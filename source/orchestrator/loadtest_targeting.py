"""Challenger-targeting override for load tests (ChallengerTargetOverride).

Lets a load test force its synthetic traffic onto a model type's canary
Triton variant instead of the router's normal random split, and reports
plainly when the requested variant isn't available — never silently
falling back to stable while presenting it as a challenger result (BR-5).

This module is the ONLY place that constructs the out-of-band
``X-Load-Test-Target-Variant`` HTTP header (see shared/load_test_context.py
for the request-scoped signal it sets on the container side). No other
code path in the orchestrator sets this header, which is what keeps the
override structurally exclusive to the load-test invocation path (BR-4).

Maps to: FR-3 (Story 2).
"""

from __future__ import annotations

import os
from typing import Literal

import httpx

from shared.load_test_context import HEADER_NAME

# The model types that can actually serve a targeted challenger request today.
# widedeep_segment_activator/metrics_enricher are rule-based — kept in the
# selectable set (Q4=B) so a future Triton/canary rollout for them doesn't
# require an API contract change, but they report "not supported" rather
# than attempting an override.
#
# The two yield models reach their canary a different way from the two
# TensorRT-backed ones. dlrm_bid_shader/ncf_deal_manager pass a target_variant
# INPUT to a Python-backend router model that owns the split. A FIL model
# accepts only input__0, so the yield containers instead select the
# ``<model>_canary`` model by NAME
# (yield_optimizer_floor/triton_inference.py). Both end up serving the
# requested variant, which is what this set is about.
#
# They were excluded while that routing did not exist. Including them then would
# have been worse than excluding them: is_canary_staged() below only probes
# whether the canary MODEL is loaded, so the check would have passed while the
# container still inferred against the stable model — returning stable results
# labelled challenger.
#
# Note this covers targeted load tests, not a live canary split: live traffic
# never sets a variant, so it always reaches the stable model. A live split for
# the yield models would still need a router of their own.
CANARY_SUPPORTED_MODEL_TYPES = frozenset({
    "dlrm_bid_shader",
    "ncf_deal_manager",
    "deal_yield_manager_floor",
    "deal_yield_manager_margin",
})

TargetVariant = Literal["current", "challenger"]


class CanaryNotSupportedError(Exception):
    """Raised when challenger-targeting is requested for a model type that
    has no Triton/canary infrastructure at all (structural limitation,
    distinct from "no canary staged" — see BR-5)."""

    def __init__(self, model_type: str):
        super().__init__(
            f"No canary supported for model type '{model_type}' — this model "
            "type has no Triton/canary infrastructure today."
        )
        self.model_type = model_type


class CanaryNotStagedError(Exception):
    """Raised when challenger-targeting is requested for a canary-capable
    model type, but no canary is currently staged (staging-state fact,
    distinct from "no canary supported" — see BR-5)."""

    def __init__(self, model_type: str):
        super().__init__(
            f"No canary is currently staged for model type '{model_type}'. "
            "Stage a canary before running a challenger-targeted load test."
        )
        self.model_type = model_type


def canary_supported(model_type: str) -> bool:
    """True if model_type has Triton/canary infrastructure today."""
    return model_type in CANARY_SUPPORTED_MODEL_TYPES


async def is_canary_staged(model_type: str) -> bool:
    """Check whether a canary is currently staged for model_type.

    Probes Triton's own /v2/models/{model}_canary/ready endpoint directly —
    the same real-health-check pattern app.py's diagnostics route already
    uses for the stable models, rather than depending on CanaryDeployer's
    in-process state (a separate process/agent the orchestrator doesn't
    share memory with). Returns False (never fabricates "staged") on any
    connection error, timeout, or non-200 response.
    """
    if not canary_supported(model_type):
        return False
    triton_url = os.environ.get("TRITON_URL", "triton-inference-server:8000")
    canary_model = f"{model_type}_canary"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"http://{triton_url}/v2/models/{canary_model}/ready")
            return resp.status_code == 200
    except Exception:
        return False


def build_override_headers(target_variant: TargetVariant) -> dict[str, str]:
    """Build the out-of-band header for a container call.

    Returns an empty dict for target_variant="current" (no override needed
    — the container's normal random-split behavior already covers "current"
    on average, and this function only exists to force "challenger").
    Returns the X-Load-Test-Target-Variant: canary header for "challenger".
    """
    if target_variant == "challenger":
        return {HEADER_NAME: "canary"}
    return {}


async def validate_challenger_target(model_type: str) -> None:
    """Raise the correct, distinct error if a challenger-targeted run can't proceed.

    Call this BEFORE starting a load-test run with target_variant="challenger"
    — never let the run silently execute against stable while presenting the
    result as a challenger result (BR-5).

    Raises:
        CanaryNotSupportedError: model_type has no Triton/canary infrastructure.
        CanaryNotStagedError: model_type supports canaries, but none is staged.
    """
    if not canary_supported(model_type):
        raise CanaryNotSupportedError(model_type)
    if not await is_canary_staged(model_type):
        raise CanaryNotStagedError(model_type)
