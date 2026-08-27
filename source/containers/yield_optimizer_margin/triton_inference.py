"""Yield Optimizer (Margin) inference via NVIDIA Triton's FIL backend.

Serves ONE single-target XGBoost model -- the deal margin value. The Triton
transport itself lives in shared/fil_inference.py (FIL's input__0/output__0
contract is identical for every FIL model); this module holds only what is
specific to the margin model.

TRITON MODEL NAME IS FIXED. "deal_yield_manager_margin" is the name of a real
deployed Triton model, a real SageMaker Model Package Group
(artf-deal-yield-manager-margin), and the directory name asserted by
source/tests/test_export_xgboost_genesis.py. Renaming it to match this
container's new name would orphan already-registered model packages and break
the genesis export, so the historical name stays.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.fil_inference import infer_single_output, is_model_ready

MARGIN_MODEL_NAME = "deal_yield_manager_margin"

# "No margin adjustment recommended." Returned when Triton is unreachable or
# the model errors, so a failed inference is a no-op rather than a fabricated
# adjustment (business-rules.md BR-9). Note this differs from the floor
# container's 1.0 -- margin is additive, floor is multiplicative.
NO_CHANGE_MARGIN_VALUE = 0.0


def predict_margin_value(
    feature_vector: list[float], target_variant: str | None = None
) -> tuple[float, str, str]:
    """Predict a margin value for one deal.

    Args:
        feature_vector: The 7-element vector from
            shared.yield_features.build_feature_vector().
        target_variant: "stable" | "canary" | None. Load-test-only override
            (see source/orchestrator/loadtest_targeting.py); always None on
            live bid-serving traffic. There is no canary router for the yield
            models yet -- this parameter is accepted for interface parity with
            the other Triton-backed containers and to avoid a signature change
            when one is added, but has no effect today.

    Returns:
        (margin_value, served_variant, served_model_version).
        margin_value == 0.0 means "no adjustment recommended".
        served_variant/served_model_version are "" until a canary router
        exists -- a real "unknown", never fabricated. Never raises.
    """
    margin_value = infer_single_output(MARGIN_MODEL_NAME, feature_vector)
    return (
        margin_value if margin_value is not None else NO_CHANGE_MARGIN_VALUE,
        "",
        "",
    )


def is_triton_ready() -> bool:
    """True only if Triton and the margin model are both ready.

    Deliberately does NOT check the floor model: this container must stay
    servable when the floor model is down (BR-5 -- the two intents are
    independent and atomic, never gated on each other).
    """
    return is_model_ready(MARGIN_MODEL_NAME)
