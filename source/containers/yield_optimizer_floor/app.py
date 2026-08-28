"""Yield Optimizer (Floor) -- ARTF container serving ADJUST_DEAL_FLOOR only.

Predicts a bidfloor multiplier per private-marketplace (PMP) deal using an
XGBoost model served by NVIDIA Triton's FIL (Forest Inference Library) backend.

This container and yield_optimizer_margin were one container until the split.
They are independent now because their two intents already were: BR-5 states
ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN are independent, atomic mutations,
never gated on each other, and FIL cannot serve a two-output tree model anyway
-- so there were always two models. One container serving both made the floor
model's availability depend on the margin model's, which nothing in the
business rules asked for.

Mutation path/payload convention verified against the ARTF v1.0 proto
(agenticrtbframework.proto): a deal is identified purely via
`path: "/imp/{imp_id}/deals/{deal_id}"` -- AdjustDealPayload carries no
deal_id field.

See aidlc-docs/construction/deal-yield-model/functional-design/ for the full
design rationale (business-logic-model.md, business-rules.md).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    AdjustDealPayload, Intent, Metadata, Mutation, Operation, RTBRequest,
    RTBResponse, intent_applicable,
)
from shared.yield_exploration import (
    FLOOR_MULTIPLIER_MAX, FLOOR_MULTIPLIER_MIN, apply_exploration,
    parse_explore_override, resolve_effective_epsilon,
)
from shared.yield_features import build_feature_vector

logger = logging.getLogger(__name__)

MODEL_VERSION = "deal-yield-floor-xgboost-v1"

USE_TRITON = os.environ.get("USE_TRITON", "").lower() in ("1", "true", "yes")

# Bounded epsilon-greedy exploration. See shared/yield_exploration.py's module
# docstring for the cold-start problem this solves: a model converged on
# "recommend no change" -- true of the genesis model by construction -- never
# emits a mutation, so it can never generate the outcome data needed to learn
# anything else.
#
# epsilon defaults to 0.1 but only takes effect on requests that are
# load-test-originated or that explicitly opt in via ext.model_params.explore
# (see resolve_effective_epsilon). A prior version read this epsilon
# unconditionally with a disabled (0.0) default, which meant a freshly deployed
# stack could NEVER produce a yield mutation for the genesis model -- confirmed
# live: every load test produced exactly 0 mutations, so no outcome event was
# ever emitted and the training pipeline had nothing to bootstrap from.
_EXPLORATION_EPSILON = float(os.environ.get("YIELD_EXPLORATION_EPSILON", "0.1"))
_EXPLORATION_FLOOR_BOUND = float(os.environ.get("YIELD_EXPLORATION_FLOOR_BOUND", "0.05"))
# Module-level so tests can substitute a seeded random.Random() for
# deterministic assertions, without monkeypatching the `random` module itself.
_exploration_rng = random.Random()

if USE_TRITON:
    from container.triton_inference import predict_floor_multiplier as _predict_floor
else:
    def _predict_floor(feature_vector, target_variant=None):
        """CPU fallback: no model available without Triton -- recommends no
        change rather than fabricating a prediction. Mirrors the safe
        no-op-recommendation contract every Triton-backed container falls back
        to on inference error (BR-9)."""
        return 1.0, "", ""


def mutate(req: RTBRequest) -> RTBResponse:
    """ARTF GetMutations -- ADJUST_DEAL_FLOOR via Triton FIL-served XGBoost."""
    if not intent_applicable(Intent.ADJUST_DEAL_FLOOR, req.applicable_intents):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    from shared.load_test_context import get_is_load_test, get_target_variant

    request_time = datetime.now(timezone.utc)
    bid_request = req.bid_request or {}

    effective_epsilon = resolve_effective_epsilon(
        get_is_load_test(),
        parse_explore_override(req.model_params),
        _EXPLORATION_EPSILON,
    )

    mutations: list[Mutation] = []
    served_model_version = ""
    any_explored = False

    for imp in bid_request.get("imp", []):
        imp_id = imp.get("id", "")
        pmp = imp.get("pmp") or {}
        for deal in pmp.get("deals", []):
            deal_id = deal.get("id", "")
            if not deal_id:
                continue

            feature_vector = build_feature_vector(bid_request, deal, request_time)
            floor_multiplier, _served_variant, resolved_version = _predict_floor(
                feature_vector, target_variant=get_target_variant()
            )
            if resolved_version:
                served_model_version = resolved_version

            floor_multiplier, explored = apply_exploration(
                floor_multiplier,
                rng=_exploration_rng,
                epsilon=effective_epsilon,
                bound=_EXPLORATION_FLOOR_BOUND,
                clamp_min=FLOOR_MULTIPLIER_MIN,
                clamp_max=FLOOR_MULTIPLIER_MAX,
            )
            any_explored = any_explored or explored

            # BR-3/BR-4: a multiplier of exactly 1.0 is "no change" and must
            # never become a mutation -- emitting a no-op REPLACE would put a
            # meaningless entry in the bidstream and a meaningless outcome
            # event in the training data.
            if floor_multiplier == 1.0:
                continue

            original_bidfloor = float(deal.get("bidfloor", 0.0) or 0.0)
            # BR-7: a bidfloor can never go negative.
            adjusted_bidfloor = max(0.0, round(original_bidfloor * floor_multiplier, 4))
            mutations.append(Mutation(
                intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
                path=f"/imp/{imp_id}/deals/{deal_id}",
                adjust_deal=AdjustDealPayload(bidfloor=adjusted_bidfloor),
            ))

    resolved_model_version = served_model_version or MODEL_VERSION
    # Disclose exploration in the served model_version -- never let an
    # exploratory probe look like a confident model recommendation to anything
    # reading this response downstream (outcome events, training data, the demo
    # UI). See shared/yield_exploration.py's module docstring.
    if any_explored:
        resolved_model_version = f"{resolved_model_version}:explore"
    return RTBResponse(
        id=req.id,
        mutations=mutations,
        metadata=Metadata(api_version="1.0", model_version=resolved_model_version),
    )


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(
        mutate, agent_name="yield-optimizer-floor",
        grpc_port=50051, mcp_port=8081, health_port=8080,
    )
