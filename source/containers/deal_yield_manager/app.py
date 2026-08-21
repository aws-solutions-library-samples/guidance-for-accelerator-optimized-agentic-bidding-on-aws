"""Deal Yield Manager -- ARTF container using XGBoost (via Triton's FIL
backend) for ADJUST_DEAL_FLOOR / ADJUST_DEAL_MARGIN.

Unlike ncf_deal_manager (which scores user-deal relevance) or the
deterministic-rules containers, this container predicts a real yield
adjustment -- a floor multiplier and a margin value -- per private
marketplace (PMP) deal, served by NVIDIA Triton's Forest Inference Library
(FIL) backend for GPU-accelerated tree-model inference. FIL is bundled in
the same nvcr.io/nvidia/tritonserver:24.08-py3 image already used by this
project's other Triton-backed containers; no new Triton image is required.

Mutation path/payload convention verified against the ARTF v1.0 proto
(agenticrtbframework.proto): a deal is identified purely via
`path: "/imp/{imp_id}/deals/{deal_id}"` -- AdjustDealPayload carries no
deal_id field.

See aidlc-docs/construction/deal-yield-model/functional-design/ for the
full design rationale (business-logic-model.md, business-rules.md).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    AdjustDealPayload, Intent, Margin, MarginCalculationType, Metadata,
    Mutation, Operation, RTBRequest, RTBResponse, intent_applicable,
)

from .features import build_feature_vector

logger = logging.getLogger(__name__)

MODEL_VERSION = "deal-yield-xgboost-v1"

USE_TRITON = os.environ.get("USE_TRITON", "").lower() in ("1", "true", "yes")

if USE_TRITON:
    from container.triton_inference import predict_yield_adjustment as _predict_yield
else:
    def _predict_yield(feature_vector, target_variant=None):
        """CPU fallback: no model available without Triton -- recommends no
        change rather than fabricating a prediction. Mirrors the safe
        no-op-recommendation contract every Triton-backed container falls
        back to on inference error (BR-9)."""
        return 1.0, 0.0, "", ""


def _margin_calculation_type(at: int | None) -> int:
    """BR-6: at=1 (guaranteed/first-price) -> CPM; at=2 (open/second-price)
    -> PERCENT; any other/missing at defaults to PERCENT (the more common
    OpenRTB convention) rather than raising."""
    if at == 1:
        return MarginCalculationType.CPM
    return MarginCalculationType.PERCENT


def mutate(req: RTBRequest) -> RTBResponse:
    """ARTF GetMutations -- ADJUST_DEAL_FLOOR / ADJUST_DEAL_MARGIN via
    Triton FIL-served XGBoost prediction per deal."""
    applicable = req.applicable_intents
    floor_ok = intent_applicable(Intent.ADJUST_DEAL_FLOOR, applicable)
    margin_ok = intent_applicable(Intent.ADJUST_DEAL_MARGIN, applicable)
    if not (floor_ok or margin_ok):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    from shared.load_test_context import get_target_variant

    request_time = datetime.now(timezone.utc)
    bid_request = req.bid_request or {}

    mutations: list[Mutation] = []
    served_model_version = ""

    for imp in bid_request.get("imp", []):
        imp_id = imp.get("id", "")
        pmp = imp.get("pmp") or {}
        for deal in pmp.get("deals", []):
            deal_id = deal.get("id", "")
            if not deal_id:
                continue

            feature_vector = build_feature_vector(bid_request, deal, request_time)
            floor_multiplier, margin_value, _served_variant, resolved_version = (
                _predict_yield(feature_vector, target_variant=get_target_variant())
            )
            if resolved_version:
                served_model_version = resolved_version

            path = f"/imp/{imp_id}/deals/{deal_id}"

            if floor_ok and floor_multiplier != 1.0:
                original_bidfloor = float(deal.get("bidfloor", 0.0) or 0.0)
                adjusted_bidfloor = max(0.0, round(original_bidfloor * floor_multiplier, 4))
                mutations.append(Mutation(
                    intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
                    path=path,
                    adjust_deal=AdjustDealPayload(bidfloor=adjusted_bidfloor),
                ))

            if margin_ok and margin_value != 0.0:
                at = deal.get("at")
                mutations.append(Mutation(
                    intent=Intent.ADJUST_DEAL_MARGIN, op=Operation.REPLACE,
                    path=path,
                    adjust_deal=AdjustDealPayload(
                        margin=Margin(
                            value=round(margin_value, 4),
                            calculation_type=_margin_calculation_type(at),
                        )
                    ),
                ))

    resolved_model_version = served_model_version or MODEL_VERSION
    return RTBResponse(
        id=req.id,
        mutations=mutations,
        metadata=Metadata(
            api_version="1.0",
            model_version=resolved_model_version,
        ),
    )


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="deal-yield-manager", grpc_port=50051, mcp_port=8081, health_port=8080)
