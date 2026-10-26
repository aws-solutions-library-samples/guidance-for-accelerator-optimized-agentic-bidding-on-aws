"""Yield Optimizer (Floor) inference via NVIDIA Triton's FIL backend.

Serves ONE single-target XGBoost model -- the deal floor multiplier. The
Triton transport itself lives in shared/fil_inference.py (FIL's
input__0/output__0 contract is identical for every FIL model); this module
holds only what is specific to the floor model.

TRITON MODEL NAME IS FIXED. "deal_yield_manager_floor" is the name of a real
deployed Triton model, a real SageMaker Model Package Group
(artf-deal-yield-manager-floor), and the directory name asserted by
source/tests/test_export_xgboost_genesis.py. Renaming it to match this
container's new name would orphan already-registered model packages and break
the genesis export, so the historical name stays.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.fil_inference import infer_single_output, is_model_ready

FLOOR_MODEL_NAME = "deal_yield_manager_floor"

# "No floor change recommended." Returned when Triton is unreachable or the
# model errors, so a failed inference is a no-op rather than a fabricated
# adjustment (business-rules.md BR-9).
NO_CHANGE_FLOOR_MULTIPLIER = 1.0


def predict_floor_multiplier(
    feature_vector: list[float], target_variant: str | None = None
) -> tuple[float, str, str]:
    """Predict a bidfloor multiplier for one deal.

    Args:
        feature_vector: The 7-element vector from
            shared.yield_features.build_feature_vector().
        target_variant: "stable" | "canary" | None. Load-test-only override
            (see source/orchestrator/loadtest_targeting.py); always None on
            live bid-serving traffic.

            "canary" selects the ``<model>_canary`` Triton model DIRECTLY, by
            name. The TensorRT-backed containers instead pass a target_variant
            INPUT to a Python-backend router model that owns the split
            (dlrm_bid_shader/triton_inference.py), but a FIL model accepts only
            input__0, so a router input is not available here. Selecting the
            model by name achieves the same thing for a targeted load test.

            This deliberately does NOT give the yield models a live canary
            split: live traffic never sets a variant, so it always reaches the
            stable model exactly as before. Splitting live traffic would still
            need a router.

    Returns:
        (floor_multiplier, served_variant, served_model_version).
        floor_multiplier == 1.0 means "no change recommended".

        ``served_variant`` is only populated when a variant was explicitly
        targeted AND that model actually returned a prediction -- so a canary
        request that fell back to the no-change default reports "", never a
        claim that the canary served it. ``served_model_version`` stays ""
        because no router declares it for these models: a real "unknown", never
        fabricated. Never raises.
    """
    model_name = (
        f"{FLOOR_MODEL_NAME}_canary" if target_variant == "canary" else FLOOR_MODEL_NAME
    )
    floor_multiplier = infer_single_output(model_name, feature_vector)
    if floor_multiplier is None:
        return NO_CHANGE_FLOOR_MULTIPLIER, "", ""
    served_variant = target_variant if target_variant in ("stable", "canary") else ""
    return floor_multiplier, served_variant, ""


def is_triton_ready() -> bool:
    """True only if Triton and the floor model are both ready.

    Deliberately does NOT check the margin model: this container must stay
    servable when the margin model is down (BR-5 -- the two intents are
    independent and atomic, never gated on each other).
    """
    return is_model_ready(FLOOR_MODEL_NAME)
