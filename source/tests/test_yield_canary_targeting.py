"""Yield containers must actually serve the canary when one is targeted.

The two yield models reach a canary differently from the TensorRT-backed pair.
dlrm_bid_shader/ncf_deal_manager pass a ``target_variant`` INPUT to a
Python-backend router model that owns the split. A FIL model accepts only
``input__0``, so no router input is available; the yield containers instead
select the ``<model>_canary`` Triton model by NAME.

Before this, ``target_variant`` was accepted and ignored, and the yield model
types were excluded from CANARY_SUPPORTED_MODEL_TYPES. That exclusion was
correct at the time: ``is_canary_staged`` only probes whether the canary MODEL is
loaded, so including them would have let the check pass while the container still
inferred against the stable model — returning stable results labelled challenger.

Scope: this covers a TARGETED load test, not a live canary split. Live traffic
never sets a variant, so it must keep reaching the stable model untouched.
"""

from __future__ import annotations

import importlib
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FEATURES = [0.0] * 7


def _floor_module():
    import containers.yield_optimizer_floor.triton_inference as mod

    return importlib.reload(mod)


def _margin_module():
    import containers.yield_optimizer_margin.triton_inference as mod

    return importlib.reload(mod)


@pytest.fixture(params=["floor", "margin"])
def model(request):
    """Both yield containers, since the two must not drift apart."""
    if request.param == "floor":
        mod = _floor_module()
        return mod, mod.predict_floor_multiplier, "deal_yield_manager_floor", 1.15, 1.0
    mod = _margin_module()
    return mod, mod.predict_margin_value, "deal_yield_manager_margin", 0.25, 0.0


def _capture(mod, value):
    """Patch the FIL transport, recording which model name it was asked for."""
    seen: list[str] = []

    def _infer(model_name, feature_vector):
        seen.append(model_name)
        return value

    return mock.patch.object(mod, "infer_single_output", _infer), seen


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def test_canary_target_selects_the_canary_model(model):
    mod, predict, base, value, _ = model
    patch, seen = _capture(mod, value)
    with patch:
        result = predict(FEATURES, target_variant="canary")
    assert seen == [f"{base}_canary"]
    assert result[0] == value
    assert result[1] == "canary"


def test_stable_target_selects_the_base_model(model):
    """The yield models are loaded directly under their base name — there is no
    ``<model>_stable`` for them, unlike the router-fronted pair."""
    mod, predict, base, value, _ = model
    patch, seen = _capture(mod, value)
    with patch:
        result = predict(FEATURES, target_variant="stable")
    assert seen == [base]
    assert result[1] == "stable"


def test_live_traffic_is_unchanged(model):
    """No variant means live serving: base model, and no variant claimed."""
    mod, predict, base, value, _ = model
    patch, seen = _capture(mod, value)
    with patch:
        result = predict(FEATURES)
    assert seen == [base]
    assert result[1] == ""


def test_unrecognised_variant_falls_back_to_the_base_model(model):
    mod, predict, base, value, _ = model
    patch, seen = _capture(mod, value)
    with patch:
        result = predict(FEATURES, target_variant="nonsense")
    assert seen == [base]
    assert result[1] == ""


# ---------------------------------------------------------------------------
# A failed inference must not claim the canary served it
# ---------------------------------------------------------------------------

def test_failed_canary_inference_does_not_claim_a_canary_served_it(model):
    """BR-9: an unreachable model returns the no-change default. Reporting
    served_variant="canary" there would attribute a fabricated no-op to the
    challenger and put it in the comparison as a real canary sample."""
    mod, predict, _base, _value, no_change = model
    with mock.patch.object(mod, "infer_single_output", lambda n, f: None):
        result = predict(FEATURES, target_variant="canary")
    assert result[0] == no_change
    assert result[1] == ""


def test_model_version_is_never_fabricated(model):
    """No router declares a resolved version for these models, so it stays an
    honest empty string rather than a guess."""
    mod, predict, _base, value, _ = model
    patch, _seen = _capture(mod, value)
    with patch:
        assert predict(FEATURES, target_variant="canary")[2] == ""


# ---------------------------------------------------------------------------
# The orchestrator must agree that these types support a canary
# ---------------------------------------------------------------------------

def test_both_yield_types_are_canary_supported():
    from orchestrator.loadtest_targeting import canary_supported

    assert canary_supported("deal_yield_manager_floor")
    assert canary_supported("deal_yield_manager_margin")


def test_rule_based_types_remain_unsupported():
    """Kept selectable but honest: they have no Triton path at all, so a
    challenger-targeted run is refused rather than silently run against a
    rule engine."""
    from orchestrator.loadtest_targeting import canary_supported

    assert not canary_supported("widedeep_segment_activator")
    assert not canary_supported("metrics_enricher")


def test_canary_support_matches_the_types_that_can_route_to_one():
    from orchestrator.loadtest_targeting import CANARY_SUPPORTED_MODEL_TYPES

    assert CANARY_SUPPORTED_MODEL_TYPES == {
        "dlrm_bid_shader",
        "ncf_deal_manager",
        "deal_yield_manager_floor",
        "deal_yield_manager_margin",
    }
