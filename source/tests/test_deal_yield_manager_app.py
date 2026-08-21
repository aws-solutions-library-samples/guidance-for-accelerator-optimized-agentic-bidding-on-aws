"""Tests for containers.deal_yield_manager.app -- ARTF mutate() entry point.

Monkeypatches _predict_yield to avoid a real Triton dependency, following
the project's existing convention of mocking only external service
boundaries (see source/tests/README.md).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.artf_types import Intent, RTBRequest

import containers.deal_yield_manager.app as deal_yield_app


def _req(bid_request: dict, applicable_intents=None) -> RTBRequest:
    return RTBRequest(
        id="req-1",
        bid_request=bid_request,
        applicable_intents=applicable_intents or ["ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"],
    )


_SAMPLE_BID_REQUEST = {
    "imp": [
        {
            "id": "imp-1",
            "pmp": {
                "deals": [
                    {"id": "deal-premium", "bidfloor": 10.0, "at": 1},
                    {"id": "deal-remnant", "bidfloor": 0.5, "at": 2},
                ]
            },
        }
    ],
    "site": {"cat": ["IAB17"]},
}


class TestMutateIntentFiltering:
    def test_no_mutations_when_neither_intent_applicable(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.5, 0.1, "", ""))
        req = _req(_SAMPLE_BID_REQUEST, applicable_intents=["BID_SHADE"])
        resp = deal_yield_app.mutate(req)
        assert resp.mutations == []

    def test_returns_early_with_no_deals(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        req = _req({"imp": [{"id": "imp-1"}]})
        resp = deal_yield_app.mutate(req)
        assert resp.mutations == []


class TestMutateFloorAdjustment:
    def test_emits_floor_mutation_when_multiplier_not_one(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.2, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        floor_muts = [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_FLOOR]
        assert len(floor_muts) == 2  # one per deal
        assert floor_muts[0].path == "/imp/imp-1/deals/deal-premium"
        assert floor_muts[0].adjust_deal.bidfloor == pytest.approx(12.0)

    def test_no_floor_mutation_when_multiplier_is_one(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        assert not [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_FLOOR]

    def test_adjusted_floor_never_negative(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (-5.0, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        floor_muts = [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_FLOOR]
        assert all(m.adjust_deal.bidfloor >= 0.0 for m in floor_muts)

    def test_floor_intent_not_requested_suppresses_mutation(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.2, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST, applicable_intents=["ADJUST_DEAL_MARGIN"])
        resp = deal_yield_app.mutate(req)
        assert not [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_FLOOR]


class TestMutateMarginAdjustment:
    def test_emits_margin_mutation_with_cpm_for_first_price(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.15, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        margin_muts = [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_MARGIN]
        premium_mutation = next(m for m in margin_muts if m.path == "/imp/imp-1/deals/deal-premium")
        assert premium_mutation.adjust_deal.margin.calculation_type == 0  # CPM

    def test_emits_margin_mutation_with_percent_for_second_price(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.15, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        margin_muts = [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_MARGIN]
        remnant_mutation = next(m for m in margin_muts if m.path == "/imp/imp-1/deals/deal-remnant")
        assert remnant_mutation.adjust_deal.margin.calculation_type == 1  # PERCENT

    def test_no_margin_mutation_when_value_is_zero(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        assert not [m for m in resp.mutations if m.intent == Intent.ADJUST_DEAL_MARGIN]


class TestMutateBothIntentsIndependent:
    def test_a_deal_can_receive_both_mutations(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.1, 0.05, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        premium_muts = [m for m in resp.mutations if m.path == "/imp/imp-1/deals/deal-premium"]
        intents = {m.intent for m in premium_muts}
        assert intents == {Intent.ADJUST_DEAL_FLOOR, Intent.ADJUST_DEAL_MARGIN}


class TestMutateModelVersion:
    def test_falls_back_to_static_model_version_when_unresolved(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        assert resp.metadata.model_version == deal_yield_app.MODEL_VERSION

    def test_uses_resolved_model_version_when_available(self, monkeypatch):
        monkeypatch.setattr(
            deal_yield_app, "_predict_yield",
            lambda fv, target_variant=None: (1.0, 0.0, "stable", "arn:aws:sagemaker:...:model-package/v3"),
        )
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        assert resp.metadata.model_version == "arn:aws:sagemaker:...:model-package/v3"


class TestMutateNeverRaises:
    def test_never_raises_on_triton_fallback(self, monkeypatch):
        # Simulate the Triton-error fallback contract (1.0, 0.0, "", "").
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)  # must not raise
        assert resp.id == "req-1"
