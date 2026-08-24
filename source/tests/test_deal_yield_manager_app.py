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
from shared.load_test_context import load_test_scope


def _req(bid_request: dict, applicable_intents=None, ext: dict | None = None) -> RTBRequest:
    return RTBRequest(
        id="req-1",
        bid_request=bid_request,
        applicable_intents=applicable_intents or ["ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"],
        ext=ext,
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


class TestMutateExploration:
    """Exercises the cold-start exploration wiring: with epsilon disabled
    (the default), behavior is completely unchanged from before exploration
    existed; with epsilon enabled AND the call scoped as load-test-origin,
    the genesis model's constant "no change" prediction can be perturbed
    into a real mutation, and that perturbation is disclosed via the
    ":explore" model_version suffix (never presented as a confident model
    recommendation). Exploration must never fire on a call NOT scoped as
    load-test-origin, regardless of epsilon -- that's the structural gate
    (get_is_load_test()) this class also verifies."""

    def test_epsilon_zero_default_produces_no_mutations_for_genesis_model(self, monkeypatch):
        """Default (epsilon=0.0) behavior for a genesis-style constant
        no-op model must be identical to pre-exploration behavior: zero
        mutations, unsuffixed model_version. (Load-test-scoped here too,
        to isolate epsilon=0 as the reason nothing explores, not the
        load-test gate.)"""
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 0.0)
        req = _req(_SAMPLE_BID_REQUEST)
        with load_test_scope(True):
            resp = deal_yield_app.mutate(req)
        assert resp.mutations == []
        assert resp.metadata.model_version == deal_yield_app.MODEL_VERSION
        assert not resp.metadata.model_version.endswith(":explore")

    def test_epsilon_one_breaks_genesis_constant_output_into_a_real_mutation(self, monkeypatch):
        """With exploration forced on AND the call scoped as load-test
        traffic, the genesis model's constant floor_multiplier==1.0/
        margin_value==0.0 output can be perturbed into a real, non-no-op
        value -- this is the actual cold-start fix: real mutations get
        emitted, which is what lets DealYieldOutcomeEvents (and therefore
        training data) exist at all."""
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_FLOOR_BOUND", 0.1)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_MARGIN_BOUND", 0.05)
        monkeypatch.setattr(deal_yield_app, "_exploration_rng", __import__("random").Random(1))
        req = _req(_SAMPLE_BID_REQUEST)
        with load_test_scope(True):
            resp = deal_yield_app.mutate(req)
        assert len(resp.mutations) > 0

    def test_explored_response_discloses_via_model_version_suffix(self, monkeypatch):
        """An explored response must never look like a confident model
        recommendation -- the ':explore' suffix is the disclosure contract
        DealYieldOutcomeEvents and any downstream training data rely on to
        distinguish exploration from a real prediction."""
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_FLOOR_BOUND", 0.1)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_MARGIN_BOUND", 0.05)
        monkeypatch.setattr(deal_yield_app, "_exploration_rng", __import__("random").Random(1))
        req = _req(_SAMPLE_BID_REQUEST)
        with load_test_scope(True):
            resp = deal_yield_app.mutate(req)
        assert resp.metadata.model_version.endswith(":explore")

    def test_unexplored_response_never_carries_explore_suffix(self, monkeypatch):
        """A resolved (non-genesis) model_version must not gain the
        ':explore' suffix when exploration didn't actually fire."""
        monkeypatch.setattr(
            deal_yield_app, "_predict_yield",
            lambda fv, target_variant=None: (1.0, 0.0, "stable", "arn:aws:sagemaker:...:model-package/v3"),
        )
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 0.0)
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        assert resp.metadata.model_version == "arn:aws:sagemaker:...:model-package/v3"

    def test_exploration_never_fires_on_live_traffic_even_with_epsilon_enabled(self, monkeypatch):
        """The structural gate: even with epsilon forced to 1.0 (always
        explore), a call NOT scoped as load-test-origin (i.e. real
        auction traffic, get_is_load_test()==False) must never explore.
        This is what makes exploration safe to enable by default --
        the epsilon value alone is not what protects live traffic."""
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_FLOOR_BOUND", 0.1)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_MARGIN_BOUND", 0.05)
        monkeypatch.setattr(deal_yield_app, "_exploration_rng", __import__("random").Random(1))
        req = _req(_SAMPLE_BID_REQUEST)
        # No load_test_scope(True) here -- simulates real bid-serving traffic.
        resp = deal_yield_app.mutate(req)
        assert resp.mutations == []
        assert not resp.metadata.model_version.endswith(":explore")


class TestResolveEffectiveEpsilon:
    """_resolve_effective_epsilon() is the single decision point for whether
    exploration is armed on a given request -- two independent signals
    (is_load_test, explore_override) can arm it, and an explicit
    explore=False always wins over either."""

    def test_neither_signal_set_is_off(self):
        assert deal_yield_app._resolve_effective_epsilon(False, None) == 0.0

    def test_load_test_alone_arms_it(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 0.1)
        assert deal_yield_app._resolve_effective_epsilon(True, None) == 0.1

    def test_explore_true_override_alone_arms_it(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 0.1)
        assert deal_yield_app._resolve_effective_epsilon(False, True) == 0.1

    def test_explore_false_override_wins_even_during_load_test(self, monkeypatch):
        """A user who has since trained a real model must be able to force
        exploration off to see its unperturbed prediction, even if this
        happens to run during a load test."""
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 0.1)
        assert deal_yield_app._resolve_effective_epsilon(True, False) == 0.0

    def test_explore_true_and_load_test_both_set_still_arms_it(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 0.1)
        assert deal_yield_app._resolve_effective_epsilon(True, True) == 0.1


class TestMutateExploreOverride:
    """Exercises the ext.model_params.explore override end-to-end through
    mutate() -- the channel the "PMP Deals — Yield Optimizer" scenario
    card's Explore toggle uses to arm exploration on a single scenario Send
    against the (otherwise constant-output) genesis model, without needing
    a load test."""

    def test_explore_true_produces_a_mutation_for_genesis_model_no_load_test(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_FLOOR_BOUND", 0.1)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_MARGIN_BOUND", 0.05)
        monkeypatch.setattr(deal_yield_app, "_exploration_rng", __import__("random").Random(1))
        req = _req(_SAMPLE_BID_REQUEST, ext={"model_params": {"explore": True}})
        # No load_test_scope(True) -- this is the point: explore=True alone
        # must be sufficient, without any load test in progress.
        resp = deal_yield_app.mutate(req)
        assert len(resp.mutations) > 0
        assert resp.metadata.model_version.endswith(":explore")

    def test_explore_false_suppresses_mutation_even_during_load_test(self, monkeypatch):
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_FLOOR_BOUND", 0.1)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_MARGIN_BOUND", 0.05)
        monkeypatch.setattr(deal_yield_app, "_exploration_rng", __import__("random").Random(1))
        req = _req(_SAMPLE_BID_REQUEST, ext={"model_params": {"explore": False}})
        with load_test_scope(True):
            resp = deal_yield_app.mutate(req)
        assert resp.mutations == []
        assert not resp.metadata.model_version.endswith(":explore")

    def test_no_override_defaults_to_load_test_signal_only(self, monkeypatch):
        """Absent any explore override, behavior is unchanged from before
        this override existed -- only get_is_load_test() decides."""
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_FLOOR_BOUND", 0.1)
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_MARGIN_BOUND", 0.05)
        monkeypatch.setattr(deal_yield_app, "_exploration_rng", __import__("random").Random(1))
        req = _req(_SAMPLE_BID_REQUEST)
        resp = deal_yield_app.mutate(req)
        assert resp.mutations == []

    def test_non_boolean_explore_value_is_treated_as_no_override(self, monkeypatch):
        """A malformed ext.model_params.explore (e.g. a stray string) must
        never be silently coerced into True/False -- it's treated as
        absent, falling back to the load-test signal only."""
        monkeypatch.setattr(deal_yield_app, "_predict_yield", lambda fv, target_variant=None: (1.0, 0.0, "", ""))
        monkeypatch.setattr(deal_yield_app, "_EXPLORATION_EPSILON", 1.0)
        req = _req(_SAMPLE_BID_REQUEST, ext={"model_params": {"explore": "yes"}})
        resp = deal_yield_app.mutate(req)
        assert resp.mutations == []
