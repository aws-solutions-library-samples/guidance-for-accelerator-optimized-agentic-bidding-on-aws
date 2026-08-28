"""Tests for the two Yield Optimizer containers' ARTF mutate() entry points.

Replaces the pre-split test_deal_yield_manager_app.py. Behavior that is
symmetric across the two containers is parametrized over both, so a regression
in one cannot hide behind the other; the genuinely asymmetric rules get their
own classes (BR-7 floor clamping, BR-6 margin calculation type).

Monkeypatches each container's _predict_* function to avoid a real Triton
dependency, following the project's convention of mocking only external service
boundaries (see source/tests/README.md).
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.artf_types import Intent, MarginCalculationType, RTBRequest
from shared.load_test_context import load_test_scope

import containers.yield_optimizer_floor.app as floor_app
import containers.yield_optimizer_margin.app as margin_app

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


def _req(bid_request=None, applicable_intents=None, ext=None) -> RTBRequest:
    return RTBRequest(
        id="req-1",
        bid_request=_SAMPLE_BID_REQUEST if bid_request is None else bid_request,
        applicable_intents=applicable_intents or ["ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"],
        ext=ext,
    )


class _Case:
    """Per-container knobs so one test body can drive either container."""

    def __init__(self, app, predict_name, intent, other_intent, no_op, changed, bound_attr):
        self.app = app
        self.predict_name = predict_name
        self.intent = intent
        self.other_intent = other_intent
        self.no_op = no_op          # value meaning "recommend no change"
        self.changed = changed      # value that should produce a mutation
        self.bound_attr = bound_attr

    def patch_predict(self, monkeypatch, value, variant="", version=""):
        monkeypatch.setattr(
            self.app, self.predict_name,
            lambda fv, target_variant=None: (value, variant, version),
        )

    def force_exploration(self, monkeypatch, epsilon=1.0, seed=1):
        monkeypatch.setattr(self.app, "_EXPLORATION_EPSILON", epsilon)
        monkeypatch.setattr(self.app, self.bound_attr, 0.1)
        monkeypatch.setattr(self.app, "_exploration_rng", random.Random(seed))


FLOOR = _Case(
    floor_app, "_predict_floor", Intent.ADJUST_DEAL_FLOOR, Intent.ADJUST_DEAL_MARGIN,
    no_op=1.0, changed=1.2, bound_attr="_EXPLORATION_FLOOR_BOUND",
)
MARGIN = _Case(
    margin_app, "_predict_margin", Intent.ADJUST_DEAL_MARGIN, Intent.ADJUST_DEAL_FLOOR,
    no_op=0.0, changed=0.15, bound_attr="_EXPLORATION_MARGIN_BOUND",
)

BOTH = pytest.mark.parametrize("case", [FLOOR, MARGIN], ids=["floor", "margin"])


@BOTH
class TestIntentScoping:
    def test_emits_only_its_own_intent(self, case, monkeypatch):
        """Each container serves exactly one intent. A container emitting the
        other's intent would double-apply it once both are deployed."""
        case.patch_predict(monkeypatch, case.changed)
        resp = case.app.mutate(_req())
        assert resp.mutations, "expected mutations for a non-no-op prediction"
        assert {m.intent for m in resp.mutations} == {case.intent}

    def test_no_mutations_when_its_intent_not_applicable(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.changed)
        resp = case.app.mutate(_req(applicable_intents=[case.other_intent.name]))
        assert resp.mutations == []

    def test_no_mutations_for_unrelated_intent(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.changed)
        resp = case.app.mutate(_req(applicable_intents=["BID_SHADE"]))
        assert resp.mutations == []

    def test_returns_early_with_no_deals(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.changed)
        resp = case.app.mutate(_req(bid_request={"imp": [{"id": "imp-1"}]}))
        assert resp.mutations == []

    def test_deal_without_an_id_is_skipped(self, case, monkeypatch):
        """A deal with no id cannot be addressed by an ARTF path
        (/imp/{imp_id}/deals/{deal_id}), so it must be skipped rather than
        producing a mutation pointing at an empty deal id."""
        case.patch_predict(monkeypatch, case.changed)
        resp = case.app.mutate(_req(bid_request={
            "imp": [{"id": "imp-1", "pmp": {"deals": [
                {"id": "", "bidfloor": 1.0, "at": 2},
                {"bidfloor": 2.0, "at": 2},
                {"id": "real", "bidfloor": 3.0, "at": 2},
            ]}}],
        }))
        assert [m.path for m in resp.mutations] == ["/imp/imp-1/deals/real"]


@BOTH
class TestNoOpSuppression:
    def test_no_mutation_for_the_no_op_value(self, case, monkeypatch):
        """BR-3/BR-4: the model's "no change" value must never become a
        mutation -- a no-op REPLACE would pollute both the bidstream and the
        training data."""
        case.patch_predict(monkeypatch, case.no_op)
        resp = case.app.mutate(_req())
        assert resp.mutations == []

    def test_emits_one_mutation_per_deal_when_changed(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.changed)
        resp = case.app.mutate(_req())
        assert len(resp.mutations) == 2
        assert [m.path for m in resp.mutations] == [
            "/imp/imp-1/deals/deal-premium",
            "/imp/imp-1/deals/deal-remnant",
        ]


@BOTH
class TestModelVersion:
    def test_falls_back_to_static_version_when_unresolved(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op)
        resp = case.app.mutate(_req())
        assert resp.metadata.model_version == case.app.MODEL_VERSION

    def test_uses_resolved_version_when_available(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op, variant="stable", version="arn:...:model-package/v3")
        resp = case.app.mutate(_req())
        assert resp.metadata.model_version == "arn:...:model-package/v3"


class TestCrossContainerIdentity:
    def test_the_two_containers_report_distinct_static_versions(self):
        """Floor and margin are independently trained and independently
        registered, so a shared version string would make it impossible to tell
        which model produced a given mutation."""
        assert floor_app.MODEL_VERSION != margin_app.MODEL_VERSION

    def test_they_target_distinct_triton_models(self):
        """The Triton model names are real deployed resources and must not
        collide -- and must keep their historical deal_yield_manager_* names,
        which match already-registered SageMaker Model Package Groups."""
        from containers.yield_optimizer_floor.triton_inference import FLOOR_MODEL_NAME
        from containers.yield_optimizer_margin.triton_inference import MARGIN_MODEL_NAME
        assert FLOOR_MODEL_NAME == "deal_yield_manager_floor"
        assert MARGIN_MODEL_NAME == "deal_yield_manager_margin"


@BOTH
class TestNeverRaises:
    def test_triton_fallback_does_not_raise(self, case, monkeypatch):
        """BR-9: on inference failure the container recommends no change rather
        than raising or fabricating a value."""
        case.patch_predict(monkeypatch, case.no_op)
        resp = case.app.mutate(_req())
        assert resp.id == "req-1"
        assert resp.mutations == []


@BOTH
class TestExplorationGate:
    """Exploration breaks the cold-start deadlock (a model converged on "no
    change" never emits a mutation, so it never generates outcome data for
    itself), but must never fire on traffic that is neither load-test-scoped nor
    explicitly opted in."""

    def test_epsilon_zero_produces_no_mutations_for_a_no_op_model(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op)
        monkeypatch.setattr(case.app, "_EXPLORATION_EPSILON", 0.0)
        with load_test_scope(True):
            resp = case.app.mutate(_req())
        assert resp.mutations == []
        assert not resp.metadata.model_version.endswith(":explore")

    def test_load_test_scope_turns_a_no_op_model_into_a_real_mutation(self, case, monkeypatch):
        """The actual cold-start fix: real mutations get emitted, which is what
        lets outcome events -- and therefore training data -- exist at all."""
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        with load_test_scope(True):
            resp = case.app.mutate(_req())
        assert len(resp.mutations) > 0

    def test_explored_response_is_disclosed_via_version_suffix(self, case, monkeypatch):
        """The ':explore' suffix is the disclosure contract downstream training
        data relies on to tell an exploratory probe from a real prediction."""
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        with load_test_scope(True):
            resp = case.app.mutate(_req())
        assert resp.metadata.model_version == f"{case.app.MODEL_VERSION}:explore"

    def test_unexplored_response_never_carries_the_suffix(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op, variant="stable", version="arn:...:model-package/v3")
        monkeypatch.setattr(case.app, "_EXPLORATION_EPSILON", 0.0)
        resp = case.app.mutate(_req())
        assert resp.metadata.model_version == "arn:...:model-package/v3"

    def test_never_fires_on_live_traffic_even_with_epsilon_forced_on(self, case, monkeypatch):
        """The structural gate: epsilon alone is not what protects live traffic.
        A call that is neither load-test-scoped nor explicitly opted in must
        never explore, even at epsilon=1.0."""
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        resp = case.app.mutate(_req())  # no load_test_scope -> real auction traffic
        assert resp.mutations == []
        assert not resp.metadata.model_version.endswith(":explore")


@BOTH
class TestExploreOverride:
    """The ext.model_params.explore channel the scenario card's Explore toggle
    uses to arm exploration on a single Send, without needing a load test."""

    def test_explore_true_alone_is_sufficient(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        resp = case.app.mutate(_req(ext={"model_params": {"explore": True}}))
        assert len(resp.mutations) > 0
        assert resp.metadata.model_version.endswith(":explore")

    def test_explore_false_wins_even_during_a_load_test(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        with load_test_scope(True):
            resp = case.app.mutate(_req(ext={"model_params": {"explore": False}}))
        assert resp.mutations == []
        assert not resp.metadata.model_version.endswith(":explore")

    def test_no_override_falls_back_to_the_load_test_signal(self, case, monkeypatch):
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        assert case.app.mutate(_req()).mutations == []

    def test_non_boolean_override_is_never_coerced(self, case, monkeypatch):
        """A stray string must be treated as absent, not silently read as
        True -- otherwise it would arm exploration on traffic the caller never
        asked to perturb."""
        case.patch_predict(monkeypatch, case.no_op)
        case.force_exploration(monkeypatch)
        assert case.app.mutate(_req(ext={"model_params": {"explore": "yes"}})).mutations == []


class TestFloorSpecificRules:
    def test_bidfloor_is_the_original_scaled_by_the_multiplier(self, monkeypatch):
        FLOOR.patch_predict(monkeypatch, 1.2)
        resp = floor_app.mutate(_req())
        premium = next(m for m in resp.mutations if m.path.endswith("deal-premium"))
        assert premium.adjust_deal.bidfloor == pytest.approx(12.0)  # 10.0 * 1.2

    def test_bidfloor_never_goes_negative(self, monkeypatch):
        """BR-7. A negative multiplier (a pathological prediction) must clamp to
        0.0, never emit a negative floor into the bidstream."""
        FLOOR.patch_predict(monkeypatch, -5.0)
        resp = floor_app.mutate(_req())
        assert resp.mutations
        assert all(m.adjust_deal.bidfloor >= 0.0 for m in resp.mutations)

    def test_a_missing_bidfloor_is_treated_as_zero_not_an_error(self, monkeypatch):
        FLOOR.patch_predict(monkeypatch, 1.5)
        resp = floor_app.mutate(_req(bid_request={
            "imp": [{"id": "imp-1", "pmp": {"deals": [{"id": "d", "at": 2}]}}],
        }))
        assert resp.mutations[0].adjust_deal.bidfloor == 0.0


class TestMarginSpecificRules:
    def test_first_price_deal_uses_cpm(self, monkeypatch):
        """BR-6: at=1 (guaranteed/first-price) -> CPM."""
        MARGIN.patch_predict(monkeypatch, 0.15)
        resp = margin_app.mutate(_req())
        premium = next(m for m in resp.mutations if m.path.endswith("deal-premium"))
        assert premium.adjust_deal.margin.calculation_type == MarginCalculationType.CPM

    def test_second_price_deal_uses_percent(self, monkeypatch):
        """BR-6: at=2 (open/second-price) -> PERCENT."""
        MARGIN.patch_predict(monkeypatch, 0.15)
        resp = margin_app.mutate(_req())
        remnant = next(m for m in resp.mutations if m.path.endswith("deal-remnant"))
        assert remnant.adjust_deal.margin.calculation_type == MarginCalculationType.PERCENT

    def test_missing_auction_type_defaults_to_percent(self, monkeypatch):
        """BR-6: an absent/unrecognized `at` defaults to PERCENT (the more
        common OpenRTB convention) rather than raising."""
        MARGIN.patch_predict(monkeypatch, 0.15)
        resp = margin_app.mutate(_req(bid_request={
            "imp": [{"id": "imp-1", "pmp": {"deals": [{"id": "d", "bidfloor": 1.0}]}}],
        }))
        assert resp.mutations[0].adjust_deal.margin.calculation_type == MarginCalculationType.PERCENT

    def test_margin_value_is_rounded_to_four_places(self, monkeypatch):
        MARGIN.patch_predict(monkeypatch, 0.123456789)
        resp = margin_app.mutate(_req())
        assert resp.mutations[0].adjust_deal.margin.value == 0.1235
