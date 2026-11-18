"""Tests for orchestrator.deal_yield_feedback -- DealYieldOutcomeEvent
emission from the orchestrator bid path.

Mocks only the external service boundary (FeedbackCollector.emit / Kinesis),
per this project's testing convention (see source/tests/README.md).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.artf_types import (
    AdjustDealPayload, Intent, Margin, MarginCalculationType, Metadata,
    Mutation, Operation, RTBRequest, RTBResponse,
)
from shared.feedback_models import DealYieldOutcomeEvent

import orchestrator.deal_yield_feedback as deal_yield_feedback


_SAMPLE_BID_REQUEST = {
    "imp": [
        {
            "id": "imp-1",
            "pmp": {
                "deals": [
                    {"id": "deal-premium", "bidfloor": 10.0, "at": 1},
                ]
            },
        }
    ],
    "site": {"cat": ["IAB17"]},
}


def _req() -> RTBRequest:
    return RTBRequest(id="req-1", bid_request=_SAMPLE_BID_REQUEST)


def _resp_with_floor_mutation() -> RTBResponse:
    return RTBResponse(
        id="req-1",
        mutations=[
            Mutation(
                intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
                path="/imp/imp-1/deals/deal-premium",
                adjust_deal=AdjustDealPayload(bidfloor=12.0),
            )
        ],
        metadata=Metadata(model_version="v1"),
    )


def _resp_with_margin_mutation() -> RTBResponse:
    return RTBResponse(
        id="req-1",
        mutations=[
            Mutation(
                intent=Intent.ADJUST_DEAL_MARGIN, op=Operation.REPLACE,
                path="/imp/imp-1/deals/deal-premium",
                adjust_deal=AdjustDealPayload(
                    margin=Margin(value=0.1, calculation_type=MarginCalculationType.CPM)
                ),
            )
        ],
        metadata=Metadata(model_version="v1"),
    )


class TestParseDealPath:
    def test_valid_path(self):
        assert deal_yield_feedback._parse_deal_path("/imp/imp-1/deals/deal-1") == ("imp-1", "deal-1")

    def test_malformed_path_returns_none(self):
        assert deal_yield_feedback._parse_deal_path("/imp/imp-1") is None

    def test_empty_path_returns_none(self):
        assert deal_yield_feedback._parse_deal_path("") is None

    def test_none_path_returns_none(self):
        assert deal_yield_feedback._parse_deal_path(None) is None


class TestAdjustDealMutationsFilter:
    def test_filters_to_adjust_deal_only(self):
        muts = [
            Mutation(intent=Intent.BID_SHADE, op=Operation.REPLACE, path="/x", adjust_deal=None),
            Mutation(intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE, path="/y",
                      adjust_deal=AdjustDealPayload(bidfloor=1.0)),
        ]
        result = deal_yield_feedback._adjust_deal_mutations(muts)
        assert len(result) == 1
        assert result[0].intent == Intent.ADJUST_DEAL_FLOOR


class TestBuildDealYieldEvent:
    def test_builds_floor_event(self):
        req = _req()
        mutation = _resp_with_floor_mutation().mutations[0]
        event = deal_yield_feedback._build_deal_yield_event(
            req, mutation, source="live", model_version="v1"
        )
        assert event is not None
        assert event.intent == "ADJUST_DEAL_FLOOR"
        assert event.imp_id == "imp-1"
        assert event.deal_id == "deal-premium"
        assert event.original_bidfloor == 10.0
        assert event.adjusted_bidfloor == 12.0
        assert event.margin_value is None

    def test_builds_margin_event(self):
        req = _req()
        mutation = _resp_with_margin_mutation().mutations[0]
        event = deal_yield_feedback._build_deal_yield_event(
            req, mutation, source="live", model_version="v1"
        )
        assert event is not None
        assert event.intent == "ADJUST_DEAL_MARGIN"
        assert event.margin_value == 0.1
        assert event.adjusted_bidfloor is None

    def test_returns_none_for_malformed_path(self):
        req = _req()
        mutation = Mutation(
            intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
            path="/not/a/deal/path",
            adjust_deal=AdjustDealPayload(bidfloor=1.0),
        )
        event = deal_yield_feedback._build_deal_yield_event(
            req, mutation, source="live", model_version="v1"
        )
        assert event is None

    def test_returns_none_when_deal_not_found(self):
        req = _req()
        mutation = Mutation(
            intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
            path="/imp/imp-1/deals/nonexistent-deal",
            adjust_deal=AdjustDealPayload(bidfloor=1.0),
        )
        event = deal_yield_feedback._build_deal_yield_event(
            req, mutation, source="live", model_version="v1"
        )
        assert event is None

    def test_won_and_price_paid_are_unknown_defaults(self):
        req = _req()
        mutation = _resp_with_floor_mutation().mutations[0]
        event = deal_yield_feedback._build_deal_yield_event(
            req, mutation, source="live", model_version="v1"
        )
        assert event.won is False
        assert event.price_paid is None

    def test_source_load_test_propagates(self):
        req = _req()
        mutation = _resp_with_floor_mutation().mutations[0]
        event = deal_yield_feedback._build_deal_yield_event(
            req, mutation, source="load_test", model_version="v1"
        )
        assert event.source == "load_test"


class TestEmitDealYieldOutcome:
    def test_noop_when_collector_not_configured(self, monkeypatch):
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", None)
        # Must not raise even with a real mutation present.
        deal_yield_feedback.emit_deal_yield_outcome(_req(), _resp_with_floor_mutation())

    def test_noop_when_no_adjust_deal_mutations(self, monkeypatch):
        mock_collector = AsyncMock()
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", mock_collector)
        resp = RTBResponse(id="req-1", mutations=[], metadata=Metadata(model_version="v1"))
        deal_yield_feedback.emit_deal_yield_outcome(_req(), resp)
        mock_collector.emit.assert_not_called()

    def test_emits_for_each_adjust_deal_mutation(self, monkeypatch):
        mock_collector = AsyncMock()
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", mock_collector)

        async def _run():
            deal_yield_feedback.emit_deal_yield_outcome(_req(), _resp_with_floor_mutation())
            await asyncio.sleep(0.01)

        asyncio.run(_run())
        mock_collector.emit.assert_called_once()
        emitted = mock_collector.emit.call_args[0][0]
        assert isinstance(emitted, DealYieldOutcomeEvent)
        assert emitted.intent == "ADJUST_DEAL_FLOOR"

    def test_never_raises_on_internal_error(self, monkeypatch):
        monkeypatch.setattr(
            deal_yield_feedback, "_deal_yield_collector",
            type("Bad", (), {"emit": lambda self, e: 1 / 0})(),
        )
        # Must not raise even though the collector is broken.
        deal_yield_feedback.emit_deal_yield_outcome(_req(), _resp_with_floor_mutation())


# ---------------------------------------------------------------------------
# Property-based tests (PBT Partial mode)
# ---------------------------------------------------------------------------

_imp_ids = st.text(min_size=1, max_size=10, alphabet=st.characters(blacklist_characters="/"))
_deal_ids = st.text(min_size=1, max_size=10, alphabet=st.characters(blacklist_characters="/"))


class TestSynthesizeLoadTestOutcome:
    """Tests for the load-test-origin bootstrap path (Logic Flow 4's
    documented resolution to the genesis-model cold-start problem: a load
    test's own synthetic scenario provides a known outcome, mirroring
    emit_load_test_bid_outcome()'s precedent, rather than leaving
    won/price_paid unknown as the live path does)."""

    @staticmethod
    def _event(intent="ADJUST_DEAL_FLOOR", *, deal_id="deal-a", **kw):
        from shared.feedback_models import DealYieldOutcomeEvent

        base = dict(
            request_id="r", timestamp=1.0, model_version="v", source="load_test",
            imp_id="1", deal_id=deal_id, intent=intent, original_bidfloor=3.0,
            hour_of_day=12, day_of_week=3,
        )
        base.update(kw)
        return DealYieldOutcomeEvent(**base)

    @classmethod
    def _mean_revenue(cls, intent, n=800, **kw):
        total = 0.0
        for i in range(n):
            event = cls._event(intent, deal_id=f"deal-{i % 40}", **kw)
            _, _, revenue = deal_yield_feedback._synthesize_load_test_outcome(
                event, seed=42, request_index=i
            )
            total += revenue
        return total / n

    def test_deterministic_same_inputs(self):
        event = self._event(adjusted_bidfloor=3.3)
        r1 = deal_yield_feedback._synthesize_load_test_outcome(event, seed=7, request_index=5)
        r2 = deal_yield_feedback._synthesize_load_test_outcome(event, seed=7, request_index=5)
        assert r1 == r2

    def test_demand_is_fixed_by_the_seed_not_the_run(self):
        """Two runs replayed at one seed face identical demand, so a
        canary-vs-stable difference is attributable to the model. Keyed on
        run_id before, which made a fresh uuid4 per run the real variable."""
        wtp = deal_yield_feedback.buyer_willingness_to_pay
        assert wtp(7, 3, "deal-a") == wtp(7, 3, "deal-a")
        assert wtp(7, 3, "deal-a") != wtp(8, 3, "deal-a")
        assert "run_id" not in deal_yield_feedback._synthesize_load_test_outcome.__code__.co_varnames

    def test_different_deal_ids_can_differ(self):
        """Real per-deal variation across a run, not a constant stuck value."""
        outcomes = {
            deal_yield_feedback._synthesize_load_test_outcome(
                self._event(deal_id=f"deal-{i}", adjusted_bidfloor=3.3),
                seed=42, request_index=0,
            )
            for i in range(50)
        }
        assert len({o[0] for o in outcomes}) > 1

    def test_price_paid_none_and_zero_revenue_when_it_does_not_clear(self):
        cleared, price_paid, revenue = deal_yield_feedback._synthesize_load_test_outcome(
            self._event(adjusted_bidfloor=99.0), seed=7, request_index=3
        )
        assert (cleared, price_paid, revenue) == (False, None, 0.0)

    def test_raising_the_floor_reduces_the_clear_rate(self):
        """The model's decision has to move its own outcome -- the whole point
        of this change. A higher floor clears less often."""
        def clear_rate(floor):
            hits = sum(
                deal_yield_feedback._synthesize_load_test_outcome(
                    self._event(deal_id=f"deal-{i % 40}", adjusted_bidfloor=floor),
                    seed=42, request_index=i,
                )[0]
                for i in range(600)
            )
            return hits / 600

        assert clear_rate(2.0) > clear_rate(3.0) > clear_rate(4.5)

    def test_floor_revenue_has_an_interior_optimum(self):
        """Reserve-price tradeoff: too low earns little per clear, too high
        forfeits clears. Neither extreme may win."""
        floors = [1.5, 2.1, 2.7, 3.0, 3.3, 3.9, 4.5]
        scores = {f: self._mean_revenue("ADJUST_DEAL_FLOOR", adjusted_bidfloor=f) for f in floors}
        best = max(scores, key=scores.get)
        assert best not in (floors[0], floors[-1]), scores

    def test_margin_percent_revenue_has_an_interior_optimum(self):
        from shared.artf_types import MarginCalculationType

        takes = [-0.3, -0.1, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        scores = {
            t: self._mean_revenue(
                "ADJUST_DEAL_MARGIN",
                margin_value=t,
                margin_calculation_type=int(MarginCalculationType.PERCENT),
            )
            for t in takes
        }
        best = max(scores, key=scores.get)
        assert best not in (takes[0], takes[-1]), scores

    def test_a_negative_margin_loses_money(self):
        """A discount clears more often and earns less. No special case for it."""
        from shared.artf_types import MarginCalculationType

        assert self._mean_revenue(
            "ADJUST_DEAL_MARGIN", margin_value=-0.3,
            margin_calculation_type=int(MarginCalculationType.PERCENT),
        ) < 0

    def test_cpm_margin_earns_the_absolute_take(self):
        """CPM adds an absolute amount rather than scaling the floor
        (shared/artf_types.py: CPM=0, PERCENT=1)."""
        from shared.artf_types import MarginCalculationType

        event = self._event(
            "ADJUST_DEAL_MARGIN", margin_value=0.4,
            margin_calculation_type=int(MarginCalculationType.CPM),
        )
        price, revenue = deal_yield_feedback._effective_price_and_revenue(event)
        assert price == 3.0 + 0.4
        assert revenue == 0.4


class TestEmitLoadTestDealYieldOutcome:
    def test_noop_when_collector_not_configured(self, monkeypatch):
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", None)
        result = deal_yield_feedback.emit_load_test_deal_yield_outcome(
            _req(), _resp_with_floor_mutation(), run_id="run-1", request_index=0, seed=42
        )
        assert result == []

    def test_noop_when_no_adjust_deal_mutations(self, monkeypatch):
        mock_collector = AsyncMock()
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", mock_collector)
        resp = RTBResponse(id="req-1", mutations=[], metadata=Metadata(model_version="v1"))
        result = deal_yield_feedback.emit_load_test_deal_yield_outcome(
            _req(), resp, run_id="run-1", request_index=0, seed=42
        )
        assert result == []
        mock_collector.emit.assert_not_called()

    def test_emits_event_with_source_load_test_and_known_outcome(self, monkeypatch):
        mock_collector = AsyncMock()
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", mock_collector)

        async def _run():
            samples = deal_yield_feedback.emit_load_test_deal_yield_outcome(
                _req(), _resp_with_floor_mutation(), run_id="run-1", request_index=0, seed=42
            )
            await asyncio.sleep(0.01)
            return samples

        samples = asyncio.run(_run())
        assert len(samples) == 1
        mock_collector.emit.assert_called_once()
        emitted = mock_collector.emit.call_args[0][0]
        assert isinstance(emitted, DealYieldOutcomeEvent)
        assert emitted.source == "load_test"
        assert emitted.intent == "ADJUST_DEAL_FLOOR"
        # Unlike the live path, won/price_paid are NOT the unknown defaults --
        # they carry the load test's own known synthetic outcome.
        if emitted.won:
            assert emitted.price_paid is not None
        else:
            assert emitted.price_paid is None

    def test_returns_one_sample_per_emitted_event_both_intents(self, monkeypatch):
        """A request with both ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN
        mutations on the same deal (BR-5: independent, atomic) must emit
        two events and return two samples."""
        mock_collector = AsyncMock()
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", mock_collector)

        resp = RTBResponse(
            id="req-1",
            mutations=[
                _resp_with_floor_mutation().mutations[0],
                _resp_with_margin_mutation().mutations[0],
            ],
            metadata=Metadata(model_version="v1"),
        )

        async def _run():
            samples = deal_yield_feedback.emit_load_test_deal_yield_outcome(
                _req(), resp, run_id="run-1", request_index=0, seed=42
            )
            await asyncio.sleep(0.01)
            return samples

        samples = asyncio.run(_run())
        assert len(samples) == 2
        assert mock_collector.emit.call_count == 2

    def test_sample_value_matches_returned_event(self, monkeypatch):
        """The returned sample value is exactly price_paid-if-won-else-0.0
        for the event actually emitted -- not a re-derived/fabricated
        number (mirrors emit_load_test_outcome()'s existing contract)."""
        mock_collector = AsyncMock()
        monkeypatch.setattr(deal_yield_feedback, "_deal_yield_collector", mock_collector)

        async def _run():
            samples = deal_yield_feedback.emit_load_test_deal_yield_outcome(
                _req(), _resp_with_floor_mutation(), run_id="run-1", request_index=0, seed=42
            )
            await asyncio.sleep(0.01)
            return samples

        samples = asyncio.run(_run())
        emitted = mock_collector.emit.call_args[0][0]
        expected = emitted.price_paid if emitted.won and emitted.price_paid is not None else 0.0
        assert samples[0] == expected

    def test_never_raises_on_internal_error(self, monkeypatch):
        monkeypatch.setattr(
            deal_yield_feedback, "_deal_yield_collector",
            type("Bad", (), {"emit": lambda self, e: 1 / 0})(),
        )
        result = deal_yield_feedback.emit_load_test_deal_yield_outcome(
            _req(), _resp_with_floor_mutation(), run_id="run-1", request_index=0, seed=42
        )
        # Errors are swallowed per-mutation; no sample recorded for the
        # failed emission, but the call itself must not raise.
        assert result == []


class TestParseDealPathProperties:
    @given(imp_id=_imp_ids, deal_id=_deal_ids)
    @settings(max_examples=200)
    def test_round_trips_wellformed_path(self, imp_id, deal_id):
        """Property: a well-formed path always round-trips to the same
        (imp_id, deal_id) pair."""
        path = f"/imp/{imp_id}/deals/{deal_id}"
        result = deal_yield_feedback._parse_deal_path(path)
        assert result == (imp_id, deal_id)

    @given(garbage=st.text())
    @settings(max_examples=200)
    def test_never_raises_on_arbitrary_input(self, garbage):
        """Property: _parse_deal_path never raises on any string input."""
        deal_yield_feedback._parse_deal_path(garbage)  # must not raise
