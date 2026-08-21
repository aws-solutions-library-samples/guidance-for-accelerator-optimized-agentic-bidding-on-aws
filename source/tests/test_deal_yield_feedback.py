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
