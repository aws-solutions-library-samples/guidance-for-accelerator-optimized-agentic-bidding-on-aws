"""Unit tests for orchestrator.feedback_integration module.

Validates that emit_bid_outcome correctly constructs BidOutcomeEvents from
RTBRequest/RTBResponse pairs, respects the fire-and-forget / non-blocking
contract, and gracefully handles the disabled-collector case.

Requirements: 1.1, 1.2, 1.6
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Ensure source/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.artf_types import AdjustBidPayload, Mutation, RTBRequest, RTBResponse, Metadata
from shared.feedback_models import BidOutcomeEvent


class TestBuildBidOutcomeEvent:
    """Tests for _build_bid_outcome_event event construction."""

    def _build(self, req: RTBRequest, resp: RTBResponse) -> BidOutcomeEvent:
        """Helper to import and call _build_bid_outcome_event."""
        from orchestrator.feedback_integration import _build_bid_outcome_event
        return _build_bid_outcome_event(req, resp, time.monotonic())

    def test_basic_event_construction(self):
        """A minimal request/response produces a valid BidOutcomeEvent."""
        req = RTBRequest(
            id="550e8400-e29b-41d4-a716-446655440000",
            bid_request={
                "imp": [{"bidfloor": 1.0}],
                "user": {"id": "user-123"},
                "site": {"domain": "example.com"},
                "device": {"devicetype": 2},
            },
            ext={"model_params": {"shade_factor": 0.7, "conversion_value": 10.0}},
        )
        mutation = Mutation(
            intent="BID_SHADE",
            adjust_bid=AdjustBidPayload(price=3.5),
        )
        resp = RTBResponse(id=req.id, mutations=[mutation])

        event = self._build(req, resp)

        assert event.request_id == "550e8400-e29b-41d4-a716-446655440000"
        assert event.shade_factor_used == 0.7
        assert event.conversion_value_estimate_used == 10.0
        assert event.shaded_price == 3.5
        assert event.original_price >= event.shaded_price
        assert event.bid_floor == 1.0
        assert event.site_domain == "example.com"
        assert event.device_type == "2"
        assert event.won is False
        assert event.impression is False
        assert event.click is False
        assert event.conversion is False
        assert event.price_paid is None
        assert event.conversion_value is None
        assert 0 <= event.hour_of_day <= 23
        # user_id_hash is a sha256 prefix of user-123
        assert event.user_id_hash != "unknown"
        assert len(event.user_id_hash) == 16

    def test_non_uuid_request_id_converted(self):
        """Non-UUID request IDs are converted to a deterministic UUID."""
        req = RTBRequest(
            id="not-a-uuid",
            bid_request={"imp": [{"bidfloor": 0.5}]},
        )
        resp = RTBResponse(id=req.id, mutations=[])

        event = self._build(req, resp)

        # Should be a valid UUID (uuid5 derived)
        import uuid
        uuid.UUID(event.request_id)  # Raises ValueError if not valid

    def test_missing_user_gives_unknown_hash(self):
        """When user.id is missing, user_id_hash should be 'unknown'."""
        req = RTBRequest(
            id="550e8400-e29b-41d4-a716-446655440000",
            bid_request={"imp": [{"bidfloor": 0.0}]},
        )
        resp = RTBResponse(id=req.id, mutations=[])

        event = self._build(req, resp)
        assert event.user_id_hash == "unknown"

    def test_default_model_params_when_ext_missing(self):
        """Default shade_factor=0.65 and conversion_value=5.0 when no ext."""
        req = RTBRequest(
            id="550e8400-e29b-41d4-a716-446655440000",
            bid_request={"imp": [{"bidfloor": 1.0}]},
        )
        resp = RTBResponse(id=req.id, mutations=[])

        event = self._build(req, resp)
        assert event.shade_factor_used == 0.65
        assert event.conversion_value_estimate_used == 5.0

    def test_price_ordering_enforced(self):
        """original_price >= shaded_price >= bid_floor is always maintained."""
        req = RTBRequest(
            id="550e8400-e29b-41d4-a716-446655440000",
            bid_request={"imp": [{"bidfloor": 2.0}]},
            ext={"model_params": {"shade_factor": 0.8, "conversion_value": 5.0}},
        )
        mutation = Mutation(
            intent="BID_SHADE",
            adjust_bid=AdjustBidPayload(price=2.5),
        )
        resp = RTBResponse(id=req.id, mutations=[mutation])

        event = self._build(req, resp)
        assert event.original_price >= event.shaded_price >= event.bid_floor

    def test_no_mutations_uses_bid_floor(self):
        """When there are no mutations, prices default to bid_floor."""
        req = RTBRequest(
            id="550e8400-e29b-41d4-a716-446655440000",
            bid_request={"imp": [{"bidfloor": 1.5}]},
        )
        resp = RTBResponse(id=req.id, mutations=[])

        event = self._build(req, resp)
        assert event.original_price == 1.5
        assert event.shaded_price == 1.5
        assert event.bid_floor == 1.5


class TestEmitBidOutcome:
    """Tests for emit_bid_outcome fire-and-forget behavior."""

    def test_noop_when_collector_is_none(self):
        """When FEEDBACK_STREAM_NAME is not set, emit is a silent no-op."""
        from orchestrator import feedback_integration

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = None
            # Should not raise
            feedback_integration.emit_bid_outcome(
                RTBRequest(id="550e8400-e29b-41d4-a716-446655440000", bid_request={}),
                RTBResponse(id="550e8400-e29b-41d4-a716-446655440000", mutations=[]),
                time.monotonic(),
            )
        finally:
            feedback_integration._feedback_collector = original

    def test_creates_async_task_when_collector_present(self):
        """When collector is present, emit creates an asyncio task."""
        from orchestrator import feedback_integration

        mock_collector = MagicMock()
        mock_collector.emit = AsyncMock()

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = mock_collector

            async def _run():
                feedback_integration.emit_bid_outcome(
                    RTBRequest(
                        id="550e8400-e29b-41d4-a716-446655440000",
                        bid_request={"imp": [{"bidfloor": 1.0}]},
                    ),
                    RTBResponse(id="550e8400-e29b-41d4-a716-446655440000", mutations=[]),
                    time.monotonic(),
                )
                # Allow the task to execute
                await asyncio.sleep(0.01)

            asyncio.run(_run())
            mock_collector.emit.assert_called_once()
            # Verify the emitted event is a BidOutcomeEvent
            emitted_event = mock_collector.emit.call_args[0][0]
            assert isinstance(emitted_event, BidOutcomeEvent)
        finally:
            feedback_integration._feedback_collector = original

    def test_never_raises_on_error(self):
        """emit_bid_outcome swallows all exceptions to protect the bid path."""
        from orchestrator import feedback_integration

        mock_collector = MagicMock()
        mock_collector.emit = AsyncMock(side_effect=RuntimeError("Kinesis down"))

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = mock_collector

            async def _run():
                # This must not raise
                feedback_integration.emit_bid_outcome(
                    RTBRequest(
                        id="550e8400-e29b-41d4-a716-446655440000",
                        bid_request={"imp": [{"bidfloor": 1.0}]},
                    ),
                    RTBResponse(id="550e8400-e29b-41d4-a716-446655440000", mutations=[]),
                    time.monotonic(),
                )
                # Allow the task to run (and fail internally)
                await asyncio.sleep(0.01)

            asyncio.run(_run())
            # If we got here, no exception was propagated
        finally:
            feedback_integration._feedback_collector = original
