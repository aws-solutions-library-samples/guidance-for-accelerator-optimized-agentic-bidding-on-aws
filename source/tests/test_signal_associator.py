"""Tests for shared.signal_associator — SignalAssociator downstream signal handling.

Validates:
- Registering bid contexts and associating downstream signals by request_id
- Enriched BidShadingOutcomeEvent emission with correct signal flags
- Monotonic signal chain enforcement (conversion → click → impression → won)
- Out-of-order signal handling (e.g., conversion before click)
- TTL-based cache expiration
- LRU eviction when cache is full
- Signals dropped when bid context not found (expired or never registered)
- Multiple signals for the same bid accumulate correctly

**Validates: Requirements 1.4**
"""

import sys
import os
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import BidShadingOutcomeEvent
from shared.signal_associator import (
    SignalAssociator,
    SignalType,
    DownstreamSignal,
    _DEFAULT_TTL_SECONDS,
    _DEFAULT_MAX_ENTRIES,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures
# ---------------------------------------------------------------------------


def _make_bid_event(
    request_id: str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    won: bool = True,
    impression: bool = False,
    click: bool = False,
    conversion: bool = False,
    conversion_value: float | None = None,
    price_paid: float | None = 3.5,
) -> BidShadingOutcomeEvent:
    """Create a valid BidShadingOutcomeEvent fixture."""
    return BidShadingOutcomeEvent(
        request_id=request_id,
        timestamp=1718000000.0,
        model_version="v1.0.0",
        source="live",
        original_price=5.0,
        shaded_price=4.0,
        bid_floor=2.0,
        won=won,
        price_paid=price_paid,
        impression=impression,
        click=click,
        conversion=conversion,
        conversion_value=conversion_value,
        user_id_hash="user_abc123",
        site_domain="example.com",
        device_type="mobile",
        hour_of_day=14,
        shade_factor_used=0.8,
        conversion_value_estimate_used=10.0,
    )


def _make_signal(
    request_id: str = "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    signal_type: SignalType = SignalType.IMPRESSION,
    conversion_value: float | None = None,
    timestamp: float = 1718000100.0,
) -> DownstreamSignal:
    """Create a valid DownstreamSignal fixture."""
    return DownstreamSignal(
        request_id=request_id,
        signal_type=signal_type,
        conversion_value=conversion_value,
        timestamp=timestamp,
    )


def _make_mock_collector() -> MagicMock:
    """Create a mock FeedbackCollector with async emit."""
    mock = MagicMock()
    mock.emit = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Tests: DownstreamSignal model validation
# ---------------------------------------------------------------------------


class TestDownstreamSignalValidation:
    """Tests for DownstreamSignal pydantic validation."""

    def test_valid_impression_signal(self):
        """Valid impression signal is accepted."""
        signal = DownstreamSignal(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type=SignalType.IMPRESSION,
            timestamp=1718000100.0,
        )
        assert signal.signal_type == SignalType.IMPRESSION
        assert signal.conversion_value is None

    def test_valid_click_signal(self):
        """Valid click signal is accepted."""
        signal = DownstreamSignal(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type=SignalType.CLICK,
            timestamp=1718000100.0,
        )
        assert signal.signal_type == SignalType.CLICK

    def test_valid_conversion_signal_with_value(self):
        """Valid conversion signal with conversion_value is accepted."""
        signal = DownstreamSignal(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type=SignalType.CONVERSION,
            conversion_value=15.5,
            timestamp=1718000100.0,
        )
        assert signal.signal_type == SignalType.CONVERSION
        assert signal.conversion_value == 15.5

    def test_conversion_value_on_non_conversion_signal_rejected(self):
        """conversion_value on a non-conversion signal raises ValueError."""
        with pytest.raises(Exception):
            DownstreamSignal(
                request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                signal_type=SignalType.IMPRESSION,
                conversion_value=10.0,
                timestamp=1718000100.0,
            )

    def test_signal_type_enum_values(self):
        """SignalType enum has the expected values."""
        assert SignalType.IMPRESSION.value == "impression"
        assert SignalType.CLICK.value == "click"
        assert SignalType.CONVERSION.value == "conversion"


# ---------------------------------------------------------------------------
# Tests: SignalAssociator registration and association
# ---------------------------------------------------------------------------


class TestSignalAssociatorRegistration:
    """Tests for registering bids and associating signals."""

    def test_register_bid_adds_to_cache(self):
        """register_bid stores the event in the cache."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event()
        associator.register_bid(event)

        assert associator.cache_size == 1

    def test_register_multiple_bids(self):
        """Multiple bids can be registered with different request_ids."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        for i in range(5):
            event = _make_bid_event(
                request_id=f"a1b2c3d4-e5f6-7890-abcd-ef123456789{i}"
            )
            associator.register_bid(event)

        assert associator.cache_size == 5

    def test_handle_signal_returns_true_when_bid_found(self):
        """handle_signal returns True when the bid context exists."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event()
        associator.register_bid(event)

        signal = _make_signal(signal_type=SignalType.IMPRESSION)
        result = asyncio.run(associator.handle_signal(signal))

        assert result is True

    def test_handle_signal_returns_false_when_bid_not_found(self):
        """handle_signal returns False when no bid context exists."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        signal = _make_signal(
            request_id="ffffffff-ffff-ffff-ffff-ffffffffffff"
        )
        result = asyncio.run(associator.handle_signal(signal))

        assert result is False

    def test_handle_signal_emits_enriched_event(self):
        """handle_signal emits an enriched event via the collector."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(won=True, impression=False)
        associator.register_bid(event)

        signal = _make_signal(signal_type=SignalType.IMPRESSION)
        asyncio.run(associator.handle_signal(signal))

        mock_collector.emit.assert_called_once()
        emitted_event = mock_collector.emit.call_args[0][0]
        assert isinstance(emitted_event, BidShadingOutcomeEvent)
        assert emitted_event.impression is True
        assert emitted_event.request_id == event.request_id


# ---------------------------------------------------------------------------
# Tests: Signal enrichment and monotonic chain
# ---------------------------------------------------------------------------


class TestSignalEnrichment:
    """Tests for signal enrichment logic and monotonic chain filling."""

    def test_impression_signal_sets_impression_true(self):
        """Impression signal sets impression=True on the event."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(won=True, impression=False, click=False, conversion=False)
        associator.register_bid(event)

        signal = _make_signal(signal_type=SignalType.IMPRESSION)
        asyncio.run(associator.handle_signal(signal))

        emitted = mock_collector.emit.call_args[0][0]
        assert emitted.impression is True
        assert emitted.click is False
        assert emitted.conversion is False
        assert emitted.won is True

    def test_click_signal_fills_chain_impression(self):
        """Click signal sets click=True and also fills impression=True."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(won=True, impression=False, click=False)
        associator.register_bid(event)

        signal = _make_signal(signal_type=SignalType.CLICK)
        asyncio.run(associator.handle_signal(signal))

        emitted = mock_collector.emit.call_args[0][0]
        assert emitted.click is True
        assert emitted.impression is True  # filled by monotonic chain
        assert emitted.conversion is False
        assert emitted.won is True

    def test_conversion_signal_fills_full_chain(self):
        """Conversion signal fills the full chain: conversion → click → impression."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(
            won=True,
            impression=False,
            click=False,
            conversion=False,
        )
        associator.register_bid(event)

        signal = _make_signal(
            signal_type=SignalType.CONVERSION, conversion_value=25.0
        )
        asyncio.run(associator.handle_signal(signal))

        emitted = mock_collector.emit.call_args[0][0]
        assert emitted.conversion is True
        assert emitted.click is True  # filled by chain
        assert emitted.impression is True  # filled by chain
        assert emitted.won is True
        assert emitted.conversion_value == 25.0

    def test_impression_signal_sets_won_true(self):
        """Impression signal on a won=False bid sets won=True (monotonic chain)."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        # A bid that initially lost but receives an impression signal
        # (edge case: impression implies won in the monotonic chain)
        event = _make_bid_event(won=False, price_paid=None, impression=False)
        associator.register_bid(event)

        signal = _make_signal(signal_type=SignalType.IMPRESSION)
        asyncio.run(associator.handle_signal(signal))

        emitted = mock_collector.emit.call_args[0][0]
        assert emitted.impression is True
        assert emitted.won is True

    def test_sequential_signals_accumulate(self):
        """Multiple signals for the same bid accumulate correctly."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(won=True, impression=False, click=False, conversion=False)
        associator.register_bid(event)

        # First: impression
        signal_imp = _make_signal(signal_type=SignalType.IMPRESSION)
        asyncio.run(associator.handle_signal(signal_imp))

        first_emit = mock_collector.emit.call_args[0][0]
        assert first_emit.impression is True
        assert first_emit.click is False

        # Second: click
        signal_click = _make_signal(signal_type=SignalType.CLICK, timestamp=1718000200.0)
        asyncio.run(associator.handle_signal(signal_click))

        second_emit = mock_collector.emit.call_args[0][0]
        assert second_emit.impression is True
        assert second_emit.click is True
        assert second_emit.conversion is False

        # Third: conversion
        signal_conv = _make_signal(
            signal_type=SignalType.CONVERSION,
            conversion_value=30.0,
            timestamp=1718000300.0,
        )
        asyncio.run(associator.handle_signal(signal_conv))

        third_emit = mock_collector.emit.call_args[0][0]
        assert third_emit.impression is True
        assert third_emit.click is True
        assert third_emit.conversion is True
        assert third_emit.conversion_value == 30.0

    def test_out_of_order_conversion_before_click(self):
        """Conversion arriving before click correctly fills the chain."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(won=True, impression=False, click=False, conversion=False)
        associator.register_bid(event)

        # Conversion arrives first (out of order)
        signal = _make_signal(
            signal_type=SignalType.CONVERSION, conversion_value=20.0
        )
        asyncio.run(associator.handle_signal(signal))

        emitted = mock_collector.emit.call_args[0][0]
        # All chain flags should be set
        assert emitted.conversion is True
        assert emitted.click is True
        assert emitted.impression is True
        assert emitted.conversion_value == 20.0

    def test_preserves_original_event_fields(self):
        """Enriched event preserves all original fields except signals."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        event = _make_bid_event(won=True, impression=False)
        associator.register_bid(event)

        signal = _make_signal(signal_type=SignalType.IMPRESSION)
        asyncio.run(associator.handle_signal(signal))

        emitted = mock_collector.emit.call_args[0][0]
        assert emitted.request_id == event.request_id
        assert emitted.timestamp == event.timestamp
        assert emitted.model_version == event.model_version
        assert emitted.original_price == event.original_price
        assert emitted.shaded_price == event.shaded_price
        assert emitted.bid_floor == event.bid_floor
        assert emitted.price_paid == event.price_paid
        assert emitted.user_id_hash == event.user_id_hash
        assert emitted.site_domain == event.site_domain
        assert emitted.device_type == event.device_type
        assert emitted.hour_of_day == event.hour_of_day
        assert emitted.shade_factor_used == event.shade_factor_used
        assert emitted.conversion_value_estimate_used == event.conversion_value_estimate_used


# ---------------------------------------------------------------------------
# Tests: Cache TTL and LRU eviction
# ---------------------------------------------------------------------------


class TestSignalAssociatorCache:
    """Tests for TTL expiration and LRU eviction behavior."""

    def test_expired_entry_returns_not_found(self):
        """Expired cache entries are treated as not found."""
        mock_collector = _make_mock_collector()
        # Use a very short TTL
        associator = SignalAssociator(
            feedback_collector=mock_collector, ttl_seconds=0.01
        )

        event = _make_bid_event()
        associator.register_bid(event)

        # Wait for expiry
        time.sleep(0.02)

        signal = _make_signal(signal_type=SignalType.IMPRESSION)
        result = asyncio.run(associator.handle_signal(signal))

        assert result is False
        mock_collector.emit.assert_not_called()

    def test_lru_eviction_when_cache_full(self):
        """Oldest entries are evicted when cache reaches max_entries."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(
            feedback_collector=mock_collector, max_entries=3
        )

        # Register 4 bids — the first should be evicted
        for i in range(4):
            event = _make_bid_event(
                request_id=f"a1b2c3d4-e5f6-7890-abcd-ef123456789{i}"
            )
            associator.register_bid(event)

        assert associator.cache_size == 3

        # The first bid (index 0) should have been evicted
        signal_0 = _make_signal(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        )
        result = asyncio.run(associator.handle_signal(signal_0))
        assert result is False

        # The last bid (index 3) should still be present
        signal_3 = _make_signal(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567893"
        )
        result = asyncio.run(associator.handle_signal(signal_3))
        assert result is True

    def test_default_ttl_is_24_hours(self):
        """Default TTL matches the expected 24 hours."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)
        assert associator.ttl_seconds == _DEFAULT_TTL_SECONDS
        assert associator.ttl_seconds == 86400

    def test_cache_size_property(self):
        """cache_size property reflects current entries count."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        assert associator.cache_size == 0

        event = _make_bid_event()
        associator.register_bid(event)
        assert associator.cache_size == 1


# ---------------------------------------------------------------------------
# Tests: Signal drops and logging
# ---------------------------------------------------------------------------


class TestSignalAssociatorDropBehavior:
    """Tests for signal drop behavior when bid context not found."""

    def test_signal_for_unknown_request_id_is_dropped(self):
        """Signal for an unregistered request_id is dropped (returns False)."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        signal = _make_signal(
            request_id="ffffffff-ffff-ffff-ffff-ffffffffffff"
        )
        result = asyncio.run(associator.handle_signal(signal))

        assert result is False
        mock_collector.emit.assert_not_called()

    def test_signal_for_expired_bid_is_dropped(self):
        """Signal for an expired bid context is dropped."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(
            feedback_collector=mock_collector, ttl_seconds=0.01
        )

        event = _make_bid_event()
        associator.register_bid(event)

        time.sleep(0.02)

        signal = _make_signal(signal_type=SignalType.CLICK)
        result = asyncio.run(associator.handle_signal(signal))

        assert result is False
        mock_collector.emit.assert_not_called()

    def test_dropped_signal_logs_warning(self, caplog):
        """Dropped signals produce a warning log."""
        mock_collector = _make_mock_collector()
        associator = SignalAssociator(feedback_collector=mock_collector)

        signal = _make_signal(
            request_id="ffffffff-ffff-ffff-ffff-ffffffffffff"
        )
        with caplog.at_level("WARNING"):
            asyncio.run(associator.handle_signal(signal))

        assert "bid context not found" in caplog.text
        assert "ffffffff-ffff-ffff-ffff-ffffffffffff" in caplog.text
