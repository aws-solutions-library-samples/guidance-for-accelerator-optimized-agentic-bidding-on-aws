"""Tests for shared.feedback_models — BidOutcomeEvent, BidOutcomeRecord, and validation.

Validates the design's validation rules:
1. request_id must be non-empty UUID format
2. original_price >= bid_floor >= 0
3. shaded_price between bid_floor and original_price
4. price_paid is null if won == false
5. conversion_value is null if conversion == false
6. Monotonic outcome signals: conversion → click → impression → won

**Validates: Requirements 1.6, 2.5**
"""

import sys
import os
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import (
    BidOutcomeEvent,
    BidOutcomeRecord,
    validate_bid_outcome,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures
# ---------------------------------------------------------------------------


def _valid_event_kwargs() -> dict:
    """Minimal valid BidOutcomeEvent keyword arguments."""
    return {
        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "timestamp": 1718000000.0,
        "model_version": "v1.2.3",
        "source": "live",
        "original_price": 5.0,
        "shaded_price": 4.0,
        "bid_floor": 2.0,
        "won": True,
        "price_paid": 3.5,
        "impression": True,
        "click": False,
        "conversion": False,
        "conversion_value": None,
        "user_id_hash": "abc123hash",
        "site_domain": "example.com",
        "device_type": "mobile",
        "hour_of_day": 14,
        "shade_factor_used": 0.8,
        "conversion_value_estimate_used": 10.0,
    }


def _valid_record_kwargs() -> dict:
    """Minimal valid BidOutcomeRecord keyword arguments."""
    return {
        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "event_timestamp": 1718000000000,
        "model_type": "dlrm_bid_shader",
        "model_version": "v1.2.3",
        "source": "live",
        "intent": "BID_SHADE",
        "original_price": 5.0,
        "shaded_price": 4.0,
        "bid_floor": 2.0,
        "price_paid": 3.5,
        "won": True,
        "impression": True,
        "click": False,
        "conversion": False,
        "conversion_value": None,
        "user_id_hash": "abc123hash",
        "site_domain_hash": "domainhash456",
        "device_type": "mobile",
        "geo_country": "US",
        "hour_of_day": 14,
        "day_of_week": 3,
        "has_video": False,
        "iab_categories": ["IAB1"],
        "shade_factor_used": 0.8,
        "conversion_value_estimate": 10.0,
        "partition_date": "2025-06-10",
        "partition_hour": 14,
    }


# ---------------------------------------------------------------------------
# BidOutcomeEvent — valid construction
# ---------------------------------------------------------------------------


class TestBidOutcomeEventValid:
    """BidOutcomeEvent valid construction tests."""

    def test_create_valid_event(self):
        """A fully valid event is created without error."""
        event = BidOutcomeEvent(**_valid_event_kwargs())
        assert event.request_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert event.won is True
        assert event.price_paid == 3.5

    def test_model_type_defaults_to_none(self):
        """model_type defaults to None when omitted (live traffic can span
        multiple model types in one response -- no single attributable
        value, a real 'unknown', never fabricated)."""
        event = BidOutcomeEvent(**_valid_event_kwargs())
        assert event.model_type is None

    def test_model_type_set_for_load_test(self):
        """model_type is preserved when explicitly set (load-test path,
        where the target model type is known)."""
        kwargs = _valid_event_kwargs()
        kwargs["model_type"] = "dlrm_bid_shader"
        event = BidOutcomeEvent(**kwargs)
        assert event.model_type == "dlrm_bid_shader"

    def test_lost_bid_no_price_paid(self):
        """Lost bid with null price_paid passes validation."""
        kwargs = _valid_event_kwargs()
        kwargs["won"] = False
        kwargs["price_paid"] = None
        kwargs["impression"] = False
        kwargs["click"] = False
        kwargs["conversion"] = False
        event = BidOutcomeEvent(**kwargs)
        assert event.won is False
        assert event.price_paid is None

    def test_full_funnel_conversion(self):
        """Full funnel: won → impression → click → conversion with value."""
        kwargs = _valid_event_kwargs()
        kwargs["won"] = True
        kwargs["impression"] = True
        kwargs["click"] = True
        kwargs["conversion"] = True
        kwargs["conversion_value"] = 25.0
        event = BidOutcomeEvent(**kwargs)
        assert event.conversion_value == 25.0

    def test_source_live(self):
        """source='live' is accepted and preserved."""
        kwargs = _valid_event_kwargs()
        kwargs["source"] = "live"
        event = BidOutcomeEvent(**kwargs)
        assert event.source == "live"

    def test_source_load_test(self):
        """source='load_test' is accepted and preserved."""
        kwargs = _valid_event_kwargs()
        kwargs["source"] = "load_test"
        event = BidOutcomeEvent(**kwargs)
        assert event.source == "load_test"

    def test_source_required(self):
        """Omitting source raises a validation error (no default)."""
        kwargs = _valid_event_kwargs()
        del kwargs["source"]
        with pytest.raises(ValueError):
            BidOutcomeEvent(**kwargs)

    def test_source_invalid_value_rejected(self):
        """An unrecognized source value raises a validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["source"] = "synthetic"
        with pytest.raises(ValueError):
            BidOutcomeEvent(**kwargs)

    def test_source_immutable(self):
        """source cannot be reassigned after construction (model is frozen)."""
        event = BidOutcomeEvent(**_valid_event_kwargs())
        with pytest.raises(Exception):
            event.source = "load_test"


# ---------------------------------------------------------------------------
# BidOutcomeEvent — validation errors
# ---------------------------------------------------------------------------


class TestBidOutcomeEventValidation:
    """BidOutcomeEvent validation rule enforcement."""

    def test_invalid_request_id_empty(self):
        """Empty request_id raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["request_id"] = ""
        with pytest.raises(ValueError, match="request_id"):
            BidOutcomeEvent(**kwargs)

    def test_invalid_request_id_not_uuid(self):
        """Non-UUID request_id raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["request_id"] = "not-a-valid-uuid"
        with pytest.raises(ValueError, match="request_id"):
            BidOutcomeEvent(**kwargs)

    def test_original_price_below_bid_floor(self):
        """original_price < bid_floor raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["original_price"] = 1.0
        kwargs["bid_floor"] = 3.0
        kwargs["shaded_price"] = 2.0
        with pytest.raises(ValueError, match="original_price"):
            BidOutcomeEvent(**kwargs)

    def test_negative_bid_floor(self):
        """Negative bid_floor raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["bid_floor"] = -1.0
        kwargs["shaded_price"] = 0.0
        with pytest.raises(ValueError, match="bid_floor"):
            BidOutcomeEvent(**kwargs)

    def test_shaded_price_below_bid_floor(self):
        """shaded_price < bid_floor raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["shaded_price"] = 1.0
        kwargs["bid_floor"] = 2.0
        with pytest.raises(ValueError, match="shaded_price.*bid_floor"):
            BidOutcomeEvent(**kwargs)

    def test_shaded_price_above_original(self):
        """shaded_price > original_price raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["shaded_price"] = 6.0
        kwargs["original_price"] = 5.0
        with pytest.raises(ValueError, match="shaded_price.*original_price"):
            BidOutcomeEvent(**kwargs)

    def test_price_paid_not_null_on_loss(self):
        """price_paid set when won is false raises validation error."""
        kwargs = _valid_event_kwargs()
        kwargs["won"] = False
        kwargs["price_paid"] = 2.0
        kwargs["impression"] = False
        kwargs["click"] = False
        kwargs["conversion"] = False
        with pytest.raises(ValueError, match="price_paid must be null"):
            BidOutcomeEvent(**kwargs)

    def test_conversion_value_not_null_when_no_conversion(self):
        """conversion_value set when conversion is false raises error."""
        kwargs = _valid_event_kwargs()
        kwargs["conversion"] = False
        kwargs["conversion_value"] = 10.0
        with pytest.raises(ValueError, match="conversion_value must be null"):
            BidOutcomeEvent(**kwargs)

    def test_monotonic_conversion_without_click(self):
        """conversion=True but click=False violates monotonic rule."""
        kwargs = _valid_event_kwargs()
        kwargs["won"] = True
        kwargs["impression"] = True
        kwargs["click"] = False
        kwargs["conversion"] = True
        kwargs["conversion_value"] = 5.0
        with pytest.raises(ValueError, match="[Mm]onotonic"):
            BidOutcomeEvent(**kwargs)

    def test_monotonic_click_without_impression(self):
        """click=True but impression=False violates monotonic rule."""
        kwargs = _valid_event_kwargs()
        kwargs["won"] = True
        kwargs["impression"] = False
        kwargs["click"] = True
        kwargs["conversion"] = False
        with pytest.raises(ValueError, match="[Mm]onotonic"):
            BidOutcomeEvent(**kwargs)

    def test_monotonic_impression_without_won(self):
        """impression=True but won=False violates monotonic rule."""
        kwargs = _valid_event_kwargs()
        kwargs["won"] = False
        kwargs["price_paid"] = None
        kwargs["impression"] = True
        kwargs["click"] = False
        kwargs["conversion"] = False
        with pytest.raises(ValueError, match="[Mm]onotonic"):
            BidOutcomeEvent(**kwargs)

    def test_hour_of_day_out_of_range(self):
        """hour_of_day outside [0, 23] raises error."""
        kwargs = _valid_event_kwargs()
        kwargs["hour_of_day"] = 25
        with pytest.raises(ValueError):
            BidOutcomeEvent(**kwargs)


# ---------------------------------------------------------------------------
# BidOutcomeRecord — valid construction
# ---------------------------------------------------------------------------


class TestBidOutcomeRecordValid:
    """BidOutcomeRecord valid construction tests."""

    def test_create_valid_record(self):
        """A fully valid record is created without error."""
        record = BidOutcomeRecord(**_valid_record_kwargs())
        assert record.model_type == "dlrm_bid_shader"
        assert record.partition_date == "2025-06-10"

    def test_all_model_types_valid(self):
        """All three model types are accepted."""
        for model_type in [
            "dlrm_bid_shader",
            "ncf_deal_manager",
            "widedeep_segment_activator",
        ]:
            kwargs = _valid_record_kwargs()
            kwargs["model_type"] = model_type
            record = BidOutcomeRecord(**kwargs)
            assert record.model_type == model_type


# ---------------------------------------------------------------------------
# BidOutcomeRecord — validation errors
# ---------------------------------------------------------------------------


class TestBidOutcomeRecordValidation:
    """BidOutcomeRecord validation rule enforcement."""

    def test_invalid_model_type(self):
        """Unknown model_type raises validation error."""
        kwargs = _valid_record_kwargs()
        kwargs["model_type"] = "unknown_model"
        with pytest.raises(ValueError, match="model_type"):
            BidOutcomeRecord(**kwargs)

    def test_price_ordering_violated(self):
        """original_price < bid_floor raises validation error."""
        kwargs = _valid_record_kwargs()
        kwargs["original_price"] = 1.0
        kwargs["bid_floor"] = 3.0
        kwargs["shaded_price"] = 2.0
        with pytest.raises(ValueError, match="original_price"):
            BidOutcomeRecord(**kwargs)

    def test_monotonic_violation_in_record(self):
        """Monotonic violation detected in record."""
        kwargs = _valid_record_kwargs()
        kwargs["won"] = True
        kwargs["impression"] = True
        kwargs["click"] = True
        kwargs["conversion"] = True
        kwargs["conversion_value"] = 10.0
        # Now break monotonic: click without impression
        kwargs["impression"] = False
        with pytest.raises(ValueError, match="[Mm]onotonic"):
            BidOutcomeRecord(**kwargs)

    def test_day_of_week_out_of_range(self):
        """day_of_week outside [0, 6] raises error."""
        kwargs = _valid_record_kwargs()
        kwargs["day_of_week"] = 7
        with pytest.raises(ValueError):
            BidOutcomeRecord(**kwargs)


# ---------------------------------------------------------------------------
# Standalone validate_bid_outcome function
# ---------------------------------------------------------------------------


class TestValidateBidOutcome:
    """Tests for the standalone validate_bid_outcome() function."""

    def test_valid_input_returns_empty_list(self):
        """Valid input returns no errors."""
        errors = validate_bid_outcome(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            won=True,
            price_paid=3.5,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        assert errors == []

    def test_multiple_violations_all_reported(self):
        """Multiple violations are all reported in the returned list."""
        errors = validate_bid_outcome(
            request_id="bad-id",
            original_price=1.0,
            shaded_price=3.0,
            bid_floor=2.0,
            won=False,
            price_paid=1.5,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        # Should report: bad UUID, original < bid_floor, shaded > original,
        # price_paid not null on loss, impression without won
        assert len(errors) >= 4

    def test_uuid_validation(self):
        """Non-UUID request_id produces an error."""
        errors = validate_bid_outcome(
            request_id="not-a-uuid",
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            won=True,
            price_paid=3.5,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        assert any("request_id" in e for e in errors)

    def test_valid_uuid_passes(self):
        """Properly formatted UUID passes."""
        test_uuid = str(uuid.uuid4())
        errors = validate_bid_outcome(
            request_id=test_uuid,
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            won=True,
            price_paid=3.5,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        assert not any("request_id" in e for e in errors)

    def test_monotonic_chain_all_true(self):
        """Full chain conversion→click→impression→won is valid."""
        errors = validate_bid_outcome(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            won=True,
            price_paid=3.5,
            impression=True,
            click=True,
            conversion=True,
            conversion_value=20.0,
        )
        assert not any("Monotonic" in e for e in errors)

    def test_monotonic_chain_all_false(self):
        """All signals false (lost bid) is valid."""
        errors = validate_bid_outcome(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            won=False,
            price_paid=None,
            impression=False,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        assert not any("Monotonic" in e for e in errors)

    def test_edge_shaded_equals_bid_floor(self):
        """shaded_price == bid_floor is valid (lower bound)."""
        errors = validate_bid_outcome(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            original_price=5.0,
            shaded_price=2.0,
            bid_floor=2.0,
            won=True,
            price_paid=2.0,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        assert errors == []

    def test_edge_shaded_equals_original(self):
        """shaded_price == original_price is valid (upper bound)."""
        errors = validate_bid_outcome(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            original_price=5.0,
            shaded_price=5.0,
            bid_floor=2.0,
            won=True,
            price_paid=4.5,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
        )
        assert errors == []
