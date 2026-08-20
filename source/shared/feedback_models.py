"""Feedback data models for the closed-loop learning system.

Defines the core event/record types used throughout the feedback pipeline:

- ``BidOutcomeEvent``: The real-time event emitted by the Feedback Collector
  after auction resolution (Kinesis ingest format).
- ``BidOutcomeRecord``: The enriched Parquet/S3 record produced by Glue ETL
  for training consumption.
- ``SignalEvent``: A downstream signal (impression/click/conversion) emitted
  to Kinesis for later ETL joining with the originating bid by ``request_id``.

Both models enforce the validation rules from the design document:
- ``request_id`` in UUID format
- ``original_price >= bid_floor >= 0``
- ``shaded_price`` between ``bid_floor`` and ``original_price``
- ``price_paid`` is null when ``won`` is false
- ``conversion_value`` is null when ``conversion`` is false
- Monotonic outcome signals: conversion → click → impression → won

Requirements: 1.4, 1.6, 2.5
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# UUID regex: standard 8-4-4-4-12 hex format
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Distinguishes real auction traffic from orchestrator-initiated load-test
# traffic. Every BidOutcomeEvent/BidOutcomeRecord carries this so downstream
# consumers (Glue ETL, training data, governance comparisons) never confuse
# the two. Required, immutable once set (see model_config frozen=True below).
OutcomeSource = Literal["live", "load_test"]

_VALID_MODEL_TYPES = frozenset(
    {"dlrm_bid_shader", "ncf_deal_manager", "widedeep_segment_activator"}
)


# ---------------------------------------------------------------------------
# BidOutcomeEvent – emitted by Feedback Collector to Kinesis
# ---------------------------------------------------------------------------


class BidOutcomeEvent(BaseModel):
    """A single bid outcome emitted after auction resolution.

    This is the canonical event format written to Kinesis by the
    Feedback Collector.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    request_id: str
    timestamp: float
    model_version: str
    source: OutcomeSource

    # Bid details
    original_price: float
    shaded_price: float
    bid_floor: float

    # Outcome
    won: bool
    price_paid: Optional[float] = None

    # Downstream signals
    impression: bool
    click: bool
    conversion: bool
    conversion_value: Optional[float] = None

    # Context features
    user_id_hash: str
    site_domain: str
    device_type: str
    hour_of_day: int = Field(ge=0, le=23)

    # Model parameters at time of bid
    shade_factor_used: float
    conversion_value_estimate_used: float

    @model_validator(mode="after")
    def _validate_bid_outcome_rules(self) -> "BidOutcomeEvent":
        _validate_common_fields(
            request_id=self.request_id,
            original_price=self.original_price,
            shaded_price=self.shaded_price,
            bid_floor=self.bid_floor,
            won=self.won,
            price_paid=self.price_paid,
            conversion=self.conversion,
            conversion_value=self.conversion_value,
            impression=self.impression,
            click=self.click,
        )
        return self


# ---------------------------------------------------------------------------
# BidOutcomeRecord – Parquet/S3 schema for training
# ---------------------------------------------------------------------------


class BidOutcomeRecord(BaseModel):
    """Schema for enriched Parquet storage in S3 (produced by Glue ETL)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    # Primary key
    request_id: str
    event_timestamp: int  # Unix millis

    # Bid context
    model_type: str
    model_version: str
    source: OutcomeSource
    intent: str

    # Pricing
    original_price: float
    shaded_price: float
    bid_floor: float
    price_paid: Optional[float] = None

    # Outcome signals
    won: bool
    impression: bool
    click: bool
    conversion: bool
    conversion_value: Optional[float] = None

    # Features (for retraining)
    user_id_hash: str
    site_domain_hash: str
    device_type: str
    geo_country: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    has_video: bool
    iab_categories: list[str] = Field(default_factory=list)

    # Parameters at time of bid
    shade_factor_used: float
    conversion_value_estimate: float

    # Partition keys
    partition_date: str
    partition_hour: int = Field(ge=0, le=23)

    @model_validator(mode="after")
    def _validate_bid_outcome_rules(self) -> "BidOutcomeRecord":
        # Validate model_type is one of the expected values
        if self.model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {sorted(_VALID_MODEL_TYPES)}, "
                f"got '{self.model_type}'"
            )

        _validate_common_fields(
            request_id=self.request_id,
            original_price=self.original_price,
            shaded_price=self.shaded_price,
            bid_floor=self.bid_floor,
            won=self.won,
            price_paid=self.price_paid,
            conversion=self.conversion,
            conversion_value=self.conversion_value,
            impression=self.impression,
            click=self.click,
        )
        return self


# ---------------------------------------------------------------------------
# Standalone validation function
# ---------------------------------------------------------------------------


def validate_bid_outcome(
    *,
    request_id: str,
    original_price: float,
    shaded_price: float,
    bid_floor: float,
    won: bool,
    price_paid: Optional[float],
    impression: bool,
    click: bool,
    conversion: bool,
    conversion_value: Optional[float],
) -> list[str]:
    """Validate bid outcome fields against design rules.

    Returns a list of validation error messages. An empty list means all
    rules pass. This function can be called independently of the pydantic
    models — useful for pre-validation in pipeline stages that do not
    instantiate the full model.

    Validation rules:
        1. request_id must be non-empty UUID format
        2. original_price >= bid_floor >= 0
        3. shaded_price between bid_floor and original_price
        4. price_paid is null if won == false
        5. conversion_value is null if conversion == false
        6. Monotonic outcomes: conversion → click → impression → won
    """
    errors: list[str] = []

    # Rule 1: request_id must be non-empty UUID format
    if not request_id or not _UUID_RE.match(request_id):
        errors.append(
            f"request_id must be a non-empty UUID, got '{request_id}'"
        )

    # Rule 2: original_price >= bid_floor >= 0
    if bid_floor < 0:
        errors.append(f"bid_floor must be >= 0, got {bid_floor}")
    if original_price < bid_floor:
        errors.append(
            f"original_price ({original_price}) must be >= bid_floor ({bid_floor})"
        )

    # Rule 3: shaded_price between bid_floor and original_price
    if shaded_price < bid_floor:
        errors.append(
            f"shaded_price ({shaded_price}) must be >= bid_floor ({bid_floor})"
        )
    if shaded_price > original_price:
        errors.append(
            f"shaded_price ({shaded_price}) must be <= original_price ({original_price})"
        )

    # Rule 4: price_paid is null if won == false
    if not won and price_paid is not None:
        errors.append("price_paid must be null when won is false")

    # Rule 5: conversion_value is null if conversion == false
    if not conversion and conversion_value is not None:
        errors.append("conversion_value must be null when conversion is false")

    # Rule 6: Monotonic outcome signals: conversion → click → impression → won
    if conversion and not click:
        errors.append(
            "Monotonic violation: conversion is true but click is false"
        )
    if click and not impression:
        errors.append(
            "Monotonic violation: click is true but impression is false"
        )
    if impression and not won:
        errors.append(
            "Monotonic violation: impression is true but won is false"
        )

    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_common_fields(
    *,
    request_id: str,
    original_price: float,
    shaded_price: float,
    bid_floor: float,
    won: bool,
    price_paid: Optional[float],
    conversion: bool,
    conversion_value: Optional[float],
    impression: bool,
    click: bool,
) -> None:
    """Run shared validation rules and raise ValueError on failure.

    Used internally by model validators.
    """
    errors = validate_bid_outcome(
        request_id=request_id,
        original_price=original_price,
        shaded_price=shaded_price,
        bid_floor=bid_floor,
        won=won,
        price_paid=price_paid,
        impression=impression,
        click=click,
        conversion=conversion,
        conversion_value=conversion_value,
    )
    if errors:
        raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# SignalEvent – downstream signal emitted to Kinesis for ETL joining
# ---------------------------------------------------------------------------

_VALID_SIGNAL_TYPES = frozenset({"impression", "click", "conversion"})


class SignalEvent(BaseModel):
    """A downstream signal (impression/click/conversion) to be joined with
    the originating bid by ``request_id`` at the Glue ETL layer.

    This event is emitted to the same Kinesis stream as BidOutcomeEvents
    but carries a ``record_type`` of ``"signal"`` to distinguish it during
    ETL processing.

    Requirements: 1.4
    """

    model_config = ConfigDict(frozen=True)

    request_id: str  # UUID linking to original bid
    signal_type: str  # "impression" | "click" | "conversion"
    timestamp: float  # When the signal occurred
    conversion_value: Optional[float] = None  # Only for conversion signals
    record_type: str = Field(default="signal", frozen=True)

    @model_validator(mode="after")
    def _validate_signal_event(self) -> "SignalEvent":
        """Validate signal event fields.

        Rules:
        - request_id must be a valid UUID
        - signal_type must be one of: impression, click, conversion
        - conversion_value must only be present when signal_type == "conversion"
        """
        # Validate request_id is a valid UUID
        if not self.request_id or not _UUID_RE.match(self.request_id):
            raise ValueError(
                f"request_id must be a valid UUID, got '{self.request_id}'"
            )

        # Validate signal_type
        if self.signal_type not in _VALID_SIGNAL_TYPES:
            raise ValueError(
                f"signal_type must be one of {sorted(_VALID_SIGNAL_TYPES)}, "
                f"got '{self.signal_type}'"
            )

        # Validate conversion_value only for conversion signals
        if self.signal_type != "conversion" and self.conversion_value is not None:
            raise ValueError(
                "conversion_value must only be provided when signal_type is 'conversion'"
            )

        return self
