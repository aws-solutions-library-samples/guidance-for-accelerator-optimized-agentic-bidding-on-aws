"""Feedback data models for the closed-loop learning system.

Defines the core event/record types used throughout the feedback pipeline:

- ``BidShadingOutcomeEvent``: The real-time event emitted by the Feedback
  Collector after auction resolution (Kinesis ingest format), for DSP-side
  bid-shading outcomes (DLRM/NCF/Wide&Deep). Not applicable to SSP-side
  containers (e.g. deal floor/margin adjustment), which have no bid price
  to shade -- see a dedicated event type for those instead.
- ``BidShadingOutcomeRecord``: The enriched Parquet/S3 record produced by
  Glue ETL for training consumption.
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
# traffic. Every BidShadingOutcomeEvent/BidShadingOutcomeRecord carries this
# so downstream consumers (Glue ETL, training data, governance comparisons)
# never confuse the two. Required, immutable once set (see model_config
# frozen=True below).
OutcomeSource = Literal["live", "load_test"]

#: An outcome signal that may not be known yet.
#:
#: These fields used to be plain `bool`, so a bid-time event -- written before the
#: auction has resolved and long before any impression, click or conversion could
#: occur -- recorded `False` on every dimension. The ETL then read those as
#: CONFIRMED NEGATIVES, which is why every training dataset was entirely negative
#: and why a model trained on it learned to predict zero.
#:
#: None means "not known yet". False means "known not to have happened".
OutcomeSignal = Optional[bool]

#: Where a record's outcome signals came from.
#:
#: Carried on the record and into the model manifest so a model trained on
#: simulated outcomes can never be mistaken for one trained on observed ones. A
#: dataset's provenance mix is reported by the trainer's dataset gate.
#:
#:   unresolved -- no outcome has been reported for this bid yet
#:   simulated  -- from source/orchestrator/outcome_simulator.py, a labelled
#:                 simulator. NOT a real advertiser response.
#:   observed   -- reported by something that actually saw the outcome
OutcomeProvenance = Literal["unresolved", "simulated", "observed"]

_VALID_MODEL_TYPES = frozenset(
    {"dlrm_bid_shader", "ncf_deal_manager", "widedeep_segment_activator"}
)


# ---------------------------------------------------------------------------
# BidShadingOutcomeEvent – emitted by Feedback Collector to Kinesis
# ---------------------------------------------------------------------------


class BidShadingOutcomeEvent(BaseModel):
    """A single DSP-side bid-shading outcome emitted after auction resolution.

    This is the canonical event format written to Kinesis by the Feedback
    Collector for bid-shading containers (DLRM/NCF/Wide&Deep). It is not
    meaningful for SSP-side containers with no bid price to shade (e.g. a
    deal floor/margin adjustment container) -- see a dedicated event type
    for those instead of overloading this one.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    request_id: str
    timestamp: float
    model_version: str
    source: OutcomeSource
    # None for live traffic: a single bid response can fan out across
    # multiple containers/model types (see orchestrator/app.py's
    # all_mutations aggregation), so there is no single model_type to
    # attribute a live event to at this call site -- a real "unknown", never
    # fabricated. Always set for load-test traffic, where the target model
    # type is known (see orchestrator/loadtest_instrumentation.py).
    model_type: Optional[str] = None

    # Bid details
    original_price: float
    shaded_price: float
    bid_floor: float

    # Outcome. None until reported — see OutcomeSignal.
    won: OutcomeSignal = None
    price_paid: Optional[float] = None

    # Downstream signals
    impression: OutcomeSignal = None
    click: OutcomeSignal = None
    conversion: OutcomeSignal = None
    conversion_value: Optional[float] = None

    #: Where the signals above came from. Defaulted so stored events stay valid;
    #: the emit paths set it explicitly.
    outcome_provenance: OutcomeProvenance = "unresolved"

    # Context features
    #
    # These are the columns the DLRM feature spec reads
    # (source/shared/dlrm_features.py). A column the spec names must be carried
    # here, or the training side reads a default while the serving side supplies
    # a real value -- the two would agree on width and disagree on meaning.
    user_id_hash: str
    site_domain: str
    device_type: str
    hour_of_day: int = Field(ge=0, le=23)
    # Defaulted rather than required: events already stored, and the other
    # producers of this type, stay valid. The two emit paths populate them
    # (orchestrator/feedback_integration.py), so a defaulted value in a fresh
    # event means an emitter has not been updated.
    day_of_week: int = Field(ge=0, le=6, default=0)
    geo_country: str = ""
    has_video: bool = False

    # Model parameters at time of bid
    #
    # Recorded for provenance, deliberately NOT model features: a prediction
    # conditioned on the policy that produced its own training data cannot be
    # used to evaluate a change to that policy. Excluded by
    # dlrm_features.py's held-out set.
    shade_factor_used: float
    conversion_value_estimate_used: float

    @property
    def partition_key(self) -> str:
        """Kinesis partition key — user_id_hash, giving per-user ordering.

        FeedbackCollector reads this rather than a specific field name so it
        can carry more than one event type. It previously read .user_id_hash
        directly, which made every DealYieldOutcomeEvent raise AttributeError
        before any Kinesis call (see DealYieldOutcomeEvent.partition_key).
        """
        return self.user_id_hash

    @model_validator(mode="after")
    def _validate_bid_outcome_rules(self) -> "BidShadingOutcomeEvent":
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
# BidShadingOutcomeRecord – Parquet/S3 schema for training
# ---------------------------------------------------------------------------


class BidShadingOutcomeRecord(BaseModel):
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

    # Outcome signals. None until reported — see OutcomeSignal.
    won: OutcomeSignal = None
    impression: OutcomeSignal = None
    click: OutcomeSignal = None
    conversion: OutcomeSignal = None
    outcome_provenance: OutcomeProvenance = "unresolved"
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
    def _validate_bid_outcome_rules(self) -> "BidShadingOutcomeRecord":
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
# DealYieldOutcomeEvent / DealYieldOutcomeRecord -- deal floor/margin outcomes
#
# Deliberately NOT an extension of BidShadingOutcomeEvent: this event has no
# bid price to shade (it's an SSP-side floor/margin decision, not a DSP-side
# bid-shading decision), so overloading BidShadingOutcomeEvent with these
# fields would produce meaningless placeholder values for fields like
# shade_factor_used. Travels through its own Kinesis stream/Firehose/Glue
# table (see deployment/feedback_pipeline_cfn.yaml's DealYieldOutcomeStream)
# rather than sharing BidOutcomeStream, since Firehose's Glue-schema-based
# Parquet conversion silently drops fields absent from the target table's
# column list.
# ---------------------------------------------------------------------------

DealYieldSource = Literal["live", "load_test"]
DealYieldIntent = Literal["ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"]


class DealYieldOutcomeEvent(BaseModel):
    """A single deal floor/margin adjustment outcome, emitted by the
    orchestrator (never by a yield container itself -- a container only
    proposes mutations and has no visibility into what happens after).
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    request_id: str
    timestamp: float
    model_version: str
    source: DealYieldSource

    # Which deal, which intent
    imp_id: str
    deal_id: str
    intent: DealYieldIntent

    # Floor/margin decision
    original_bidfloor: float
    adjusted_bidfloor: Optional[float] = None  # set for ADJUST_DEAL_FLOOR
    margin_value: Optional[float] = None  # set for ADJUST_DEAL_MARGIN
    margin_calculation_type: Optional[int] = None  # set for ADJUST_DEAL_MARGIN

    # Outcome -- unknown ("live") until a future downstream-signal path
    # exists for deal-level outcomes; a real "unknown", never fabricated.
    won: bool = False
    price_paid: Optional[float] = None

    # Context features (same signals build_feature_vector() reads)
    auction_type: Optional[int] = None
    category_tier: float = 0.0
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)

    @property
    def partition_key(self) -> str:
        """Kinesis partition key — deal_id, giving per-deal ordering.

        This event has no user_id_hash: a deal-level floor/margin adjustment
        is not attributable to one user, and the meaningful ordering guarantee
        is per deal (a deal's successive floor/margin adjustments stay in
        sequence on one shard).

        FeedbackCollector used to read .user_id_hash directly, so emitting this
        event raised AttributeError before any Kinesis call. That line sat
        outside the method's try/except and ran inside an
        asyncio.create_task(...) whose result nothing awaited, so the failure
        surfaced nowhere: no records reached the stream, no error was logged,
        and callers still counted the event as an emitted sample.
        """
        return self.deal_id


class DealYieldOutcomeRecord(BaseModel):
    """Schema for enriched Parquet storage in S3 (produced by Glue ETL),
    mirroring BidShadingOutcomeRecord's pattern for this event's own
    fields."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    request_id: str
    event_timestamp: int  # Unix millis
    model_version: str
    source: DealYieldSource
    imp_id: str
    deal_id: str
    intent: str
    original_bidfloor: float
    adjusted_bidfloor: Optional[float] = None
    margin_value: Optional[float] = None
    margin_calculation_type: Optional[int] = None
    won: bool
    price_paid: Optional[float] = None
    auction_type: Optional[int] = None
    category_tier: float
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    partition_date: str
    partition_hour: int = Field(ge=0, le=23)


# ---------------------------------------------------------------------------
# Standalone validation function
# ---------------------------------------------------------------------------


def validate_bid_outcome(
    *,
    request_id: str,
    original_price: float,
    shaded_price: float,
    bid_floor: float,
    won: OutcomeSignal,
    price_paid: Optional[float],
    impression: OutcomeSignal,
    click: OutcomeSignal,
    conversion: OutcomeSignal,
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

    # Rules 4-6 read the signals as TRI-STATE: True, False, or None for "not
    # reported yet". `not won` would treat None as False, which is the conflation
    # this change exists to remove -- so each rule tests `is True` / `is not True`
    # explicitly.

    # Rule 4: price_paid only when the bid is known to have won
    if won is not True and price_paid is not None:
        state = "unknown" if won is None else "false"
        errors.append(f"price_paid must be null when won is {state}")

    # Rule 5: conversion_value only when a conversion is known to have happened
    if conversion is not True and conversion_value is not None:
        state = "unknown" if conversion is None else "false"
        errors.append(f"conversion_value must be null when conversion is {state}")

    # Rule 6: Monotonic outcome signals: conversion → click → impression → won.
    # A reported downstream signal implies its upstream HAPPENED, so an upstream
    # that is unknown is as much a violation as one that is false — a conversion
    # cannot coexist with an unknown click.
    for downstream_name, downstream, upstream_name, upstream in (
        ("conversion", conversion, "click", click),
        ("click", click, "impression", impression),
        ("impression", impression, "won", won),
    ):
        if downstream is True and upstream is not True:
            state = "unknown" if upstream is None else "false"
            errors.append(
                f"Monotonic violation: {downstream_name} is true but "
                f"{upstream_name} is {state}"
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
    won: OutcomeSignal,
    price_paid: Optional[float],
    conversion: OutcomeSignal,
    conversion_value: Optional[float],
    impression: OutcomeSignal,
    click: OutcomeSignal,
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
    #: Where this signal came from. Travels with the signal into the enriched
    #: outcome event and on into the training record, so a dataset's provenance
    #: mix is recoverable at training time.
    provenance: OutcomeProvenance = "observed"

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
