"""Feedback integration — emits BidOutcomeEvents from the orchestrator bid path.

Provides a single public function ``emit_bid_outcome()`` that constructs a
BidOutcomeEvent from the RTBRequest/RTBResponse and fires it to Kinesis via
``asyncio.create_task`` (fire-and-forget, non-blocking, < 1 ms overhead).

The FeedbackCollector instance is lazily initialized as a module-level
singleton controlled by environment variables:

- ``FEEDBACK_STREAM_NAME`` — Kinesis stream name. If unset, emission is
  disabled and ``emit_bid_outcome()`` is a no-op.
- ``FEEDBACK_STREAM_REGION`` — AWS region (falls back to
  ``AWS_REGION`` → ``AWS_DEFAULT_REGION`` → ``us-east-1``).

Requirements: 1.1, 1.2, 1.6
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from shared.artf_types import RTBRequest, RTBResponse
from shared.feedback_collector import FeedbackCollector
from shared.feedback_models import BidOutcomeEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton FeedbackCollector
# ---------------------------------------------------------------------------

_FEEDBACK_STREAM_NAME: Optional[str] = os.environ.get("FEEDBACK_STREAM_NAME")
_FEEDBACK_REGION: str = os.environ.get(
    "FEEDBACK_STREAM_REGION",
    os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
)

_feedback_collector: Optional[FeedbackCollector] = None

if _FEEDBACK_STREAM_NAME:
    _feedback_collector = FeedbackCollector(
        stream_name=_FEEDBACK_STREAM_NAME,
        region=_FEEDBACK_REGION,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_bid_outcome(
    req: RTBRequest,
    resp: RTBResponse,
    start_time: float,
    *,
    source: str = "live",
    model_version: str | None = None,
) -> None:
    """Fire-and-forget: build and emit a BidOutcomeEvent via asyncio.create_task.

    Emits exactly one event per served bid. The ``asyncio.create_task`` call
    itself adds < 1 ms and never blocks the bid response path.

    This function NEVER raises — any error is logged and swallowed to ensure
    the bid response is never impacted.

    Parameters
    ----------
    req : RTBRequest
        The inbound bid request.
    resp : RTBResponse
        The assembled response with mutations.
    start_time : float
        The ``time.monotonic()`` value captured at the start of request
        processing (retained for future latency telemetry; not used for
        the event timestamp which uses wall-clock ``time.time()``).
    source : str
        "live" (default) for real auction traffic, "load_test" for
        orchestrator-initiated load-test traffic. Never set by any caller on
        the real bid-serving path other than the default.
    model_version : str | None
        The resolved served model version for this request (e.g. from the
        container's response metadata). Falls back to a static placeholder
        if not provided, so this call never fails when the caller doesn't
        have a resolved version available.
    """
    if _feedback_collector is None:
        return

    try:
        event = _build_bid_outcome_event(
            req, resp, start_time, source=source, model_version=model_version
        )
        asyncio.create_task(_feedback_collector.emit(event))
    except Exception:
        # Never impact the bid response path
        logger.warning("Failed to emit bid outcome event", exc_info=True)


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------


def emit_load_test_bid_outcome(
    *,
    request_id: str,
    model_version: str,
    model_type: str,
    won: bool,
    shaded_price: float,
    original_price: float,
    bid_floor: float,
    price_paid: float | None,
    impression: bool,
    click: bool,
    conversion: bool,
    conversion_value: float | None,
    shade_factor_used: float,
    conversion_value_estimate_used: float,
) -> None:
    """Fire-and-forget: emit a real, load-test-origin BidOutcomeEvent.

    Unlike emit_bid_outcome() (which derives win/impression/click/conversion
    as False, to be filled in later by downstream live signals — there ARE
    no live signals for a load test), this constructs the event directly
    from a load test's already-known synthetic outcome (generated via
    source/closed_loop_demo's existing scenario patterns — see
    orchestrator/loadtest_instrumentation.py). Uses the SAME
    FeedbackCollector/Kinesis fire-and-forget path emit_bid_outcome() uses
    (BR-7 — exactly once per targeted request) and NEVER raises, matching
    emit_bid_outcome()'s error-swallowing contract.
    """
    if _feedback_collector is None:
        return

    try:
        event = BidOutcomeEvent(
            request_id=request_id,
            timestamp=time.time(),
            model_version=model_version,
            model_type=model_type,
            source="load_test",
            original_price=original_price,
            shaded_price=shaded_price,
            bid_floor=bid_floor,
            won=won,
            price_paid=price_paid,
            impression=impression,
            click=click,
            conversion=conversion,
            conversion_value=conversion_value,
            # A real hash of the request_id (deterministic per request, no
            # actual user identity involved) rather than the literal string
            # "load-test" -- the ETL's validate_no_raw_pii() checks
            # user_id_hash against a "looks like a hash" pattern and would
            # otherwise drop every load-test record as suspected raw PII
            # (confirmed live: 100% of load-test records were dropped before
            # this fix). site_domain/device_type are not hash-checked columns,
            # so the literal "load-test" marker is fine for those.
            user_id_hash=hashlib.sha256(f"load-test-{request_id}".encode()).hexdigest()[:16],
            site_domain="load-test",
            device_type="load-test",
            hour_of_day=datetime.now(timezone.utc).hour,
            shade_factor_used=shade_factor_used,
            conversion_value_estimate_used=conversion_value_estimate_used,
        )
        asyncio.create_task(_feedback_collector.emit(event))
    except Exception:
        logger.warning("Failed to emit load-test bid outcome event", exc_info=True)


def _build_bid_outcome_event(
    req: RTBRequest,
    resp: RTBResponse,
    start_time: float,
    *,
    source: str = "live",
    model_version: str | None = None,
) -> BidOutcomeEvent:
    """Construct a BidOutcomeEvent from the RTBRequest/RTBResponse data.

    Extracts available context from the bid_request and model_params.
    Fields that arrive later (won, impression, click, conversion) default
    to False — they are updated asynchronously via downstream signals.

    ``start_time`` is ignored for the timestamp field (we use wall-clock
    time.time() instead) but retained in the signature for future use.

    ``model_version``, when provided, is the caller's resolved served model
    version (e.g. read from the container's response metadata, which itself
    resolves it from the Triton router's served_variant/served_model_version
    outputs — see source/containers/*/app.py and
    source/triton/router/model.py). Falls back to a static placeholder if
    not provided, so this never fails for callers that don't have a
    resolved version available yet.
    """
    bid_request = req.bid_request or {}
    imp_list = bid_request.get("imp", [{}])
    first_imp = imp_list[0] if imp_list else {}
    bid_floor = first_imp.get("bidfloor", 0.0)

    # Extract model parameters used (from ext.model_params if available)
    model_params = req.model_params or {}
    shade_factor_used = float(model_params.get("shade_factor", 0.65))
    conversion_value_estimate_used = float(
        model_params.get("conversion_value", 5.0)
    )

    # Compute original_price from mutations: look for BID_SHADE adjust_bid mutations
    original_price = bid_floor
    shaded_price = bid_floor
    for mutation in resp.mutations:
        if mutation.adjust_bid and mutation.adjust_bid.price:
            shaded_price = mutation.adjust_bid.price
            # Original is the unshaded price (shaded_price / shade_factor)
            if shade_factor_used > 0:
                original_price = shaded_price / shade_factor_used
            else:
                original_price = shaded_price
            break

    # Ensure price ordering: original_price >= shaded_price >= bid_floor
    original_price = max(original_price, shaded_price)
    shaded_price = max(shaded_price, bid_floor)
    original_price = max(original_price, bid_floor)

    # Extract user identity info for hashing
    user_data = bid_request.get("user", {})
    user_id_raw = (
        user_data.get("id", "") or user_data.get("buyeruid", "") or ""
    )
    user_id_hash = (
        hashlib.sha256(user_id_raw.encode()).hexdigest()[:16]
        if user_id_raw
        else "unknown"
    )

    # Extract context features
    site = bid_request.get("site", {})
    site_domain = site.get("domain", "unknown")
    device = bid_request.get("device", {})
    device_type = device.get("devicetype", "unknown")
    if isinstance(device_type, int):
        device_type = str(device_type)

    # Compute hour_of_day from current time
    hour_of_day = datetime.now(timezone.utc).hour

    # Generate a proper UUID request_id
    request_id = req.id
    # Ensure it's in UUID format for the BidOutcomeEvent validation
    try:
        uuid.UUID(request_id)
    except (ValueError, AttributeError):
        # If the original request id is not a UUID, create a deterministic one from it
        request_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(request_id)))

    return BidOutcomeEvent(
        request_id=request_id,
        timestamp=time.time(),
        model_version=model_version or "orchestrator-v1",
        source=source,
        original_price=original_price,
        shaded_price=shaded_price,
        bid_floor=bid_floor,
        won=False,
        price_paid=None,
        impression=False,
        click=False,
        conversion=False,
        conversion_value=None,
        user_id_hash=user_id_hash,
        site_domain=site_domain,
        device_type=device_type,
        hour_of_day=hour_of_day,
        shade_factor_used=shade_factor_used,
        conversion_value_estimate_used=conversion_value_estimate_used,
    )
