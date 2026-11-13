"""Feedback integration — emits BidOutcomeEvents from the orchestrator bid path.

Provides a single public function ``emit_bid_outcome()`` that constructs a
BidShadingOutcomeEvent from the RTBRequest/RTBResponse and fires it to Kinesis via
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
from shared.feedback_collector import FeedbackCollector, fire_and_forget_emit
from shared.feedback_models import BidShadingOutcomeEvent

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


# The SignalAssociator whose cache a later signal is looked up in. Injected by the
# orchestrator at import time (app.py owns the singleton, because POST /v1/signals
# must resolve the SAME instance this module registers into — two associators means
# every signal misses).
_signal_associator = None


def set_signal_associator(associator) -> None:
    """Give this module the associator that POST /v1/signals reads from."""
    global _signal_associator
    _signal_associator = associator


def get_feedback_collector() -> Optional[FeedbackCollector]:
    """The collector the bid path emits through, or None when emission is off.

    Exposed so the signal endpoint and the SignalAssociator write to the SAME stream
    the bid events went to. They must: the ETL joins a signal to its bid by
    `request_id` within one table, so a signal emitted to a different stream can never
    be joined, and nothing reports an error — the row simply stays unlabelled.
    """
    return _feedback_collector


def feedback_stream_name() -> Optional[str]:
    """The Kinesis stream name in use, or None when emission is disabled."""
    return _FEEDBACK_STREAM_NAME


#: The outcome simulator, built lazily on first use and only when switched on. A
#: deployment with a real signal feed never constructs one.
_outcome_simulator = None
_outcome_simulator_checked = False


def _get_outcome_simulator():
    """The simulator, or None when it is switched off or unusable.

    Resolved once per process. Returns None when OUTCOME_SIMULATOR_ENABLED is unset
    — which is the default, so a deployment gets no synthetic outcomes unless it asks
    for them.
    """
    global _outcome_simulator, _outcome_simulator_checked
    if _outcome_simulator_checked:
        return _outcome_simulator
    _outcome_simulator_checked = True

    if _signal_associator is None:
        return None
    try:
        try:
            from orchestrator.outcome_simulator import OutcomeSimulator, is_enabled
        except ImportError:  # pragma: no cover - container-relative import
            from container.outcome_simulator import OutcomeSimulator, is_enabled
        if not is_enabled():
            return None
        _outcome_simulator = OutcomeSimulator(_signal_associator)
        logger.warning(
            "Outcome simulator ENABLED: win=%.3f impression=%.3f click=%.3f "
            "conversion=%.3f. All outcomes are SYNTHETIC and labelled "
            "provenance='simulated'.",
            _outcome_simulator.config.win_rate,
            _outcome_simulator.config.impression_rate,
            _outcome_simulator.config.click_rate,
            _outcome_simulator.config.conversion_rate,
        )
    except Exception:
        logger.warning("Outcome simulator not available", exc_info=True)
        _outcome_simulator = None
    return _outcome_simulator


def reset_outcome_simulator() -> None:
    """Drop the resolved simulator so the next call re-reads the environment.

    For tests that toggle OUTCOME_SIMULATOR_ENABLED; not used in production.
    """
    global _outcome_simulator, _outcome_simulator_checked
    _outcome_simulator = None
    _outcome_simulator_checked = False


def register_bid_context(event: BidShadingOutcomeEvent) -> None:
    """Make a bid findable by a later signal.

    A no-op when no associator has been injected, which is the case in unit tests
    and in any deployment without the signal path wired — an unregistered bid means
    a dropped signal, not a failed bid.
    """
    if _signal_associator is None:
        return
    try:
        _signal_associator.register_bid(event)
    except Exception:
        logger.warning("Failed to register bid context for signal association",
                       exc_info=True)


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
    """Fire-and-forget: build and emit a BidShadingOutcomeEvent via asyncio.create_task.

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

        # Register the bid so a later signal can find it.
        #
        # SignalAssociator.register_bid had NO production caller — only a test — so
        # its cache was always empty and every signal arriving at POST /v1/signals
        # was dropped with "bid context not found". The signal path existed end to
        # end and could never complete. This is the missing call.
        register_bid_context(event)

        fire_and_forget_emit(
            _feedback_collector.emit(event),
            description=f"bid outcome emit (request={event.request_id})",
        )

        # Synthetic outcomes, only when explicitly switched on. Scheduled AFTER the
        # register above, since a signal for an unregistered bid is dropped.
        simulator = _get_outcome_simulator()
        if simulator is not None:
            fire_and_forget_emit(
                simulator.apply(event),
                description=f"simulated outcome (request={event.request_id})",
            )
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
    bid_request: dict | None = None,
) -> None:
    """Fire-and-forget: emit a real, load-test-origin BidShadingOutcomeEvent.

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
        event = BidShadingOutcomeEvent(
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
            # A load test's outcomes are generated from a scenario pattern, not
            # reported by anything that saw an auction. Labelled accordingly so a
            # model trained on load-test traffic is distinguishable from one trained
            # on observed outcomes.
            outcome_provenance="simulated",
            # A real hash of the request_id (deterministic per request, no
            # actual user identity involved) rather than the literal string
            # "load-test" -- the ETL's validate_no_raw_pii() checks
            # user_id_hash against a "looks like a hash" pattern and would
            # otherwise drop every load-test record as suspected raw PII
            # (confirmed live: 100% of load-test records were dropped before
            # this fix). site_domain/device_type are not hash-checked columns,
            # so the literal "load-test" marker is fine for those.
            user_id_hash=hashlib.sha256(f"load-test-{request_id}".encode()).hexdigest()[:16],
            **_load_test_context(bid_request),
            shade_factor_used=shade_factor_used,
            conversion_value_estimate_used=conversion_value_estimate_used,
        )
        fire_and_forget_emit(
            _feedback_collector.emit(event),
            description=f"load-test bid outcome emit (request={event.request_id})",
        )
    except Exception:
        logger.warning("Failed to emit load-test bid outcome event", exc_info=True)


def _load_test_context(bid_request: dict | None) -> dict:
    """The context columns for a load-test event, from the request that was sent.

    The load-test generator varies `site.domain`, `device.devicetype` and
    `device.geo.country` (see loadtest.py's `_device_block` and each profile's
    `domains` pool), so the faithful thing is to record what the request
    actually carried rather than a marker. Two categoricals held at a constant
    across a run give their embedding tables one value to separate, which is
    indistinguishable from a trained table that learned nothing.

    Falls back to the `"load-test"` marker when no request is supplied, so
    callers that have not been threaded through still produce a valid event.
    `site_domain` and `device_type` are not hash-checked by the ETL's
    `validate_no_raw_pii`, so the marker is safe for them -- unlike
    `user_id_hash`, which must look like a hash or every record is dropped.
    """
    if not bid_request:
        now = datetime.now(timezone.utc)
        return {
            "site_domain": "load-test",
            "device_type": "load-test",
            "geo_country": "",
            "has_video": False,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
        }

    imps = bid_request.get("imp") or [{}]
    first_imp = imps[0] if imps else {}
    site = bid_request.get("site") or bid_request.get("app") or {}
    device = bid_request.get("device") or {}
    geo = device.get("geo") or {}

    device_type = device.get("devicetype", "unknown")
    if isinstance(device_type, int):
        device_type = str(device_type)

    now = datetime.now(timezone.utc)
    return {
        "site_domain": site.get("domain", "unknown"),
        "device_type": device_type,
        "geo_country": geo.get("country", "") or "",
        "has_video": bool(first_imp.get("video")),
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
    }


def _build_bid_outcome_event(
    req: RTBRequest,
    resp: RTBResponse,
    start_time: float,
    *,
    source: str = "live",
    model_version: str | None = None,
) -> BidShadingOutcomeEvent:
    """Construct a BidShadingOutcomeEvent from the RTBRequest/RTBResponse data.

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

    # geo_country and has_video are read by the DLRM feature spec
    # (source/shared/dlrm_features.py). Absent encodes to "" / False, which the
    # spec maps to its reserved "absent" slot rather than to a real value's.
    geo = device.get("geo", {}) or {}
    geo_country = geo.get("country", "") or ""
    has_video = bool(first_imp.get("video"))

    # One clock read for both calendar fields, so hour_of_day and day_of_week
    # cannot straddle a midnight boundary between two separate calls.
    now = datetime.now(timezone.utc)
    hour_of_day = now.hour
    day_of_week = now.weekday()

    # Generate a proper UUID request_id
    request_id = req.id
    # Ensure it's in UUID format for the BidShadingOutcomeEvent validation
    try:
        uuid.UUID(request_id)
    except (ValueError, AttributeError):
        # If the original request id is not a UUID, create a deterministic one from it
        request_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(request_id)))

    return BidShadingOutcomeEvent(
        request_id=request_id,
        timestamp=time.time(),
        model_version=model_version or "orchestrator-v1",
        source=source,
        original_price=original_price,
        shaded_price=shaded_price,
        bid_floor=bid_floor,
        # Not known yet. This event is written AT BID TIME: the auction has not
        # resolved, and no impression, click or conversion could have occurred. These
        # used to be `False`, which the ETL read as confirmed negatives — the reason
        # every training dataset was entirely negative. The outcome arrives later, via
        # POST /v1/signals → SignalAssociator, which re-emits this event enriched.
        won=None,
        price_paid=None,
        impression=None,
        click=None,
        conversion=None,
        conversion_value=None,
        outcome_provenance="unresolved",
        user_id_hash=user_id_hash,
        site_domain=site_domain,
        device_type=device_type,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        geo_country=geo_country,
        has_video=has_video,
        shade_factor_used=shade_factor_used,
        conversion_value_estimate_used=conversion_value_estimate_used,
    )
