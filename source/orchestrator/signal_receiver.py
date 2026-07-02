"""Signal Receiver — REST endpoint for downstream signal ingestion.

Provides a ``POST /v1/signals`` endpoint that accepts impression, click,
and conversion signals and associates them with the originating bid by
``request_id``. Signals are:

1. Validated (UUID format, allowed signal_type, conversion_value rules)
2. Emitted to the Kinesis feedback stream as a ``SignalEvent`` record
   (fire-and-forget, same pattern as bid outcomes)
3. Passed to the in-memory ``SignalAssociator`` for immediate enrichment

The Glue ETL job joins SignalEvents with their originating BidOutcomeEvents
by ``request_id`` to produce complete training records.

Requirements: 1.4
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from shared.feedback_collector import FeedbackCollector
from shared.feedback_models import SignalEvent
from shared.signal_associator import DownstreamSignal, SignalAssociator, SignalType

logger = logging.getLogger(__name__)


async def receive_signal(
    request: Request,
    signal_associator: SignalAssociator,
    feedback_collector: Optional[FeedbackCollector],
) -> JSONResponse:
    """POST /v1/signals — receive a downstream signal and associate it with a bid.

    Accepts impression, click, or conversion signals identified by ``request_id``.
    Validates the payload, emits a ``SignalEvent`` to Kinesis for ETL joining,
    and passes the signal to the in-memory ``SignalAssociator`` for immediate
    enrichment of the cached bid context.

    Request body (JSON):
        {
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "impression" | "click" | "conversion",
            "conversion_value": 10.5,   // optional, only for conversion signals
            "timestamp": 1718000000.0   // optional, defaults to current time
        }

    Responses:
        200: Signal accepted and emitted
        400: Invalid JSON body
        422: Validation error (bad UUID, invalid signal_type, etc.)
    """
    # Parse JSON body
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body"}, status_code=400
        )

    # Default timestamp to current time if not provided
    if "timestamp" not in body or body.get("timestamp") is None:
        body["timestamp"] = time.time()

    # Validate via the SignalEvent model (UUID check, signal_type, conversion_value)
    try:
        signal_event = SignalEvent(**body)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Invalid signal payload: {exc}"}, status_code=422
        )

    # Fire-and-forget: emit the raw signal event to Kinesis for ETL joining
    if feedback_collector is not None:
        try:
            await _emit_signal_event(feedback_collector, signal_event)
        except Exception:
            # Never block the response — same pattern as bid outcome emission
            logger.warning(
                "Failed to emit signal event to Kinesis for request_id=%s",
                signal_event.request_id,
                exc_info=True,
            )

    # Also pass to the in-memory SignalAssociator for immediate enrichment
    try:
        downstream_signal = DownstreamSignal(
            request_id=signal_event.request_id,
            signal_type=SignalType(signal_event.signal_type),
            conversion_value=signal_event.conversion_value,
            timestamp=signal_event.timestamp,
        )
        associated = await signal_associator.handle_signal(downstream_signal)
    except Exception:
        # In-memory association failure doesn't block the response;
        # the ETL layer will perform the join by request_id.
        logger.warning(
            "In-memory signal association failed for request_id=%s",
            signal_event.request_id,
            exc_info=True,
        )
        associated = False

    return JSONResponse(
        {
            "status": "accepted",
            "request_id": signal_event.request_id,
            "signal_type": signal_event.signal_type,
            "associated": associated,
        },
        status_code=200,
    )


async def _emit_signal_event(
    collector: FeedbackCollector, event: SignalEvent
) -> None:
    """Emit a SignalEvent to the Kinesis feedback stream.

    Uses the same stream as bid outcomes but with ``record_type: "signal"``
    to distinguish during ETL processing. Partitions by ``request_id`` so
    that signals for the same bid land on the same shard for ordering.
    """
    import asyncio

    record_bytes = json.dumps(event.model_dump(), default=str).encode("utf-8")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: collector._kinesis_client.put_record(
            StreamName=collector._stream_name,
            Data=record_bytes,
            PartitionKey=event.request_id,
        ),
    )
