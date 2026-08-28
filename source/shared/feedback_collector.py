"""Feedback Collector — emits bid outcome events to Kinesis Data Streams.

Writes outcome event records to Amazon Kinesis without blocking the
real-time bid path. Implements:

- Partitioning by the event's ``partition_key`` (``user_id_hash`` for bid
  outcomes, ``deal_id`` for deal-yield outcomes)
- Batching up to 500 records per ``PutRecords`` call
- Bounded internal queue with drop-oldest backpressure
- On write failure: log error, increment CloudWatch metric, drop event (no retry)

Carries more than one event type: BidShadingOutcomeEvent (via
orchestrator/feedback_integration.py and shared/signal_associator.py) and
DealYieldOutcomeEvent (via orchestrator/deal_yield_feedback.py), each to its own
stream. Events are therefore typed structurally, by the OutcomeEvent protocol
below, rather than as one concrete model -- reading a bid-only field name
directly is what silently broke deal-yield emission.

Requirements: 1.1, 1.3, 1.5
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any, Optional, Protocol, runtime_checkable

import boto3


@runtime_checkable
class OutcomeEvent(Protocol):
    """What FeedbackCollector needs of an event: a Kinesis partition key and a
    JSON-serializable dump. Both BidShadingOutcomeEvent and
    DealYieldOutcomeEvent satisfy this."""

    @property
    def partition_key(self) -> str: ...

    def model_dump(self) -> dict[str, Any]: ...


# Strong references to in-flight emit tasks. asyncio only holds a weak
# reference to a running task, so a task nothing awaits can be garbage
# collected mid-flight; keeping it here until completion prevents that.
_PENDING_EMITS: set[asyncio.Task] = set()


def fire_and_forget_emit(coro, *, description: str) -> asyncio.Task:
    """Schedule an emit coroutine the caller will not await, with failures
    logged instead of discarded.

    A bare ``asyncio.create_task(collector.emit(event))`` loses exceptions: the
    result is never retrieved, so a raise inside the coroutine surfaces only as
    a late, detached "Task exception was never retrieved" message -- if at all.
    That is precisely how deal-yield emission failed invisibly for every event
    while the caller went on counting each one as emitted. The done-callback
    below makes that failure mode loud at the point it happens.

    Still fire-and-forget: returns immediately and never raises into the
    caller's path.
    """
    task = asyncio.create_task(coro)
    _PENDING_EMITS.add(task)

    def _log_failure(finished: asyncio.Task) -> None:
        _PENDING_EMITS.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.error("%s failed: %r", description, exc, exc_info=exc)

    task.add_done_callback(_log_failure)
    return task

logger = logging.getLogger(__name__)

# Kinesis PutRecords limit
_MAX_RECORDS_PER_PUT = 500

# Default internal queue capacity (drop-oldest when full)
_DEFAULT_MAX_QUEUE_SIZE = 10000

# CloudWatch metric for emit errors
_CW_METRIC_NAMESPACE = "ARTF/FeedbackCollector"
_CW_METRIC_NAME = "EmitErrors"


class FeedbackCollector:
    """Emits bid outcome events to Kinesis Data Streams.

    Uses a bounded internal queue with drop-oldest backpressure to decouple
    event emission from the bid response path. On any Kinesis write failure,
    logs the error, increments a CloudWatch error metric, and drops the event
    without retrying.
    """

    def __init__(
        self,
        stream_name: str,
        region: str,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._stream_name = stream_name
        self._region = region
        self._max_queue_size = max_queue_size

        # Bounded deque for drop-oldest backpressure
        self._queue: deque[OutcomeEvent] = deque(maxlen=max_queue_size)

        # boto3 clients (sync — run in executor for async)
        self._kinesis_client = boto3.client(
            "kinesis", region_name=region
        )
        self._cloudwatch_client = boto3.client(
            "cloudwatch", region_name=region
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def emit(self, event: OutcomeEvent) -> None:
        """Write a single outcome event to Kinesis (async, fire-and-forget).

        The event is enqueued internally with drop-oldest backpressure, then
        written to Kinesis asynchronously. On failure the event is dropped.
        """
        self._enqueue(event)
        await self._flush_queue(batch_size=1)

    async def emit_batch(self, events: list[OutcomeEvent]) -> None:
        """Batch-write outcome events to Kinesis (up to 500 per PutRecords call).

        Events are enqueued with drop-oldest backpressure and flushed in
        batches of up to 500 records per PutRecords API call.
        """
        for event in events:
            self._enqueue(event)

        await self._flush_queue(batch_size=_MAX_RECORDS_PER_PUT)

    # ------------------------------------------------------------------
    # Internal queue management
    # ------------------------------------------------------------------

    def _enqueue(self, event: OutcomeEvent) -> None:
        """Add event to bounded queue. Oldest events are dropped when full."""
        # deque(maxlen=N) automatically drops the oldest item on append
        # when at capacity — this implements drop-oldest backpressure.
        self._queue.append(event)

    async def _flush_queue(self, batch_size: int) -> None:
        """Drain the internal queue in batches and write to Kinesis."""
        while self._queue:
            batch: list[OutcomeEvent] = []
            for _ in range(min(batch_size, len(self._queue))):
                batch.append(self._queue.popleft())

            await self._put_records(batch)

    # ------------------------------------------------------------------
    # Kinesis write
    # ------------------------------------------------------------------

    async def _put_records(self, events: list[OutcomeEvent]) -> None:
        """Write a batch of events to Kinesis via PutRecords.

        On any failure: logs, increments CloudWatch metric, drops events.
        """
        if not events:
            return

        # Built INSIDE the try: serialization and partition-key access can both
        # fail on a malformed or unexpected event type, and when this ran above
        # the try any such error escaped _put_records entirely. Since callers
        # invoke emit() via asyncio.create_task(...) without awaiting it, the
        # exception became an unretrieved task exception -- no records written,
        # nothing logged, no error metric. Keeping it here means every failure
        # goes through the documented log-and-drop path below.
        try:
            records = [
                {
                    "Data": self._serialize_event(event),
                    "PartitionKey": event.partition_key,
                }
                for event in events
            ]

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._kinesis_client.put_records(
                    StreamName=self._stream_name,
                    Records=records,
                ),
            )

            # Check for partial failures within the response
            failed_count = response.get("FailedRecordCount", 0)
            if failed_count > 0:
                logger.error(
                    "Kinesis PutRecords partial failure: %d/%d records failed "
                    "for stream %s",
                    failed_count,
                    len(records),
                    self._stream_name,
                )
                await self._increment_error_metric(count=failed_count)

        except Exception:
            # Counts off `events`, not `records`: when the failure happens while
            # building `records` that name is still unbound, and referencing it
            # here would raise NameError from inside the handler -- re-hiding
            # the very error this block exists to report.
            logger.exception(
                "Kinesis PutRecords failed for stream %s (%d records dropped)",
                self._stream_name,
                len(events),
            )
            await self._increment_error_metric(count=len(events))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_event(event: OutcomeEvent) -> bytes:
        """Serialize an outcome event to JSON bytes for Kinesis."""
        return json.dumps(event.model_dump(), default=str).encode("utf-8")

    # ------------------------------------------------------------------
    # CloudWatch error metric
    # ------------------------------------------------------------------

    async def _increment_error_metric(self, count: int = 1) -> None:
        """Increment the FeedbackCollector/EmitErrors CloudWatch metric."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=_CW_METRIC_NAMESPACE,
                    MetricData=[
                        {
                            "MetricName": _CW_METRIC_NAME,
                            "Value": count,
                            "Unit": "Count",
                        }
                    ],
                ),
            )
        except Exception:
            # Best-effort metric emission — do not propagate
            logger.warning(
                "Failed to emit CloudWatch error metric for FeedbackCollector"
            )

    # ------------------------------------------------------------------
    # Properties (useful for testing/inspection)
    # ------------------------------------------------------------------

    @property
    def queue_size(self) -> int:
        """Current number of events in the internal queue."""
        return len(self._queue)

    @property
    def max_queue_size(self) -> int:
        """Maximum capacity of the internal queue."""
        return self._max_queue_size
