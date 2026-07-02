"""Feedback Collector — emits bid outcome events to Kinesis Data Streams.

Writes BidOutcomeEvent records to Amazon Kinesis without blocking the
real-time bid path. Implements:

- Partitioning by ``user_id_hash`` for per-user ordering
- Batching up to 500 records per ``PutRecords`` call
- Bounded internal queue with drop-oldest backpressure
- On write failure: log error, increment CloudWatch metric, drop event (no retry)

Requirements: 1.1, 1.3, 1.5
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Optional

import boto3

from shared.feedback_models import BidOutcomeEvent

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
        self._queue: deque[BidOutcomeEvent] = deque(maxlen=max_queue_size)

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

    async def emit(self, event: BidOutcomeEvent) -> None:
        """Write a single outcome event to Kinesis (async, fire-and-forget).

        The event is enqueued internally with drop-oldest backpressure, then
        written to Kinesis asynchronously. On failure the event is dropped.
        """
        self._enqueue(event)
        await self._flush_queue(batch_size=1)

    async def emit_batch(self, events: list[BidOutcomeEvent]) -> None:
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

    def _enqueue(self, event: BidOutcomeEvent) -> None:
        """Add event to bounded queue. Oldest events are dropped when full."""
        # deque(maxlen=N) automatically drops the oldest item on append
        # when at capacity — this implements drop-oldest backpressure.
        self._queue.append(event)

    async def _flush_queue(self, batch_size: int) -> None:
        """Drain the internal queue in batches and write to Kinesis."""
        while self._queue:
            batch: list[BidOutcomeEvent] = []
            for _ in range(min(batch_size, len(self._queue))):
                batch.append(self._queue.popleft())

            await self._put_records(batch)

    # ------------------------------------------------------------------
    # Kinesis write
    # ------------------------------------------------------------------

    async def _put_records(self, events: list[BidOutcomeEvent]) -> None:
        """Write a batch of events to Kinesis via PutRecords.

        On any failure: logs, increments CloudWatch metric, drops events.
        """
        if not events:
            return

        records = [
            {
                "Data": self._serialize_event(event),
                "PartitionKey": event.user_id_hash,
            }
            for event in events
        ]

        try:
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
            logger.exception(
                "Kinesis PutRecords failed for stream %s (%d records dropped)",
                self._stream_name,
                len(records),
            )
            await self._increment_error_metric(count=len(records))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_event(event: BidOutcomeEvent) -> bytes:
        """Serialize a BidOutcomeEvent to JSON bytes for Kinesis."""
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
