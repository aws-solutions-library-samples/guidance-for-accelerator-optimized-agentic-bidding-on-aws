"""Tests for shared.feedback_collector — FeedbackCollector emit/emit_batch.

Validates:
- Single event emission to Kinesis with correct partition key
- Batch emission respecting the 500-record PutRecords limit
- Drop-oldest backpressure when the internal queue is full
- Error handling: log + CloudWatch metric + drop on Kinesis failure
- Partial failure handling (FailedRecordCount > 0)

**Validates: Requirements 1.1, 1.3, 1.5**
"""

import sys
import os
import asyncio
import json
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import BidShadingOutcomeEvent
from shared.feedback_collector import (
    FeedbackCollector,
    _MAX_RECORDS_PER_PUT,
    _CW_METRIC_NAMESPACE,
    _CW_METRIC_NAME,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures
# ---------------------------------------------------------------------------


def _make_event(user_id_hash: str = "user_abc123", request_id_suffix: str = "0") -> BidShadingOutcomeEvent:
    """Create a valid BidShadingOutcomeEvent with a given user_id_hash."""
    return BidShadingOutcomeEvent(
        request_id=f"a1b2c3d4-e5f6-7890-abcd-ef123456789{request_id_suffix}",
        timestamp=1718000000.0,
        model_version="v1.0.0",
        source="live",
        original_price=5.0,
        shaded_price=4.0,
        bid_floor=2.0,
        won=True,
        price_paid=3.5,
        impression=True,
        click=False,
        conversion=False,
        conversion_value=None,
        user_id_hash=user_id_hash,
        site_domain="example.com",
        device_type="mobile",
        hour_of_day=14,
        shade_factor_used=0.8,
        conversion_value_estimate_used=10.0,
    )


def _make_events(count: int, user_id_hash: str = "user_abc123") -> list[BidShadingOutcomeEvent]:
    """Create a list of valid BidOutcomeEvents."""
    events = []
    for i in range(count):
        # Use hex chars for the suffix to keep it a valid UUID
        suffix = f"{i:x}"[-1]  # single hex char
        events.append(_make_event(user_id_hash=user_id_hash, request_id_suffix=suffix))
    return events


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFeedbackCollectorEmit:
    """Tests for FeedbackCollector.emit (single event)."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Patch boto3.client to return mocks for kinesis and cloudwatch."""
        mock_kinesis = MagicMock()
        mock_cloudwatch = MagicMock()

        # Default successful response
        mock_kinesis.put_records.return_value = {
            "FailedRecordCount": 0,
            "Records": [{"SequenceNumber": "seq1", "ShardId": "shard-0"}],
        }

        def client_factory(service, **kwargs):
            if service == "kinesis":
                return mock_kinesis
            elif service == "cloudwatch":
                return mock_cloudwatch
            return MagicMock()

        with patch("boto3.client", side_effect=client_factory):
            yield mock_kinesis, mock_cloudwatch

    def test_emit_single_event_calls_put_records(self, mock_boto3_clients):
        """emit() writes one event to Kinesis via put_records."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event(user_id_hash="partition_key_user")
        asyncio.run(collector.emit(event))

        mock_kinesis.put_records.assert_called_once()
        call_kwargs = mock_kinesis.put_records.call_args[1]
        assert call_kwargs["StreamName"] == "test-stream"
        assert len(call_kwargs["Records"]) == 1
        assert call_kwargs["Records"][0]["PartitionKey"] == "partition_key_user"

    def test_emit_partitions_by_user_id_hash(self, mock_boto3_clients):
        """Partition key is set to the event's user_id_hash field."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event(user_id_hash="specific_user_hash")
        asyncio.run(collector.emit(event))

        call_kwargs = mock_kinesis.put_records.call_args[1]
        assert call_kwargs["Records"][0]["PartitionKey"] == "specific_user_hash"

    def test_emit_serializes_event_to_json(self, mock_boto3_clients):
        """Event data is serialized as JSON bytes."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event()
        asyncio.run(collector.emit(event))

        call_kwargs = mock_kinesis.put_records.call_args[1]
        data = call_kwargs["Records"][0]["Data"]
        assert isinstance(data, bytes)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["request_id"] == event.request_id
        assert parsed["user_id_hash"] == event.user_id_hash
        assert parsed["won"] is True

    def test_emit_queue_is_drained(self, mock_boto3_clients):
        """After emit(), the internal queue should be empty."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event()
        asyncio.run(collector.emit(event))

        assert collector.queue_size == 0


class TestFeedbackCollectorEmitBatch:
    """Tests for FeedbackCollector.emit_batch (batch writes)."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Patch boto3.client to return mocks for kinesis and cloudwatch."""
        mock_kinesis = MagicMock()
        mock_cloudwatch = MagicMock()

        mock_kinesis.put_records.return_value = {
            "FailedRecordCount": 0,
            "Records": [],
        }

        def client_factory(service, **kwargs):
            if service == "kinesis":
                return mock_kinesis
            elif service == "cloudwatch":
                return mock_cloudwatch
            return MagicMock()

        with patch("boto3.client", side_effect=client_factory):
            yield mock_kinesis, mock_cloudwatch

    def test_emit_batch_under_limit(self, mock_boto3_clients):
        """Batch of <500 events is sent in a single PutRecords call."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        events = _make_events(10)
        asyncio.run(collector.emit_batch(events))

        mock_kinesis.put_records.assert_called_once()
        call_kwargs = mock_kinesis.put_records.call_args[1]
        assert len(call_kwargs["Records"]) == 10

    def test_emit_batch_splits_at_500(self, mock_boto3_clients):
        """Batch of >500 events is split into multiple PutRecords calls."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        events = _make_events(501)
        asyncio.run(collector.emit_batch(events))

        # Should have 2 calls: one with 500, one with 1
        assert mock_kinesis.put_records.call_count == 2
        first_call = mock_kinesis.put_records.call_args_list[0][1]
        second_call = mock_kinesis.put_records.call_args_list[1][1]
        assert len(first_call["Records"]) == 500
        assert len(second_call["Records"]) == 1

    def test_emit_batch_exactly_500(self, mock_boto3_clients):
        """Batch of exactly 500 events is sent in a single PutRecords call."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        events = _make_events(500)
        asyncio.run(collector.emit_batch(events))

        mock_kinesis.put_records.assert_called_once()
        call_kwargs = mock_kinesis.put_records.call_args[1]
        assert len(call_kwargs["Records"]) == 500

    def test_emit_batch_empty_list(self, mock_boto3_clients):
        """Empty event list does not call PutRecords."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        asyncio.run(collector.emit_batch([]))

        mock_kinesis.put_records.assert_not_called()


class TestFeedbackCollectorBackpressure:
    """Tests for drop-oldest backpressure behavior."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Patch boto3.client to return mocks."""
        mock_kinesis = MagicMock()
        mock_cloudwatch = MagicMock()

        mock_kinesis.put_records.return_value = {
            "FailedRecordCount": 0,
            "Records": [],
        }

        def client_factory(service, **kwargs):
            if service == "kinesis":
                return mock_kinesis
            elif service == "cloudwatch":
                return mock_cloudwatch
            return MagicMock()

        with patch("boto3.client", side_effect=client_factory):
            yield mock_kinesis, mock_cloudwatch

    def test_drop_oldest_when_queue_full(self, mock_boto3_clients):
        """When the queue is full, oldest events are dropped on enqueue."""
        mock_kinesis, _ = mock_boto3_clients

        # Use a tiny queue for testing
        collector = FeedbackCollector(
            stream_name="test-stream",
            region="us-east-1",
            max_queue_size=3,
        )

        # Don't flush - directly enqueue to test backpressure
        event_a = _make_event(user_id_hash="user_a")
        event_b = _make_event(user_id_hash="user_b")
        event_c = _make_event(user_id_hash="user_c")
        event_d = _make_event(user_id_hash="user_d")

        collector._enqueue(event_a)
        collector._enqueue(event_b)
        collector._enqueue(event_c)
        assert collector.queue_size == 3

        # Adding a fourth drops the oldest (event_a)
        collector._enqueue(event_d)
        assert collector.queue_size == 3

        # The oldest item in queue should now be event_b
        oldest = collector._queue[0]
        assert oldest.user_id_hash == "user_b"

        # The newest should be event_d
        newest = collector._queue[-1]
        assert newest.user_id_hash == "user_d"

    def test_max_queue_size_property(self, mock_boto3_clients):
        """max_queue_size property returns configured value."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream",
            region="us-east-1",
            max_queue_size=5000,
        )
        assert collector.max_queue_size == 5000

    def test_default_queue_size(self, mock_boto3_clients):
        """Default max_queue_size is 10000."""
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )
        assert collector.max_queue_size == 10000


class TestFeedbackCollectorErrorHandling:
    """Tests for error handling: log, metric, drop behavior."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Patch boto3.client to return mocks."""
        mock_kinesis = MagicMock()
        mock_cloudwatch = MagicMock()

        def client_factory(service, **kwargs):
            if service == "kinesis":
                return mock_kinesis
            elif service == "cloudwatch":
                return mock_cloudwatch
            return MagicMock()

        with patch("boto3.client", side_effect=client_factory):
            yield mock_kinesis, mock_cloudwatch

    def test_kinesis_exception_logs_and_increments_metric(
        self, mock_boto3_clients, caplog
    ):
        """On Kinesis exception: logs error, increments CW metric, drops event."""
        mock_kinesis, mock_cloudwatch = mock_boto3_clients
        mock_kinesis.put_records.side_effect = Exception("Throttled")

        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event()
        with caplog.at_level("ERROR"):
            asyncio.run(collector.emit(event))

        # Event was dropped (queue empty)
        assert collector.queue_size == 0

        # Error was logged
        assert "Kinesis PutRecords failed" in caplog.text

        # CloudWatch metric was incremented
        mock_cloudwatch.put_metric_data.assert_called_once()
        metric_call = mock_cloudwatch.put_metric_data.call_args[1]
        assert metric_call["Namespace"] == _CW_METRIC_NAMESPACE
        assert metric_call["MetricData"][0]["MetricName"] == _CW_METRIC_NAME
        assert metric_call["MetricData"][0]["Value"] == 1

    def test_partial_failure_logs_and_increments_metric(
        self, mock_boto3_clients, caplog
    ):
        """On partial PutRecords failure: logs error, increments metric for failed count."""
        mock_kinesis, mock_cloudwatch = mock_boto3_clients
        mock_kinesis.put_records.return_value = {
            "FailedRecordCount": 3,
            "Records": [
                {"SequenceNumber": "seq1", "ShardId": "shard-0"},
                {"ErrorCode": "ProvisionedThroughputExceededException", "ErrorMessage": "Rate exceeded"},
                {"SequenceNumber": "seq2", "ShardId": "shard-0"},
                {"ErrorCode": "InternalFailure", "ErrorMessage": "Internal error"},
                {"SequenceNumber": "seq3", "ShardId": "shard-0"},
                {"ErrorCode": "ProvisionedThroughputExceededException", "ErrorMessage": "Rate exceeded"},
            ],
        }

        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        events = _make_events(6)
        with caplog.at_level("ERROR"):
            asyncio.run(collector.emit_batch(events))

        # Partial failure was logged
        assert "partial failure" in caplog.text
        assert "3/6" in caplog.text

        # CloudWatch metric was incremented with count=3
        mock_cloudwatch.put_metric_data.assert_called_once()
        metric_call = mock_cloudwatch.put_metric_data.call_args[1]
        assert metric_call["MetricData"][0]["Value"] == 3

    def test_no_retry_on_failure(self, mock_boto3_clients):
        """On failure, events are dropped — put_records is called only once."""
        mock_kinesis, _ = mock_boto3_clients
        mock_kinesis.put_records.side_effect = Exception("Stream not found")

        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event()
        asyncio.run(collector.emit(event))

        # Only one call — no retry
        mock_kinesis.put_records.assert_called_once()

    def test_cloudwatch_metric_failure_does_not_propagate(
        self, mock_boto3_clients
    ):
        """If CloudWatch metric emission fails, it does not raise."""
        mock_kinesis, mock_cloudwatch = mock_boto3_clients
        mock_kinesis.put_records.side_effect = Exception("Kinesis error")
        mock_cloudwatch.put_metric_data.side_effect = Exception("CW error")

        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event()
        # Should not raise even though both Kinesis and CW fail
        asyncio.run(collector.emit(event))
        assert collector.queue_size == 0

    def test_successful_emit_does_not_increment_metric(self, mock_boto3_clients):
        """On success, CloudWatch error metric is NOT incremented."""
        mock_kinesis, mock_cloudwatch = mock_boto3_clients
        mock_kinesis.put_records.return_value = {
            "FailedRecordCount": 0,
            "Records": [{"SequenceNumber": "seq1", "ShardId": "shard-0"}],
        }

        collector = FeedbackCollector(
            stream_name="test-stream", region="us-east-1"
        )

        event = _make_event()
        asyncio.run(collector.emit(event))

        mock_cloudwatch.put_metric_data.assert_not_called()
