"""Regression tests for deal-yield outcome emission reaching Kinesis.

The bug these cover, observed on a live stack: FeedbackCollector built its
Kinesis records with ``event.user_id_hash``. BidShadingOutcomeEvent has that
field; DealYieldOutcomeEvent does not (a deal-level floor/margin adjustment is
not attributable to one user). So every deal-yield event raised AttributeError.

Three properties made it invisible rather than loud:

1. the record-building line sat OUTSIDE _put_records' try/except, so the error
   bypassed the documented log-and-drop path
2. callers invoked emit() via ``asyncio.create_task(...)`` without awaiting, so
   the exception was never retrieved and never logged
3. callers appended to their sample list immediately after create_task, before
   the task ran, so a run reported N emitted samples with 0 records written

Net effect on the deployed stack: nv5-deal-yield-outcome-stream had zero
IncomingRecords ever, raw-deal-yield-outcomes/ in S3 held only a .keep file, and
the deal-yield Glue job failed every run with "Unable to infer schema for
Parquet" because its input prefix was empty -- which in turn meant no yield
load-test run could ever become trainable.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.feedback_collector import FeedbackCollector  # noqa: E402
from shared.feedback_models import (  # noqa: E402
    BidShadingOutcomeEvent,
    DealYieldOutcomeEvent,
)


def _deal_yield_event(**overrides) -> DealYieldOutcomeEvent:
    base = dict(
        request_id="req-1",
        timestamp=1_787_000_000.0,
        model_version="v1",
        source="load_test",
        imp_id="1",
        deal_id="deal-abc",
        intent="ADJUST_DEAL_FLOOR",
        original_bidfloor=1.0,
        adjusted_bidfloor=1.25,
        hour_of_day=12,
        day_of_week=3,
    )
    base.update(overrides)
    return DealYieldOutcomeEvent(**base)


def _bid_event(**overrides) -> BidShadingOutcomeEvent:
    # BidShadingOutcomeEvent validates request_id as a non-empty UUID.
    base = dict(
        request_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        timestamp=1_787_000_000.0,
        model_type="dlrm_bid_shader",
        model_version="v1",
        source="load_test",
        original_price=2.0,
        shaded_price=1.5,
        bid_floor=1.0,
        won=True,
        price_paid=1.5,
        impression=True,
        click=False,
        conversion=False,
        user_id_hash="user-xyz",
        site_domain="example.com",
        device_type="mobile",
        hour_of_day=12,
        shade_factor_used=0.75,
        conversion_value_estimate_used=10.0,
    )
    base.update(overrides)
    return BidShadingOutcomeEvent(**base)


@pytest.fixture
def kinesis():
    """Patch boto3 so no AWS call is made, capturing put_records kwargs."""
    calls: list[dict] = []

    def _put_records(**kwargs):
        calls.append(kwargs)
        return {"FailedRecordCount": 0}

    client = mock.Mock(put_records=_put_records, put_metric_data=mock.Mock())
    with mock.patch("boto3.client", return_value=client):
        yield calls


# ---------------------------------------------------------------------------
# The actual regression: a deal-yield event must reach Kinesis
# ---------------------------------------------------------------------------

def test_deal_yield_event_reaches_kinesis(kinesis):
    collector = FeedbackCollector(stream_name="deal-yield-stream", region="us-east-1")
    asyncio.run(collector.emit(_deal_yield_event()))

    assert len(kinesis) == 1, "deal-yield event must produce exactly one PutRecords call"
    assert kinesis[0]["StreamName"] == "deal-yield-stream"
    assert len(kinesis[0]["Records"]) == 1


def test_deal_yield_partition_key_is_deal_id(kinesis):
    """Per-deal ordering: a deal's successive adjustments stay on one shard."""
    collector = FeedbackCollector(stream_name="deal-yield-stream", region="us-east-1")
    asyncio.run(collector.emit(_deal_yield_event(deal_id="deal-999")))

    assert kinesis[0]["Records"][0]["PartitionKey"] == "deal-999"


def test_deal_yield_payload_keeps_distinguishing_fields(kinesis):
    """The fields that justify a separate stream must survive serialization."""
    collector = FeedbackCollector(stream_name="deal-yield-stream", region="us-east-1")
    asyncio.run(collector.emit(_deal_yield_event(intent="ADJUST_DEAL_MARGIN", margin_value=0.2)))

    payload = json.loads(kinesis[0]["Records"][0]["Data"].decode("utf-8"))
    assert payload["deal_id"] == "deal-abc"
    assert payload["intent"] == "ADJUST_DEAL_MARGIN"
    assert payload["margin_value"] == 0.2


def test_emit_does_not_raise_for_deal_yield_event(kinesis):
    """The precise old failure: AttributeError escaping emit()."""
    collector = FeedbackCollector(stream_name="deal-yield-stream", region="us-east-1")

    async def go():
        task = asyncio.create_task(collector.emit(_deal_yield_event()))
        await asyncio.sleep(0)
        await task
        return task

    task = asyncio.run(go())
    assert task.exception() is None


# ---------------------------------------------------------------------------
# Bid-outcome behavior must be unchanged
# ---------------------------------------------------------------------------

def test_bid_partition_key_still_user_id_hash(kinesis):
    """Existing per-user ordering must be preserved by the partition_key move."""
    collector = FeedbackCollector(stream_name="bid-stream", region="us-east-1")
    asyncio.run(collector.emit(_bid_event(user_id_hash="specific-user")))

    assert kinesis[0]["Records"][0]["PartitionKey"] == "specific-user"


def test_both_event_types_expose_partition_key():
    assert _bid_event().partition_key == "user-xyz"
    assert _deal_yield_event().partition_key == "deal-abc"


# ---------------------------------------------------------------------------
# Failures must be logged and dropped, never escape
# ---------------------------------------------------------------------------

def test_unserializable_event_is_logged_and_dropped_not_raised(caplog):
    """Record building now happens inside the try. An event that cannot supply
    a partition key must take the log-and-drop path rather than escaping as an
    unretrieved task exception."""

    class Broken:
        @property
        def partition_key(self):
            raise AttributeError("no partition key")

        def model_dump(self):
            return {}

    client = mock.Mock(put_records=mock.Mock(), put_metric_data=mock.Mock())
    with mock.patch("boto3.client", return_value=client):
        collector = FeedbackCollector(stream_name="s", region="us-east-1")
        with caplog.at_level("ERROR"):
            asyncio.run(collector.emit(Broken()))

    assert client.put_records.call_count == 0
    assert "PutRecords failed" in caplog.text
    assert collector.queue_size == 0, "event must be dropped, not left queued"


def test_error_metric_counts_events_when_record_building_fails():
    """The except handler must not reference the unbound `records` name -- doing
    so would raise NameError from inside the handler and re-hide the error."""

    class Broken:
        @property
        def partition_key(self):
            raise AttributeError("no partition key")

        def model_dump(self):
            return {}

    client = mock.Mock(put_records=mock.Mock(), put_metric_data=mock.Mock())
    with mock.patch("boto3.client", return_value=client):
        collector = FeedbackCollector(stream_name="s", region="us-east-1")
        asyncio.run(collector.emit_batch([Broken(), Broken()]))

    client.put_metric_data.assert_called()
    metric = client.put_metric_data.call_args[1]["MetricData"][0]
    assert metric["Value"] == 2


# ---------------------------------------------------------------------------
# fire_and_forget_emit surfaces failures instead of discarding them
# ---------------------------------------------------------------------------

def test_fire_and_forget_emit_logs_coroutine_failure(caplog):
    from shared.feedback_collector import fire_and_forget_emit

    async def boom():
        raise RuntimeError("kinesis exploded")

    async def go():
        task = fire_and_forget_emit(boom(), description="test emit")
        await asyncio.gather(task, return_exceptions=True)

    with caplog.at_level("ERROR"):
        asyncio.run(go())

    assert "test emit" in caplog.text
    assert "kinesis exploded" in caplog.text


def test_fire_and_forget_emit_returns_immediately_and_runs_coroutine():
    from shared.feedback_collector import fire_and_forget_emit

    ran: list[bool] = []

    async def work():
        ran.append(True)

    async def go():
        task = fire_and_forget_emit(work(), description="test emit")
        assert not ran, "must not block the caller waiting for the coroutine"
        await task

    asyncio.run(go())
    assert ran == [True]


def test_pending_emits_does_not_leak_after_completion():
    """Tasks are strongly referenced while in flight, then released."""
    from shared.feedback_collector import _PENDING_EMITS, fire_and_forget_emit

    async def work():
        await asyncio.sleep(0)

    async def go():
        task = fire_and_forget_emit(work(), description="test emit")
        assert task in _PENDING_EMITS
        await task

    before = len(_PENDING_EMITS)
    asyncio.run(go())
    assert len(_PENDING_EMITS) == before
