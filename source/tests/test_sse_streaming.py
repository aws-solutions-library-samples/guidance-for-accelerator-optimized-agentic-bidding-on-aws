"""Unit tests for SSE streaming endpoint (Task 7.2).

Validates:
- GET /v1/loadtest/{id}/stream returns text/event-stream content type
- Progress events are emitted with correct SSE format
- Complete event is emitted with full LoadTestStatus when test finishes
- 404 returned for non-existent test IDs
- Already-completed tests emit a single 'complete' event immediately

**Validates: Requirements 10.5, 10.6**
"""

import asyncio
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from orchestrator.loadtest import (
    LoadTestStatus,
    _active_tests,
    _cancel_flags,
    _progress_latencies,
    _progress_errors,
    _progress_completed,
    _progress_start_time,
    _progress_per_container_latencies,
    _progress_per_container_mutations,
    _sse_event_generator,
    stream_loadtest,
)
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import StreamingResponse


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_completed_status(test_id: str = "lt-test123") -> LoadTestStatus:
    """Create a completed LoadTestStatus for testing."""
    return LoadTestStatus(
        id=test_id,
        state="complete",
        preset="1k",
        total_requests=1000,
        completed=1000,
        errors=5,
        elapsed_ms=2100.0,
        rps=476.19,
        latency_p50=4.2,
        latency_p95=12.1,
        latency_p99=28.7,
        latency_min=1.1,
        latency_avg=6.3,
        latency_max=45.2,
        histogram={"lt_10ms": 800, "10_30ms": 150, "30_50ms": 40, "gt_50ms": 10},
        per_container=[
            {"name": "dlrm-bid-shader", "avg_latency_ms": 3.5, "total_mutations": 1000},
            {"name": "metrics-enricher", "avg_latency_ms": 2.1, "total_mutations": 2000},
        ],
    )


def _make_running_status(test_id: str = "lt-running123") -> LoadTestStatus:
    """Create a running LoadTestStatus for testing."""
    return LoadTestStatus(
        id=test_id,
        state="running",
        preset="1k",
        total_requests=1000,
        completed=0,
        errors=0,
        elapsed_ms=0.0,
        rps=0.0,
        latency_p50=0.0,
        latency_p95=0.0,
        latency_p99=0.0,
        latency_min=0.0,
        latency_avg=0.0,
        latency_max=0.0,
        histogram={"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0},
        per_container=[],
    )


def _cleanup_test_data(test_id: str) -> None:
    """Remove test data from shared state."""
    _active_tests.pop(test_id, None)
    _cancel_flags.pop(test_id, None)
    _progress_latencies.pop(test_id, None)
    _progress_errors.pop(test_id, None)
    _progress_completed.pop(test_id, None)
    _progress_start_time.pop(test_id, None)
    _progress_per_container_latencies.pop(test_id, None)
    _progress_per_container_mutations.pop(test_id, None)


# Create a minimal Starlette app for testing the stream endpoint
test_app = Starlette(routes=[
    Route("/v1/loadtest/{id}/stream", stream_loadtest, methods=["GET"]),
])


# ---------------------------------------------------------------------------
# Tests: SSE event format
# ---------------------------------------------------------------------------

class TestSSEEventFormat:
    """Verify SSE events follow the correct format."""

    def test_completed_test_returns_complete_event(self):
        """A completed test emits a single 'complete' event immediately."""
        test_id = "lt-completed-fmt"
        _active_tests[test_id] = _make_completed_status(test_id)

        try:
            client = TestClient(test_app)
            with client.stream("GET", f"/v1/loadtest/{test_id}/stream") as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

                # Read the full response
                content = response.read().decode()

            # Verify SSE format: "event: complete\ndata: {...}\n\n"
            assert "event: complete\n" in content
            assert "data: " in content

            # Extract the data payload
            lines = content.strip().split("\n")
            data_line = [l for l in lines if l.startswith("data: ")][0]
            payload = json.loads(data_line[len("data: "):])

            assert payload["id"] == test_id
            assert payload["state"] == "complete"
            assert payload["completed"] == 1000
            assert payload["total_requests"] == 1000
            assert payload["latency_p50"] == 4.2
            assert payload["histogram"]["lt_10ms"] == 800
        finally:
            _cleanup_test_data(test_id)

    def test_404_for_nonexistent_test(self):
        """Non-existent test ID returns 404."""
        client = TestClient(test_app)
        response = client.get("/v1/loadtest/lt-nonexistent/stream")
        assert response.status_code == 404
        assert "not found" in response.json()["error"].lower()

    def test_progress_event_format(self):
        """Progress events contain the expected fields."""
        test_id = "lt-progress-fmt"
        _active_tests[test_id] = _make_running_status(test_id)
        _progress_latencies[test_id] = [5.0, 8.0, 12.0, 25.0, 55.0]
        _progress_errors[test_id] = 1
        _progress_completed[test_id] = 5
        _progress_start_time[test_id] = time.monotonic() - 1.0  # 1 second ago

        try:
            # Use the generator directly to test one iteration
            gen = _sse_event_generator(test_id)

            async def _get_first_event():
                event = await gen.__anext__()
                # After getting first event, mark test as complete to stop generator
                _active_tests[test_id] = _make_completed_status(test_id)
                return event

            event = asyncio.run(_get_first_event())

            # Verify SSE format
            assert event.startswith("event: progress\n")
            assert "data: " in event
            assert event.endswith("\n\n")

            # Extract and verify payload
            data_line = [l for l in event.strip().split("\n") if l.startswith("data: ")][0]
            payload = json.loads(data_line[len("data: "):])

            assert "completed" in payload
            assert payload["completed"] == 5
            assert payload["total"] == 1000
            assert "rps" in payload
            assert "elapsed_ms" in payload
            assert "latency_p50" in payload
            assert "latency_p95" in payload
            assert "latency_p99" in payload
            assert "errors" in payload
            assert payload["errors"] == 1
            assert "histogram" in payload
            assert "lt_10ms" in payload["histogram"]
            assert "10_30ms" in payload["histogram"]
            assert "30_50ms" in payload["histogram"]
            assert "gt_50ms" in payload["histogram"]
        finally:
            _cleanup_test_data(test_id)


# ---------------------------------------------------------------------------
# Tests: SSE streaming behavior
# ---------------------------------------------------------------------------

class TestSSEStreamingBehavior:
    """Verify SSE streaming behavior and lifecycle."""

    def test_stream_response_headers(self):
        """SSE response has correct headers for streaming."""
        test_id = "lt-headers"
        _active_tests[test_id] = _make_completed_status(test_id)

        try:
            client = TestClient(test_app)
            with client.stream("GET", f"/v1/loadtest/{test_id}/stream") as response:
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                assert response.headers["cache-control"] == "no-cache"
                response.read()
        finally:
            _cleanup_test_data(test_id)

    def test_cancelled_test_emits_complete_event(self):
        """A cancelled test emits a 'complete' event with state='cancelled'."""
        test_id = "lt-cancelled"
        status = _make_completed_status(test_id)
        status.state = "cancelled"
        status.completed = 500
        _active_tests[test_id] = status

        try:
            client = TestClient(test_app)
            with client.stream("GET", f"/v1/loadtest/{test_id}/stream") as response:
                content = response.read().decode()

            data_line = [l for l in content.strip().split("\n") if l.startswith("data: ")][0]
            payload = json.loads(data_line[len("data: "):])
            assert payload["state"] == "cancelled"
            assert payload["completed"] == 500
        finally:
            _cleanup_test_data(test_id)

    def test_progress_histogram_consistency(self):
        """Progress event histogram bucket counts sum to completed count."""
        test_id = "lt-hist-consistency"
        _active_tests[test_id] = _make_running_status(test_id)
        # 10 latencies: 3 < 10ms, 4 in 10-30ms, 2 in 30-50ms, 1 > 50ms
        _progress_latencies[test_id] = [2.0, 5.0, 9.0, 12.0, 15.0, 20.0, 28.0, 35.0, 45.0, 60.0]
        _progress_errors[test_id] = 0
        _progress_completed[test_id] = 10
        _progress_start_time[test_id] = time.monotonic() - 0.5

        try:
            gen = _sse_event_generator(test_id)

            async def _get_first_event():
                event = await gen.__anext__()
                _active_tests[test_id] = _make_completed_status(test_id)
                return event

            event = asyncio.run(_get_first_event())
            data_line = [l for l in event.strip().split("\n") if l.startswith("data: ")][0]
            payload = json.loads(data_line[len("data: "):])

            histogram = payload["histogram"]
            total_buckets = histogram["lt_10ms"] + histogram["10_30ms"] + histogram["30_50ms"] + histogram["gt_50ms"]
            assert total_buckets == 10  # matches number of latencies

            # Verify specific bucket assignments
            assert histogram["lt_10ms"] == 3   # 2.0, 5.0, 9.0
            assert histogram["10_30ms"] == 4   # 12.0, 15.0, 20.0, 28.0
            assert histogram["30_50ms"] == 2   # 35.0, 45.0
            assert histogram["gt_50ms"] == 1   # 60.0
        finally:
            _cleanup_test_data(test_id)

    def test_progress_rps_computation(self):
        """RPS is computed as completed / elapsed_seconds."""
        test_id = "lt-rps"
        _active_tests[test_id] = _make_running_status(test_id)
        _progress_latencies[test_id] = [5.0] * 100
        _progress_errors[test_id] = 0
        _progress_completed[test_id] = 100
        # Set start time to 2 seconds ago
        _progress_start_time[test_id] = time.monotonic() - 2.0

        try:
            gen = _sse_event_generator(test_id)

            async def _get_first_event():
                event = await gen.__anext__()
                _active_tests[test_id] = _make_completed_status(test_id)
                return event

            event = asyncio.run(_get_first_event())
            data_line = [l for l in event.strip().split("\n") if l.startswith("data: ")][0]
            payload = json.loads(data_line[len("data: "):])

            # 100 completed in ~2 seconds = ~50 RPS
            assert payload["rps"] > 40  # Allow some tolerance for timing
            assert payload["rps"] < 60
            assert payload["elapsed_ms"] > 1900
            assert payload["elapsed_ms"] < 2200
        finally:
            _cleanup_test_data(test_id)
