"""Tests for the signal receiver endpoint (POST /v1/signals).

Validates:
- Valid signals are accepted and emitted to Kinesis
- Invalid request_id (not a UUID) is rejected with 422
- Invalid signal_type is rejected with 422
- conversion_value only accepted for conversion signals
- Timestamp defaults to current time when not provided
- Invalid JSON body returns 400

**Validates: Requirements 1.4**
"""

import sys
import os
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import SignalEvent
from shared.signal_associator import SignalAssociator, SignalType, DownstreamSignal
from shared.feedback_collector import FeedbackCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockRequest:
    """Minimal mock of a Starlette Request for testing."""

    def __init__(self, body: dict | bytes | None = None):
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("No body")
        if isinstance(self._body, bytes):
            return json.loads(self._body)
        return self._body


class _InvalidJsonRequest:
    """Mock request that raises on json()."""

    async def json(self):
        raise ValueError("Invalid JSON")


def _make_mock_collector() -> MagicMock:
    mock = MagicMock(spec=FeedbackCollector)
    mock.emit = AsyncMock()
    mock._kinesis_client = MagicMock()
    mock._stream_name = "test-stream"
    return mock


def _make_mock_associator() -> MagicMock:
    mock = MagicMock(spec=SignalAssociator)
    mock.handle_signal = AsyncMock(return_value=True)
    return mock


# ---------------------------------------------------------------------------
# Tests: SignalEvent model validation
# ---------------------------------------------------------------------------


class TestSignalEventModel:
    """Tests for the SignalEvent pydantic model."""

    def test_valid_impression_signal(self):
        event = SignalEvent(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type="impression",
            timestamp=1718000000.0,
        )
        assert event.signal_type == "impression"
        assert event.conversion_value is None
        assert event.record_type == "signal"

    def test_valid_click_signal(self):
        event = SignalEvent(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type="click",
            timestamp=1718000000.0,
        )
        assert event.signal_type == "click"

    def test_valid_conversion_signal_with_value(self):
        event = SignalEvent(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type="conversion",
            timestamp=1718000000.0,
            conversion_value=25.50,
        )
        assert event.signal_type == "conversion"
        assert event.conversion_value == 25.50

    def test_invalid_request_id_not_uuid(self):
        with pytest.raises(Exception) as exc_info:
            SignalEvent(
                request_id="not-a-uuid",
                signal_type="impression",
                timestamp=1718000000.0,
            )
        assert "UUID" in str(exc_info.value)

    def test_empty_request_id_rejected(self):
        with pytest.raises(Exception) as exc_info:
            SignalEvent(
                request_id="",
                signal_type="impression",
                timestamp=1718000000.0,
            )
        assert "UUID" in str(exc_info.value)

    def test_invalid_signal_type_rejected(self):
        with pytest.raises(Exception) as exc_info:
            SignalEvent(
                request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                signal_type="purchase",
                timestamp=1718000000.0,
            )
        assert "signal_type" in str(exc_info.value)

    def test_conversion_value_on_non_conversion_rejected(self):
        with pytest.raises(Exception) as exc_info:
            SignalEvent(
                request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                signal_type="impression",
                timestamp=1718000000.0,
                conversion_value=10.0,
            )
        assert "conversion_value" in str(exc_info.value)

    def test_conversion_value_on_click_rejected(self):
        with pytest.raises(Exception) as exc_info:
            SignalEvent(
                request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                signal_type="click",
                timestamp=1718000000.0,
                conversion_value=5.0,
            )
        assert "conversion_value" in str(exc_info.value)

    def test_record_type_is_signal(self):
        event = SignalEvent(
            request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            signal_type="impression",
            timestamp=1718000000.0,
        )
        assert event.record_type == "signal"


# ---------------------------------------------------------------------------
# Tests: Signal receiver endpoint
# ---------------------------------------------------------------------------


class TestSignalReceiverEndpoint:
    """Tests for the receive_signal endpoint handler."""

    def test_valid_impression_accepted(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "impression",
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["status"] == "accepted"
        assert body["request_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert body["signal_type"] == "impression"

    def test_valid_conversion_accepted(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "conversion",
            "conversion_value": 15.5,
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["status"] == "accepted"
        assert body["signal_type"] == "conversion"

    def test_invalid_request_id_rejected(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "not-a-valid-uuid",
            "signal_type": "impression",
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 422
        body = json.loads(response.body)
        assert "error" in body
        assert "UUID" in body["error"]

    def test_invalid_signal_type_rejected(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "purchase",
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 422
        body = json.loads(response.body)
        assert "error" in body
        assert "signal_type" in body["error"]

    def test_conversion_value_on_impression_rejected(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "impression",
            "conversion_value": 10.0,
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 422
        body = json.loads(response.body)
        assert "error" in body
        assert "conversion_value" in body["error"]

    def test_invalid_json_returns_400(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _InvalidJsonRequest()
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 400
        body = json.loads(response.body)
        assert "error" in body
        assert "Invalid JSON" in body["error"]

    def test_timestamp_defaults_to_current_time(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        before = time.time()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "click",
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        after = time.time()
        assert response.status_code == 200
        call_args = mock_associator.handle_signal.call_args[0][0]
        assert before <= call_args.timestamp <= after

    def test_emits_to_kinesis_via_collector(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "impression",
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 200
        mock_collector._kinesis_client.put_record.assert_called_once()
        call_kwargs = mock_collector._kinesis_client.put_record.call_args[1]
        assert call_kwargs["StreamName"] == "test-stream"
        assert call_kwargs["PartitionKey"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        record_data = json.loads(call_kwargs["Data"])
        assert record_data["record_type"] == "signal"
        assert record_data["signal_type"] == "impression"
        assert record_data["request_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_kinesis_failure_does_not_block_response(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_collector._kinesis_client.put_record.side_effect = Exception(
            "Kinesis unavailable"
        )
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "impression",
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 200

    def test_no_collector_still_works(self):
        from orchestrator.signal_receiver import receive_signal

        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "click",
            "timestamp": 1718000000.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, None)
        )
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["status"] == "accepted"

    def test_associator_called_with_correct_signal(self):
        from orchestrator.signal_receiver import receive_signal

        mock_collector = _make_mock_collector()
        mock_associator = _make_mock_associator()
        request = _MockRequest({
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "signal_type": "conversion",
            "conversion_value": 42.0,
            "timestamp": 1718000500.0,
        })
        response = asyncio.run(
            receive_signal(request, mock_associator, mock_collector)
        )
        assert response.status_code == 200
        mock_associator.handle_signal.assert_called_once()
        signal_arg = mock_associator.handle_signal.call_args[0][0]
        assert isinstance(signal_arg, DownstreamSignal)
        assert signal_arg.request_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert signal_arg.signal_type == SignalType.CONVERSION
        assert signal_arg.conversion_value == 42.0
        assert signal_arg.timestamp == 1718000500.0
