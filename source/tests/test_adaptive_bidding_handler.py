"""Tests for agents.adaptive_bidding.handler — AgentCore HTTP entrypoint (Adaptive Bidding Strategy Agent).

Validates:
- Handler accepts an EventBridge payload and returns a JSON response
- Handler returns 200 with parameter updates when adjustments are needed
- Handler returns 200 with empty updates when within tolerance
- Handler returns 500 with error details on failure

**Validates: Requirements 7.1, 7.6, 11.1**
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.testclient import TestClient

from agents.adaptive_bidding.handler import app
from agents.adaptive_bidding.agent import ParameterUpdate
from shared.parameter_store import ParameterState


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures
# ---------------------------------------------------------------------------


def _mock_cloudwatch_response(total_bids=5000, wins=1000):
    """Create a deterministic CloudWatch GetMetricData response."""
    return {
        "MetricDataResults": [
            {"Id": "total_bids", "Values": [total_bids]},
            {"Id": "wins", "Values": [wins]},
            {"Id": "avg_price_paid", "Values": [2.5]},
            {"Id": "avg_shaded_price", "Values": [1.75]},
            {"Id": "total_revenue", "Values": [3000.0]},
            {"Id": "total_cost", "Values": [2500.0]},
        ]
    }


def _mock_parameter_state(param_name="shade_factor", value=0.65, version=5):
    """Create a deterministic ParameterState fixture."""
    bounds = {
        "shade_factor": (0.3, 0.95, 0.05),
        "conversion_value": (1.0, 50.0, 2.5),
    }
    min_val, max_val, max_delta = bounds.get(param_name, (0.0, 100.0, 5.0))
    return ParameterState(
        model_type="dlrm_bid_shader",
        parameter_name=param_name,
        current_value=value,
        previous_value=value - 0.01,
        updated_at=1700000000.0,
        updated_by="adaptive_bidding_agent",
        version=version,
        min_value=min_val,
        max_value=max_val,
        max_delta_per_update=max_delta,
        reason="Previous adjustment",
        confidence=0.85,
    )


def _mock_current_params():
    """Create a dict of current parameters matching ParameterStore format."""
    return {
        "shade_factor": _mock_parameter_state("shade_factor", 0.65),
        "conversion_value": _mock_parameter_state("conversion_value", 10.0),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandlerInvocation:
    """Test the HTTP handler accepts EventBridge payloads and returns responses."""

    @patch("agents.adaptive_bidding.handler.boto3")
    @patch("agents.adaptive_bidding.handler.ParameterStore")
    def test_accepts_eventbridge_payload_returns_response(
        self, mock_store_cls, mock_boto3
    ):
        """Handler accepts an EventBridge Scheduler JSON payload and returns 200."""
        # Set up mocks
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _mock_current_params()
        mock_store.update_parameter.return_value = _mock_parameter_state()
        mock_store_cls.return_value = mock_store

        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = _mock_cloudwatch_response(
            total_bids=5000, wins=1000
        )
        mock_cw.put_metric_data = MagicMock()
        mock_boto3.client.return_value = mock_cw

        client = TestClient(app)

        # EventBridge Scheduler sends a JSON payload with schedule context
        payload = {
            "source": "aws.scheduler",
            "detail-type": "Scheduled Event",
            "detail": {},
        }
        response = client.post("/invocations", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert "updates" in body
        assert "timestamp" in body
        assert "duration_ms" in body
        assert isinstance(body["updates"], list)

    @patch("agents.adaptive_bidding.handler.boto3")
    @patch("agents.adaptive_bidding.handler.ParameterStore")
    def test_returns_parameter_updates_when_adjustment_needed(
        self, mock_store_cls, mock_boto3
    ):
        """Handler returns updates with new parameter values when win_rate is off-target."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _mock_current_params()
        mock_store.update_parameter.return_value = _mock_parameter_state()
        mock_store_cls.return_value = mock_store

        mock_cw = MagicMock()
        # Win rate 0.2 is below target 0.35 → shade_factor should increase
        mock_cw.get_metric_data.return_value = _mock_cloudwatch_response(
            total_bids=5000, wins=1000
        )
        mock_cw.put_metric_data = MagicMock()
        mock_boto3.client.return_value = mock_cw

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["updates_count"] >= 1

        # Check that shade_factor was updated (win_rate below target)
        shade_updates = [
            u for u in body["updates"] if u["parameter_name"] == "shade_factor"
        ]
        assert len(shade_updates) == 1
        assert shade_updates[0]["new_value"] > shade_updates[0]["old_value"]

    @patch("agents.adaptive_bidding.handler.boto3")
    @patch("agents.adaptive_bidding.handler.ParameterStore")
    def test_returns_empty_updates_when_within_tolerance(
        self, mock_store_cls, mock_boto3
    ):
        """Handler returns 200 with empty updates when win_rate is at target."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _mock_current_params()
        mock_store_cls.return_value = mock_store

        mock_cw = MagicMock()
        # Win rate 0.35 is exactly at target → no adjustment
        mock_cw.get_metric_data.return_value = _mock_cloudwatch_response(
            total_bids=5000, wins=1750
        )
        mock_cw.put_metric_data = MagicMock()
        mock_boto3.client.return_value = mock_cw

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["updates_count"] == 0
        assert body["updates"] == []

    @patch("agents.adaptive_bidding.handler.boto3")
    @patch("agents.adaptive_bidding.handler.ParameterStore")
    def test_returns_500_on_failure(self, mock_store_cls, mock_boto3):
        """Handler returns 500 with error details when an exception occurs."""
        mock_store_cls.side_effect = Exception("DynamoDB connection failed")

        mock_cw = MagicMock()
        mock_boto3.client.return_value = mock_cw

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert "DynamoDB connection failed" in body["error"]
        assert "timestamp" in body
        assert "duration_ms" in body

    @patch("agents.adaptive_bidding.handler.boto3")
    @patch("agents.adaptive_bidding.handler.ParameterStore")
    def test_handles_empty_body(self, mock_store_cls, mock_boto3):
        """Handler works correctly even with an empty POST body."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _mock_current_params()
        mock_store_cls.return_value = mock_store

        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = _mock_cloudwatch_response(
            total_bids=5000, wins=1750
        )
        mock_cw.put_metric_data = MagicMock()
        mock_boto3.client.return_value = mock_cw

        client = TestClient(app)
        # Send request with no body (Content-Type not set to JSON)
        response = client.post("/invocations")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"

    @patch("agents.adaptive_bidding.handler.boto3")
    @patch("agents.adaptive_bidding.handler.ParameterStore")
    def test_invocations_endpoint_handles_repeat_invocation(
        self, mock_store_cls, mock_boto3
    ):
        """The /invocations endpoint handles repeated invocations (session reuse)."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _mock_current_params()
        mock_store_cls.return_value = mock_store

        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = _mock_cloudwatch_response(
            total_bids=5000, wins=1750
        )
        mock_cw.put_metric_data = MagicMock()
        mock_boto3.client.return_value = mock_cw

        client = TestClient(app)
        response = client.post("/invocations", json={})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"


class TestHealthEndpoint:
    """Test the health/ping endpoint."""

    def test_ping_returns_healthy(self):
        """The /ping endpoint returns Healthy status."""
        client = TestClient(app)
        response = client.get("/ping")

        assert response.status_code == 200
        assert response.json() == {"status": "Healthy"}
