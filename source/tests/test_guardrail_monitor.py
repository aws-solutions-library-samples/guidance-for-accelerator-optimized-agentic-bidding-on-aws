"""Unit tests for GuardrailMonitor: automatic rollback on deployment metric breaches.

Tests cover:
- p99 latency violation detected (canary 1.3x stable → violated)
- p99 latency within threshold (canary 1.1x stable → not violated)
- Error rate violation detected (canary at 2% → violated)
- Error rate within threshold (canary at 0.5% → not violated)
- handle_violation triggers rollback on the deployer
- Audit record written with correct reason
- Rollback completes (verify canary_deployer.rollback called)
- NIM re-optimization flagged for latency regressions

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deployment.canary_deployer import (
    CanaryDeployer,
    DeploymentState,
)
from deployment.guardrail_monitor import (
    AuditRecord,
    GuardrailCheckResult,
    GuardrailConfig,
    GuardrailMonitor,
)
from deployment.model_deployer import (
    HttpResponse,
    NIMOptimizer,
    TritonModelLoader,
)


# ---------------------------------------------------------------------------
# Helpers: Mock CloudWatch Client
# ---------------------------------------------------------------------------


class MockCloudWatchClient:
    """A mock CloudWatch client that returns pre-configured metric responses.

    Uses deterministic, pre-configured responses — no randomness.
    """

    def __init__(self):
        self._responses: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def set_metric_response(
        self,
        metric_name: str,
        version: str,
        datapoints: list[dict[str, Any]],
    ) -> None:
        """Configure a response for a specific metric + version query."""
        key = f"{metric_name}:{version}"
        self._responses[key] = {"Datapoints": datapoints}

    async def get_metric_statistics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: list[dict[str, str]],
        start_time: float,
        end_time: float,
        period: int,
        statistics: list[str],
    ) -> dict[str, Any]:
        """Return pre-configured metric data."""
        self.calls.append({
            "namespace": namespace,
            "metric_name": metric_name,
            "dimensions": dimensions,
            "start_time": start_time,
            "end_time": end_time,
            "period": period,
            "statistics": statistics,
        })

        # Find version from dimensions
        version = ""
        for dim in dimensions:
            if dim.get("Name") == "ModelVersion":
                version = dim.get("Value", "")

        key = f"{metric_name}:{version}"
        if key in self._responses:
            return self._responses[key]

        return {"Datapoints": []}


# ---------------------------------------------------------------------------
# Helpers: Mock HTTP Client (same pattern as test_canary_deployer.py)
# ---------------------------------------------------------------------------


@dataclass
class RecordedCall:
    """A recorded HTTP call for verification."""

    method: str
    url: str
    json_body: dict | None = None


class MockHttpClient:
    """A mock HTTP client that returns pre-configured responses."""

    def __init__(self, responses: dict[str, HttpResponse] | None = None):
        self._responses: dict[str, HttpResponse] = responses or {}
        self.calls: list[RecordedCall] = []

    def set_response(self, url: str, response: HttpResponse) -> None:
        self._responses[url] = response

    def _find_response(self, url: str) -> HttpResponse:
        if url in self._responses:
            return self._responses[url]
        for key, resp in self._responses.items():
            if url.startswith(key):
                return resp
        return HttpResponse(status=404, body=b"Not Found", headers={})

    async def get(self, url: str) -> HttpResponse:
        self.calls.append(RecordedCall(method="GET", url=url))
        return self._find_response(url)

    async def post(self, url: str, json_body: dict | None = None) -> HttpResponse:
        self.calls.append(RecordedCall(method="POST", url=url, json_body=json_body))
        return self._find_response(url)


def _ok_json(data: dict) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


def _ok_bytes(data: bytes) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=data,
        headers={"content-type": "application/octet-stream"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRITON_URL = "http://triton:8000"
NIM_ENDPOINT = "http://nim-service:8080"
MODEL_BUCKET = "artf-model-bucket"
MODEL_NAME = "dlrm_bid_shader"
ARTIFACT_URI = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary model repository path."""
    return str(tmp_path / "models")


def _make_deployer(http_client: MockHttpClient, repo_path: str) -> CanaryDeployer:
    """Create a CanaryDeployer with mocked HTTP dependencies."""
    triton_loader = TritonModelLoader(
        triton_url=TRITON_URL,
        model_repository_path=repo_path,
        http_client=http_client,
    )
    nim_optimizer = NIMOptimizer(
        nim_endpoint=NIM_ENDPOINT,
        model_bucket=MODEL_BUCKET,
        region="us-east-1",
        http_client=http_client,
    )
    return CanaryDeployer(
        triton_loader=triton_loader,
        nim_optimizer=nim_optimizer,
    )


def _setup_successful_deploy_responses(
    client: MockHttpClient,
    model_name: str = MODEL_NAME,
) -> None:
    """Configure mock responses for a successful canary deploy."""
    # get_loaded_versions
    client.set_response(
        f"{TRITON_URL}/v2/models/{model_name}",
        _ok_json({"name": model_name, "versions": ["1"]}),
    )
    # Artifact download
    client.set_response(ARTIFACT_URI, _ok_bytes(b"engine-plan-data"))
    # Triton load
    client.set_response(
        f"{TRITON_URL}/v2/repository/models/{model_name}/load",
        _ok_json({}),
    )
    # Health check for new version (v2)
    client.set_response(
        f"{TRITON_URL}/v2/models/{model_name}/versions/2/ready",
        _ok_json({}),
    )
    # Unload (for rollback)
    client.set_response(
        f"{TRITON_URL}/v2/repository/models/{model_name}/unload",
        _ok_json({}),
    )


async def _deploy_canary(
    http_client: MockHttpClient, repo_path: str
) -> CanaryDeployer:
    """Deploy a canary and return the deployer in canary_active state."""
    _setup_successful_deploy_responses(http_client)
    deployer = _make_deployer(http_client, repo_path)
    await deployer.deploy_canary(
        model_name=MODEL_NAME,
        artifact_uri=ARTIFACT_URI,
        initial_traffic_pct=5.0,
    )
    return deployer


# ---------------------------------------------------------------------------
# Tests: check_guardrails — latency violations
# ---------------------------------------------------------------------------


class TestLatencyViolation:
    """Tests for p99 latency guardrail checks."""

    @pytest.mark.asyncio
    async def test_latency_violation_detected(self, tmp_repo):
        """Canary at 1.3x stable latency triggers a violation (Req 6.1)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        # Stable p99 = 10ms, Canary p99 = 13ms (1.3x > 1.2x threshold)
        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 13.0}],
        )
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.005}],
        )

        monitor = GuardrailMonitor(deployer, cw_client)
        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is True
        assert result.violation_reason == "latency_regression"
        assert result.canary_latency_p99 == 13.0
        assert result.stable_latency_p99 == 10.0

    @pytest.mark.asyncio
    async def test_latency_within_threshold_not_violated(self, tmp_repo):
        """Canary at 1.1x stable latency is within threshold (no violation)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        # Stable p99 = 10ms, Canary p99 = 11ms (1.1x < 1.2x threshold)
        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 11.0}],
        )
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.005}],
        )

        monitor = GuardrailMonitor(deployer, cw_client)
        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False
        assert result.violation_reason is None
        assert result.canary_latency_p99 == 11.0
        assert result.stable_latency_p99 == 10.0

    @pytest.mark.asyncio
    async def test_latency_exactly_at_threshold_not_violated(self, tmp_repo):
        """Canary at exactly 1.2x stable latency is NOT violated (needs to exceed)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        # Stable p99 = 10ms, Canary p99 = 12ms (exactly 1.2x, not > 1.2x)
        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 12.0}],
        )
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.005}],
        )

        monitor = GuardrailMonitor(deployer, cw_client)
        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False


# ---------------------------------------------------------------------------
# Tests: check_guardrails — error rate violations
# ---------------------------------------------------------------------------


class TestErrorRateViolation:
    """Tests for error rate guardrail checks."""

    @pytest.mark.asyncio
    async def test_error_rate_violation_detected(self, tmp_repo):
        """Canary at 2% error rate triggers a violation (Req 6.1)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        # Good latency but high error rate
        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 10.5}],
        )
        # 2% error rate > 1% threshold
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.02}],
        )

        monitor = GuardrailMonitor(deployer, cw_client)
        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is True
        assert result.violation_reason == "error_rate_breach"
        assert result.canary_error_rate == 0.02

    @pytest.mark.asyncio
    async def test_error_rate_within_threshold_not_violated(self, tmp_repo):
        """Canary at 0.5% error rate is within threshold (no violation)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 10.5}],
        )
        # 0.5% error rate < 1% threshold
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.005}],
        )

        monitor = GuardrailMonitor(deployer, cw_client)
        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False
        assert result.canary_error_rate == 0.005

    @pytest.mark.asyncio
    async def test_error_rate_exactly_at_threshold_not_violated(self, tmp_repo):
        """Canary at exactly 1% error rate is NOT violated (needs to exceed)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 10.5}],
        )
        # Exactly 1% — threshold is "exceeds 1%", so 1% is not violated
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.01}],
        )

        monitor = GuardrailMonitor(deployer, cw_client)
        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False


# ---------------------------------------------------------------------------
# Tests: handle_violation — rollback and audit
# ---------------------------------------------------------------------------


class TestHandleViolation:
    """Tests for handle_violation triggering rollback and writing audit records."""

    @pytest.mark.asyncio
    async def test_handle_violation_triggers_rollback(self, tmp_repo):
        """handle_violation calls canary_deployer.rollback() (Req 6.2)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        # Verify canary is active before violation
        state_before = deployer.get_state(MODEL_NAME)
        assert state_before.canary_version == 2
        assert state_before.status == "canary_active"

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.005,
            detected_at=time.time(),
        )

        audit = await monitor.handle_violation(MODEL_NAME, violation)

        # Verify rollback occurred — state should be stable with no canary
        state_after = deployer.get_state(MODEL_NAME)
        assert state_after.status == "stable"
        assert state_after.canary_version is None
        assert state_after.canary_traffic_pct == 0.0
        assert state_after.control_traffic_pct == 100.0

        # Verify unload was called (rollback unloads canary)
        unload_calls = [
            c for c in http_client.calls
            if c.method == "POST" and "unload" in c.url
        ]
        assert len(unload_calls) >= 1

    @pytest.mark.asyncio
    async def test_audit_record_has_correct_reason_latency(self, tmp_repo):
        """Audit record has reason 'latency_regression' for latency violations (Req 6.3)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.003,
            detected_at=1000.0,
        )

        audit = await monitor.handle_violation(MODEL_NAME, violation)

        assert audit.reason == "latency_regression"
        assert audit.model_name == MODEL_NAME
        assert audit.canary_version == 2
        assert audit.stable_version == 1
        assert audit.action == "automatic_rollback"
        assert audit.timestamp == 1000.0
        assert audit.metrics_snapshot["canary_latency_p99"] == 15.0
        assert audit.metrics_snapshot["stable_latency_p99"] == 10.0
        assert audit.metrics_snapshot["canary_error_rate"] == 0.003

    @pytest.mark.asyncio
    async def test_audit_record_has_correct_reason_error_rate(self, tmp_repo):
        """Audit record has reason 'error_rate_breach' for error rate violations (Req 6.4)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="error_rate_breach",
            canary_latency_p99=10.5,
            stable_latency_p99=10.0,
            canary_error_rate=0.05,
            detected_at=2000.0,
        )

        audit = await monitor.handle_violation(MODEL_NAME, violation)

        assert audit.reason == "error_rate_breach"
        assert audit.metrics_snapshot["canary_error_rate"] == 0.05

    @pytest.mark.asyncio
    async def test_nim_reoptimization_flagged_for_latency_regression(self, tmp_repo):
        """NIM re-optimization is requested for latency regressions (Req 6.3)."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.003,
            detected_at=time.time(),
        )

        audit = await monitor.handle_violation(MODEL_NAME, violation)

        assert audit.request_nim_reoptimization is True

    @pytest.mark.asyncio
    async def test_nim_reoptimization_not_flagged_for_error_rate(self, tmp_repo):
        """NIM re-optimization is NOT requested for error rate breaches."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="error_rate_breach",
            canary_latency_p99=10.5,
            stable_latency_p99=10.0,
            canary_error_rate=0.05,
            detected_at=time.time(),
        )

        audit = await monitor.handle_violation(MODEL_NAME, violation)

        assert audit.request_nim_reoptimization is False

    @pytest.mark.asyncio
    async def test_audit_records_accumulate(self, tmp_repo):
        """Multiple violations accumulate audit records."""
        http_client = MockHttpClient()

        # First canary deploy + rollback
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        violation1 = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.003,
            detected_at=1000.0,
        )

        await monitor.handle_violation(MODEL_NAME, violation1)

        assert len(monitor.audit_records) == 1
        assert monitor.audit_records[0].reason == "latency_regression"


# ---------------------------------------------------------------------------
# Tests: check_guardrails — no canary active
# ---------------------------------------------------------------------------


class TestNoCanaryActive:
    """Tests for check_guardrails when no canary is active."""

    @pytest.mark.asyncio
    async def test_no_violation_when_no_canary(self, tmp_repo):
        """check_guardrails returns no violation when no canary is deployed."""
        http_client = MockHttpClient()
        # Set up responses for getting versions (no canary deployed)
        http_client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}",
            _ok_json({"name": MODEL_NAME, "versions": ["1"]}),
        )
        deployer = _make_deployer(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()
        monitor = GuardrailMonitor(deployer, cw_client)

        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False
        # No CloudWatch calls should have been made
        assert len(cw_client.calls) == 0


# ---------------------------------------------------------------------------
# Tests: monitor_loop
# ---------------------------------------------------------------------------


class TestMonitorLoop:
    """Tests for the monitor_loop that runs periodic checks."""

    @pytest.mark.asyncio
    async def test_monitor_loop_detects_and_handles_violation(self, tmp_repo):
        """monitor_loop detects a violation and triggers rollback."""
        http_client = MockHttpClient()
        deployer = await _deploy_canary(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        # Configure metrics that will trigger a violation
        cw_client.set_metric_response(
            "InferenceLatency", "1",
            [{"Timestamp": time.time(), "p99": 10.0}],
        )
        cw_client.set_metric_response(
            "InferenceLatency", "2",
            [{"Timestamp": time.time(), "p99": 15.0}],  # 1.5x > 1.2x
        )
        cw_client.set_metric_response(
            "InferenceErrorRate", "2",
            [{"Timestamp": time.time(), "Average": 0.005}],
        )

        # Use a short check interval for testing
        config = GuardrailConfig(check_interval_seconds=0.01)
        monitor = GuardrailMonitor(deployer, cw_client, config)

        audit = await monitor.monitor_loop(MODEL_NAME)

        assert audit is not None
        assert audit.reason == "latency_regression"
        assert audit.model_name == MODEL_NAME

        # Verify rollback occurred
        state = deployer.get_state(MODEL_NAME)
        assert state.status == "stable"
        assert state.canary_version is None

    @pytest.mark.asyncio
    async def test_monitor_loop_exits_when_no_canary(self, tmp_repo):
        """monitor_loop exits with None if canary is removed externally."""
        http_client = MockHttpClient()
        # Don't deploy a canary — deployer is in stable state
        http_client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}",
            _ok_json({"name": MODEL_NAME, "versions": ["1"]}),
        )
        deployer = _make_deployer(http_client, tmp_repo)
        cw_client = MockCloudWatchClient()

        config = GuardrailConfig(check_interval_seconds=0.01)
        monitor = GuardrailMonitor(deployer, cw_client, config)

        # Should exit immediately since no canary is active
        result = await monitor.monitor_loop(MODEL_NAME)

        assert result is None
