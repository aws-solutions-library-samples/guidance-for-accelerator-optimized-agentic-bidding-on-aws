"""Unit tests for GuardrailMonitor: automatic rollback on deployment metric breaches.

Option-D note: stable and canary are SEPARATE Triton models (<model>_stable and
<model>_canary), so the monitor reads per-variant serving metrics by MODEL NAME
(not by version), and the DeploymentState identifies the canary by `canary_model`.

Tests cover:
- p99 latency violation detected (canary 1.3x stable → violated)
- p99 latency within / exactly-at threshold → not violated
- Error rate violation detected / within / exactly-at threshold
- handle_violation triggers rollback and writes an audit record
- Audit reason is latency_regression / error_rate_breach
- Re-optimization flagged for latency regressions, not error-rate breaches
- No violation when no canary is active
- monitor_loop detects a violation and rolls back; exits when no canary

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deployment.canary_deployer import DeploymentState
from deployment.guardrail_monitor import (
    GuardrailCheckResult,
    GuardrailConfig,
    GuardrailMonitor,
)


MODEL_NAME = "dlrm_bid_shader"
STABLE_MODEL = f"{MODEL_NAME}_stable"
CANARY_MODEL = f"{MODEL_NAME}_canary"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeDeployer:
    """Holds a DeploymentState and clears the canary on rollback."""

    def __init__(self, state: DeploymentState):
        self._state = state
        self.rollback_calls: list[str] = []

    def get_state(self, model_name: str) -> DeploymentState:
        return self._state

    async def rollback(self, model_name: str) -> DeploymentState:
        self.rollback_calls.append(model_name)
        self._state = DeploymentState(
            model_name=model_name,
            current_version=self._state.current_version,
            status="stable",
        )
        return self._state


def _canary_state() -> DeploymentState:
    return DeploymentState(
        model_name=MODEL_NAME,
        current_version=1,
        canary_model=CANARY_MODEL,
        canary_engine_uri="s3://b/opt/dlrm/model.engine",
        canary_traffic_pct=5.0,
        status="canary_active",
    )


class MockCloudWatchClient:
    """Deterministic CloudWatch stand-in keyed by (metric_name, ModelName)."""

    def __init__(self):
        self._responses: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def set_metric_response(self, metric_name: str, model_variant: str, datapoints: list[dict]):
        self._responses[f"{metric_name}:{model_variant}"] = {"Datapoints": datapoints}

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
        self.calls.append({"metric_name": metric_name, "dimensions": dimensions})
        model_variant = ""
        for dim in dimensions:
            if dim.get("Name") == "ModelName":
                model_variant = dim.get("Value", "")
        return self._responses.get(f"{metric_name}:{model_variant}", {"Datapoints": []})


def _make_monitor(cw: MockCloudWatchClient, config: GuardrailConfig | None = None):
    deployer = FakeDeployer(_canary_state())
    return GuardrailMonitor(deployer, cw, config), deployer


# ---------------------------------------------------------------------------
# Latency violations
# ---------------------------------------------------------------------------


class TestLatencyViolation:
    @pytest.mark.asyncio
    async def test_latency_violation_detected(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 13.0}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.005}])
        monitor, _ = _make_monitor(cw)

        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is True
        assert result.violation_reason == "latency_regression"
        assert result.canary_latency_p99 == 13.0
        assert result.stable_latency_p99 == 10.0

    @pytest.mark.asyncio
    async def test_latency_within_threshold_not_violated(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 11.0}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.005}])
        monitor, _ = _make_monitor(cw)

        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False
        assert result.violation_reason is None

    @pytest.mark.asyncio
    async def test_latency_exactly_at_threshold_not_violated(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 12.0}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.005}])
        monitor, _ = _make_monitor(cw)

        result = await monitor.check_guardrails(MODEL_NAME)
        assert result.is_violated is False


# ---------------------------------------------------------------------------
# Error rate violations
# ---------------------------------------------------------------------------


class TestErrorRateViolation:
    @pytest.mark.asyncio
    async def test_error_rate_violation_detected(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 10.5}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.02}])
        monitor, _ = _make_monitor(cw)

        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is True
        assert result.violation_reason == "error_rate_breach"
        assert result.canary_error_rate == 0.02

    @pytest.mark.asyncio
    async def test_error_rate_within_threshold_not_violated(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 10.5}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.005}])
        monitor, _ = _make_monitor(cw)

        result = await monitor.check_guardrails(MODEL_NAME)
        assert result.is_violated is False
        assert result.canary_error_rate == 0.005

    @pytest.mark.asyncio
    async def test_error_rate_exactly_at_threshold_not_violated(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 10.5}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.01}])
        monitor, _ = _make_monitor(cw)

        result = await monitor.check_guardrails(MODEL_NAME)
        assert result.is_violated is False


# ---------------------------------------------------------------------------
# handle_violation — rollback + audit
# ---------------------------------------------------------------------------


class TestHandleViolation:
    @pytest.mark.asyncio
    async def test_handle_violation_triggers_rollback(self):
        cw = MockCloudWatchClient()
        monitor, deployer = _make_monitor(cw)

        state_before = deployer.get_state(MODEL_NAME)
        assert state_before.canary_model == CANARY_MODEL
        assert state_before.status == "canary_active"

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.005,
            detected_at=1000.0,
        )
        await monitor.handle_violation(MODEL_NAME, violation)

        assert deployer.rollback_calls == [MODEL_NAME]
        state_after = deployer.get_state(MODEL_NAME)
        assert state_after.status == "stable"
        assert state_after.canary_model is None
        assert state_after.control_traffic_pct == 100.0

    @pytest.mark.asyncio
    async def test_audit_record_latency_regression(self):
        cw = MockCloudWatchClient()
        monitor, _ = _make_monitor(cw)

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
        assert audit.canary_model == CANARY_MODEL
        assert audit.stable_model == STABLE_MODEL
        assert audit.action == "automatic_rollback"
        assert audit.timestamp == 1000.0
        assert audit.metrics_snapshot["canary_latency_p99"] == 15.0
        assert audit.metrics_snapshot["stable_latency_p99"] == 10.0
        assert audit.metrics_snapshot["canary_error_rate"] == 0.003

    @pytest.mark.asyncio
    async def test_audit_record_error_rate_breach(self):
        cw = MockCloudWatchClient()
        monitor, _ = _make_monitor(cw)

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
    async def test_reoptimization_flagged_for_latency_regression(self):
        cw = MockCloudWatchClient()
        monitor, _ = _make_monitor(cw)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.003,
            detected_at=time.time(),
        )
        audit = await monitor.handle_violation(MODEL_NAME, violation)
        assert audit.request_reoptimization is True

    @pytest.mark.asyncio
    async def test_reoptimization_not_flagged_for_error_rate(self):
        cw = MockCloudWatchClient()
        monitor, _ = _make_monitor(cw)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="error_rate_breach",
            canary_latency_p99=10.5,
            stable_latency_p99=10.0,
            canary_error_rate=0.05,
            detected_at=time.time(),
        )
        audit = await monitor.handle_violation(MODEL_NAME, violation)
        assert audit.request_reoptimization is False

    @pytest.mark.asyncio
    async def test_audit_records_accumulate(self):
        cw = MockCloudWatchClient()
        monitor, _ = _make_monitor(cw)

        violation = GuardrailCheckResult(
            is_violated=True,
            violation_reason="latency_regression",
            canary_latency_p99=15.0,
            stable_latency_p99=10.0,
            canary_error_rate=0.003,
            detected_at=1000.0,
        )
        await monitor.handle_violation(MODEL_NAME, violation)

        assert len(monitor.audit_records) == 1
        assert monitor.audit_records[0].reason == "latency_regression"


# ---------------------------------------------------------------------------
# No canary active
# ---------------------------------------------------------------------------


class TestNoCanaryActive:
    @pytest.mark.asyncio
    async def test_no_violation_when_no_canary(self):
        cw = MockCloudWatchClient()
        deployer = FakeDeployer(DeploymentState(model_name=MODEL_NAME, status="stable"))
        monitor = GuardrailMonitor(deployer, cw)

        result = await monitor.check_guardrails(MODEL_NAME)

        assert result.is_violated is False
        # No CloudWatch queries when there is no canary to compare.
        assert len(cw.calls) == 0


# ---------------------------------------------------------------------------
# monitor_loop
# ---------------------------------------------------------------------------


class TestMonitorLoop:
    @pytest.mark.asyncio
    async def test_monitor_loop_detects_and_handles_violation(self):
        cw = MockCloudWatchClient()
        cw.set_metric_response("InferenceLatency", STABLE_MODEL, [{"Timestamp": 1.0, "p99": 10.0}])
        cw.set_metric_response("InferenceLatency", CANARY_MODEL, [{"Timestamp": 1.0, "p99": 15.0}])
        cw.set_metric_response("InferenceErrorRate", CANARY_MODEL, [{"Timestamp": 1.0, "Average": 0.005}])
        monitor, deployer = _make_monitor(cw, GuardrailConfig(check_interval_seconds=0.0))

        audit = await monitor.monitor_loop(MODEL_NAME)

        assert audit is not None
        assert audit.reason == "latency_regression"
        assert audit.model_name == MODEL_NAME
        assert deployer.get_state(MODEL_NAME).canary_model is None

    @pytest.mark.asyncio
    async def test_monitor_loop_exits_when_no_canary(self):
        cw = MockCloudWatchClient()
        deployer = FakeDeployer(DeploymentState(model_name=MODEL_NAME, status="stable"))
        monitor = GuardrailMonitor(deployer, cw, GuardrailConfig(check_interval_seconds=0.0))

        result = await monitor.monitor_loop(MODEL_NAME)
        assert result is None
