"""Unit tests for the centralized CloudWatch MetricsEmitter.

Tests cover:
- Each emit method constructs the correct CloudWatch PutMetricData call
- Dimensions are correctly set on each metric
- Exceptions are caught and logged (never propagate to callers)
- TrainingDuration is only emitted when outcome is "completed"
- Generic counter helper works with arbitrary dimensions

Requirements: 13.1
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.observability import MetricsEmitter, DEFAULT_NAMESPACE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REGION = "us-east-1"
NAMESPACE = DEFAULT_NAMESPACE


def _make_emitter(
    cloudwatch_client: MagicMock | None = None,
    namespace: str = NAMESPACE,
) -> tuple[MetricsEmitter, MagicMock]:
    """Create a MetricsEmitter with a mocked CloudWatch client."""
    if cloudwatch_client is None:
        cloudwatch_client = MagicMock()
    emitter = MetricsEmitter(
        region=REGION,
        namespace=namespace,
        cloudwatch_client=cloudwatch_client,
    )
    return emitter, cloudwatch_client


# ---------------------------------------------------------------------------
# Tests: emit_training_outcome
# ---------------------------------------------------------------------------


class TestEmitTrainingOutcome:
    """Tests for emit_training_outcome method."""

    @pytest.mark.asyncio
    async def test_emits_training_job_outcome_metric(self):
        """Emits TrainingJobOutcome with correct dimensions."""
        emitter, cw = _make_emitter()

        await emitter.emit_training_outcome(
            model_type="dlrm_bid_shader",
            outcome="started",
            duration_s=0.0,
        )

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == NAMESPACE

        metric_data = call_kwargs["MetricData"]
        # Only TrainingJobOutcome when outcome != "completed"
        assert len(metric_data) == 1
        assert metric_data[0]["MetricName"] == "TrainingJobOutcome"
        assert metric_data[0]["Value"] == 1.0
        assert metric_data[0]["Unit"] == "Count"
        dims = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims == {"ModelType": "dlrm_bid_shader", "Outcome": "started"}

    @pytest.mark.asyncio
    async def test_emits_training_duration_on_completed(self):
        """Emits both TrainingJobOutcome and TrainingDuration when completed."""
        emitter, cw = _make_emitter()

        await emitter.emit_training_outcome(
            model_type="ncf_deal_manager",
            outcome="completed",
            duration_s=3600.5,
        )

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args[1]
        metric_data = call_kwargs["MetricData"]

        assert len(metric_data) == 2

        # First: TrainingJobOutcome
        assert metric_data[0]["MetricName"] == "TrainingJobOutcome"
        dims_0 = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims_0 == {"ModelType": "ncf_deal_manager", "Outcome": "completed"}

        # Second: TrainingDuration
        assert metric_data[1]["MetricName"] == "TrainingDuration"
        assert metric_data[1]["Value"] == 3600.5
        assert metric_data[1]["Unit"] == "Seconds"
        dims_1 = {d["Name"]: d["Value"] for d in metric_data[1]["Dimensions"]}
        assert dims_1 == {"ModelType": "ncf_deal_manager"}

    @pytest.mark.asyncio
    async def test_does_not_emit_duration_on_failed(self):
        """Does NOT emit TrainingDuration when outcome is 'failed'."""
        emitter, cw = _make_emitter()

        await emitter.emit_training_outcome(
            model_type="widedeep_segment_activator",
            outcome="failed",
            duration_s=120.0,
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        metric_data = call_kwargs["MetricData"]
        assert len(metric_data) == 1
        assert metric_data[0]["MetricName"] == "TrainingJobOutcome"

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """CloudWatch exceptions are caught and logged, never propagated."""
        emitter, cw = _make_emitter()
        cw.put_metric_data.side_effect = RuntimeError("CloudWatch unavailable")

        # Should not raise
        await emitter.emit_training_outcome(
            model_type="dlrm_bid_shader",
            outcome="started",
            duration_s=0.0,
        )


# ---------------------------------------------------------------------------
# Tests: emit_ab_test_result
# ---------------------------------------------------------------------------


class TestEmitABTestResult:
    """Tests for emit_ab_test_result method."""

    @pytest.mark.asyncio
    async def test_emits_all_three_ab_metrics(self):
        """Emits ABTestDecision, ABTestPValue, and ABTestLift in one call."""
        emitter, cw = _make_emitter()

        await emitter.emit_ab_test_result(
            model_type="dlrm_bid_shader",
            decision="promote",
            p_value=0.03,
            lift=0.12,
        )

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == NAMESPACE

        metric_data = call_kwargs["MetricData"]
        assert len(metric_data) == 3

        # ABTestDecision
        assert metric_data[0]["MetricName"] == "ABTestDecision"
        assert metric_data[0]["Value"] == 1.0
        assert metric_data[0]["Unit"] == "Count"
        dims_0 = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims_0 == {"ModelType": "dlrm_bid_shader", "Decision": "promote"}

        # ABTestPValue
        assert metric_data[1]["MetricName"] == "ABTestPValue"
        assert metric_data[1]["Value"] == 0.03
        assert metric_data[1]["Unit"] == "None"
        dims_1 = {d["Name"]: d["Value"] for d in metric_data[1]["Dimensions"]}
        assert dims_1 == {"ModelType": "dlrm_bid_shader"}

        # ABTestLift
        assert metric_data[2]["MetricName"] == "ABTestLift"
        assert metric_data[2]["Value"] == 0.12
        assert metric_data[2]["Unit"] == "None"
        dims_2 = {d["Name"]: d["Value"] for d in metric_data[2]["Dimensions"]}
        assert dims_2 == {"ModelType": "dlrm_bid_shader"}

    @pytest.mark.asyncio
    async def test_reject_decision_dimensions(self):
        """Decision dimension correctly reflects 'reject'."""
        emitter, cw = _make_emitter()

        await emitter.emit_ab_test_result(
            model_type="ncf_deal_manager",
            decision="reject",
            p_value=0.52,
            lift=-0.04,
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        metric_data = call_kwargs["MetricData"]
        dims = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims["Decision"] == "reject"
        assert metric_data[2]["Value"] == -0.04

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """CloudWatch exceptions are caught and logged."""
        emitter, cw = _make_emitter()
        cw.put_metric_data.side_effect = Exception("Network timeout")

        await emitter.emit_ab_test_result(
            model_type="dlrm_bid_shader",
            decision="inconclusive",
            p_value=0.15,
            lift=0.01,
        )


# ---------------------------------------------------------------------------
# Tests: emit_deployment_transition
# ---------------------------------------------------------------------------


class TestEmitDeploymentTransition:
    """Tests for emit_deployment_transition method."""

    @pytest.mark.asyncio
    async def test_emits_deployment_transition_metric(self):
        """Emits DeploymentTransition with ModelType and Transition dimensions."""
        emitter, cw = _make_emitter()

        await emitter.emit_deployment_transition(
            model_type="dlrm_bid_shader",
            transition="canary_deployed",
        )

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == NAMESPACE

        metric_data = call_kwargs["MetricData"]
        assert len(metric_data) == 1
        assert metric_data[0]["MetricName"] == "DeploymentTransition"
        assert metric_data[0]["Value"] == 1.0
        assert metric_data[0]["Unit"] == "Count"
        dims = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims == {"ModelType": "dlrm_bid_shader", "Transition": "canary_deployed"}

    @pytest.mark.asyncio
    async def test_promoted_transition(self):
        """Transition 'promoted' is set correctly."""
        emitter, cw = _make_emitter()

        await emitter.emit_deployment_transition(
            model_type="ncf_deal_manager",
            transition="promoted",
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        dims = {d["Name"]: d["Value"] for d in call_kwargs["MetricData"][0]["Dimensions"]}
        assert dims["Transition"] == "promoted"

    @pytest.mark.asyncio
    async def test_guardrail_rollback_transition(self):
        """Transition 'guardrail_rollback' is set correctly."""
        emitter, cw = _make_emitter()

        await emitter.emit_deployment_transition(
            model_type="widedeep_segment_activator",
            transition="guardrail_rollback",
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        dims = {d["Name"]: d["Value"] for d in call_kwargs["MetricData"][0]["Dimensions"]}
        assert dims == {
            "ModelType": "widedeep_segment_activator",
            "Transition": "guardrail_rollback",
        }

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """CloudWatch exceptions are caught and logged."""
        emitter, cw = _make_emitter()
        cw.put_metric_data.side_effect = RuntimeError("Service unavailable")

        await emitter.emit_deployment_transition(
            model_type="dlrm_bid_shader",
            transition="rolled_back",
        )


# ---------------------------------------------------------------------------
# Tests: emit_governance_decision
# ---------------------------------------------------------------------------


class TestEmitGovernanceDecision:
    """Tests for emit_governance_decision method."""

    @pytest.mark.asyncio
    async def test_emits_governance_decision_metric(self):
        """Emits GovernanceDecision with all three dimensions."""
        emitter, cw = _make_emitter()

        await emitter.emit_governance_decision(
            model_type="dlrm_bid_shader",
            decision="approve",
            reason_category="statistical",
        )

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == NAMESPACE

        metric_data = call_kwargs["MetricData"]
        assert len(metric_data) == 1
        assert metric_data[0]["MetricName"] == "GovernanceDecision"
        assert metric_data[0]["Value"] == 1.0
        assert metric_data[0]["Unit"] == "Count"
        dims = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims == {
            "ModelType": "dlrm_bid_shader",
            "Decision": "approve",
            "ReasonCategory": "statistical",
        }

    @pytest.mark.asyncio
    async def test_reject_with_guardrail_violation(self):
        """Decision and ReasonCategory dimensions for a guardrail rejection."""
        emitter, cw = _make_emitter()

        await emitter.emit_governance_decision(
            model_type="ncf_deal_manager",
            decision="reject",
            reason_category="guardrail_violation",
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        dims = {d["Name"]: d["Value"] for d in call_kwargs["MetricData"][0]["Dimensions"]}
        assert dims == {
            "ModelType": "ncf_deal_manager",
            "Decision": "reject",
            "ReasonCategory": "guardrail_violation",
        }

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """CloudWatch exceptions are caught and logged."""
        emitter, cw = _make_emitter()
        cw.put_metric_data.side_effect = Exception("Access denied")

        await emitter.emit_governance_decision(
            model_type="dlrm_bid_shader",
            decision="reject",
            reason_category="manual_override",
        )


# ---------------------------------------------------------------------------
# Tests: emit_counter
# ---------------------------------------------------------------------------


class TestEmitCounter:
    """Tests for the generic emit_counter helper."""

    @pytest.mark.asyncio
    async def test_emits_counter_with_custom_dimensions(self):
        """Emits a generic counter with arbitrary dimension key-value pairs."""
        emitter, cw = _make_emitter()

        await emitter.emit_counter(
            metric_name="CustomEvent",
            dimensions={"Service": "orchestrator", "EventType": "timeout"},
            count=3.0,
        )

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == NAMESPACE

        metric_data = call_kwargs["MetricData"]
        assert len(metric_data) == 1
        assert metric_data[0]["MetricName"] == "CustomEvent"
        assert metric_data[0]["Value"] == 3.0
        assert metric_data[0]["Unit"] == "Count"
        dims = {d["Name"]: d["Value"] for d in metric_data[0]["Dimensions"]}
        assert dims == {"Service": "orchestrator", "EventType": "timeout"}

    @pytest.mark.asyncio
    async def test_default_count_is_one(self):
        """Default count value is 1.0 when not specified."""
        emitter, cw = _make_emitter()

        await emitter.emit_counter(
            metric_name="SimpleCounter",
            dimensions={"Component": "feedback_collector"},
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["MetricData"][0]["Value"] == 1.0

    @pytest.mark.asyncio
    async def test_empty_dimensions(self):
        """Works with no dimensions."""
        emitter, cw = _make_emitter()

        await emitter.emit_counter(
            metric_name="GlobalCounter",
            dimensions={},
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["MetricData"][0]["Dimensions"] == []

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """CloudWatch exceptions are caught and logged."""
        emitter, cw = _make_emitter()
        cw.put_metric_data.side_effect = RuntimeError("Throttled")

        await emitter.emit_counter(
            metric_name="SomeMetric",
            dimensions={"Key": "Value"},
        )


# ---------------------------------------------------------------------------
# Tests: Custom namespace
# ---------------------------------------------------------------------------


class TestCustomNamespace:
    """Tests that a custom namespace is used in all emitted metrics."""

    @pytest.mark.asyncio
    async def test_custom_namespace_used(self):
        """Custom namespace is passed to put_metric_data."""
        custom_ns = "MyApp/CustomMetrics"
        emitter, cw = _make_emitter(namespace=custom_ns)

        await emitter.emit_deployment_transition(
            model_type="dlrm_bid_shader",
            transition="promoted",
        )

        call_kwargs = cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == custom_ns
