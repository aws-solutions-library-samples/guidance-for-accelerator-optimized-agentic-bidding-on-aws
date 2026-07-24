"""Tests for ModelPromotionGovernanceAgent validation pipeline and decisions.

Validates the full governance pipeline:
- Full pipeline: optimize → deploy → AB pass → promote → registry updated
- Guardrail breach: deploy → AB detects guardrail violation → reject + rollback
- Inconclusive: AB reaches max duration → inconclusive + rollback
- Model optimization failure → reject with reason
- Canary load failure → reject with reason
- Audit record written for promote/reject/inconclusive
- Registry status updated correctly ("Approved" vs "Rejected")

Mock: Model Optimizer (HTTP), Triton API (HTTP), SageMaker registry client (boto3), CloudWatch (boto3)
Real: ABEvaluator logic, decision logic

**Validates: Requirements 4.2, 4.5, 4.6, 4.7, 4.8, 10.1, 10.3**
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig, ABTestResult, TestStatus
from agents.governance.governance_agent import GovernanceDecision, ModelPromotionGovernanceAgent


# ---------------------------------------------------------------------------
# Deterministic fixture data (same arrays as test_ab_evaluator.py)
# ---------------------------------------------------------------------------

CONTROL_BASELINE = [
    1.82, 2.15, 1.97, 2.03, 2.10, 1.88, 2.22, 1.95, 2.08, 1.91,
    2.01, 1.99, 2.14, 1.87, 2.06, 2.11, 1.93, 2.04, 1.96, 2.09,
    2.00, 1.85, 2.17, 1.94, 2.07, 1.90, 2.12, 1.98, 2.05, 1.89,
    2.02, 2.13, 1.92, 2.08, 1.86, 2.16, 1.95, 2.03, 1.97, 2.10,
    1.88, 2.19, 1.93, 2.06, 1.91, 2.11, 1.96, 2.04, 2.00, 1.84,
]

TREATMENT_WINNER = [
    2.42, 2.55, 2.47, 2.63, 2.50, 2.38, 2.62, 2.45, 2.58, 2.41,
    2.51, 2.49, 2.64, 2.37, 2.56, 2.61, 2.43, 2.54, 2.46, 2.59,
    2.50, 2.35, 2.67, 2.44, 2.57, 2.40, 2.52, 2.48, 2.55, 2.39,
    2.52, 2.63, 2.42, 2.58, 2.36, 2.66, 2.45, 2.53, 2.47, 2.60,
    2.38, 2.69, 2.43, 2.56, 2.41, 2.61, 2.46, 2.54, 2.50, 2.34,
]

TREATMENT_LOSER = [
    1.42, 1.55, 1.47, 1.63, 1.50, 1.38, 1.62, 1.45, 1.58, 1.41,
    1.51, 1.49, 1.64, 1.37, 1.56, 1.61, 1.43, 1.54, 1.46, 1.59,
    1.50, 1.35, 1.67, 1.44, 1.57, 1.40, 1.52, 1.48, 1.55, 1.39,
    1.52, 1.63, 1.42, 1.58, 1.36, 1.66, 1.45, 1.53, 1.47, 1.60,
    1.38, 1.69, 1.43, 1.56, 1.41, 1.61, 1.46, 1.54, 1.50, 1.34,
]

TREATMENT_IDENTICAL = [
    1.83, 2.14, 1.98, 2.02, 2.11, 1.87, 2.21, 1.96, 2.07, 1.92,
    2.00, 2.01, 2.13, 1.88, 2.05, 2.12, 1.94, 2.03, 1.97, 2.08,
    2.01, 1.86, 2.16, 1.93, 2.06, 1.91, 2.11, 1.99, 2.04, 1.90,
    2.01, 2.14, 1.91, 2.09, 1.85, 2.17, 1.94, 2.04, 1.96, 2.11,
    1.87, 2.20, 1.92, 2.07, 1.90, 2.12, 1.95, 2.05, 1.99, 1.85,
]

# Guardrail data: latency regression (negated — lower is worse)
GUARDRAIL_CONTROL_LATENCY_NEG = [
    -10.2, -10.5, -9.8, -10.1, -10.3, -9.9, -10.4, -10.0, -10.2, -10.1,
    -10.3, -9.7, -10.5, -10.0, -10.2, -10.1, -9.8, -10.4, -10.0, -10.3,
    -10.1, -9.9, -10.2, -10.0, -10.4, -9.8, -10.3, -10.1, -10.2, -10.0,
    -10.5, -9.7, -10.1, -10.3, -10.0, -10.2, -9.9, -10.4, -10.1, -10.0,
    -10.2, -10.3, -9.8, -10.1, -10.0, -10.4, -9.9, -10.2, -10.1, -10.3,
]

GUARDRAIL_TREATMENT_LATENCY_NEG = [
    -15.2, -15.5, -14.8, -15.1, -15.3, -14.9, -15.4, -15.0, -15.2, -15.1,
    -15.3, -14.7, -15.5, -15.0, -15.2, -15.1, -14.8, -15.4, -15.0, -15.3,
    -15.1, -14.9, -15.2, -15.0, -15.4, -14.8, -15.3, -15.1, -15.2, -15.0,
    -15.5, -14.7, -15.1, -15.3, -15.0, -15.2, -14.9, -15.4, -15.1, -15.0,
    -15.2, -15.3, -14.8, -15.1, -15.0, -15.4, -14.9, -15.2, -15.1, -15.3,
]


# ---------------------------------------------------------------------------
# Mock Dependencies
# ---------------------------------------------------------------------------


class MockModelOptimizer:
    """Mock Model Optimizer that returns a deterministic optimized URI."""

    def __init__(self, *, should_fail: bool = False, error_message: str = "optimizer error"):
        self._should_fail = should_fail
        self._error_message = error_message
        self.optimize_calls: list[dict] = []

    async def optimize(self, model_artifact_uri: str, model_name: str) -> str:
        self.optimize_calls.append({
            "model_artifact_uri": model_artifact_uri,
            "model_name": model_name,
        })
        if self._should_fail:
            raise RuntimeError(self._error_message)
        return f"s3://models/optimized/{model_name}/model.engine"


class MockCanaryDeployer:
    """Mock canary deployer that tracks deploy/promote/rollback calls."""

    def __init__(self, *, deploy_should_fail: bool = False):
        self._deploy_should_fail = deploy_should_fail
        self.deploy_calls: list[dict] = []
        self.promote_calls: list[str] = []
        self.rollback_calls: list[str] = []

    async def deploy_canary(
        self, model_name: str, artifact_uri: str, initial_traffic_pct: float
    ) -> None:
        self.deploy_calls.append({
            "model_name": model_name,
            "artifact_uri": artifact_uri,
            "initial_traffic_pct": initial_traffic_pct,
        })
        if self._deploy_should_fail:
            raise RuntimeError("Canary load failed: health check timeout")

    async def promote(self, model_name: str) -> None:
        self.promote_calls.append(model_name)

    async def rollback(self, model_name: str) -> None:
        self.rollback_calls.append(model_name)


class MockGuardrailMonitor:
    """Mock guardrail monitor that returns pre-configured violations."""

    def __init__(self, violations: list[str] | None = None):
        self._violations = violations or []

    async def check(self, model_name: str) -> list[str]:
        return self._violations


class MockModelRegistryClient:
    """Mock boto3 SageMaker client for update_model_package."""

    def __init__(self):
        self.update_calls: list[dict] = []

    def update_model_package(self, **kwargs) -> dict:
        self.update_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


class MockAuditStore:
    """Mock audit store (DynamoDB-like) that records put_record calls."""

    def __init__(self):
        self.records: list[dict] = []

    async def put_record(self, record: dict) -> None:
        self.records.append(record)


# ---------------------------------------------------------------------------
# Helper: default test config
# ---------------------------------------------------------------------------


def _default_ab_config() -> ABTestConfig:
    return ABTestConfig(
        model_type="dlrm_bid_shader",
        control_version="v1.0",
        treatment_version="v1.1",
        traffic_percentage=5.0,
        min_samples=10,
        max_duration_hours=0.001,  # Very short for tests (~3.6s)
        significance_level=0.05,
        primary_metric="revenue_per_bid",
        guardrail_metrics=["latency_p99"],
    )


def _build_agent(
    nim_should_fail: bool = False,
    deploy_should_fail: bool = False,
    guardrail_violations: list[str] | None = None,
) -> tuple[
    ModelPromotionGovernanceAgent,
    MockModelOptimizer,
    MockCanaryDeployer,
    MockGuardrailMonitor,
    MockModelRegistryClient,
    MockAuditStore,
]:
    """Build a ModelPromotionGovernanceAgent with all mock dependencies."""
    nim = MockModelOptimizer(should_fail=nim_should_fail)
    canary = MockCanaryDeployer(deploy_should_fail=deploy_should_fail)
    guardrail = MockGuardrailMonitor(violations=guardrail_violations)
    registry = MockModelRegistryClient()
    audit = MockAuditStore()

    agent = ModelPromotionGovernanceAgent(
        model_optimizer=nim,
        canary_deployer=canary,
        ab_evaluator_factory=lambda config: ABEvaluator(config),
        guardrail_monitor=guardrail,
        model_registry_client=registry,
        audit_store=audit,
    )

    return agent, nim, canary, guardrail, registry, audit


# ---------------------------------------------------------------------------
# Helper: metrics collector factories
# ---------------------------------------------------------------------------


def _make_winning_collector():
    """Collector that returns data where treatment clearly beats control."""
    call_count = 0

    async def collect(model_name, config):
        nonlocal call_count
        call_count += 1
        return CONTROL_BASELINE, TREATMENT_WINNER, None

    return collect


def _make_losing_collector():
    """Collector that returns data where treatment clearly loses."""
    async def collect(model_name, config):
        return CONTROL_BASELINE, TREATMENT_LOSER, None

    return collect


def _make_guardrail_breach_collector():
    """Collector that returns data with a guardrail regression."""
    async def collect(model_name, config):
        guardrail_data = {
            "latency_p99": (
                GUARDRAIL_CONTROL_LATENCY_NEG,
                GUARDRAIL_TREATMENT_LATENCY_NEG,
            ),
        }
        return CONTROL_BASELINE, TREATMENT_WINNER, guardrail_data

    return collect


def _make_inconclusive_collector():
    """Collector that returns data where no significance is reached."""
    async def collect(model_name, config):
        return CONTROL_BASELINE, TREATMENT_IDENTICAL, None

    return collect


# ---------------------------------------------------------------------------
# Tests: Full pipeline — promote path
# ---------------------------------------------------------------------------


class TestPromotePipeline:
    """Test the happy path: optimize → deploy → AB pass → promote."""

    @pytest.mark.asyncio
    async def test_full_pipeline_promote(self):
        """Treatment wins → promote → registry Approved → audit written."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert result.decision == "promote"
        assert "outperforms" in result.reason

    @pytest.mark.asyncio
    async def test_nim_called_with_correct_args(self):
        """NIM optimizer is called with the raw artifact URI."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(nim.optimize_calls) == 1
        assert nim.optimize_calls[0]["model_artifact_uri"] == "s3://models/raw/dlrm/model.pt"
        assert nim.optimize_calls[0]["model_name"] == "dlrm_bid_shader"

    @pytest.mark.asyncio
    async def test_canary_deployed_with_optimized_uri(self):
        """Canary deployer receives the optimized URI from NIM."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(canary.deploy_calls) == 1
        assert canary.deploy_calls[0]["artifact_uri"] == "s3://models/optimized/dlrm_bid_shader/model.engine"
        assert canary.deploy_calls[0]["initial_traffic_pct"] == 5.0

    @pytest.mark.asyncio
    async def test_promote_calls_canary_promote(self):
        """On promote decision, canary_deployer.promote() is called."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert "dlrm_bid_shader" in canary.promote_calls
        assert len(canary.rollback_calls) == 0

    @pytest.mark.asyncio
    async def test_promote_updates_registry_approved(self):
        """On promote, registry is updated with 'Approved' status."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(registry.update_calls) == 1
        assert registry.update_calls[0]["ModelApprovalStatus"] == "Approved"
        assert registry.update_calls[0]["ModelPackageArn"] == "arn:aws:sagemaker:us-east-1:123:model-package/v1.1"


# ---------------------------------------------------------------------------
# Tests: Guardrail breach → reject + rollback
# ---------------------------------------------------------------------------


class TestGuardrailBreach:
    """Test guardrail violation forces reject + rollback."""

    @pytest.mark.asyncio
    async def test_guardrail_breach_rejects(self):
        """Guardrail breach → reject regardless of primary metric improvement."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_guardrail_breach_collector(),
        )

        assert result.decision == "reject"
        assert "Guardrail" in result.reason or "guardrail" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_guardrail_breach_triggers_rollback(self):
        """On guardrail breach, canary_deployer.rollback() is called."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_guardrail_breach_collector(),
        )

        assert "dlrm_bid_shader" in canary.rollback_calls
        assert len(canary.promote_calls) == 0

    @pytest.mark.asyncio
    async def test_guardrail_breach_updates_registry_rejected(self):
        """On guardrail breach, registry is updated with 'Rejected' status."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_guardrail_breach_collector(),
        )

        assert len(registry.update_calls) == 1
        assert registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"


# ---------------------------------------------------------------------------
# Tests: Inconclusive → rollback
# ---------------------------------------------------------------------------


class TestInconclusive:
    """Test max duration reached → inconclusive + rollback."""

    @pytest.mark.asyncio
    async def test_inconclusive_when_no_significance(self):
        """When AB test doesn't reach significance by max duration → inconclusive."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()
        # Short max_duration ensures test times out quickly
        config.max_duration_hours = 0.0001  # ~0.36 seconds

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_inconclusive_collector(),
        )

        assert result.decision == "inconclusive"
        assert "Inconclusive" in result.reason or "inconclusive" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_inconclusive_triggers_rollback(self):
        """On inconclusive, canary_deployer.rollback() is called."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()
        config.max_duration_hours = 0.0001

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_inconclusive_collector(),
        )

        assert "dlrm_bid_shader" in canary.rollback_calls
        assert len(canary.promote_calls) == 0

    @pytest.mark.asyncio
    async def test_inconclusive_updates_registry_rejected(self):
        """On inconclusive, registry is updated with 'Rejected' status."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()
        config.max_duration_hours = 0.0001

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_inconclusive_collector(),
        )

        assert len(registry.update_calls) == 1
        assert registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"


# ---------------------------------------------------------------------------
# Tests: NIM optimization failure
# ---------------------------------------------------------------------------


class TestNIMFailure:
    """Test NIM optimization failure → reject."""

    @pytest.mark.asyncio
    async def test_nim_failure_rejects(self):
        """NIM optimization failure → reject with reason."""
        agent, nim, canary, _, registry, audit = _build_agent(nim_should_fail=True)
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert result.decision == "reject"
        assert "Model optimization failed" in result.reason

    @pytest.mark.asyncio
    async def test_nim_failure_no_canary_deploy(self):
        """On NIM failure, canary is never deployed."""
        agent, nim, canary, _, registry, audit = _build_agent(nim_should_fail=True)
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(canary.deploy_calls) == 0
        assert len(canary.promote_calls) == 0
        assert len(canary.rollback_calls) == 0

    @pytest.mark.asyncio
    async def test_nim_failure_updates_registry_rejected(self):
        """On NIM failure, registry is updated with 'Rejected'."""
        agent, nim, canary, _, registry, audit = _build_agent(nim_should_fail=True)
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(registry.update_calls) == 1
        assert registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"


# ---------------------------------------------------------------------------
# Tests: Canary load failure
# ---------------------------------------------------------------------------


class TestCanaryLoadFailure:
    """Test canary deployment failure → reject."""

    @pytest.mark.asyncio
    async def test_canary_failure_rejects(self):
        """Canary load failure → reject with reason."""
        agent, nim, canary, _, registry, audit = _build_agent(deploy_should_fail=True)
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert result.decision == "reject"
        assert "Canary deployment failed" in result.reason

    @pytest.mark.asyncio
    async def test_canary_failure_nim_still_called(self):
        """Even when canary fails, NIM optimization happens first."""
        agent, nim, canary, _, registry, audit = _build_agent(deploy_should_fail=True)
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        # NIM was called (it succeeded)
        assert len(nim.optimize_calls) == 1
        # But no promote or rollback (deploy failed before AB test)
        assert len(canary.promote_calls) == 0
        assert len(canary.rollback_calls) == 0

    @pytest.mark.asyncio
    async def test_canary_failure_updates_registry_rejected(self):
        """On canary failure, registry is updated with 'Rejected'."""
        agent, nim, canary, _, registry, audit = _build_agent(deploy_should_fail=True)
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(registry.update_calls) == 1
        assert registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"


# ---------------------------------------------------------------------------
# Tests: Audit records
# ---------------------------------------------------------------------------


class TestAuditRecords:
    """Test audit record written for promote/reject/inconclusive."""

    @pytest.mark.asyncio
    async def test_audit_record_written_on_promote(self):
        """Promote writes an audit record with correct fields."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(audit.records) == 1
        record = audit.records[0]
        assert record["actor"] == "model_promotion_governance_agent"
        assert record["decision"] == "promote"
        assert record["model_type"] == "dlrm_bid_shader"
        assert record["version_arn"] == "arn:aws:sagemaker:us-east-1:123:model-package/v1.1"
        assert "timestamp" in record
        assert "reason" in record
        assert "metrics" in record

    @pytest.mark.asyncio
    async def test_audit_record_written_on_reject(self):
        """Reject writes an audit record."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_guardrail_breach_collector(),
        )

        assert len(audit.records) == 1
        record = audit.records[0]
        assert record["decision"] == "reject"
        assert record["actor"] == "model_promotion_governance_agent"

    @pytest.mark.asyncio
    async def test_audit_record_written_on_inconclusive(self):
        """Inconclusive writes an audit record."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()
        config.max_duration_hours = 0.0001

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_inconclusive_collector(),
        )

        assert len(audit.records) == 1
        record = audit.records[0]
        assert record["decision"] == "inconclusive"
        assert record["actor"] == "model_promotion_governance_agent"

    @pytest.mark.asyncio
    async def test_audit_record_written_on_nim_failure(self):
        """NIM failure writes an audit record."""
        agent, nim, canary, _, registry, audit = _build_agent(nim_should_fail=True)
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(audit.records) == 1
        record = audit.records[0]
        assert record["decision"] == "reject"
        assert "Model optimization failed" in record["reason"]

    @pytest.mark.asyncio
    async def test_audit_record_written_on_canary_failure(self):
        """Canary failure writes an audit record."""
        agent, nim, canary, _, registry, audit = _build_agent(deploy_should_fail=True)
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert len(audit.records) == 1
        record = audit.records[0]
        assert record["decision"] == "reject"
        assert "Canary" in record["reason"] or "canary" in record["reason"].lower()


# ---------------------------------------------------------------------------
# Tests: Registry status updated correctly
# ---------------------------------------------------------------------------


class TestRegistryStatus:
    """Test that registry status is Approved/Rejected per the decision."""

    @pytest.mark.asyncio
    async def test_promote_sets_approved(self):
        """Promote → ModelApprovalStatus = 'Approved'."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert registry.update_calls[0]["ModelApprovalStatus"] == "Approved"

    @pytest.mark.asyncio
    async def test_reject_sets_rejected(self):
        """Reject → ModelApprovalStatus = 'Rejected'."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_losing_collector(),
        )

        assert registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"

    @pytest.mark.asyncio
    async def test_inconclusive_sets_rejected(self):
        """Inconclusive → ModelApprovalStatus = 'Rejected'."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()
        config.max_duration_hours = 0.0001

        await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_inconclusive_collector(),
        )

        assert registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"


# ---------------------------------------------------------------------------
# Tests: Guardrail monitor integration
# ---------------------------------------------------------------------------


class TestGuardrailMonitorIntegration:
    """Test that runtime guardrail_monitor violations trigger reject."""

    @pytest.mark.asyncio
    async def test_monitor_violation_forces_reject(self):
        """When guardrail_monitor.check() returns violations → reject + rollback."""
        agent, nim, canary, _, registry, audit = _build_agent(
            guardrail_violations=["latency_p99: 120ms > 100ms threshold"]
        )
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            # Use winning collector — but guardrail_monitor overrides
            collect_metrics=_make_winning_collector(),
        )

        assert result.decision == "reject"
        assert "dlrm_bid_shader" in canary.rollback_calls


# ---------------------------------------------------------------------------
# Tests: Decision metrics in result
# ---------------------------------------------------------------------------


class TestDecisionMetrics:
    """Test that the returned GovernanceDecision contains proper metrics."""

    @pytest.mark.asyncio
    async def test_promote_decision_has_metrics(self):
        """Promote decision includes AB test metrics."""
        agent, nim, canary, _, registry, audit = _build_agent()
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert "p_value" in result.metrics
        assert "relative_lift" in result.metrics
        assert "control_metric" in result.metrics
        assert "treatment_metric" in result.metrics
        assert result.metrics["p_value"] < 0.05
        assert result.metrics["relative_lift"] > 0.0

    @pytest.mark.asyncio
    async def test_nim_failure_decision_has_stage(self):
        """NIM failure decision includes stage info in metrics."""
        agent, nim, canary, _, registry, audit = _build_agent(nim_should_fail=True)
        config = _default_ab_config()

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn="arn:aws:sagemaker:us-east-1:123:model-package/v1.1",
            artifact_uri="s3://models/raw/dlrm/model.pt",
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=_make_winning_collector(),
        )

        assert result.metrics["stage"] == "nim_optimization"
        assert "error" in result.metrics
