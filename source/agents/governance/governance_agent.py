"""Model Promotion Governance Agent: validation pipeline and promotion decisions.

Orchestrates model validation through a full pipeline:
  Model Optimizer (TensorRT) → canary deploy → A/B test → promote/reject

Promotion criteria:
- Treatment primary metric > control primary metric
- p-value < significance level
- No guardrail metric regressed

Rejection triggers:
- Any guardrail breach → immediate reject + rollback
- Model optimization failure → reject with reason
- Canary load failure → reject with reason
- Max duration reached without significance → inconclusive + rollback

Writes audit records for every decision.

Requirements: 4.2, 4.5, 4.6, 4.7, 4.8, 10.1, 10.3
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig, ABTestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GovernanceDecision:
    """Result of the governance validation pipeline."""

    decision: str  # "promote" | "reject" | "inconclusive"
    reason: str
    model_type: str
    version_arn: str
    metrics: dict[str, Any] = field(default_factory=dict)
    audit_record_id: str | None = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GovernancePipelineError(Exception):
    """Raised when the governance pipeline encounters a fatal error."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"Governance pipeline failed at {stage}: {message}")
        self.stage = stage


# ---------------------------------------------------------------------------
# Model Governance Agent
# ---------------------------------------------------------------------------


class ModelPromotionGovernanceAgent:
    """Orchestrates model validation and promotion decisions.

    Deployed as an AgentCore Runtime. Invoked by EventBridge when a new
    model version is registered in SageMaker Model Registry. Each invocation
    runs the full validation pipeline:
      model optimization → canary deploy → A/B test → promote/reject.

    Uses AgentCore Identity (workload identity) to obtain credentials for:
    - SageMaker Model Registry (update model package status)
    - DynamoDB (audit trail writes)
    - CloudWatch (read A/B test metrics)
    - Triton + Model Optimizer (build engines, deploy canary/stable models)

    Args:
        model_optimizer: ModelOptimizer instance for model optimization.
        canary_deployer: CanaryDeployer instance for traffic management.
        ab_evaluator_factory: Callable that creates an ABEvaluator for a given ABTestConfig.
        guardrail_monitor: Object with `check(model_name)` method returning guardrail violations.
        model_registry_client: boto3 SageMaker client for updating model package status.
        audit_store: Object with `put_record(record)` method for writing audit records.
    """

    def __init__(
        self,
        model_optimizer: Any,
        canary_deployer: Any,
        ab_evaluator_factory: Callable[[ABTestConfig], ABEvaluator],
        guardrail_monitor: Any,
        model_registry_client: Any,
        audit_store: Any,
    ):
        self._model_optimizer = model_optimizer
        self._canary_deployer = canary_deployer
        self._ab_evaluator_factory = ab_evaluator_factory
        self._guardrail_monitor = guardrail_monitor
        self._model_registry_client = model_registry_client
        self._audit_store = audit_store

    async def on_new_model_version(
        self,
        model_type: str,
        version_arn: str,
        artifact_uri: str,
        ab_test_config: ABTestConfig | None = None,
        evaluation_interval_seconds: float = 600.0,
        collect_metrics: Callable[..., Any] | None = None,
        stage_for_comparison: bool = False,
    ) -> GovernanceDecision:
        """Run the full validation pipeline for a new model version.

        Steps:
        1. Optimize the artifact via the Model Optimizer (TensorRT) → optimized engine URI
        2. Deploy as canary (5% initial traffic) via canary_deployer
        3. Run A/B test: collect metrics, evaluate periodically
        4. Decision:
           - If treatment > control AND p < significance AND no guardrail → promote
           - If any guardrail breaches → reject + rollback immediately
           - If max duration reached without significance → inconclusive + rollback
        5. Update model registry status ("Approved" or "Rejected")
        6. Write audit record with full context

        Args:
            model_type: The model type being validated (e.g. "dlrm_bid_shader").
            version_arn: ARN of the new model version in SageMaker Model Registry.
            artifact_uri: S3 URI of the raw model artifact.
            ab_test_config: Configuration for the A/B test. If None, uses defaults.
            evaluation_interval_seconds: How often to evaluate the AB test (default 600s).
            collect_metrics: Async callable that returns (control_data, treatment_data, guardrail_data).
                            If None, uses a no-op that returns empty data (for testing).
            stage_for_comparison: When True (a load-test-triggered version), stop
                            after optimize + canary-stage at 0% traffic and return a
                            "staged" decision — no A/B test, no auto-promote, registry
                            status left unchanged. The operator runs the comparison
                            manually. When False (default; scheduled retraining), the
                            full automated optimize → canary → A/B → promote/reject
                            pipeline runs (Steps 2-6 below).

        Returns:
            GovernanceDecision with decision, reason, and metrics. ``decision`` is
            "staged" in the stage_for_comparison path, else one of
            "promote"/"reject"/"inconclusive".
        """
        # Default AB test configuration
        if ab_test_config is None:
            ab_test_config = ABTestConfig(
                model_type=model_type,
                control_version="current",
                treatment_version=version_arn,
                traffic_percentage=5.0,
                min_samples=100,
                max_duration_hours=4.0,
                significance_level=0.05,
                primary_metric="revenue_per_bid",
                guardrail_metrics=["latency_p99", "error_rate"],
            )

        # Step 1: Model optimize (TensorRT engine build via the Model Optimizer service)
        try:
            optimized_uri = await self._model_optimizer.optimize(
                model_artifact_uri=artifact_uri,
                model_name=model_type,
            )
        except Exception as e:
            reason = f"Model optimization failed: {e}"
            logger.error(reason)
            decision = GovernanceDecision(
                decision="reject",
                reason=reason,
                model_type=model_type,
                version_arn=version_arn,
                metrics={"stage": "model_optimization", "error": str(e)},
            )
            await self._update_registry_status(version_arn, "Rejected", reason)
            await self._write_audit_record(
                model_type, version_arn, "reject", reason, decision.metrics
            )
            return decision

        # Stage-only branch (load-test-triggered version): the operator wants to
        # compare this version against the current one MANUALLY, so we stage it as
        # the challenger canary and stop — no automated A/B, no auto-promote. It is
        # deployed at 0% live traffic (the load-test challenger override still
        # reaches it by variant, so real auction traffic is never affected), and
        # the registry approval status is left unchanged (PendingManualApproval):
        # staging is not an approval, and the operator's later Promote remains the
        # real approval action. Scheduled retraining (stage_for_comparison=False)
        # keeps the full automated pipeline below.
        if stage_for_comparison:
            try:
                await self._canary_deployer.deploy_canary(
                    model_name=model_type,
                    artifact_uri=optimized_uri,
                    initial_traffic_pct=0.0,
                    canary_version_arn=version_arn,
                )
            except Exception as e:
                reason = f"Canary staging failed: {e}"
                logger.error(reason)
                metrics = {"stage": "stage_for_comparison", "error": str(e)}
                # A staging failure is operational, not a model-quality rejection,
                # so the registry status is left untouched — the version stays
                # eligible once the transient cause (e.g. optimizer capacity) clears.
                await self._write_audit_record(
                    model_type, version_arn, "reject", reason, metrics
                )
                return GovernanceDecision(
                    decision="reject",
                    reason=reason,
                    model_type=model_type,
                    version_arn=version_arn,
                    metrics=metrics,
                )

            reason = (
                "Staged as challenger canary for manual comparison "
                "(load-test-triggered version); no automated A/B run."
            )
            metrics = {"stage": "staged_for_comparison", "canary_traffic_pct": 0.0}
            await self._write_audit_record(
                model_type, version_arn, "staged", reason, metrics
            )
            return GovernanceDecision(
                decision="staged",
                reason=reason,
                model_type=model_type,
                version_arn=version_arn,
                metrics=metrics,
            )

        # Step 2: Deploy as canary
        try:
            await self._canary_deployer.deploy_canary(
                model_name=model_type,
                artifact_uri=optimized_uri,
                initial_traffic_pct=ab_test_config.traffic_percentage,
            )
        except Exception as e:
            reason = f"Canary deployment failed: {e}"
            logger.error(reason)
            decision = GovernanceDecision(
                decision="reject",
                reason=reason,
                model_type=model_type,
                version_arn=version_arn,
                metrics={"stage": "canary_deploy", "error": str(e)},
            )
            await self._update_registry_status(version_arn, "Rejected", reason)
            await self._write_audit_record(
                model_type, version_arn, "reject", reason, decision.metrics
            )
            return decision

        # Step 3: Run A/B test
        try:
            ab_result = await self._run_ab_test(
                model_name=model_type,
                config=ab_test_config,
                evaluation_interval_seconds=evaluation_interval_seconds,
                collect_metrics=collect_metrics,
            )
        except Exception as e:
            # Unexpected error during AB test — rollback and reject
            reason = f"A/B test error: {e}"
            logger.error(reason)
            await self._canary_deployer.rollback(model_type)
            decision = GovernanceDecision(
                decision="reject",
                reason=reason,
                model_type=model_type,
                version_arn=version_arn,
                metrics={"stage": "ab_test", "error": str(e)},
            )
            await self._update_registry_status(version_arn, "Rejected", reason)
            await self._write_audit_record(
                model_type, version_arn, "reject", reason, decision.metrics
            )
            return decision

        # Step 4: Make decision based on AB result
        metrics = {
            "control_metric": ab_result.control_metric,
            "treatment_metric": ab_result.treatment_metric,
            "relative_lift": ab_result.relative_lift,
            "p_value": ab_result.p_value,
            "samples_control": ab_result.samples_control,
            "samples_treatment": ab_result.samples_treatment,
            "guardrail_violations": ab_result.guardrail_violations,
            "recommendation": ab_result.recommendation,
        }

        if ab_result.recommendation == "promote":
            # Promote: treatment > control, p < significance, no guardrail breach
            await self._canary_deployer.promote(model_type)
            decision = GovernanceDecision(
                decision="promote",
                reason=(
                    f"Treatment outperforms control "
                    f"(lift={ab_result.relative_lift:.4f}, "
                    f"p={ab_result.p_value:.4f})"
                ),
                model_type=model_type,
                version_arn=version_arn,
                metrics=metrics,
            )
            await self._update_registry_status(version_arn, "Approved", decision.reason)

        elif ab_result.recommendation == "reject":
            # Reject: guardrail breach or treatment significantly worse
            await self._canary_deployer.rollback(model_type)
            if ab_result.guardrail_violations:
                reason = (
                    f"Guardrail violation: "
                    f"{', '.join(ab_result.guardrail_violations)}"
                )
            else:
                reason = (
                    f"Treatment underperforms control "
                    f"(lift={ab_result.relative_lift:.4f}, "
                    f"p={ab_result.p_value:.4f})"
                )
            decision = GovernanceDecision(
                decision="reject",
                reason=reason,
                model_type=model_type,
                version_arn=version_arn,
                metrics=metrics,
            )
            await self._update_registry_status(version_arn, "Rejected", reason)

        else:
            # Inconclusive: max duration reached without significance
            await self._canary_deployer.rollback(model_type)
            reason = (
                f"Inconclusive after max duration "
                f"({ab_test_config.max_duration_hours}h): "
                f"p={ab_result.p_value:.4f}"
            )
            decision = GovernanceDecision(
                decision="inconclusive",
                reason=reason,
                model_type=model_type,
                version_arn=version_arn,
                metrics=metrics,
            )
            await self._update_registry_status(version_arn, "Rejected", reason)

        # Step 6: Write audit record
        await self._write_audit_record(
            model_type, version_arn, decision.decision, decision.reason, metrics
        )

        return decision

    async def _run_ab_test(
        self,
        model_name: str,
        config: ABTestConfig,
        evaluation_interval_seconds: float = 600.0,
        collect_metrics: Callable[..., Any] | None = None,
    ) -> ABTestResult:
        """Periodically collect metrics and evaluate using ABEvaluator.

        Runs the evaluation loop until one of:
        - ABEvaluator recommends "promote" (treatment wins)
        - ABEvaluator recommends "reject" (guardrail breach or treatment loses)
        - max_duration_hours is reached → returns last result with "extend"

        Args:
            model_name: Logical model name for the test.
            config: ABTestConfig with test parameters.
            evaluation_interval_seconds: Seconds between evaluations.
            collect_metrics: Async callable returning
                (control_data: list[float], treatment_data: list[float],
                 guardrail_data: dict[str, tuple[list[float], list[float]]] | None).

        Returns:
            The final ABTestResult.
        """
        evaluator = self._ab_evaluator_factory(config)
        max_duration_seconds = config.max_duration_hours * 3600.0
        start_time = time.monotonic()

        # Default collect_metrics returns empty data (used in testing with pre-supplied data)
        if collect_metrics is None:
            async def _empty_collector(model_name, config):
                return [], [], None
            collect_metrics = _empty_collector

        last_result: ABTestResult | None = None

        while True:
            elapsed = time.monotonic() - start_time

            # Check max duration
            if elapsed >= max_duration_seconds:
                if last_result is not None:
                    return last_result
                # Never got a result — return inconclusive
                return ABTestResult(
                    test_id=f"ab_{config.model_type}_{config.control_version}_vs_{config.treatment_version}",
                    status=__import__("agents.governance.ab_evaluator", fromlist=["TestStatus"]).TestStatus.INCONCLUSIVE,
                    control_metric=0.0,
                    treatment_metric=0.0,
                    relative_lift=0.0,
                    p_value=1.0,
                    samples_control=0,
                    samples_treatment=0,
                    guardrail_violations=[],
                    recommendation="extend",
                )

            # Collect metrics
            control_data, treatment_data, guardrail_data = await collect_metrics(
                model_name, config
            )

            # Also check guardrail_monitor for runtime guardrail breaches
            guardrail_violations = []
            if self._guardrail_monitor is not None:
                try:
                    guardrail_violations = await self._guardrail_monitor.check(model_name)
                except Exception as e:
                    logger.warning("Guardrail monitor check failed: %s", e)

            # If guardrail monitor reports violations, inject them into guardrail_data
            if guardrail_violations and guardrail_data is None:
                guardrail_data = {}

            # Evaluate
            result = evaluator.evaluate(
                control_data=control_data,
                treatment_data=treatment_data,
                guardrail_data=guardrail_data,
            )

            # If guardrail monitor found violations not caught by the evaluator,
            # override the result to reject
            if guardrail_violations and not result.guardrail_violations:
                result = ABTestResult(
                    test_id=result.test_id,
                    status=result.status,
                    control_metric=result.control_metric,
                    treatment_metric=result.treatment_metric,
                    relative_lift=result.relative_lift,
                    p_value=result.p_value,
                    samples_control=result.samples_control,
                    samples_treatment=result.samples_treatment,
                    guardrail_violations=guardrail_violations,
                    recommendation="reject",
                )

            last_result = result

            # Early stop on decisive recommendation
            if result.recommendation in ("promote", "reject"):
                return result

            # Wait for next evaluation interval
            await asyncio.sleep(evaluation_interval_seconds)

    async def _update_registry_status(
        self, version_arn: str, status: str, reason: str
    ) -> None:
        """Update model version status in SageMaker Model Registry.

        Calls `update_model_package` to set ModelApprovalStatus to
        "Approved" or "Rejected".

        Args:
            version_arn: ARN of the model package version.
            status: "Approved" or "Rejected".
            reason: Reason for the status change.
        """
        try:
            self._model_registry_client.update_model_package(
                ModelPackageArn=version_arn,
                ModelApprovalStatus=status,
                ApprovalDescription=reason,
            )
            logger.info(
                "Updated model registry: %s → %s (%s)",
                version_arn,
                status,
                reason,
            )
        except Exception as e:
            logger.error(
                "Failed to update model registry for %s: %s",
                version_arn,
                e,
            )
            raise

    async def _write_audit_record(
        self,
        model_type: str,
        version_arn: str,
        decision: str,
        reason: str,
        metrics: dict[str, Any],
    ) -> None:
        """Append an audit record with timestamp, actor, decision, and metrics.

        Args:
            model_type: The model type (e.g. "dlrm_bid_shader").
            version_arn: ARN of the model version.
            decision: "promote", "reject", or "inconclusive".
            reason: Human-readable reason for the decision.
            metrics: Metrics snapshot associated with the decision.
        """
        record = {
            "timestamp": time.time(),
            "actor": "model_promotion_governance_agent",
            "model_type": model_type,
            "version_arn": version_arn,
            "decision": decision,
            "reason": reason,
            "metrics": metrics,
        }

        try:
            await self._audit_store.put_record(record)
            logger.info(
                "Audit record written: %s %s → %s",
                model_type,
                version_arn,
                decision,
            )
        except Exception as e:
            logger.error(
                "Failed to write audit record for %s %s: %s",
                model_type,
                version_arn,
                e,
            )
            raise
