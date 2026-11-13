"""Centralized CloudWatch metrics emitter for the closed-loop learning system.

Provides a unified interface for emitting observability metrics across
all loops: training outcomes, A/B test results, deployment state transitions,
governance decisions, and generic counters.

All methods are fire-and-forget — exceptions are caught and logged but never
propagated to callers. This ensures observability instrumentation cannot
disrupt the operational loops it monitors.

Requirements: 13.1
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import boto3

logger = logging.getLogger(__name__)

# Default CloudWatch namespace for all closed-loop metrics
DEFAULT_NAMESPACE = "ARTF/ClosedLoop"

# ---------------------------------------------------------------------------
# Central registry of all CloudWatch metric namespaces and metric names used
# across the closed-loop learning system. Each module emits metrics under its
# own namespace; this registry documents and cross-references them.
# ---------------------------------------------------------------------------

METRICS: dict[str, dict[str, Any]] = {
    "feedback": {
        "namespace": "ARTF/FeedbackCollector",
        "metrics": {
            "emit_errors": "EmitErrors",
            "events_emitted": "EventsEmitted",
            "batch_size": "BatchSize",
        },
    },
    "training": {
        "namespace": "ARTF/Training",
        "metrics": {
            "job_started": "JobStarted",
            "job_completed": "JobCompleted",
            "job_failed": "JobFailed",
            "consecutive_failures": "ConsecutiveFailures",
            "training_duration_seconds": "TrainingDurationSeconds",
        },
    },
    "ab_test": {
        "namespace": "ARTF/ABTest",
        "metrics": {
            "test_started": "TestStarted",
            "test_completed": "TestCompleted",
            "promote_decision": "PromoteDecision",
            "reject_decision": "RejectDecision",
            "inconclusive_decision": "InconclusiveDecision",
            "p_value": "PValue",
            "relative_lift": "RelativeLift",
        },
    },
    "parameter_updates": {
        "namespace": "ARTF/BidOutcome",
        "metrics": {
            "parameter_update": "ParameterUpdate",
            "adjustment_magnitude": "AdjustmentMagnitude",
            "skipped_insufficient_samples": "SkippedInsufficientSamples",
        },
    },
    "deployment": {
        "namespace": "ARTF/Deployment",
        "metrics": {
            "canary_deployed": "CanaryDeployed",
            "canary_promoted": "CanaryPromoted",
            "canary_rolled_back": "CanaryRolledBack",
            "guardrail_violation": "GuardrailViolation",
            "nim_optimization_started": "NIMOptimizationStarted",
            "nim_optimization_completed": "NIMOptimizationCompleted",
            "nim_optimization_failed": "NIMOptimizationFailed",
        },
    },
    "inference": {
        "namespace": "ARTF/Inference",
        "metrics": {
            "latency_p99": "InferenceLatency",
            "error_rate": "InferenceErrorRate",
            "parameter_cache_hit": "ParameterCacheHit",
            "parameter_cache_miss": "ParameterCacheMiss",
        },
    },
}


class MetricsEmitter:
    """Emits CloudWatch metrics for the closed-loop learning system.

    Covers training job outcomes, A/B test results, deployment state
    transitions, governance decisions, and generic counters.

    All emit methods are fire-and-forget: exceptions are caught and logged
    but never propagated. This prevents metric instrumentation from
    affecting operational code paths.

    Args:
        region: AWS region for the CloudWatch client.
        namespace: CloudWatch namespace for all metrics (default: ARTF/ClosedLoop).
        cloudwatch_client: Optional pre-configured boto3 CloudWatch client
            (useful for testing). If None, creates a new client.
    """

    def __init__(
        self,
        region: str,
        namespace: str = DEFAULT_NAMESPACE,
        cloudwatch_client: Any | None = None,
        default_dimensions: list[dict[str, str]] | None = None,
    ) -> None:
        self._region = region
        self._namespace = namespace
        self._default_dimensions = default_dimensions or []

        if cloudwatch_client is not None:
            self._cloudwatch_client = cloudwatch_client
        else:
            self._cloudwatch_client = boto3.client("cloudwatch", region_name=region)

    # ------------------------------------------------------------------
    # Generic emit helpers
    # ------------------------------------------------------------------

    async def emit(
        self,
        namespace: str,
        metric_name: str,
        value: float,
        unit: str = "Count",
        dimensions: list[dict[str, str]] | None = None,
    ) -> None:
        """Emit a single metric to CloudWatch. Best-effort — never raises.

        Args:
            namespace: CloudWatch namespace (e.g. "ARTF/FeedbackCollector").
            metric_name: The metric name (e.g. "EmitErrors").
            value: Numeric value to record.
            unit: CloudWatch unit (default "Count"). One of: Seconds,
                Microseconds, Milliseconds, Bytes, Kilobytes, Megabytes,
                Gigabytes, Terabytes, Bits, Percent, Count, None, etc.
            dimensions: Optional list of {"Name": ..., "Value": ...} dicts.
                Merged with default_dimensions provided at construction.
        """
        try:
            merged_dims = list(self._default_dimensions)
            if dimensions:
                merged_dims.extend(dimensions)

            metric_data: list[dict[str, Any]] = [
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    **({"Dimensions": merged_dims} if merged_dims else {}),
                }
            ]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=namespace,
                    MetricData=metric_data,
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit metric %s/%s", namespace, metric_name
            )

    async def emit_batch(
        self,
        namespace: str,
        metric_data: list[dict[str, Any]],
    ) -> None:
        """Emit multiple metrics in one PutMetricData call. Best-effort.

        Each item in metric_data should be a CloudWatch MetricDatum dict
        with at least MetricName and Value. If Dimensions are not provided
        per-datum, default_dimensions are applied.

        Args:
            namespace: CloudWatch namespace for all metrics in the batch.
            metric_data: List of CloudWatch MetricDatum dicts. Each should
                contain at minimum {"MetricName": str, "Value": float}.
                Optional keys: "Unit", "Dimensions", "Timestamp".
        """
        if not metric_data:
            return

        try:
            # Apply default dimensions to any datum missing its own
            prepared: list[dict[str, Any]] = []
            for datum in metric_data:
                d = dict(datum)
                if "Dimensions" not in d and self._default_dimensions:
                    d["Dimensions"] = list(self._default_dimensions)
                if "Unit" not in d:
                    d["Unit"] = "Count"
                prepared.append(d)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=namespace,
                    MetricData=prepared,
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit batch metrics to namespace %s (%d items)",
                namespace,
                len(metric_data),
            )

    # ------------------------------------------------------------------
    # Training outcomes
    # ------------------------------------------------------------------

    async def emit_training_outcome(
        self, model_type: str, outcome: str, duration_s: float
    ) -> None:
        """Emit metrics for a training job outcome.

        Emits:
        - TrainingJobOutcome (Count=1) with dimensions ModelType and Outcome
        - TrainingDuration (Seconds) when outcome is "completed"

        Args:
            model_type: The model type (e.g. "dlrm_bid_shader").
            outcome: One of "started", "completed", "failed".
            duration_s: Training duration in seconds (used when outcome="completed").
        """
        try:
            metric_data: list[dict[str, Any]] = [
                {
                    "MetricName": "TrainingJobOutcome",
                    "Value": 1.0,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "ModelType", "Value": model_type},
                        {"Name": "Outcome", "Value": outcome},
                    ],
                }
            ]

            if outcome == "completed":
                metric_data.append(
                    {
                        "MetricName": "TrainingDuration",
                        "Value": duration_s,
                        "Unit": "Seconds",
                        "Dimensions": [
                            {"Name": "ModelType", "Value": model_type},
                        ],
                    }
                )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=metric_data,
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit training outcome metric (model_type=%s, outcome=%s)",
                model_type,
                outcome,
            )

    # ------------------------------------------------------------------
    # A/B test results
    # ------------------------------------------------------------------

    async def emit_ab_test_result(
        self, model_type: str, decision: str, p_value: float, lift: float
    ) -> None:
        """Emit metrics for an A/B test result.

        Emits:
        - ABTestDecision (Count=1) with dimensions ModelType and Decision
        - ABTestPValue (None unit) with dimension ModelType
        - ABTestLift (None unit) with dimension ModelType

        Args:
            model_type: The model type (e.g. "dlrm_bid_shader").
            decision: One of "promote", "reject", "inconclusive".
            p_value: The p-value from the statistical test.
            lift: The relative lift of treatment over control.
        """
        try:
            metric_data: list[dict[str, Any]] = [
                {
                    "MetricName": "ABTestDecision",
                    "Value": 1.0,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "ModelType", "Value": model_type},
                        {"Name": "Decision", "Value": decision},
                    ],
                },
                {
                    "MetricName": "ABTestPValue",
                    "Value": p_value,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "ModelType", "Value": model_type},
                    ],
                },
                {
                    "MetricName": "ABTestLift",
                    "Value": lift,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "ModelType", "Value": model_type},
                    ],
                },
            ]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=metric_data,
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit A/B test result metric (model_type=%s, decision=%s)",
                model_type,
                decision,
            )

    # ------------------------------------------------------------------
    # Deployment state transitions
    # ------------------------------------------------------------------

    async def emit_deployment_transition(
        self, model_type: str, transition: str
    ) -> None:
        """Emit a metric for a deployment state transition.

        Emits:
        - DeploymentTransition (Count=1) with dimensions ModelType and Transition

        Args:
            model_type: The model type (e.g. "dlrm_bid_shader").
            transition: One of "canary_deployed", "promoted", "rolled_back",
                "guardrail_rollback".
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=[
                        {
                            "MetricName": "DeploymentTransition",
                            "Value": 1.0,
                            "Unit": "Count",
                            "Dimensions": [
                                {"Name": "ModelType", "Value": model_type},
                                {"Name": "Transition", "Value": transition},
                            ],
                        }
                    ],
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit deployment transition metric "
                "(model_type=%s, transition=%s)",
                model_type,
                transition,
            )

    # ------------------------------------------------------------------
    # Governance decisions
    # ------------------------------------------------------------------

    async def emit_governance_decision(
        self, model_type: str, decision: str, reason_category: str
    ) -> None:
        """Emit a metric for a governance decision.

        Emits:
        - GovernanceDecision (Count=1) with dimensions ModelType, Decision,
          and ReasonCategory

        Args:
            model_type: The model type (e.g. "dlrm_bid_shader").
            decision: The governance decision (e.g. "approve", "reject").
            reason_category: The category of reason (e.g. "statistical",
                "guardrail_violation", "manual_override").
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=[
                        {
                            "MetricName": "GovernanceDecision",
                            "Value": 1.0,
                            "Unit": "Count",
                            "Dimensions": [
                                {"Name": "ModelType", "Value": model_type},
                                {"Name": "Decision", "Value": decision},
                                {"Name": "ReasonCategory", "Value": reason_category},
                            ],
                        }
                    ],
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit governance decision metric "
                "(model_type=%s, decision=%s, reason_category=%s)",
                model_type,
                decision,
                reason_category,
            )

    # ------------------------------------------------------------------
    # Generic counter
    # ------------------------------------------------------------------

    async def emit_counter(
        self, metric_name: str, dimensions: dict[str, str], count: float = 1.0
    ) -> None:
        """Emit a generic counter metric.

        A helper for emitting arbitrary count-based metrics with custom
        dimensions under the configured namespace.

        Args:
            metric_name: The CloudWatch metric name.
            dimensions: Key-value pairs for metric dimensions.
            count: The count value to emit (default 1.0).
        """
        try:
            cw_dimensions = [
                {"Name": key, "Value": value}
                for key, value in dimensions.items()
            ]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._cloudwatch_client.put_metric_data(
                    Namespace=self._namespace,
                    MetricData=[
                        {
                            "MetricName": metric_name,
                            "Value": count,
                            "Unit": "Count",
                            "Dimensions": cw_dimensions,
                        }
                    ],
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit counter metric (metric_name=%s)",
                metric_name,
            )
