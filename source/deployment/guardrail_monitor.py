"""Guardrail Monitor: automatic rollback on deployment metric breaches.

Periodically checks canary deployment metrics (p99 latency, error rate) against
configured thresholds. Triggers automatic rollback via the CanaryDeployer within
60 seconds of detecting a violation, marks the canary version as rejected, and
writes an audit record.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CloudWatch Client Protocol — injectable for testing
# ---------------------------------------------------------------------------


class CloudWatchClient(Protocol):
    """Protocol for CloudWatch metric queries. Mocked at the boundary."""

    async def get_metric_statistics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: list[dict[str, str]],
        start_time: float,
        end_time: float,
        period: int,
        statistics: list[str],
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GuardrailConfig:
    """Configuration for deployment guardrail thresholds."""

    latency_multiplier_threshold: float = 1.2  # 1.2x current p99 latency
    error_rate_threshold: float = 0.01  # 1% error rate
    check_interval_seconds: float = 10.0  # Check every 10s
    rollback_deadline_seconds: float = 60.0  # Must rollback within 60s of detection


@dataclass
class GuardrailCheckResult:
    """Result of a single guardrail check."""

    is_violated: bool
    violation_reason: str | None = None
    canary_latency_p99: float = 0.0
    stable_latency_p99: float = 0.0
    canary_error_rate: float = 0.0
    detected_at: float = 0.0


@dataclass
class AuditRecord:
    """An audit trail record for a guardrail rollback event.

    In Option D the stable and canary variants are separate Triton models
    (``<model>_stable`` / ``<model>_canary``), so a variant is identified by its
    model name rather than a version integer.
    """

    timestamp: float
    model_name: str
    canary_model: str
    stable_model: str
    reason: str  # "latency_regression" | "error_rate_breach"
    metrics_snapshot: dict[str, float]
    action: str = "automatic_rollback"
    request_reoptimization: bool = False


# ---------------------------------------------------------------------------
# Guardrail Monitor
# ---------------------------------------------------------------------------


class GuardrailMonitor:
    """Monitors canary deployments and triggers automatic rollback on violation.

    Uses CloudWatch metrics to compare canary vs stable model performance.
    If the canary's p99 latency exceeds 1.2x the stable version's p99,
    or if the canary's error rate exceeds 1%, triggers an immediate rollback
    via the CanaryDeployer and records the event in the audit trail.

    Requirements: 6.1, 6.2, 6.3, 6.4
    """

    def __init__(
        self,
        canary_deployer: Any,
        cloudwatch_client: CloudWatchClient,
        config: GuardrailConfig | None = None,
    ):
        self._deployer = canary_deployer
        self._cloudwatch = cloudwatch_client
        self._config = config or GuardrailConfig()
        self._audit_records: list[AuditRecord] = []

    @property
    def audit_records(self) -> list[AuditRecord]:
        """Return the list of audit records for inspection."""
        return list(self._audit_records)

    async def check_guardrails(self, model_name: str) -> GuardrailCheckResult:
        """Check canary vs stable metrics for guardrail violations.

        Queries CloudWatch for the canary and stable versions' p99 latency
        and error rate. Returns a result indicating whether a violation
        was detected and the reason.

        Args:
            model_name: Logical model name to check.

        Returns:
            GuardrailCheckResult with violation status and metrics.
        """
        state = self._deployer.get_state(model_name)

        if state.canary_model is None:
            return GuardrailCheckResult(is_violated=False)

        # In Option D, stable and canary are SEPARATE Triton models
        # (<model>_stable and <model>_canary), so their serving metrics are keyed
        # by model name, not by a version integer.
        stable_model = f"{model_name}_stable"
        canary_model = state.canary_model

        now = time.time()
        window_start = now - 60.0  # Look at last 60 seconds of data

        # Query stable model p99 latency
        stable_latency = await self._query_p99_latency(
            model_variant=stable_model,
            start_time=window_start,
            end_time=now,
        )

        # Query canary model p99 latency
        canary_latency = await self._query_p99_latency(
            model_variant=canary_model,
            start_time=window_start,
            end_time=now,
        )

        # Query canary error rate
        canary_error_rate = await self._query_error_rate(
            model_variant=canary_model,
            start_time=window_start,
            end_time=now,
        )

        # Check latency threshold: canary p99 > stable p99 * 1.2
        latency_threshold = stable_latency * self._config.latency_multiplier_threshold
        if stable_latency > 0 and canary_latency > latency_threshold:
            return GuardrailCheckResult(
                is_violated=True,
                violation_reason="latency_regression",
                canary_latency_p99=canary_latency,
                stable_latency_p99=stable_latency,
                canary_error_rate=canary_error_rate,
                detected_at=now,
            )

        # Check error rate threshold: canary error rate > 1%
        if canary_error_rate > self._config.error_rate_threshold:
            return GuardrailCheckResult(
                is_violated=True,
                violation_reason="error_rate_breach",
                canary_latency_p99=canary_latency,
                stable_latency_p99=stable_latency,
                canary_error_rate=canary_error_rate,
                detected_at=now,
            )

        return GuardrailCheckResult(
            is_violated=False,
            canary_latency_p99=canary_latency,
            stable_latency_p99=stable_latency,
            canary_error_rate=canary_error_rate,
        )

    async def handle_violation(
        self, model_name: str, result: GuardrailCheckResult
    ) -> AuditRecord:
        """Handle a guardrail violation by rolling back and auditing.

        Triggers canary_deployer.rollback() to restore 100% traffic to
        the stable version, marks the canary as rejected, and writes
        an audit record.

        Args:
            model_name: Model that violated guardrails.
            result: The check result with violation details.

        Returns:
            The audit record created for this event.
        """
        state = self._deployer.get_state(model_name)
        canary_model = state.canary_model or f"{model_name}_canary"
        stable_model = f"{model_name}_stable"

        # Trigger rollback — restores 100% to stable, unloads canary
        await self._deployer.rollback(model_name)

        # Request a re-optimization (more aggressive TensorRT settings) for latency
        # regressions; error-rate breaches are not addressed by re-optimizing.
        request_reopt = result.violation_reason == "latency_regression"

        # Create audit record
        audit_record = AuditRecord(
            timestamp=result.detected_at or time.time(),
            model_name=model_name,
            canary_model=canary_model,
            stable_model=stable_model,
            reason=result.violation_reason or "unknown",
            metrics_snapshot={
                "canary_latency_p99": result.canary_latency_p99,
                "stable_latency_p99": result.stable_latency_p99,
                "canary_error_rate": result.canary_error_rate,
            },
            action="automatic_rollback",
            request_reoptimization=request_reopt,
        )

        self._audit_records.append(audit_record)

        logger.warning(
            "Guardrail rollback for %s: canary %s rolled back. "
            "Reason: %s. Metrics: latency canary=%.2fms stable=%.2fms, "
            "error_rate=%.4f. Re-optimization requested: %s",
            model_name,
            canary_model,
            result.violation_reason,
            result.canary_latency_p99,
            result.stable_latency_p99,
            result.canary_error_rate,
            request_reopt,
        )

        return audit_record

    async def monitor_loop(self, model_name: str) -> AuditRecord | None:
        """Run periodic guardrail checks until a violation is detected or canary ends.

        Checks metrics every config.check_interval_seconds. On violation,
        calls handle_violation and exits the loop.

        This is intended to be called by the Governance Agent during A/B testing.

        Args:
            model_name: Model name to monitor.

        Returns:
            The AuditRecord if a violation triggered rollback, or None if the
            canary was promoted/removed before any violation.
        """
        while True:
            state = self._deployer.get_state(model_name)
            if state.canary_model is None:
                # Canary no longer active (promoted or already rolled back)
                return None

            result = await self.check_guardrails(model_name)

            if result.is_violated:
                audit = await self.handle_violation(model_name, result)
                return audit

            await asyncio.sleep(self._config.check_interval_seconds)

    # -----------------------------------------------------------------------
    # Private: CloudWatch query helpers
    # -----------------------------------------------------------------------

    async def _query_p99_latency(
        self,
        model_variant: str,
        start_time: float,
        end_time: float,
    ) -> float:
        """Query CloudWatch for p99 latency of a specific Triton model variant.

        ``model_variant`` is the served Triton model name (``<model>_stable`` or
        ``<model>_canary``). Returns the p99 latency in milliseconds, or 0.0 if no
        data is available (honest "no data" — never fabricated).
        """
        response = await self._cloudwatch.get_metric_statistics(
            namespace="ARTF/Inference",
            metric_name="InferenceLatency",
            dimensions=[
                {"Name": "ModelName", "Value": model_variant},
            ],
            start_time=start_time,
            end_time=end_time,
            period=60,
            statistics=["p99"],
        )

        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return 0.0

        # Return the most recent p99 value
        latest = max(datapoints, key=lambda dp: dp.get("Timestamp", 0))
        return latest.get("p99", 0.0)

    async def _query_error_rate(
        self,
        model_variant: str,
        start_time: float,
        end_time: float,
    ) -> float:
        """Query CloudWatch for error rate of a specific Triton model variant.

        ``model_variant`` is the served Triton model name (``<model>_stable`` or
        ``<model>_canary``). Returns the error rate in [0.0, 1.0], or 0.0 if no data.
        """
        response = await self._cloudwatch.get_metric_statistics(
            namespace="ARTF/Inference",
            metric_name="InferenceErrorRate",
            dimensions=[
                {"Name": "ModelName", "Value": model_variant},
            ],
            start_time=start_time,
            end_time=end_time,
            period=60,
            statistics=["Average"],
        )

        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return 0.0

        # Return the most recent average error rate
        latest = max(datapoints, key=lambda dp: dp.get("Timestamp", 0))
        return latest.get("Average", 0.0)
