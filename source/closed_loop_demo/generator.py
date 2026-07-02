"""Synthetic input-metric generator for the closed-loop demo.

Emits the *input* market metrics for a scenario to the **real** CloudWatch
namespaces the Part 2 loops read from (``ARTF/BidOutcome`` and
``ARTF/Inference``). This is realistic test input the user controls — it is
never a fabricated *result*.

Every emit returns an ``EmitEvidence`` record listing exactly which metrics and
values were written and when, so a consumer can independently verify the input.

Note on CloudWatch read-after-write: ``PutMetricData`` datapoints take a few
seconds (occasionally up to ~1 minute) before ``GetMetricData`` returns them.
The invoker's "direct" decision mode does not depend on this round-trip; it
feeds the same controlled values to the real agent synchronously. The emitted
metrics still populate real CloudWatch for the metrics view and for the
scheduled (every-5-minute) agent invocation in a live deployment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from closed_loop_demo.scenarios import (
    BidOutcomeMetrics,
    InferenceMetrics,
    Scenario,
)

logger = logging.getLogger(__name__)

BID_OUTCOME_NAMESPACE = "ARTF/BidOutcome"
INFERENCE_NAMESPACE = "ARTF/Inference"


@dataclass
class EmitEvidence:
    """Record of exactly what synthetic input was written to CloudWatch."""

    namespace: str
    emitted: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None
    timestamp: float = 0.0

    def add(self, metric_name: str, value: float, dimensions: dict[str, str] | None = None) -> None:
        self.emitted.append(
            {
                "metric_name": metric_name,
                "value": value,
                "dimensions": dimensions or {},
            }
        )


def _dims(model_type: str, extra: dict[str, str] | None = None) -> list[dict[str, str]]:
    dims = [{"Name": "ModelType", "Value": model_type}]
    if extra:
        dims.extend({"Name": k, "Value": v} for k, v in extra.items())
    return dims


def emit_bid_outcome_metrics(
    cloudwatch_client: Any,
    metrics: BidOutcomeMetrics,
    model_type: str,
    timestamp: Optional[float] = None,
) -> EmitEvidence:
    """Emit the six ``ARTF/BidOutcome`` metrics for a scenario.

    A single datapoint per metric at ``timestamp`` reproduces the target value
    under both Sum and Average statistics (the agent uses Sum for counts/totals
    and Average for the price metrics).

    Best-effort: never raises. Returns evidence with ``ok``/``error`` set.
    """
    ts = timestamp if timestamp is not None else (time.time() - 30.0)
    ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    evidence = EmitEvidence(namespace=BID_OUTCOME_NAMESPACE, timestamp=ts)

    metric_map = metrics.as_metric_map()
    metric_data = []
    for name, value in metric_map.items():
        unit = "Count" if name in ("TotalBids", "Wins") else "None"
        metric_data.append(
            {
                "MetricName": name,
                "Value": value,
                "Unit": unit,
                "Timestamp": ts_dt,
                "Dimensions": [],  # aggregate metrics (agent queries without dimensions)
            }
        )
        evidence.add(name, value)

    try:
        cloudwatch_client.put_metric_data(
            Namespace=BID_OUTCOME_NAMESPACE,
            MetricData=metric_data,
        )
        evidence.ok = True
    except Exception as exc:  # best-effort — report honestly, never raise
        logger.warning("Failed to emit bid-outcome metrics: %s", exc)
        evidence.error = str(exc)

    return evidence


def emit_inference_metrics(
    cloudwatch_client: Any,
    metrics: InferenceMetrics,
    model_type: str,
    stable_version: int = 1,
    canary_version: int = 2,
    timestamp: Optional[float] = None,
) -> EmitEvidence:
    """Emit ``ARTF/Inference`` latency / error-rate metrics for stable + canary.

    Populates the metrics view with real datapoints. Best-effort.
    """
    ts = timestamp if timestamp is not None else (time.time() - 30.0)
    ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    evidence = EmitEvidence(namespace=INFERENCE_NAMESPACE, timestamp=ts)

    metric_data = [
        {
            "MetricName": "InferenceLatency",
            "Value": metrics.stable_latency_p99_ms,
            "Unit": "Milliseconds",
            "Timestamp": ts_dt,
            "Dimensions": _dims(model_type, {"ModelVersion": str(stable_version)}),
        },
        {
            "MetricName": "InferenceLatency",
            "Value": metrics.canary_latency_p99_ms,
            "Unit": "Milliseconds",
            "Timestamp": ts_dt,
            "Dimensions": _dims(model_type, {"ModelVersion": str(canary_version)}),
        },
        {
            "MetricName": "InferenceErrorRate",
            "Value": metrics.canary_error_rate,
            "Unit": "None",
            "Timestamp": ts_dt,
            "Dimensions": _dims(model_type, {"ModelVersion": str(canary_version)}),
        },
    ]
    evidence.add("InferenceLatency", metrics.stable_latency_p99_ms, {"ModelVersion": str(stable_version)})
    evidence.add("InferenceLatency", metrics.canary_latency_p99_ms, {"ModelVersion": str(canary_version)})
    evidence.add("InferenceErrorRate", metrics.canary_error_rate, {"ModelVersion": str(canary_version)})

    try:
        cloudwatch_client.put_metric_data(
            Namespace=INFERENCE_NAMESPACE,
            MetricData=metric_data,
        )
        evidence.ok = True
    except Exception as exc:
        logger.warning("Failed to emit inference metrics: %s", exc)
        evidence.error = str(exc)

    return evidence


def emit_for_scenario(
    cloudwatch_client: Any,
    scenario: Scenario,
    timestamp: Optional[float] = None,
) -> list[EmitEvidence]:
    """Emit all input metrics a scenario defines. Returns a list of evidence."""
    out: list[EmitEvidence] = []
    if scenario.bid_metrics is not None:
        out.append(
            emit_bid_outcome_metrics(
                cloudwatch_client, scenario.bid_metrics, scenario.model_type, timestamp
            )
        )
    if scenario.inference_metrics is not None:
        out.append(
            emit_inference_metrics(
                cloudwatch_client, scenario.inference_metrics, scenario.model_type, timestamp=timestamp
            )
        )
    return out
