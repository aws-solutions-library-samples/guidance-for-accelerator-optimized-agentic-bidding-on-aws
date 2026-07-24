"""Synthetic input-metric generator for the closed-loop demo.

Emits the *input* market metrics for a scenario to the **real** CloudWatch
namespaces the Part 2 loops read from (``ARTF/BidOutcome`` and
``ARTF/Inference``). This is realistic test input the user controls — it is
never a fabricated *result*.

Every emit returns an ``EmitEvidence`` record listing exactly which metrics and
values were written and when, so a consumer can independently verify the input.

Note on CloudWatch read-after-write: ``PutMetricData`` datapoints take a few
seconds (occasionally up to ~1 minute) before ``GetMetricData`` returns them.
For agentic scenarios, the Adaptive Bidding agent is invoked immediately after
this module returns (directly by the browser, not by this module) and reads
these same metrics back via ``GetMetricData`` — so ``emit_bid_outcome_metrics``
polls (bounded, ``max_wait_seconds``) for real read-after-write confirmation
before returning, rather than guessing a fixed sleep or letting the agent race
an unconfirmed write. ``EmitEvidence.visible`` reports whether that confirmation
succeeded, so a caller can see honestly if the window may still be empty.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
    visible: Optional[bool] = None          # confirmed queryable via GetMetricData (agentic only)
    visible_wait_seconds: Optional[float] = None

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


def _wait_for_metric_visibility(
    cloudwatch_client: Any,
    namespace: str,
    metric_name: str,
    window_start: datetime,
    window_end: datetime,
    *,
    max_wait_seconds: float = 12.0,
    poll_interval_seconds: float = 1.5,
) -> tuple[bool, float]:
    """Poll ``GetMetricData`` until the just-written datapoint is actually queryable.

    ``PutMetricData`` is eventually consistent — the docstring above notes it can
    take a few seconds (occasionally up to ~1 minute) before ``GetMetricData``
    returns a just-written point. Rather than guessing a fixed sleep, poll for real
    read-after-write confirmation with a bounded timeout so callers get an honest
    ``visible`` flag instead of silently racing the agent's own read.
    """
    started = time.time()
    query = {
        "Id": "visibility_check",
        "MetricStat": {
            "Metric": {"Namespace": namespace, "MetricName": metric_name},
            "Period": 300,
            "Stat": "Sum",
        },
    }
    while time.time() - started < max_wait_seconds:
        try:
            resp = cloudwatch_client.get_metric_data(
                MetricDataQueries=[query], StartTime=window_start, EndTime=window_end
            )
            results = resp.get("MetricDataResults", [])
            if results and results[0].get("Values"):
                return True, time.time() - started
        except Exception as exc:  # best-effort — report honestly, never raise
            logger.warning("Visibility check for %s/%s failed: %s", namespace, metric_name, exc)
            return False, time.time() - started
        time.sleep(poll_interval_seconds)
    return False, time.time() - started


def emit_bid_outcome_metrics(
    cloudwatch_client: Any,
    metrics: BidOutcomeMetrics,
    model_type: str,
    timestamp: Optional[float] = None,
    *,
    wait_for_visibility: bool = True,
) -> EmitEvidence:
    """Emit the six ``ARTF/BidOutcome`` metrics for a scenario.

    A single datapoint per metric at ``timestamp`` reproduces the target value
    under both Sum and Average statistics (the agent uses Sum for counts/totals
    and Average for the price metrics).

    If ``wait_for_visibility`` is True (default), this blocks briefly (bounded,
    ``max_wait_seconds``) polling ``GetMetricData`` for the ``TotalBids`` datapoint
    to actually become queryable before returning. This closes the real
    read-after-write race where the Adaptive Bidding agent (invoked immediately
    after this call returns) would otherwise query an empty window and honestly
    report "zero bids" even though metrics were just emitted — not a bug in the
    agent, just PutMetricData's eventual-consistency window.

    Best-effort: never raises. Returns evidence with ``ok``/``error``/``visible`` set.
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

    if wait_for_visibility:
        visible, waited = _wait_for_metric_visibility(
            cloudwatch_client,
            BID_OUTCOME_NAMESPACE,
            "TotalBids",
            window_start=ts_dt - timedelta(minutes=1),
            window_end=datetime.now(tz=timezone.utc) + timedelta(minutes=1),
        )
        evidence.visible = visible
        evidence.visible_wait_seconds = round(waited, 2)
        if not visible:
            logger.warning(
                "Bid-outcome metrics not confirmed queryable in CloudWatch after %.1fs; "
                "the agent's next read may still see an empty window.",
                waited,
            )

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
    *,
    wait_for_visibility: bool = True,
) -> list[EmitEvidence]:
    """Emit all input metrics a scenario defines. Returns a list of evidence.

    ``wait_for_visibility`` (default True) blocks briefly confirming the bid-outcome
    metrics are actually queryable before returning — see ``emit_bid_outcome_metrics``.
    """
    out: list[EmitEvidence] = []
    if scenario.bid_metrics is not None:
        out.append(
            emit_bid_outcome_metrics(
                cloudwatch_client,
                scenario.bid_metrics,
                scenario.model_type,
                timestamp,
                wait_for_visibility=wait_for_visibility,
            )
        )
    if scenario.inference_metrics is not None:
        out.append(
            emit_inference_metrics(
                cloudwatch_client, scenario.inference_metrics, scenario.model_type, timestamp=timestamp
            )
        )
    return out
