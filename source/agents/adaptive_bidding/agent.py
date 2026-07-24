"""Adaptive Bidding Strategy Agent — reasoning agent (Strands + Amazon Bedrock).

Each invocation the agent:
  1. reads the REAL market state from CloudWatch (win rate, ROI, prices, sample counts),
  2. reads the current bidding parameters from the DynamoDB Parameter Store,
  3. reasons (via a Bedrock model) about whether and how to adjust ``shade_factor``
     and ``conversion_value`` to improve bidding outcomes, and
  4. writes any adjustments back to the Parameter Store.

There is deliberately **no hardcoded target win rate, learning rate, tolerance, or
proportional-gradient formula** in this agent. The model reasons over the real numbers
and explains its choice. Hard safety limits are enforced by the Parameter Store write
layer, not here:
  - global bounds ``shade_factor ∈ [0.3, 0.95]``, ``conversion_value ∈ [1.0, 50.0]``
  - per-update max delta (prevents oscillation)
  - optimistic-concurrency version checks

No fabrication / no fallback: the agent reads real metrics and writes real updates. If
the Bedrock model is unavailable, the invocation raises — it does NOT silently fall back
to a formula or invent an adjustment.

Requirements: 7.2, 7.3, 7.4, 7.5, 7.7, 8.5, 10.1 (parent spec); Req 2.7, 2.8, 2.9 (this spec).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-decision configuration (data plumbing only — not tuning constants)
# ---------------------------------------------------------------------------

MODEL_TYPE = "dlrm_bid_shader"           # ARTF model type (parameter partition key)
CLOUDWATCH_NAMESPACE = "ARTF/BidOutcome"  # namespace the RTB pipeline emits outcomes to
DEFAULT_WINDOW_MINUTES = 5                # how far back to aggregate market metrics


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MarketState:
    """Aggregated market metrics over a time window, computed from CloudWatch."""

    window_start: float
    window_end: float
    total_bids: int
    wins: int
    win_rate: float
    avg_price_paid: float
    avg_shaded_price: float
    total_revenue: float
    total_cost: float
    roi: float
    competitive_pressure: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_minutes": round((self.window_end - self.window_start) / 60.0, 2),
            "total_bids": self.total_bids,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "avg_price_paid": round(self.avg_price_paid, 4),
            "avg_shaded_price": round(self.avg_shaded_price, 4),
            "total_revenue": round(self.total_revenue, 4),
            "total_cost": round(self.total_cost, 4),
            "roi": round(self.roi, 4),
            "competitive_pressure": round(self.competitive_pressure, 4),
        }


@dataclass
class ParameterUpdate:
    """A single parameter adjustment applied by the agent."""

    parameter_name: str
    old_value: float
    new_value: float
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# CloudWatch helpers
# ---------------------------------------------------------------------------


def _build_metric_query(metric_id: str, metric_name: str, stat: str, namespace: str) -> dict:
    """Build a CloudWatch GetMetricData query structure."""
    return {
        "Id": metric_id,
        "MetricStat": {
            "Metric": {"Namespace": namespace, "MetricName": metric_name},
            "Period": 300,
            "Stat": stat,
        },
    }


def _parse_metric_response(response: dict) -> dict[str, float]:
    """Parse a CloudWatch GetMetricData response into {id: most-recent-value}."""
    values: dict[str, float] = {}
    for result in response.get("MetricDataResults", []):
        metric_id = result.get("Id", "")
        datapoints = result.get("Values", [])
        values[metric_id] = float(datapoints[0]) if datapoints else 0.0
    return values


def compute_market_state(
    cloudwatch_client: Any,
    *,
    namespace: str = CLOUDWATCH_NAMESPACE,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> MarketState:
    """Query CloudWatch and compute the current MarketState from real metrics.

    This is a read of real telemetry — it never fabricates values. If a metric has
    no datapoints in the window, it is reported as 0 (honest "no data"), and the
    reasoning layer is told the sample counts so it can decline to act on thin data.
    """
    now = datetime.now(tz=timezone.utc)
    start_time = now - timedelta(minutes=window_minutes)

    queries = [
        _build_metric_query("total_bids", "TotalBids", "Sum", namespace),
        _build_metric_query("wins", "Wins", "Sum", namespace),
        _build_metric_query("avg_price_paid", "AvgPricePaid", "Average", namespace),
        _build_metric_query("avg_shaded_price", "AvgShadedPrice", "Average", namespace),
        _build_metric_query("total_revenue", "TotalRevenue", "Sum", namespace),
        _build_metric_query("total_cost", "TotalCost", "Sum", namespace),
    ]

    response = cloudwatch_client.get_metric_data(
        MetricDataQueries=queries,
        StartTime=start_time,
        EndTime=now,
    )
    m = _parse_metric_response(response)

    total_bids = int(m.get("total_bids", 0))
    wins = int(m.get("wins", 0))
    avg_price_paid = m.get("avg_price_paid", 0.0)
    avg_shaded_price = m.get("avg_shaded_price", 0.0)
    total_revenue = m.get("total_revenue", 0.0)
    total_cost = m.get("total_cost", 0.0)

    win_rate = wins / total_bids if total_bids > 0 else 0.0
    roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0.0
    competitive_pressure = 1.0 - win_rate

    return MarketState(
        window_start=start_time.timestamp(),
        window_end=now.timestamp(),
        total_bids=total_bids,
        wins=wins,
        win_rate=win_rate,
        avg_price_paid=avg_price_paid,
        avg_shaded_price=avg_shaded_price,
        total_revenue=total_revenue,
        total_cost=total_cost,
        roi=roi,
        competitive_pressure=competitive_pressure,
    )


# ---------------------------------------------------------------------------
# Async bridge — run ParameterStore coroutines from a synchronous worker thread
# ---------------------------------------------------------------------------


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine to completion from a thread with no running loop.

    The reasoning cycle runs inside ``asyncio.to_thread`` (no active event loop),
    so a fresh loop here is safe. ParameterStore's I/O is boto3 (synchronous under
    the hood), so there are no cross-loop async resources to leak.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# System prompt — objective + constraints, NOT a decision formula
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Adaptive Bidding Strategy Agent for a real-time bidding platform.

Your job each invocation: decide whether the bidding parameters shade_factor and
conversion_value should be adjusted, based ONLY on the real market metrics you read
this cycle, then apply any adjustments.

Objective:
- Maximize return on ad spend (ROI = (revenue - cost) / cost) while keeping a healthy,
  competitive win rate. Bidding too low loses valuable impressions; bidding too high
  overpays and erodes ROI.

How to work:
1. Call get_market_state to read the current real market metrics.
2. Call read_current_parameters to see the current shade_factor and conversion_value,
   their versions, and their allowed bounds.
3. Reason about what the metrics imply. There is no fixed target or formula — weigh
   win rate, ROI, price paid vs. shaded price, and competitive pressure together.
4. If, and only if, the data justifies a change, call write_parameter for each
   parameter you want to move, with a concise reason grounded in the metrics and a
   confidence in [0,1]. Make at most one adjustment per parameter this cycle.
5. If the sample counts are too small to draw a conclusion, or the parameters are
   already well-placed, make NO change and say so.

Hard rules:
- Never invent metrics. Reason only from the values the tools return.
- The parameter store enforces absolute bounds and a maximum change per update; if a
  write is rejected, respect that — do not try to circumvent it.
- Finish with a one-paragraph rationale summarizing what you observed and did.
"""


# ---------------------------------------------------------------------------
# AdaptiveBiddingStrategyAgent
# ---------------------------------------------------------------------------


class AdaptiveBiddingStrategyAgent:
    """Reasoning agent that adjusts bidding parameters from real market metrics.

    The agent is stateless between invocations — all state is read from CloudWatch
    (metrics) and DynamoDB (current parameters). Decisions are made by a Bedrock model
    via Strands tools; hard safety limits are enforced by the Parameter Store.

    Args:
        parameter_store: shared.parameter_store.ParameterStore instance (async API).
        cloudwatch_client: boto3 CloudWatch client.
        model_id: Bedrock model id for the reasoning layer (required).
        region: AWS region for the Bedrock model.
        window_minutes: market-metric aggregation window.
        namespace: CloudWatch namespace to read outcomes from.
        model_type: ARTF model type (parameter partition key).
    """

    def __init__(
        self,
        parameter_store: Any,
        cloudwatch_client: Any,
        *,
        model_id: str,
        region: str,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        namespace: str = CLOUDWATCH_NAMESPACE,
        model_type: str = MODEL_TYPE,
    ) -> None:
        if not model_id:
            raise ValueError("model_id is required for the Adaptive Bidding reasoning agent")
        self._parameter_store = parameter_store
        self._cloudwatch = cloudwatch_client
        self._model_id = model_id
        self._region = region
        self._window_minutes = window_minutes
        self._namespace = namespace
        self._model_type = model_type
        self._applied: list[ParameterUpdate] = []

    # ------------------------------------------------------------------
    # Core operations (also directly unit-testable without Strands)
    # ------------------------------------------------------------------

    def get_market_state(self) -> MarketState:
        """Read the current market state from CloudWatch (real telemetry)."""
        return compute_market_state(
            self._cloudwatch,
            namespace=self._namespace,
            window_minutes=self._window_minutes,
        )

    def read_parameters(self) -> dict[str, Any]:
        """Read current parameters from the Parameter Store."""
        return _run_coro(self._parameter_store.read_all_parameters(self._model_type))

    def apply_update(
        self, parameter_name: str, new_value: float, reason: str, confidence: float
    ) -> dict[str, Any]:
        """Apply one parameter update through the Parameter Store.

        Bounds, per-update max delta, and optimistic version are enforced by the store.
        Rejections are returned honestly (not swallowed, not retried around the bound).
        """
        # Import lazily so this module imports without the store's optional deps.
        from shared.parameter_store import OptimisticLockError, ParameterBoundsError

        params = _run_coro(self._parameter_store.read_all_parameters(self._model_type))
        current = params.get(parameter_name)
        if current is None:
            return {
                "status": "rejected",
                "parameter_name": parameter_name,
                "reason": f"parameter '{parameter_name}' does not exist in the store",
            }

        try:
            updated = _run_coro(
                self._parameter_store.update_parameter(
                    model_type=self._model_type,
                    parameter_name=parameter_name,
                    new_value=float(new_value),
                    updated_by="adaptive_bidding_agent",
                    reason=reason,
                    confidence=float(confidence),
                    expected_version=current.version,
                )
            )
        except ParameterBoundsError as exc:
            return {
                "status": "rejected",
                "parameter_name": parameter_name,
                "reason": f"bounds/max-delta rejected the write: {exc}",
            }
        except OptimisticLockError as exc:
            return {
                "status": "rejected",
                "parameter_name": parameter_name,
                "reason": f"optimistic lock rejected the write: {exc}",
            }

        update = ParameterUpdate(
            parameter_name=parameter_name,
            old_value=current.current_value,
            new_value=updated.current_value,
            reason=reason,
            confidence=float(confidence),
        )
        self._applied.append(update)
        logger.info(
            "Applied %s: %.4f -> %.4f (%s)",
            parameter_name,
            update.old_value,
            update.new_value,
            reason,
        )
        return {
            "status": "applied",
            "parameter_name": parameter_name,
            "old_value": update.old_value,
            "new_value": update.new_value,
            "version": updated.version,
        }

    def emit_update_event(self) -> None:
        """Emit a CloudWatch ParameterUpdate metric for each applied change (real event)."""
        if not self._applied:
            return
        metric_data = [
            {
                "MetricName": "ParameterUpdate",
                "Dimensions": [
                    {"Name": "ParameterName", "Value": u.parameter_name},
                    {"Name": "ModelType", "Value": self._model_type},
                ],
                "Value": abs(u.new_value - u.old_value),
                "Unit": "None",
                "Timestamp": datetime.now(tz=timezone.utc),
            }
            for u in self._applied
        ]
        try:
            self._cloudwatch.put_metric_data(Namespace=self._namespace, MetricData=metric_data)
        except Exception as exc:  # noqa: BLE001 - telemetry emission must not fail the cycle
            logger.error("Failed to emit parameter-update event: %s", exc)

    # ------------------------------------------------------------------
    # Reasoning cycle (runs inside a worker thread; builds the Strands agent)
    # ------------------------------------------------------------------

    def _build_tools(self) -> list:
        """Build Strands tools as closures over this agent's real clients."""
        from strands import tool

        @tool
        def get_market_state() -> dict:
            """Read the current real market metrics (win rate, ROI, prices, sample counts)."""
            return self.get_market_state().to_dict()

        @tool
        def read_current_parameters() -> dict:
            """Read current bidding parameters, their versions, and their allowed bounds."""
            params = self.read_parameters()
            return {
                name: {
                    "current_value": p.current_value,
                    "version": p.version,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "max_delta_per_update": p.max_delta_per_update,
                }
                for name, p in params.items()
            }

        @tool
        def write_parameter(parameter_name: str, new_value: float, reason: str, confidence: float) -> dict:
            """Apply an adjustment to a bidding parameter.

            The parameter store enforces absolute bounds, the maximum change per update,
            and optimistic version checks; a rejected write is returned as-is.

            Args:
                parameter_name: "shade_factor" or "conversion_value".
                new_value: the proposed new value.
                reason: concise, metric-grounded reason for the change.
                confidence: your confidence in the change, in [0, 1].
            """
            return self.apply_update(parameter_name, new_value, reason, confidence)

        return [get_market_state, read_current_parameters, write_parameter]

    def run_cycle(self) -> dict[str, Any]:
        """Run one reasoning cycle. Intended to be called via asyncio.to_thread.

        Returns a dict with the applied updates, the model's rationale, and the market
        state snapshot. Raises if the reasoning model cannot be reached (no fallback).
        """
        from strands import Agent
        from strands.models import BedrockModel

        self._applied = []

        # Snapshot the market state up front so it is included in the response even if
        # the model chooses to call the tool again.
        market_state = self.get_market_state()

        model = BedrockModel(model_id=self._model_id, region_name=self._region)
        agent = Agent(
            model=model,
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )

        prompt = (
            "A new 5-minute market window has closed. Read the current market state and "
            "parameters, reason about whether shade_factor and/or conversion_value should "
            "change, apply any warranted adjustments, and summarize your reasoning."
        )
        result = agent(prompt)
        rationale = str(result)

        return {
            "updates": [
                {
                    "parameter_name": u.parameter_name,
                    "old_value": u.old_value,
                    "new_value": u.new_value,
                    "reason": u.reason,
                    "confidence": u.confidence,
                }
                for u in self._applied
            ],
            "updates_count": len(self._applied),
            "rationale": rationale,
            "market_state": market_state.to_dict(),
        }
