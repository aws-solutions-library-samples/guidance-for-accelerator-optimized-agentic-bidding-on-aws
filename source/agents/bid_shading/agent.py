"""Bid Shading Strategy Agent — evaluates market state and adjusts bidding parameters.

Deployed as a Bedrock AgentCore Runtime, invoked every 5 minutes by EventBridge
Scheduler. The agent reads recent market metrics from CloudWatch, computes
parameter adjustments using a proportional policy, writes updates to the
DynamoDB-backed Parameter Store, and emits a parameter-update event.

All decisions are deterministic given the same inputs — no randomness.

Requirements: 7.2, 7.3, 7.4, 7.5, 7.7, 8.5, 10.1
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MarketState:
    """Aggregated market metrics over a time window.

    Computed from CloudWatch BidOutcome metrics.
    """

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


@dataclass
class ParameterUpdate:
    """Describes a single parameter adjustment made by the agent."""

    parameter_name: str
    old_value: float
    new_value: float
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "target_win_rate": 0.35,
    "win_rate_tolerance": 0.05,
    "max_adjustment": 0.05,
    "learning_rate": 0.1,
    "min_samples": 1000,
    "model_type": "dlrm_bid_shader",
    "window_minutes": 5,
    "cloudwatch_namespace": "ARTF/BidOutcome",
}


# ---------------------------------------------------------------------------
# BidShadingStrategyAgent
# ---------------------------------------------------------------------------


class BidShadingStrategyAgent:
    """Agent that adjusts shade_factor and conversion_value based on market state.

    The agent is stateless between invocations — all state is read from
    CloudWatch (metrics) and DynamoDB (current parameters).

    Parameters:
        parameter_store: An instance of shared.parameter_store.ParameterStore.
        cloudwatch_client: A boto3 CloudWatch client.
        config: Optional overrides for DEFAULT_CONFIG keys.
    """

    def __init__(
        self,
        parameter_store: Any,
        cloudwatch_client: Any,
        config: dict | None = None,
    ) -> None:
        self._parameter_store = parameter_store
        self._cloudwatch = cloudwatch_client
        self._config = {**DEFAULT_CONFIG, **(config or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_and_adjust(self) -> list[ParameterUpdate]:
        """Run a full evaluation cycle: read metrics → compute → write.

        Returns:
            List of ParameterUpdate objects for each parameter that was changed.
            Empty list if no adjustments were made (insufficient data or within
            tolerance).
        """
        window_minutes = self._config["window_minutes"]
        model_type = self._config["model_type"]
        min_samples = self._config["min_samples"]

        # Step 1: Get market state from CloudWatch
        state = await self.get_market_state(window_minutes=window_minutes)

        # Step 2: Check minimum samples threshold (Req 7.7)
        if state.total_bids < min_samples:
            logger.info(
                "Insufficient samples (%d < %d), skipping adjustment",
                state.total_bids,
                min_samples,
            )
            return []

        # Step 3: Read current parameters from the store
        current_params = await self._parameter_store.read_all_parameters(model_type)

        # Step 4: Compute adjustments using the policy
        updates = self.compute_adjustment(state, current_params)

        # Step 5: Write updates to parameter store with version and reason
        applied_updates: list[ParameterUpdate] = []
        for update in updates:
            if update.old_value == update.new_value:
                continue  # No change needed

            param_state = current_params.get(update.parameter_name)
            if param_state is None:
                logger.warning(
                    "Parameter '%s' not found in store, skipping",
                    update.parameter_name,
                )
                continue

            # Write via the parameter store (Req 7.5 — version + reason)
            await self._parameter_store.update_parameter(
                model_type=model_type,
                parameter_name=update.parameter_name,
                new_value=update.new_value,
                updated_by="bid_shading_agent",
                reason=update.reason,
                confidence=update.confidence,
                expected_version=param_state.version,
            )

            applied_updates.append(update)

            logger.info(
                "Updated %s: %.4f → %.4f (reason: %s)",
                update.parameter_name,
                update.old_value,
                update.new_value,
                update.reason,
            )

        # Step 6: Emit parameter-update event via CloudWatch (Req 7.5)
        if applied_updates:
            self._emit_parameter_update_event(applied_updates)

        return applied_updates

    async def get_market_state(self, window_minutes: int = 5) -> MarketState:
        """Query CloudWatch metrics and compute the current MarketState.

        Args:
            window_minutes: How far back to look for metrics (default 5).

        Returns:
            MarketState populated from CloudWatch metric data.
        """
        namespace = self._config["cloudwatch_namespace"]
        now = datetime.now(tz=timezone.utc)
        start_time = now - timedelta(minutes=window_minutes)

        metric_queries = [
            self._build_metric_query("total_bids", "TotalBids", "Sum", namespace),
            self._build_metric_query("wins", "Wins", "Sum", namespace),
            self._build_metric_query("avg_price_paid", "AvgPricePaid", "Average", namespace),
            self._build_metric_query("avg_shaded_price", "AvgShadedPrice", "Average", namespace),
            self._build_metric_query("total_revenue", "TotalRevenue", "Sum", namespace),
            self._build_metric_query("total_cost", "TotalCost", "Sum", namespace),
        ]

        response = self._cloudwatch.get_metric_data(
            MetricDataQueries=metric_queries,
            StartTime=start_time,
            EndTime=now,
        )

        # Parse metric results into a dict keyed by Id
        metric_values = self._parse_metric_response(response)

        total_bids = int(metric_values.get("total_bids", 0))
        wins = int(metric_values.get("wins", 0))
        avg_price_paid = metric_values.get("avg_price_paid", 0.0)
        avg_shaded_price = metric_values.get("avg_shaded_price", 0.0)
        total_revenue = metric_values.get("total_revenue", 0.0)
        total_cost = metric_values.get("total_cost", 0.0)

        # Derived metrics
        win_rate = wins / total_bids if total_bids > 0 else 0.0
        roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0.0
        competitive_pressure = 1.0 - win_rate  # simple proxy

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

    def compute_adjustment(
        self, state: MarketState, current_params: dict
    ) -> list[ParameterUpdate]:
        """Compute parameter adjustments based on market state and current values.

        Pure logic — no I/O. All decisions are deterministic.

        Args:
            state: Current MarketState from CloudWatch.
            current_params: Dict mapping parameter_name → ParameterState.

        Returns:
            List of ParameterUpdate objects (may include no-ops if within tolerance).
        """
        updates: list[ParameterUpdate] = []

        # --- shade_factor policy ---
        shade_update = self._compute_shade_factor_adjustment(state, current_params)
        updates.append(shade_update)

        # --- conversion_value policy ---
        cv_update = self._compute_conversion_value_adjustment(state, current_params)
        updates.append(cv_update)

        return updates

    # ------------------------------------------------------------------
    # Private: shade_factor policy (from design pseudocode)
    # ------------------------------------------------------------------

    def _compute_shade_factor_adjustment(
        self, state: MarketState, current_params: dict
    ) -> ParameterUpdate:
        """Policy gradient approximation for shade_factor optimization.

        Direction: increase shade_factor → bid higher → win more.
        """
        target = self._config["target_win_rate"]
        tolerance = self._config["win_rate_tolerance"]
        max_adj = self._config["max_adjustment"]
        lr = self._config["learning_rate"]

        # Get current shade_factor from parameter store
        shade_param = current_params.get("shade_factor")
        if shade_param is not None:
            current_factor = shade_param.current_value
        else:
            # Fallback: estimate from market state
            current_factor = (
                state.avg_shaded_price / state.avg_price_paid
                if state.avg_price_paid > 0
                else 0.65
            )

        # Error signal
        error = state.win_rate - target

        if abs(error) <= tolerance:
            # Within acceptable range — no adjustment (Req 7.3)
            return ParameterUpdate(
                parameter_name="shade_factor",
                old_value=current_factor,
                new_value=current_factor,
                reason="Win rate within tolerance",
                confidence=min(0.95, state.total_bids / 10000),
            )

        # Proportional adjustment (from design pseudocode)
        # If win_rate < target: error < 0 → raw_adjustment > 0 → bid more aggressively
        # If win_rate > target: error > 0 → raw_adjustment < 0 → bid less aggressively
        raw_adjustment = -error * lr

        # Incorporate ROI signal: if ROI is high, be more aggressive
        roi_signal = max(0.0, min(1.0, state.roi))
        raw_adjustment *= 0.5 + 0.5 * roi_signal

        # Bound the adjustment to ±max_adjustment (Req 7.4 — ±5%)
        adjustment = max(-max_adj, min(max_adj, raw_adjustment))

        # Clamp new value to global bounds [0.3, 0.95] (Req 8.1)
        new_value = max(0.3, min(0.95, current_factor + adjustment))

        return ParameterUpdate(
            parameter_name="shade_factor",
            old_value=current_factor,
            new_value=new_value,
            reason=f"Win rate {state.win_rate:.3f} vs target {target:.3f}, ROI {state.roi:.3f}",
            confidence=min(0.95, state.total_bids / 10000),
        )

    # ------------------------------------------------------------------
    # Private: conversion_value policy
    # ------------------------------------------------------------------

    def _compute_conversion_value_adjustment(
        self, state: MarketState, current_params: dict
    ) -> ParameterUpdate:
        """Adjust conversion_value based on ROI and win rate signals.

        Policy:
        - If ROI is negative and win_rate is above target: decrease (overbidding)
        - If ROI is positive and win_rate is below target: increase (underbidding)
        """
        target = self._config["target_win_rate"]
        max_adj = self._config["max_adjustment"]
        lr = self._config["learning_rate"]

        # Get current conversion_value from parameter store
        cv_param = current_params.get("conversion_value")
        if cv_param is not None:
            current_cv = cv_param.current_value
        else:
            current_cv = 10.0  # Default starting point

        # Determine adjustment direction
        adjustment = 0.0
        reason = "No conversion_value adjustment needed"

        if state.roi < 0 and state.win_rate > target:
            # Overbidding: ROI is negative but we're winning enough
            # → decrease conversion_value to bid less
            adjustment = -abs(state.roi) * lr * current_cv
            reason = (
                f"ROI negative ({state.roi:.3f}) with high win rate "
                f"({state.win_rate:.3f}), reducing conversion_value"
            )
        elif state.roi > 0 and state.win_rate < target:
            # Underbidding: ROI is positive but we're not winning enough
            # → increase conversion_value to bid more
            adjustment = state.roi * lr * current_cv
            reason = (
                f"ROI positive ({state.roi:.3f}) with low win rate "
                f"({state.win_rate:.3f}), increasing conversion_value"
            )

        if adjustment == 0.0:
            return ParameterUpdate(
                parameter_name="conversion_value",
                old_value=current_cv,
                new_value=current_cv,
                reason=reason,
                confidence=min(0.95, state.total_bids / 10000),
            )

        # Bound to ±5% of current value (Req 7.4)
        max_delta = max_adj * current_cv
        adjustment = max(-max_delta, min(max_delta, adjustment))

        # Clamp to global bounds [1.0, 50.0] (Req 8.1)
        new_value = max(1.0, min(50.0, current_cv + adjustment))

        return ParameterUpdate(
            parameter_name="conversion_value",
            old_value=current_cv,
            new_value=new_value,
            reason=reason,
            confidence=min(0.95, state.total_bids / 10000),
        )

    # ------------------------------------------------------------------
    # Private: CloudWatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metric_query(
        metric_id: str, metric_name: str, stat: str, namespace: str
    ) -> dict:
        """Build a CloudWatch GetMetricData query structure."""
        return {
            "Id": metric_id,
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": metric_name,
                },
                "Period": 300,  # 5 minutes
                "Stat": stat,
            },
        }

    @staticmethod
    def _parse_metric_response(response: dict) -> dict[str, float]:
        """Parse CloudWatch GetMetricData response into a simple dict.

        Returns the most recent value for each metric (last datapoint).
        """
        values: dict[str, float] = {}
        for result in response.get("MetricDataResults", []):
            metric_id = result.get("Id", "")
            datapoints = result.get("Values", [])
            if datapoints:
                # Take the most recent value (first in descending order)
                values[metric_id] = float(datapoints[0])
            else:
                values[metric_id] = 0.0
        return values

    # ------------------------------------------------------------------
    # Private: Event emission
    # ------------------------------------------------------------------

    def _emit_parameter_update_event(self, updates: list[ParameterUpdate]) -> None:
        """Emit a parameter-update custom metric to CloudWatch (Req 7.5).

        This signals downstream systems that parameters have been updated.
        """
        namespace = self._config["cloudwatch_namespace"]
        metric_data = []

        for update in updates:
            metric_data.append(
                {
                    "MetricName": "ParameterUpdate",
                    "Dimensions": [
                        {"Name": "ParameterName", "Value": update.parameter_name},
                        {"Name": "ModelType", "Value": self._config["model_type"]},
                    ],
                    "Value": abs(update.new_value - update.old_value),
                    "Unit": "None",
                    "Timestamp": datetime.now(tz=timezone.utc),
                }
            )

        if metric_data:
            try:
                self._cloudwatch.put_metric_data(
                    Namespace=namespace,
                    MetricData=metric_data,
                )
            except Exception as exc:
                logger.error("Failed to emit parameter-update event: %s", exc)
