"""Tests for agents.adaptive_bidding.agent — AdaptiveBiddingStrategyAgent.

Validates:
- compute_adjustment with win_rate below target → increases shade_factor
- compute_adjustment with win_rate above target + tolerance → decreases shade_factor
- compute_adjustment within tolerance → no change
- max adjustment bounded to ±5%
- min_samples threshold (fewer samples → no adjustment)
- conversion_value adjustment logic
- evaluate_and_adjust integration (market state → compute → write)

**Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.7, 8.5, 10.1**
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.adaptive_bidding.agent import (
    AdaptiveBiddingStrategyAgent,
    MarketState,
    ParameterUpdate,
    DEFAULT_CONFIG,
)
from shared.parameter_store import ParameterState


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures (no fabricated data — real domain values)
# ---------------------------------------------------------------------------


def _make_market_state(
    total_bids: int = 5000,
    wins: int = 1500,
    win_rate: float = 0.30,
    avg_price_paid: float = 2.50,
    avg_shaded_price: float = 1.75,
    total_revenue: float = 3000.0,
    total_cost: float = 2500.0,
    roi: float = 0.2,
    competitive_pressure: float = 0.7,
) -> MarketState:
    """Create a deterministic MarketState fixture."""
    now = 1700000000.0
    return MarketState(
        window_start=now - 300,
        window_end=now,
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


def _make_param_state(
    parameter_name: str = "shade_factor",
    current_value: float = 0.65,
    version: int = 5,
) -> ParameterState:
    """Create a deterministic ParameterState fixture."""
    bounds = {
        "shade_factor": (0.3, 0.95, 0.05),
        "conversion_value": (1.0, 50.0, 2.5),
    }
    min_val, max_val, max_delta = bounds.get(parameter_name, (0.0, 100.0, 5.0))
    return ParameterState(
        model_type="dlrm_bid_shader",
        parameter_name=parameter_name,
        current_value=current_value,
        previous_value=current_value - 0.01,
        updated_at=1700000000.0,
        updated_by="adaptive_bidding_agent",
        version=version,
        min_value=min_val,
        max_value=max_val,
        max_delta_per_update=max_delta,
        reason="Previous adjustment",
        confidence=0.85,
    )


def _make_current_params(
    shade_factor: float = 0.65, conversion_value: float = 10.0
) -> dict:
    """Create a dict of current parameters matching ParameterStore format."""
    return {
        "shade_factor": _make_param_state("shade_factor", shade_factor),
        "conversion_value": _make_param_state("conversion_value", conversion_value),
    }


def _make_agent(config_overrides: dict | None = None) -> AdaptiveBiddingStrategyAgent:
    """Create a AdaptiveBiddingStrategyAgent with mocked external dependencies."""
    mock_store = MagicMock()
    mock_cloudwatch = MagicMock()
    config = config_overrides or {}
    return AdaptiveBiddingStrategyAgent(
        parameter_store=mock_store,
        cloudwatch_client=mock_cloudwatch,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests: shade_factor policy
# ---------------------------------------------------------------------------


class TestShadeFactorPolicy:
    """Test shade_factor adjustment logic."""

    def test_win_rate_below_target_increases_shade_factor(self):
        """When win_rate < target - tolerance, shade_factor should increase."""
        agent = _make_agent()
        # win_rate=0.25 is below target=0.35 by more than tolerance=0.05
        state = _make_market_state(win_rate=0.25, roi=0.5)
        params = _make_current_params(shade_factor=0.65)

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade_update.new_value > shade_update.old_value
        assert shade_update.old_value == 0.65

    def test_win_rate_above_target_plus_tolerance_decreases_shade_factor(self):
        """When win_rate > target + tolerance, shade_factor should decrease."""
        agent = _make_agent()
        # win_rate=0.50 is above target=0.35 + tolerance=0.05
        state = _make_market_state(win_rate=0.50, roi=0.3)
        params = _make_current_params(shade_factor=0.65)

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade_update.new_value < shade_update.old_value

    def test_win_rate_within_tolerance_no_change(self):
        """When win_rate is within [target - tolerance, target + tolerance], no change."""
        agent = _make_agent()
        # win_rate=0.36 is within target=0.35 ± tolerance=0.05
        state = _make_market_state(win_rate=0.36, roi=0.2)
        params = _make_current_params(shade_factor=0.65)

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade_update.new_value == shade_update.old_value
        assert "within tolerance" in shade_update.reason.lower()

    def test_win_rate_at_target_boundary_no_change(self):
        """When win_rate is exactly at target, no change."""
        agent = _make_agent()
        state = _make_market_state(win_rate=0.35, roi=0.2)
        params = _make_current_params(shade_factor=0.65)

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade_update.new_value == shade_update.old_value

    def test_max_adjustment_bounded_positive(self):
        """Adjustment should never exceed +max_adjustment (5%)."""
        agent = _make_agent({"learning_rate": 1.0})  # Aggressive lr to push the limit
        # Very low win_rate → wants a large positive adjustment
        state = _make_market_state(win_rate=0.05, roi=1.0)
        params = _make_current_params(shade_factor=0.65)

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        max_adj = DEFAULT_CONFIG["max_adjustment"]
        actual_adjustment = shade_update.new_value - shade_update.old_value
        assert actual_adjustment <= max_adj + 1e-9

    def test_max_adjustment_bounded_negative(self):
        """Adjustment should never exceed -max_adjustment (5%)."""
        agent = _make_agent({"learning_rate": 1.0})  # Aggressive lr
        # Very high win_rate → wants a large negative adjustment
        state = _make_market_state(win_rate=0.95, roi=0.5)
        params = _make_current_params(shade_factor=0.65)

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        max_adj = DEFAULT_CONFIG["max_adjustment"]
        actual_adjustment = shade_update.new_value - shade_update.old_value
        assert actual_adjustment >= -max_adj - 1e-9

    def test_shade_factor_clamped_to_lower_bound(self):
        """shade_factor cannot go below 0.3 even with large negative adjustment."""
        agent = _make_agent({"learning_rate": 1.0})
        state = _make_market_state(win_rate=0.95, roi=0.8)
        params = _make_current_params(shade_factor=0.31)  # Near lower bound

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade_update.new_value >= 0.3

    def test_shade_factor_clamped_to_upper_bound(self):
        """shade_factor cannot go above 0.95 even with large positive adjustment."""
        agent = _make_agent({"learning_rate": 1.0})
        state = _make_market_state(win_rate=0.05, roi=1.0)
        params = _make_current_params(shade_factor=0.94)  # Near upper bound

        updates = agent.compute_adjustment(state, params)
        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade_update.new_value <= 0.95

    def test_roi_signal_amplifies_adjustment(self):
        """Higher ROI should produce a larger adjustment magnitude."""
        agent = _make_agent()
        params = _make_current_params(shade_factor=0.65)

        # Same win_rate deficit, different ROI
        state_low_roi = _make_market_state(win_rate=0.20, roi=0.0)
        state_high_roi = _make_market_state(win_rate=0.20, roi=1.0)

        updates_low = agent.compute_adjustment(state_low_roi, params)
        updates_high = agent.compute_adjustment(state_high_roi, params)

        shade_low = next(u for u in updates_low if u.parameter_name == "shade_factor")
        shade_high = next(u for u in updates_high if u.parameter_name == "shade_factor")

        # Higher ROI means bigger adjustment (both positive since win_rate < target)
        adj_low = shade_low.new_value - shade_low.old_value
        adj_high = shade_high.new_value - shade_high.old_value

        assert adj_high > adj_low


# ---------------------------------------------------------------------------
# Tests: conversion_value policy
# ---------------------------------------------------------------------------


class TestConversionValuePolicy:
    """Test conversion_value adjustment logic."""

    def test_negative_roi_high_win_rate_decreases_cv(self):
        """When ROI is negative and win_rate above target → decrease (overbidding)."""
        agent = _make_agent()
        state = _make_market_state(win_rate=0.50, roi=-0.2)
        params = _make_current_params(conversion_value=10.0)

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        assert cv_update.new_value < cv_update.old_value

    def test_positive_roi_low_win_rate_increases_cv(self):
        """When ROI is positive and win_rate below target → increase (underbidding)."""
        agent = _make_agent()
        state = _make_market_state(win_rate=0.20, roi=0.5)
        params = _make_current_params(conversion_value=10.0)

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        assert cv_update.new_value > cv_update.old_value

    def test_positive_roi_high_win_rate_no_change(self):
        """When ROI is positive and win_rate above target → no change."""
        agent = _make_agent()
        state = _make_market_state(win_rate=0.50, roi=0.5)
        params = _make_current_params(conversion_value=10.0)

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        assert cv_update.new_value == cv_update.old_value

    def test_negative_roi_low_win_rate_no_change(self):
        """When ROI is negative and win_rate below target → no change."""
        agent = _make_agent()
        state = _make_market_state(win_rate=0.20, roi=-0.2)
        params = _make_current_params(conversion_value=10.0)

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        assert cv_update.new_value == cv_update.old_value

    def test_cv_max_adjustment_bounded(self):
        """conversion_value adjustment bounded to ±5% of current value."""
        agent = _make_agent({"learning_rate": 10.0})  # Very aggressive
        state = _make_market_state(win_rate=0.50, roi=-5.0)  # Large negative ROI
        params = _make_current_params(conversion_value=10.0)

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        max_delta = 0.05 * 10.0  # 5% of current value = 0.5
        actual_delta = abs(cv_update.new_value - cv_update.old_value)
        assert actual_delta <= max_delta + 1e-9

    def test_cv_clamped_to_lower_bound(self):
        """conversion_value cannot go below 1.0."""
        agent = _make_agent({"learning_rate": 10.0})
        state = _make_market_state(win_rate=0.50, roi=-5.0)
        params = _make_current_params(conversion_value=1.05)  # Near lower bound

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        assert cv_update.new_value >= 1.0

    def test_cv_clamped_to_upper_bound(self):
        """conversion_value cannot go above 50.0."""
        agent = _make_agent({"learning_rate": 10.0})
        state = _make_market_state(win_rate=0.10, roi=5.0)
        params = _make_current_params(conversion_value=49.5)  # Near upper bound

        updates = agent.compute_adjustment(state, params)
        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")

        assert cv_update.new_value <= 50.0


# ---------------------------------------------------------------------------
# Tests: min_samples threshold (Req 7.7)
# ---------------------------------------------------------------------------


class TestMinSamplesThreshold:
    """Test that the agent skips adjustments when samples are insufficient."""

    def test_below_min_samples_returns_empty(self):
        """Fewer than min_samples → evaluate_and_adjust returns empty list."""
        mock_store = AsyncMock()
        mock_cw = MagicMock()

        # CloudWatch returns low sample count
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [500]},  # Below default 1000
                {"Id": "wins", "Values": [150]},
                {"Id": "avg_price_paid", "Values": [2.0]},
                {"Id": "avg_shaded_price", "Values": [1.5]},
                {"Id": "total_revenue", "Values": [200.0]},
                {"Id": "total_cost", "Values": [150.0]},
            ]
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=mock_store,
            cloudwatch_client=mock_cw,
            config={"min_samples": 1000},
        )

        result = asyncio.run(agent.evaluate_and_adjust())
        assert result == []
        # Parameter store should not be read for parameters
        mock_store.read_all_parameters.assert_not_called()

    def test_at_min_samples_proceeds(self):
        """Exactly at min_samples → proceeds with adjustments."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _make_current_params()
        mock_store.update_parameter.return_value = _make_param_state()

        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [1000]},  # Exactly at threshold
                {"Id": "wins", "Values": [200]},  # win_rate=0.2, below target
                {"Id": "avg_price_paid", "Values": [2.0]},
                {"Id": "avg_shaded_price", "Values": [1.5]},
                {"Id": "total_revenue", "Values": [500.0]},
                {"Id": "total_cost", "Values": [400.0]},
            ]
        }
        mock_cw.put_metric_data = MagicMock()

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=mock_store,
            cloudwatch_client=mock_cw,
            config={"min_samples": 1000},
        )

        result = asyncio.run(agent.evaluate_and_adjust())
        # Should proceed and at least attempt adjustments
        mock_store.read_all_parameters.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: evaluate_and_adjust integration
# ---------------------------------------------------------------------------


class TestEvaluateAndAdjustIntegration:
    """Integration test for the full cycle: market state → compute → write."""

    def test_full_cycle_writes_updates(self):
        """A full cycle should read metrics, compute, and write to parameter store."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _make_current_params(
            shade_factor=0.65, conversion_value=10.0
        )
        mock_store.update_parameter.return_value = _make_param_state()

        mock_cw = MagicMock()
        # Win rate below target → should trigger shade_factor increase
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [5000]},
                {"Id": "wins", "Values": [1000]},  # 0.2 win rate
                {"Id": "avg_price_paid", "Values": [2.5]},
                {"Id": "avg_shaded_price", "Values": [1.75]},
                {"Id": "total_revenue", "Values": [3000.0]},
                {"Id": "total_cost", "Values": [2500.0]},
            ]
        }
        mock_cw.put_metric_data = MagicMock()

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=mock_store,
            cloudwatch_client=mock_cw,
        )

        result = asyncio.run(agent.evaluate_and_adjust())

        # Should have written at least shade_factor (win_rate < target triggers it)
        assert len(result) >= 1
        shade_updates = [u for u in result if u.parameter_name == "shade_factor"]
        assert len(shade_updates) == 1
        assert shade_updates[0].new_value > shade_updates[0].old_value

        # Parameter store update_parameter was called
        mock_store.update_parameter.assert_called()

        # CloudWatch put_metric_data was called (parameter-update event)
        mock_cw.put_metric_data.assert_called()

    def test_no_change_within_tolerance_skips_write(self):
        """When within tolerance, no write to parameter store."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _make_current_params()

        mock_cw = MagicMock()
        # Win rate exactly at target, positive ROI and above target → no cv change either
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [5000]},
                {"Id": "wins", "Values": [1750]},  # 0.35 win rate = target
                {"Id": "avg_price_paid", "Values": [2.5]},
                {"Id": "avg_shaded_price", "Values": [1.75]},
                {"Id": "total_revenue", "Values": [3000.0]},
                {"Id": "total_cost", "Values": [2500.0]},
            ]
        }
        mock_cw.put_metric_data = MagicMock()

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=mock_store,
            cloudwatch_client=mock_cw,
        )

        result = asyncio.run(agent.evaluate_and_adjust())

        # Within tolerance → no updates written
        assert result == []
        mock_store.update_parameter.assert_not_called()
        mock_cw.put_metric_data.assert_not_called()

    def test_emits_parameter_update_event(self):
        """When parameters change, a CloudWatch metric event is emitted (Req 7.5)."""
        mock_store = AsyncMock()
        mock_store.read_all_parameters.return_value = _make_current_params()
        mock_store.update_parameter.return_value = _make_param_state()

        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [5000]},
                {"Id": "wins", "Values": [1000]},  # win_rate=0.2
                {"Id": "avg_price_paid", "Values": [2.5]},
                {"Id": "avg_shaded_price", "Values": [1.75]},
                {"Id": "total_revenue", "Values": [3000.0]},
                {"Id": "total_cost", "Values": [2500.0]},
            ]
        }
        mock_cw.put_metric_data = MagicMock()

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=mock_store,
            cloudwatch_client=mock_cw,
        )

        asyncio.run(agent.evaluate_and_adjust())

        # put_metric_data should have been called with ParameterUpdate metric
        mock_cw.put_metric_data.assert_called_once()
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == "ARTF/BidOutcome"
        metric_data = call_kwargs["MetricData"]
        assert any(m["MetricName"] == "ParameterUpdate" for m in metric_data)


# ---------------------------------------------------------------------------
# Tests: get_market_state
# ---------------------------------------------------------------------------


class TestGetMarketState:
    """Test CloudWatch metric queries and MarketState construction."""

    def test_market_state_computation(self):
        """MarketState should correctly derive win_rate and roi from raw metrics."""
        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [10000]},
                {"Id": "wins", "Values": [3500]},
                {"Id": "avg_price_paid", "Values": [3.0]},
                {"Id": "avg_shaded_price", "Values": [2.1]},
                {"Id": "total_revenue", "Values": [5000.0]},
                {"Id": "total_cost", "Values": [4000.0]},
            ]
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=MagicMock(),
            cloudwatch_client=mock_cw,
        )

        state = asyncio.run(agent.get_market_state(window_minutes=5))

        assert state.total_bids == 10000
        assert state.wins == 3500
        assert abs(state.win_rate - 0.35) < 1e-9
        assert state.avg_price_paid == 3.0
        assert state.avg_shaded_price == 2.1
        assert state.total_revenue == 5000.0
        assert state.total_cost == 4000.0
        assert abs(state.roi - 0.25) < 1e-9  # (5000-4000)/4000
        assert abs(state.competitive_pressure - 0.65) < 1e-9  # 1 - 0.35

    def test_zero_bids_handles_gracefully(self):
        """With zero bids, derived metrics should be 0 (no division errors)."""
        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": [0]},
                {"Id": "wins", "Values": [0]},
                {"Id": "avg_price_paid", "Values": [0.0]},
                {"Id": "avg_shaded_price", "Values": [0.0]},
                {"Id": "total_revenue", "Values": [0.0]},
                {"Id": "total_cost", "Values": [0.0]},
            ]
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=MagicMock(),
            cloudwatch_client=mock_cw,
        )

        state = asyncio.run(agent.get_market_state())

        assert state.total_bids == 0
        assert state.win_rate == 0.0
        assert state.roi == 0.0

    def test_missing_metrics_default_to_zero(self):
        """If CloudWatch returns empty Values, metrics default to 0."""
        mock_cw = MagicMock()
        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "total_bids", "Values": []},
                {"Id": "wins", "Values": []},
                {"Id": "avg_price_paid", "Values": []},
                {"Id": "avg_shaded_price", "Values": []},
                {"Id": "total_revenue", "Values": []},
                {"Id": "total_cost", "Values": []},
            ]
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=MagicMock(),
            cloudwatch_client=mock_cw,
        )

        state = asyncio.run(agent.get_market_state())

        assert state.total_bids == 0
        assert state.win_rate == 0.0


# ---------------------------------------------------------------------------
# Tests: confidence calculation
# ---------------------------------------------------------------------------


class TestConfidence:
    """Test that confidence is derived from sample size."""

    def test_confidence_proportional_to_samples(self):
        """Confidence should increase with more samples, capped at 0.95."""
        agent = _make_agent()
        params = _make_current_params()

        state_low = _make_market_state(total_bids=1000, win_rate=0.20, roi=0.5)
        state_high = _make_market_state(total_bids=9500, win_rate=0.20, roi=0.5)

        updates_low = agent.compute_adjustment(state_low, params)
        updates_high = agent.compute_adjustment(state_high, params)

        shade_low = next(u for u in updates_low if u.parameter_name == "shade_factor")
        shade_high = next(u for u in updates_high if u.parameter_name == "shade_factor")

        assert shade_low.confidence < shade_high.confidence
        assert shade_high.confidence <= 0.95

    def test_confidence_capped_at_095(self):
        """Confidence never exceeds 0.95 regardless of sample size."""
        agent = _make_agent()
        params = _make_current_params()

        state = _make_market_state(total_bids=100000, win_rate=0.20, roi=0.5)
        updates = agent.compute_adjustment(state, params)
        shade = next(u for u in updates if u.parameter_name == "shade_factor")

        assert shade.confidence == 0.95
