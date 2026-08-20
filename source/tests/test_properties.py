"""Property-based tests (Hypothesis) for the closed-loop learning system.

Asserts universal properties that must hold across all valid inputs:
- Reward function always returns values in [-1.0, 1.0]
- Bid shading agent adjustments stay within parameter bounds
- A/B evaluator p-value is always in [0, 1]
- Canary deployer traffic always totals 100%

**Validates: Requirements 8.1, 13.3**
"""

from __future__ import annotations

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from shared.feedback_models import BidOutcomeRecord
from training.reward import compute_rl_reward
from agents.adaptive_bidding.agent import (
    AdaptiveBiddingStrategyAgent,
    MarketState,
    ParameterUpdate,
)
from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig
from deployment.canary_deployer import DeploymentState


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for device_type (used in BidOutcomeRecord)
device_types = st.sampled_from(["mobile", "desktop", "tablet"])

# Strategy for model_type
model_types = st.sampled_from(["dlrm_bid_shader", "ncf_deal_manager", "widedeep_segment_activator"])

# Strategy for UUID strings
uuid_strategy = st.builds(lambda: str(uuid.uuid4()))


def _bid_outcome_strategy():
    """Strategy that produces valid BidOutcomeRecords respecting all constraints.

    Key constraints:
    - original_price >= shaded_price >= bid_floor >= 0
    - Monotonic outcome signals: conversion -> click -> impression -> won
    - price_paid only when won == true
    - conversion_value only when conversion == true
    """

    @st.composite
    def build_record(draw):
        # Prices: bid_floor <= shaded_price <= original_price, all >= 0
        bid_floor = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        shaded_price = draw(st.floats(min_value=bid_floor, max_value=bid_floor + 100.0, allow_nan=False, allow_infinity=False))
        original_price = draw(st.floats(min_value=shaded_price, max_value=shaded_price + 100.0, allow_nan=False, allow_infinity=False))

        # Monotonic outcome signals: conversion -> click -> impression -> won
        # Pick a "level" to determine which signals are true
        # level 0: nothing won; level 1: won only; level 2: won+impression;
        # level 3: won+impression+click; level 4: won+impression+click+conversion
        level = draw(st.integers(min_value=0, max_value=4))
        won = level >= 1
        impression = level >= 2
        click = level >= 3
        conversion = level >= 4

        # price_paid only when won
        price_paid = draw(st.floats(min_value=0.01, max_value=original_price + 50.0, allow_nan=False, allow_infinity=False)) if won else None

        # conversion_value only when conversion
        conversion_value = draw(st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)) if conversion else None

        # Other fields
        request_id = draw(uuid_strategy)
        event_timestamp = draw(st.integers(min_value=1_000_000_000_000, max_value=2_000_000_000_000))
        model_type = draw(model_types)
        device_type = draw(device_types)
        hour_of_day = draw(st.integers(min_value=0, max_value=23))
        day_of_week = draw(st.integers(min_value=0, max_value=6))

        shade_factor_used = draw(st.floats(min_value=0.3, max_value=0.95, allow_nan=False, allow_infinity=False))
        conversion_value_estimate = draw(st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False))

        return BidOutcomeRecord(
            request_id=request_id,
            event_timestamp=event_timestamp,
            model_type=model_type,
            model_version="v1.0.0",
            source="live",
            intent="BID_SHADE",
            original_price=original_price,
            shaded_price=shaded_price,
            bid_floor=bid_floor,
            price_paid=price_paid,
            won=won,
            impression=impression,
            click=click,
            conversion=conversion,
            conversion_value=conversion_value,
            user_id_hash="abc123def456",
            site_domain_hash="site_hash_001",
            device_type=device_type,
            geo_country="US",
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            has_video=False,
            iab_categories=["IAB1"],
            shade_factor_used=shade_factor_used,
            conversion_value_estimate=conversion_value_estimate,
            partition_date="2025-06-10",
            partition_hour=hour_of_day,
        )

    return build_record()


def _market_state_strategy():
    """Strategy that produces valid MarketState instances."""

    @st.composite
    def build_state(draw):
        total_bids = draw(st.integers(min_value=1, max_value=100_000))
        wins = draw(st.integers(min_value=0, max_value=total_bids))
        win_rate = wins / total_bids

        avg_price_paid = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
        avg_shaded_price = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
        total_revenue = draw(st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False))
        total_cost = draw(st.floats(min_value=0.01, max_value=1_000_000.0, allow_nan=False, allow_infinity=False))
        roi = (total_revenue - total_cost) / total_cost
        competitive_pressure = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

        return MarketState(
            window_start=1718000000.0,
            window_end=1718000300.0,
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

    return build_state()


# ---------------------------------------------------------------------------
# A mock ParameterState for the bid shading agent tests
# ---------------------------------------------------------------------------

class _MockParameterState:
    """Minimal parameter state object matching what compute_adjustment expects."""

    def __init__(self, current_value: float, version: int = 1):
        self.current_value = current_value
        self.version = version


# ---------------------------------------------------------------------------
# Property 1: Reward bounds
# ---------------------------------------------------------------------------


class TestRewardBoundsProperty:
    """For any valid BidOutcomeRecord, compute_rl_reward returns a value in [-1.0, 1.0]."""

    @given(record=_bid_outcome_strategy())
    @settings(max_examples=200)
    def test_reward_always_in_bounds(self, record: BidOutcomeRecord):
        """**Validates: Requirements 13.3**

        Property: For all valid BidOutcomeRecords, the reward function
        returns a value in the closed interval [-1.0, 1.0].
        """
        reward = compute_rl_reward(record)
        assert -1.0 <= reward <= 1.0, (
            f"Reward {reward} out of bounds for record with "
            f"won={record.won}, original_price={record.original_price}, "
            f"shaded_price={record.shaded_price}"
        )


# ---------------------------------------------------------------------------
# Property 2: Parameter bounds after adjustment
# ---------------------------------------------------------------------------


class TestParameterBoundsProperty:
    """After compute_adjustment, parameters stay within global bounds and +-5% max change."""

    @given(
        state=_market_state_strategy(),
        shade_factor=st.floats(min_value=0.3, max_value=0.95, allow_nan=False, allow_infinity=False),
        conversion_value=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_shade_factor_within_bounds(
        self, state: MarketState, shade_factor: float, conversion_value: float
    ):
        """**Validates: Requirements 8.1**

        Property: After compute_adjustment, the shade_factor new_value
        is always within [0.3, 0.95].
        """
        current_params = {
            "shade_factor": _MockParameterState(shade_factor),
            "conversion_value": _MockParameterState(conversion_value),
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=None,
            cloudwatch_client=None,
            config={"target_win_rate": 0.35, "win_rate_tolerance": 0.05,
                    "max_adjustment": 0.05, "learning_rate": 0.1,
                    "min_samples": 1000, "model_type": "dlrm_bid_shader"},
        )

        updates = agent.compute_adjustment(state, current_params)

        shade_update = next(u for u in updates if u.parameter_name == "shade_factor")
        assert 0.3 <= shade_update.new_value <= 0.95, (
            f"shade_factor {shade_update.new_value} out of bounds [0.3, 0.95] "
            f"(old={shade_update.old_value}, win_rate={state.win_rate})"
        )

    @given(
        state=_market_state_strategy(),
        shade_factor=st.floats(min_value=0.3, max_value=0.95, allow_nan=False, allow_infinity=False),
        conversion_value=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_conversion_value_within_bounds(
        self, state: MarketState, shade_factor: float, conversion_value: float
    ):
        """**Validates: Requirements 8.1**

        Property: After compute_adjustment, the conversion_value new_value
        is always within [1.0, 50.0].
        """
        current_params = {
            "shade_factor": _MockParameterState(shade_factor),
            "conversion_value": _MockParameterState(conversion_value),
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=None,
            cloudwatch_client=None,
            config={"target_win_rate": 0.35, "win_rate_tolerance": 0.05,
                    "max_adjustment": 0.05, "learning_rate": 0.1,
                    "min_samples": 1000, "model_type": "dlrm_bid_shader"},
        )

        updates = agent.compute_adjustment(state, current_params)

        cv_update = next(u for u in updates if u.parameter_name == "conversion_value")
        assert 1.0 <= cv_update.new_value <= 50.0, (
            f"conversion_value {cv_update.new_value} out of bounds [1.0, 50.0] "
            f"(old={cv_update.old_value}, roi={state.roi}, win_rate={state.win_rate})"
        )

    @given(
        state=_market_state_strategy(),
        shade_factor=st.floats(min_value=0.3, max_value=0.95, allow_nan=False, allow_infinity=False),
        conversion_value=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_adjustment_magnitude_bounded(
        self, state: MarketState, shade_factor: float, conversion_value: float
    ):
        """**Validates: Requirements 8.1**

        Property: The magnitude of any single adjustment never exceeds 5%
        (max_adjustment config = 0.05 for shade_factor, 5% of value for conversion_value).
        """
        current_params = {
            "shade_factor": _MockParameterState(shade_factor),
            "conversion_value": _MockParameterState(conversion_value),
        }

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=None,
            cloudwatch_client=None,
            config={"target_win_rate": 0.35, "win_rate_tolerance": 0.05,
                    "max_adjustment": 0.05, "learning_rate": 0.1,
                    "min_samples": 1000, "model_type": "dlrm_bid_shader"},
        )

        updates = agent.compute_adjustment(state, current_params)

        for update in updates:
            delta = abs(update.new_value - update.old_value)
            if update.parameter_name == "shade_factor":
                # shade_factor max adjustment is 0.05 absolute
                assert delta <= 0.05 + 1e-9, (
                    f"shade_factor adjustment {delta} exceeds max 0.05"
                )
            elif update.parameter_name == "conversion_value":
                # conversion_value max adjustment is 5% of current value
                max_delta = 0.05 * conversion_value
                assert delta <= max_delta + 1e-9, (
                    f"conversion_value adjustment {delta} exceeds max "
                    f"{max_delta} (5% of {conversion_value})"
                )


# ---------------------------------------------------------------------------
# Property 3: A/B p-value in [0, 1]
# ---------------------------------------------------------------------------


class TestABPValueProperty:
    """For any two lists of floats, p_value is always in [0, 1]."""

    @given(
        control=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=200,
        ),
        treatment=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=200,
        ),
    )
    @settings(max_examples=200)
    def test_p_value_always_in_unit_interval(
        self, control: list[float], treatment: list[float]
    ):
        """**Validates: Requirements 13.3**

        Property: For any two lists of floats (control, treatment) with at
        least 2 elements each, the computed p_value is always in [0, 1].
        """
        config = ABTestConfig(
            model_type="dlrm_bid_shader",
            control_version="v1.0",
            treatment_version="v1.1",
            traffic_percentage=10.0,
            min_samples=2,  # Low threshold so we test the statistical logic
            max_duration_hours=2.0,
            significance_level=0.05,
            primary_metric="revenue_per_bid",
        )

        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(control, treatment)

        assert 0.0 <= result.p_value <= 1.0, (
            f"p_value {result.p_value} out of [0, 1] for "
            f"control (n={len(control)}), treatment (n={len(treatment)})"
        )


# ---------------------------------------------------------------------------
# Property 4: Canary total traffic == 100%
# ---------------------------------------------------------------------------


class TestCanaryTrafficProperty:
    """For any DeploymentState with a canary, control + canary traffic == 100%."""

    @given(
        canary_pct=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_deployment_state_traffic_totals_100(self, canary_pct: float):
        """**Validates: Requirements 13.3**

        Property: For any DeploymentState, canary_traffic_pct + control_traffic_pct
        always equals 100.0.
        """
        state = DeploymentState(
            model_name="test_model",
            current_version=1,
            canary_version=2,
            canary_traffic_pct=canary_pct,
            status="canary_active",
        )

        total = state.canary_traffic_pct + state.control_traffic_pct
        assert abs(total - 100.0) < 1e-9, (
            f"Total traffic {total} != 100.0 "
            f"(canary={state.canary_traffic_pct}, control={state.control_traffic_pct})"
        )

    @given(
        adjustments=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=200)
    def test_traffic_total_after_adjustments(self, adjustments: list[float]):
        """**Validates: Requirements 13.3**

        Property: After any sequence of valid traffic adjustments (pct in [0, 100]),
        total traffic (canary + control) remains exactly 100%.
        """
        state = DeploymentState(
            model_name="test_model",
            current_version=1,
            canary_version=2,
            canary_traffic_pct=5.0,
            status="canary_active",
        )

        for pct in adjustments:
            state.canary_traffic_pct = pct
            total = state.canary_traffic_pct + state.control_traffic_pct
            assert abs(total - 100.0) < 1e-9, (
                f"After setting canary to {pct}%, total traffic {total} != 100.0"
            )
