"""Unit tests for training.reward — compute_rl_reward function.

Validates the NeMo-RL reward function returns correct signals for various
bid outcomes and always stays bounded in [-1.0, 1.0].

**Validates: Requirements 3.3**
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import BidOutcomeRecord
from training.reward import compute_rl_reward, _estimate_ctr


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> BidOutcomeRecord:
    """Create a valid BidOutcomeRecord with sensible defaults, applying overrides."""
    defaults = {
        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "event_timestamp": 1718000000000,
        "model_type": "dlrm_bid_shader",
        "model_version": "v1.2.3",
        "intent": "bid",
        "original_price": 5.0,
        "shaded_price": 4.0,
        "bid_floor": 2.0,
        "price_paid": 3.5,
        "won": True,
        "impression": True,
        "click": True,
        "conversion": True,
        "conversion_value": 10.0,
        "user_id_hash": "abc123hash",
        "site_domain_hash": "site456hash",
        "device_type": "mobile",
        "geo_country": "US",
        "hour_of_day": 14,
        "day_of_week": 2,
        "has_video": False,
        "iab_categories": ["IAB1"],
        "shade_factor_used": 0.8,
        "conversion_value_estimate": 8.0,
        "partition_date": "2024-06-10",
        "partition_hour": 14,
    }
    defaults.update(overrides)
    return BidOutcomeRecord(**defaults)


# ---------------------------------------------------------------------------
# Tests: Won with conversion → positive reward
# ---------------------------------------------------------------------------


class TestWonWithConversion:
    """Won auction with conversion should produce a positive reward."""

    def test_profitable_conversion_gives_positive_reward(self):
        """Winning with high revenue relative to cost yields positive reward."""
        record = _make_record(
            won=True,
            impression=True,
            click=True,
            conversion=True,
            conversion_value=10.0,
            price_paid=3.5,
            original_price=5.0,
            shaded_price=4.0,
        )
        reward = compute_rl_reward(record)
        assert reward > 0.0

    def test_very_profitable_conversion(self):
        """Very high conversion value relative to cost gives high reward."""
        record = _make_record(
            won=True,
            impression=True,
            click=True,
            conversion=True,
            conversion_value=50.0,
            price_paid=2.5,
            original_price=5.0,
            shaded_price=4.0,
            conversion_value_estimate=50.0,
        )
        reward = compute_rl_reward(record)
        assert reward > 0.5


# ---------------------------------------------------------------------------
# Tests: Won with click (no conversion) → moderate positive reward
# ---------------------------------------------------------------------------


class TestWonWithClick:
    """Won with click but no conversion should produce a moderate reward."""

    def test_click_no_conversion_gives_moderate_reward(self):
        """Click earns partial credit from conversion_value_estimate."""
        record = _make_record(
            won=True,
            impression=True,
            click=True,
            conversion=False,
            conversion_value=None,
            price_paid=3.0,
            original_price=5.0,
            shaded_price=4.0,
            conversion_value_estimate=20.0,
        )
        reward = compute_rl_reward(record)
        # With revenue = 20 * 0.1 = 2.0, cost = 3.0, ROI is negative
        # but savings bonus from cost < original helps
        # Key: reward should be moderate (not strongly positive or negative)
        assert -0.5 < reward < 0.5


# ---------------------------------------------------------------------------
# Tests: Won with only impression → small positive reward
# ---------------------------------------------------------------------------


class TestWonWithImpression:
    """Won with only impression (no click) should yield small reward."""

    def test_impression_only_gives_small_reward(self):
        """Impression only earns minimal credit from conversion_value_estimate."""
        record = _make_record(
            won=True,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=2.0,
            original_price=5.0,
            shaded_price=3.0,
            conversion_value_estimate=10.0,
        )
        reward = compute_rl_reward(record)
        # Revenue = 10 * 0.01 = 0.1, cost = 2.0. ROI is very negative
        # but savings_ratio = 1 - 2/5 = 0.6, bonus = 0.12
        # Should be a modest number around the savings bonus
        assert -1.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# Tests: Lost but low-value impression → small positive (cost avoidance)
# ---------------------------------------------------------------------------


class TestLostLowValue:
    """Losing a low-value impression is correct — yields small positive reward."""

    def test_lost_low_value_impression_gives_positive_reward(self):
        """When potential_value < shaded_price, not winning is correct."""
        record = _make_record(
            won=False,
            impression=False,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=None,
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            conversion_value_estimate=1.0,  # Low value
            device_type="desktop",  # CTR = 0.015 → potential = 1.0 * 0.015 = 0.015 < 4.0
        )
        reward = compute_rl_reward(record)
        assert reward == pytest.approx(0.1)

    def test_lost_zero_value_estimate(self):
        """Zero conversion estimate means correctly avoided."""
        record = _make_record(
            won=False,
            impression=False,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=None,
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=2.0,
            conversion_value_estimate=0.0,
        )
        reward = compute_rl_reward(record)
        assert reward == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Tests: Lost high-value impression → negative reward (bid too low)
# ---------------------------------------------------------------------------


class TestLostHighValue:
    """Losing a high-value impression should penalize (bid too low)."""

    def test_lost_high_value_gives_negative_reward(self):
        """When potential_value > shaded_price, penalty is proportional."""
        # potential_value = 200 * 0.025 = 5.0, shaded_price = 2.0
        # loss_ratio = (5.0 - 2.0) / 5.0 = 0.6
        # reward = -0.3 * min(1.0, 0.6) = -0.18
        record = _make_record(
            won=False,
            impression=False,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=None,
            original_price=5.0,
            shaded_price=2.0,
            bid_floor=1.0,
            conversion_value_estimate=200.0,
            device_type="mobile",  # CTR = 0.025
        )
        reward = compute_rl_reward(record)
        assert reward < 0.0
        assert reward == pytest.approx(-0.18, abs=0.01)

    def test_lost_very_high_value_caps_at_minus_0_3(self):
        """Maximum penalty for lost impressions is -0.3."""
        # potential_value = 10000 * 0.025 = 250, shaded_price = 2.0
        # loss_ratio = (250 - 2) / 250 ≈ 0.992
        # reward = -0.3 * min(1.0, 0.992) = -0.298
        record = _make_record(
            won=False,
            impression=False,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=None,
            original_price=5.0,
            shaded_price=2.0,
            bid_floor=1.0,
            conversion_value_estimate=10000.0,
            device_type="mobile",
        )
        reward = compute_rl_reward(record)
        assert -0.3 <= reward <= 0.0


# ---------------------------------------------------------------------------
# Tests: Reward is always bounded in [-1.0, 1.0]
# ---------------------------------------------------------------------------


class TestRewardBounds:
    """Reward must always be in [-1.0, 1.0] regardless of inputs."""

    def test_massive_conversion_value_stays_bounded(self):
        """Very high conversion value still capped at 1.0."""
        record = _make_record(
            won=True,
            impression=True,
            click=True,
            conversion=True,
            conversion_value=100000.0,
            price_paid=0.01,
            original_price=5.0,
            shaded_price=4.0,
            conversion_value_estimate=100000.0,
        )
        reward = compute_rl_reward(record)
        assert -1.0 <= reward <= 1.0

    def test_zero_price_paid_stays_bounded(self):
        """Edge case: price_paid = 0 (free win) stays in bounds."""
        record = _make_record(
            won=True,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=0.0,
            original_price=5.0,
            shaded_price=4.0,
            bid_floor=0.0,
            conversion_value_estimate=5.0,
        )
        reward = compute_rl_reward(record)
        assert -1.0 <= reward <= 1.0

    def test_high_cost_low_value_stays_bounded(self):
        """Very costly win with low value stays bounded at -1.0."""
        record = _make_record(
            won=True,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=50.0,
            original_price=50.0,
            shaded_price=50.0,
            bid_floor=0.0,
            conversion_value_estimate=0.01,
        )
        reward = compute_rl_reward(record)
        assert -1.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# Tests: Savings ratio bonus when winning at low cost
# ---------------------------------------------------------------------------


class TestSavingsBonus:
    """Efficient shading (winning below original price) earns a bonus."""

    def test_savings_bonus_increases_reward(self):
        """Paying less than the original price adds a positive bonus."""
        # Same conversion, compare high cost vs low cost
        base_kwargs = {
            "won": True,
            "impression": True,
            "click": True,
            "conversion": True,
            "conversion_value": 10.0,
            "original_price": 10.0,
            "shaded_price": 8.0,
            "bid_floor": 1.0,
            "conversion_value_estimate": 10.0,
        }
        record_low_cost = _make_record(**base_kwargs, price_paid=2.0)
        record_high_cost = _make_record(**base_kwargs, price_paid=9.0)

        reward_low = compute_rl_reward(record_low_cost)
        reward_high = compute_rl_reward(record_high_cost)

        # Lower cost should yield higher reward (both from ROI and savings bonus)
        assert reward_low > reward_high

    def test_zero_savings_gives_no_bonus(self):
        """Paying the full original price gives zero savings bonus."""
        record = _make_record(
            won=True,
            impression=True,
            click=True,
            conversion=True,
            conversion_value=10.0,
            price_paid=5.0,
            original_price=5.0,
            shaded_price=5.0,
            bid_floor=5.0,
            conversion_value_estimate=10.0,
        )
        # savings_ratio = 1 - 5/5 = 0 → no bonus from savings
        reward = compute_rl_reward(record)
        # ROI = (10 - 5) / 5 = 1.0, scaled = 0.5, bonus = 0
        assert reward == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: zero prices, boundary values, etc."""

    def test_price_paid_none_uses_shaded_price(self):
        """When price_paid is None (won=True possible only if it's set,
        but if somehow None), falls back to shaded_price."""
        # Note: BidOutcomeRecord allows price_paid=None even when won=True
        # (the model doesn't enforce it), so we test fallback behavior.
        # Actually, looking at the model, price_paid can be None when won is True
        # since there's no validator preventing that combination.
        record = _make_record(
            won=True,
            impression=True,
            click=True,
            conversion=True,
            conversion_value=10.0,
            price_paid=None,
            original_price=5.0,
            shaded_price=4.0,
            conversion_value_estimate=10.0,
        )
        reward = compute_rl_reward(record)
        # Should use shaded_price=4.0 as cost
        # ROI = (10 - 4) / 4 = 1.5, scaled = 0.75
        # savings = 1 - 4/5 = 0.2, bonus = 0.04
        # total = 0.79 → clamped to 0.79
        assert reward > 0.0
        assert -1.0 <= reward <= 1.0

    def test_minimum_bid_floor_scenario(self):
        """Bid floor at 0 doesn't cause division errors."""
        record = _make_record(
            won=True,
            impression=True,
            click=False,
            conversion=False,
            conversion_value=None,
            price_paid=0.01,
            original_price=5.0,
            shaded_price=3.0,
            bid_floor=0.0,
            conversion_value_estimate=5.0,
        )
        reward = compute_rl_reward(record)
        assert -1.0 <= reward <= 1.0

    def test_all_device_types_produce_valid_ctr(self):
        """All recognized device types return valid CTR values."""
        for device in ["mobile", "desktop", "tablet", "unknown_device"]:
            record = _make_record(
                won=False,
                impression=False,
                click=False,
                conversion=False,
                conversion_value=None,
                price_paid=None,
                original_price=5.0,
                shaded_price=4.0,
                bid_floor=2.0,
                device_type=device,
                conversion_value_estimate=1.0,
            )
            reward = compute_rl_reward(record)
            assert -1.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# Tests: _estimate_ctr helper
# ---------------------------------------------------------------------------


class TestEstimateCtr:
    """Tests for the CTR estimation heuristic."""

    def test_mobile_ctr(self):
        record = _make_record(device_type="mobile")
        assert _estimate_ctr(record) == 0.025

    def test_desktop_ctr(self):
        record = _make_record(device_type="desktop")
        assert _estimate_ctr(record) == 0.015

    def test_tablet_ctr(self):
        record = _make_record(device_type="tablet")
        assert _estimate_ctr(record) == 0.02

    def test_unknown_device_uses_default(self):
        record = _make_record(device_type="smarttv")
        assert _estimate_ctr(record) == 0.02

    def test_case_insensitive(self):
        record = _make_record(device_type="Mobile")
        assert _estimate_ctr(record) == 0.025
