"""Tests for shared.yield_exploration -- bounded epsilon-greedy exploration
used to break the Yield Optimizer's cold-start deadlock (a model that always
predicts "no change" never emits a mutation, so it can never generate the
outcome data needed to learn anything else).

Property-tested per this project's PBT conventions (business bounds,
determinism under a seeded rng, disclosure via the returned `explored` flag).

Replaces the pre-split test_exploration.py. The API is single-scalar now
because the floor and margin models live in separate containers; see
TestIndependentPerModelCoinFlip below for the one behavioral guarantee that
genuinely changed.
"""

from __future__ import annotations

import os
import random
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.yield_exploration import (
    FLOOR_MULTIPLIER_MAX,
    FLOOR_MULTIPLIER_MIN,
    MARGIN_VALUE_MAX,
    MARGIN_VALUE_MIN,
    apply_exploration,
    parse_explore_override,
    resolve_effective_epsilon,
)

FLOOR_KW = dict(bound=0.05, clamp_min=FLOOR_MULTIPLIER_MIN, clamp_max=FLOOR_MULTIPLIER_MAX)
MARGIN_KW = dict(bound=0.02, clamp_min=MARGIN_VALUE_MIN, clamp_max=MARGIN_VALUE_MAX)


class TestEpsilonGating:
    def test_epsilon_zero_never_explores(self):
        rng = random.Random(1)
        for _ in range(200):
            value, explored = apply_exploration(1.0, rng=rng, epsilon=0.0, **FLOOR_KW)
            assert explored is False
            assert value == 1.0

    def test_negative_epsilon_never_explores(self):
        rng = random.Random(1)
        _, explored = apply_exploration(1.0, rng=rng, epsilon=-0.5, **FLOOR_KW)
        assert explored is False

    def test_epsilon_one_always_explores(self):
        rng = random.Random(1)
        for _ in range(200):
            _, explored = apply_exploration(1.0, rng=rng, epsilon=1.0, **FLOOR_KW)
            assert explored is True

    def test_partial_epsilon_produces_a_mix(self):
        """A real coin flip each call -- not all-or-nothing across a run."""
        rng = random.Random(42)
        outcomes = [
            apply_exploration(1.0, rng=rng, epsilon=0.5, **FLOOR_KW)[1]
            for _ in range(500)
        ]
        assert True in outcomes
        assert False in outcomes


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Same seeded rng state produces the same decision -- needed so a
        load test run is reproducible (matches this project's existing
        fixed-seed convention for load-test scenario generation)."""
        a = apply_exploration(1.0, rng=random.Random(7), epsilon=0.5, **FLOOR_KW)
        b = apply_exploration(1.0, rng=random.Random(7), epsilon=0.5, **FLOOR_KW)
        assert a == b


class TestBoundsAndClamping:
    @given(
        value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        bound=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def test_explored_floor_always_within_hard_clamps(self, value, bound, seed):
        """Property: regardless of the base prediction or configured bound, an
        explored output never exceeds the hard safety clamps -- the guard
        against a pathological base-prediction + oversized-bound combination
        producing an absurd floor multiplier."""
        result, explored = apply_exploration(
            value, rng=random.Random(seed), epsilon=1.0,
            bound=bound, clamp_min=FLOOR_MULTIPLIER_MIN, clamp_max=FLOOR_MULTIPLIER_MAX,
        )
        assert explored is True
        assert FLOOR_MULTIPLIER_MIN <= result <= FLOOR_MULTIPLIER_MAX

    @given(
        value=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        bound=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def test_explored_margin_always_within_hard_clamps(self, value, bound, seed):
        result, explored = apply_exploration(
            value, rng=random.Random(seed), epsilon=1.0,
            bound=bound, clamp_min=MARGIN_VALUE_MIN, clamp_max=MARGIN_VALUE_MAX,
        )
        assert explored is True
        assert MARGIN_VALUE_MIN <= result <= MARGIN_VALUE_MAX

    def test_unexplored_output_is_returned_verbatim_even_out_of_normal_range(self):
        """When not exploring, the model's own prediction passes through
        completely unchanged -- exploration must never silently modify a real
        model recommendation."""
        value, explored = apply_exploration(-100.0, rng=random.Random(1), epsilon=0.0, **FLOOR_KW)
        assert explored is False
        assert value == -100.0


class TestIndependentPerModelCoinFlip:
    """Pins the one guarantee that CHANGED at the container split.

    Before the split, one container held both outputs and perturbed them from a
    single coin flip, so floor and margin always explored together. Two
    independent containers handle two independent requests with two independent
    RNGs, so a shared flip is structurally impossible -- each model decides on
    its own now. That matches BR-5 (ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN are
    independent atomic mutations, never gated on each other); the old
    "same decision, so same flip" coupling was the weaker justification.

    Consequence this test documents: for one deal you may now see an explored
    floor alongside an unexplored margin. Both remain individually disclosed via
    their own response's ":explore" model_version suffix, so no training row is
    ever mislabeled -- only the correlation between the two is gone.
    """

    def test_two_models_can_reach_different_decisions_for_the_same_deal(self):
        floor_rng = random.Random(1)
        margin_rng = random.Random(2)
        disagreed = False
        for _ in range(500):
            _, floor_explored = apply_exploration(1.0, rng=floor_rng, epsilon=0.5, **FLOOR_KW)
            _, margin_explored = apply_exploration(0.0, rng=margin_rng, epsilon=0.5, **MARGIN_KW)
            if floor_explored != margin_explored:
                disagreed = True
                break
        assert disagreed, (
            "independent RNGs must be able to disagree; if they never do, the "
            "two containers are somehow sharing exploration state"
        )

    def test_exploration_actually_moves_the_value_off_its_no_op(self):
        """An explored floor must differ from 1.0 (and an explored margin from
        0.0), otherwise exploration could not break the cold-start deadlock --
        the no-op check in each container would drop it before it ever became a
        mutation."""
        rng = random.Random(3)
        for _ in range(100):
            value, explored = apply_exploration(1.0, rng=rng, epsilon=1.0, **FLOOR_KW)
            if explored:
                assert value != 1.0


class TestResolveEffectiveEpsilon:
    """The load-test/demo-only gate. Exploration must never arm itself on
    traffic the container cannot identify as load-test or demo-originated,
    regardless of how the epsilon env var is set."""

    def test_plain_production_traffic_never_explores(self):
        assert resolve_effective_epsilon(False, None, 0.1) == 0.0

    def test_load_test_traffic_arms_epsilon(self):
        assert resolve_effective_epsilon(True, None, 0.1) == 0.1

    def test_explicit_opt_in_arms_epsilon_without_a_load_test(self):
        assert resolve_effective_epsilon(False, True, 0.1) == 0.1

    def test_explicit_opt_out_beats_the_load_test_signal(self):
        """Lets a user who has since trained a real model turn exploration off
        and observe that model's unperturbed prediction."""
        assert resolve_effective_epsilon(True, False, 0.1) == 0.0

    def test_configured_epsilon_of_zero_stays_zero_even_when_armed(self):
        assert resolve_effective_epsilon(True, True, 0.0) == 0.0


class TestParseExploreOverride:
    def test_real_bools_pass_through(self):
        assert parse_explore_override({"explore": True}) is True
        assert parse_explore_override({"explore": False}) is False

    def test_absent_or_empty_is_no_override(self):
        assert parse_explore_override(None) is None
        assert parse_explore_override({}) is None
        assert parse_explore_override({"explore": None}) is None

    def test_non_bool_is_never_coerced(self):
        """A stray string must not silently arm exploration on traffic the
        caller never asked to perturb."""
        for bogus in ("true", "yes", 1, 0, "", [], {}):
            assert parse_explore_override({"explore": bogus}) is None
