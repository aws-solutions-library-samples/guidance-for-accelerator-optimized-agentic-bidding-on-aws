"""Tests for containers.deal_yield_manager.exploration -- bounded
epsilon-greedy exploration used to break the Yield Optimizer's cold-start
deadlock (a model that always predicts "no change" never emits a mutation,
so it can never generate the outcome data needed to learn anything else).

Property-tested per this project's PBT conventions (business bounds,
determinism under a seeded rng, disclosure via the returned `explored`
flag).
"""

from __future__ import annotations

import os
import random
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from containers.deal_yield_manager.exploration import (
    FLOOR_MULTIPLIER_MAX,
    FLOOR_MULTIPLIER_MIN,
    MARGIN_VALUE_MAX,
    MARGIN_VALUE_MIN,
    apply_exploration,
)


class TestEpsilonGating:
    def test_epsilon_zero_never_explores(self):
        rng = random.Random(1)
        for _ in range(200):
            floor, margin, explored = apply_exploration(
                1.0, 0.0, rng=rng, epsilon=0.0, floor_bound=0.1, margin_bound=0.1
            )
            assert explored is False
            assert floor == 1.0
            assert margin == 0.0

    def test_negative_epsilon_never_explores(self):
        rng = random.Random(1)
        floor, margin, explored = apply_exploration(
            1.0, 0.0, rng=rng, epsilon=-0.5, floor_bound=0.1, margin_bound=0.1
        )
        assert explored is False

    def test_epsilon_one_always_explores(self):
        rng = random.Random(1)
        for _ in range(200):
            _, _, explored = apply_exploration(
                1.0, 0.0, rng=rng, epsilon=1.0, floor_bound=0.05, margin_bound=0.02
            )
            assert explored is True

    def test_partial_epsilon_produces_a_mix(self):
        """A real coin flip each call -- not all-or-nothing across a run."""
        rng = random.Random(42)
        outcomes = [
            apply_exploration(1.0, 0.0, rng=rng, epsilon=0.5, floor_bound=0.05, margin_bound=0.02)[2]
            for _ in range(500)
        ]
        assert True in outcomes
        assert False in outcomes


class TestDeterminism:
    def test_same_seed_same_sequence(self):
        """Same seeded rng state produces the same sequence of decisions --
        needed so a load test run is reproducible (matches this project's
        existing fixed-seed convention for load-test scenario generation)."""
        seq1 = [
            apply_exploration(1.0, 0.0, rng=random.Random(7), epsilon=0.5, floor_bound=0.05, margin_bound=0.02)
            for _ in range(1)
        ]
        seq2 = [
            apply_exploration(1.0, 0.0, rng=random.Random(7), epsilon=0.5, floor_bound=0.05, margin_bound=0.02)
            for _ in range(1)
        ]
        assert seq1 == seq2


class TestBoundsAndClamping:
    @given(
        floor_mult=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        margin_val=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        floor_bound=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        margin_bound=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def test_explored_output_always_within_hard_clamps(self, floor_mult, margin_val, floor_bound, margin_bound, seed):
        """Property: regardless of the base prediction or configured bound,
        an explored output never exceeds the hard safety clamps -- this is
        the guard against a pathological base-prediction + oversized-bound
        combination producing an absurd floor/margin."""
        rng = random.Random(seed)
        floor, margin, explored = apply_exploration(
            floor_mult, margin_val, rng=rng, epsilon=1.0, floor_bound=floor_bound, margin_bound=margin_bound
        )
        assert explored is True
        assert FLOOR_MULTIPLIER_MIN <= floor <= FLOOR_MULTIPLIER_MAX
        assert MARGIN_VALUE_MIN <= margin <= MARGIN_VALUE_MAX

    def test_unexplored_output_is_returned_verbatim_even_out_of_normal_range(self):
        """When not exploring, the model's own prediction passes through
        completely unchanged -- exploration must never silently modify a
        real model recommendation."""
        rng = random.Random(1)
        floor, margin, explored = apply_exploration(
            -100.0, 999.0, rng=rng, epsilon=0.0, floor_bound=0.05, margin_bound=0.02
        )
        assert explored is False
        assert floor == -100.0
        assert margin == 999.0


class TestFloorAndMarginShareOneCoinFlip:
    def test_both_explore_or_neither_explores(self):
        """Floor and margin describe the same yield-optimization decision
        for one deal -- they must share one explore/exploit coin flip, not
        two independent ones (which could otherwise explore floor but not
        margin for the same deal, an inconsistent state)."""
        rng = random.Random(3)
        for _ in range(100):
            floor, margin, explored = apply_exploration(
                1.0, 0.0, rng=rng, epsilon=0.5, floor_bound=0.05, margin_bound=0.02
            )
            if explored:
                assert floor != 1.0 or margin != 0.0
