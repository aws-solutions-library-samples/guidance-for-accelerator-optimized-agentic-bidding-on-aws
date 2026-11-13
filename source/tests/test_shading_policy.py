"""Tests for the bid shading policy — Item 3.1, 3.2, 3.4.

The two properties the policy exists to guarantee are monotonicity in expected value
and boundedness by the auction's floor and the buyer's original bid. Both are asserted
as property-based tests over generated inputs, per task 3.4, because the whole point of
choosing a small parametric form (decision D2) was that these hold by algebra rather
than by the examples someone happened to write down.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from shared.shading_policy import (  # noqa: E402
    BASE_BOUNDS,
    CURVATURE_BOUNDS,
    GENESIS_POLICY,
    POLICY_VERSION,
    SLOPE_BOUNDS,
    PolicyParameterError,
    ShadingPolicy,
    clamp,
    policy_search_space,
)


# ---------------------------------------------------------------------------
# Generators — constrained to the domain, per PBT-07
# ---------------------------------------------------------------------------


def policies() -> st.SearchStrategy[ShadingPolicy]:
    """Valid policies: the bounds are what make the properties true."""
    return st.builds(
        ShadingPolicy,
        base=st.floats(*BASE_BOUNDS, allow_nan=False, allow_infinity=False),
        slope=st.floats(*SLOPE_BOUNDS, allow_nan=False, allow_infinity=False),
        curvature=st.floats(*CURVATURE_BOUNDS, allow_nan=False, allow_infinity=False),
    )


def evs() -> st.SearchStrategy[float]:
    """Expected values in a realistic CPM range, plus zero."""
    return st.floats(
        min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False
    )


def prices() -> st.SearchStrategy[float]:
    return st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    )


# ---------------------------------------------------------------------------
# 3.4 — monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    @settings(max_examples=300, deadline=None)
    @given(policies(), evs(), evs(), prices(), prices())
    def test_higher_ev_never_prices_lower(self, policy, ev_a, ev_b, floor, bid):
        """ev_1 >= ev_2 implies price_1 >= price_2, for a fixed floor and bid."""
        lo, hi = sorted((ev_a, ev_b))
        p_lo = policy.price(lo, floor=floor, original_bid=bid)
        p_hi = policy.price(hi, floor=floor, original_bid=bid)
        assert p_hi >= p_lo - 1e-9

    @settings(max_examples=300, deadline=None)
    @given(policies(), evs(), evs())
    def test_raw_price_is_monotone_before_clamping(self, policy, ev_a, ev_b):
        """The clamp preserves monotonicity, but the form has it on its own."""
        lo, hi = sorted((ev_a, ev_b))
        assert policy.raw_price(hi) >= policy.raw_price(lo) - 1e-9

    @settings(max_examples=200, deadline=None)
    @given(policies())
    def test_zero_ev_prices_at_base(self, policy):
        assert policy.raw_price(0.0) == pytest.approx(policy.base)


# ---------------------------------------------------------------------------
# 3.4 — boundedness
# ---------------------------------------------------------------------------


class TestBoundedness:
    @settings(max_examples=300, deadline=None)
    @given(policies(), evs(), prices(), prices())
    def test_price_is_always_within_floor_and_original_bid(
        self, policy, ev, floor, bid
    ):
        price = policy.price(ev, floor=floor, original_bid=bid)
        if floor <= bid:
            assert floor - 1e-9 <= price <= bid + 1e-9
        else:
            # No value exists in the interval. The floor wins, and the returned price
            # exceeding the bid is the signal the caller acts on.
            assert price == pytest.approx(floor)

    @settings(max_examples=300, deadline=None)
    @given(policies(), evs(), prices())
    def test_never_prices_above_the_buyers_own_bid(self, policy, ev, bid):
        """Shading bids less than the buyer offered. Never more."""
        assert policy.price(ev, floor=0.0, original_bid=bid) <= bid + 1e-9

    @settings(max_examples=200, deadline=None)
    @given(prices(), prices(), prices())
    def test_clamp_is_idempotent(self, price, floor, bid):
        once = clamp(price, floor=floor, original_bid=bid)
        assert clamp(once, floor=floor, original_bid=bid) == pytest.approx(once)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"slope": -0.01},
            {"slope": SLOPE_BOUNDS[1] + 0.01},
            {"base": -0.01},
            {"base": BASE_BOUNDS[1] + 1},
            {"curvature": 0.0},
            {"curvature": CURVATURE_BOUNDS[1] + 0.01},
        ],
    )
    def test_out_of_bounds_is_refused_not_clamped(self, kwargs):
        """Clamping a bad parameter hides which caller produced it."""
        with pytest.raises(PolicyParameterError):
            ShadingPolicy(**kwargs)

    def test_nan_is_refused(self):
        with pytest.raises(PolicyParameterError):
            ShadingPolicy(slope=float("nan"))

    def test_negative_ev_is_refused(self):
        with pytest.raises(PolicyParameterError, match="ev must be non-negative"):
            GENESIS_POLICY.raw_price(-1.0)

    def test_negative_floor_is_refused(self):
        with pytest.raises(PolicyParameterError, match="floor"):
            GENESIS_POLICY.price(1.0, floor=-1.0, original_bid=5.0)

    def test_search_space_matches_the_enforced_bounds(self):
        """One source for the bounds, or a search drifts from what is validated."""
        space = policy_search_space()
        assert space["base"] == BASE_BOUNDS
        assert space["slope"] == SLOPE_BOUNDS
        assert space["curvature"] == CURVATURE_BOUNDS

    @settings(max_examples=200, deadline=None)
    @given(
        st.floats(*BASE_BOUNDS, allow_nan=False),
        st.floats(*SLOPE_BOUNDS, allow_nan=False),
        st.floats(*CURVATURE_BOUNDS, allow_nan=False),
    )
    def test_every_point_in_the_search_space_is_constructible(self, b, s, c):
        """A search must not be able to propose a parameter set that then throws."""
        ShadingPolicy(base=b, slope=s, curvature=c)


# ---------------------------------------------------------------------------
# Genesis equivalence — swapping the call site in is not a behaviour change
# ---------------------------------------------------------------------------


class TestGenesisEquivalence:
    @settings(max_examples=300, deadline=None)
    @given(
        evs(),
        prices(),
        prices(),
        st.floats(*SLOPE_BOUNDS, allow_nan=False, allow_infinity=False),
    )
    def test_reproduces_the_previous_hardcoded_arithmetic(
        self, ev, floor, bid, shade_factor
    ):
        """The old line was `max(min(original, ev * shade_factor), floor)`."""
        expected = max(min(bid, ev * shade_factor), floor)
        actual = ShadingPolicy(slope=shade_factor).price(
            ev, floor=floor, original_bid=bid
        )
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_genesis_defaults_to_the_old_shade_factor(self):
        assert GENESIS_POLICY.slope == 0.65
        assert GENESIS_POLICY.base == 0.0
        assert GENESIS_POLICY.curvature == 1.0


# ---------------------------------------------------------------------------
# Serialisation into the manifest and back
# ---------------------------------------------------------------------------


class TestSerialisation:
    @settings(max_examples=200, deadline=None)
    @given(policies())
    def test_round_trip_through_a_mapping(self, policy):
        assert ShadingPolicy.from_mapping(policy.as_dict()) == policy

    def test_as_dict_carries_the_policy_version(self):
        assert GENESIS_POLICY.as_dict()["policy_version"] == POLICY_VERSION

    def test_absent_parameters_yield_the_default(self):
        """A model published before policy parameters existed still serves."""
        assert ShadingPolicy.from_mapping(None) == GENESIS_POLICY
        assert ShadingPolicy.from_mapping({}) == GENESIS_POLICY

    def test_a_partial_mapping_overrides_only_what_it_names(self):
        p = ShadingPolicy.from_mapping(
            {"curvature": 0.8}, default=ShadingPolicy(slope=0.5)
        )
        assert p.slope == 0.5
        assert p.curvature == 0.8

    def test_an_unknown_policy_version_is_refused(self):
        """Serving the genesis policy instead would price against the wrong params."""
        with pytest.raises(PolicyParameterError, match="policy_version"):
            ShadingPolicy.from_mapping({"policy_version": POLICY_VERSION + 1})

    def test_an_out_of_bounds_stored_parameter_is_refused(self):
        with pytest.raises(PolicyParameterError):
            ShadingPolicy.from_mapping({"slope": 99.0})


# ---------------------------------------------------------------------------
# 3.1 — the shader calls the policy and nothing else prices a bid
# ---------------------------------------------------------------------------


class TestSingleCallSite:
    def test_shader_uses_the_policy(self):
        source = (_SOURCE / "containers" / "dlrm_bid_shader" / "app.py").read_text()
        assert "from shared.shading_policy import ShadingPolicy" in source
        assert "policy.price(" in source

    def test_shader_no_longer_computes_a_price_inline(self):
        source = (_SOURCE / "containers" / "dlrm_bid_shader" / "app.py").read_text()
        assert "min(original_price, ev * shade_factor)" not in source
        assert "shaded = max(shaded, floor)" not in source
