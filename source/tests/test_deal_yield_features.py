"""Tests for shared.yield_features -- pure feature
engineering for the Yield Optimizer (deal floor/margin) model.

Property-based tests (PBT Partial mode -- pure functions, per this
project's Extension Configuration) validate the invariants declared in
aidlc-docs/construction/deal-yield-model/nfr-requirements/nfr-requirements.md.
"""

import os
import sys
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.yield_features import (
    FEATURE_VECTOR_LENGTH,
    build_feature_vector,
    classify_content_tier,
)


FIXED_TIME = datetime(2025, 6, 10, 18, 30, 0, tzinfo=timezone.utc)  # a Tuesday


class TestClassifyContentTier:
    def test_premium_category_returns_2(self):
        assert classify_content_tier(["IAB17"]) == 2.0

    def test_standard_category_returns_1(self):
        assert classify_content_tier(["IAB2"]) == 1.0

    def test_unrecognized_category_returns_0(self):
        assert classify_content_tier(["IAB999"]) == 0.0

    def test_empty_list_returns_0(self):
        assert classify_content_tier([]) == 0.0

    def test_none_returns_0(self):
        assert classify_content_tier(None) == 0.0

    def test_premium_takes_precedence_over_standard(self):
        assert classify_content_tier(["IAB2", "IAB17"]) == 2.0


class TestBuildFeatureVector:
    def test_returns_fixed_length(self):
        bid_request = {"site": {"cat": ["IAB17"]}}
        deal = {"id": "deal-1", "bidfloor": 5.0, "at": 1}
        vec = build_feature_vector(bid_request, deal, FIXED_TIME)
        assert len(vec) == FEATURE_VECTOR_LENGTH

    def test_first_price_onehot(self):
        vec = build_feature_vector({}, {"bidfloor": 1.0, "at": 1}, FIXED_TIME)
        assert vec[0] == 1.0
        assert vec[1] == 0.0

    def test_second_price_onehot(self):
        vec = build_feature_vector({}, {"bidfloor": 1.0, "at": 2}, FIXED_TIME)
        assert vec[0] == 0.0
        assert vec[1] == 1.0

    def test_missing_at_is_neither(self):
        vec = build_feature_vector({}, {"bidfloor": 1.0}, FIXED_TIME)
        assert vec[0] == 0.0
        assert vec[1] == 0.0

    def test_bidfloor_passthrough(self):
        vec = build_feature_vector({}, {"bidfloor": 7.5}, FIXED_TIME)
        assert vec[2] == 7.5

    def test_remnant_bidfloor_tier(self):
        vec = build_feature_vector({}, {"bidfloor": 0.5}, FIXED_TIME)
        assert vec[3] == 0.0

    def test_premium_bidfloor_tier(self):
        vec = build_feature_vector({}, {"bidfloor": 10.0}, FIXED_TIME)
        assert vec[3] == 2.0

    def test_mid_bidfloor_tier(self):
        vec = build_feature_vector({}, {"bidfloor": 3.0}, FIXED_TIME)
        assert vec[3] == 1.0

    def test_category_tier_from_site(self):
        vec = build_feature_vector({"site": {"cat": ["IAB17"]}}, {"bidfloor": 1.0}, FIXED_TIME)
        assert vec[4] == 2.0

    def test_category_tier_from_app_fallback(self):
        vec = build_feature_vector({"app": {"cat": ["IAB17"]}}, {"bidfloor": 1.0}, FIXED_TIME)
        assert vec[4] == 2.0

    def test_hour_norm(self):
        vec = build_feature_vector({}, {"bidfloor": 1.0}, FIXED_TIME)
        assert vec[5] == FIXED_TIME.hour / 24.0

    def test_weekday_norm(self):
        vec = build_feature_vector({}, {"bidfloor": 1.0}, FIXED_TIME)
        assert vec[6] == FIXED_TIME.weekday() / 7.0

    def test_missing_bidfloor_defaults_to_zero(self):
        vec = build_feature_vector({}, {}, FIXED_TIME)
        assert vec[2] == 0.0

    def test_defaults_to_now_when_no_request_time_given(self):
        vec = build_feature_vector({}, {"bidfloor": 1.0})
        assert len(vec) == FEATURE_VECTOR_LENGTH

    def test_never_raises_on_malformed_deal(self):
        # No exception for a deal missing all expected fields.
        vec = build_feature_vector({}, {"unexpected_field": "x"}, FIXED_TIME)
        assert len(vec) == FEATURE_VECTOR_LENGTH


# ---------------------------------------------------------------------------
# Property-based tests (PBT Partial mode)
# ---------------------------------------------------------------------------

_bidfloors = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
_at_values = st.one_of(st.none(), st.integers(min_value=0, max_value=5))
_iab_lists = st.lists(st.sampled_from(["IAB1", "IAB2", "IAB8", "IAB9", "IAB13", "IAB17", "IAB18", "IAB19", "IAB20", "IAB999"]))


class TestFeatureVectorProperties:
    @given(bidfloor=_bidfloors, at=_at_values)
    @settings(max_examples=200)
    def test_output_length_always_constant(self, bidfloor, at):
        """Property: build_feature_vector always returns exactly
        FEATURE_VECTOR_LENGTH floats, regardless of input."""
        deal = {"bidfloor": bidfloor, "at": at}
        vec = build_feature_vector({}, deal, FIXED_TIME)
        assert len(vec) == FEATURE_VECTOR_LENGTH
        assert all(isinstance(x, float) for x in vec)

    @given(at=_at_values)
    @settings(max_examples=200)
    def test_auction_type_onehot_mutually_exclusive(self, at):
        """Property: is_first_price and is_second_price are never both 1.0."""
        deal = {"bidfloor": 1.0, "at": at}
        vec = build_feature_vector({}, deal, FIXED_TIME)
        assert not (vec[0] == 1.0 and vec[1] == 1.0)

    @given(bidfloor=_bidfloors)
    @settings(max_examples=200)
    def test_bidfloor_tier_always_valid_ordinal(self, bidfloor):
        """Property: bidfloor_tier is always one of {0.0, 1.0, 2.0}."""
        deal = {"bidfloor": bidfloor}
        vec = build_feature_vector({}, deal, FIXED_TIME)
        assert vec[3] in (0.0, 1.0, 2.0)

    @given(categories=_iab_lists)
    @settings(max_examples=200)
    def test_category_tier_always_valid_ordinal(self, categories):
        """Property: classify_content_tier always returns one of {0.0, 1.0, 2.0}."""
        assert classify_content_tier(categories) in (0.0, 1.0, 2.0)
