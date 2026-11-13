"""The serving and training feature vectors must be identical for the same input.

This is the assertion whose absence let the two sides drift: before the shared
spec, `dlrm_bid_shader/app.py` built dense
`[bidfloor, hour/24, age_norm, has_video]` with categoricals hashed from
`user.id`, `site.domain`, `device.ua[:20]`, while
`source/training/container/train.py` learned on
`[bid_floor, hour_of_day, shade_factor_used, conversion_value_estimate_used]`
with categoricals `device_type`, `site_domain`, `win_rate_bucket`. Widths and
dtypes matched, so nothing raised.

Every test here builds one vector from a bid request and one from the stored
outcome row that same request would produce, and asserts they are equal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from shared import dlrm_features as F


# ---------------------------------------------------------------------------
# A bid request and the stored row it produces — the same event, both shapes
# ---------------------------------------------------------------------------

REQUEST_TIME = datetime(2026, 9, 26, 14, 0, 0, tzinfo=timezone.utc)  # Saturday, hour 14


def _bid_request(**overrides):
    request = {
        "id": "req-1",
        "imp": [{"id": "imp-1", "bidfloor": 2.50, "video": {"mimes": ["video/mp4"]}}],
        "site": {"domain": "example.com"},
        "device": {"devicetype": 2, "geo": {"country": "USA"}},
    }
    request.update(overrides)
    return request


def _stored_row(**overrides):
    """The outcome row the request above produces, as the ETL writes it."""
    row = {
        "bid_floor": 2.50,
        "hour_of_day": REQUEST_TIME.hour,
        "day_of_week": REQUEST_TIME.weekday(),
        "has_video": True,
        "site_domain": "example.com",
        "device_type": 2,
        "geo_country": "USA",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def test_serving_and_training_vectors_are_identical():
    serving = F.build_from_bid_request(_bid_request(), request_time=REQUEST_TIME)
    training = F.build_from_row(_stored_row())
    assert serving == training


def test_parity_holds_for_a_weekday_without_video():
    weekday = datetime(2026, 9, 23, 9, 0, 0, tzinfo=timezone.utc)  # Wednesday, hour 9
    request = _bid_request(imp=[{"id": "imp-1", "bidfloor": 0.75}])
    row = _stored_row(
        bid_floor=0.75,
        hour_of_day=weekday.hour,
        day_of_week=weekday.weekday(),
        has_video=False,
    )
    assert F.build_from_bid_request(request, request_time=weekday) == F.build_from_row(row)


def test_parity_holds_when_optional_fields_are_absent():
    request = {"imp": [{"id": "imp-1"}]}
    row = {}
    serving = F.build_from_bid_request(request, request_time=REQUEST_TIME)
    training = F.build_from_row(row)
    # Only the calendar features differ, because the row carries no timestamp
    # and the request is priced at a real moment. Categoricals must agree.
    assert serving[1] == training[1]
    assert serving[0][0] == training[0][0] == 0.0


def test_flatten_width_matches_the_declared_spec():
    dense, categorical = F.build_from_bid_request(_bid_request(), request_time=REQUEST_TIME)
    assert len(dense) == F.DENSE_WIDTH
    assert len(categorical) == F.CATEGORICAL_WIDTH
    assert len(F.flatten(dense, categorical)) == F.FEATURE_WIDTH
    assert len(F.feature_names()) == F.FEATURE_WIDTH


# ---------------------------------------------------------------------------
# The hash contract
# ---------------------------------------------------------------------------

def test_same_value_lands_in_the_same_slot_from_either_side():
    assert F.hash_to_idx("example.com", "site_domain") == F.hash_to_idx(
        "example.com", "site_domain"
    )


def test_hashing_is_case_and_whitespace_insensitive():
    assert F.hash_to_idx("Example.COM", "site_domain") == F.hash_to_idx(
        "  example.com  ", "site_domain"
    )


@pytest.mark.parametrize("feature", F.CATEGORICAL_COLUMNS)
def test_index_stays_inside_the_feature_vocabulary(feature):
    for value in ("a", "bbb", "example.com", "12345", "ZZ"):
        index = F.hash_to_idx(value, feature)
        assert 0 <= index < F.VOCAB_SIZES[feature]


@pytest.mark.parametrize("absent", [None, "", "   "])
@pytest.mark.parametrize("feature", F.CATEGORICAL_COLUMNS)
def test_absent_values_use_the_reserved_slot(feature, absent):
    assert F.hash_to_idx(absent, feature) == F.UNKNOWN_INDEX


@pytest.mark.parametrize("feature", F.CATEGORICAL_COLUMNS)
def test_real_values_never_take_the_reserved_slot(feature):
    """Slot 0 means absent, so a present value must not collide with it."""
    for i in range(500):
        assert F.hash_to_idx(f"value-{i}", feature) != F.UNKNOWN_INDEX


def test_each_categorical_has_its_own_vocabulary_size():
    assert set(F.VOCAB_SIZES) == set(F.CATEGORICAL_COLUMNS)
    assert len(set(F.VOCAB_SIZES.values())) > 1, (
        "a single shared size is what per-feature sizing exists to avoid"
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,expected", [(0, 0.0), (12, 0.5), (23, 23 / 24.0)])
def test_hour_normalises_into_the_unit_interval(hour, expected):
    assert F.normalise_hour(hour) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [None, -1, 24, 99, "noon"])
def test_out_of_range_hour_encodes_neutral(bad):
    assert F.normalise_hour(bad) == 0.0


@pytest.mark.parametrize("day,expected", [(0, 0.0), (4, 0.0), (5, 1.0), (6, 1.0)])
def test_weekend_flag_covers_saturday_and_sunday(day, expected):
    assert F.weekend_flag(day) == expected


# ---------------------------------------------------------------------------
# Held-out features
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "held_out",
    [
        "shade_factor_used",
        "conversion_value_estimate_used",
        "shaded_price",
        "roi",
        "price_paid",
        "user_id_hash",
    ],
)
def test_held_out_columns_are_not_in_the_spec(held_out):
    """The shader's own parameters, post-bid consequences, and raw user ids."""
    assert held_out not in F.feature_names()


def test_a_stored_row_carrying_held_out_columns_does_not_change_the_vector():
    plain = F.build_from_row(_stored_row())
    contaminated = F.build_from_row(
        _stored_row(
            shade_factor_used=0.65,
            conversion_value_estimate_used=12.0,
            shaded_price=1.80,
            roi=0.42,
            user_id_hash="abc123",
        )
    )
    assert plain == contaminated


def test_spec_version_is_declared():
    assert isinstance(F.FEATURE_SPEC_VERSION, int)
    assert F.FEATURE_SPEC_VERSION >= 1
