"""Floor resolution: examples and properties."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from demand.artfhouse.floors import (
    SUPPORTED_CURRENCY,
    BindingFloor,
    CurrencyMismatch,
    FloorSource,
    assert_supported_currency,
    resolve,
)


def test_deal_floor_binds_when_higher():
    floor = resolve({"bidfloor": 2.0}, {"bidfloor": 5.0})
    assert floor.value == 5.0
    assert floor.bound_by is FloorSource.DEAL


def test_impression_floor_binds_when_higher():
    floor = resolve({"bidfloor": 7.0}, {"bidfloor": 5.0})
    assert floor.value == 7.0
    assert floor.bound_by is FloorSource.IMPRESSION


def test_impression_floor_binds_when_there_is_no_deal():
    floor = resolve({"bidfloor": 1.5}, None)
    assert floor.value == 1.5
    assert floor.bound_by is FloorSource.IMPRESSION


def test_absent_floor_is_zero_not_an_error():
    floor = resolve({}, None)
    assert floor.value == 0.0


def test_equal_cpm_clears_the_floor():
    # Inclusive minimum: a bid exactly at the floor is not dropped.
    assert BindingFloor(5.0, FloorSource.DEAL).clears(5.0) is True


def test_cpm_below_the_floor_does_not_clear():
    assert BindingFloor(5.0, FloorSource.DEAL).clears(4.99) is False


def test_a_tie_is_reported_as_impression_bound():
    # Equal floors: the deal floor is not higher, so the impression binds. Stated so
    # the tie is a decision rather than an accident of comparison order.
    floor = resolve({"bidfloor": 3.0}, {"bidfloor": 3.0})
    assert floor.bound_by is FloorSource.IMPRESSION


def test_supported_currency_accepted():
    assert_supported_currency(SUPPORTED_CURRENCY) is None
    assert_supported_currency(None) is None
    assert_supported_currency("usd") is None


def test_other_currency_is_an_error_not_a_rescale():
    with pytest.raises(CurrencyMismatch):
        assert_supported_currency("EUR")


# ----------------------------------------------------------------- properties

money = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)


@settings(max_examples=200)
@given(imp_floor=money, deal_floor=money)
def test_property_binding_floor_is_never_below_either_input(imp_floor, deal_floor):
    floor = resolve({"bidfloor": imp_floor}, {"bidfloor": deal_floor})
    assert floor.value >= imp_floor
    assert floor.value >= deal_floor


@settings(max_examples=200)
@given(imp_floor=money, deal_floor=money)
def test_property_binding_floor_is_one_of_its_inputs(imp_floor, deal_floor):
    floor = resolve({"bidfloor": imp_floor}, {"bidfloor": deal_floor})
    assert floor.value in (imp_floor, deal_floor)


@settings(max_examples=200)
@given(imp_floor=money, deal_floor=money)
def test_property_resolution_is_idempotent(imp_floor, deal_floor):
    once = resolve({"bidfloor": imp_floor}, {"bidfloor": deal_floor})
    twice = resolve({"bidfloor": once.value}, {"bidfloor": deal_floor})
    assert twice.value == once.value


@settings(max_examples=200)
@given(floor_value=money, cpm=money)
def test_property_clears_is_exactly_greater_or_equal(floor_value, cpm):
    floor = BindingFloor(floor_value, FloorSource.DEAL)
    assert floor.clears(cpm) == (cpm >= floor_value)


@settings(max_examples=200)
@given(value=money)
def test_property_a_cpm_equal_to_the_floor_always_clears(value):
    assert BindingFloor(value, FloorSource.IMPRESSION).clears(value) is True
