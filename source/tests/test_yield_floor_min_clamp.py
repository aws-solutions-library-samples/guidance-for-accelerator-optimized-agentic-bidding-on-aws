"""The floor container clamps an adjusted deal floor to the deal's own minimum.

A deal record's ``terms.ext.mindealfloor`` rides on the request as
``imp.pmp.deals[].ext.mindealfloor``. The floor model may move the deal floor; it
may not move it under that minimum. The model is stubbed with a fixed multiplier
and exploration is off (no explore override, not a load test), so each case is
exact arithmetic.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.artf_types import Intent, RTBRequest  # noqa: E402
import containers.yield_optimizer_floor.app as floor_app  # noqa: E402


def _run(monkeypatch, multiplier, deal):
    monkeypatch.setattr(floor_app, "_predict_floor", lambda fv, target_variant=None: (multiplier, "", ""))
    req = RTBRequest(
        id="r", applicable_intents=["ADJUST_DEAL_FLOOR"],
        bid_request={"imp": [{"id": "imp-1", "pmp": {"deals": [deal]}}], "app": {"cat": ["653"]}},
    )
    return floor_app.mutate(req).mutations


def _deal(bidfloor=12.0, ext=None):
    deal = {"id": "deal-x", "bidfloor": bidfloor, "bidfloorcur": "USD", "at": 2, "guar": 0}
    if ext is not None:
        deal["ext"] = ext
    return deal


def _floors(mutations):
    return [m.adjust_deal.bidfloor for m in mutations if m.intent == Intent.ADJUST_DEAL_FLOOR]


def test_a_floor_moved_under_the_minimum_lands_on_the_minimum(monkeypatch):
    assert _floors(_run(monkeypatch, 0.3, _deal(ext={"mindealfloor": 5.0}))) == [5.0]


def test_a_floor_moved_above_the_minimum_is_left_alone(monkeypatch):
    assert _floors(_run(monkeypatch, 0.8, _deal(ext={"mindealfloor": 5.0}))) == [9.6]


def test_a_raised_floor_is_unaffected_by_the_minimum(monkeypatch):
    assert _floors(_run(monkeypatch, 1.2, _deal(ext={"mindealfloor": 5.0}))) == [14.4]


def test_no_minimum_keeps_the_existing_arithmetic(monkeypatch):
    assert _floors(_run(monkeypatch, 0.3, _deal())) == [3.6]


@pytest.mark.parametrize("ext", [
    {"mindealfloor": True},
    {"mindealfloor": "5.00"},
    {"mindealfloor": -1.0},
    {"mindealfloor": float("nan")},
    {"mindealfloor": float("inf")},
    {"mindealfloor": None},
    {},
    "mindealfloor=5",
    [5.0],
])
def test_a_minimum_that_is_not_a_finite_non_negative_number_is_ignored(monkeypatch, ext):
    assert _floors(_run(monkeypatch, 0.3, _deal(ext=ext))) == [3.6]


def test_an_integer_minimum_is_honoured(monkeypatch):
    assert _floors(_run(monkeypatch, 0.3, _deal(ext={"mindealfloor": 5}))) == [5.0]


def test_rounding_to_four_places_is_preserved(monkeypatch):
    assert _floors(_run(monkeypatch, 0.123456, _deal(bidfloor=10.0, ext={"mindealfloor": 1.0}))) == [1.2346]


def test_the_floor_never_goes_negative(monkeypatch):
    assert _floors(_run(monkeypatch, -0.5, _deal(ext={"mindealfloor": 0.0}))) == [0.0]


def test_a_clamp_that_lands_on_the_original_floor_emits_nothing(monkeypatch):
    # The minimum equals the deal floor, so any downward move is undone: no change.
    assert _run(monkeypatch, 0.5, _deal(bidfloor=12.0, ext={"mindealfloor": 12.0})) == []


def test_a_minimum_above_the_original_floor_yields_the_minimum(monkeypatch):
    assert _floors(_run(monkeypatch, 0.9, _deal(bidfloor=4.0, ext={"mindealfloor": 5.0}))) == [5.0]


def test_a_multiplier_of_one_is_still_no_change(monkeypatch):
    assert _run(monkeypatch, 1.0, _deal(bidfloor=4.0, ext={"mindealfloor": 5.0})) == []
