"""The outcome event carries every column the DLRM feature spec reads.

`BidShadingOutcomeEvent` gained `day_of_week`, `geo_country` and `has_video`
because `shared/dlrm_features.py` names them. A column the spec reads but the
event does not carry means the training side reads a default while the serving
side supplies a real value — equal widths, different meanings.

The last test in this module is the one that matters most: it runs a generated
load-test request through the context extraction and then through both feature
build paths, and asserts the two vectors agree. That chains generator → event →
spec, which is the path the drift actually travelled.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from orchestrator.feedback_integration import _load_test_context
from orchestrator.loadtest import (
    TRAFFIC_SCENARIO_KEYS,
    _tmpl_shade,
    get_traffic_profile,
)
from shared import dlrm_features as F
from shared.feedback_models import BidShadingOutcomeEvent


def _event_kwargs(**overrides):
    kwargs = {
        "request_id": "11111111-1111-1111-1111-111111111111",
        "timestamp": 1_800_000_000.0,
        "model_version": "dlrm-test-v1",
        "source": "live",
        "original_price": 3.00,
        "shaded_price": 2.00,
        "bid_floor": 1.00,
        "won": False,
        "price_paid": None,
        "impression": False,
        "click": False,
        "conversion": False,
        "conversion_value": None,
        "user_id_hash": "abcdef0123456789",
        "site_domain": "espn.com",
        "device_type": "2",
        "hour_of_day": 14,
        "shade_factor_used": 0.65,
        "conversion_value_estimate_used": 12.0,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 1.0a — the schema change is additive
# ---------------------------------------------------------------------------

def test_event_still_constructs_without_the_new_fields():
    """Additive with defaults, so stored events and other producers stay valid."""
    event = BidShadingOutcomeEvent(**_event_kwargs())
    assert event.day_of_week == 0
    assert event.geo_country == ""
    assert event.has_video is False


def test_event_carries_the_new_fields_when_supplied():
    event = BidShadingOutcomeEvent(
        **_event_kwargs(day_of_week=5, geo_country="CAN", has_video=True)
    )
    assert (event.day_of_week, event.geo_country, event.has_video) == (5, "CAN", True)


@pytest.mark.parametrize("bad_day", [-1, 7, 99])
def test_day_of_week_is_range_checked(bad_day):
    with pytest.raises(Exception):
        BidShadingOutcomeEvent(**_event_kwargs(day_of_week=bad_day))


def test_new_fields_survive_serialisation():
    """FeedbackCollector emits json.dumps(event.model_dump()), so the fields
    have to be in the dump to reach the ETL at all."""
    event = BidShadingOutcomeEvent(
        **_event_kwargs(day_of_week=6, geo_country="GBR", has_video=True)
    )
    dumped = event.model_dump()
    assert dumped["day_of_week"] == 6
    assert dumped["geo_country"] == "GBR"
    assert dumped["has_video"] is True


def test_every_spec_column_is_present_on_the_event():
    """The assertion that keeps the two in step as the spec changes."""
    event_fields = set(BidShadingOutcomeEvent.model_fields)
    # hour_norm and is_weekend are derived by the spec from these stored columns.
    derived = {"hour_norm": "hour_of_day", "is_weekend": "day_of_week"}
    for column in F.feature_names():
        stored = derived.get(column, column)
        assert stored in event_fields, (
            f"feature spec reads {column!r} but the event has no {stored!r} column"
        )


# ---------------------------------------------------------------------------
# 1.0b — the generator produces the context, and it is recorded
# ---------------------------------------------------------------------------

def test_generated_shade_request_carries_devicetype_and_country():
    """Without these the serving path reads absent categoricals, so the load
    test would not exercise the features it is meant to produce data for."""
    rng = random.Random(4242)
    profile = get_traffic_profile(None)
    for idx in range(25):
        device = _tmpl_shade(rng, idx, profile)["bid_request"]["device"]
        assert device["devicetype"] in (2, 4, 5)
        assert device["geo"]["country"] in ("USA", "CAN", "GBR", "DEU", "AUS")


def test_devicetype_agrees_with_the_user_agent():
    """An iPhone UA reporting a desktop devicetype would be a contradiction the
    containers could learn from."""
    rng = random.Random(7)
    profile = get_traffic_profile(None)
    for idx in range(60):
        device = _tmpl_shade(rng, idx, profile)["bid_request"]["device"]
        ua, devicetype = device["ua"], device["devicetype"]
        if "iPhone" in ua or "Android" in ua:
            assert devicetype == 4
        elif "iPad" in ua:
            assert devicetype == 5
        else:
            assert devicetype == 2


def test_load_test_context_records_the_request_not_a_marker():
    rng = random.Random(99)
    payload = _tmpl_shade(rng, 0, get_traffic_profile(None))
    context = _load_test_context(payload["bid_request"])
    assert context["site_domain"] != "load-test"
    assert context["device_type"] != "load-test"
    assert context["site_domain"] in get_traffic_profile(None).domains
    assert context["geo_country"] != ""


def test_load_test_context_varies_across_requests():
    """Finding 3: two categoricals held constant give their embedding tables one
    value to separate, which is indistinguishable from having learned nothing."""
    rng = random.Random(2026)
    profile = get_traffic_profile(None)
    contexts = [
        _load_test_context(_tmpl_shade(rng, i, profile)["bid_request"])
        for i in range(200)
    ]
    assert len({c["site_domain"] for c in contexts}) > 1
    assert len({c["device_type"] for c in contexts}) > 1
    assert len({c["geo_country"] for c in contexts}) > 1


def test_load_test_context_falls_back_to_the_marker_without_a_request():
    context = _load_test_context(None)
    assert context["site_domain"] == "load-test"
    assert context["device_type"] == "load-test"
    assert 0 <= context["hour_of_day"] <= 23
    assert 0 <= context["day_of_week"] <= 6


@pytest.mark.parametrize("profile_key", list(TRAFFIC_SCENARIO_KEYS))
def test_every_profile_produces_varied_context(profile_key):
    rng = random.Random(11)
    profile = get_traffic_profile(profile_key)
    contexts = [
        _load_test_context(_tmpl_shade(rng, i, profile)["bid_request"])
        for i in range(120)
    ]
    assert len({c["site_domain"] for c in contexts}) > 1, profile_key
    assert len({c["device_type"] for c in contexts}) > 1, profile_key


# ---------------------------------------------------------------------------
# The chain that matters: generator -> event context -> both feature paths
# ---------------------------------------------------------------------------

def test_features_agree_across_the_full_load_test_chain():
    """Build features from the generated request, and from the outcome row that
    request produces, and assert they match.

    This is the end-to-end form of the parity assertion. The module-level test
    in test_dlrm_feature_parity.py proves the spec is self-consistent; this
    proves the event schema carries enough to reproduce the serving vector.
    """
    rng = random.Random(31337)
    profile = get_traffic_profile(None)
    at = datetime(2026, 9, 26, 14, 0, 0, tzinfo=timezone.utc)  # Saturday

    for idx in range(50):
        bid_request = _tmpl_shade(rng, idx, profile)["bid_request"]
        serving = F.build_from_bid_request(bid_request, request_time=at)

        context = _load_test_context(bid_request)
        row = {
            "bid_floor": bid_request["imp"][0]["bidfloor"],
            "hour_of_day": at.hour,
            "day_of_week": at.weekday(),
            "has_video": context["has_video"],
            "site_domain": context["site_domain"],
            "device_type": context["device_type"],
            "geo_country": context["geo_country"],
        }
        training = F.build_from_row(row)

        assert serving == training, f"request {idx} diverged: {serving} != {training}"
