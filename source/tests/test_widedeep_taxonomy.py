"""Tests for the Audience Activator's taxonomy support.

The legacy-equivalence test is the one that matters most: this is the only unit in
the Auction Theater feature that changes code on the live bid path, so requests that
do not opt into a modern taxonomy must behave as they did before.

The expected legacy output below was CAPTURED from the implementation as it stood
before this change, not typed from an expectation. See
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from containers.widedeep_segment_activator.app import mutate  # noqa: E402
from shared.artf_types import Intent, Operation, RTBRequest  # noqa: E402
from shared import iab_taxonomy as T  # noqa: E402

_SAMPLES = os.path.join(os.path.dirname(__file__), "..", "frontend-react", "public", "samples")

# Captured from the pre-change implementation. Age identifiers are listed
# separately because they are the one documented behavioural change (BR-13/BR-20):
# the container now emits the taxonomy's five-year ranges instead of the ten-year
# ranges it invented.
LEGACY_BASELINE = {
    "banner-basic.json": {
        "non_age": ["ctx-mobile", "int-sports"],
        "old_age": ["demo-35-44"],
    },
    "bid-shading.json": {"non_age": [], "old_age": []},
    "isv-ecosystem.json": {
        "non_age": ["ctx-mobile", "ctx-premium", "int-auto", "int-sports"],
        "old_age": ["demo-35-44"],
    },
    "video-deals.json": {
        "non_age": ["ctx-premium", "ctx-video"],
        "old_age": ["demo-35-44"],
    },
    "yield-optimizer.json": {"non_age": [], "old_age": []},
}

_AGE_IDS = {b.id for b in T.AGE_BUCKETS}


def _segments(payload: dict) -> list[str]:
    response = mutate(RTBRequest(**payload))
    for mutation in response.mutations:
        if mutation.ids:
            return list(mutation.ids.id)
    return []


def _request(cat, cattax, *, yob=1991, intents=("ACTIVATE_SEGMENTS",)) -> dict:
    return {
        "id": "test-request",
        "applicable_intents": list(intents),
        "bid_request": {
            "imp": [{"id": "1", "bidfloor": 0.9, "banner": {"w": 300, "h": 250}}],
            "site": {"domain": "example.com", "cat": cat, "cattax": cattax},
            "user": {"yob": yob},
            "device": {"ua": "Mozilla/5.0 (Macintosh)"},
        },
    }


# ---------------------------------------------------------------------------
# Legacy equivalence (BR-20, NFR2-6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", sorted(LEGACY_BASELINE))
def test_legacy_non_age_output_is_unchanged(fixture):
    with open(os.path.join(_SAMPLES, fixture), encoding="utf-8") as handle:
        payload = json.load(handle)
    site = payload["bid_request"].get("site") or payload["bid_request"].get("app") or {}
    assert site.get("cattax") in (None, 1), "fixture is expected to be on the legacy path"

    produced = _segments(payload)
    non_age = sorted(s for s in produced if s not in _AGE_IDS)
    assert non_age == sorted(LEGACY_BASELINE[fixture]["non_age"])


@pytest.mark.parametrize("fixture", sorted(LEGACY_BASELINE))
def test_legacy_age_identifiers_became_taxonomy_ids(fixture):
    """The one documented behavioural change on the legacy path."""
    with open(os.path.join(_SAMPLES, fixture), encoding="utf-8") as handle:
        payload = json.load(handle)
    produced = _segments(payload)
    expected_count = len(LEGACY_BASELINE[fixture]["old_age"])
    age_ids = [s for s in produced if s in _AGE_IDS]
    assert len(age_ids) == expected_count
    assert not [s for s in produced if s.startswith("demo-")], "invented demo- buckets are gone"


def test_legacy_content_codes_still_map_to_vendor_interest_segments():
    """cattax 1 keeps the Content 1.0 code map, so IAB17 still yields int-sports."""
    assert _segments(_request(["IAB17"], 1)) == sorted(["int-sports", T.age_bucket_id(1991)])
    assert _segments(_request(["IAB17"], None)) == sorted(["int-sports", T.age_bucket_id(1991)])


# ---------------------------------------------------------------------------
# Modern taxonomy path
# ---------------------------------------------------------------------------

def test_content_3_1_category_yields_a_real_audience_segment():
    produced = _segments(_request(["192"], 9))
    assert "350" in produced
    assert T.condensed_name("350") == "Interest | Family and Relationships | Parenting"


def test_content_3_0_is_also_supported():
    assert "350" in _segments(_request(["192"], 7))


def test_multiple_categories_all_resolve():
    produced = _segments(_request(["192", "483", "680"], 9))
    assert {"350", "607", "733"} <= set(produced)


def test_content_2_x_derives_no_category_segments():
    """Its ids are not interchangeable with 3.x, so it is unrecognised, not guessed at."""
    for cattax in (2, 5, 6):
        produced = _segments(_request(["192"], cattax))
        assert "350" not in produced
        # Rules unrelated to the content taxonomy still run.
        assert T.age_bucket_id(1991) in produced


def test_unrecognised_taxonomy_does_not_fall_back_to_the_legacy_map():
    produced = _segments(_request(["IAB17"], 99))
    assert "int-sports" not in produced


# ---------------------------------------------------------------------------
# The privacy gate (NFR2-12, NFR2-17)
# ---------------------------------------------------------------------------

def test_flagged_category_derives_no_segment():
    produced = _segments(_request(["186"], 9))
    assert "348" not in produced, "186 carries Special Category Data"


def test_unflagged_sibling_still_resolves_when_a_flagged_one_is_present():
    """The gate is per category, not per request."""
    produced = _segments(_request(["192", "186"], 9))
    assert "350" in produced
    assert "348" not in produced


def test_flagged_grandchild_is_also_withheld():
    produced = _segments(_request(["197"], 9))
    assert "355" not in produced


def test_page_content_alone_never_yields_a_life_stage_segment():
    """A parenting article does not make the reader a parent.

    98 Parents with Children must be unreachable from content categories. It is a
    personal circumstance, and inferring one from what someone read is the inference
    the Special Category Data flag exists to discourage.
    """
    produced = _segments(_request(["192", "186", "196", "197"], 9))
    assert "98" not in produced


def test_withheld_category_contributes_no_score(monkeypatch):
    """BR-19: a withheld category must not push a segment over the threshold.

    With only a flagged category present, the sole remaining activation is the age
    bucket, so the segment list must contain nothing category-derived.
    """
    produced = _segments(_request(["186"], 9))
    assert produced == [T.age_bucket_id(1991)]


# ---------------------------------------------------------------------------
# Asserted audience data (FR-32)
# ---------------------------------------------------------------------------

def _request_with_asserted(names, *, cattax=9, cat=("192",), yob=1991) -> dict:
    payload = _request(list(cat), cattax, yob=yob)
    payload["bid_request"]["user"]["data"] = [
        {"id": "dmp-1", "name": "ExampleDMP",
         "segment": [{"id": f"s{i}", "name": n} for i, n in enumerate(names)]}
    ]
    return payload


def test_asserted_life_stage_yields_the_real_identifier():
    produced = _segments(_request_with_asserted(["Parents with Children"]))
    assert "98" in produced


def test_the_parenting_narrative_produces_all_three_identifiers():
    """FR-32 names 350, 354 and 98. All three must come from real signals."""
    produced = _segments(_request_with_asserted(
        ["Parents with Children", "Parenting Babies and Toddlers"]
    ))
    assert {"350", "354", "98"} <= set(produced)


def test_asserted_name_that_matches_no_node_falls_back_to_the_keyword_map():
    produced = _segments(_request_with_asserted(["auto enthusiasts"], cat=[], cattax=1))
    assert "int-auto" in produced


def test_asserted_names_work_on_the_legacy_path_too():
    """user.data has nothing to do with cattax, which describes content categories."""
    produced = _segments(_request_with_asserted(["Parents with Children"], cattax=1, cat=["IAB17"]))
    assert "98" in produced
    assert "int-sports" in produced


def test_malformed_asserted_data_does_not_raise():
    payload = _request(["192"], 9)
    payload["bid_request"]["user"]["data"] = [
        {}, {"segment": []}, {"segment": [{}]}, {"segment": [{"name": None}]},
    ]
    assert "350" in _segments(payload)


# ---------------------------------------------------------------------------
# Contract stability (NFR2-7, NFR2-8)
# ---------------------------------------------------------------------------

def test_mutation_shape_is_unchanged():
    response = mutate(RTBRequest(**_request(["192"], 9)))
    assert len(response.mutations) == 1
    mutation = response.mutations[0]
    assert mutation.intent == Intent.ACTIVATE_SEGMENTS
    assert mutation.op == Operation.ADD
    assert mutation.path == "/user/data/segment"
    assert mutation.ids is not None


def test_intent_gate_still_short_circuits():
    response = mutate(RTBRequest(**_request(["192"], 9, intents=("BID_SHADE",))))
    assert response.mutations == []


def test_score_segments_keeps_its_signature():
    from containers.widedeep_segment_activator.app import _score_segments
    import inspect
    params = list(inspect.signature(_score_segments).parameters)
    assert params == ["bid_request", "threshold"]


def test_output_is_sorted_and_deduplicated():
    produced = _segments(_request(["192", "192", "483"], 9))
    assert produced == sorted(produced)
    assert len(produced) == len(set(produced))


# ---------------------------------------------------------------------------
# Robustness (NFR2-10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cat", [
    [],
    ["not-a-real-category"],
    ["192", "not-a-real-category"],
    ["", "192"],
    [None, "192"],
    [123, "192"],
])
def test_malformed_categories_never_raise_and_do_not_block_the_others(cat):
    produced = _segments(_request(cat, 9))
    if "192" in [c for c in cat if isinstance(c, str)]:
        assert "350" in produced


def test_missing_site_and_user_do_not_raise():
    payload = {
        "id": "t",
        "applicable_intents": ["ACTIVATE_SEGMENTS"],
        "bid_request": {"imp": [{"id": "1"}]},
    }
    assert _segments(payload) == []


def test_threshold_override_still_applies():
    payload = _request(["192"], 9)
    payload["ext"] = {"model_params": {"segment_threshold": 0.99}}
    assert _segments(payload) == []
