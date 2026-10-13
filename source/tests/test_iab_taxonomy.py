"""Tests for shared/iab_taxonomy.

The property tests are the NFR-4 obligation for this unit: the module is pure, so
its invariants are checkable across generated input rather than only at examples.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import iab_taxonomy as T  # noqa: E402


# ---------------------------------------------------------------------------
# Bundled data
# ---------------------------------------------------------------------------

def test_tables_loaded_with_expected_shape():
    stats = T.stats()
    assert stats["content_categories"] == 705
    assert stats["interest_segments"] == 496
    assert stats["named_segments"] == 1558
    assert stats["emittable_segments"] == 694
    assert stats["matchable_leaf_names"] == 655
    assert stats["age_buckets"] == 13
    assert stats["special_category_data_rows"] == 63
    assert stats["content_taxonomy_version"] == "3.1"
    assert stats["audience_taxonomy_version"] == "1.1"


def test_every_demographic_node_has_a_name_not_just_age_ranges():
    """An earlier version registered only Age Range, leaving real ids nameless.

    98 Parents with Children is the case that exposed it: the parenting scenario
    names it explicitly, and it resolved to nothing.
    """
    assert T.condensed_name("98") == "Demographic | Household Data | Parents with Children"
    for household_node in ("93", "97", "99", "100", "101"):
        assert T.condensed_name(household_node) is not None


def test_category_ids_are_not_all_numeric():
    """Content Taxonomy 3.1 contains alphanumeric ids, so nothing may parse them as ints."""
    non_numeric = [cid for cid in T.CONTENT_CATEGORIES if not cid.isdigit()]
    assert non_numeric, "expected alphanumeric category ids such as JLBCU7"
    for cid in ("JLBCU7", "8VZQHL", "SPSHQ5"):
        assert cid in T.CONTENT_CATEGORIES


# ---------------------------------------------------------------------------
# Taxonomy selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cattax,expected", [
    (None, T.TAXONOMY_CONTENT_1_0),
    (1, T.TAXONOMY_CONTENT_1_0),
    (7, T.TAXONOMY_CONTENT_3_X),
    (9, T.TAXONOMY_CONTENT_3_X),
    # Content 2.x is deliberately unrecognised: its ids are not interchangeable
    # with 3.x, so resolving them against a 3.1 table would mis-resolve silently.
    (2, T.TAXONOMY_UNRECOGNISED),
    (5, T.TAXONOMY_UNRECOGNISED),
    (6, T.TAXONOMY_UNRECOGNISED),
    (4, T.TAXONOMY_UNRECOGNISED),
    (99, T.TAXONOMY_UNRECOGNISED),
    ("9", T.TAXONOMY_UNRECOGNISED),
    (True, T.TAXONOMY_UNRECOGNISED),
])
def test_resolve_taxonomy(cattax, expected):
    assert T.resolve_taxonomy(cattax) == expected


# ---------------------------------------------------------------------------
# The Special Category Data gate
# ---------------------------------------------------------------------------

def test_scd_flag_is_per_row_not_inherited():
    """186 is flagged, its child 192 is not, and two grandchildren are.

    This is what makes the gate testable on ancestors: if the flag were inherited,
    flagging 197 and 198 separately would have been redundant.
    """
    assert T.is_special_category_data("186") is True
    assert T.is_special_category_data("192") is False
    assert T.is_special_category_data("197") is True
    assert T.is_special_category_data("198") is True


def test_gate_withholds_flagged_and_allows_unflagged_sibling():
    outcomes = {o.category_id: o for o in T.segments_for_categories(["186", "192"])}
    assert outcomes["186"].disposition == T.WITHHELD
    assert outcomes["186"].segment_id is None
    assert outcomes["192"].disposition == T.MAPPED
    assert outcomes["192"].segment_id == "350"


def test_dispositions_distinguish_withheld_from_no_equivalent():
    outcomes = {o.category_id: o.disposition
                for o in T.segments_for_categories(["192", "186", "IAB17", "frontpage"])}
    assert outcomes["192"] == T.MAPPED
    assert outcomes["186"] == T.WITHHELD
    # A Content 1.0 code is not in the 3.1 table at all.
    assert outcomes["IAB17"] == T.UNRECOGNISED
    assert outcomes["frontpage"] == T.UNRECOGNISED


def test_segment_for_category_does_not_apply_the_gate():
    """The gate belongs to segments_for_categories, so this stays usable for inspection."""
    assert T.segment_for_category("186") == "348"
    assert T.is_special_category_data("186") is True


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,segment", [
    ("192", "350"),   # Family and Relationships / Parenting
    ("186", "348"),   # Family and Relationships
    ("196", "354"),   # ... / Parenting Babies and Toddlers
    ("483", "607"),   # Sports
    ("680", "733"),   # Video Gaming
    ("210", "368"),   # Food & Drink
    ("1", "243"),     # Automotive
])
def test_known_mappings(category, segment):
    assert T.segment_for_category(category) == segment


def test_categories_describing_content_rather_than_interest_do_not_map():
    """Not a gap: these describe a page, not a durable user interest."""
    for name in ("Crime", "Disasters", "Politics", "Religion & Spirituality"):
        matches = [c for c in T.CONTENT_CATEGORIES.values()
                   if c.tier_path and c.tier_path[0] == name and len(c.tier_path) == 1]
        assert matches, f"expected a tier-1 category named {name}"
        assert T.segment_for_category(matches[0].id) is None


def test_mapping_coverage_is_what_the_design_claims():
    mapped = sum(1 for cid in T.CONTENT_CATEGORIES if T.segment_for_category(cid))
    assert mapped == 565, f"design records 565 of 705 resolving, got {mapped}"


def test_unknown_category_returns_none_rather_than_raising():
    assert T.segment_for_category("no-such-category") is None
    assert T.segment_for_category("") is None


# ---------------------------------------------------------------------------
# Age buckets
# ---------------------------------------------------------------------------

def test_age_buckets_come_from_the_taxonomy():
    ids = {b.id for b in T.AGE_BUCKETS}
    assert {"3", "4", "5", "6", "7"} <= ids
    first = T.AGE_BUCKETS[0]
    assert (first.lower_age, first.upper_age) == (18, 20)
    # The top bucket is open ended.
    assert T.AGE_BUCKETS[-1].upper_age is None


def test_age_bucket_id_uses_five_year_ranges():
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # 36 and 44 landed in the same invented ten-year bucket; the taxonomy separates them.
    assert T.age_bucket_id(1990, now=fixed) == "7"    # 36 -> 35-39
    assert T.age_bucket_id(1982, now=fixed) == "8"    # 44 -> 40-44
    assert T.age_bucket_id(1990, now=fixed) != T.age_bucket_id(1982, now=fixed)


@pytest.mark.parametrize("bad", [None, "1990", 3000, 1800, True, 1.5])
def test_age_bucket_id_rejects_unusable_input(bad):
    assert T.age_bucket_id(bad) is None


def test_age_below_the_lowest_bucket_returns_none():
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert T.age_bucket_id(2020, now=fixed) is None   # age 6, below 18


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def test_condensed_names_are_the_taxonomy_s_own():
    assert T.condensed_name("350") == "Interest | Family and Relationships | Parenting"
    assert T.condensed_name("7") == "Demographic | Age Range | 35-39"


def test_condensed_name_unknown_is_none_not_a_guess():
    assert T.condensed_name("no-such-segment") is None


def test_segment_names_covers_every_mappable_segment():
    names = T.segment_names()
    for cid in T.CONTENT_CATEGORIES:
        seg = T.segment_for_category(cid)
        if seg:
            assert seg in names, f"segment {seg} has no name"


def test_emittable_names_exclude_unreachable_purchase_intent():
    emittable = T.segment_names(emittable_only=True)
    everything = T.segment_names()
    assert len(emittable) == 694
    assert len(everything) == 1558
    # Interest and Demographic are reachable; Purchase Intent is not.
    assert "350" in emittable and "98" in emittable and "7" in emittable
    purchase_intent = [sid for sid, name in everything.items()
                       if name.startswith("Purchase Intent")]
    assert purchase_intent, "expected Purchase Intent nodes to exist"
    assert not any(sid in emittable for sid in purchase_intent)


# ---------------------------------------------------------------------------
# Asserted-name matching
# ---------------------------------------------------------------------------

def test_asserted_name_reaches_a_life_stage_segment():
    """98 is reachable ONLY from asserted data, never from page content."""
    assert T.segment_for_asserted_name("Parents with Children") == "98"


@pytest.mark.parametrize("asserted,expected", [
    ("Parents with Children", "98"),
    ("parents_with_children", "98"),
    ("PARENTS WITH CHILDREN", "98"),
    ("  Parents   with  Children  ", "98"),
    ("Parenting", "350"),
    ("Parenting Babies and Toddlers", "354"),
    ("Sports", "607"),
    ("Empty Nest (Adults, Children left home)", "101"),
])
def test_asserted_name_normalisation(asserted, expected):
    assert T.segment_for_asserted_name(asserted) == expected


@pytest.mark.parametrize("asserted", [
    "totally made up segment",
    # A vendor prefix defeats the match deliberately: matching is exact after
    # normalisation, never a substring, because substring matching across 655 node
    # names would produce false positives.
    "HH: Parents with Children",
    "parent",
    "",
    None,
    123,
])
def test_asserted_name_no_match_is_none_not_a_guess(asserted):
    assert T.segment_for_asserted_name(asserted) is None


def test_asserted_name_never_reaches_purchase_intent():
    """The taxonomy marks that tier with an asterisk and defers to a separate
    classification extension, so a name match must not claim purchase intent."""
    everything = T.segment_names()
    for sid, name in everything.items():
        if not name.startswith("Purchase Intent"):
            continue
        leaf = name.split("|")[-1].strip()
        matched = T.segment_for_asserted_name(leaf)
        assert matched != sid


# ---------------------------------------------------------------------------
# Properties (NFR-4)
# ---------------------------------------------------------------------------

_CATEGORY_IDS = sorted(T.CONTENT_CATEGORIES)


@settings(max_examples=300, deadline=None)
@given(st.lists(st.text(), max_size=12))
def test_never_raises_for_arbitrary_input(ids):
    T.segments_for_categories(ids)


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(_CATEGORY_IDS), max_size=10))
def test_every_mapped_segment_exists_in_the_taxonomy(ids):
    names = T.segment_names()
    for outcome in T.segments_for_categories(ids):
        if outcome.disposition == T.MAPPED:
            assert outcome.segment_id in names


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(_CATEGORY_IDS), max_size=10))
def test_flagged_categories_never_yield_a_segment(ids):
    for outcome in T.segments_for_categories(ids):
        if T.is_special_category_data(outcome.category_id):
            assert outcome.disposition == T.WITHHELD
            assert outcome.segment_id is None


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(_CATEGORY_IDS), max_size=10))
def test_one_outcome_per_usable_input_in_order(ids):
    outcomes = T.segments_for_categories(ids)
    assert [o.category_id for o in outcomes] == ids


@settings(max_examples=200, deadline=None)
@given(st.lists(st.text(min_size=1), max_size=10))
def test_disposition_is_always_one_of_the_four(ids):
    allowed = {T.MAPPED, T.WITHHELD, T.NO_EQUIVALENT, T.UNRECOGNISED}
    for outcome in T.segments_for_categories(ids):
        assert outcome.disposition in allowed


@settings(max_examples=300, deadline=None)
@given(st.integers(min_value=1901, max_value=2026))
def test_age_bucket_is_always_a_real_taxonomy_id_or_none(yob):
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = T.age_bucket_id(yob, now=fixed)
    if result is not None:
        assert result in {b.id for b in T.AGE_BUCKETS}


@settings(max_examples=200, deadline=None)
@given(st.sampled_from(_CATEGORY_IDS))
def test_mapping_is_deterministic(cid):
    assert T.segment_for_category(cid) == T.segment_for_category(cid)


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=60))
def test_asserted_name_never_raises_and_returns_a_real_id_or_none(name):
    result = T.segment_for_asserted_name(name)
    if result is not None:
        assert result in T.segment_names()
        assert result in T.segment_names(emittable_only=True)


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=40))
def test_normalisation_is_idempotent(name):
    once = T.normalise_asserted_name(name)
    assert T.normalise_asserted_name(once) == once


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=40))
def test_normalisation_yields_no_leading_trailing_or_double_spaces(name):
    result = T.normalise_asserted_name(name)
    assert result == result.strip()
    assert "  " not in result
