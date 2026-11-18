"""Eligibility: examples and the completeness property."""

from hypothesis import given, settings
from hypothesis import strategies as st

from demand.artfhouse.catalog import CampaignCatalog
from demand.artfhouse.eligibility import evaluate
from demand.artfhouse.exclusion import ExclusionReason

CATALOG = CampaignCatalog()
ALL_DEAL_IDS = [d for c in CATALOG.all() for d in c.deal_ids]


def imp_with(deals=(), categories=None, floor=1.0):
    imp = {"id": "imp-1", "bidfloor": floor, "pmp": {"deals": list(deals)}}
    if categories is not None:
        imp["ext"] = {"artf": {"categories": list(categories)}}
    return imp


def by_campaign(candidates):
    return {c.campaign.campaign_id: c for c in candidates}


def test_every_campaign_is_considered():
    candidates = evaluate(imp_with(), CATALOG)
    assert len(candidates) == len(CATALOG.all())


def test_a_matching_live_deal_makes_a_campaign_eligible():
    candidates = by_campaign(evaluate(imp_with([{"id": "deal-home-premium"}]), CATALOG))
    cedar = candidates["camp-cedar"]
    assert cedar.eligible
    assert cedar.deal_id == "deal-home-premium"


def test_no_deal_on_the_impression_excludes_with_that_reason():
    candidates = by_campaign(evaluate(imp_with(), CATALOG))
    assert candidates["camp-cedar"].excluded_because is ExclusionReason.NO_DEAL_ON_IMPRESSION


def test_a_suppressed_deal_excludes_with_suppression_not_a_missing_deal():
    deals = [{"id": "deal-finance-pmp", "ext": {"artf": {"suppressed": True}}}]
    candidates = by_campaign(evaluate(imp_with(deals), CATALOG))
    harbour = candidates["camp-harbour"]
    assert harbour.excluded_because is ExclusionReason.DEAL_SUPPRESSED
    # The distinction FR-13 requires: not confused with a below-floor rejection.
    assert harbour.excluded_because is not ExclusionReason.BELOW_FLOOR


def test_suppression_flag_is_also_read_at_the_top_level_of_ext():
    deals = [{"id": "deal-finance-pmp", "ext": {"suppressed": True}}]
    candidates = by_campaign(evaluate(imp_with(deals), CATALOG))
    assert candidates["camp-harbour"].excluded_because is ExclusionReason.DEAL_SUPPRESSED


def test_targeting_mismatch_excludes_with_that_reason():
    deals = [{"id": "deal-auto-brand"}]
    candidates = by_campaign(evaluate(imp_with(deals, categories=["finance"]), CATALOG))
    assert candidates["camp-vantage"].excluded_because is ExclusionReason.NOT_TARGETED


def test_targeting_matches_when_the_impression_declares_no_categories():
    # Absence of a signal is not a mismatch: excluding here would reject a campaign
    # for a reason the request never stated.
    deals = [{"id": "deal-auto-brand"}]
    candidates = by_campaign(evaluate(imp_with(deals), CATALOG))
    assert candidates["camp-vantage"].eligible


def test_an_open_market_campaign_is_eligible_with_no_deal():
    candidates = by_campaign(evaluate(imp_with(), CATALOG))
    assert candidates["camp-openfield"].eligible
    assert candidates["camp-openfield"].deal_id is None


def test_below_floor_is_not_decided_here():
    # It needs the binding floor, so it is decided when bids are built. No candidate
    # leaves this stage carrying BELOW_FLOOR.
    candidates = evaluate(imp_with([{"id": "deal-auto-brand"}], floor=99.0), CATALOG)
    assert all(c.excluded_because is not ExclusionReason.BELOW_FLOOR for c in candidates)


# ----------------------------------------------------------------- properties

deal_entries = st.lists(
    st.fixed_dictionaries(
        {
            "id": st.sampled_from(ALL_DEAL_IDS + ["deal-unknown"]),
            "ext": st.fixed_dictionaries({"artf": st.fixed_dictionaries({"suppressed": st.booleans()})}),
        }
    ),
    max_size=6,
)

category_lists = st.lists(
    st.sampled_from(["home", "retail", "automotive", "finance", "sport"]), max_size=3
)


@settings(max_examples=200)
@given(deals=deal_entries, categories=category_lists, floor=st.floats(0, 50, allow_nan=False))
def test_property_one_candidate_per_campaign_always(deals, categories, floor):
    """Completeness: nothing is silently dropped, and nothing is duplicated."""
    candidates = evaluate(imp_with(deals, categories, floor), CATALOG)
    assert len(candidates) == len(CATALOG.all())
    ids = [c.campaign.campaign_id for c in candidates]
    assert sorted(ids) == sorted(c.campaign_id for c in CATALOG.all())


@settings(max_examples=200)
@given(deals=deal_entries, categories=category_lists)
def test_property_every_candidate_is_eligible_xor_excluded(deals, categories):
    for candidate in evaluate(imp_with(deals, categories), CATALOG):
        assert candidate.eligible == (candidate.excluded_because is None)


@settings(max_examples=200)
@given(deals=deal_entries, categories=category_lists)
def test_property_an_exclusion_reason_is_always_a_known_member(deals, categories):
    for candidate in evaluate(imp_with(deals, categories), CATALOG):
        if candidate.excluded_because is not None:
            assert candidate.excluded_because in tuple(ExclusionReason)


@settings(max_examples=200)
@given(deals=deal_entries, categories=category_lists)
def test_property_evaluation_is_deterministic(deals, categories):
    imp = imp_with(deals, categories)
    first = [(c.campaign.campaign_id, c.excluded_because) for c in evaluate(imp, CATALOG)]
    second = [(c.campaign.campaign_id, c.excluded_because) for c in evaluate(imp, CATALOG)]
    assert first == second
