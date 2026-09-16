"""Bid construction: the partition property and the seatbid round-trip."""

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from demand.artfhouse.bids import SEAT, build, to_excluded_ext, to_seatbid
from demand.artfhouse.catalog import CampaignCatalog, highest_declared_cpm
from demand.artfhouse.eligibility import evaluate
from demand.artfhouse.exclusion import ExclusionReason
from demand.artfhouse.floors import BindingFloor, FloorSource, resolve

CATALOG = CampaignCatalog()
ALL_DEAL_IDS = [d for c in CATALOG.all() for d in c.deal_ids]


def imp_with(deals=(), floor=1.0):
    return {"id": "imp-1", "bidfloor": floor, "pmp": {"deals": list(deals)}}


def floors_for(candidates, imp):
    return {c.campaign.campaign_id: resolve(imp, c.deal) for c in candidates}


def build_for(imp):
    candidates = evaluate(imp, CATALOG)
    return candidates, build(imp["id"], candidates, floors_for(candidates, imp))


def test_one_bid_per_clearing_campaign():
    imp = imp_with([{"id": "deal-home-premium"}], floor=1.0)
    _, result = build_for(imp)
    assert any(b.campaign_id == "camp-cedar" for b in result.bids)


def test_price_is_the_catalogued_cpm_not_the_floor():
    imp = imp_with([{"id": "deal-home-premium"}], floor=1.0)
    _, result = build_for(imp)
    cedar = next(b for b in result.bids if b.campaign_id == "camp-cedar")
    assert cedar.price == 6.35
    assert cedar.price != cedar.binding_floor


def test_a_campaign_below_the_floor_becomes_a_below_floor_exclusion():
    # Vantage declares 1.80; a floor of 5.0 excludes it.
    imp = imp_with([{"id": "deal-auto-brand"}], floor=5.0)
    _, result = build_for(imp)
    vantage = next(e for e in result.exclusions if e.campaign_id == "camp-vantage")
    assert vantage.reason is ExclusionReason.BELOW_FLOOR
    assert all(b.campaign_id != "camp-vantage" for b in result.bids)


def test_a_cpm_equal_to_the_floor_still_bids():
    imp = imp_with([{"id": "deal-auto-brand"}], floor=1.80)
    _, result = build_for(imp)
    assert any(b.campaign_id == "camp-vantage" for b in result.bids)


def test_the_deal_floor_can_exclude_where_the_impression_floor_would_not():
    # This is the ADJUST_DEAL_FLOOR demonstration: same impression floor, raised deal
    # floor, and a campaign that would otherwise have offered does not.
    imp = imp_with([{"id": "deal-auto-brand", "bidfloor": 4.0}], floor=1.0)
    _, result = build_for(imp)
    vantage = next(e for e in result.exclusions if e.campaign_id == "camp-vantage")
    assert vantage.reason is ExclusionReason.BELOW_FLOOR


def test_the_binding_floor_source_is_carried_for_the_ui():
    imp = imp_with([{"id": "deal-home-premium", "bidfloor": 5.0}], floor=1.0)
    _, result = build_for(imp)
    cedar = next(b for b in result.bids if b.campaign_id == "camp-cedar")
    assert cedar.floor_bound_by == FloorSource.DEAL.value


def test_seatbid_is_single_with_no_aliases():
    imp = imp_with([{"id": "deal-home-premium"}])
    _, result = build_for(imp)
    seatbid = to_seatbid(result.bids)
    assert seatbid["seat"] == SEAT
    assert isinstance(seatbid["bid"], list)


def test_seatbid_carries_campaign_identity_where_the_frontend_reads_it():
    imp = imp_with([{"id": "deal-home-premium"}])
    _, result = build_for(imp)
    bid = to_seatbid(result.bids)["bid"][0]
    artf = bid["ext"]["prebid"]["artf"]
    assert artf["campaignId"]
    assert artf["campaignName"]


def test_excluded_ext_carries_identity_and_reason():
    imp = imp_with()
    _, result = build_for(imp)
    excluded = to_excluded_ext(result.exclusions)
    assert excluded
    for entry in excluded:
        assert entry["campaignId"]
        assert entry["exclusionReason"] in [r.value for r in ExclusionReason]


def test_no_bid_carries_a_dealid_key_when_there_is_no_deal():
    # An open-market offer has no deal; emitting dealid=None would read as a deal.
    imp = imp_with()
    _, result = build_for(imp)
    openfield = next(b for b in result.bids if b.campaign_id == "camp-openfield")
    bid = next(b for b in to_seatbid(result.bids)["bid"] if b["id"] == openfield.bid_id)
    assert "dealid" not in bid


def test_catalogue_maximum_is_exposed_so_the_ceiling_can_be_derived():
    # BR-19: the adapter's ceiling must exceed this, or an eligible bid is emitted
    # here and silently dropped by Prebid's price-range filter.
    assert highest_declared_cpm(CATALOG) == 6.35


# ----------------------------------------------------------------- properties

deal_entries = st.lists(
    st.fixed_dictionaries(
        {
            "id": st.sampled_from(ALL_DEAL_IDS),
            "bidfloor": st.floats(0, 12, allow_nan=False, allow_infinity=False),
        }
    ),
    max_size=6,
)


@settings(max_examples=200)
@given(deals=deal_entries, floor=st.floats(0, 12, allow_nan=False, allow_infinity=False))
def test_property_bids_and_exclusions_partition_the_candidates(deals, floor):
    """Every candidate becomes exactly one of a bid or an exclusion."""
    imp = imp_with(deals, floor)
    candidates, result = build_for(imp)
    assert len(result.bids) + len(result.exclusions) == len(candidates)

    bid_ids = {b.campaign_id for b in result.bids}
    excluded_ids = {e.campaign_id for e in result.exclusions}
    assert bid_ids.isdisjoint(excluded_ids)
    assert bid_ids | excluded_ids == {c.campaign.campaign_id for c in candidates}


@settings(max_examples=200)
@given(deals=deal_entries, floor=st.floats(0, 12, allow_nan=False, allow_infinity=False))
def test_property_no_emitted_bid_is_below_its_binding_floor(deals, floor):
    imp = imp_with(deals, floor)
    _, result = build_for(imp)
    for bid in result.bids:
        assert bid.price >= bid.binding_floor


@settings(max_examples=200)
@given(deals=deal_entries, floor=st.floats(0, 12, allow_nan=False, allow_infinity=False))
def test_property_seatbid_round_trip_preserves_every_bid(deals, floor):
    imp = imp_with(deals, floor)
    _, result = build_for(imp)
    seatbid = to_seatbid(result.bids)
    # Through JSON, as it would travel.
    revived = json.loads(json.dumps(seatbid))
    assert len(revived["bid"]) == len(result.bids)
    prices = sorted(b["price"] for b in revived["bid"])
    assert prices == sorted(b.price for b in result.bids)


@settings(max_examples=200)
@given(floor_value=st.floats(0, 50, allow_nan=False), cpm=st.floats(0, 50, allow_nan=False))
def test_property_clearing_decides_bid_versus_exclusion(floor_value, cpm):
    floor = BindingFloor(floor_value, FloorSource.DEAL)
    assert floor.clears(cpm) == (cpm >= floor_value)
