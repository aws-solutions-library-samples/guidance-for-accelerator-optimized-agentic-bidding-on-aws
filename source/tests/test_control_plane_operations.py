"""The ten control-plane operations, against an in-memory store seeded from the real
seed file (through the seeder's own ``items`` so the table shape is the deployed one).

What these pin:
  - every write stamps managed_by/updated_at and reports the 30 s propagation window
  - a created campaign is paused; activation refuses one that could never offer
  - validation is records.py's (the seed's rules), including the one-holder-per-deal rule
  - entitlement by adomain (campaigns) and seller (deals); "not yours" reads as "not found"
  - deals: origin is the stack's, archived is final, sellerstatus drives activity
  - the records the readers see after a write load into the readers unchanged
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from control_plane import records
from control_plane.operations import (
    ControlPlaneOperations,
    InvalidRequest,
    MemoryStore,
    NotFound,
    public_record,
    to_decimals,
    to_plain,
)
from control_plane.scope import Entitlement, NotEntitled
from demand.artfhouse.catalog import CampaignCatalog, campaigns_from_records
from shared import deal_library

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
_SEEDER = os.path.join(_REPO, "deployment", "scripts", "seed_deals_table.py")
ORIGIN = "tst-prebid.example"
NOW = 1_700_000_000.0

ALL = Entitlement(frozenset(), unrestricted=True, client_id="client-all")
CEDAR_ONLY = Entitlement(frozenset({"cedarandco.example"}), unrestricted=False, client_id="client-cedar")
PARENTING_ONLY = Entitlement(frozenset({"parenting-weekly.example"}), unrestricted=False, client_id="client-pw")


@pytest.fixture(scope="module")
def seed():
    with open(os.environ["DEALS_SEED_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def seeder():
    spec = importlib.util.spec_from_file_location("seed_deals_table_ops_tests", _SEEDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store(seed, seeder):
    return MemoryStore(seeder.items(seed, ORIGIN, "paused"))


@pytest.fixture
def ops(store):
    clock = {"t": NOW}

    def tick():
        clock["t"] += 1
        return clock["t"]

    return ControlPlaneOperations(store, deal_origin=ORIGIN, price_ceiling=50.0, clock=tick)


def new_campaign(**over):
    base = {
        "campaign_id": "camp-new", "campaign_name": "New Buyer", "adomain": "newbuyer.example",
        "creative_id": "cr-new-300x250", "width": 300, "height": 250, "declared_cpm": 4.5,
        "deal_ids": [], "open_market": True, "target_categories": ["home"], "media_type": "banner",
    }
    base.update(over)
    return base


def new_deal(**over):
    base = {
        "id": "deal-new-premium", "name": "New premium", "seller": "newpub.example",
        "desc": "A new deal", "adtypes": [1], "pubcount": 1, "dinventory": 1, "auxdata": 3,
        "terms": {"dealfloor": 2.5, "cur": "USD", "guar": 0, "pricetype": 1},
        "inventory": {"inclinventory": [1], "sitedomains": ["newpub.example"], "cat": ["192"], "cattax": 9},
    }
    base.update(over)
    return base


# ----------------------------------------------------------------- campaigns


def test_list_campaigns_shows_the_seed_paused_with_seed_provenance(ops, seed):
    out = ops.list_campaigns(ALL)
    assert out["count"] == len(seed["campaigns"])
    assert {c["status"] for c in out["campaigns"]} == {"paused"}
    assert {c["managed_by"] for c in out["campaigns"]} == {"seed"}


def test_create_campaign_is_paused_and_marked(ops, store):
    out = ops.create_campaign(ALL, **new_campaign())
    assert out["campaign"]["status"] == "paused"
    assert out["visible_within_seconds"] == 30
    item = store.get("campaign", "camp-new")
    assert item["managed_by"] == "control-plane"
    assert item["updated_at"].endswith("Z")
    assert item["seq"] >= 1_000_000, "created records sort after the seed"
    assert isinstance(item["declared_cpm"], Decimal), "the table shape, not a float"
    assert "managed_by" not in out["campaign"] and "seq" not in out["campaign"]


def test_create_refuses_a_duplicate_id(ops):
    ops.create_campaign(ALL, **new_campaign())
    with pytest.raises(InvalidRequest, match="already exists"):
        ops.create_campaign(ALL, **new_campaign())


def test_create_refuses_a_bad_id(ops):
    with pytest.raises(InvalidRequest, match="lowercase"):
        ops.create_campaign(ALL, **new_campaign(campaign_id="Camp_New"))


def test_create_refuses_a_deal_another_campaign_holds(ops):
    with pytest.raises(InvalidRequest, match="already held by camp-cedar"):
        ops.create_campaign(ALL, **new_campaign(deal_ids=["deal-home-premium"], open_market=False))


def test_create_refuses_an_unknown_deal(ops):
    with pytest.raises(InvalidRequest, match="not a deal"):
        ops.create_campaign(ALL, **new_campaign(deal_ids=["deal-nowhere"], open_market=False))


def test_create_refuses_a_campaign_that_could_never_offer(ops):
    with pytest.raises(InvalidRequest, match="at least one deal or set open_market"):
        ops.create_campaign(ALL, **new_campaign(deal_ids=[], open_market=False))


def test_create_refuses_a_cpm_above_the_ceiling(ops):
    with pytest.raises(InvalidRequest, match="price ceiling"):
        ops.create_campaign(ALL, **new_campaign(declared_cpm=49.0, audience_uplift={"segment_ids": ["350"], "cpm": 2.0}))


def test_create_refuses_video_without_duration(ops):
    with pytest.raises(InvalidRequest, match="duration"):
        ops.create_campaign(ALL, **new_campaign(media_type="video"))


def test_create_refuses_unknown_fields_and_status(ops):
    with pytest.raises(InvalidRequest, match="unknown campaign fields"):
        ops.create_campaign(ALL, **new_campaign(status="active"))


def test_create_checks_entitlement_before_touching_deals(ops):
    with pytest.raises(NotEntitled, match="advertiser 'newbuyer.example'"):
        ops.create_campaign(CEDAR_ONLY, **new_campaign())


def test_activate_refuses_a_campaign_with_no_active_deal_and_no_open_market(ops, store):
    deal = store.get("deal", "deal-home-premium")
    store.put({**deal, "sellerstatus": 1})
    ledger = store.get("deal", "deal-home-premium-ledger")
    store.put({**ledger, "sellerstatus": 1})
    with pytest.raises(InvalidRequest, match="holds no active deal"):
        ops.activate_campaign(ALL, campaign_id="camp-cedar")


def test_activate_and_pause_round_trip_and_reach_the_demand_catalog(ops, store):
    out = ops.activate_campaign(ALL, campaign_id="camp-cedar")
    assert out["campaign"]["status"] == "active" and out["previous_status"] == "paused"
    assert store.get("campaign", "camp-cedar")["managed_by"] == "control-plane"

    catalog = CampaignCatalog(campaigns_from_records([public_record(i) for i in store.query("campaign")]))
    cedar = next(c for c in catalog.all() if c.campaign_id == "camp-cedar")
    assert not cedar.paused
    assert sum(1 for c in catalog.all() if c.paused) == len(catalog.all()) - 1

    out = ops.pause_campaign(ALL, campaign_id="camp-cedar")
    assert out["campaign"]["status"] == "paused" and out["previous_status"] == "active"


def test_update_changes_only_the_supplied_fields(ops, store):
    before = public_record(store.get("campaign", "camp-cedar"))
    out = ops.update_campaign(ALL, campaign_id="camp-cedar", declared_cpm=7.25, note="raised")
    after = out["campaign"]
    assert after["declared_cpm"] == 7.25 and after["note"] == "raised"
    assert out["changed"] == ["declared_cpm", "note"]
    for key in before:
        if key not in ("declared_cpm", "note"):
            assert after[key] == before[key], key
    assert after["status"] == "paused", "update never moves status"


def test_update_with_nothing_to_change_is_refused(ops):
    with pytest.raises(InvalidRequest, match="no fields"):
        ops.update_campaign(ALL, campaign_id="camp-cedar")
    with pytest.raises(InvalidRequest, match="cannot be updated"):
        ops.update_campaign(ALL, campaign_id="camp-cedar", status="active")


def test_update_revalidates_the_merged_record(ops):
    with pytest.raises(InvalidRequest, match="price ceiling"):
        ops.update_campaign(ALL, campaign_id="camp-cedar", declared_cpm=60.0)


def test_moving_a_campaign_to_another_advertiser_needs_both_entitlements(ops):
    with pytest.raises(NotEntitled):
        ops.update_campaign(CEDAR_ONLY, campaign_id="camp-cedar", adomain="other.example")
    out = ops.update_campaign(ALL, campaign_id="camp-cedar", adomain="other.example")
    assert out["campaign"]["adomain"] == "other.example"


def test_not_yours_reads_as_not_found(ops):
    with pytest.raises(NotFound):
        ops.get_campaign(PARENTING_ONLY, campaign_id="camp-cedar")
    with pytest.raises(NotFound):
        ops.get_campaign(ALL, campaign_id="camp-does-not-exist")
    with pytest.raises(NotFound):
        ops.pause_campaign(PARENTING_ONLY, campaign_id="camp-cedar")


def test_list_is_filtered_by_entitlement_status_and_advertiser(ops):
    assert {c["adomain"] for c in ops.list_campaigns(CEDAR_ONLY)["campaigns"]} == {"cedarandco.example"}
    ops.activate_campaign(ALL, campaign_id="camp-cedar")
    assert [c["campaign_id"] for c in ops.list_campaigns(ALL, status="active")["campaigns"]] == ["camp-cedar"]
    assert ops.list_campaigns(ALL, advertiser="CEDARANDCO.example")["count"] == 1
    with pytest.raises(InvalidRequest):
        ops.list_campaigns(ALL, status="stopped")


def test_get_campaign_reports_which_deals_are_active(ops, store):
    deal = store.get("deal", "deal-home-premium")
    store.put({**deal, "sellerstatus": 1})
    out = ops.get_campaign(ALL, campaign_id="camp-cedar")
    assert "deal-home-premium" not in out["active_deal_ids"]
    assert "deal-home-premium-ledger" in out["active_deal_ids"]
    assert out["managed_by"] == "seed"


# --------------------------------------------------------------------- deals


def test_create_deal_sets_origin_and_created_and_defaults_active(ops, store):
    out = ops.create_deal(ALL, deal=new_deal())
    deal = out["deal"]
    assert deal["origin"] == ORIGIN
    assert deal["created"].endswith("Z")
    assert deal["sellerstatus"] == 0
    assert store.get("deal", "deal-new-premium")["managed_by"] == "control-plane"
    assert out["visible_within_seconds"] == 30


def test_create_deal_refuses_a_caller_origin_and_unknown_fields(ops):
    with pytest.raises(InvalidRequest, match="origin is not caller-settable"):
        ops.create_deal(ALL, deal=new_deal(origin="evil.example"))
    with pytest.raises(InvalidRequest, match="not settable here"):
        ops.create_deal(ALL, deal=new_deal(created="2026-01-01T00:00:00Z"))
    with pytest.raises(InvalidRequest, match="not Deals API"):
        ops.create_deal(ALL, deal=new_deal(bogus=1))


def test_create_deal_requires_a_seller_and_entitlement_to_it(ops):
    with pytest.raises(InvalidRequest, match="seller is required"):
        ops.create_deal(ALL, deal=new_deal(seller=None) | {} if False else {k: v for k, v in new_deal().items() if k != "seller"})
    with pytest.raises(NotEntitled, match="seller 'newpub.example'"):
        ops.create_deal(CEDAR_ONLY, deal=new_deal())


def test_create_deal_applies_the_seed_rules(ops):
    with pytest.raises(InvalidRequest, match="sellerstatus"):
        ops.create_deal(ALL, deal=new_deal(sellerstatus=7))
    with pytest.raises(InvalidRequest, match="dealfloor"):
        ops.create_deal(ALL, deal=new_deal(terms={"dealfloor": -1}))
    with pytest.raises(InvalidRequest, match="already exists"):
        ops.create_deal(ALL, deal=new_deal(id="deal-home-premium", seller="hearth-and-home.example"))


def test_update_deal_pauses_and_the_readers_see_it(ops, store):
    out = ops.update_deal(ALL, deal_id="deal-home-premium", changes={"sellerstatus": 1})
    assert out["deal"]["sellerstatus"] == 1 and out["sellerstatus_name"] == "paused"
    assert out["changed"] == ["sellerstatus"]

    book = deal_library.DealBook([public_record(i) for i in store.query("deal")])
    assert book.by_id("deal-home-premium") is None
    assert "deal-home-premium" in book.inactive_deal_ids()

    got = ops.get_deal(ALL, deal_id="deal-home-premium")
    assert got["active"] is False and got["held_by_campaigns"] == ["camp-cedar"]


def test_archived_is_final(ops):
    ops.update_deal(ALL, deal_id="deal-home-premium", changes={"sellerstatus": 4})
    with pytest.raises(InvalidRequest, match="archived"):
        ops.update_deal(ALL, deal_id="deal-home-premium", changes={"sellerstatus": 0})


def test_update_deal_cannot_change_id_or_origin(ops):
    with pytest.raises(InvalidRequest, match="cannot be updated"):
        ops.update_deal(ALL, deal_id="deal-home-premium", changes={"origin": "x.example"})
    with pytest.raises(InvalidRequest, match="cannot be updated"):
        ops.update_deal(ALL, deal_id="deal-home-premium", changes={"id": "deal-renamed"})
    with pytest.raises(InvalidRequest, match="non-empty"):
        ops.update_deal(ALL, deal_id="deal-home-premium", changes={})


def test_deal_entitlement_is_by_seller(ops):
    with pytest.raises(NotFound):
        ops.get_deal(CEDAR_ONLY, deal_id="deal-home-premium")
    mine = ops.list_deals(PARENTING_ONLY)["deals"]
    assert mine and {d["seller"] for d in mine} == {"parenting-weekly.example"}
    assert all(d["held_by_campaign"] for d in mine), "every seeded deal is held by a campaign"


def test_list_deals_filters(ops):
    ops.update_deal(ALL, deal_id="deal-home-premium", changes={"sellerstatus": 1})
    paused = ops.list_deals(ALL, sellerstatus=1)["deals"]
    assert [d["id"] for d in paused] == ["deal-home-premium"]
    assert paused[0]["sellerstatus_name"] == "paused"
    assert ops.list_deals(ALL, seller="espn.com")["count"] >= 1
    with pytest.raises(InvalidRequest):
        ops.list_deals(ALL, sellerstatus=9)


def test_updated_records_still_load_into_both_readers(ops, store, seed):
    ops.create_campaign(ALL, **new_campaign())
    ops.create_deal(ALL, deal=new_deal())
    ops.update_campaign(ALL, campaign_id="camp-new", deal_ids=["deal-new-premium"], open_market=False)
    ops.activate_campaign(ALL, campaign_id="camp-new")

    campaigns = campaigns_from_records([public_record(i) for i in store.query("campaign")])
    catalog = CampaignCatalog(campaigns)
    assert catalog.by_deal("deal-new-premium").campaign_id == "camp-new"
    assert len(campaigns) == len(seed["campaigns"]) + 1
    assert campaigns[-1].campaign_id == "camp-new", "created records come after the seed"

    book = deal_library.DealBook([public_record(i) for i in store.query("deal")])
    assert book.by_id("deal-new-premium").bidfloor == 2.5
    assert [d.id for d in book.deals_for({"site": {"domain": "newpub.example"}})] == ["deal-new-premium"]


# -------------------------------------------------------------- conversions


# Magnitudes bounded to the domain's (prices, sizes, counts): past 2**53 a float that
# happens to be integral no longer round-trips through Decimal's text form exactly.
_json_scalar = st.one_of(st.integers(min_value=-10**9, max_value=10**9), st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False), st.text(max_size=20), st.booleans(), st.none())
_json = st.recursive(_json_scalar, lambda inner: st.one_of(st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=8), inner, max_size=4)), max_leaves=20)


@settings(max_examples=200, deadline=None)
@given(_json)
def test_decimal_round_trip_preserves_json_values(value):
    assert to_plain(to_decimals(value)) == value


@settings(max_examples=100, deadline=None)
@given(st.floats(min_value=0.0001, max_value=100.0))
def test_prices_survive_the_table_exactly(price):
    stored = to_decimals(float(price))
    assert isinstance(stored, Decimal)
    assert str(stored) == repr(float(price))
    assert to_plain(stored) == float(price)
