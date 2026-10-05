"""Audience uplift: a buyer paying more for a matched audience.

The point of the Theater's with/without comparison is that the ARTF containers'
mutations change the auction. Deal activation changes WHO can offer; this is the
other half, a buyer changing WHAT it offers because the Audience Activator placed the
user in a segment it targets. Pinned here:

- the price is declared CPM + uplift exactly when a target segment is present;
- the uplift is reported on the bid, and absent (not zero) when none applied;
- the uplifted price is the one compared against the floor;
- the catalog ceiling includes uplift (BR-19);
- the parenting scenario end to end: baseline vs. the request the containers build.
"""
import copy
import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from demand.artfhouse.bids import build
from demand.artfhouse.catalog import AudienceUplift, Campaign, CampaignCatalog, highest_declared_cpm
from demand.artfhouse.eligibility import evaluate
from demand.artfhouse.exclusion import ExclusionReason
from demand.artfhouse.floors import resolve
from demand.artfhouse.service import DemandDecisionService, user_segment_ids

SAMPLES = Path(__file__).resolve().parents[1] / "frontend-react" / "public" / "samples"
CATALOG = CampaignCatalog()
BRIGHTSTART = next(c for c in CATALOG.all() if c.campaign_id == "camp-brightstart")
FAMILYNET = next(c for c in CATALOG.all() if c.campaign_id == "camp-familynet")


def _parenting_request():
    return json.loads((SAMPLES / "parenting-narrative.json").read_text())["bid_request"]


def _with_artf_mutations(bid_request):
    """The request as the ARTF containers leave it for this scenario."""
    out = copy.deepcopy(bid_request)
    out["imp"][0]["pmp"]["deals"] += [
        {"id": "deal-parenting-premium", "bidfloor": 3.40, "at": 1},
        {"id": "deal-family-network", "bidfloor": 2.10, "at": 2},
    ]
    out["user"]["data"].append({
        "id": "artf", "name": "artf",
        "segment": [{"id": s} for s in ("350", "354", "7", "98", "ctx-mobile", "ctx-premium")],
    })
    return out


# ---------------------------------------------------------------- the rule


def test_price_for_is_declared_cpm_without_a_matching_segment():
    assert BRIGHTSTART.price_for(()) == 4.10
    assert BRIGHTSTART.price_for(("seg-hh-parents", "7", "98")) == 4.10


def test_price_for_adds_the_uplift_on_any_target_segment():
    assert BRIGHTSTART.price_for(("350",)) == 4.50
    assert BRIGHTSTART.price_for(("354",)) == 4.50
    assert BRIGHTSTART.price_for(("350", "354")) == 4.50
    assert FAMILYNET.price_for(("98",)) == 3.30


def test_a_campaign_without_uplift_prices_on_content_only():
    remnant = next(c for c in CATALOG.all() if c.campaign_id == "camp-remnant-open")
    assert remnant.audience_uplift is None
    assert remnant.price_for(("350", "354", "98")) == remnant.declared_cpm


@given(st.lists(st.text(min_size=1, max_size=6), max_size=8))
def test_uplift_never_exceeds_the_ceiling_and_never_drops_below_declared(segments):
    price = BRIGHTSTART.price_for(tuple(segments))
    assert BRIGHTSTART.declared_cpm <= price <= BRIGHTSTART.ceiling_cpm


def test_the_ceiling_includes_uplift_for_br19():
    assert BRIGHTSTART.ceiling_cpm == 4.50
    # The video campaign's 13.40 is still the catalog maximum; uplift on a 4.10
    # campaign does not move it, and the test that the ceiling stays under 20
    # (test_artf_demand_bids) still holds.
    assert highest_declared_cpm(CATALOG) == 13.40
    uplifted = CampaignCatalog((
        Campaign("c", "C", "c.example", "cr", 300, 250, 13.00,
                 audience_uplift=AudienceUplift(("1",), 1.00)),
    ))
    assert highest_declared_cpm(uplifted) == 14.00


# ------------------------------------------------------------- the bid


def test_segment_ids_are_read_across_every_provider():
    assert user_segment_ids(_with_artf_mutations(_parenting_request())) == (
        "seg-hh-parents", "seg-int-babies", "350", "354", "7", "98", "ctx-mobile", "ctx-premium",
    )
    assert user_segment_ids({}) == ()
    assert user_segment_ids({"user": {"data": [{"segment": [{"id": 350}]}]}}) == ("350",)


def test_the_bid_reports_the_uplift_that_applied():
    imp = {"id": "imp-1", "bidfloor": 2.60, "pmp": {"deals": [{"id": "deal-parenting-premium", "bidfloor": 3.40}]}}
    candidates = evaluate(imp, CATALOG)
    floors = {c.campaign.campaign_id: resolve(imp, c.deal) for c in candidates}
    result = build("imp-1", candidates, floors, ("350", "7"))
    bid = next(b for b in result.bids if b.campaign_id == "camp-brightstart")
    assert bid.price == 4.50
    assert bid.declared_cpm == 4.10
    assert bid.uplift_cpm == 0.40
    assert bid.uplift_segments == ("350",)


def test_the_bid_reports_no_uplift_when_none_applied():
    imp = {"id": "imp-1", "bidfloor": 2.60, "pmp": {"deals": [{"id": "deal-parenting-premium", "bidfloor": 3.40}]}}
    candidates = evaluate(imp, CATALOG)
    floors = {c.campaign.campaign_id: resolve(imp, c.deal) for c in candidates}
    result = build("imp-1", candidates, floors, ("seg-hh-parents",))
    bid = next(b for b in result.bids if b.campaign_id == "camp-brightstart")
    assert bid.price == 4.10
    assert bid.uplift_cpm is None
    assert bid.uplift_segments == ()


def test_the_uplifted_price_is_what_clears_the_floor():
    # Floor between the declared CPM and the uplifted one: the matched buyer clears,
    # the unmatched one is excluded below_floor.
    imp = {"id": "imp-1", "bidfloor": 2.60, "pmp": {"deals": [{"id": "deal-parenting-premium", "bidfloor": 4.30}]}}
    candidates = evaluate(imp, CATALOG)
    floors = {c.campaign.campaign_id: resolve(imp, c.deal) for c in candidates}
    matched = build("imp-1", candidates, floors, ("354",))
    assert any(b.campaign_id == "camp-brightstart" and b.price == 4.50 for b in matched.bids)
    unmatched = build("imp-1", candidates, floors, ())
    assert any(
        e.campaign_id == "camp-brightstart" and e.reason == ExclusionReason.BELOW_FLOOR
        for e in unmatched.exclusions
    )


# ------------------------------------------------- the scenario, end to end


def test_parenting_baseline_offers_only_the_remnant_deal():
    response = DemandDecisionService().decide(_parenting_request())
    bids = [(b["price"], b.get("dealid")) for s in response["seatbid"] for b in s["bid"]]
    assert bids == [(2.75, "deal-remnant-open")]
    excluded = {e["campaignId"]: e["exclusionReason"] for e in response["ext"]["artf"]["excluded"]}
    assert excluded["camp-brightstart"] == "no_deal_on_impression"
    assert excluded["camp-familynet"] == "no_deal_on_impression"


def test_parenting_with_artf_mutations_offers_the_activated_deals_at_uplifted_prices():
    response = DemandDecisionService().decide(_with_artf_mutations(_parenting_request()))
    bids = {
        b.get("dealid"): (b["price"], b["ext"]["prebid"]["artf"].get("audienceUplift"))
        for s in response["seatbid"] for b in s["bid"]
    }
    assert bids == {
        "deal-parenting-premium": (4.50, {"cpm": 0.40, "segments": ["350", "354"]}),
        "deal-family-network": (3.30, {"cpm": 0.25, "segments": ["98"]}),
        "deal-remnant-open": (2.75, None),
    }
    for seat in response["seatbid"]:
        for bid in seat["bid"]:
            artf = bid["ext"]["prebid"]["artf"]
            assert "declaredCpm" in artf
            if artf.get("audienceUplift") is None:
                assert "audienceUplift" not in artf
