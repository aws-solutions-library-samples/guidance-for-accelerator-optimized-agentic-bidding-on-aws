"""The catalog against the scenarios the product actually ships.

WHY THIS FILE EXISTS

For a long time none of the shipped scenarios carried a deal id the catalog held:
the samples use deal-premium-auto, deal-parenting-premium, deal-premium-video and
so on, while the catalog held deal-home-premium, deal-retail-run, deal-auto-brand
and deal-finance-pmp. Nothing failed. The demo simply showed this seat offering
camp-openfield at 2.05 open-market and losing to the simulator on every scenario,
and the deal path -- the point of the ARTF story -- never ran outside a hand-built
request.

A unit test over the catalog alone could not have caught that, because the catalog
was internally consistent. These tests read the SCENARIO FILES and check the
catalog against them, so the two cannot drift apart again silently.

They also check the prices clear the binding floor. A campaign whose deal exists
but whose CPM is under the floor is excluded, which looks the same on screen as no
campaign at all.
"""

import json
from pathlib import Path

import pytest

from demand.artfhouse.catalog import CampaignCatalog
from demand.artfhouse.exclusion import ExclusionReason
from demand.artfhouse.service import DemandDecisionService

SAMPLES = (
    Path(__file__).resolve().parents[2] / "source" / "frontend-react" / "public" / "samples"
)
if not SAMPLES.is_dir():  # running from inside source/
    SAMPLES = Path(__file__).resolve().parents[1] / "frontend-react" / "public" / "samples"

CATALOG = CampaignCatalog()
SERVICE = DemandDecisionService()


def scenario_files():
    return sorted(SAMPLES.glob("*.json"))


def deal_slots():
    """Every (scenario, imp, deal) the shipped samples declare."""
    out = []
    for path in scenario_files():
        payload = json.loads(path.read_text())
        for imp in (payload.get("bid_request") or {}).get("imp") or []:
            for deal in (imp.get("pmp") or {}).get("deals") or []:
                out.append((path.name, imp, deal))
    return out


def test_the_samples_directory_was_found():
    # Guards the path arithmetic above: an empty glob would make every test below
    # pass by vacuity.
    assert scenario_files(), f"no scenario files under {SAMPLES}"
    assert deal_slots(), "no deals found in any scenario"


@pytest.mark.parametrize("scenario,imp,deal", deal_slots(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_scenario_deal_has_a_campaign(scenario, imp, deal):
    deal_id = deal.get("id")
    campaign = CATALOG.by_deal(deal_id)
    assert campaign is not None, (
        f"{scenario} offers {deal_id} and no campaign transacts on it, so this seat "
        f"can never bid that deal"
    )


@pytest.mark.parametrize("scenario,imp,deal", deal_slots(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_scenario_deal_campaign_clears_the_binding_floor(scenario, imp, deal):
    campaign = CATALOG.by_deal(deal.get("id"))
    assert campaign is not None

    # The binding floor is the HIGHER of the impression's floor and the deal's --
    # the same rule floors.resolve applies.
    imp_floor = float(imp.get("bidfloor") or 0.0)
    deal_floor = float(deal.get("bidfloor") or 0.0)
    binding = max(imp_floor, deal_floor)

    assert campaign.declared_cpm >= binding, (
        f"{scenario}: {campaign.campaign_id} declares {campaign.declared_cpm} but the "
        f"binding floor is {binding} (imp {imp_floor}, deal {deal_floor}), so it is "
        f"excluded below_floor and never appears"
    )


@pytest.mark.parametrize("scenario,imp,deal", deal_slots(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_scenario_deal_campaign_can_be_placed_in_the_slot(scenario, imp, deal):
    campaign = CATALOG.by_deal(deal.get("id"))
    assert campaign is not None

    slots = [k for k in ("banner", "video", "audio", "native") if imp.get(k)]
    assert campaign.media_type in slots, (
        f"{scenario}: {campaign.campaign_id} is {campaign.media_type} but the "
        f"impression offers {slots}, so its creative could never render"
    )


@pytest.mark.parametrize("scenario", [p.name for p in scenario_files()])
def test_each_scenario_with_deals_produces_at_least_one_deal_bid(scenario):
    """End to end over the demand service, not just the catalog.

    This is the assertion that would have failed before: every deal-bearing
    scenario must yield at least one bid carrying a dealid.
    """
    payload = json.loads((SAMPLES / scenario).read_text())
    bid_request = payload.get("bid_request") or {}

    has_deals = any(
        (imp.get("pmp") or {}).get("deals") for imp in bid_request.get("imp") or []
    )
    if not has_deals:
        pytest.skip(f"{scenario} declares no deals")

    response = SERVICE.decide(bid_request)
    deal_bids = [
        bid
        for seat in response.get("seatbid") or []
        for bid in seat.get("bid") or []
        if bid.get("dealid")
    ]
    assert deal_bids, (
        f"{scenario} carries deals but this seat produced no deal bid; "
        f"exclusions: "
        + ", ".join(
            f"{e.get('campaignId')}={e.get('exclusionReason')}"
            for e in (response.get("ext", {}).get("artf", {}).get("excluded") or [])
        )
    )


def test_video_slots_are_not_offered_a_banner_creative():
    """The rule that makes the video scenarios honest.

    Before media_type existed every campaign was implicitly a banner and offered on
    anything, so a 300x250 banner was bid into a 640x480 video slot -- an offer that
    could never have rendered.
    """
    video_imp = {
        "id": "imp-1",
        "bidfloor": 8.0,
        "video": {"w": 640, "h": 480, "minduration": 15, "maxduration": 30},
        "pmp": {"deals": [{"id": "deal-premium-video", "bidfloor": 10.0}]},
    }
    response = SERVICE.decide({"id": "r", "imp": [video_imp]})

    for seat in response.get("seatbid") or []:
        for bid in seat.get("bid") or []:
            campaign = next(
                c for c in CATALOG.all()
                if c.campaign_id == bid["ext"]["prebid"]["artf"]["campaignId"]
            )
            assert campaign.media_type == "video", (
                f"{campaign.campaign_id} is {campaign.media_type} and was bid into a "
                f"video-only slot"
            )

    excluded = response.get("ext", {}).get("artf", {}).get("excluded") or []
    reasons = {e.get("exclusionReason") for e in excluded}
    assert ExclusionReason.MEDIA_TYPE_UNSUPPORTED.value in reasons


def test_a_video_bid_declares_its_media_type_and_duration():
    video_imp = {
        "id": "imp-1",
        "bidfloor": 8.0,
        "video": {"w": 640, "h": 480, "minduration": 15, "maxduration": 30},
        "pmp": {"deals": [{"id": "deal-premium-video", "bidfloor": 10.0}]},
    }
    response = SERVICE.decide({"id": "r", "imp": [video_imp]})
    bids = [b for s in response.get("seatbid") or [] for b in s.get("bid") or []]
    assert bids

    for bid in bids:
        # OpenRTB 2.6: 2 is video. Without it the adapter has to guess one type for
        # every bid, and a video bid reported as a banner cannot render.
        assert bid["mtype"] == 2
        assert isinstance(bid["dur"], int)


def test_a_banner_bid_declares_banner_and_omits_duration():
    banner_imp = {
        "id": "imp-1",
        "bidfloor": 1.0,
        "banner": {"w": 300, "h": 250},
        "pmp": {"deals": [{"id": "deal-home-premium", "bidfloor": 5.0}]},
    }
    response = SERVICE.decide({"id": "r", "imp": [banner_imp]})
    bids = [b for s in response.get("seatbid") or [] for b in s.get("bid") or []]
    assert bids

    for bid in bids:
        assert bid["mtype"] == 1
        assert "dur" not in bid


def test_no_two_campaigns_claim_the_same_deal():
    # CampaignCatalog raises on construction if they do; this states the property
    # explicitly so the guard is not silently removed.
    seen = {}
    for campaign in CATALOG.all():
        for deal_id in campaign.deal_ids:
            assert deal_id not in seen, (
                f"{deal_id} claimed by both {seen[deal_id]} and {campaign.campaign_id}"
            )
            seen[deal_id] = campaign.campaign_id
