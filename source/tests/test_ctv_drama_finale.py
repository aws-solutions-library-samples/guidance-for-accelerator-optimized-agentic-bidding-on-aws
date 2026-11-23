"""The CTV finale scenario end to end, without Prebid.

The sample request alone leaves Northgate Motors NOT_TARGETED: nothing in the seed
can bid on it. An ADD_CIDS mutation on /imp/imp-1/metric, applied by the real ARTF
applier, adds cid:premium-drama-finale, and the open-market campaign becomes the
only eligible candidate at the 8.00 impression floor.

The request carries no private marketplace and the campaign holds no deal, so the
outcome does not depend on the Deal Scorer: a suppressed deal cannot take the
campaign out of the auction, and an activated one cannot let it in.
"""
import json
from pathlib import Path

from demand.artfhouse import floors
from demand.artfhouse.catalog import CampaignCatalog
from demand.artfhouse.eligibility import evaluate
from demand.artfhouse.exclusion import ExclusionReason
from shared import artf_applier
from shared.artf_types import AddMetricsPayload, Intent, Metric, Mutation, Operation

SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "frontend-react" / "public" / "samples" / "ctv-drama-finale.json"
)
CAMPAIGN_ID = "camp-ctv-drama-auto"

ADD_CIDS = Mutation(
    intent=Intent.ADD_CIDS,
    op=Operation.ADD,
    path="/imp/imp-1/metric",
    add_metrics=AddMetricsPayload(metric=[
        Metric(type="cid:iris_8a92b23a77", value=1.0, vendor="iris.tv"),
        Metric(type="cid:premium-drama-finale", value=1.0, vendor="iris.tv"),
    ]),
)


def _request():
    return json.loads(SAMPLE.read_text())["bid_request"]


def _by_campaign(imp):
    return {c.campaign.campaign_id: c for c in evaluate(imp, CampaignCatalog())}


def test_without_content_ids_nothing_in_the_seed_is_eligible():
    candidates = _by_campaign(_request()["imp"][0])
    assert candidates[CAMPAIGN_ID].excluded_because is ExclusionReason.NOT_TARGETED
    assert [cid for cid, c in candidates.items() if c.eligible] == []


def test_the_request_offers_no_deal_and_the_campaign_holds_none():
    imp = _request()["imp"][0]
    assert "pmp" not in imp
    campaign = _by_campaign(imp)[CAMPAIGN_ID].campaign
    assert campaign.open_market and campaign.deal_ids == ()


def test_add_cids_unlocks_the_open_market_campaign_at_the_impression_floor():
    result = artf_applier.apply(_request(), [ADD_CIDS])
    assert [d.applied for d in result.dispositions] == [True], result.dispositions

    imp = result.bid_request["imp"][0]
    candidates = _by_campaign(imp)
    eligible = [cid for cid, c in candidates.items() if c.eligible]
    assert eligible == [CAMPAIGN_ID]

    winner = candidates[CAMPAIGN_ID]
    assert winner.deal_id is None
    floor = floors.resolve(imp, winner.deal)
    assert floor.value == 8.0
    assert floor.clears(winner.campaign.declared_cpm)


def test_the_template_example_path_is_rejected():
    """The template's commented example writes /imp/{id}/ext/cids, which the applier rejects."""
    wrong = ADD_CIDS.model_copy(update={"path": "/imp/imp-1/ext/cids"})
    result = artf_applier.apply(_request(), [wrong])
    assert [d.applied for d in result.dispositions] == [False]
