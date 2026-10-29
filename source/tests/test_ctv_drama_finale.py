"""The CTV finale scenario end to end, without Prebid.

The sample request alone leaves Northgate Motors NOT_TARGETED: nothing in the seed
can bid on it. An ADD_CIDS mutation on /imp/imp-1/metric, applied by the real ARTF
applier, adds cid:premium-drama-finale, and the deal campaign becomes the only
eligible candidate at the 10.00 deal floor.
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
DEAL_ID = "deal-ctv-drama-open-auto"

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


def test_add_cids_unlocks_the_deal_campaign_at_the_deal_floor():
    result = artf_applier.apply(_request(), [ADD_CIDS])
    assert [d.applied for d in result.dispositions] == [True], result.dispositions

    imp = result.bid_request["imp"][0]
    candidates = _by_campaign(imp)
    eligible = [cid for cid, c in candidates.items() if c.eligible]
    assert eligible == [CAMPAIGN_ID]

    winner = candidates[CAMPAIGN_ID]
    assert winner.deal_id == DEAL_ID
    floor = floors.resolve(imp, winner.deal)
    assert floor.value == 10.0
    assert floor.clears(winner.campaign.declared_cpm)


def test_the_template_example_path_is_rejected():
    """The template's commented example writes /imp/{id}/ext/cids, which the applier rejects."""
    wrong = ADD_CIDS.model_copy(update={"path": "/imp/imp-1/ext/cids"})
    result = artf_applier.apply(_request(), [wrong])
    assert [d.applied for d in result.dispositions] == [False]
