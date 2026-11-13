"""Content identifier targeting: imp.metric[].type values with the cid: prefix."""
from hypothesis import given, settings
from hypothesis import strategies as st

from demand.artfhouse.catalog import Campaign, CampaignCatalog, campaigns_from_records
from demand.artfhouse.eligibility import evaluate
from demand.artfhouse.exclusion import ExclusionReason

TARGET = "cid:premium-drama-finale"
DEAL = {"id": "deal-ctv-drama-open-auto", "bidfloor": 10.0, "at": 1}


def _campaign(**overrides):
    fields = dict(
        campaign_id="camp-ctv-drama-auto",
        campaign_name="Northgate Motors Finale PMP",
        adomain="northgatemotors.example",
        creative_id="cr-northgate-1280x720",
        width=1280,
        height=720,
        declared_cpm=18.5,
        deal_ids=("deal-ctv-drama-open-auto",),
        open_market=False,
        media_type="video",
        duration=30,
        target_content_ids=(TARGET,),
    )
    fields.update(overrides)
    return Campaign(**fields)


def _imp(metric=None, deals=(DEAL,)):
    imp = {
        "id": "imp-1",
        "video": {"mimes": ["video/mp4"], "minduration": 15, "maxduration": 30, "w": 1280, "h": 720},
        "bidfloor": 8.0,
        "pmp": {"private_auction": 0, "deals": [dict(d) for d in deals]},
    }
    if metric is not None:
        imp["metric"] = metric
    return imp


def _only(imp, campaign):
    (candidate,) = evaluate(imp, CampaignCatalog([campaign]))
    return candidate


def test_a_matching_content_id_makes_the_deal_campaign_eligible():
    c = _only(_imp([{"type": TARGET, "value": 1.0, "vendor": "iris.tv"}]), _campaign())
    assert c.eligible and c.deal_id == "deal-ctv-drama-open-auto"


def test_a_different_content_id_is_not_targeted():
    c = _only(_imp([{"type": "cid:other-show", "value": 1.0}]), _campaign())
    assert c.excluded_because is ExclusionReason.NOT_TARGETED


def test_an_absent_content_id_is_not_targeted():
    assert _only(_imp(), _campaign()).excluded_because is ExclusionReason.NOT_TARGETED


def test_metrics_without_the_cid_prefix_do_not_count():
    imp = _imp([{"type": "viewability", "value": 0.82}, {"type": "premium-drama-finale", "value": 1.0}])
    assert _only(imp, _campaign()).excluded_because is ExclusionReason.NOT_TARGETED


def test_malformed_metric_entries_are_ignored():
    imp = _imp(["cid:premium-drama-finale", None, {"type": 7, "value": 1.0}])
    assert _only(imp, _campaign()).excluded_because is ExclusionReason.NOT_TARGETED


def test_the_rule_applies_on_the_open_market_branch():
    campaign = _campaign(deal_ids=(), open_market=True)
    assert _only(_imp([{"type": TARGET, "value": 1.0}], deals=()), campaign).eligible
    assert _only(_imp(deals=()), campaign).excluded_because is ExclusionReason.NOT_TARGETED


def test_a_campaign_without_content_targets_ignores_the_signal():
    campaign = _campaign(target_content_ids=())
    assert _only(_imp(), campaign).eligible
    assert _only(_imp([{"type": "cid:other-show", "value": 1.0}]), campaign).eligible


def test_records_without_the_field_load_with_no_content_targets():
    (campaign,) = campaigns_from_records([{
        "campaign_id": "c", "campaign_name": "n", "adomain": "a.example", "creative_id": "cr",
        "width": 300, "height": 250, "declared_cpm": 1.0,
    }])
    assert campaign.target_content_ids == ()


def test_records_with_the_field_load_it_as_a_tuple():
    (campaign,) = campaigns_from_records([{
        "campaign_id": "c", "campaign_name": "n", "adomain": "a.example", "creative_id": "cr",
        "width": 1280, "height": 720, "declared_cpm": 18.5, "target_content_ids": [TARGET],
    }])
    assert campaign.target_content_ids == (TARGET,)


metric_types = st.one_of(
    st.just(TARGET),
    st.text(min_size=0, max_size=24),
    st.text(min_size=1, max_size=20).map(lambda s: "cid:" + s),
)


@settings(max_examples=200, derandomize=True)
@given(st.lists(st.fixed_dictionaries({"type": metric_types, "value": st.floats(0, 1)}), max_size=6))
def test_property_eligible_exactly_when_the_target_id_is_present(metrics):
    c = _only(_imp(metrics), _campaign())
    assert c.eligible == any(m["type"] == TARGET for m in metrics)
    if not c.eligible:
        assert c.excluded_because is ExclusionReason.NOT_TARGETED
