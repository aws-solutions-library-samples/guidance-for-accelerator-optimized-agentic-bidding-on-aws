"""The two status fields the control plane writes, as the readers see them.

Campaign ``status`` is read by the demand service; deal ``sellerstatus`` is read by
both the demand service and shared/deal_library.py (the Deal Scorer and the
orchestrator). The rule in every reader is the same: absence means active, so a
table written before the fields existed behaves exactly as before, and only an
explicit paused/non-zero value takes a record off the market.

Also covers the seeder's two new switches: ``--campaign-status`` and
``--skip-managed``.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import os
import sys

import pytest

from demand.artfhouse.catalog import CampaignCatalog, campaign_status, campaigns_from_records
from demand.artfhouse.eligibility import evaluate
from demand.artfhouse.exclusion import ExclusionReason
from shared import deal_library

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
_SEEDER = os.path.join(_REPO, "deployment", "scripts", "seed_deals_table.py")


@pytest.fixture(scope="module")
def seed():
    with open(os.environ["DEALS_SEED_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def seeder():
    spec = importlib.util.spec_from_file_location("seed_deals_table_status_tests", _SEEDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def imp_with(deals=()):
    return {"id": "imp-1", "bidfloor": 1.0, "banner": {"w": 300, "h": 250}, "pmp": {"deals": list(deals)}}


def by_campaign(candidates):
    return {c.campaign.campaign_id: c for c in candidates}


# ------------------------------------------------------------ campaign status


def test_a_record_without_status_is_active():
    assert campaign_status({}) == "active"
    assert campaign_status({"status": "active"}) == "active"
    assert campaign_status({"status": "paused"}) == "paused"


@pytest.mark.parametrize("odd", ["PAUSED", "Paused", "", None, 0, "stopped"])
def test_only_an_exact_paused_silences_a_campaign(odd):
    # The failure mode to avoid is a typo in the table silencing demand. Anything
    # that is not the one word is treated as active and will show up bidding.
    assert campaign_status({"status": odd}) == "active"


def test_campaigns_from_records_carry_status(seed):
    records = copy.deepcopy(seed["campaigns"])
    records[0]["status"] = "paused"
    campaigns = campaigns_from_records(records)
    assert campaigns[0].paused is True
    assert all(not c.paused for c in campaigns[1:])


def test_a_paused_campaign_is_considered_and_declines_with_that_reason(seed):
    records = copy.deepcopy(seed["campaigns"])
    cedar = next(r for r in records if r["campaign_id"] == "camp-cedar")
    cedar["status"] = "paused"
    catalog = CampaignCatalog(campaigns_from_records(records))

    candidates = by_campaign(evaluate(imp_with([{"id": "deal-home-premium"}]), catalog))
    assert len(candidates) == len(records), "a paused campaign must still be listed"
    assert candidates["camp-cedar"].excluded_because is ExclusionReason.CAMPAIGN_PAUSED
    assert candidates["camp-cedar"].deal is None


def test_pausing_is_the_only_change_between_the_two_catalogs(seed):
    # With every campaign active the decision is exactly what it was before the
    # field existed: no row changes outcome or reason.
    before = evaluate(imp_with([{"id": "deal-home-premium"}]), CampaignCatalog(campaigns_from_records(seed["campaigns"])))
    stamped = [dict(r, status="active") for r in seed["campaigns"]]
    after = evaluate(imp_with([{"id": "deal-home-premium"}]), CampaignCatalog(campaigns_from_records(stamped)))
    assert [(c.campaign.campaign_id, c.excluded_because, c.deal_id) for c in before] == \
        [(c.campaign.campaign_id, c.excluded_because, c.deal_id) for c in after]


# -------------------------------------------------------- deal sellerstatus


def test_an_inactive_deal_excludes_the_holding_campaign(seed):
    catalog = CampaignCatalog(campaigns_from_records(seed["campaigns"]), inactive_deal_ids={"deal-home-premium"})
    candidates = by_campaign(evaluate(imp_with([{"id": "deal-home-premium"}]), catalog))
    assert candidates["camp-cedar"].excluded_because is ExclusionReason.DEAL_INACTIVE
    assert candidates["camp-cedar"].deal_id == "deal-home-premium"


def test_inactive_beats_suppressed_as_the_reason(seed):
    deal = {"id": "deal-home-premium", "ext": {"suppressed": True}}
    catalog = CampaignCatalog(campaigns_from_records(seed["campaigns"]), inactive_deal_ids={"deal-home-premium"})
    candidates = by_campaign(evaluate(imp_with([deal]), catalog))
    assert candidates["camp-cedar"].excluded_because is ExclusionReason.DEAL_INACTIVE


def test_a_campaign_with_one_active_and_one_inactive_deal_transacts_on_the_active_one(seed):
    cedar = next(r for r in seed["campaigns"] if r["campaign_id"] == "camp-cedar")
    assert len(cedar["deal_ids"]) >= 2, "the fixture relies on camp-cedar holding two deals"
    first, second = cedar["deal_ids"][:2]
    catalog = CampaignCatalog(campaigns_from_records(seed["campaigns"]), inactive_deal_ids={first})
    candidates = by_campaign(evaluate(imp_with([{"id": first}, {"id": second}]), catalog))
    assert candidates["camp-cedar"].eligible
    assert candidates["camp-cedar"].deal_id == second


def test_deal_library_treats_missing_sellerstatus_as_active():
    assert deal_library.deal_is_active({"id": "d"}) is True
    assert deal_library.deal_is_active({"id": "d", "sellerstatus": 0}) is True
    assert deal_library.deal_is_active({"id": "d", "sellerstatus": "0"}) is True


@pytest.mark.parametrize("status", [1, 2, 3, 4, "1", "paused", None, 1.5])
def test_deal_library_treats_anything_but_zero_as_inactive(status):
    assert deal_library.deal_is_active({"id": "d", "sellerstatus": status}) is False


def test_deal_book_keeps_inactive_deals_out_of_the_request_path(seed):
    records = copy.deepcopy(seed["deals"])
    target = records[0]
    target["sellerstatus"] = 1
    book = deal_library.DealBook(records)

    assert len(book) == len(records), "the book still describes the whole table"
    assert target["id"] in book.all_deal_ids()
    assert target["id"] in book.inactive_deal_ids()
    assert book.by_id(target["id"]) is None

    domain = target["inventory"]["sitedomains"][0]
    offered = {d.id for d in book.deals_for({"site": {"domain": domain}})}
    assert target["id"] not in offered


def test_deal_book_is_unchanged_when_every_deal_is_active(seed):
    plain = deal_library.DealBook(copy.deepcopy(seed["deals"]))
    stamped = deal_library.DealBook([dict(d, sellerstatus=0) for d in seed["deals"]])
    for deal in seed["deals"]:
        domain = deal["inventory"].get("sitedomains", [None])[0]
        if domain:
            assert [d.id for d in plain.deals_for({"site": {"domain": domain}})] == \
                [d.id for d in stamped.deals_for({"site": {"domain": domain}})]
    assert plain.inactive_deal_ids() == stamped.inactive_deal_ids() == frozenset()


def test_storage_keys_include_the_control_plane_marks():
    # The control plane stamps every item it writes. Readers must strip those marks
    # so a deal record stays a pure Deals API object.
    for key in ("managed_by", "updated_at"):
        assert key in deal_library.STORAGE_KEYS
    from demand.artfhouse import deals_source
    for key in ("managed_by", "updated_at"):
        assert key in deals_source.STORAGE_KEYS


# ------------------------------------------------------------------- seeder


def test_items_stamp_the_requested_campaign_status(seeder, seed):
    paused = [i for i in seeder.items(seed, "prebid.example", "paused") if i["record_type"] == "campaign"]
    active = [i for i in seeder.items(seed, "prebid.example") if i["record_type"] == "campaign"]
    assert paused and all(i["status"] == "paused" for i in paused)
    assert active and all(i["status"] == "active" for i in active)
    deals = [i for i in seeder.items(seed, "prebid.example", "paused") if i["record_type"] == "deal"]
    assert all("status" not in i for i in deals), "status is a campaign field, not a deal field"


def test_items_reject_an_unknown_campaign_status(seeder, seed):
    with pytest.raises(seeder.SeedError):
        list(seeder.items(seed, "prebid.example", "stopped"))


class _Conditional(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def test_write_unmanaged_skips_items_the_control_plane_owns(seeder, seed):
    managed = {("campaign", "camp-cedar"), ("deal", "deal-home-premium")}
    puts = []

    class _Table:
        def put_item(self, Item, ConditionExpression=None):
            assert ConditionExpression == "attribute_not_exists(managed_by)"
            puts.append((Item["record_type"], Item["id"]))
            if (Item["record_type"], Item["id"]) in managed:
                raise _Conditional("ConditionalCheckFailedException")

    rows = list(seeder.items(seed, "prebid.example", "paused"))
    written, skipped = seeder.write_unmanaged(_Table(), rows)
    assert skipped == 2
    assert written == len(rows) - 2
    assert len(puts) == len(rows), "every row is attempted; the table decides"


def test_write_unmanaged_raises_on_anything_but_the_condition(seeder, seed):
    class _Table:
        def put_item(self, Item, ConditionExpression=None):
            raise _Conditional("ProvisionedThroughputExceededException")

    with pytest.raises(_Conditional):
        seeder.write_unmanaged(_Table(), list(seeder.items(seed, "prebid.example")))


def test_main_passes_status_and_skip_managed_through(seeder, seed, tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_once(args, loaded, origin):
        seen["status"] = args.campaign_status
        seen["skip"] = args.skip_managed
        return (40, 5)

    monkeypatch.setattr(seeder, "_seed_once", fake_once)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed))
    code = seeder.main([
        "--table", "t", "--role-arn", "arn:aws:iam::000000000000:role/x", "--seed", str(path),
        "--region", "us-east-1", "--campaign-status", "paused", "--skip-managed",
    ])
    assert code == 0
    assert seen == {"status": "paused", "skip": True}
    out = capsys.readouterr().out
    assert "40 items written, 5 left to the control plane" in out
    assert "campaigns paused" in out
