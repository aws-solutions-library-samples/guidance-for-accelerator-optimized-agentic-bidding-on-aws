"""The demand catalog's record source: the deals table, or the seed file in tests.

No test here reaches DynamoDB. The table path is exercised with a stub Table
object that returns two pages the way boto3's Table.query does.
"""
import json
import os
from decimal import Decimal

import pytest

from demand.artfhouse import deals_source
from demand.artfhouse.catalog import CampaignCatalog, campaigns_from_records


class _PagedTable:
    """Answers query() with two pages, recording each call's arguments."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self._pages[len(self.calls) - 1]


def _item(campaign_id, seq, cpm):
    return {
        "record_type": "campaign", "id": campaign_id, "seq": Decimal(seq),
        "campaign_id": campaign_id, "campaign_name": campaign_id.title(),
        "adomain": f"{campaign_id}.example", "creative_id": f"cr-{campaign_id}",
        "width": Decimal("320"), "height": Decimal("480"),
        "declared_cpm": Decimal(cpm), "deal_ids": [f"deal-{campaign_id}"],
        "open_market": False, "target_categories": ["travel"], "media_type": "banner",
        "duration": None, "audience_uplift": {"segment_ids": ["719"], "cpm": Decimal("0.60")},
    }


def test_the_table_path_reads_every_page_in_seed_order_and_strips_storage_keys():
    table = _PagedTable([
        {"Items": [_item("b", 1, "12.25")], "LastEvaluatedKey": {"record_type": "campaign", "id": "b"}},
        {"Items": [_item("a", 0, "12.70")]},
    ])
    records = deals_source.query_records(table, "campaign")
    assert len(table.calls) == 2
    assert table.calls[1]["ExclusiveStartKey"] == {"record_type": "campaign", "id": "b"}
    assert [r["campaign_id"] for r in records] == ["a", "b"]
    assert all("record_type" not in r and "seq" not in r for r in records)


def test_decimals_become_the_literals_the_catalog_compares_against():
    table = _PagedTable([{"Items": [_item("a", 0, "12.70")]}])
    (record,) = deals_source.query_records(table, "campaign")
    assert record["declared_cpm"] == 12.70 and isinstance(record["declared_cpm"], float)
    assert record["width"] == 320 and isinstance(record["width"], int)
    assert record["audience_uplift"]["cpm"] == 0.60
    (campaign,) = campaigns_from_records([record])
    assert campaign.ceiling_cpm == 13.30


@pytest.mark.parametrize("table,seed", [("", ""), ("t-deals", "/tmp/seed.json")])
def test_exactly_one_source_must_be_configured(monkeypatch, table, seed):
    monkeypatch.setenv(deals_source.ENV_TABLE, table)
    monkeypatch.setenv(deals_source.ENV_SEED_FILE, seed)
    with pytest.raises(ValueError, match="DEALS_TABLE.*DEALS_SEED_FILE"):
        deals_source.configured_source()


def test_the_table_source_is_used_when_configured(monkeypatch):
    table = _PagedTable([{"Items": [_item("a", 0, "1.00")]}])
    monkeypatch.setenv(deals_source.ENV_TABLE, "stack-deals")
    monkeypatch.delenv(deals_source.ENV_SEED_FILE, raising=False)
    names = []
    monkeypatch.setattr(deals_source, "_table", lambda name: names.append(name) or table)
    assert [r["campaign_id"] for r in deals_source.campaign_records()] == ["a"]
    assert names == ["stack-deals"]


def test_the_default_catalog_is_every_seed_campaign_in_seed_order():
    with open(os.environ[deals_source.ENV_SEED_FILE], encoding="utf-8") as handle:
        seed = json.load(handle)
    campaigns = CampaignCatalog().all()
    assert campaigns == campaigns_from_records(seed["campaigns"])
    assert [c.campaign_id for c in campaigns] == [r["campaign_id"] for r in seed["campaigns"]]


def test_one_campaign_per_deal_still_raises():
    record = deals_source.campaign_records()[0]
    twin = dict(record, campaign_id="camp-twin")
    with pytest.raises(ValueError, match="claimed by both"):
        CampaignCatalog(campaigns_from_records([record, twin]))
