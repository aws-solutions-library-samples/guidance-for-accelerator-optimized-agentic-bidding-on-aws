"""The deals seed file against the Deals API v1.0 schema and its own rules.

Imports deployment/scripts/seed_deals_table.py by path and runs its validator
over deployment/seed/deals_and_campaigns.json. No AWS call is made: the write
path is driven with a stub table.
"""
import copy
import importlib.util
import json
import os
import sys
from decimal import Decimal

import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
_SCRIPT = os.path.join(_REPO, "deployment", "scripts", "seed_deals_table.py")


@pytest.fixture(scope="module")
def seeder():
    spec = importlib.util.spec_from_file_location("seed_deals_table_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seed():
    with open(os.environ["DEALS_SEED_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


def test_the_seed_validates(seeder, seed):
    seeder.validate(seed)


def test_the_seed_declares_its_format_and_the_deals_api_version(seed):
    assert seed["format"] == "artf-deals-seed/1"
    assert seed["dealsapi"] == "1.0"


def test_every_deal_carries_the_required_deals_api_fields(seed):
    for deal in seed["deals"]:
        assert {"id", "origin", "terms"} <= set(deal)


@pytest.mark.parametrize("path,value", [
    (("sellerstatus",), 5),
    (("adtypes",), [5]),
    (("adtypes",), []),
    (("auxdata",), 5),
    (("pubcount",), 3),
    (("dinventory",), 3),
    (("terms", "pricetype"), 4),
    (("terms", "pricetype"), 499),
    (("terms", "guar"), 2),
    (("inventory", "inclinventory"), [7]),
    (("sellerstatus",), True),
])
def test_an_out_of_range_enum_is_rejected(seeder, seed, path, value):
    broken = copy.deepcopy(seed)
    target = broken["deals"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(seeder.SeedError):
        seeder.validate(broken)


def test_an_exchange_specific_pricetype_is_allowed(seeder, seed):
    seed["deals"][0]["terms"]["pricetype"] = 500
    seeder.validate(seed)


@pytest.mark.parametrize("field", ["id", "origin", "terms"])
def test_a_missing_required_field_is_rejected(seeder, seed, field):
    del seed["deals"][0][field]
    with pytest.raises(seeder.SeedError, match=field):
        seeder.validate(seed)


def test_a_field_outside_the_deals_api_is_rejected(seeder, seed):
    seed["deals"][0]["bidfloor"] = 1.0
    with pytest.raises(seeder.SeedError, match="not Deals API Deal fields"):
        seeder.validate(seed)


def test_a_terms_extension_other_than_the_minimum_is_rejected(seeder, seed):
    seed["deals"][0]["terms"]["ext"] = {"maxdealfloor": 9.0}
    with pytest.raises(seeder.SeedError, match="terms.ext"):
        seeder.validate(seed)


def test_a_minimum_above_the_deal_floor_is_rejected(seeder, seed):
    deal = seed["deals"][0]
    deal["terms"]["ext"] = {"mindealfloor": deal["terms"]["dealfloor"] + 1}
    with pytest.raises(seeder.SeedError, match="above terms.dealfloor"):
        seeder.validate(seed)


def test_a_duplicate_deal_id_is_rejected(seeder, seed):
    seed["deals"].append(copy.deepcopy(seed["deals"][0]))
    with pytest.raises(seeder.SeedError, match="duplicate deal id"):
        seeder.validate(seed)


def test_a_campaign_holding_an_unknown_deal_is_rejected(seeder, seed):
    seed["campaigns"][0]["deal_ids"].append("deal-nowhere")
    with pytest.raises(seeder.SeedError, match="not a deal in the seed"):
        seeder.validate(seed)


def test_a_deal_held_by_two_campaigns_is_rejected(seeder, seed):
    seed["campaigns"][1]["deal_ids"].append(seed["campaigns"][0]["deal_ids"][0])
    with pytest.raises(seeder.SeedError, match="already held"):
        seeder.validate(seed)


def test_every_deal_is_held_by_a_campaign(seed):
    held = {d for c in seed["campaigns"] for d in c["deal_ids"]}
    assert held == {d["id"] for d in seed["deals"]}


def test_campaign_ids_are_unique(seed):
    ids = [c["campaign_id"] for c in seed["campaigns"]]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- origin


def test_every_deal_stores_the_origin_template(seeder, seed):
    assert {d["origin"] for d in seed["deals"]} == {seeder.ORIGIN_TEMPLATE}
    assert seeder.ORIGIN_TEMPLATE == "{stackprefix}-prebid.example"


def test_the_origin_is_rendered_from_the_deploy_prefix(seeder):
    assert seeder.render_origin("tr3") == "tr3-prebid.example"
    assert seeder.render_origin("TR3") == "tr3-prebid.example"
    assert seeder.render_origin("") == "prebid.example"


@pytest.mark.parametrize("prefix", ["-bad", "a_b", "a.b", "x" * 60])
def test_an_invalid_prefix_raises(seeder, prefix):
    with pytest.raises(seeder.SeedError):
        seeder.render_origin(prefix)


# ------------------------------------------------- one publisher per deal


def test_every_deal_belongs_to_one_publisher(seed):
    for deal in seed["deals"]:
        inventory = deal["inventory"]
        assert deal["pubcount"] == 1 and deal["dinventory"] == 1
        places = list(inventory.get("sitedomains", [])) + list(inventory.get("appbundles", []))
        assert len(places) == 1, deal["id"]
        if "sitedomains" in inventory:
            assert deal["seller"] == inventory["sitedomains"][0]
            assert inventory["inclinventory"] == [1]
        else:
            assert deal["seller"] == "ridesharer.example"
            assert inventory["inclinventory"] == [2]


def test_adtypes_match_the_holding_campaigns_media_type(seed):
    media = {d: c["media_type"] for c in seed["campaigns"] for d in c["deal_ids"]}
    for deal in seed["deals"]:
        assert deal["adtypes"] == ([1] if media[deal["id"]] == "banner" else [2]), deal["id"]


# ------------------------------------------------- records the catalog reads


def test_every_campaign_record_loads_into_the_catalog_in_seed_order(seed):
    from demand.artfhouse.catalog import CampaignCatalog, campaigns_from_records

    loaded = CampaignCatalog(campaigns_from_records(seed["campaigns"])).all()
    assert [c.campaign_id for c in loaded] == [r["campaign_id"] for r in seed["campaigns"]]


def test_every_deal_carries_a_positive_usd_floor_and_a_deals_api_pricetype(seed):
    for deal in seed["deals"]:
        terms = deal["terms"]
        assert terms["dealfloor"] > 0, deal["id"]
        assert terms["cur"] == "USD", deal["id"]
        assert terms["pricetype"] in (1, 2, 3), deal["id"]


def test_every_campaign_carries_a_positive_price_and_a_renderable_creative(seed):
    for campaign in seed["campaigns"]:
        assert campaign["declared_cpm"] > 0, campaign["campaign_id"]
        assert campaign["width"] > 0 and campaign["height"] > 0, campaign["campaign_id"]
        assert campaign["media_type"] in ("banner", "video"), campaign["campaign_id"]
        if campaign["media_type"] == "video":
            assert campaign["duration"] and campaign["duration"] > 0, campaign["campaign_id"]
        else:
            assert campaign["duration"] is None, campaign["campaign_id"]
        uplift = campaign["audience_uplift"]
        if uplift is not None:
            assert uplift["cpm"] > 0 and uplift["segment_ids"], campaign["campaign_id"]


def test_only_the_guaranteed_deal_is_guaranteed(seed):
    assert [d["id"] for d in seed["deals"] if d["terms"]["guar"] == 1] == ["deal-guaranteed-premium"]


def test_the_catalog_maximum_is_the_highest_seed_ceiling_and_stays_under_the_top_bucket(seed):
    from demand.artfhouse.catalog import CampaignCatalog, campaigns_from_records, highest_declared_cpm

    ceilings = [
        c["declared_cpm"] + ((c["audience_uplift"] or {}).get("cpm") or 0.0)
        for c in seed["campaigns"]
    ]
    catalog = CampaignCatalog(campaigns_from_records(seed["campaigns"]))
    assert highest_declared_cpm(catalog) == pytest.approx(max(ceilings))
    # Prebid's default price granularity tops out at 20; above it hb_pb is clamped.
    assert max(ceilings) < 20.0


# ------------------------------------------------------------ the minimum


def test_only_the_midday_deals_carry_a_minimum_of_five(seed):
    with_minimum = {
        d["id"]: d["terms"]["ext"]["mindealfloor"]
        for d in seed["deals"] if "ext" in d["terms"]
    }
    assert with_minimum == {
        "deal-ridesharer-eats-open-kitchen": 5.0,
        "deal-ridesharer-eats-open-noodles": 5.0,
    }


# --------------------------------------------------------------- writing


def test_items_add_record_type_and_seq_and_render_the_origin(seeder, seed):
    rows = list(seeder.items(seed, "tr3-prebid.example"))
    assert len(rows) == len(seed["deals"]) + len(seed["campaigns"])
    deals = [r for r in rows if r["record_type"] == "deal"]
    campaigns = [r for r in rows if r["record_type"] == "campaign"]
    assert [r["seq"] for r in deals] == list(range(len(seed["deals"])))
    assert [r["seq"] for r in campaigns] == list(range(len(seed["campaigns"])))
    assert {r["origin"] for r in deals} == {"tr3-prebid.example"}
    assert all(r["id"] == r["campaign_id"] for r in campaigns)
    assert deals[0]["terms"]["dealfloor"] == Decimal("3.4")
    # The seed itself is not mutated.
    assert seed["deals"][0]["origin"] == seeder.ORIGIN_TEMPLATE


def test_write_issues_one_put_per_record_and_nothing_else(seeder, seed):
    calls = []

    class _Writer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.append(("flush",))
            return False

        def put_item(self, Item):
            calls.append(("put", Item["record_type"], Item["id"]))

        def __getattr__(self, name):
            raise AssertionError(f"unexpected writer call {name}")

    class _Table:
        def batch_writer(self):
            return _Writer()

        def __getattr__(self, name):
            raise AssertionError(f"unexpected table call {name}")

    written = seeder.write(_Table(), seeder.items(seed, "prebid.example"))
    puts = [c for c in calls if c[0] == "put"]
    assert written == len(puts) == len(seed["deals"]) + len(seed["campaigns"])
    assert calls[-1] == ("flush",)


def test_main_fails_without_touching_aws_on_an_invalid_seed(seeder, seed, tmp_path, capsys):
    del seed["deals"][0]["origin"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(seed))
    code = seeder.main(["--table", "t", "--role-arn", "arn:aws:iam::000000000000:role/x",
                        "--seed", str(bad), "--region", "us-east-1"])
    assert code == 1
    assert "origin" in capsys.readouterr().err


class _Denied(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def test_seeding_retries_access_denied_while_iam_propagates(seeder, monkeypatch):
    outcomes = [_Denied("AccessDenied"), _Denied("AccessDeniedException"), 43]
    calls, sleeps = [], []

    def fake_once(args, seed, origin):
        calls.append(1)
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(seeder, "_seed_once", fake_once)
    written = seeder._seed_with_retry(None, {}, "o", attempts=4, delay=1, sleep=sleeps.append)
    assert written == 43
    assert len(calls) == 3
    assert sleeps == [1, 1]


def test_seeding_gives_up_after_the_last_attempt(seeder, monkeypatch):
    def always_denied(args, seed, origin):
        raise _Denied("AccessDenied")

    monkeypatch.setattr(seeder, "_seed_once", always_denied)
    sleeps = []
    with pytest.raises(_Denied):
        seeder._seed_with_retry(None, {}, "o", attempts=3, delay=1, sleep=sleeps.append)
    assert sleeps == [1, 1]


def test_seeding_does_not_retry_other_errors(seeder, monkeypatch):
    calls = []

    def throttled(args, seed, origin):
        calls.append(1)
        raise _Denied("ValidationException")

    monkeypatch.setattr(seeder, "_seed_once", throttled)
    with pytest.raises(_Denied):
        seeder._seed_with_retry(None, {}, "o", attempts=5, delay=1, sleep=lambda _s: None)
    assert len(calls) == 1
