"""The two RideSharer Theater scenarios, checked across every copy that describes them.

A scenario is the ScenarioCard.jsx entry, its sample payload, the deal records in
the deals seed (read through the Deal Scorer's loader) and the campaign records
(read through the demand catalog's loader). These tests hold the four in step and
pin the copy rule: a card states context and intended direction, and the only
dollar figures it may quote are the request's own floors.
"""
import importlib.util
import json
import os
import re
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from demand.artfhouse.catalog import CampaignCatalog  # noqa: E402
from demand.artfhouse.eligibility import evaluate  # noqa: E402
from demand.artfhouse.exclusion import ExclusionReason  # noqa: E402

_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")
_CARD = os.path.join(_ROOT, "frontend-react", "src", "components", "ScenarioCard.jsx")

AIRPORT = "ridesharer-airport-evening.json"
MIDDAY = "ridesharer-suburban-midday.json"
CARDS = {
    "ridesharer-airport": ("RideSharer: Airport Run, Raise the Floor", AIRPORT),
    "ridesharer-midday": ("RideSharer: Midday Ride, Lower the Floor", MIDDAY),
}
RIDESHARER_DEALS = (
    "deal-ridesharer-airport-open-travel",
    "deal-ridesharer-airport-open-luggage",
    "deal-ridesharer-eats-open-kitchen",
    "deal-ridesharer-eats-open-noodles",
)
AIRPORT_CAMPAIGNS = ("camp-altimara-travel", "camp-ivoryrow-luggage")
RESTAURANT_CAMPAIGNS = ("camp-basilbrine-kitchen", "camp-maplecorner-noodles")


def _load(name):
    with open(os.path.join(_SAMPLES, name), encoding="utf-8") as handle:
        return json.load(handle)


def _deals(payload):
    return payload["bid_request"]["imp"][0]["pmp"]["deals"]


@pytest.fixture(scope="module")
def library():
    path = os.path.join(_ROOT, "shared", "deal_library.py")
    spec = importlib.util.spec_from_file_location("deal_library_for_ridesharer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load()
    return module


@pytest.fixture(scope="module")
def catalog():
    return CampaignCatalog()


@pytest.fixture(scope="module")
def seed_deals():
    with open(os.environ["DEALS_SEED_FILE"], encoding="utf-8") as handle:
        return {d["id"]: d for d in json.load(handle)["deals"]}


@pytest.fixture(scope="module")
def card_source():
    with open(_CARD, encoding="utf-8") as handle:
        return handle.read()


def _block(card_source, card_id):
    start = card_source.index(f'id: "{card_id}"')
    block_start = card_source.rindex("\n  {\n", 0, start)
    block_end = card_source.index("\n  },", start)
    return card_source[block_start:block_end]


def _field(block, name):
    match = re.search(rf'\n    {name}: "((?:[^"\\]|\\.)*)"', block)
    assert match, f"no {name} in block"
    return match.group(1)


# ----------------------------------------------------------------- the cards


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_the_card_exists_with_its_name_file_and_surface(card_source, card_id):
    name, file_name = CARDS[card_id]
    block = _block(card_source, card_id)
    assert _field(block, "name") == name
    assert f'file: "{file_name}"' in block
    assert "surface: SURFACE_REQUEST" in block


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_the_card_names_every_campaign_holding_a_deal_on_its_payload(card_source, catalog, card_id):
    demand = _field(_block(card_source, card_id), "demand")
    for deal in _deals(_load(CARDS[card_id][1])):
        campaign = catalog.by_deal(deal["id"])
        assert campaign is not None
        assert campaign.campaign_name in demand, f"{card_id} does not name {campaign.campaign_name}"


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_every_dollar_figure_in_the_copy_is_a_floor_or_a_seeded_price(card_source, catalog, card_id):
    """A figure is either one of the request's own floors or a seeded CPM (declared,
    or with its audience uplift) of a campaign holding a deal on the request."""
    block = _block(card_source, card_id)
    payload = _load(CARDS[card_id][1])
    imp = payload["bid_request"]["imp"][0]
    allowed = {float(imp["bidfloor"])} | {float(d["bidfloor"]) for d in _deals(payload)}
    allowed |= {float(d["ext"]["mindealfloor"]) for d in _deals(payload) if "ext" in d}
    for deal in _deals(payload):
        campaign = catalog.by_deal(deal["id"])
        allowed |= {round(campaign.declared_cpm, 2), round(campaign.ceiling_cpm, 2)}
    for name in ("page", "audience", "demand"):
        for figure in re.findall(r"\$(\d+(?:\.\d+)?)", _field(block, name)):
            assert float(figure) in allowed, f"{card_id} {name} quotes ${figure}, not a floor or seeded price"


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_the_card_says_the_floor_is_decided_at_run_time(card_source, card_id):
    demand = _field(_block(card_source, card_id), "demand")
    assert "Yield Optimizer" in demand and "at run time" in demand


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_the_card_default_floor_is_the_payloads_impression_floor(card_source, card_id):
    block = _block(card_source, card_id)
    assert "defaults: { bidFloor: 5.0 }" in block
    assert _load(CARDS[card_id][1])["bid_request"]["imp"][0]["bidfloor"] == 5.0


# --------------------------------------------------------------- the payloads


def test_the_airport_deals_are_first_price_at_twelve_with_no_minimum():
    for deal in _deals(_load(AIRPORT)):
        assert (deal["bidfloor"], deal["at"], deal["bidfloorcur"], deal["guar"]) == (12.0, 1, "USD", 0)
        assert "ext" not in deal


def test_the_midday_deals_are_second_price_at_twelve_with_a_five_dollar_minimum(seed_deals):
    payload = _load(MIDDAY)
    assert payload["bid_request"]["imp"][0]["bidfloor"] == 5.0
    for deal in _deals(payload):
        assert (deal["bidfloor"], deal["at"], deal["bidfloorcur"], deal["guar"]) == (12.0, 2, "USD", 0)
        assert deal["ext"]["mindealfloor"] == 5.0
        assert deal["ext"]["mindealfloor"] == seed_deals[deal["id"]]["terms"]["ext"]["mindealfloor"]


@pytest.mark.parametrize("sample", [AIRPORT, MIDDAY])
def test_each_sample_deal_is_the_openrtb_object_built_from_its_record(library, sample):
    built = {d.id: d.as_openrtb() for d in library.deals_for(_load(sample)["bid_request"])}
    for deal in _deals(_load(sample)):
        assert deal == built[deal["id"]]


def test_both_payloads_are_the_same_app_placement():
    a, b = _load(AIRPORT)["bid_request"], _load(MIDDAY)["bid_request"]
    assert a["app"] == b["app"]
    assert a["app"]["bundle"] == "com.ridesharer.app"
    assert "site" not in a and "site" not in b
    assert a["imp"][0]["banner"] == b["imp"][0]["banner"] == {"w": 320, "h": 480, "pos": 7}


@pytest.mark.parametrize("sample", [AIRPORT, MIDDAY])
def test_the_library_book_for_the_app_is_the_four_ridesharer_deals(library, sample):
    book = library.deals_for(_load(sample)["bid_request"])
    assert tuple(d.id for d in book) == RIDESHARER_DEALS


# ------------------------------------------------------------- the demand side


def _with_all_four(sample, library):
    """The impression with every RideSharer deal on it, as after a library activation."""
    imp = json.loads(json.dumps(_load(sample)["bid_request"]["imp"][0]))
    imp["pmp"]["deals"] = [d.as_openrtb() for d in library.deals_for(_load(sample)["bid_request"])]
    return imp


def _reasons(imp, catalog):
    return {c.campaign.campaign_id: c.excluded_because for c in evaluate(imp, catalog)}


def test_targeting_turns_the_restaurants_away_on_the_airport_ride(library, catalog):
    reasons = _reasons(_with_all_four(AIRPORT, library), catalog)
    for campaign_id in RESTAURANT_CAMPAIGNS:
        assert reasons[campaign_id] is ExclusionReason.NOT_TARGETED
    for campaign_id in AIRPORT_CAMPAIGNS:
        assert reasons[campaign_id] is None


def test_targeting_turns_the_airport_buyers_away_on_the_midday_ride(library, catalog):
    reasons = _reasons(_with_all_four(MIDDAY, library), catalog)
    for campaign_id in AIRPORT_CAMPAIGNS:
        assert reasons[campaign_id] is ExclusionReason.NOT_TARGETED
    for campaign_id in RESTAURANT_CAMPAIGNS:
        assert reasons[campaign_id] is None


def test_the_midday_restaurants_sit_between_the_minimum_and_the_base_floor(catalog):
    for campaign_id in RESTAURANT_CAMPAIGNS:
        campaign = next(c for c in catalog.all() if c.campaign_id == campaign_id)
        assert 5.0 <= campaign.declared_cpm < 12.0


def test_the_airport_buyers_clear_the_base_floor(catalog):
    for campaign_id in AIRPORT_CAMPAIGNS:
        campaign = next(c for c in catalog.all() if c.campaign_id == campaign_id)
        assert campaign.declared_cpm > 12.0
        assert campaign.ceiling_cpm <= 13.30
