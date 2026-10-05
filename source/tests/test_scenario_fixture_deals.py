"""Every request-side Theater scenario offers only its open/remnant deals.

The Auction Theater runs each request scenario twice, with and without ARTF, and
compares the outcomes. A fixture that pre-declares the premium deal the Deal Scorer
would activate hands the with-ARTF outcome to the baseline pass, and the comparison
shows two identical columns. So the rule for every request fixture is:

  * the publisher offers only the deal(s) whose id contains "open" or "remnant";
  * every other deal a campaign in the artfhouse catalog transacts on for that
    publisher is in the Deal Scorer's library (containers/ncf_deal_manager/
    deal_library.py), so it can still reach the auction -- by activation;
  * a fixture that can receive activations declares ACTIVATE_DEALS, and its
    ScenarioCard entry tags that intent and lists the NCF Deal Manager.

Response-side fixtures (response-*.json) carry a bid response, not a bid request,
and are not covered here.
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

_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")
_CARD = os.path.join(_ROOT, "frontend-react", "src", "components", "ScenarioCard.jsx")

REQUEST_FIXTURES = (
    "banner-basic.json",
    "finance-news.json",
    "home-lifestyle.json",
    "isv-ecosystem.json",
    "parenting-narrative.json",
    "video-deals.json",
    "yield-optimizer.json",
)


def _load(name):
    with open(os.path.join(_SAMPLES, name), encoding="utf-8") as handle:
        return json.load(handle)


def _offered(payload):
    for imp in payload["bid_request"]["imp"]:
        for deal in ((imp.get("pmp") or {}).get("deals") or []):
            yield imp["id"], deal


@pytest.fixture(scope="module")
def deal_library():
    path = os.path.join(_ROOT, "containers", "ncf_deal_manager", "deal_library.py")
    spec = importlib.util.spec_from_file_location("ncf_deal_library_for_scenarios", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def card_source():
    with open(_CARD, encoding="utf-8") as handle:
        return handle.read()


def _card_block(card_source, fixture):
    start = card_source.index(f'file: "{fixture}"')
    block_start = card_source.rindex("\n  {\n", 0, start)
    block_end = card_source.index("\n  },", start)
    return card_source[block_start:block_end]


@pytest.mark.parametrize("fixture", REQUEST_FIXTURES)
def test_the_publisher_offers_only_open_or_remnant_deals(fixture):
    offered = [deal["id"] for _, deal in _offered(_load(fixture))]
    stray = [d for d in offered if not re.search(r"open|remnant", d)]
    assert stray == [], f"{fixture} pre-declares {stray}; those belong in the Deal Scorer's library"


@pytest.mark.parametrize("fixture", REQUEST_FIXTURES)
def test_every_offered_deal_is_in_the_library_at_the_same_terms(fixture, deal_library):
    payload = _load(fixture)
    book = {d.id: d for d in deal_library.deals_for(payload["bid_request"])}
    for imp_id, deal in _offered(payload):
        assert deal["id"] in book, f"{fixture} {imp_id} offers {deal['id']} which the library does not know"
        assert (book[deal["id"]].bidfloor, book[deal["id"]].at) == (deal["bidfloor"], deal["at"])


@pytest.mark.parametrize("fixture", REQUEST_FIXTURES)
def test_the_catalog_deals_for_this_publisher_are_all_reachable(fixture, deal_library):
    # Reachable = offered on the request, or activatable from the library. A deal a
    # campaign would bid on that is neither can never transact in this scenario.
    payload = _load(fixture)
    book = {d.id for d in deal_library.deals_for(payload["bid_request"])}
    offered = {deal["id"] for _, deal in _offered(payload)}
    catalog = CampaignCatalog()
    expected_deals = set()
    for campaign in catalog.all():
        if campaign.deal_ids and set(campaign.deal_ids) & (book | offered):
            expected_deals |= set(campaign.deal_ids)
    assert expected_deals <= (book | offered)


@pytest.mark.parametrize("fixture", REQUEST_FIXTURES)
def test_a_fixture_with_activatable_deals_declares_activate_deals(fixture, deal_library):
    payload = _load(fixture)
    book = deal_library.deals_for(payload["bid_request"])
    activatable = [
        d.id
        for imp in payload["bid_request"]["imp"]
        if imp.get("pmp") is not None
        for d in deal_library.not_on_request(book, (imp["pmp"].get("deals") or []))
    ]
    if activatable:
        assert "ACTIVATE_DEALS" in payload["applicable_intents"], (
            f"{fixture} leaves {activatable} to the Deal Scorer but does not declare ACTIVATE_DEALS"
        )


@pytest.mark.parametrize("fixture", REQUEST_FIXTURES)
def test_the_card_tags_every_intent_the_fixture_declares(fixture, card_source):
    block = _card_block(card_source, fixture)
    for intent in _load(fixture)["applicable_intents"]:
        assert f'label: "{intent}"' in block, f"card for {fixture} does not tag {intent}"


@pytest.mark.parametrize("fixture", REQUEST_FIXTURES)
def test_the_card_lists_the_deal_manager_when_deals_can_be_activated(fixture, card_source):
    block = _card_block(card_source, fixture)
    declares = "ACTIVATE_DEALS" in _load(fixture)["applicable_intents"]
    assert ('key: "ncf_deal_manager"' in block) == declares


def test_video_deals_has_no_request_side_deal_count_control(card_source):
    # The slider sliced the pre-declared deal list. With one remnant deal left on the
    # request and the rest in the library, there is nothing left for it to slice.
    assert '"numDeals"' not in _card_block(card_source, "video-deals.json")
