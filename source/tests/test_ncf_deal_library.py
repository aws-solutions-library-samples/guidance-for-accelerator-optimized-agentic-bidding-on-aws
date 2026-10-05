"""The Deal Scorer's publisher deal library and how activation uses it.

Sell-side ACTIVATE_DEALS activates deals the publisher has but did not offer on the
request. These tests pin:

- the library mirrors the demand catalog (every library deal is one an artfhouse
  campaign transacts on, and every scenario deal the catalog knows is in the library);
- a library deal that scores high enough is activated AND its floor follows as an
  ADJUST_DEAL_FLOOR on the deal's own path, in that order;
- a library deal is never suppressed;
- a library deal the request already offers is scored once, as a request deal;
- an unknown publisher gets today's behaviour exactly.

The NCF model is replaced by a scripted scorer: the inline NeuMF is randomly
initialised, so its scores are not a fixture.
"""
import importlib.util
import json
import os
import sys

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from shared.artf_types import Intent, Operation, RTBRequest  # noqa: E402
from demand.artfhouse.catalog import CampaignCatalog  # noqa: E402

_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")


@pytest.fixture(scope="module")
def ncf():
    path = os.path.join(_ROOT, "containers", "ncf_deal_manager", "app.py")
    spec = importlib.util.spec_from_file_location("ncf_deal_manager_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scripted(ncf, monkeypatch):
    """Install a scorer that activates the ids in `high`, suppresses those in `low`."""
    calls: list[list[str]] = []

    def install(high=(), low=()):
        def _score(uid_hash, deal_ids, activate_threshold=0.0, suppress_threshold=0.0):
            calls.append(list(deal_ids))
            return (
                [d for d in deal_ids if d in high],
                [d for d in deal_ids if d in low],
                "", "",
            )
        monkeypatch.setattr(ncf, "_score_deals", _score)
        return calls

    return install


def _request(bid_request, intents=("ACTIVATE_DEALS", "SUPPRESS_DEALS", "ADJUST_DEAL_FLOOR")):
    return RTBRequest(id="t", bid_request=bid_request, applicable_intents=list(intents))


def _parenting():
    with open(os.path.join(_SAMPLES, "parenting-narrative.json"), encoding="utf-8") as handle:
        return json.load(handle)["bid_request"]


# --------------------------------------------------------------- the library


class TestLibraryMirrorsTheCatalog:
    def test_every_library_deal_is_one_a_campaign_transacts_on(self, ncf):
        catalog_deals = {d for c in CampaignCatalog().all() for d in c.deal_ids}
        assert ncf.deal_library.all_deal_ids() <= catalog_deals

    def test_every_catalog_deal_is_in_the_library(self, ncf):
        catalog_deals = {d for c in CampaignCatalog().all() for d in c.deal_ids}
        assert catalog_deals <= ncf.deal_library.all_deal_ids()

    def test_parenting_publisher_book(self, ncf):
        deals = ncf.deal_library.deals_for(_parenting())
        assert [(d.id, d.bidfloor, d.at) for d in deals] == [
            ("deal-parenting-premium", 3.40, 1),
            ("deal-family-network", 2.10, 2),
            ("deal-remnant-open", 0.85, 3),
        ]

    def test_category_fallback_when_the_site_has_no_domain(self, ncf):
        deals = ncf.deal_library.deals_for({"site": {"cat": ["192"]}})
        assert {d.id for d in deals} == {"deal-parenting-premium", "deal-family-network", "deal-remnant-open"}

    def test_unknown_publisher_has_an_empty_book(self, ncf):
        assert ncf.deal_library.deals_for({"site": {"domain": "nowhere.example", "cat": ["1"]}}) == ()
        assert ncf.deal_library.deals_for({}) == ()
        assert ncf.deal_library.deals_for(None) == ()

    def test_not_on_request_excludes_offered_deals(self, ncf):
        library = ncf.deal_library.deals_for(_parenting())
        left = ncf.deal_library.not_on_request(library, [{"id": "deal-remnant-open"}])
        assert [d.id for d in left] == ["deal-parenting-premium", "deal-family-network"]


# ------------------------------------------------------------- activation


class TestActivationFromTheLibrary:
    def test_the_shipped_parenting_fixture_offers_only_the_remnant_deal(self):
        deals = _parenting()["imp"][0]["pmp"]["deals"]
        assert [d["id"] for d in deals] == ["deal-remnant-open"]

    def test_library_deals_are_scored_alongside_request_deals(self, ncf, scripted):
        calls = scripted(high=())
        ncf.mutate(_request(_parenting()))
        assert calls == [["deal-remnant-open", "deal-parenting-premium", "deal-family-network"]]

    def test_an_activated_library_deal_gets_its_floor_right_after(self, ncf, scripted):
        scripted(high=("deal-parenting-premium", "deal-family-network", "deal-remnant-open"))
        resp = ncf.mutate(_request(_parenting()))
        kinds = [(m.intent, m.op, m.path) for m in resp.mutations]
        assert kinds == [
            (Intent.ACTIVATE_DEALS, Operation.ADD, "/imp/imp-1"),
            (Intent.ADJUST_DEAL_FLOOR, Operation.REPLACE, "/imp/imp-1/deals/deal-parenting-premium"),
            (Intent.ADJUST_DEAL_FLOOR, Operation.REPLACE, "/imp/imp-1/deals/deal-family-network"),
        ]
        assert resp.mutations[0].ids.id == ["deal-remnant-open", "deal-parenting-premium", "deal-family-network"]
        assert resp.mutations[1].adjust_deal.bidfloor == 3.40
        assert resp.mutations[2].adjust_deal.bidfloor == 2.10

    def test_a_request_deal_that_is_activated_gets_no_floor_mutation(self, ncf, scripted):
        # Its floor is the publisher's, already on the request.
        scripted(high=("deal-remnant-open",))
        resp = ncf.mutate(_request(_parenting()))
        assert [m.intent for m in resp.mutations] == [Intent.ACTIVATE_DEALS]

    def test_a_library_deal_that_scores_low_is_not_suppressed(self, ncf, scripted):
        scripted(high=(), low=("deal-parenting-premium", "deal-family-network", "deal-remnant-open"))
        resp = ncf.mutate(_request(_parenting()))
        assert [m.intent for m in resp.mutations] == [Intent.SUPPRESS_DEALS]
        assert resp.mutations[0].ids.id == ["deal-remnant-open"]

    def test_narrowing_away_the_floor_intent_activates_without_a_floor(self, ncf, scripted):
        scripted(high=("deal-parenting-premium",))
        resp = ncf.mutate(_request(_parenting(), intents=("ACTIVATE_DEALS",)))
        assert [m.intent for m in resp.mutations] == [Intent.ACTIVATE_DEALS]
        assert resp.mutations[0].ids.id == ["deal-parenting-premium"]

    def test_narrowing_away_activation_emits_nothing_for_the_library(self, ncf, scripted):
        scripted(high=("deal-parenting-premium",), low=("deal-remnant-open",))
        resp = ncf.mutate(_request(_parenting(), intents=("SUPPRESS_DEALS",)))
        assert [m.intent for m in resp.mutations] == [Intent.SUPPRESS_DEALS]

    def test_an_unknown_publisher_behaves_as_before(self, ncf, scripted):
        calls = scripted(high=("deal-x",))
        bid_request = {
            "id": "r", "site": {"domain": "nowhere.example"},
            "imp": [{"id": "imp-9", "pmp": {"deals": [{"id": "deal-x", "bidfloor": 1.0}]}}],
        }
        resp = ncf.mutate(_request(bid_request))
        assert calls == [["deal-x"]]
        assert [(m.intent, m.path) for m in resp.mutations] == [(Intent.ACTIVATE_DEALS, "/imp/imp-9")]

    def test_an_unknown_publisher_with_no_deals_emits_nothing(self, ncf, scripted):
        calls = scripted(high=("anything",))
        resp = ncf.mutate(_request({"id": "r", "imp": [{"id": "imp-9"}]}))
        assert calls == []
        assert resp.mutations == []

    def test_library_deals_already_offered_are_not_duplicated(self, ncf, scripted):
        calls = scripted(high=("deal-parenting-premium",))
        bid_request = _parenting()
        bid_request["imp"][0]["pmp"]["deals"].append({"id": "deal-parenting-premium", "bidfloor": 3.4, "at": 1})
        resp = ncf.mutate(_request(bid_request))
        assert calls == [["deal-remnant-open", "deal-parenting-premium", "deal-family-network"]]
        # Offered on the request, so its floor is already there: no floor mutation.
        assert [m.intent for m in resp.mutations] == [Intent.ACTIVATE_DEALS]

    def _isv(self):
        with open(os.path.join(_SAMPLES, "isv-ecosystem.json"), encoding="utf-8") as handle:
            return json.load(handle)["bid_request"]

    def test_the_shipped_isv_fixture_offers_no_deals_but_opens_a_pmp_on_the_leaderboard(self):
        imps = {imp["id"]: imp for imp in self._isv()["imp"]}
        assert imps["imp-1"]["pmp"]["deals"] == []
        assert "pmp" not in imps["imp-2"]

    def test_library_deals_go_only_onto_impressions_with_a_pmp(self, ncf, scripted):
        # Both Autoline deals are in cnn.com's book. The leaderboard (imp-1) opened a
        # private marketplace with no deals in it; the 300x600 rail (imp-2) did not
        # open one at all. The library fills the first and leaves the second alone --
        # otherwise a 970x250 deal creative would land in the rail.
        calls = scripted(high=("deal-premium-auto", "deal-standard-auto"))
        resp = ncf.mutate(_request(self._isv()))
        assert calls == [["deal-premium-auto", "deal-standard-auto"]]
        assert [(m.intent, m.path) for m in resp.mutations] == [
            (Intent.ACTIVATE_DEALS, "/imp/imp-1"),
            (Intent.ADJUST_DEAL_FLOOR, "/imp/imp-1/deals/deal-premium-auto"),
            (Intent.ADJUST_DEAL_FLOOR, "/imp/imp-1/deals/deal-standard-auto"),
        ]
        assert resp.mutations[0].ids.id == ["deal-premium-auto", "deal-standard-auto"]
        assert resp.mutations[1].adjust_deal.bidfloor == 6.00
        assert resp.mutations[2].adjust_deal.bidfloor == 3.50

    def test_an_empty_pmp_is_still_an_open_private_marketplace(self, ncf, scripted):
        scripted(high=("deal-parenting-premium",))
        bid_request = _parenting()
        bid_request["imp"][0]["pmp"] = {"private_auction": 0, "deals": []}
        resp = ncf.mutate(_request(bid_request))
        assert [m.intent for m in resp.mutations] == [Intent.ACTIVATE_DEALS, Intent.ADJUST_DEAL_FLOOR]
