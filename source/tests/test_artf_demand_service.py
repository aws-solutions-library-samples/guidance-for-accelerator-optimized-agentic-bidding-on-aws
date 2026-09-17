"""The service end to end, and the response contract the frontend reads."""

import pytest

from demand.artfhouse.exclusion import ExclusionReason
from demand.artfhouse.floors import CurrencyMismatch
from demand.artfhouse.service import DemandDecisionService, MalformedRequest, validate

SERVICE = DemandDecisionService()


def request_with(deals=(), floor=1.0, cur=None):
    req = {
        "id": "req-1",
        "imp": [{"id": "imp-1", "bidfloor": floor, "pmp": {"deals": list(deals)}}],
    }
    if cur is not None:
        req["cur"] = cur
    return req


def excluded_of(response):
    return response.get("ext", {}).get("artf", {}).get("excluded", [])


def bids_of(response):
    seatbids = response.get("seatbid", [])
    return seatbids[0]["bid"] if seatbids else []


# ------------------------------------------------------------------ validation


def test_validate_accepts_a_well_formed_request():
    assert validate(request_with()) == []


def test_validate_rejects_a_non_object():
    assert validate("not a request") != []
    assert validate(None) != []


def test_validate_requires_an_id_and_impressions():
    assert "request has no id" in validate({"imp": [{"id": "i"}]})
    assert "request has no impressions" in validate({"id": "r"})


def test_malformed_request_raises_rather_than_returning_an_empty_response():
    # An empty bid response asserts that no campaign wished to offer. Returning it
    # for an unparseable request would state something untrue about demand.
    with pytest.raises(MalformedRequest):
        SERVICE.decide({"imp": []})


def test_unsupported_currency_raises_rather_than_rescaling():
    with pytest.raises(CurrencyMismatch):
        SERVICE.decide(request_with(cur="EUR"))


def test_decide_accepts_the_openrtb_currency_ARRAY_end_to_end():
    """The whole path, with the shape a real exchange sends.

    Regression for the defect that made this endpoint answer every live auction with
    a 500: ``cur`` is an array in OpenRTB 2.x and Prebid Server sends ``['USD']``, but
    the currency check treated it as a string. Every unit test used a bare string, so
    the suite was green while nothing worked end to end.

    Asserted through ``decide`` rather than only on the helper, because the helper
    passing in isolation is what the old tests already proved.
    """
    response = SERVICE.decide(
        request_with(deals=[{"id": "deal-home-premium", "bidfloor": 2.0}], cur=["USD"])
    )

    bids = [bid for seat in response.get("seatbid", []) for bid in seat.get("bid", [])]
    assert bids, "a spec-shaped request must produce bids"
    # The response states ONE currency, which is correct: cur is an array on the
    # request and a single string on the response.
    assert response["cur"] == "USD"


def test_decide_accepts_a_currency_array_that_also_allows_others():
    response = SERVICE.decide(
        request_with(deals=[{"id": "deal-home-premium", "bidfloor": 2.0}], cur=["EUR", "USD"])
    )
    assert [bid for seat in response.get("seatbid", []) for bid in seat.get("bid", [])]


def test_decide_rejects_a_currency_array_without_the_supported_one():
    with pytest.raises(CurrencyMismatch):
        SERVICE.decide(request_with(cur=["EUR", "GBP"]))


# --------------------------------------------------------------- the response


def test_every_considered_campaign_appears_as_a_bid_or_an_exclusion():
    response = SERVICE.decide(request_with([{"id": "deal-home-premium"}]))
    seen = {b["ext"]["prebid"]["artf"]["campaignId"] for b in bids_of(response)}
    seen |= {e["campaignId"] for e in excluded_of(response)}
    assert len(seen) == 5  # the whole catalog


def test_a_suppressed_deal_is_distinguishable_from_a_below_floor_rejection():
    deals = [
        {"id": "deal-finance-pmp", "ext": {"artf": {"suppressed": True}}},
        {"id": "deal-auto-brand", "bidfloor": 9.0},
    ]
    response = SERVICE.decide(request_with(deals))
    reasons = {e["campaignId"]: e["exclusionReason"] for e in excluded_of(response)}
    assert reasons["camp-harbour"] == ExclusionReason.DEAL_SUPPRESSED.value
    assert reasons["camp-vantage"] == ExclusionReason.BELOW_FLOOR.value


def test_the_excluded_block_is_present_when_a_campaign_made_no_offer():
    # This block is what the frontend reads for never-offered candidates:
    # ext.seatnonbid cannot carry them, because Prebid records no seatnonbid entry
    # for a seat that did bid.
    response = SERVICE.decide(request_with([{"id": "deal-home-premium"}]))
    assert excluded_of(response)


def test_a_response_with_no_bids_omits_seatbid_rather_than_sending_it_empty():
    response = SERVICE.decide(request_with(floor=99.0))
    assert "seatbid" not in response
    assert excluded_of(response)


def test_the_response_echoes_the_request_id_for_correlation():
    response = SERVICE.decide(request_with())
    assert response["id"] == "req-1"


def test_the_response_states_its_currency():
    assert SERVICE.decide(request_with())["cur"] == "USD"


def test_no_winner_or_clearing_price_is_computed():
    # Prebid resolves the auction. Nothing here may claim a winner.
    response = SERVICE.decide(request_with([{"id": "deal-home-premium"}]))
    text = repr(response)
    assert "winner" not in text
    assert "clearing" not in text
    for bid in bids_of(response):
        assert "targeting" not in bid.get("ext", {}).get("prebid", {})


def test_the_same_request_yields_the_same_response():
    first = SERVICE.decide(request_with([{"id": "deal-home-premium"}]))
    second = SERVICE.decide(request_with([{"id": "deal-home-premium"}]))
    assert first == second


def test_a_floor_adjustment_admits_demand_the_original_floor_excluded():
    """FR-42's shape: lowering the binding floor lets a campaign offer."""
    high = SERVICE.decide(request_with([{"id": "deal-auto-brand", "bidfloor": 5.0}]))
    low = SERVICE.decide(request_with([{"id": "deal-auto-brand", "bidfloor": 1.0}]))

    high_reasons = {e["campaignId"]: e["exclusionReason"] for e in excluded_of(high)}
    assert high_reasons["camp-vantage"] == ExclusionReason.BELOW_FLOOR.value

    low_bidders = {b["ext"]["prebid"]["artf"]["campaignId"] for b in bids_of(low)}
    assert "camp-vantage" in low_bidders


def test_deal_suppression_removes_a_candidate_before_bidding():
    """FR-43's shape: the campaign is present with a reason, not absent."""
    plain = SERVICE.decide(request_with([{"id": "deal-finance-pmp"}]))
    assert "camp-harbour" in {b["ext"]["prebid"]["artf"]["campaignId"] for b in bids_of(plain)}

    suppressed = SERVICE.decide(
        request_with([{"id": "deal-finance-pmp", "ext": {"artf": {"suppressed": True}}}])
    )
    assert "camp-harbour" not in {
        b["ext"]["prebid"]["artf"]["campaignId"] for b in bids_of(suppressed)
    }
    reasons = {e["campaignId"]: e["exclusionReason"] for e in excluded_of(suppressed)}
    assert reasons["camp-harbour"] == ExclusionReason.DEAL_SUPPRESSED.value


# ---------------------------------------------------------------------------
# CATEGORY TARGETING THROUGH PREBID.
#
# Prebid Server rewrites imp.ext per bidder and removes keys it does not
# recognise, treating an unknown key as a bidder name. A request carrying
# imp.ext.artf came back with
#
#   request.imp[0].ext.prebid.bidder.artf was dropped with a reason:
#   request.imp[0].ext.prebid.bidder contains unknown bidder: artf
#
# so the category signal never reached this endpoint through an auction. It was
# masked because an impression with NO categories matches every campaign, which
# meant a targeted campaign bid for a reason that had nothing to do with
# targeting. imp.ext.data is first-party data that Prebid preserves, so it is
# read as well.
# ---------------------------------------------------------------------------

def request_with_categories(categories, location="data"):
    """A request whose only variable is WHERE the categories are declared.

    The deal is present because camp-cedar bids only on deal-home-premium: without
    it cedar is excluded for no_deal_on_impression, and a test that then observed
    "cedar did not bid" would be measuring the missing deal rather than targeting.
    """
    imp = {
        "id": "imp-1",
        "bidfloor": 1.0,
        "pmp": {"deals": [{"id": "deal-home-premium", "bidfloor": 5.0}]},
    }
    if location == "data":
        imp["ext"] = {"data": {"artf": {"categories": list(categories)}}}
    elif location == "direct":
        imp["ext"] = {"artf": {"categories": list(categories)}}
    else:
        raise AssertionError(f"unknown location {location!r}")
    return {"id": "req-cat", "imp": [imp]}


def crids_of(response):
    return {b.get("crid") for b in bids_of(response)}


def test_categories_under_imp_ext_data_are_read():
    # The Prebid-safe location. Regression: before the fix this was ignored, so a
    # non-matching category still admitted every targeted campaign.
    response = SERVICE.decide(request_with_categories(["home"], "data"))
    assert "cr-cedar-300x250" in crids_of(response)


def test_categories_under_imp_ext_artf_are_still_read_for_direct_callers():
    # Callers that reach this endpoint without Prebid in the path.
    response = SERVICE.decide(request_with_categories(["home"], "direct"))
    assert "cr-cedar-300x250" in crids_of(response)


def test_a_non_matching_category_excludes_a_targeted_campaign():
    # THE check that proves targeting is exercised at all. camp-cedar targets
    # home and lifestyle, so a finance impression must not admit it. Without this,
    # a passing "cedar bid" says nothing: an impression with no categories matches
    # everything.
    response = SERVICE.decide(request_with_categories(["finance"], "data"))
    assert "cr-cedar-300x250" not in crids_of(response)


def test_a_non_matching_category_still_admits_the_open_market_campaign():
    # Exclusion has to be selective. camp-openfield declares no target categories,
    # so it is unrestricted and must survive a category that excludes others.
    response = SERVICE.decide(request_with_categories(["finance"], "data"))
    assert "cr-openfield-300x250" in crids_of(response)


def test_the_data_location_wins_when_both_are_present():
    # Read order is not arbitrary: the Prebid-safe location is authoritative,
    # because that is the one an auction can actually deliver.
    imp = {
        "id": "imp-1",
        "bidfloor": 1.0,
        "pmp": {"deals": [{"id": "deal-home-premium", "bidfloor": 5.0}]},
        "ext": {
            "data": {"artf": {"categories": ["finance"]}},
            "artf": {"categories": ["home"]},
        },
    }
    response = SERVICE.decide({"id": "req-cat", "imp": [imp]})
    assert "cr-cedar-300x250" not in crids_of(response)
