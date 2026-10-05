"""The live-auction route: what it does when Prebid is there, and when it is not.

The behaviour under test that matters most is the ABSENCE case. This route feeds a
screen that claims to show a real auction, so "Prebid is not deployed" and "the
auction produced no bids" must not render the same, and neither may be answered
with a stored response presented as live.
"""

import asyncio
import json

import pytest

from orchestrator import auction_api


class FakeRequest:
    """Minimal stand-in for a Starlette Request: the handler reads json() and
    query_params."""

    def __init__(self, body, raise_exc=None, query=None):
        self._body = body
        self._raise = raise_exc
        self.query_params = dict(query or {})

    async def json(self):
        if self._raise is not None:
            raise self._raise
        return self._body


def run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def body_of(response):
    return json.loads(response.body)


VALID_REQUEST = {"id": "req-1", "imp": [{"id": "imp-1"}]}


@pytest.fixture(autouse=True)
def _no_switch(monkeypatch):
    """Default every test to "Prebid not deployed" so setting it is explicit."""
    monkeypatch.delenv(auction_api.PREBID_AUCTION_URL_ENV, raising=False)


# ------------------------------------------------------------------ the switch

def test_status_reports_not_configured_when_the_switch_is_absent():
    response = run(auction_api.auction_status_handler(None))
    assert response.status_code == 200
    payload = body_of(response)
    assert payload["prebid"] == "not_configured"
    assert payload["endpoint"] is None
    # The detail has to tell the reader what to do, not merely that something is off.
    assert "--with-prebid" in payload["detail"]


def test_status_reports_configured_once_the_switch_is_set(monkeypatch):
    monkeypatch.setenv(
        auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction"
    )
    payload = body_of(run(auction_api.auction_status_handler(None)))
    assert payload["prebid"] == "configured"
    assert payload["endpoint"] == "https://prebid/openrtb2/auction"


def test_a_blank_switch_counts_as_absent(monkeypatch):
    # --destroy may clear the variable by setting it empty rather than removing it,
    # and an empty endpoint must not read as "configured".
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "   ")
    assert body_of(run(auction_api.auction_status_handler(None)))["prebid"] == "not_configured"


# -------------------------------------------------- refusing to invent a result

def test_running_without_prebid_returns_501_and_no_auction():
    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    assert response.status_code == 501
    payload = body_of(response)
    assert payload["error"] == "prebid_not_deployed"
    # The decisive assertion: nothing that could be mistaken for an auction result.
    assert "seatbid" not in payload
    assert payload["prebid"] == "not_configured"


def test_malformed_json_is_a_400(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    response = run(
        auction_api.run_auction_handler(
            FakeRequest(None, raise_exc=ValueError("not json"))
        )
    )
    assert response.status_code == 400
    assert body_of(response)["error"] == "invalid_json"


@pytest.mark.parametrize("payload", [{}, {"imp": []}, [1, 2], "nope"])
def test_a_request_without_impressions_is_a_400(monkeypatch, payload):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    response = run(auction_api.run_auction_handler(FakeRequest(payload)))
    assert response.status_code == 400
    assert body_of(response)["error"] == "invalid_bid_request"


# ------------------------------------------------- passing the auction through

class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeClient:
    """Stands in for httpx.AsyncClient. Records what was sent."""

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise = raise_exc
        self.posted = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.posted = {"url": url, "json": json}
        if self._raise is not None:
            raise self._raise
        return self._response


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(auction_api.httpx, "AsyncClient", lambda **kw: client)


AUCTION = {
    "id": "req-1",
    "cur": "USD",
    "seatbid": [
        {"seat": "artfhouse", "bid": [{"crid": "cr-cedar-300x250", "price": 6.35}]},
        {"seat": "amt", "bid": [{"crid": "banner_creative_1", "price": 3.25}]},
    ],
}


def test_the_auction_is_returned_unchanged(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, AUCTION))
    _patch_client(monkeypatch, client)

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))

    # Prebid's own seatbid, untouched: no bid added, no price altered.
    assert payload["seatbid"] == AUCTION["seatbid"]
    assert payload["cur"] == "USD"
    # The request reached Prebid as given, apart from the bidder enablement and
    # debug flag the orchestrator adds -- covered by the preparation tests below.
    sent = client.posted["json"]
    assert sent["id"] == VALID_REQUEST["id"]
    assert sent["imp"][0]["id"] == VALID_REQUEST["imp"][0]["id"]
    assert sent["ext"]["prebid"]["debug"] == 1
    assert set(sent) - set(VALID_REQUEST) == {"ext"}


def test_metadata_states_where_the_numbers_came_from(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, dict(AUCTION))))

    meta = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))["artf_meta"]

    assert meta["source"] == "prebid"
    assert meta["endpoint"] == "https://prebid/openrtb2/auction"
    assert meta["seats"] == ["amt", "artfhouse"]
    assert isinstance(meta["hop_ms"], int)


def test_an_empty_auction_is_reported_as_an_empty_auction(monkeypatch):
    # No bids is a legitimate outcome and must come back as 200 with no seatbid --
    # distinct from the 501 that means Prebid is not deployed.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, {"id": "req-1", "cur": "USD"})))

    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    payload = body_of(response)

    assert response.status_code == 200
    assert payload.get("seatbid") is None
    assert payload["artf_meta"]["seats"] == []


# --------------------------------------------------------- failures, as failures

def test_a_timeout_is_a_504_not_an_empty_auction(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(
        monkeypatch, FakeClient(raise_exc=auction_api.httpx.TimeoutException("too slow"))
    )
    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    assert response.status_code == 504
    payload = body_of(response)
    assert payload["error"] == "prebid_timeout"
    assert "seatbid" not in payload


def test_an_unreachable_prebid_is_a_502(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(
        monkeypatch, FakeClient(raise_exc=auction_api.httpx.ConnectError("refused"))
    )
    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    assert response.status_code == 502
    assert body_of(response)["error"] == "prebid_unreachable"


def test_a_non_json_response_is_a_502(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, None, text="<html>nope</html>")))
    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    assert response.status_code == 502
    assert body_of(response)["error"] == "prebid_returned_non_json"


def test_a_rejection_from_prebid_is_surfaced_with_its_status(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(
        monkeypatch, FakeClient(FakeResponse(400, {"message": "invalid request"}))
    )
    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    payload = body_of(response)
    assert response.status_code == 502
    assert payload["error"] == "prebid_rejected_request"
    # Prebid's own account of the refusal is preserved for diagnosis.
    assert payload["status"] == 400
    assert payload["response"] == {"message": "invalid request"}


# ---------------------------------------------------------------------------
# PER-CAMPAIGN EXCLUSIONS.
#
# A bidder's own response body is not part of an OpenRTB bid response: Prebid
# returns bids, and the artfhouse adapter extracts bids and nothing else. So the
# demand endpoint's account of WHICH campaign lost and WHY reaches us only inside
# ext.debug.httpcalls.artfhouse[].responsebody, which is why the orchestrator asks
# Prebid for debug output.
#
# The distinction these tests protect: an EMPTY list asserts "nothing was
# excluded", while an ABSENT list means "we could not tell". Rendering the second
# as the first would put a false statement on screen.
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    {"campaignId": "camp-harbour", "exclusionReason": "no_deal_on_impression"},
    {"campaignId": "camp-vantage", "exclusionReason": "no_deal_on_impression"},
]


def _auction_with_httpcalls(responsebody, bidder="artfhouse"):
    return {
        "id": "req-1",
        "seatbid": [{"seat": "artfhouse", "bid": [{"crid": "c", "price": 1.0}]}],
        "ext": {"debug": {"httpcalls": {bidder: [{"status": 200, "responsebody": responsebody}]}}},
    }


def test_debug_is_requested_so_the_exclusions_can_arrive(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    run(auction_api.run_auction_handler(FakeRequest(dict(VALID_REQUEST))))

    assert client.posted["json"]["ext"]["prebid"]["debug"] == 1


def test_requesting_debug_does_not_mutate_the_caller_s_request(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, dict(AUCTION))))

    original = {"id": "req-1", "imp": [{"id": "imp-1"}], "ext": {"prebid": {"targeting": {}}}}
    snapshot = json.loads(json.dumps(original))

    run(auction_api.run_auction_handler(FakeRequest(original)))

    assert original == snapshot


def test_exclusions_are_lifted_from_the_demand_response(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    body = json.dumps({"ext": {"artf": {"excluded": EXCLUSIONS}}})
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, _auction_with_httpcalls(body))))

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))

    assert payload["ext"]["artf"]["excluded"] == EXCLUSIONS
    assert payload["artf_meta"]["excluded_source"].startswith("artfhouse response")


def test_an_empty_exclusion_list_is_preserved_as_empty(monkeypatch):
    # "Nothing was excluded" is a real answer and must survive as [], not become
    # absent -- the screen can then say so truthfully.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    body = json.dumps({"ext": {"artf": {"excluded": []}}})
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, _auction_with_httpcalls(body))))

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))

    assert payload["ext"]["artf"]["excluded"] == []
    assert payload["artf_meta"]["excluded_source"].startswith("artfhouse response")


@pytest.mark.parametrize(
    "auction,expected_reason",
    [
        ({"id": "r", "ext": {}}, "no httpcalls"),
        ({"id": "r", "ext": {"debug": {"httpcalls": {"amt": [{"status": 200}]}}}},
         "artfhouse was not called"),
    ],
)
def test_no_exclusion_list_is_reported_as_unavailable(monkeypatch, auction, expected_reason):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, auction)))

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))

    # Absent, NOT an empty list: the difference between "could not tell" and
    # "nothing was excluded".
    assert "excluded" not in (payload.get("ext") or {}).get("artf", {})
    assert payload["artf_meta"]["excluded_source"].startswith("unavailable")
    assert expected_reason in payload["artf_meta"]["excluded_source"]


def test_an_unparseable_demand_response_does_not_invent_exclusions(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(
        monkeypatch, FakeClient(FakeResponse(200, _auction_with_httpcalls("<not json>")))
    )

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))

    assert "excluded" not in (payload.get("ext") or {}).get("artf", {})
    assert payload["artf_meta"]["excluded_source"].startswith("unavailable")


# ---------------------------------------------------------------------------
# PREPARING AN ARTF SCENARIO FOR AN AUCTION.
#
# Prebid routes an impression to a bidder only when imp.ext.<bidder> is present --
# that key is how a publisher's prebid.js config declares which bidders to call.
# The ARTF scenarios are bidstream requests with no such keys, so submitted
# verbatim they reach NO bidder and return an empty auction that reads as "nobody
# wanted this impression".
#
# What these tests pin down is the boundary: enablement is added, and nothing that
# constitutes an ANSWER is. No bid, no price, no campaign, no deal, no category.
# ---------------------------------------------------------------------------

def test_both_seats_are_enabled_on_a_bare_impression():
    # The shape of source/frontend-react/public/samples/banner-basic.json: an imp
    # with no ext at all.
    body = {"id": "r", "imp": [{"id": "imp-1", "bidfloor": 1.0, "banner": {"w": 300, "h": 250}}]}
    out, prepared = auction_api._prepare_for_auction(body)

    ext = out["imp"][0]["ext"]
    assert ext["artfhouse"] == {}
    assert ext["amt"]["placementId"] == "artf-imp-1"
    assert out["ext"]["prebid"]["returnallbidstatus"] is True
    assert out["ext"]["prebid"]["targeting"]["includewinners"] is True
    assert out["ext"]["prebid"]["multibid"] == [
        {"bidder": "artfhouse", "maxbids": 3},
        {"bidder": "amt", "maxbids": 3},
    ]
    assert len(prepared) == 5


def test_preparation_adds_nothing_that_constitutes_a_bid():
    body = {"id": "r", "imp": [{"id": "imp-1", "bidfloor": 1.0}]}
    out, _ = auction_api._prepare_for_auction(body)

    imp = out["imp"][0]
    # No price, no creative, no deal, no campaign, and no category invented.
    assert "pmp" not in imp
    assert imp["bidfloor"] == 1.0
    assert "artf" not in imp["ext"]
    assert "data" not in imp["ext"]
    assert "seatbid" not in out


def test_existing_bidder_params_are_left_alone():
    # A caller that already declared its own params keeps them: preparation fills
    # gaps, it does not overwrite intent.
    body = {
        "id": "r",
        "imp": [{"id": "imp-1", "ext": {"amt": {"placementId": "mine"}, "artfhouse": {"x": 1}}}],
    }
    out, prepared = auction_api._prepare_for_auction(body)

    assert out["imp"][0]["ext"]["amt"] == {"placementId": "mine"}
    assert out["imp"][0]["ext"]["artfhouse"] == {"x": 1}
    assert not any("ext.amt" in p or "ext.artfhouse" in p for p in prepared)


def test_categories_are_relocated_not_invented():
    body = {"id": "r", "imp": [{"id": "imp-1", "ext": {"artf": {"categories": ["home"]}}}]}
    out, prepared = auction_api._prepare_for_auction(body)

    # Same value, new location -- the one Prebid preserves.
    assert out["imp"][0]["ext"]["data"]["artf"]["categories"] == ["home"]
    assert any("data.artf.categories" in p for p in prepared)


def test_an_impression_without_categories_gains_none():
    body = {"id": "r", "imp": [{"id": "imp-1"}]}
    out, prepared = auction_api._prepare_for_auction(body)

    assert "data" not in out["imp"][0]["ext"]
    assert not any("categories" in p for p in prepared)


def test_an_existing_data_location_is_not_overwritten():
    body = {
        "id": "r",
        "imp": [{
            "id": "imp-1",
            "ext": {
                "artf": {"categories": ["home"]},
                "data": {"artf": {"categories": ["finance"]}},
            },
        }],
    }
    out, _ = auction_api._prepare_for_auction(body)

    # The Prebid-safe location is authoritative; the legacy one does not clobber it.
    assert out["imp"][0]["ext"]["data"]["artf"]["categories"] == ["finance"]


def test_preparation_is_reported_so_it_is_visible(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, dict(AUCTION))))

    body = {"id": "r", "imp": [{"id": "imp-1"}]}
    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(body))))

    prepared = payload["artf_meta"]["prepared"]
    assert any("artfhouse" in p for p in prepared)
    assert any("amt" in p for p in prepared)


def test_preparation_does_not_mutate_the_caller_s_request():
    body = {"id": "r", "imp": [{"id": "imp-1", "ext": {"artf": {"categories": ["home"]}}}]}
    snapshot = json.loads(json.dumps(body))
    auction_api._prepare_for_auction(body)
    assert body == snapshot


def test_a_plain_text_rejection_is_reported_as_a_rejection(monkeypatch):
    # Prebid answers a malformed bid request with 400 and a bare sentence, not
    # JSON. Calling that "non-JSON" buries the reason, which is the one thing the
    # caller needs since the fault is in the request they sent. Example: a
    # scenario carrying pmp.private_auction as a JSON boolean where OpenRTB
    # specifies an integer is rejected as a whole request.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    message = (
        "Invalid request format: Error decoding bidRequest: Cannot deserialize value "
        "of type `java.lang.Integer` from Boolean value"
    )
    _patch_client(monkeypatch, FakeClient(FakeResponse(400, None, text=message)))

    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST)))
    payload = body_of(response)

    assert response.status_code == 502
    assert payload["error"] == "prebid_rejected_request"
    assert payload["status"] == 400
    # The reason survives to the caller rather than being replaced by a category.
    assert "java.lang.Integer" in payload["detail"]


def test_a_non_json_body_with_a_200_is_still_reported_as_non_json(monkeypatch):
    # The other branch: a 200 whose body is not a bid response at all.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, None, text="<html>nope</html>")))

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST))))
    assert payload["error"] == "prebid_returned_non_json"


# --------------------------------------------------------------------------- #
# Targeting: without it Prebid states no winner, and every scenario reads unsold.
#
# The scenario payloads under source/frontend-react/public/samples/ carry no
# ext.prebid at all. Prebid emits seatbid[].bid[].ext.prebid.targeting only when
# the request asks for targeting, so before this was added every live auction
# returned bids that no consumer could attribute a win to -- and the theater
# rendered "No winning offer" on all six scenarios while the auction itself was
# working. Verified against the deployed server: identical request, targeting
# absent -> no bid carries targeting; targeting requested -> the top bid carries
# hb_bidder / hb_pb / hb_deal.
# --------------------------------------------------------------------------- #


def test_targeting_is_requested_so_prebid_names_a_winner():
    body = {"id": "r", "imp": [{"id": "imp-1", "banner": {"w": 300, "h": 250}}]}
    out, prepared = auction_api._prepare_for_auction(body)
    assert out["ext"]["prebid"]["targeting"] == {"includewinners": True}
    assert any("includewinners" in note for note in prepared)


def test_a_callers_own_targeting_is_never_overwritten():
    # An operator asking for a different price granularity or preferdeals keeps it;
    # this layer supplies a default, it does not impose a policy.
    body = {
        "id": "r",
        "imp": [{"id": "imp-1"}],
        "ext": {"prebid": {"targeting": {"includewinners": False, "preferdeals": True}}},
    }
    out, prepared = auction_api._prepare_for_auction(body)
    assert out["ext"]["prebid"]["targeting"] == {
        "includewinners": False,
        "preferdeals": True,
    }
    assert not any("includewinners" in note for note in prepared)


def test_targeting_request_adds_no_bid_and_no_price():
    body = {"id": "r", "imp": [{"id": "imp-1", "bidfloor": 2.5}]}
    out, _ = auction_api._prepare_for_auction(body)
    # Asking to be told the winner is not the same as supplying one.
    assert "seatbid" not in out
    assert out["imp"][0]["bidfloor"] == 2.5
    assert "pricegranularity" not in out["ext"]["prebid"]["targeting"]


def test_the_callers_dict_is_not_mutated_by_preparation():
    body = {"id": "r", "imp": [{"id": "imp-1"}]}
    auction_api._prepare_for_auction(body)
    assert "ext" not in body
    assert body["imp"][0] == {"id": "imp-1"}


# --------------------------------------------------------------------------- #
# Seat non-bid annotation: the price a seat returned, against the floor it faced.
#
# Prebid reports a seat whose bid was unusable as NonBidReason 0 (NO_BID) and says
# nothing else, so "the floor turned this demand away" and "the seat answered with
# nothing" arrive identical. The bidder's own response IS in ext.debug.httpcalls,
# so the price is recoverable. Verified live: in isv-ecosystem the amt simulator
# returned 3.25 for imp-1 (floor 4.00) and 2.75 for imp-2 (floor 3.00), and Prebid
# reported both as statuscode 0.
#
# What is deliberately NOT done: claiming the floor was the cause. Prebid said
# NO_BID. These tests pin the numbers and the absence of a verdict.
# --------------------------------------------------------------------------- #


def _auction_with_debug(sim_bids, *, seat="amt", nonbids=(("imp-1", 0),), cur="USD"):
    """An auction response shaped like Prebid's, with a debug httpcalls block."""
    return {
        "id": "r",
        "cur": cur,
        "seatbid": [],
        "ext": {
            "seatnonbid": [
                {"seat": seat, "nonbid": [{"impid": i, "statuscode": c} for i, c in nonbids]}
            ],
            "debug": {
                "httpcalls": {
                    seat: [
                        {
                            "uri": "http://amt-simulator/amt-exchange",
                            "responsebody": json.dumps(
                                {"seatbid": [{"bid": [dict(b) for b in sim_bids]}]}
                            ),
                        }
                    ]
                }
            },
        },
    }


REQUEST_TWO_IMPS = {
    "id": "r",
    "imp": [{"id": "imp-1", "bidfloor": 4.0}, {"id": "imp-2", "bidfloor": 3.0}],
}


def test_seat_nonbid_carries_the_price_it_returned_and_the_floor_it_faced():
    auction = _auction_with_debug(
        [{"impid": "imp-1", "price": 3.25}, {"impid": "imp-2", "price": 2.75}],
        nonbids=(("imp-1", 0), ("imp-2", 0)),
    )
    count = auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS)
    assert count == 2

    entries = auction["ext"]["seatnonbid"][0]["nonbid"]
    by_imp = {e["impid"]: e["ext"]["artf"]["observed"] for e in entries}
    assert by_imp["imp-1"]["returnedPrice"] == 3.25
    assert by_imp["imp-1"]["impFloor"] == 4.0
    assert by_imp["imp-2"]["returnedPrice"] == 2.75
    assert by_imp["imp-2"]["impFloor"] == 3.0
    assert by_imp["imp-1"]["currency"] == "USD"
    assert "debug httpcalls" in by_imp["imp-1"]["source"]


def test_the_annotation_states_no_reason_and_no_verdict():
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}])
    auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS)
    observed = auction["ext"]["seatnonbid"][0]["nonbid"][0]["ext"]["artf"]["observed"]
    # Two numbers and their provenance. No exclusionReason, no "below_floor", no
    # claim about why Prebid dropped the bid -- Prebid said NO_BID and only Prebid
    # knows whether the floor or a validation rule discarded it.
    assert set(observed) == {"returnedPrice", "impFloor", "currency", "source"}
    assert "exclusionReason" not in auction["ext"]["seatnonbid"][0]["nonbid"][0]["ext"]["artf"]


def test_prebids_own_fields_on_the_nonbid_are_left_alone():
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}])
    auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS)
    entry = auction["ext"]["seatnonbid"][0]["nonbid"][0]
    assert entry["impid"] == "imp-1"
    assert entry["statuscode"] == 0


def test_a_campaign_identity_already_on_the_nonbid_survives_annotation():
    # The demand endpoint puts campaign identity in the same ext.artf namespace.
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}], seat="artfhouse")
    auction["ext"]["seatnonbid"][0]["nonbid"][0]["ext"] = {
        "artf": {"campaignId": "camp-x", "campaignName": "Camp X", "exclusionReason": "below_floor"}
    }
    auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS)
    artf = auction["ext"]["seatnonbid"][0]["nonbid"][0]["ext"]["artf"]
    assert artf["campaignId"] == "camp-x"
    assert artf["exclusionReason"] == "below_floor"
    assert artf["observed"]["returnedPrice"] == 3.25


def test_no_annotation_when_the_floor_is_unknown():
    # An imp with no bidfloor: half an observation is worse than none, because
    # "returned 3.25 against a floor of —" reads as a floor that was looked up and
    # came back empty.
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}])
    count = auction_api._annotate_seat_nonbids(auction, {"id": "r", "imp": [{"id": "imp-1"}]})
    assert count == 0
    assert "ext" not in auction["ext"]["seatnonbid"][0]["nonbid"][0]


def test_no_annotation_when_the_debug_block_is_absent():
    auction = {
        "id": "r",
        "ext": {"seatnonbid": [{"seat": "amt", "nonbid": [{"impid": "imp-1", "statuscode": 0}]}]},
    }
    assert auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS) == 0
    assert "ext" not in auction["ext"]["seatnonbid"][0]["nonbid"][0]


def test_a_price_is_matched_to_its_own_seat_only():
    # Two seats, one price each. Keying on impid alone would attribute amt's price
    # to artfhouse's non-bid on the same impression.
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}], seat="amt")
    auction["ext"]["seatnonbid"].append(
        {"seat": "artfhouse", "nonbid": [{"impid": "imp-1", "statuscode": 0}]}
    )
    auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS)
    amt, artfhouse = auction["ext"]["seatnonbid"]
    assert amt["nonbid"][0]["ext"]["artf"]["observed"]["returnedPrice"] == 3.25
    assert "ext" not in artfhouse["nonbid"][0]


def test_a_boolean_price_is_not_a_price():
    # json true is an int subclass in Python; it must not become a CPM.
    auction = _auction_with_debug([{"impid": "imp-1", "price": True}])
    assert auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS) == 0


def test_unparseable_debug_body_annotates_nothing_and_does_not_raise():
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}])
    auction["ext"]["debug"]["httpcalls"]["amt"][0]["responsebody"] = "not json"
    assert auction_api._annotate_seat_nonbids(auction, REQUEST_TWO_IMPS) == 0


def test_floors_come_from_the_prepared_request_so_a_raised_floor_is_the_one_reported():
    # A container raising the floor to 9.00 before the auction means 9.00 is what the
    # demand faced, and 9.00 is what a reader must be shown.
    auction = _auction_with_debug([{"impid": "imp-1", "price": 3.25}])
    auction_api._annotate_seat_nonbids(auction, {"id": "r", "imp": [{"id": "imp-1", "bidfloor": 9.0}]})
    observed = auction["ext"]["seatnonbid"][0]["nonbid"][0]["ext"]["artf"]["observed"]
    assert observed["impFloor"] == 9.0


def test_the_handler_wires_the_annotation_and_reports_the_count(monkeypatch):
    # The unit tests above call _annotate_seat_nonbids directly, which passes whether
    # or not the handler ever calls it. This one goes through run_auction_handler.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    auction = _auction_with_debug(
        [{"impid": "imp-1", "price": 3.25}], nonbids=(("imp-1", 0),)
    )
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, auction)))

    request = {"id": "r", "imp": [{"id": "imp-1", "bidfloor": 4.0}]}
    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(request))))

    assert payload["artf_meta"]["seat_nonbids_annotated"] == 1
    observed = payload["ext"]["seatnonbid"][0]["nonbid"][0]["ext"]["artf"]["observed"]
    assert observed["returnedPrice"] == 3.25
    assert observed["impFloor"] == 4.0


def test_the_count_is_zero_rather_than_absent_when_nothing_could_be_annotated(monkeypatch):
    # Zero with a non-empty seatnonbid tells a consumer the debug block did not name
    # the price, which is different from there being no non-bids.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    auction = {
        "id": "r",
        "ext": {"seatnonbid": [{"seat": "amt", "nonbid": [{"impid": "imp-1", "statuscode": 0}]}]},
    }
    _patch_client(monkeypatch, FakeClient(FakeResponse(200, auction)))
    payload = body_of(
        run(auction_api.run_auction_handler(FakeRequest({"id": "r", "imp": [{"id": "imp-1", "bidfloor": 4.0}]})))
    )
    assert payload["artf_meta"]["seat_nonbids_annotated"] == 0
    assert payload["ext"]["seatnonbid"][0]["nonbid"][0] == {"impid": "imp-1", "statuscode": 0}


# --------------------------------------------------------------------------- #
# The `artf` mode: a per-request, opt-in way to run an auction with the ARTF
# extension point proposing nothing. Exists for the Theater's baseline pass.
#
# What these tests protect: the DEFAULT. An absent parameter, an empty parameter,
# and an explicit `on` all run the auction with ARTF, and nothing but an explicit
# `off` on a single request can change that. There is no environment variable and
# no config flag -- an operator cannot accidentally run the stack without ARTF.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", [None, {}, {"artf": ""}, {"artf": "on"}, {"artf": "ON"}])
def test_the_default_is_artf_on_and_no_marker_is_stamped(monkeypatch, query):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST, query=query))))

    sent = client.posted["json"]
    assert "artf" not in sent["ext"]["prebid"]
    assert payload["artf_meta"]["artf_mutations"] == "requested"
    assert not any("bypass" in note for note in payload["artf_meta"]["prepared"])


def test_artf_off_stamps_the_bypass_marker_and_reports_it(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    payload = body_of(
        run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST, query={"artf": "off"})))
    )

    sent = client.posted["json"]
    # The marker is the exact boolean the orchestrator's /v1/mutations checks for,
    # on the TOP-LEVEL ext (Prebid drops unknown keys under ext.prebid).
    assert sent["ext"]["artf"]["bypass"] is True
    assert "artf" not in sent["ext"]["prebid"]
    # Everything else the endpoint adds is still added: a baseline auction is the
    # same auction, minus the extension point.
    assert sent["ext"]["prebid"]["debug"] == 1
    assert sent["ext"]["prebid"]["targeting"]["includewinners"] is True
    assert sent["imp"][0]["ext"]["artfhouse"] == {}
    # Stated on the response, and listed among what was added to the request.
    assert payload["artf_meta"]["artf_mutations"] == "bypassed"
    assert any("ext.artf.bypass" in note for note in payload["artf_meta"]["prepared"])


@pytest.mark.parametrize("bad", ["true", "1", "yes", "false", "0", "disabled", "off; drop"])
def test_a_value_outside_the_allowlist_is_a_400(monkeypatch, bad):
    # SECURITY-05: an allowlist, not a truthiness check. "true" is not "off", and a
    # request that asked for something unrecognised must not quietly get either
    # behaviour.
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    response = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST, query={"artf": bad})))

    assert response.status_code == 400
    assert body_of(response)["error"] == "invalid_artf_mode"
    # Rejected before the hop: Prebid never saw the request.
    assert client.posted is None


def test_the_bypass_marker_keeps_what_the_caller_already_put_under_artf():
    body = {
        "id": "r",
        "imp": [{"id": "imp-1"}],
        "ext": {"artf": {"note": "kept"}, "prebid": {"debug": 1}},
    }
    snapshot = json.loads(json.dumps(body))

    out, note = auction_api._with_artf_bypass(body)

    assert out["ext"]["artf"] == {"note": "kept", "bypass": True}
    # Siblings under ext are untouched; nothing is written under ext.prebid.
    assert out["ext"]["prebid"] == {"debug": 1}
    assert "bypass" in note
    # Shallow copies all the way down: the caller's dict is untouched.
    assert body == snapshot


def test_artf_off_is_rejected_the_same_way_when_prebid_is_absent():
    # The mode is validated, but a 501 for "not deployed" still wins over a 200 of
    # anything. No auction, with or without ARTF, is invented in Prebid's absence.
    response = run(
        auction_api.run_auction_handler(FakeRequest(VALID_REQUEST, query={"artf": "off"}))
    )
    assert response.status_code == 501
    assert body_of(response)["error"] == "prebid_not_deployed"


# --------------------------------------------------------------- intents param


def test_intents_are_stated_on_top_level_ext_artf_and_do_not_touch_ext_prebid(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    payload = body_of(run(auction_api.run_auction_handler(FakeRequest(
        VALID_REQUEST, query={"artf": "on", "intents": "ACTIVATE_SEGMENTS,ADD_METRICS, activate_deals,ADD_METRICS"},
    ))))

    sent = client.posted["json"]
    # Normalised, de-duplicated, order kept; on the ext Prebid carries through.
    assert sent["ext"]["artf"]["applicable_intents"] == ["ACTIVATE_SEGMENTS", "ADD_METRICS", "ACTIVATE_DEALS"]
    assert "artf" not in sent["ext"]["prebid"]
    # Not a bypass: the marker is a separate key and is absent here.
    assert "bypass" not in sent["ext"]["artf"]
    assert payload["artf_meta"]["artf_mutations"] == "requested"
    assert any("ext.artf.applicable_intents" in note for note in payload["artf_meta"]["prepared"])


def test_intents_and_bypass_share_ext_artf_without_clobbering(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    run(auction_api.run_auction_handler(FakeRequest(
        VALID_REQUEST, query={"artf": "off", "intents": "ACTIVATE_DEALS"},
    )))

    sent = client.posted["json"]
    assert sent["ext"]["artf"] == {"bypass": True, "applicable_intents": ["ACTIVATE_DEALS"]}


def test_absent_or_empty_intents_add_nothing(monkeypatch):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST, query={"intents": "  "})))
    assert "artf" not in client.posted["json"]["ext"]


@pytest.mark.parametrize("bad", ["ACTIVATE_SEGMENTS,BOGUS", "1,2", "ADD_METRICS;x", "<script>"])
def test_an_unknown_intent_is_a_400_not_a_silent_drop(monkeypatch, bad):
    monkeypatch.setenv(auction_api.PREBID_AUCTION_URL_ENV, "https://prebid/openrtb2/auction")
    client = FakeClient(FakeResponse(200, dict(AUCTION)))
    _patch_client(monkeypatch, client)

    resp = run(auction_api.run_auction_handler(FakeRequest(VALID_REQUEST, query={"intents": bad})))
    assert resp.status_code == 400
    assert body_of(resp)["error"] == "invalid_intents"
    assert client.posted is None
