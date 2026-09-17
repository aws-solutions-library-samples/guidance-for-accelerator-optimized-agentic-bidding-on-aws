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
    """Minimal stand-in for a Starlette Request: the handler reads only json()."""

    def __init__(self, body, raise_exc=None):
        self._body = body
        self._raise = raise_exc

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
    # And the request reached Prebid as given.
    assert client.posted["json"] == VALID_REQUEST


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
