"""The request handler: status codes, and what must not reach a caller or a log."""
import json

import pytest

from demand.artfhouse.handler import handle
from demand.artfhouse.service import DemandDecisionService


@pytest.fixture(scope="module")
def service():
    # Built from DEALS_SEED_FILE (tests/conftest.py), as the snapshot builds it.
    return DemandDecisionService()


def valid_request(deals=(), floor=1.0):
    return {
        "id": "req-1",
        "imp": [{"id": "imp-1", "bidfloor": floor, "pmp": {"deals": list(deals)}}],
    }


def raw(body):
    return json.dumps(body).encode()


def test_a_valid_request_returns_200_and_a_bid_response(service):
    status, body = handle(service, raw(valid_request([{"id": "deal-home-premium"}])))
    assert status == 200
    assert body["id"] == "req-1"
    assert body["seatbid"][0]["seat"] == "artfhouse"


def test_unparseable_body_is_400_not_an_empty_bid_response(service):
    status, body = handle(service, b"{not json")
    assert status == 400
    assert "seatbid" not in body


def test_a_missing_body_is_400(service):
    assert handle(service, None)[0] == 400


def test_a_json_array_body_is_400_rather_than_being_coerced(service):
    assert handle(service, b"[1,2,3]")[0] == 400


def test_a_structurally_invalid_request_is_400_with_a_reason(service):
    status, body = handle(service, raw({"imp": []}))
    assert status == 400
    assert "malformed" in body["error"]


def test_an_unsupported_currency_is_400(service):
    request = valid_request()
    request["cur"] = "GBP"
    status, body = handle(service, raw(request))
    assert status == 400
    assert "GBP" in body["error"]


class Boom:
    def decide(self, _request):
        raise RuntimeError("/srv/secret/path exploded")


def test_an_unexpected_error_is_500_with_a_generic_message():
    status, body = handle(Boom(), raw(valid_request()))
    assert status == 500
    assert body == {"error": "internal error"}


def test_an_unexpected_error_does_not_leak_internals_to_the_caller():
    text = json.dumps(handle(Boom(), raw(valid_request()))[1])
    assert "/srv/secret/path" not in text
    assert "RuntimeError" not in text
    assert "Traceback" not in text


def test_no_bid_request_content_is_logged(service, caplog):
    """A bid request carries user data; this endpoint has no reason to log it."""
    request = valid_request([{"id": "deal-home-premium"}])
    request["user"] = {"id": "user-should-not-appear", "buyeruid": "buyer-should-not-appear"}
    request["device"] = {"ip": "203.0.113.7"}
    with caplog.at_level("INFO", logger="artfhouse.demand"):
        handle(service, raw(request))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "user-should-not-appear" not in logged
    assert "buyer-should-not-appear" not in logged
    assert "203.0.113.7" not in logged


def test_the_log_carries_the_request_id_and_counts_for_correlation(service, caplog):
    with caplog.at_level("INFO", logger="artfhouse.demand"):
        handle(service, raw(valid_request([{"id": "deal-home-premium"}])))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "req-1" in logged
    assert "offers" in logged
    assert "excluded" in logged
