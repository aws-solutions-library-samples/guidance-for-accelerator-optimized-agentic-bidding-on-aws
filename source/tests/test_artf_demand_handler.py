"""The Lambda handler: status codes, and what must not reach a caller or a log."""

import json

from demand.artfhouse import handler as handler_module
from demand.artfhouse.handler import lambda_handler


def proxy_event(body):
    return {"body": json.dumps(body) if not isinstance(body, str) else body}


def valid_request(deals=(), floor=1.0):
    return {
        "id": "req-1",
        "imp": [{"id": "imp-1", "bidfloor": floor, "pmp": {"deals": list(deals)}}],
    }


def body_of(response):
    return json.loads(response["body"])


def test_a_valid_request_returns_200_and_a_bid_response():
    response = lambda_handler(proxy_event(valid_request([{"id": "deal-home-premium"}])))
    assert response["statusCode"] == 200
    body = body_of(response)
    assert body["id"] == "req-1"
    assert body["seatbid"][0]["seat"] == "artfhouse"


def test_a_direct_invocation_is_accepted_as_the_bid_request_itself():
    response = lambda_handler(valid_request())
    assert response["statusCode"] == 200


def test_unparseable_body_is_400_not_an_empty_bid_response():
    response = lambda_handler({"body": "{not json"})
    assert response["statusCode"] == 400
    assert "seatbid" not in body_of(response)


def test_a_missing_body_is_400():
    assert lambda_handler({"body": None})["statusCode"] == 400


def test_a_json_array_body_is_400_rather_than_being_coerced():
    assert lambda_handler({"body": "[1,2,3]"})["statusCode"] == 400


def test_a_structurally_invalid_request_is_400_with_a_reason():
    response = lambda_handler(proxy_event({"imp": []}))
    assert response["statusCode"] == 400
    assert "malformed" in body_of(response)["error"]


def test_an_unsupported_currency_is_400():
    request = valid_request()
    request["cur"] = "GBP"
    response = lambda_handler(proxy_event(request))
    assert response["statusCode"] == 400
    assert "GBP" in body_of(response)["error"]


def test_an_unexpected_error_is_500_with_a_generic_message(monkeypatch):
    class Boom:
        def decide(self, _request):
            raise RuntimeError("internal detail that must not leak /srv/secret/path")

    monkeypatch.setattr(handler_module, "_service", Boom())
    response = lambda_handler(proxy_event(valid_request()))
    assert response["statusCode"] == 500
    assert body_of(response) == {"error": "internal error"}


def test_an_unexpected_error_does_not_leak_internals_to_the_caller(monkeypatch):
    class Boom:
        def decide(self, _request):
            raise RuntimeError("/srv/secret/path exploded")

    monkeypatch.setattr(handler_module, "_service", Boom())
    body = lambda_handler(proxy_event(valid_request()))["body"]
    assert "/srv/secret/path" not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body


def test_the_response_declares_json(caplog):
    response = lambda_handler(proxy_event(valid_request()))
    assert response["headers"]["Content-Type"] == "application/json"


def test_no_bid_request_content_is_logged(caplog):
    """A bid request carries user data; this endpoint has no reason to log it."""
    request = valid_request([{"id": "deal-home-premium"}])
    request["user"] = {"id": "user-should-not-appear", "buyeruid": "buyer-should-not-appear"}
    request["device"] = {"ip": "203.0.113.7"}

    with caplog.at_level("INFO", logger="artfhouse.demand"):
        lambda_handler(proxy_event(request))

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "user-should-not-appear" not in logged
    assert "buyer-should-not-appear" not in logged
    assert "203.0.113.7" not in logged


def test_the_log_carries_the_request_id_and_counts_for_correlation(caplog):
    with caplog.at_level("INFO", logger="artfhouse.demand"):
        lambda_handler(proxy_event(valid_request([{"id": "deal-home-premium"}])))

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "req-1" in logged
    assert "offers" in logged
    assert "excluded" in logged
