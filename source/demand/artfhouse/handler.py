"""Lambda entrypoint for the ``artfhouse`` demand endpoint.

Sits behind an API Gateway HTTP API with a Cognito JWT authorizer, OUTSIDE the
seller's EKS cluster. The placement is deliberate: this endpoint represents the
BUYER's demand, and putting it in the seller's cluster would collapse the party
separation the demonstration exists to show.

Authorization is API Gateway's, not this handler's. The handler validates structure
and decides; it does not authenticate.

Failure semantics:
  - a malformed request is a 400 ERROR, never an empty bid response. An empty
    response asserts that no campaign wished to offer, which is a substantive claim
    about demand and would be untrue of a request that could not be parsed;
  - an unsupported currency is a 400, not a silent rescale;
  - an unexpected error is a 500 with a generic message. No stack trace, no internal
    path and no request content reaches the caller.

Logging is structured and carries the request id for correlation. NO BID REQUEST
CONTENT IS LOGGED: a bid request carries user data, and this endpoint has no reason
to persist it anywhere.
"""

import json
import logging
import os
from typing import Any, Optional

from .floors import CurrencyMismatch
from .service import DemandDecisionService, MalformedRequest

logger = logging.getLogger("artfhouse.demand")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('{"level":"%(levelname)s","msg":%(message)s}'))
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

#: One service instance per container, so the catalog is built once.
_service = DemandDecisionService()


def _log(event: str, **fields: Any) -> None:
    """Structured log line. Only the fields passed here are logged."""
    logger.info(json.dumps({"event": event, **fields}))


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _parse_body(event: dict) -> Optional[dict]:
    """The bid request from an API Gateway proxy event, or a direct invocation."""
    if "body" in event:
        raw = event.get("body")
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            return raw if isinstance(raw, dict) else None
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    # Direct invocation: the event *is* the bid request.
    return event if isinstance(event, dict) else None


def lambda_handler(event: dict, context: Any = None) -> dict:
    """API Gateway proxy handler."""
    bid_request = _parse_body(event)

    if bid_request is None:
        _log("request_unreadable")
        return _response(400, {"error": "request body is not a JSON object"})

    request_id = bid_request.get("id")

    try:
        bid_response = _service.decide(bid_request)
    except MalformedRequest as exc:
        # The reason is safe to return: it names structural problems in the caller's
        # own request and reveals nothing about this service.
        _log("request_malformed", request_id=request_id, reason=str(exc))
        return _response(400, {"error": f"malformed bid request: {exc}"})
    except CurrencyMismatch as exc:
        _log("currency_unsupported", request_id=request_id, reason=str(exc))
        return _response(400, {"error": str(exc)})
    except Exception:  # noqa: BLE001 - last resort, must not leak internals
        logger.exception(json.dumps({"event": "unhandled_error", "request_id": request_id}))
        return _response(500, {"error": "internal error"})

    offers = len(bid_response.get("seatbid", [{}])[0].get("bid", [])) if bid_response.get("seatbid") else 0
    excluded = len(bid_response.get("ext", {}).get("artf", {}).get("excluded", []))
    _log("decided", request_id=request_id, offers=offers, excluded=excluded)

    return _response(200, bid_response)


#: Alias for deployments that expect the conventional name.
handler = lambda_handler
