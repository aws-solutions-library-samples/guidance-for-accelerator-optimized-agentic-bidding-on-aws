"""Request handling for the ``artfhouse`` demand endpoint, independent of transport.

Called by server.py once the bearer token has been checked. ``handle`` validates
structure and decides; it reads only the in-memory catalog it is given.

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
from .service import MalformedRequest

logger = logging.getLogger("artfhouse.demand")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('{"level":"%(levelname)s","msg":%(message)s}'))
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def _log(event: str, **fields: Any) -> None:
    """Structured log line. Only the fields passed here are logged."""
    logger.info(json.dumps({"event": event, **fields}))


def parse_body(raw: Any) -> Optional[dict]:
    """The bid request from a raw body (bytes, str or an already-parsed object)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def handle(service: Any, raw_body: Any) -> tuple[int, dict]:
    """(status, body) for one bid request. Never raises."""
    bid_request = parse_body(raw_body)
    if bid_request is None:
        _log("request_unreadable")
        return 400, {"error": "request body is not a JSON object"}
    request_id = bid_request.get("id")
    try:
        bid_response = service.decide(bid_request)
    except MalformedRequest as exc:
        # The reason is safe to return: it names structural problems in the caller's
        # own request and reveals nothing about this service.
        _log("request_malformed", request_id=request_id, reason=str(exc))
        return 400, {"error": f"malformed bid request: {exc}"}
    except CurrencyMismatch as exc:
        _log("currency_unsupported", request_id=request_id, reason=str(exc))
        return 400, {"error": str(exc)}
    except Exception:  # noqa: BLE001 - last resort, must not leak internals
        logger.exception(json.dumps({"event": "unhandled_error", "request_id": request_id}))
        return 500, {"error": "internal error"}
    offers = len(bid_response.get("seatbid", [{}])[0].get("bid", [])) if bid_response.get("seatbid") else 0
    excluded = len(bid_response.get("ext", {}).get("artf", {}).get("excluded", []))
    _log("decided", request_id=request_id, offers=offers, excluded=excluded)
    return 200, bid_response
