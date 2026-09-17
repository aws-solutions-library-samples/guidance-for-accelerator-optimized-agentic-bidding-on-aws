#!/usr/bin/env python3
"""Serve the AWS Prebid guidance's bidder simulator over plain HTTP, in-cluster.

WHY THIS WRAPPER EXISTS

The release ships the simulator as a Lambda handler fronted by an ALB. The handler
itself is a pure function of its input and needs nothing from Lambda, so the only
thing missing is something to turn an HTTP request into the event shape it expects
and its return value back into an HTTP response. That is all this file does.

The handler is used COMPLETELY UNMODIFIED. It is the release's file (Apache-2.0),
mounted next to this one at runtime and imported; it is never copied into this
repository, which is MIT-0.

WHY IN-CLUSTER RATHER THAN API GATEWAY

The caller is Prebid Server's `amt` adapter, running in this same cluster. That
adapter builds its outbound call with `BidderUtil.defaultRequest(...)`, which
sends default headers and nothing else -- no bearer token, no API key, no
signature -- and it is used unmodified. Any public endpoint for it would therefore
have to accept unauthenticated requests. Behind a ClusterIP Service there is no
public endpoint to authenticate: the only things that can reach it are pods in
this cluster.

WHAT IS REAL AND WHAT IS NOT

The competition is real: this is a second seat submitting its own bid, which
Prebid compares against the same floor as every other seat, producing a real
winner and a real reported loser. The PRICES are not market data -- they are the
release's static creatives at fixed CPMs. The simulator is labelled as such
everywhere it appears.

The release's handler can also inject artificial delays and timeouts, driven by
four probability variables. They all default to ZERO here. A demo that
manufactures its own failures to look lifelike is showing something that did not
happen; if you want that behaviour for load testing, set the variables
deliberately (see the README section on the simulator's knobs).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The release's handler is mounted alongside this file rather than installed, so
# the directory holding both has to be importable before the import below.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

try:
    import handler as upstream_handler  # the release's file, unmodified
except ImportError as exc:  # pragma: no cover - a deployment fault, not a code path
    print(
        f"FATAL: the release's simulator handler is not importable: {exc}\n"
        "It is mounted from a ConfigMap built by deploy_prebid.sh; an empty or "
        "partial mount means the ConfigMap was not created from the fetched release.",
        file=sys.stderr,
    )
    raise

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("amt-simulator")

BID_PATH = os.environ.get("AMT_SIMULATOR_PATH", "/amt-exchange")
PORT = int(os.environ.get("PORT", "8080"))

# The handler reads these with os.environ[...] and raises if any is missing, so
# they are defaulted here rather than left to fail at the first request. Zero
# means "never inject a delay or a timeout".
_REQUIRED_DEFAULTS = {
    "BID_RESPONSES_DELAY_PERCENTAGE": "0",
    "BID_RESPONSES_TIMEOUT_PERCENTAGE": "0",
    "A_BID_RESPONSE_DELAY_PROBABILITY": "0",
    "A_BID_RESPONSE_TIMEOUT_PROBABILITY": "0",
}
for _name, _default in _REQUIRED_DEFAULTS.items():
    os.environ.setdefault(_name, _default)


class SimulatorHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # Liveness and readiness. Deliberately NOT a probe of the bid path: a
        # readiness check that exercised the bidder would report the pod unready
        # for reasons that have nothing to do with whether it can serve.
        if self.path.rstrip("/") in ("/health", "/healthz", ""):
            self._send(200, b'{"status":"ok","role":"amt bid simulator"}', "application/json")
            return
        self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0].rstrip("/") != BID_PATH.rstrip("/"):
            self._send(404, b'{"error":"not found"}', "application/json")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""

        # The handler expects the ALB/API-Gateway event shape and reads only
        # `body`, as a string. Nothing else about the Lambda event is consulted,
        # which is why this adapter can be this thin.
        event = {"body": raw.decode("utf-8", errors="replace")}

        try:
            result = upstream_handler.lambda_handler(event, None)
        except Exception:
            # Reported as a 500 with nothing invented in its place. Returning a
            # synthetic bid here would put a fabricated price into an auction and
            # a fabricated winner on screen.
            log.exception("the simulator handler raised; returning 500 with no bid")
            self._send(502, b'{"error":"simulator handler failed"}', "application/json")
            return

        status = int(result.get("statusCode", 200))
        body = result.get("body") or ""
        if not isinstance(body, str):
            body = json.dumps(body)
        content_type = (result.get("headers") or {}).get("Content-Type", "application/json")
        self._send(status, body.encode("utf-8"), content_type)

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    log.info(
        "amt bid simulator listening on :%d, bid path %s "
        "(delay/timeout injection: %s/%s)",
        PORT,
        BID_PATH,
        os.environ["BID_RESPONSES_DELAY_PERCENTAGE"],
        os.environ["BID_RESPONSES_TIMEOUT_PERCENTAGE"],
    )
    ThreadingHTTPServer(("0.0.0.0", PORT), SimulatorHandler).serve_forever()


if __name__ == "__main__":
    main()
