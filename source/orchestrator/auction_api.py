"""Run a real auction through Prebid Server, from inside the cluster.

WHY THIS LIVES IN THE ORCHESTRATOR

Prebid Server is a ClusterIP Service on this cluster: it has no public address, and
it deliberately does not get one -- exposing it would publish an unauthenticated
auction endpoint. The browser therefore cannot call it. The orchestrator already
sits on the only public path the frontend uses (`/api/*` behind CloudFront), so it
is the natural place to make the hop.

WHAT THE SWITCH IS, AND WHAT IT IS NOT

PREBID_AUCTION_URL is set by deploy_prebid.sh and removed by its --destroy. When it
is absent, this endpoint says so and returns 501. It does NOT fall back to
replaying a stored response: a stored auction presented as a live one is the single
most misleading thing this endpoint could do, because the whole point of the screen
it feeds is that the competition is real.

The switch is deployment state, not a runtime probe. Probing the Service on each
request would make the answer depend on cluster conditions at that instant, so a
transient failure would read as "Prebid was never deployed".

TLS TO PREBID

Prebid's entrypoint generates a self-signed certificate at container start, so
there is no authority to verify it against and no stable key to pin. The hop is
cluster-internal to a named Service, and verification is disabled for that reason
alone -- stated here rather than left as an unexplained flag.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

# Set by deploy_prebid.sh, cleared by --destroy.
PREBID_AUCTION_URL_ENV = "PREBID_AUCTION_URL"

# The auction's own tmax governs bidder timeouts; this bounds the hop itself. Set
# above a cold start: the first auction after a pod restart was measured taking
# longer than a 1500 ms tmax while connections and credentials warm up, and a
# tighter client timeout here would report that as an orchestrator failure.
HOP_TIMEOUT_S = 15.0


def prebid_auction_url() -> Optional[str]:
    value = (os.environ.get(PREBID_AUCTION_URL_ENV) or "").strip()
    return value or None


def _status_payload() -> dict:
    url = prebid_auction_url()
    return {
        "prebid": "configured" if url else "not_configured",
        # The address is useful for diagnosis and carries no credential. It is not
        # reachable from outside the cluster.
        "endpoint": url,
        "detail": (
            "Prebid Server is deployed and reachable from the orchestrator."
            if url
            else
            "Prebid Server is not deployed. Deploy it with deploy.sh --with-prebid. "
            "No auction is simulated in its absence."
        ),
    }


async def auction_status_handler(request: Request) -> JSONResponse:
    """Report whether a live auction is available. Never guesses."""
    return JSONResponse(_status_payload())


async def run_auction_handler(request: Request) -> JSONResponse:
    """POST an OpenRTB request to Prebid and return its response unchanged.

    The response body is Prebid's own. Nothing is added to `seatbid`, no price is
    adjusted, and no field is synthesised: the consumer is a screen that claims to
    show a real auction, so what it receives has to be what the exchange said.
    Timing and the resolved endpoint are reported alongside it, under `artf_meta`,
    so a reader can tell where the numbers came from.
    """
    url = prebid_auction_url()
    if not url:
        # 501, not 200-with-something-plausible. An empty auction and an
        # undeployed exchange are different facts and must not render the same.
        return JSONResponse(
            {
                "error": "prebid_not_deployed",
                **_status_payload(),
            },
            status_code=501,
        )

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(
            {"error": "invalid_json", "detail": str(exc)}, status_code=400
        )

    if not isinstance(body, dict) or not body.get("imp"):
        return JSONResponse(
            {
                "error": "invalid_bid_request",
                "detail": "an OpenRTB bid request with at least one imp is required",
            },
            status_code=400,
        )

    started = time.monotonic()
    try:
        # verify=False: see the module docstring -- Prebid presents a certificate it
        # generated for itself at container start.
        async with httpx.AsyncClient(verify=False, timeout=HOP_TIMEOUT_S) as client:
            response = await client.post(
                url, json=body, headers={"Content-Type": "application/json"}
            )
    except httpx.TimeoutException as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.warning("auction hop to %s timed out after %dms: %s", url, elapsed_ms, exc)
        return JSONResponse(
            {
                "error": "prebid_timeout",
                "detail": f"no response from Prebid within {HOP_TIMEOUT_S:.0f}s",
                "elapsed_ms": elapsed_ms,
            },
            status_code=504,
        )
    except httpx.HTTPError as exc:
        log.warning("auction hop to %s failed: %s", url, exc)
        return JSONResponse(
            {"error": "prebid_unreachable", "detail": str(exc)}, status_code=502
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    try:
        auction = response.json()
    except ValueError:
        # Reported as what it is. Inventing a bid response here would put a
        # fabricated auction on the screen.
        return JSONResponse(
            {
                "error": "prebid_returned_non_json",
                "status": response.status_code,
                "body_prefix": response.text[:200],
            },
            status_code=502,
        )

    if response.status_code != 200:
        return JSONResponse(
            {
                "error": "prebid_rejected_request",
                "status": response.status_code,
                "response": auction,
            },
            status_code=502,
        )

    if isinstance(auction, dict):
        auction["artf_meta"] = {
            "source": "prebid",
            "endpoint": url,
            "hop_ms": elapsed_ms,
            "seats": sorted(
                {
                    seat.get("seat")
                    for seat in (auction.get("seatbid") or [])
                    if seat.get("seat")
                }
            ),
        }
    return JSONResponse(auction)
