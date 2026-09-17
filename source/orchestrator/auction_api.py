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

    body, prepared = _prepare_for_auction(body)
    body = _with_debug_enabled(body)

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
        # A rejection is reported as a rejection even when its body is plain text.
        # Prebid answers a malformed bid request with 400 and a bare sentence, and
        # calling that "non-JSON" buries the actual reason -- which is the one thing
        # the caller needs, since the fault is in the request they sent.
        if response.status_code != 200:
            return JSONResponse(
                {
                    "error": "prebid_rejected_request",
                    "status": response.status_code,
                    "detail": response.text[:400],
                },
                status_code=502,
            )
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
        excluded, excluded_source = _lift_artf_exclusions(auction)
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
            # Where the exclusion list came from, or why there is none. A consumer
            # showing "no campaign was excluded" needs to know the difference
            # between an empty list and an unavailable one.
            "excluded_source": excluded_source,
            # What this endpoint added to the request before submitting it. Listed
            # so a reader can see that the auction was ENABLED, not authored.
            "prepared": prepared,
        }
        if excluded is not None:
            auction.setdefault("ext", {}).setdefault("artf", {})["excluded"] = excluded
    return JSONResponse(auction)


def _prepare_for_auction(body: dict) -> tuple[dict, list[str]]:
    """Make an ARTF-shaped bid request runnable as a Prebid auction.

    WHAT THIS DOES, AND WHAT IT REFUSES TO DO

    Prebid routes an impression to a bidder only when `imp.ext.<bidder>` is
    present -- that key is how a publisher's prebid.js configuration declares which
    bidders to call. The ARTF scenarios are bidstream requests and carry no such
    keys, so submitted verbatim they reach NO bidder and return an empty auction
    that looks like nobody wanted the impression.

    So this adds the bidder enablement a publisher's page would have supplied, and
    NOTHING else. It does not add a bid, a price, a campaign, a deal, or a category:
    what each seat then bids is entirely the bidders' own answer. Everything added
    is listed in artf_meta.prepared so it is visible rather than implied.

    It also moves `imp.ext.artf.categories` to `imp.ext.data.artf.categories`,
    because Prebid deletes unrecognised `imp.ext` keys as unknown bidder names --
    the same relocation the demand endpoint reads. The VALUE is untouched; only its
    location changes, so a scenario that declares categories keeps them and one
    that declares none gains none.
    """
    prepared: list[str] = []
    out = dict(body)
    imps = []

    for imp in out.get("imp") or []:
        imp = dict(imp)
        ext = dict(imp.get("ext") or {})

        if "artfhouse" not in ext:
            # Empty on purpose: the adapter takes no parameters. Presence is what
            # routes the impression to it.
            ext["artfhouse"] = {}
            prepared.append(f"imp[{imp.get('id')}].ext.artfhouse (enable the ARTF seat)")

        if "amt" not in ext:
            # placementId is required by the AMT adapter's params schema
            # (minLength 1). Derived from the impression id so it is stable and
            # traceable; the simulator does not vary its answer by placement.
            ext["amt"] = {"placementId": f"artf-{imp.get('id') or 'imp'}"}
            prepared.append(f"imp[{imp.get('id')}].ext.amt.placementId (enable the simulator seat)")

        artf = ext.get("artf")
        if isinstance(artf, dict) and artf.get("categories"):
            data = dict(ext.get("data") or {})
            data_artf = dict(data.get("artf") or {})
            if not data_artf.get("categories"):
                data_artf["categories"] = artf["categories"]
                data["artf"] = data_artf
                ext["data"] = data
                prepared.append(
                    f"imp[{imp.get('id')}].ext.data.artf.categories "
                    f"(moved from imp.ext.artf, which Prebid drops)"
                )

        imp["ext"] = ext
        imps.append(imp)

    if imps:
        out["imp"] = imps

    ext = dict(out.get("ext") or {})
    prebid = dict(ext.get("prebid") or {})
    if "returnallbidstatus" not in prebid:
        # Without it a seat that was called and did not bid is indistinguishable
        # from a seat that was never called, so the losers cannot be shown.
        prebid["returnallbidstatus"] = True
        prepared.append("ext.prebid.returnallbidstatus (report the seats that did not bid)")
    ext["prebid"] = prebid
    out["ext"] = ext

    return out, prepared


def _with_debug_enabled(body: dict) -> dict:
    """Ask Prebid for its debug output, which is what carries the exclusions.

    A bidder's own response body is not part of an OpenRTB bid response: Prebid
    returns bids, and the artfhouse adapter's makeBids extracts bids and nothing
    else. So the per-campaign exclusion reasons the demand endpoint reports --
    which campaign lost, and why -- reach us only inside
    ext.debug.httpcalls.artfhouse[].responsebody.

    Set here rather than left to the caller because the screen this feeds always
    needs the losers, and a request that forgot the flag would silently show an
    auction with no explanation of who was excluded.

    Copied shallowly, with ext and ext.prebid copied too, so a caller's dict is not
    mutated as a side effect of being posted.
    """
    out = dict(body)
    ext = dict(out.get("ext") or {})
    prebid = dict(ext.get("prebid") or {})
    prebid["debug"] = 1
    ext["prebid"] = prebid
    out["ext"] = ext
    return out


def _lift_artf_exclusions(auction: dict) -> tuple[Optional[list], str]:
    """Pull the demand endpoint's own exclusion list out of Prebid's debug output.

    Returns (excluded, source). `excluded` is None when the list is genuinely
    unavailable, never [] -- an empty list asserts "nothing was excluded", which is
    a different claim from "we could not tell".

    This is the demand endpoint's own account of its decision, carried verbatim by
    Prebid. Nothing here reconstructs or infers a reason: if the response does not
    contain one, none is reported.
    """
    debug = (auction.get("ext") or {}).get("debug") or {}
    httpcalls = debug.get("httpcalls") or {}
    if not httpcalls:
        return None, "unavailable: prebid returned no httpcalls"

    calls = httpcalls.get("artfhouse") or []
    if not calls:
        return None, "unavailable: artfhouse was not called"

    for call in calls:
        raw = call.get("responsebody")
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        artf = (parsed.get("ext") or {}).get("artf") or {}
        if "excluded" in artf:
            return artf["excluded"], "artfhouse response, via prebid debug httpcalls"

    return None, "unavailable: no artf exclusion list in the artfhouse response"
