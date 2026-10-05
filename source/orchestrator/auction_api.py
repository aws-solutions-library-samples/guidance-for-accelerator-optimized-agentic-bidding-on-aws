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

#: The seats this endpoint enables on an ARTF request. Named once: the imp-level
#: enablement below and the multibid request both need the same list, and a seat
#: enabled without a multibid entry silently loses its non-winning bids.
_BIDDERS = ("artfhouse", "amt")

#: Bids kept per seat per impression. Above 1 so a seat's losing offers survive into
#: the response; bounded so a bidder returning a long list cannot make the response
#: unbounded. Matches the ceiling the repo's own diagnostic fixture uses.
_MAX_BIDS_PER_BIDDER = 3

#: The `artf` query parameter: whether the ARTF extension point is asked to mutate
#: this auction. `on` is the default and the only behaviour when the parameter is
#: absent. `off` is per request and opt-in -- the Theater's baseline pass sends it,
#: nothing else does, and there is deliberately no environment variable or config
#: flag that turns it on for the whole stack. An allowlist, not a truthiness check
#: (SECURITY-05): anything other than these two values is a 400.
ARTF_MODE_PARAM = "artf"
ARTF_MODE_ON = "on"
ARTF_MODE_OFF = "off"
_ARTF_MODES = frozenset({ARTF_MODE_ON, ARTF_MODE_OFF})

#: Where the bypass marker travels: the request's TOP-LEVEL `ext.artf`, which Prebid
#: Server carries through as an opaque extension. It must not go under `ext.prebid`:
#: that object is parsed into a typed model and unknown keys are dropped before the
#: hook builds its envelope, so a marker there is silently lost and the auction runs
#: WITH every container while claiming otherwise. The
#: orchestrator's `/v1/mutations` reads the same key via `is_artf_bypass`.
ARTF_BYPASS_KEY = "bypass"


#: The `intents` query parameter: the scenario's ARTF `applicable_intents`, comma
#: separated. Carried onto the request's top-level `ext.artf.applicable_intents`
#: so the request Prebid receives STATES which intents the scenario called for.
#:
#: It does not narrow anything today, and that is deliberate. The hook asks the
#: extension point for its configured intent set regardless of the request (its
#: reader looks under `ext.prebid.artf`, which Prebid drops, and the set was opened
#: to the full list on purpose). This parameter is for the record, and for a
#: future hook that chooses to read it; the auction's mutations are decided by the
#: hook's configuration and the containers, not by this value.
#:
#: An allowlist of intent names (SECURITY-05): anything else is a 400.
INTENTS_PARAM = "intents"
INTENTS_KEY = "applicable_intents"
_INTENT_NAMES = frozenset(
    name for name in (
        "ACTIVATE_SEGMENTS", "ACTIVATE_DEALS", "SUPPRESS_DEALS", "ADJUST_DEAL_FLOOR",
        "ADJUST_DEAL_MARGIN", "BID_SHADE", "ADD_METRICS", "ADD_CIDS",
    )
)


def intents_of(request: Request) -> tuple[Optional[list[str]], Optional[str]]:
    """The validated `intents` list, or (None, None) when absent.

    Returns (None, reason) when a value is not on the allowlist, so the handler can
    name the bad entry rather than silently dropping it.
    """
    raw = request.query_params.get(INTENTS_PARAM)
    if raw is None or raw.strip() == "":
        return None, None
    names = [part.strip().upper() for part in raw.split(",") if part.strip()]
    bad = [n for n in names if n not in _INTENT_NAMES]
    if bad:
        return None, f"unknown intent(s): {', '.join(sorted(set(bad)))}"
    # De-duplicated, order kept.
    return list(dict.fromkeys(names)), None


def artf_mode_of(request: Request) -> Optional[str]:
    """The validated `artf` mode, or None when the value is not on the allowlist."""
    raw = request.query_params.get(ARTF_MODE_PARAM)
    if raw is None or raw == "":
        return ARTF_MODE_ON
    value = raw.strip().lower()
    return value if value in _ARTF_MODES else None


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

    artf_mode = artf_mode_of(request)
    if artf_mode is None:
        return JSONResponse(
            {
                "error": "invalid_artf_mode",
                "detail": (
                    f"'{ARTF_MODE_PARAM}' must be '{ARTF_MODE_ON}' or "
                    f"'{ARTF_MODE_OFF}'"
                ),
            },
            status_code=400,
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

    intents, intents_error = intents_of(request)
    if intents_error:
        return JSONResponse(
            {"error": "invalid_intents", "detail": intents_error}, status_code=400
        )

    body, prepared = _prepare_for_auction(body)
    if artf_mode == ARTF_MODE_OFF:
        body, note = _with_artf_bypass(body)
        prepared.append(note)
    if intents:
        body, note = _with_applicable_intents(body, intents)
        prepared.append(note)
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
        # `body` is the prepared request, so the floors reported are the ones Prebid
        # enforced -- including any a container raised.
        observed_count = _annotate_seat_nonbids(auction, body)
        auction["artf_meta"] = {
            "source": "prebid",
            "endpoint": url,
            "hop_ms": elapsed_ms,
            # Whether the ARTF extension point was asked to mutate this auction.
            # "bypassed" means the request carried the marker and the orchestrator
            # answered the hook with zero mutations before reaching any container;
            # "requested" is every other auction. A consumer comparing two auctions
            # needs this on the response itself, not inferred from which call it
            # made.
            "artf_mutations": "bypassed" if artf_mode == ARTF_MODE_OFF else "requested",
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
            # How many seat non-bids carry an observed price and floor. Zero with a
            # non-empty seatnonbid means the debug block did not name the seat's
            # price, which a consumer should report as unknown rather than absent.
            "seat_nonbids_annotated": observed_count,
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

    if "targeting" not in prebid:
        # Prebid emits seatbid[].bid[].ext.prebid.targeting ONLY when the request
        # asks for targeting, and `includewinners` is what puts the unprefixed
        # hb_bidder / hb_pb / hb_deal keys on the top bid for each impression. Those
        # keys ARE the winner: Prebid resolves the auction, and this is where it
        # records the result. Without them the response carries bids and no statement
        # of which one won, so a consumer either reports no winner or has to rank the
        # bids itself -- and ranking them itself would be inventing an auction result
        # that the exchange did not give.
        #
        # Requested here rather than in each scenario payload: the scenarios are ARTF
        # bidstream requests, and this is a requirement of running one as an auction,
        # exactly like the bidder enablement above. Every scenario needs it, and one
        # that forgot it would silently render as an unsold impression.
        #
        # Only includewinners is set. includebidderkeys defaults to true, which adds
        # each seat's own hb_pb_BIDDER keys; harmless, and useful for diagnosis.
        prebid["targeting"] = {"includewinners": True}
        prepared.append("ext.prebid.targeting.includewinners (mark the winning bid)")

    if "multibid" not in prebid and _BIDDERS:
        # Prebid keeps ONE bid per bidder per impression unless asked otherwise, and
        # silently discards the rest -- they appear in neither `seatbid` nor
        # `ext.seatnonbid`, so a real losing offer leaves no trace at all. Verified on
        # the deployed stack: artfhouse bid 7.20 on deal-premium-auto and 4.60 on
        # deal-standard-auto for the same impression; without multibid only the 7.20
        # survived the response, and the 4.60 offer was unreportable.
        #
        # A losing offer is the substance of this column, so the offers it shows must
        # include the ones that lost.
        prebid["multibid"] = [{"bidder": name, "maxbids": _MAX_BIDS_PER_BIDDER} for name in _BIDDERS]
        prepared.append(
            f"ext.prebid.multibid (keep up to {_MAX_BIDS_PER_BIDDER} bids per seat, "
            f"so a seat's losing offers are still reported)"
        )

    ext["prebid"] = prebid
    out["ext"] = ext

    return out, prepared


def _with_artf_bypass(body: dict) -> tuple[dict, str]:
    """Mark the request so the ARTF extension point proposes nothing.

    The marker is `ext.artf.bypass: true` on the request's top-level ext. Prebid's
    hook serialises the parsed bid request -- top-level ext included -- into the
    envelope it POSTs to the orchestrator's `/v1/mutations`, which reads the marker
    and returns an empty mutation set WITHOUT fanning out to any container. The
    seats then bid on the request exactly as the publisher sent it.

    This is how the Theater's "without ARTF mutations" pass is a real auction and
    not a derivation: the same Prebid Server, the same seats, the same request --
    minus the one step being demonstrated.

    Copied shallowly, with ext and ext.artf copied too, so the caller's dict is not
    mutated. Anything the caller already put under `ext.artf` is kept.
    """
    out = dict(body)
    ext = dict(out.get("ext") or {})
    artf = dict(ext.get("artf") or {})
    artf[ARTF_BYPASS_KEY] = True
    ext["artf"] = artf
    out["ext"] = ext
    return out, (
        f"ext.artf.{ARTF_BYPASS_KEY} (baseline pass: the ARTF extension point "
        "proposes nothing, so no container is consulted)"
    )


def _with_applicable_intents(body: dict, intents: list[str]) -> tuple[dict, str]:
    """Record the scenario's applicable intents on the request's top-level ext.

    Same location as the bypass marker, for the same reason: top-level `ext.artf`
    survives Prebid, `ext.prebid.artf` does not. See INTENTS_PARAM for why this is a
    statement on the request and not a control: the hook does not read it, and the
    mutations it asks for are not narrowed by it. Anything the caller already put
    under `ext.artf` is kept.
    """
    out = dict(body)
    ext = dict(out.get("ext") or {})
    artf = dict(ext.get("artf") or {})
    artf[INTENTS_KEY] = list(intents)
    ext["artf"] = artf
    out["ext"] = ext
    return out, (
        f"ext.artf.{INTENTS_KEY} (the scenario's intents, stated on the request; "
        "the hook's configured set still decides what is asked for)"
    )


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


def _seat_returned_prices(auction: dict) -> dict[tuple[str, str], float]:
    """What each seat actually returned, per impression, from Prebid's debug output.

    WHY THIS IS NEEDED

    Prebid reports a seat that bid unusably as NonBidReason 0 -- NO_BID -- and
    nothing more. So a seat whose bid was below the impression's floor and a seat
    that genuinely answered with nothing are the same entry in `ext.seatnonbid`, and
    a viewer is told only "no bid" for what may be the most interesting decision on
    the screen: the floor turning demand away.

    The bidder's own response body IS available, in `ext.debug.httpcalls[seat][]`,
    because this endpoint asks for debug. That is where the price it offered lives.

    WHAT THIS IS NOT

    This reads a number the bidder sent. It does not decide, and must not be used to
    claim, WHY Prebid dropped the bid: Prebid said NO_BID, and only Prebid knows
    whether the floor, a validation rule or something else discarded it. The price is
    evidence for the reader, not a verdict from us.

    Keyed by (seat, impid). A price appearing twice for one key keeps the first,
    since a later duplicate is not more authoritative than an earlier one.
    """
    debug = (auction.get("ext") or {}).get("debug") or {}
    httpcalls = debug.get("httpcalls") or {}
    prices: dict[tuple[str, str], float] = {}

    for seat, calls in httpcalls.items():
        for call in calls or []:
            raw = call.get("responsebody")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            for seatbid in parsed.get("seatbid") or []:
                for bid in (seatbid or {}).get("bid") or []:
                    impid = bid.get("impid")
                    price = bid.get("price")
                    # bool is a subclass of int; a JSON `true` here is not a price.
                    if not impid or isinstance(price, bool) or not isinstance(price, (int, float)):
                        continue
                    prices.setdefault((seat, impid), float(price))

    return prices


def _imp_floors(request_body: dict) -> dict[str, float]:
    """Each impression's floor, from the request as submitted.

    The floor is `imp.bidfloor` on the request Prebid was given -- which is the
    request AFTER the ARTF containers mutated it, so a floor a container raised is
    the floor reported here. That is the point: the sell-side decision and the
    demand it turned away are the same story.
    """
    floors: dict[str, float] = {}
    for imp in request_body.get("imp") or []:
        impid = imp.get("id")
        floor = imp.get("bidfloor")
        if not impid or isinstance(floor, bool) or not isinstance(floor, (int, float)):
            continue
        floors[impid] = float(floor)
    return floors


def _annotate_seat_nonbids(auction: dict, request_body: dict) -> int:
    """Attach the observed price and floor to each seat-level non-bid.

    Writes `ext.artf.observed = {returnedPrice, impFloor, currency, source}` onto the
    nonbid entry, leaving every field Prebid set untouched. `ext.artf` is used because
    it is already this repo's namespace on a nonbid, and `observed` is a sibling of
    the campaign identity the demand endpoint puts there -- so a consumer reading one
    is not broken by the other.

    An entry gets no `observed` block when either number is missing, rather than a
    partial one: "returned 2.75 against a floor of null" invites the reader to fill
    in the blank themselves.

    Returns the number of entries annotated, for artf_meta.
    """
    prices = _seat_returned_prices(auction)
    floors = _imp_floors(request_body)
    if not prices or not floors:
        return 0

    currency = auction.get("cur") or None
    annotated = 0

    for seatnonbid in (auction.get("ext") or {}).get("seatnonbid") or []:
        seat = (seatnonbid or {}).get("seat")
        for entry in (seatnonbid or {}).get("nonbid") or []:
            impid = entry.get("impid")
            price = prices.get((seat, impid))
            floor = floors.get(impid)
            if price is None or floor is None:
                continue
            ext = dict(entry.get("ext") or {})
            artf = dict(ext.get("artf") or {})
            artf["observed"] = {
                "returnedPrice": price,
                "impFloor": floor,
                "currency": currency,
                "source": "bidder response, via prebid debug httpcalls",
            }
            ext["artf"] = artf
            entry["ext"] = ext
            annotated += 1

    return annotated
