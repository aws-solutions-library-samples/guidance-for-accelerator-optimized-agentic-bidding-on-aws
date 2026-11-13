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

import copy
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

#: One client for the Prebid hop, created on first use. A client per auction paid a
#: fresh TLS handshake every call (measured ~7 ms of an ~8.5 ms bare round trip);
#: the pooled connection is reused instead. Closed by the app's lifespan.
_hop_client: Optional[httpx.AsyncClient] = None


def _get_hop_client() -> httpx.AsyncClient:
    global _hop_client
    if _hop_client is None:
        # verify=False: see the module docstring -- Prebid presents a certificate it
        # generated for itself at container start. The only target is PREBID_AUCTION_URL,
        # https://prebid-server.<namespace>.svc.cluster.local/openrtb2/auction.
        _hop_client = httpx.AsyncClient(verify=False, timeout=HOP_TIMEOUT_S)  # nosec B501  # self-signed cert on the in-cluster Prebid Server Service
    return _hop_client


async def close_hop_client() -> None:
    """Close the pooled Prebid client. Called on application shutdown."""
    global _hop_client
    client, _hop_client = _hop_client, None
    if client is not None:
        await client.aclose()

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
    # Only the with-ARTF pass asks for the module trace: that is where the hook's
    # analytics tags travel, and the baseline pass has no hook report to lift.
    caller_set_trace = False
    if artf_mode == ARTF_MODE_ON:
        body, caller_set_trace = _with_module_trace(body)
        if not caller_set_trace:
            prepared.append(
                "ext.prebid.trace=verbose (so the ARTF hook's analytics tags reach the "
                "response; removed again before returning)"
            )

    started = time.monotonic()
    try:
        response = await _get_hop_client().post(
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
        # `body` is the prepared request, so the impression floors reported are the
        # ones Prebid enforced on non-deal bids (no deal floor raises them).
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
        if artf_mode == ARTF_MODE_ON:
            auction["artf_meta"]["extension_point"] = _lift_extension_point(auction)
            if not caller_set_trace:
                _strip_module_trace(auction)
    return JSONResponse(auction)


#: The request floors object that turns on Prebid's deal-floor enforcement and sets
#: no impression floor (PBS-Java 3.43.0). A request floors object without `data` is
#: rejected by PriceFloorRulesValidator ("Price floor rules data must be present")
#: and replaced with an empty one, which drops `enforcement`. So one rule is sent
#: whose key is neither a media type nor the `*` wildcard, with no `default`:
#: BasicPriceFloorResolver matches nothing, and every imp.bidfloor stays as sent.
#: The enforcer then compares a deal bid with the floor the artfhouse adapter
#: attaches to it, max(imp.bidfloor, deal bidfloor).
_DEAL_FLOOR_ENFORCEMENT: dict = {
    "enforcement": {"floorDeals": True},
    "data": {
        "currency": "USD",
        "modelGroups": [
            {"schema": {"fields": ["mediaType"]}, "values": {"artf-no-floor-rule": 0}},
        ],
    },
}


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

    It turns on Prebid's deal-floor enforcement (`ext.prebid.floors`, see
    _DEAL_FLOOR_ENFORCEMENT) unless the request carries floors of its own, which are
    then left untouched. No impression floor changes either way.

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

    if prebid.get("floors") is None:
        # Prebid skips floor enforcement on a bid with a dealid unless the request
        # sets ext.prebid.floors.enforcement.floorDeals (the account already sets
        # enforce-deal-floors). See _DEAL_FLOOR_ENFORCEMENT for why a rule block is
        # sent with it and why it changes no impression floor.
        prebid["floors"] = copy.deepcopy(_DEAL_FLOOR_ENFORCEMENT)
        prepared.append(
            "ext.prebid.floors (turn on PBS deal-floor enforcement; one rule that matches "
            "no impression, so impression floors are unchanged)"
        )
    else:
        prepared.append(
            "ext.prebid.floors left as sent (the request carries its own floors, so "
            "deal-floor enforcement follows them)"
        )

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


def _with_module_trace(body: dict) -> tuple[dict, bool]:
    """Ask Prebid for its verbose module trace, merged into the existing ext.prebid.

    PBS-Java copies hook analytics tags into the response only at verbose trace
    level (see ArtfAnalyticsTagWriter's javadoc), and those tags are the only place
    the hook reports which mutations it received and what it did with each.

    Returns (body, caller_set_trace). A trace level the caller chose is kept as is,
    and the flag tells the handler not to strip a trace the caller asked for.
    Copied shallowly, with ext and ext.prebid copied too, like _with_debug_enabled.
    """
    out = dict(body)
    ext = dict(out.get("ext") or {})
    prebid = dict(ext.get("prebid") or {})
    caller_set_trace = "trace" in prebid
    if not caller_set_trace:
        prebid["trace"] = "verbose"
    ext["prebid"] = prebid
    out["ext"] = ext
    return out, caller_set_trace


#: The ARTF hook's module code (ArtfModule.CODE). Its analytics tags are keyed by it
#: in the module trace. Mirrors prebid/verify_auction.py, which the orchestrator
#: image does not ship.
ARTF_HOOK_MODULE_CODE = "artf-orchestrator"
_ACTIVITY_EXTENSION_POINT = "artf-extension-point"
_ACTIVITY_MUTATIONS = "artf-mutations"
_ANSWERED_OUTCOMES = frozenset({"mutations_returned", "no_mutations"})
_EXTENSION_POINT_SOURCE = "prebid hook analytics tags (ext.prebid.modules.trace, verbose)"


def _module_trace(auction: dict) -> Optional[dict]:
    modules = ((auction.get("ext") or {}).get("prebid") or {}).get("modules")
    if not isinstance(modules, dict):
        return None
    trace = modules.get("trace")
    return trace if isinstance(trace, dict) else None


def _artf_hook_activities(auction: dict) -> list[dict]:
    """Every analytics activity the ARTF hook wrote, from the verbose module trace.

    Path: ext.prebid.modules.trace.stages[].outcomes[].groups[].invocationresults[]
    .analyticstags.activities[]. PBS serialises lower-cased keys; camelCase is
    tolerated. Other modules' activities are ignored.
    """
    trace = _module_trace(auction) or {}
    found: list[dict] = []
    for stage in trace.get("stages") or []:
        for outcome in (stage or {}).get("outcomes") or []:
            for group in (outcome or {}).get("groups") or []:
                group = group or {}
                for inv in group.get("invocationresults") or group.get("invocationResults") or []:
                    inv = inv or {}
                    hook_id = inv.get("hookid") or inv.get("hookId") or {}
                    module = hook_id.get("module-code") or hook_id.get("moduleCode") or ""
                    if module and module != ARTF_HOOK_MODULE_CODE:
                        continue
                    tags = inv.get("analyticstags") or inv.get("analyticsTags") or {}
                    for activity in tags.get("activities") or []:
                        if isinstance(activity, dict):
                            found.append(activity)
    return found


def _activity_values(activities: list[dict], name: str) -> Optional[dict]:
    for activity in activities:
        if activity.get("name") != name:
            continue
        for result in activity.get("results") or []:
            values = (result or {}).get("values")
            if isinstance(values, dict):
                return values
    return None


def _mutation_key(mutation: Any) -> str:
    """Canonical identity of one mutation: its intent, op, path and payload."""
    if not isinstance(mutation, dict):
        return json.dumps(mutation, sort_keys=True, default=str)
    return json.dumps(
        {
            k: mutation.get(k)
            for k in ("intent", "op", "path", "ids", "adjust_deal", "adjust_bid", "add_metrics")
        },
        sort_keys=True,
        default=str,
    )


def _stamp_dispositions(response: dict, dispositions: Optional[list]) -> dict:
    """Return a copy of the hook's RTBResponse with a disposition on every mutation.

    Survivors (`response.mutations[i]`) take the hook's `dispositions[i]`: applied,
    or rejected with the hook's reason. When the hook reported no list of the same
    length, the survivor is `unknown`.

    Per-container mutations (`metadata.containers[].mutations`) are the orchestrator's
    raw proposals. The survivors are an in-order subset of them (resolve_conflicts),
    so they are matched by walking the containers in order. A container mutation
    that lost a (path, intent) contest is `superseded`, named after the winner in
    `metadata.conflicts`. Derived from this response alone: no stored state.
    """
    out = json.loads(json.dumps(response))
    raw_survivors = out.get("mutations") or []
    usable = isinstance(dispositions, list) and len(dispositions) == len(raw_survivors)
    survivors = []
    for index, mutation in enumerate(raw_survivors):
        if not isinstance(mutation, dict):
            continue
        entry = dispositions[index] if usable else None
        if not isinstance(entry, dict) or not isinstance(entry.get("applied"), bool):
            mutation["disposition"] = "unknown"
            mutation["disposition_reason"] = None
        elif entry["applied"]:
            mutation["disposition"] = "applied"
            mutation["disposition_reason"] = None
        else:
            mutation["disposition"] = "rejected"
            mutation["disposition_reason"] = entry.get("reason") or "rejected by the exchange"
        survivors.append(mutation)

    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    contests: dict[tuple[Any, Any], dict] = {}
    for conflict in metadata.get("conflicts") or []:
        if isinstance(conflict, dict):
            contests[(conflict.get("path"), conflict.get("intent"))] = conflict

    pointer = 0
    for container in metadata.get("containers") or []:
        if not isinstance(container, dict):
            continue
        name = container.get("name")
        for mutation in container.get("mutations") or []:
            if not isinstance(mutation, dict):
                continue
            contest = contests.get((mutation.get("path"), mutation.get("intent")))
            lost = (
                contest is not None
                and name in (contest.get("losers") or [])
                and name != contest.get("winner")
            )
            if (
                not lost
                and pointer < len(survivors)
                and _mutation_key(mutation) == _mutation_key(survivors[pointer])
            ):
                mutation["disposition"] = survivors[pointer]["disposition"]
                mutation["disposition_reason"] = survivors[pointer]["disposition_reason"]
                pointer += 1
                continue
            mutation["disposition"] = "superseded"
            winner = (contest or {}).get("winner")
            mutation["disposition_reason"] = (
                f"superseded by {winner}" if winner and winner != name else "superseded"
            )
    return out


def _lift_extension_point(auction: dict) -> dict:
    """The ARTF hook's own report of this auction, as artf_meta.extension_point.

    `response` is the RTBResponse the hook received, with a disposition on every
    mutation, so a consumer narrates what the auction actually ran rather than a
    separate call. `available` is False, with a `reason`, when the response carries
    no usable report. Pure: reads `auction`, writes nothing.
    """
    if _module_trace(auction) is None:
        return {
            "available": False,
            "source": _EXTENSION_POINT_SOURCE,
            "reason": "Prebid returned no module trace, so the ARTF hook's report is not in the response",
        }
    activities = _artf_hook_activities(auction)
    call = _activity_values(activities, _ACTIVITY_EXTENSION_POINT)
    if call is None:
        return {
            "available": False,
            "source": _EXTENSION_POINT_SOURCE,
            "reason": "the module trace carries no ARTF hook activity",
        }
    outcome = call.get("outcome")
    applied_values = _activity_values(activities, _ACTIVITY_MUTATIONS) or {}
    raw_response = call.get("response")
    if outcome in _ANSWERED_OUTCOMES and not isinstance(raw_response, dict):
        return {
            "available": False,
            "source": _EXTENSION_POINT_SOURCE,
            "reason": (
                "this Prebid image's ARTF hook does not report its mutations; "
                "redeploy Prebid"
            ),
            "correlation_id": call.get("correlation_id"),
            "outcome": outcome,
        }
    response = (
        _stamp_dispositions(raw_response, applied_values.get("dispositions"))
        if isinstance(raw_response, dict)
        else None
    )
    return {
        "available": True,
        "source": _EXTENSION_POINT_SOURCE,
        "correlation_id": call.get("correlation_id"),
        "outcome": outcome,
        "latency_ms": call.get("latency_ms"),
        "request_mutated": bool(call.get("request_mutated")),
        "applied": applied_values.get("applied", 0),
        "rejected": applied_values.get("rejected", 0),
        "response": response,
    }


def _strip_module_trace(auction: dict) -> None:
    """Remove the verbose module trace this endpoint asked for, after lifting it.

    ext.debug (httpcalls, which the offers view reads) is not touched.
    """
    prebid = (auction.get("ext") or {}).get("prebid")
    if not isinstance(prebid, dict):
        return
    modules = prebid.get("modules")
    if not isinstance(modules, dict):
        return
    modules.pop("trace", None)
    if not modules:
        prebid.pop("modules", None)


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

    The floor is `imp.bidfloor` on the request Prebid was given. Deal floors do not
    raise it: an ADJUST_DEAL_FLOOR or ADJUST_DEAL_MARGIN mutation writes the deal's
    own floor, and a deal bid is held to max(impression floor, deal floor) by the
    demand endpoint and by Prebid's deal-floor enforcement. So this is the floor a
    non-deal bid faced.
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
