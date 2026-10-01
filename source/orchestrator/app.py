"""ARTF Orchestrator — fans out RTBRequests to intent containers.

Calls each registered ARTF container via **gRPC** (the primary ARTF
protocol) with a JSON-RPC/MCP fallback.  Merges returned mutations into
a single RTBResponse.

Also exposes a REST endpoint for the frontend (``POST /v1/mutations``)
and a container health dashboard (``GET /v1/containers``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import grpc
import httpx
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.artf_types import (  # noqa: E402
    ConflictModel,
    ContainerInvocationModel,
    Metadata,
    Mutation,
    RTBRequest,
    RTBResponse,
)
from orchestrator.container_registry import (  # noqa: E402
    ERROR_STATUSES,
    RAN_STATUSES,
    SOURCE_CODE,
    STATUS_DISABLED,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
    ContainerCallOutcome,
    ContainerRegistryStore,
    RegistryEntry,
    RegistryRecordNotFound,
    RegistryStoreUnavailable,
    build_claims,
    derive_status,
    merge_registry,
    resolve_conflicts,
    select_active,
    shared_intents,
)
from shared.feedback_collector import FeedbackCollector  # noqa: E402
from shared.signal_associator import SignalAssociator  # noqa: E402
from orchestrator.signal_receiver import receive_signal as _receive_signal_handler  # noqa: E402
from orchestrator.feedback_integration import emit_bid_outcome  # noqa: E402
from orchestrator.deal_yield_feedback import emit_deal_yield_outcome  # noqa: E402

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Container registry
# ---------------------------------------------------------------------------

_GRPC_METHOD = "/com.iabtechlab.bidstream.mutation.services.v1.RTBExtensionPoint/GetMutations"

# ``display_name`` and ``description`` are served by GET /v1/containers so a
# client does not have to carry a build-time label table. They matter because
# store-defined containers (see orchestrator/container_registry.py) have names
# that are not known when the frontend is built, so the API has to be the source
# of labels for those; having the six built-ins answer the same way keeps one
# code path instead of two. The display names match RENAME_MAP.md.
CONTAINERS = [
    {
        "name": "dlrm-bid-shader",
        "display_name": "Bid Pricer",
        "description": "Prices the bid with the DLRM model on Triton.",
        "intents": {"BID_SHADE"},
        "grpc": os.environ.get("DLRM_GRPC", os.environ.get("DLRM_URL", "http://localhost:50061")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("DLRM_MCP", os.environ.get("DLRM_URL", "http://localhost:8091")),
    },
    {
        "name": "widedeep-segment-activator",
        "display_name": "Audience Activator",
        "description": "Activates IAB audience segments from bid-request signals (rule-based).",
        "intents": {"ACTIVATE_SEGMENTS"},
        "grpc": os.environ.get("WIDEDEEP_GRPC", os.environ.get("WIDEDEEP_URL", "http://localhost:50062")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("WIDEDEEP_MCP", os.environ.get("WIDEDEEP_URL", "http://localhost:8092")),
    },
    {
        "name": "ncf-deal-manager",
        "display_name": "Deal Scorer",
        "description": "Activates and suppresses deals with the NCF model on Triton.",
        "intents": {"ACTIVATE_DEALS", "SUPPRESS_DEALS"},
        "grpc": os.environ.get("NCF_GRPC", os.environ.get("NCF_URL", "http://localhost:50063")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("NCF_MCP", os.environ.get("NCF_URL", "http://localhost:8093")),
    },
    {
        "name": "metrics-enricher",
        "display_name": "Signals Enricher",
        "description": "Adds viewability and brand-safety metrics (rule-based).",
        "intents": {"ADD_METRICS"},
        "grpc": os.environ.get("METRICS_GRPC", os.environ.get("METRICS_URL", "http://localhost:50064")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("METRICS_MCP", os.environ.get("METRICS_URL", "http://localhost:8094")),
    },
    # The Yield Optimizer is two containers, one per intent. Each is called
    # only for its own intent, so a request asking for just ADJUST_DEAL_FLOOR
    # never fans out to the margin model (BR-5: the two intents are
    # independent and atomic). YIELD_FLOOR_URL/YIELD_MARGIN_URL remain
    # supported as single-value fallbacks for parity with the other four.
    {
        "name": "yield-optimizer-floor",
        "display_name": "Yield Optimizer — Floor",
        "description": "Sets the deal bid floor with an XGBoost/FIL model on Triton.",
        "intents": {"ADJUST_DEAL_FLOOR"},
        "grpc": os.environ.get("YIELD_FLOOR_GRPC", os.environ.get("YIELD_FLOOR_URL", "http://localhost:50065")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("YIELD_FLOOR_MCP", os.environ.get("YIELD_FLOOR_URL", "http://localhost:8095")),
    },
    {
        "name": "yield-optimizer-margin",
        "display_name": "Yield Optimizer — Margin",
        "description": "Sets the deal margin with an XGBoost/FIL model on Triton.",
        "intents": {"ADJUST_DEAL_MARGIN"},
        "grpc": os.environ.get("YIELD_MARGIN_GRPC", os.environ.get("YIELD_MARGIN_URL", "http://localhost:50066")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("YIELD_MARGIN_MCP", os.environ.get("YIELD_MARGIN_URL", "http://localhost:8096")),
    },
]


# ---------------------------------------------------------------------------
# Effective registry — the six above, plus whatever the store defines
# ---------------------------------------------------------------------------
#
# The store half lets a container be described and switched on or off without
# rebuilding this image. Lazily constructed so importing this module never needs
# AWS credentials (same idiom as closed_loop_api._get_parameter_store).

_REGISTRY_STORE: ContainerRegistryStore | None = None


def _registry_store() -> ContainerRegistryStore:
    global _REGISTRY_STORE
    if _REGISTRY_STORE is None:
        _REGISTRY_STORE = ContainerRegistryStore()
    return _REGISTRY_STORE


def _effective_registry() -> tuple[list[RegistryEntry], list[str]]:
    """The merged registry: code-defined containers plus store-defined ones.

    Reads through the store's TTL cache, so this costs at most one DynamoDB
    Query per TTL window per replica however high the request rate — the bid
    path never pays a per-request round trip for it. A store failure yields the
    last known good records, and a store that has never been read yields none,
    which merges to exactly the six code-defined containers.
    """
    return merge_registry(CONTAINERS, _registry_store().get_records())


def _priorities() -> dict[str, int]:
    """Container name to precedence, from the same TTL-cached registry.

    Separate from ``_fan_out`` on purpose. ``_fan_out`` already returns
    invocations in registry order, so the ordering half of the tie-break is
    derivable from the list it returns and only the priority has to be looked up.
    Widening ``_fan_out``'s return type would have forced churn in the tests that
    pin its behaviour, for information already in hand.

    Adds no I/O: the registry is cached, so this is a dict comprehension over
    values already in memory (NFR-1).
    """
    entries, _ = _effective_registry()
    return {e.name: e.priority for e in entries}


def _resolve_for_response(
    invocations: list[ContainerInvocationModel],
) -> tuple[list[Mutation], list[ConflictModel]]:
    """Resolve competing mutations and annotate the invocations in place.

    Two containers may claim the same intent, and both get called. But a consumer
    applies the list in order and, for a deal floor, the applier overwrites — so
    only the last mutation for a given path had any effect. Returning both left
    the earlier one looking applied when it was not.

    Per-container ``mutations`` is deliberately left intact: the container really
    did compute it, and the Auction Theater and the pipeline read their stops from
    ``metadata.containers`` rather than from the flattened list, so the displaced
    mutation stays visible there. Only the flattened list is filtered, and
    ``superseded`` records how many of each container's mutations lost.
    """
    claims = build_claims(invocations, _priorities())
    survivors, conflicts, superseded = resolve_conflicts(claims)

    for inv in invocations:
        inv.superseded = superseded.get(inv.name, 0)

    conflict_models = [
        ConflictModel(
            path=c.path, intent=c.intent, winner=c.winner, losers=list(c.losers)
        )
        for c in conflicts
    ]
    return survivors, conflict_models


def _filter_containers(applicable_intents: list[str] | None) -> list[dict]:
    """Active containers whose intents overlap ``applicable_intents``, as dicts.

    Kept in its original shape and name because ``loadtest.py`` imports it and
    passes its results straight to ``_call_container_timed``. It now consults the
    merged registry, so an inactive container is excluded here too and a load
    test never drives traffic at something the operator switched off.

    Empty or absent ``applicable_intents`` still means "all intents apply".
    Callers needing to know *why* a container was not called should use
    ``select_active`` directly — this function only returns the ones to call.
    """
    entries, _ = _effective_registry()
    to_call, _ = select_active(entries, applicable_intents)
    return [e.as_container_dict() for e in to_call]





# ---------------------------------------------------------------------------
# gRPC caller (primary ARTF protocol)
# ---------------------------------------------------------------------------

async def _call_grpc(target: str, payload_bytes: bytes, timeout_s: float) -> list[Mutation]:
    """Call a container's RTBExtensionPoint.GetMutations via gRPC."""
    try:
        channel = grpc.aio.insecure_channel(target)
        response_bytes = await channel.unary_unary(
            _GRPC_METHOD,
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )(payload_bytes, timeout=timeout_s)
        await channel.close()
        data = json.loads(response_bytes)
        return [Mutation(**m) for m in data.get("mutations", [])]
    except Exception as exc:
        print(f"[orchestrator] gRPC to {target} failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# MCP/JSON-RPC caller (fallback)
# ---------------------------------------------------------------------------

async def _call_mcp(
    client: httpx.AsyncClient, base_url: str, payload: dict, timeout_s: float
) -> ContainerCallOutcome:
    """Call a container's extend_rtb tool via MCP JSON-RPC.

    Returns a ``ContainerCallOutcome`` rather than a bare mutation list. It used
    to swallow every failure and return ``[]``, which meant the caller could not
    tell "nothing answered" from "answered with nothing" — and so labelled a
    completely absent container as ``ok``.
    """
    try:
        rpc_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "extend_rtb", "arguments": payload},
        }
        resp = await client.post(f"{base_url}/mcp", json=rpc_body, timeout=timeout_s)
    except Exception as exc:
        # Nothing answered on the wire.
        print(f"[orchestrator] MCP to {base_url} failed: {exc}")
        return ContainerCallOutcome(reached=False, error=type(exc).__name__)

    if resp.status_code != 200:
        print(f"[orchestrator] MCP to {base_url} returned {resp.status_code}: {resp.text[:200]}")
        return ContainerCallOutcome(reached=True, error=f"MCP HTTP {resp.status_code}")

    try:
        data = resp.json()
        result = data.get("result", {})
        # MCP returns a content array whose text holds the RTBResponse JSON.
        for content_item in result.get("content", []):
            if content_item.get("type") == "text":
                rtb_resp = json.loads(content_item["text"])
                mutations = [Mutation(**m) for m in rtb_resp.get("mutations", [])]
                metadata = rtb_resp.get("metadata") or {}
                model_version = metadata.get("model_version", "") or ""
                return ContainerCallOutcome(
                    reached=True,
                    mutations=mutations,
                    model_version=model_version,
                    # Carried through rather than dropped here: an empty mutation
                    # list with a reason attached is a different fact from an empty
                    # one without, and only the container knows which it is.
                    abstained_reason=metadata.get("abstained_reason") or None,
                )
        # Some implementations put mutations directly on the result.
        if "mutations" in result:
            mutations = [Mutation(**m) for m in result.get("mutations", [])]
            return ContainerCallOutcome(reached=True, mutations=mutations)
        # A 200 that carries neither shape is a protocol error, not an empty
        # answer — saying "no mutations" here would misreport a broken container
        # as a well-behaved one.
        if data.get("error"):
            return ContainerCallOutcome(reached=True, error=f"MCP error: {str(data['error'])[:120]}")
        return ContainerCallOutcome(reached=True, error="MCP response carried no mutations field")
    except Exception as exc:
        print(f"[orchestrator] MCP response from {base_url} unparseable: {exc}")
        return ContainerCallOutcome(reached=True, error=f"unparseable MCP response ({type(exc).__name__})")


async def _call_http(client: httpx.AsyncClient, base_url: str, payload: dict, timeout_s: float) -> list[Mutation]:
    """Call a container's /mutate REST endpoint (simplest fallback)."""
    try:
        # Try the health endpoint first to construct the mutate URL
        mutate_url = base_url.rstrip("/")
        # The containers serve /mutate on the same port as health when using FastAPI
        # But with the multi-protocol server, there's no /mutate — only /mcp
        # So this is a no-op fallback
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Dispatch to a single container (gRPC first, MCP fallback)
# ---------------------------------------------------------------------------

async def _call_container(
    client: httpx.AsyncClient,
    container: dict,
    payload: dict,
    payload_bytes: bytes,
    timeout_s: float,
    *,
    headers: dict[str, str] | None = None,
) -> ContainerCallOutcome:
    """Call a container's /mutate REST endpoint, falling back to MCP JSON-RPC.

    ``headers``, when provided, are forwarded on the REST call only — used
    exclusively by the orchestrator's load-test invocation path to set the
    out-of-band X-Load-Test-Target-Variant header (see
    orchestrator/loadtest_targeting.py). Never set by this function's other
    callers (get_mutations / real bid-serving).

    Returns a ``ContainerCallOutcome`` carrying whether anything answered, what
    it returned, and any transport or protocol error. ``model_version`` is the
    container's per-request-resolved served model version (from
    RTBResponse.metadata), or "" if it didn't return one — never fabricated.

    **A parsed 200 is definitive**, mutations or not. This function used to fall
    through to the MCP attempt whenever the REST call returned zero mutations,
    so a healthy container that legitimately produced nothing was called twice on
    every request. It also made an empty result ambiguous, which is what allowed
    an unreachable container to be reported as ``ok``.
    """
    rest_error: str | None = None
    rest_reached = False

    try:
        resp = await client.post(
            f"{container['mcp']}/mutate", json=payload, timeout=timeout_s, headers=headers
        )
        rest_reached = True
        if resp.status_code == 200:
            try:
                data = resp.json()
                mutations = [Mutation(**m) for m in data.get("mutations", [])]
                metadata = data.get("metadata") or {}
                model_version = metadata.get("model_version", "") or ""
                return ContainerCallOutcome(
                    reached=True,
                    mutations=mutations,
                    model_version=model_version,
                    abstained_reason=metadata.get("abstained_reason") or None,
                )
            except Exception as exc:
                rest_error = f"unparseable /mutate response ({type(exc).__name__})"
                print(f"[orchestrator] /mutate response from {container['mcp']} unparseable: {exc}")
        else:
            rest_error = f"/mutate HTTP {resp.status_code}"
    except Exception as exc:
        rest_error = type(exc).__name__
        print(f"[orchestrator] REST /mutate to {container['mcp']} failed: {exc}")

    # Fallback to MCP JSON-RPC (no header support on this path today — the
    # load-test-only override only needs to work on the REST path, which is
    # the one load test's container calls exercise).
    mcp_outcome = await _call_mcp(client, container["mcp"], payload, timeout_s)
    if mcp_outcome.reached and mcp_outcome.error is None:
        return mcp_outcome

    # Neither transport produced a usable answer. Preserve the distinction
    # between "something answered badly" and "nothing answered at all", and
    # report the REST error in preference to the MCP one since /mutate is the
    # primary path.
    return ContainerCallOutcome(
        reached=rest_reached or mcp_outcome.reached,
        error=rest_error or mcp_outcome.error or "no response",
    )


async def _call_container_timed(
    client: httpx.AsyncClient,
    container: dict,
    payload: dict,
    payload_bytes: bytes,
    timeout_s: float,
    *,
    headers: dict[str, str] | None = None,
) -> ContainerInvocationModel:
    """Wrap ``_call_container`` with a wall-clock timer and outcome status.

    Records ``latency_ms`` in every branch using ``time.monotonic()``.
    Returns a ``ContainerInvocationModel`` whose ``status`` comes from
    ``derive_status`` over the real call outcome:

    - ``ok`` — reached, returned mutations
    - ``no_mutations`` — reached, deliberately returned none
    - ``unreachable`` — nothing answered on the wire
    - ``error`` — answered, but the response was unusable
    - ``timeout`` — exceeded the request's tmax budget
    - ``failed`` — retained for an unexpected exception in this wrapper itself

    Before this, every one of the first four collapsed to ``ok``, so a container
    that was scaled to zero or had no Service endpoints reported itself healthy
    on every request.

    ``container`` is a plain dict (``name``/``intents``/``grpc``/``mcp``, and
    optionally ``display_name``) — the shape loadtest.py and its tests pass.
    ``headers`` is forwarded to ``_call_container`` unchanged.
    """
    start = time.monotonic()
    display_name = container.get("display_name") or ""
    try:
        outcome = await asyncio.wait_for(
            _call_container(client, container, payload, payload_bytes, timeout_s, headers=headers),
            timeout=timeout_s,
        )
        latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        return ContainerInvocationModel(
            name=container["name"],
            status=derive_status(outcome),
            latency_ms=latency_ms,
            mutations=outcome.mutations,
            model_version=outcome.model_version,
            display_name=display_name,
            abstained_reason=outcome.abstained_reason,
        )
    except asyncio.TimeoutError:
        latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        return ContainerInvocationModel(
            name=container["name"],
            status=STATUS_TIMEOUT,
            latency_ms=latency_ms,
            mutations=[],
            display_name=display_name,
        )
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        print(f"[orchestrator] container {container['name']} failed: {exc}")
        return ContainerInvocationModel(
            name=container["name"],
            status="failed",
            latency_ms=latency_ms,
            mutations=[],
            display_name=display_name,
        )


# ---------------------------------------------------------------------------
# Fan-out — one implementation, used by both the REST and MCP entry points
# ---------------------------------------------------------------------------

async def _fan_out(
    payload: dict,
    payload_bytes: bytes,
    applicable_intents: list | None,
    *,
    timeout_s: float,
    headers: dict[str, str] | None = None,
) -> list[ContainerInvocationModel]:
    """Call every active, intent-matching container and return one entry each.

    Extracted because POST /v1/mutations and the MCP ``tools/call`` proxy each
    carried their own copy of the filter + gather + backfill. Two copies is how
    they drifted, and gating on activation state in only one of them would mean a
    container the operator switched off still ran on the other path. One
    implementation makes that impossible rather than merely unlikely.

    Every registry entry appears in the result, in registry order, whatever
    happened — containers that were not called get ``disabled`` or ``skipped``
    with ``latency_ms=0``.
    """
    entries, _warnings = _effective_registry()
    to_call, not_called = select_active(entries, applicable_intents)

    async with httpx.AsyncClient() as client:
        tasks = [
            _call_container_timed(
                client, e.as_container_dict(), payload, payload_bytes, timeout_s, headers=headers
            )
            for e in to_call
        ]
        invocations: list[ContainerInvocationModel] = await asyncio.gather(*tasks)

    by_name = {inv.name: inv for inv in invocations}
    for entry, reason in not_called:
        by_name[entry.name] = ContainerInvocationModel(
            name=entry.name,
            status=reason,
            latency_ms=0,
            mutations=[],
            display_name=entry.display_name,
        )

    # Registry order is also the mutation attribution order, so it is preserved
    # rather than following completion order.
    return [by_name[e.name] for e in entries if e.name in by_name]


# ---------------------------------------------------------------------------
# Starlette routes
# ---------------------------------------------------------------------------

async def get_mutations(request: Request) -> JSONResponse:
    body = await request.json()
    req = RTBRequest(**body)
    tmax = max(req.tmax, 10)
    timeout_s = tmax / 1000.0
    payload = req.model_dump()
    payload_bytes = json.dumps(payload).encode()

    # Only call containers that are active AND whose intents match
    # applicable_intents. An inactive container is reported "disabled" and never
    # invoked; one whose intents don't match is "skipped". Different facts, so
    # different labels.
    applicable = getattr(req, "applicable_intents", None) or body.get("applicable_intents")

    start = time.monotonic()
    all_invocations = await _fan_out(
        payload, payload_bytes, applicable, timeout_s=timeout_s
    )

    # Preserve canonical registry order for both the flattened mutations list
    # and the per-container attribution surfaced via metadata. Competing claims
    # on the same (path, intent) are resolved here rather than left for the
    # consumer to overwrite, so both the Prebid hook and the frontend receive one
    # already-decided set.
    all_mutations, conflicts = _resolve_for_response(all_invocations)

    elapsed_ms = (time.monotonic() - start) * 1000

    # Detect network path — RTB Fabric adds identifying headers when traffic
    # flows through its managed infrastructure.
    fabric_link_id = request.headers.get("x-rtb-fabric-link-id")
    network_path = "rtb-fabric" if fabric_link_id else "direct"

    metadata_dict = {
        "api_version": "1.0",
        "model_version": f"orchestrator-v1 ({len(all_mutations)} mutations, {elapsed_ms:.1f}ms)",
        "containers": all_invocations,
        "network_path": network_path,
        "total_latency_ms": round(elapsed_ms, 2),
    }
    if fabric_link_id:
        metadata_dict["rtb_fabric_link_id"] = fabric_link_id

    resp = RTBResponse(
        id=req.id,
        mutations=all_mutations,
        metadata=Metadata(
            api_version="1.0",
            model_version=f"orchestrator-v1 ({len(all_mutations)} mutations, {elapsed_ms:.1f}ms)",
            containers=all_invocations,
            # None rather than [] so a reader can tell "nothing was contested"
            # from "this orchestrator does not report contests".
            conflicts=conflicts or None,
        ),
    )
    # Merge network_path into the serialized response metadata
    resp_dict = resp.model_dump()
    resp_dict.setdefault("metadata", {}).update({
        "network_path": network_path,
        "total_latency_ms": round(elapsed_ms, 2),
    })
    if fabric_link_id:
        resp_dict["metadata"]["rtb_fabric_link_id"] = fabric_link_id

    # Emit bid outcome event (fire-and-forget, non-blocking)
    emit_bid_outcome(req, resp, start)
    # Emit deal yield outcome event(s) for any adjust_deal mutations
    # (fire-and-forget, non-blocking, independent of emit_bid_outcome above)
    emit_deal_yield_outcome(req, resp)

    return JSONResponse(resp_dict)


async def list_containers(request: Request) -> JSONResponse:
    """Container health endpoint.

    Each entry includes evidence proving the probe ran (timestamp, latency,
    resolved DNS address, HTTP status) so consumers can verify the result
    rather than trust an opaque "ready" label.
    """

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _resolve(hostname_with_port: str) -> str | None:
        """Best-effort DNS resolution. Returns the resolved IP, or None on failure."""
        if not hostname_with_port:
            return None
        host = hostname_with_port.split("://", 1)[-1]  # strip scheme if present
        host = host.split("/", 1)[0]                   # strip path
        host = host.rsplit(":", 1)[0] if ":" in host else host  # strip port
        try:
            return socket.gethostbyname(host)
        except OSError:
            return None

    async def _probe_http(client: httpx.AsyncClient, url: str) -> dict:  # nosemgrep: useless-inner-function
        """Probe an HTTP endpoint and return evidence about the call."""
        evidence: dict = {
            "url": url,
            "checkedAt": _now_iso(),
            "resolvedAddress": _resolve(url),
            "latencyMs": None,
            "httpStatus": None,
            "ok": False,
            "error": None,
        }
        start = time.monotonic()
        try:
            resp = await client.get(url)
            evidence["latencyMs"] = round((time.monotonic() - start) * 1000, 1)
            evidence["httpStatus"] = resp.status_code
            evidence["ok"] = resp.status_code == 200
        except Exception as exc:
            evidence["latencyMs"] = round((time.monotonic() - start) * 1000, 1)
            evidence["error"] = type(exc).__name__
        return evidence

    async def _probe_grpc(target: str) -> dict:  # nosemgrep: useless-inner-function
        """Probe gRPC channel readiness and return evidence about the call."""
        evidence: dict = {
            "target": target,
            "checkedAt": _now_iso(),
            "resolvedAddress": _resolve(target),
            "latencyMs": None,
            "ok": False,
            "error": None,
        }
        start = time.monotonic()
        try:
            channel = grpc.aio.insecure_channel(target)
            await asyncio.wait_for(channel.channel_ready(), timeout=1.0)
            await channel.close()
            evidence["latencyMs"] = round((time.monotonic() - start) * 1000, 1)
            evidence["ok"] = True
        except Exception as exc:
            evidence["latencyMs"] = round((time.monotonic() - start) * 1000, 1)
            evidence["error"] = type(exc).__name__
        return evidence

    results = []

    # Check Triton health first (shared dependency for GPU-backed containers)
    triton_url = os.environ.get("TRITON_URL", "triton-inference-server:8000")
    triton_health_url = f"http://{triton_url}/v2/health/ready"

    async with httpx.AsyncClient(timeout=2.0) as tc:
        triton_evidence = await _probe_http(tc, triton_health_url)
    triton_ready = triton_evidence["ok"]

    # Check individual Triton model readiness. widedeep_segment_activator is
    # intentionally absent — it is no longer served by Triton (segment
    # activation is rule-based; see
    # source/containers/widedeep_segment_activator/app.py). Probing for it
    # here would always report UNAVAILABLE and misrepresent a healthy
    # rules-based container as degraded.
    #
    # The yield models are TWO independent single-output FIL models, not one —
    # Triton's FIL backend does not support multi-output regression (confirmed
    # against NVIDIA's FIL backend docs; see
    # source/triton/model_repository/deal_yield_manager_floor/config.pbtxt).
    # A model literally named "deal_yield_manager" was never registered with
    # Triton, so probing for one always 400s. Each yield container now owns
    # exactly one of these models, so every container below maps to at most a
    # single model name — the tuple special-case this code used to carry is
    # gone. The MODEL names keep their historical deal_yield_manager_* spelling
    # because they are real registered Triton models and real SageMaker Model
    # Package Groups; only the CONTAINERS were renamed.
    triton_models: dict[str, dict] = {}
    if triton_ready:
        model_names = [
            "dlrm_bid_shader",
            "ncf_deal_manager",
            "deal_yield_manager_floor",
            "deal_yield_manager_margin",
        ]
        async with httpx.AsyncClient(timeout=2.0) as tc:
            for model_name in model_names:
                ev = await _probe_http(tc, f"http://{triton_url}/v2/models/{model_name}/ready")
                triton_models[model_name] = {
                    "state": "READY" if ev["ok"] else "UNAVAILABLE",
                    "evidence": ev,
                }

    # Map each container to its single Triton model name, or None for the
    # rules-based containers (widedeep-segment-activator and metrics-enricher
    # have no Triton model at all).
    container_to_model = {
        "dlrm-bid-shader": "dlrm_bid_shader",
        "widedeep-segment-activator": None,  # rules-based, no Triton model
        "ncf-deal-manager": "ncf_deal_manager",
        "metrics-enricher": None,  # rules-based, no Triton model
        "yield-optimizer-floor": "deal_yield_manager_floor",
        "yield-optimizer-margin": "deal_yield_manager_margin",
    }

    registry_entries, registry_warnings = _effective_registry()
    intent_clashes = shared_intents(registry_entries)

    async with httpx.AsyncClient(timeout=2.0) as client:
        for c in registry_entries:
            # An inactive container is not probed. Probing it would cost the
            # panel up to two seconds per container (this loop is sequential) to
            # learn something the operator already decided, and reporting the
            # result would claim a health verdict about a container that is
            # deliberately out of the flow. "not_probed" is the truthful value
            # for a check that did not run.
            if not c.active:
                results.append({
                    "name": c.name,
                    "displayName": c.display_name,
                    "description": c.description,
                    "intents": sorted(c.intents),
                    "active": False,
                    "configurable": c.configurable,
                    "priority": c.priority,
                    "source": c.source,
                    "grpc": c.grpc,
                    "mcp": c.endpoint,
                    "urlScope": "cluster-dns",
                    "status": STATUS_DISABLED,
                    "containerStatus": "not_probed",
                    "inferenceStatus": "not_probed",
                    "protocol": "none",
                    "tritonModel": container_to_model.get(c.name),
                    "evidence": {
                        "httpProbe": None,
                        "grpcProbe": None,
                        "tritonModelProbes": [],
                        "note": "Not probed: the container is inactive, so it is not called.",
                    },
                })
                continue

            container_status = "unknown"
            protocol = "none"
            inference_status = "unknown"

            # Check container health via HTTP (MCP endpoint)
            health_url = c.endpoint.rstrip("/") + "/health/ready"
            http_probe = await _probe_http(client, health_url)
            if http_probe["ok"]:
                container_status = "ready"
                protocol = "mcp"

            # Check gRPC health if HTTP failed
            grpc_probe = None
            if container_status != "ready":
                grpc_probe = await _probe_grpc(c.grpc)
                if grpc_probe["ok"]:
                    container_status = "ready"
                    protocol = "grpc"
                else:
                    container_status = "unreachable"

            # Determine inference readiness (depends on Triton for GPU containers).
            # A store-defined container has no entry in container_to_model, so
            # .get() returns None and it takes the rules-based branch — correct,
            # since a user's own container has no Triton model unless they add
            # one, and inventing a model probe for it would report on something
            # that does not exist.
            model_name = container_to_model.get(c.name)
            if model_name is None:
                # Rules-based container — no Triton dependency
                inference_status = "ready" if container_status == "ready" else "unavailable"
            elif not triton_ready:
                inference_status = "gpu_offline"
            elif triton_models.get(model_name, {}).get("state") == "READY":
                inference_status = "ready"
            else:
                inference_status = "model_unavailable"

            # Overall status: container must be reachable AND inference must work
            if container_status == "ready" and inference_status == "ready":
                overall = "ready"
            elif container_status == "ready" and inference_status == "gpu_offline":
                overall = "degraded"
            elif container_status == "unreachable":
                overall = "unreachable"
            else:
                overall = "degraded"

            entry = {
                "name": c.name,
                "displayName": c.display_name,
                "description": c.description,
                "intents": sorted(c.intents),
                "active": True,
                "configurable": c.configurable,
                # Precedence when two containers claim the same intent. Higher
                # wins; equal falls back to registry order, which is the
                # behaviour that predates precedence existing.
                "priority": c.priority,
                "source": c.source,
                "grpc": c.grpc,
                "mcp": c.endpoint,
                "urlScope": "cluster-dns",  # NOT reachable from a browser; resolves only inside the EKS cluster
                "status": overall,
                "containerStatus": container_status,
                "inferenceStatus": inference_status,
                "protocol": protocol,
                "tritonModel": model_name,
                "evidence": {
                    "httpProbe": http_probe,
                    "grpcProbe": grpc_probe,
                    # Kept as a list (0 entries for a rules-based container, 1
                    # otherwise) so the response shape the UI reads is stable.
                    "tritonModelProbes": [
                        {"model": model_name, **triton_models.get(model_name, {}).get("evidence", {})}
                    ] if model_name is not None else [],
                },
            }
            results.append(entry)

    # Registry state as evidence, so the UI can say WHY a container is not
    # configurable instead of just disabling a button. tableConfigured=false
    # means the feature is off; tableReachable=false means it is on and broken.
    registry_block = _registry_store().snapshot()
    registry_block["warnings"] = registry_warnings
    registry_block["sharedIntents"] = intent_clashes

    return JSONResponse({
        "containers": results,
        "registry": registry_block,
        "triton": {
            "ready": triton_ready,
            "url": triton_url,
            "urlScope": "cluster-dns",
            "models": {name: info["state"] for name, info in triton_models.items()},
            "evidence": {
                "healthProbe": triton_evidence,
                "modelProbes": {name: info["evidence"] for name, info in triton_models.items()},
            },
        },
        "urlsNote": (
            "All grpc/mcp/triton URLs are Kubernetes-internal DNS names "
            "(<service>:<port>) and resolve only from inside the EKS cluster. "
            "Each entry's `evidence` includes the resolved IP, latency, HTTP "
            "status code, and timestamp from the orchestrator's probe."
        ),
    })


async def set_container_active(request: Request) -> JSONResponse:
    """POST /v1/containers/{name}/active — activate or deactivate a container.

    Body: ``{"active": true|false}``.

    Only store-defined containers can be toggled. A request naming one of the
    six code-defined containers is refused with 409 rather than ignored, so the
    caller learns the built-in bid path is not UI-mutable instead of watching a
    switch silently spring back.

    Fails closed: on any error the stored flag is left alone and the response
    carries the real reason. The effect is not immediate — every orchestrator
    replica reads the registry through a TTL cache, so the response reports the
    window rather than implying the change is global at once.
    """
    name = request.path_params.get("name", "")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Request body must be JSON."}, status_code=400)

    if not isinstance(body, dict) or not isinstance(body.get("active"), bool):
        # No coercion of "true"/1 — an ambiguous request is rejected rather than
        # guessed at, because guessing wrong silently changes bid behaviour.
        return JSONResponse(
            {"error": 'Body must be an object with a boolean "active" field.'},
            status_code=400,
        )
    active = body["active"]

    entries, _ = _effective_registry()
    entry = next((e for e in entries if e.name == name), None)
    if entry is None:
        return JSONResponse(
            {"error": f"No container named '{name}' in the registry."}, status_code=404
        )
    if not entry.configurable:
        return JSONResponse(
            {
                "error": (
                    f"'{name}' is a built-in container defined in the orchestrator's code. "
                    f"Built-in containers cannot be renamed, re-targeted or deactivated from "
                    f"the API."
                ),
                "source": entry.source,
            },
            status_code=409,
        )

    # CognitoAuthMiddleware has already validated the token and attached the
    # claims as request.state.user (auth.py:174). This only reads the subject off
    # them for provenance, and falls back to "unknown" rather than inventing an
    # identity — the middleware can be bypassed for health paths, and a claim
    # that is not there should not be filled in.
    updated_by = "unknown"
    claims = getattr(request.state, "user", None)
    if isinstance(claims, dict):
        updated_by = claims.get("sub") or claims.get("username") or claims.get("email") or "unknown"

    store = _registry_store()
    try:
        record = store.set_active(name, active, updated_by=updated_by)
    except RegistryRecordNotFound as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except RegistryStoreUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:  # defensive: never leak an unhandled 500 body
        return JSONResponse(
            {"error": f"Could not update '{name}': {type(exc).__name__}: {exc}"},
            status_code=500,
        )

    ttl = int(store.ttl_seconds)
    verb = "active" if active else "inactive"
    payload = {
        "ok": True,
        "container": name,
        "active": bool(record.get("active", active)),
        "effectiveWithinSeconds": ttl,
        "updatedAt": record.get("updated_at"),
        "message": (
            f"{entry.display_name or name} is now {verb}. Orchestrator replicas read the "
            f"registry on a {ttl}s cache, so bid traffic reflects this within {ttl} seconds."
        ),
    }

    # Activating a container that shares an intent with an already-active one is
    # allowed, and both will be called. But only one mutation per (path, intent)
    # survives, so the operator needs to be told which — otherwise the losing
    # container looks like it is contributing when it is not.
    if active:
        contest = _describe_intent_contest(name)
        if contest:
            payload["intentContest"] = contest

    return JSONResponse(payload)


def _describe_intent_contest(name: str) -> dict | None:
    """Who else claims this container's intents, and who would win.

    Returns None when nothing is contested. The winner is derived from the same
    rule the bid path uses — higher priority, then later registry order — so the
    message cannot disagree with what actually happens.
    """
    entries, _ = _effective_registry()
    by_name = {e.name: e for e in entries}
    subject = by_name.get(name)
    if subject is None:
        return None

    order = {e.name: i for i, e in enumerate(entries)}
    contested: list[dict] = []

    for intent in sorted(subject.intents):
        rivals = [
            e for e in entries
            if e.name != name and e.active and intent in e.intents
        ]
        if not rivals:
            continue
        claimants = [subject, *rivals]
        winner = max(claimants, key=lambda e: (e.priority, order.get(e.name, 0)))
        contested.append({
            "intent": intent,
            "claimants": [
                {"name": e.name, "priority": e.priority, "source": e.source}
                for e in claimants
            ],
            "winner": winner.name,
        })

    if not contested:
        return None

    return {
        "contested": contested,
        "message": (
            "More than one active container claims these intents. Every claimant is "
            "called, but for a given path only the highest-priority one's mutation is "
            "returned; equal priorities fall back to registry order, where store "
            "containers follow built-in ones. Set 'priority' on a registry record to "
            "change the outcome."
        ),
    }


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def mcp_proxy(request: Request) -> JSONResponse:
    """Proxy MCP JSON-RPC calls to the first available container's /mcp endpoint."""
    body = await request.json()
    method = body.get("method", "")

    # For initialize and tools/list, respond directly (orchestrator acts as MCP server)
    if method == "initialize":
        session_id = str(uuid.uuid4())
        return JSONResponse(
            {"jsonrpc": "2.0", "id": body.get("id"), "result": {
                "name": "nvidia-artf-recommenders", "version": "0.1.0",
                "protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
            }},
            headers={"Mcp-Session-Id": session_id},
        )

    if method == "notifications/initialized":
        return JSONResponse({"jsonrpc": "2.0", "id": body.get("id"), "result": {}})

    if method == "tools/list":
        return JSONResponse({"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": [{
            "name": "extend_rtb",
            "description": "Process OpenRTB bid request/response through Accelerator-optimized Agentic Bidding containers and return proposed mutations.",
            "inputSchema": {
                "type": "object", "required": ["id", "bid_request"],
                "properties": {
                    "id": {"type": "string"}, "tmax": {"type": "integer", "default": 100},
                    "bid_request": {"type": "object"}, "bid_response": {"type": "object"},
                    "lifecycle": {"type": "string"}, "originator": {"type": "object"},
                    "applicable_intents": {"type": "array", "items": {"type": "string"}},
                },
            },
        }]}}
        )

    if method == "tools/call":
        # Route extend_rtb through the orchestrator's mutation pipeline
        params = body.get("params", {})
        arguments = params.get("arguments", {})
        try:
            req = RTBRequest(**arguments)
            # Reuse the same mutation logic as POST /v1/mutations
            tmax = max(req.tmax, 10)
            timeout_s = tmax / 1000.0
            payload = req.model_dump()
            payload_bytes = json.dumps(payload).encode()

            # Same fan-out as POST /v1/mutations, including activation gating —
            # one implementation, so this path cannot drift from that one.
            applicable = getattr(req, "applicable_intents", None) or arguments.get("applicable_intents")

            start = time.monotonic()
            all_invocations = await _fan_out(
                payload, payload_bytes, applicable, timeout_s=timeout_s
            )

            # Same resolution as POST /v1/mutations. These two paths have drifted
            # before, which is why the fan-out was unified; the resolution goes
            # through the same helper for the same reason.
            all_mutations, conflicts = _resolve_for_response(all_invocations)

            elapsed_ms = (time.monotonic() - start) * 1000
            resp = RTBResponse(
                id=req.id, mutations=all_mutations,
                metadata=Metadata(
                    api_version="1.0",
                    model_version=f"orchestrator-v1 ({len(all_mutations)} mutations, {elapsed_ms:.1f}ms)",
                    containers=all_invocations,
                    conflicts=conflicts or None,
                ),
            )
            # Emit bid outcome event (fire-and-forget, non-blocking)
            emit_bid_outcome(req, resp, start)
            # Emit deal yield outcome event(s) for any adjust_deal mutations
            emit_deal_yield_outcome(req, resp)

            return JSONResponse({"jsonrpc": "2.0", "id": body.get("id"), "result": {
                "content": [{"type": "text", "text": json.dumps(resp.model_dump())}],
            }}, headers={"Mcp-Session-Id": request.headers.get("mcp-session-id", "")})
        except Exception as exc:
            return JSONResponse({"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32000, "message": str(exc)}})

    return JSONResponse({"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": f"Method not found: {method}"}})


# ---------------------------------------------------------------------------
# GPU node group control (cost management)
# ---------------------------------------------------------------------------

_EKS_CLUSTER = os.environ.get("EKS_CLUSTER_NAME", "nvidia-artf-recommenders-triton")
_GPU_NODEGROUP = os.environ.get("GPU_NODEGROUP_NAME", "gpu-inference")
_AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-1"))


# Upper bound on a legitimate cold start, measured from the node group's last
# scaling change: EC2 provisioning + node join, then the ~15GB
# nvcr.io/nvidia/tritonserver image pull (~6 min observed on g5.xlarge).
# Past this the pod is not "loading" any more -- something is wrong and the UI
# must say so rather than showing an indefinite progress state.
_TRITON_START_GRACE_S = 720


def _humanize_duration(seconds: float) -> str:
    """Coarse duration for operator-facing status text ("4d 17h", not "6666 min")."""
    total = int(seconds)
    if total < 3600:
        return f"{max(1, total // 60)} min"
    if total < 86400:
        h, m = divmod(total // 60, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    d, rem_h = divmod(total // 3600, 24)
    return f"{d}d {rem_h}h" if rem_h else f"{d}d"


async def gpu_status(request: Request) -> JSONResponse:
    """GET /v1/gpu/status — return GPU node group scaling state + Triton readiness.

    Reports a distinct `blocked` state when the node group has been settled at
    desiredSize>0 for longer than a cold start can account for. Triton spent 4
    days unschedulable (stale optimize Jobs held both GPUs) while this endpoint
    reported only `tritonReady: false`, which the UI rendered as "starting" --
    a permanent deadlock displayed as normal startup. The probe result is
    returned as evidence so the UI can show why, not just that.
    """
    import boto3
    try:
        eks = boto3.client("eks", region_name=_AWS_REGION)
        ng = eks.describe_nodegroup(clusterName=_EKS_CLUSTER, nodegroupName=_GPU_NODEGROUP)
        scaling = ng["nodegroup"]["scalingConfig"]
        ng_status = ng["nodegroup"]["status"]
        desired = scaling["desiredSize"]

        # Seconds since the node group's last scaling change. Server-side
        # timestamp from EKS, so it survives orchestrator restarts and is
        # consistent across replicas.
        settled_for = None
        modified_at = ng["nodegroup"].get("modifiedAt")
        if modified_at is not None:
            try:
                settled_for = max(0.0, time.time() - modified_at.timestamp())
            except AttributeError:
                settled_for = max(0.0, time.time() - float(modified_at))

        # Probe Triton, keeping the outcome as evidence rather than collapsing
        # it to a bare bool.
        triton_ready = False
        triton_detail = None
        if desired > 0:
            triton_url = os.environ.get("TRITON_URL", "triton-inference-server:8000")
            try:
                async with httpx.AsyncClient(timeout=2.0) as tc:
                    resp = await tc.get(f"http://{triton_url}/v2/health/ready")
                    triton_ready = resp.status_code == 200
                    if not triton_ready:
                        triton_detail = f"/v2/health/ready returned HTTP {resp.status_code}"
            except Exception as exc:
                triton_detail = f"{type(exc).__name__} connecting to {triton_url}"
        else:
            triton_detail = "GPU node group scaled to zero"

        if desired == 0:
            triton_state = "stopped"
        elif ng_status == "UPDATING":
            triton_state = "scaling_up" if desired > 0 else "scaling_down"
        elif triton_ready:
            triton_state = "ready"
        elif settled_for is None or settled_for < _TRITON_START_GRACE_S:
            triton_state = "starting"
        else:
            triton_state = "blocked"
            triton_detail = (
                f"Triton has not become ready in {_humanize_duration(settled_for)} since the GPU "
                f"node group settled at {desired} node(s) — longer than a cold start takes. "
                f"Last probe: {triton_detail}. Check: kubectl describe pod -l app=triton "
                f"(a Pending pod usually means nothing has a free nvidia.com/gpu)."
            )

        return JSONResponse({
            "status": ng_status,
            "desiredSize": desired,
            "minSize": scaling["minSize"],
            "maxSize": scaling["maxSize"],
            "running": desired > 0 and triton_ready,
            "tritonReady": triton_ready,
            "tritonState": triton_state,
            "tritonDetail": triton_detail,
            "secondsSinceNodegroupChange": int(settled_for) if settled_for is not None else None,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def gpu_start(request: Request) -> JSONResponse:
    """POST /v1/gpu/start — scale GPU node group to 3 nodes for Triton + NVIDIA containers."""
    import boto3
    try:
        eks = boto3.client("eks", region_name=_AWS_REGION)
        eks.update_nodegroup_config(
            clusterName=_EKS_CLUSTER,
            nodegroupName=_GPU_NODEGROUP,
            scalingConfig={"minSize": 1, "maxSize": 5, "desiredSize": 3},
        )
        return JSONResponse({"ok": True, "message": "GPU node group scaling to 3 nodes. Triton + NVIDIA containers will be ready in ~3-5 minutes."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def gpu_stop(request: Request) -> JSONResponse:
    """POST /v1/gpu/stop — scale GPU node group to 0."""
    import boto3
    try:
        eks = boto3.client("eks", region_name=_AWS_REGION)
        eks.update_nodegroup_config(
            clusterName=_EKS_CLUSTER,
            nodegroupName=_GPU_NODEGROUP,
            scalingConfig={"minSize": 0, "maxSize": 5, "desiredSize": 0},
        )
        return JSONResponse({"ok": True, "message": "GPU node group scaling to 0. GPU costs will stop in ~1-2 minutes."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Signal Associator (downstream signal → bid association)
# ---------------------------------------------------------------------------

try:
    from orchestrator.feedback_integration import (
        get_feedback_collector as _get_feedback_collector,
        set_signal_associator as _set_signal_associator,
    )
except ImportError:  # pragma: no cover - container-relative import
    from container.feedback_integration import (  # type: ignore
        get_feedback_collector as _get_feedback_collector,
        set_signal_associator as _set_signal_associator,
    )

# Only a last-resort name for the enriched-event stream. The deployment sets
# FEEDBACK_STREAM_NAME, which feedback_integration's collector already uses, and that
# collector is what this module prefers -- a second collector on a second stream name
# would put enriched outcome events somewhere the ETL never reads, with no error
# anywhere: the rows would just stay unlabelled.
_KINESIS_STREAM = os.environ.get(
    "FEEDBACK_KINESIS_STREAM",
    os.environ.get("FEEDBACK_STREAM_NAME", "artf-bid-outcomes"),
)
_SIGNAL_ASSOCIATOR: SignalAssociator | None = None


def _get_signal_associator() -> SignalAssociator:
    """Lazy-initialize the signal associator singleton.

    The instance is handed to feedback_integration so the bid path registers into
    the SAME cache this endpoint looks up in. Two associators would mean every
    signal misses its bid, which is indistinguishable from no signals arriving.

    It reuses the bid path's collector, so a bid event and the enriched event that
    replaces it land on the same stream and the ETL can de-duplicate them by
    request_id.
    """
    global _SIGNAL_ASSOCIATOR
    if _SIGNAL_ASSOCIATOR is None:
        collector = _get_feedback_collector()
        if collector is None:
            collector = FeedbackCollector(
                stream_name=_KINESIS_STREAM,
                region=_AWS_REGION,
            )
        _SIGNAL_ASSOCIATOR = SignalAssociator(feedback_collector=collector)
        _set_signal_associator(_SIGNAL_ASSOCIATOR)
    return _SIGNAL_ASSOCIATOR


# Wire the associator at import, not on the first signal. Lazily creating it when a
# signal arrives means the bid path has nothing to register into until then, so every
# signal before the first one misses — and a missed signal is indistinguishable from
# no signal having been sent.
try:
    _get_signal_associator()
except Exception as _exc:  # pragma: no cover - must never block startup
    print(f"[orchestrator] signal associator not wired: {_exc}")


async def receive_signal(request: Request) -> JSONResponse:
    """POST /v1/signals — receive a downstream signal and associate it with an originating bid.

    Delegates to the signal_receiver module which validates the payload,
    emits a SignalEvent to Kinesis for ETL joining, and passes the signal
    to the in-memory SignalAssociator for immediate enrichment.

    Requirements: 1.4
    """
    associator = _get_signal_associator()
    # `_feedback_collector` used to be named here and was never defined in this
    # module, so every request to this endpoint raised NameError and returned 500 --
    # no SignalEvent ever reached Kinesis. The signal_receiver tests pass a collector
    # in explicitly, so none of them exercised this wiring.
    return await _receive_signal_handler(
        request=request,
        signal_associator=associator,
        feedback_collector=_get_feedback_collector(),
    )


try:
    from orchestrator.loadtest import start_loadtest, get_loadtest, get_loadtest_history, cancel_loadtest, stream_loadtest  # noqa: E402
except ImportError:
    from container.loadtest import start_loadtest, get_loadtest, get_loadtest_history, cancel_loadtest, stream_loadtest  # noqa: E402

# Governance panel API (Train-from-Load-Test + Governance Outcome Comparison
# feature, Unit 2: train-from-load-test). Imported defensively like the
# closed-loop API below — if unavailable, these routes are simply not
# registered and the rest of the orchestrator is unaffected.
_GOVERNANCE_API_AVAILABLE = False
try:
    try:
        from orchestrator.governance_api import (  # noqa: E402
            training_estimate_handler as gov_training_estimate,
            train_handler as gov_train,
            eligible_runs_handler as gov_eligible_runs,
            trainable_runs_handler as gov_trainable_runs,
            sweep_status_handler as gov_sweep_status,
            stage_canary_handler as gov_stage_canary,
            compare_handler as gov_compare,
            comparison_pair_handler as gov_comparison_pair,
            promote_handler as gov_promote,
        )
    except ImportError:
        from governance_api import (  # noqa: E402
            training_estimate_handler as gov_training_estimate,
            train_handler as gov_train,
            eligible_runs_handler as gov_eligible_runs,
            trainable_runs_handler as gov_trainable_runs,
            sweep_status_handler as gov_sweep_status,
            stage_canary_handler as gov_stage_canary,
            compare_handler as gov_compare,
            comparison_pair_handler as gov_comparison_pair,
            promote_handler as gov_promote,
        )
    _GOVERNANCE_API_AVAILABLE = True
except Exception as _gov_exc:  # pragma: no cover - depends on image contents
    logging.getLogger(__name__).warning(
        "Governance training-trigger API unavailable (routes disabled): %s", _gov_exc
    )


def _governance_routes(prefix: str) -> list:
    """Build the governance routes (training trigger + comparison/promotion) under a given prefix."""
    if not _GOVERNANCE_API_AVAILABLE:
        return []
    return [
        Route(f"{prefix}/v1/governance/training-estimate", gov_training_estimate, methods=["GET"]),
        Route(f"{prefix}/v1/governance/train", gov_train, methods=["POST"]),
        Route(f"{prefix}/v1/governance/eligible-runs", gov_eligible_runs, methods=["GET"]),
        Route(f"{prefix}/v1/governance/trainable-runs", gov_trainable_runs, methods=["GET"]),
        Route(f"{prefix}/v1/governance/sweep-status", gov_sweep_status, methods=["GET"]),
        Route(f"{prefix}/v1/governance/comparison-pair", gov_comparison_pair, methods=["GET"]),
        Route(f"{prefix}/v1/governance/compare", gov_compare, methods=["POST"]),
        Route(f"{prefix}/v1/governance/stage-canary", gov_stage_canary, methods=["POST"]),
        Route(f"{prefix}/v1/governance/promote", gov_promote, methods=["POST"]),
    ]

# Closed-loop demo API (Part 2 visualization + controllable synthetic input).
# Imported defensively: it depends on the closed_loop_demo + agents packages,
# which must be present in the image. If they are not, the closed-loop routes
# are simply not registered — the rest of the orchestrator is unaffected.
_CLOSED_LOOP_AVAILABLE = False
try:
    try:
        from orchestrator.closed_loop_api import (  # noqa: E402
            list_scenarios_handler as cl_scenarios,
            generate_handler as cl_generate,
            sample_outcomes_handler as cl_sample_outcomes,
            parameters_handler as cl_parameters,
            audit_handler as cl_audit,
            models_handler as cl_models,
            metrics_handler as cl_metrics,
            schedule_status_handler as cl_schedule_status,
            schedule_toggle_handler as cl_schedule_toggle,
        )
    except ImportError:
        from closed_loop_api import (  # noqa: E402
            list_scenarios_handler as cl_scenarios,
            generate_handler as cl_generate,
            sample_outcomes_handler as cl_sample_outcomes,
            parameters_handler as cl_parameters,
            audit_handler as cl_audit,
            models_handler as cl_models,
            metrics_handler as cl_metrics,
            schedule_status_handler as cl_schedule_status,
            schedule_toggle_handler as cl_schedule_toggle,
        )
    _CLOSED_LOOP_AVAILABLE = True
except Exception as _cl_exc:  # pragma: no cover - depends on image contents
    logging.getLogger(__name__).warning(
        "Closed-loop demo API unavailable (routes disabled): %s", _cl_exc
    )


def _closed_loop_routes(prefix: str) -> list:
    """Build the closed-loop routes under a given prefix ('' or '/api')."""
    if not _CLOSED_LOOP_AVAILABLE:
        return []
    return [
        Route(f"{prefix}/v1/closed-loop/scenarios", cl_scenarios, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/generate", cl_generate, methods=["POST"]),
        Route(f"{prefix}/v1/closed-loop/sample-outcomes", cl_sample_outcomes, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/parameters", cl_parameters, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/audit", cl_audit, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/models", cl_models, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/metrics", cl_metrics, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/schedule", cl_schedule_status, methods=["GET"]),
        Route(f"{prefix}/v1/closed-loop/schedule", cl_schedule_toggle, methods=["POST"]),
    ]


# Live auction through Prebid Server. Prebid is a ClusterIP Service with no public
# address, so the browser cannot reach it; the orchestrator makes the hop on the
# `/api/*` path the frontend already uses. Registered UNCONDITIONALLY when the module
# imports: the routes exist whether or not Prebid is deployed, and report that
# honestly, because a missing route and an undeployed exchange are different facts
# and a 404 would not distinguish them.
_AUCTION_AVAILABLE = False
try:
    try:
        from orchestrator.auction_api import (  # noqa: E402
            auction_status_handler as auction_status,
            run_auction_handler as auction_run,
        )
    except ImportError:
        from auction_api import (  # noqa: E402
            auction_status_handler as auction_status,
            run_auction_handler as auction_run,
        )
    _AUCTION_AVAILABLE = True
except Exception as _auction_exc:  # pragma: no cover - depends on image contents
    logging.getLogger(__name__).warning(
        "Live auction API unavailable (routes disabled): %s", _auction_exc
    )


def _auction_routes(prefix: str) -> list:
    """Build the live-auction routes under a given prefix ('' or '/api')."""
    if not _AUCTION_AVAILABLE:
        return []
    return [
        Route(f"{prefix}/v1/auction/status", auction_status, methods=["GET"]),
        Route(f"{prefix}/v1/auction/run", auction_run, methods=["POST"]),
    ]


routes = [
    Route("/v1/mutations", get_mutations, methods=["POST"]),
    Route("/v1/containers", list_containers),
    Route("/v1/containers/{name}/active", set_container_active, methods=["POST"]),
    Route("/v1/signals", receive_signal, methods=["POST"]),
    Route("/v1/gpu/status", gpu_status),
    Route("/v1/gpu/start", gpu_start, methods=["POST"]),
    Route("/v1/gpu/stop", gpu_stop, methods=["POST"]),
    Route("/v1/loadtest", start_loadtest, methods=["POST"]),
    Route("/v1/loadtest/history", get_loadtest_history, methods=["GET"]),
    Route("/v1/loadtest/{id}/stream", stream_loadtest, methods=["GET"]),
    Route("/v1/loadtest/{id}", get_loadtest, methods=["GET"]),
    Route("/v1/loadtest/{id}", cancel_loadtest, methods=["DELETE"]),
    Route("/mcp", mcp_proxy, methods=["POST", "GET", "DELETE", "OPTIONS"]),
    Route("/health/live", health),
    Route("/health/ready", health),
    # CloudFront proxies /api/* from the frontend
    Route("/api/v1/mutations", get_mutations, methods=["POST"]),
    Route("/api/v1/containers", list_containers),
    Route("/api/v1/containers/{name}/active", set_container_active, methods=["POST"]),
    Route("/api/v1/signals", receive_signal, methods=["POST"]),
    Route("/api/v1/gpu/status", gpu_status),
    Route("/api/v1/gpu/start", gpu_start, methods=["POST"]),
    Route("/api/v1/gpu/stop", gpu_stop, methods=["POST"]),
    Route("/api/v1/loadtest", start_loadtest, methods=["POST"]),
    Route("/api/v1/loadtest/history", get_loadtest_history, methods=["GET"]),
    Route("/api/v1/loadtest/{id}/stream", stream_loadtest, methods=["GET"]),
    Route("/api/v1/loadtest/{id}", get_loadtest, methods=["GET"]),
    Route("/api/v1/loadtest/{id}", cancel_loadtest, methods=["DELETE"]),
    Route("/api/mcp", mcp_proxy, methods=["POST", "GET", "DELETE", "OPTIONS"]),
    Route("/api/health/ready", health),
    # RTB Fabric path — same handlers, different prefix for CloudFront routing
    Route("/fabric/v1/mutations", get_mutations, methods=["POST"]),
    Route("/fabric/v1/containers", list_containers),
    Route("/fabric/v1/containers/{name}/active", set_container_active, methods=["POST"]),
    Route("/fabric/mcp", mcp_proxy, methods=["POST", "GET", "DELETE", "OPTIONS"]),
    Route("/fabric/health/ready", health),
]

# Closed-loop demo routes on both the direct ('/v1/...') and CloudFront ('/api/v1/...') prefixes.
routes += _closed_loop_routes("")
routes += _closed_loop_routes("/api")

# Governance training-trigger routes (Train-from-Load-Test feature, Unit 2).
routes += _governance_routes("")
routes += _governance_routes("/api")

# Both prefixes, for the same reason the others use both: CloudFront routes /api/*
# to this service, while in-cluster callers use the bare path.
routes += _auction_routes("")
routes += _auction_routes("/api")

app = Starlette(routes=routes)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["Mcp-Session-Id"])

# Cognito JWT auth — rejects unauthenticated requests on all non-health endpoints
try:
    from orchestrator.auth import CognitoAuthMiddleware
except ImportError:
    try:
        from container.auth import CognitoAuthMiddleware
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from auth import CognitoAuthMiddleware
app.add_middleware(CognitoAuthMiddleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
