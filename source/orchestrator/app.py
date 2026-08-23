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

from shared.artf_types import ContainerInvocationModel, Metadata, Mutation, RTBRequest, RTBResponse  # noqa: E402
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

CONTAINERS = [
    {
        "name": "dlrm-bid-shader",
        "intents": {"BID_SHADE"},
        "grpc": os.environ.get("DLRM_GRPC", os.environ.get("DLRM_URL", "http://localhost:50061")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("DLRM_MCP", os.environ.get("DLRM_URL", "http://localhost:8091")),
    },
    {
        "name": "widedeep-segment-activator",
        "intents": {"ACTIVATE_SEGMENTS"},
        "grpc": os.environ.get("WIDEDEEP_GRPC", os.environ.get("WIDEDEEP_URL", "http://localhost:50062")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("WIDEDEEP_MCP", os.environ.get("WIDEDEEP_URL", "http://localhost:8092")),
    },
    {
        "name": "ncf-deal-manager",
        "intents": {"ACTIVATE_DEALS", "SUPPRESS_DEALS"},
        "grpc": os.environ.get("NCF_GRPC", os.environ.get("NCF_URL", "http://localhost:50063")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("NCF_MCP", os.environ.get("NCF_URL", "http://localhost:8093")),
    },
    {
        "name": "metrics-enricher",
        "intents": {"ADD_METRICS"},
        "grpc": os.environ.get("METRICS_GRPC", os.environ.get("METRICS_URL", "http://localhost:50064")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("METRICS_MCP", os.environ.get("METRICS_URL", "http://localhost:8094")),
    },
    {
        "name": "deal-yield-manager",
        "intents": {"ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"},
        "grpc": os.environ.get("YIELD_GRPC", os.environ.get("YIELD_URL", "http://localhost:50065")).replace("http://", "").rstrip("/"),
        "mcp": os.environ.get("YIELD_MCP", os.environ.get("YIELD_URL", "http://localhost:8095")),
    },

]


def _filter_containers(applicable_intents: list[str] | None) -> list[dict]:
    """Filter CONTAINERS to only those whose intents overlap with applicable_intents.
    If applicable_intents is None or empty, all containers are called (backward compat)."""
    if not applicable_intents:
        return CONTAINERS
    requested = set(applicable_intents)
    return [c for c in CONTAINERS if c["intents"] & requested]





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

async def _call_mcp(client: httpx.AsyncClient, base_url: str, payload: dict, timeout_s: float) -> list[Mutation]:
    """Call a container's extend_rtb tool via MCP JSON-RPC."""
    try:
        rpc_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "extend_rtb", "arguments": payload},
        }
        resp = await client.post(f"{base_url}/mcp", json=rpc_body, timeout=timeout_s)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", {})
            # MCP returns content array with text containing the RTBResponse JSON
            for content_item in result.get("content", []):
                if content_item.get("type") == "text":
                    rtb_resp = json.loads(content_item["text"])
                    return [Mutation(**m) for m in rtb_resp.get("mutations", [])]
            # If no content array, check if result itself has mutations (direct response)
            if "mutations" in result:
                return [Mutation(**m) for m in result.get("mutations", [])]
        else:
            print(f"[orchestrator] MCP to {base_url} returned {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        print(f"[orchestrator] MCP to {base_url} failed: {exc}")
    return []


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
) -> tuple[list[Mutation], str]:
    """Call a container's /mutate REST endpoint (MCP fallback).

    ``headers``, when provided, are forwarded on the REST call only — used
    exclusively by the orchestrator's load-test invocation path to set the
    out-of-band X-Load-Test-Target-Variant header (see
    orchestrator/loadtest_targeting.py). Never set by this function's other
    callers (get_mutations / real bid-serving).

    Returns (mutations, model_version). model_version is the container's
    per-request-resolved served model version (from RTBResponse.metadata),
    or "" if the container didn't return one — never fabricated.
    """
    # Try simple REST /mutate first (most reliable)
    try:
        resp = await client.post(
            f"{container['mcp']}/mutate", json=payload, timeout=timeout_s, headers=headers
        )
        if resp.status_code == 200:
            data = resp.json()
            mutations = [Mutation(**m) for m in data.get("mutations", [])]
            model_version = (data.get("metadata") or {}).get("model_version", "")
            if mutations:
                return mutations, model_version
    except Exception as exc:
        print(f"[orchestrator] REST /mutate to {container['mcp']} failed: {exc}")

    # Fallback to MCP JSON-RPC (no header support on this path today — the
    # load-test-only override only needs to work on the REST path, which is
    # the one load test's container calls exercise).
    mutations = await _call_mcp(client, container["mcp"], payload, timeout_s)
    return mutations, ""


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
    Returns a ``ContainerInvocationModel`` with:

    - ``status="ok"`` and the produced mutations on success,
    - ``status="timeout"`` and ``mutations=[]`` on ``asyncio.TimeoutError``,
    - ``status="failed"`` and ``mutations=[]`` on any other exception.

    ``headers`` is forwarded to ``_call_container`` unchanged — see its
    docstring for the load-test-only usage.
    """
    start = time.monotonic()
    try:
        mutations, model_version = await asyncio.wait_for(
            _call_container(client, container, payload, payload_bytes, timeout_s, headers=headers),
            timeout=timeout_s,
        )
        latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        return ContainerInvocationModel(
            name=container["name"],
            status="ok",
            latency_ms=latency_ms,
            mutations=mutations,
            model_version=model_version,
        )
    except asyncio.TimeoutError:
        latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        return ContainerInvocationModel(
            name=container["name"],
            status="timeout",
            latency_ms=latency_ms,
            mutations=[],
        )
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000.0, 2)
        print(f"[orchestrator] container {container['name']} failed: {exc}")
        return ContainerInvocationModel(
            name=container["name"],
            status="failed",
            latency_ms=latency_ms,
            mutations=[],
        )


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

    # Only call containers whose intents match applicable_intents
    applicable = getattr(req, "applicable_intents", None) or body.get("applicable_intents")
    active_containers = _filter_containers(applicable)

    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        tasks = [_call_container_timed(client, c, payload, payload_bytes, timeout_s) for c in active_containers]
        invocations: list[ContainerInvocationModel] = await asyncio.gather(*tasks)

    # For containers that were NOT called (filtered out), add a "skipped" entry
    # so the frontend knows they weren't invoked (not that they failed).
    active_names = {c["name"] for c in active_containers}
    all_invocations = []
    for c in CONTAINERS:
        if c["name"] in active_names:
            inv = next(i for i in invocations if i.name == c["name"])
            all_invocations.append(inv)
        else:
            all_invocations.append(ContainerInvocationModel(
                name=c["name"], status="skipped", latency_ms=0, mutations=[],
            ))

    # Preserve canonical CONTAINERS registry order for both the flattened
    # mutations list and the per-container attribution surfaced via metadata.
    all_mutations: list[Mutation] = [m for inv in all_invocations for m in inv.mutations]

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
    # deal_yield_manager is served as TWO independent single-output FIL
    # models, not one — Triton's FIL backend does not support multi-output
    # regression (confirmed against NVIDIA's FIL backend docs; see
    # aidlc-docs/construction/deal-yield-training-pipeline/tasks.md Group 1
    # and source/triton/model_repository/deal_yield_manager_floor/config.pbtxt).
    # A model literally named "deal_yield_manager" was never registered with
    # Triton post-correction, so probing for it always 400s and the
    # container health panel misreports the Yield Optimizer as
    # model_unavailable even when both sub-models are actually ready.
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

    # Map container names to their Triton model name(s). widedeep-segment-activator
    # maps to None like metrics-enricher — both are rules-based, no Triton model.
    # deal-yield-manager maps to a tuple of both sub-model names (see note above) —
    # every other entry maps to a single model name string.
    container_to_model = {
        "dlrm-bid-shader": "dlrm_bid_shader",
        "widedeep-segment-activator": None,  # rules-based, no Triton model
        "ncf-deal-manager": "ncf_deal_manager",
        "metrics-enricher": None,  # rules-based, no Triton model
        "deal-yield-manager": ("deal_yield_manager_floor", "deal_yield_manager_margin"),
    }

    async with httpx.AsyncClient(timeout=2.0) as client:
        for c in CONTAINERS:
            container_status = "unknown"
            protocol = "none"
            inference_status = "unknown"

            # Check container health via HTTP (MCP endpoint)
            health_url = c["mcp"].rstrip("/") + "/health/ready"
            http_probe = await _probe_http(client, health_url)
            if http_probe["ok"]:
                container_status = "ready"
                protocol = "mcp"

            # Check gRPC health if HTTP failed
            grpc_probe = None
            if container_status != "ready":
                grpc_probe = await _probe_grpc(c["grpc"])
                if grpc_probe["ok"]:
                    container_status = "ready"
                    protocol = "grpc"
                else:
                    container_status = "unreachable"

            # Determine inference readiness (depends on Triton for GPU containers).
            # deal-yield-manager maps to a tuple of two sub-model names (floor +
            # margin) — both must be READY for the container to be reported ready.
            model_name = container_to_model.get(c["name"])
            model_names_for_container = (
                model_name if isinstance(model_name, tuple)
                else ((model_name,) if model_name is not None else ())
            )
            if model_name is None:
                # Rules-based container (metrics-enricher) — no Triton dependency
                inference_status = "ready" if container_status == "ready" else "unavailable"
            elif not triton_ready:
                inference_status = "gpu_offline"
            elif all(
                triton_models.get(name, {}).get("state") == "READY"
                for name in model_names_for_container
            ):
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
                "name": c["name"],
                "grpc": c["grpc"],
                "mcp": c["mcp"],
                "urlScope": "cluster-dns",  # NOT reachable from a browser; resolves only inside the EKS cluster
                "status": overall,
                "containerStatus": container_status,
                "inferenceStatus": inference_status,
                "protocol": protocol,
                # "+".join over a 1-tuple is just that model's name, so this is
                # unchanged for every container except deal-yield-manager
                # (which now shows "deal_yield_manager_floor+deal_yield_manager_margin").
                "tritonModel": "+".join(model_names_for_container) or None,
                "evidence": {
                    "httpProbe": http_probe,
                    "grpcProbe": grpc_probe,
                    # One evidence entry per Triton sub-model probed for this
                    # container (usually 0 or 1; 2 for deal-yield-manager).
                    "tritonModelProbes": [
                        {"model": name, **triton_models.get(name, {}).get("evidence", {})}
                        for name in model_names_for_container
                    ],
                },
            }
            results.append(entry)

    return JSONResponse({
        "containers": results,
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

            # Only call containers whose intents match applicable_intents
            applicable = getattr(req, "applicable_intents", None) or arguments.get("applicable_intents")
            active_containers = _filter_containers(applicable)

            start = time.monotonic()
            async with httpx.AsyncClient() as client:
                tasks = [_call_container_timed(client, c, payload, payload_bytes, timeout_s) for c in active_containers]
                invocations: list[ContainerInvocationModel] = await asyncio.gather(*tasks)

            # For containers that were NOT called, add a "skipped" entry
            active_names = {c["name"] for c in active_containers}
            all_invocations = []
            for c in CONTAINERS:
                if c["name"] in active_names:
                    inv = next(i for i in invocations if i.name == c["name"])
                    all_invocations.append(inv)
                else:
                    all_invocations.append(ContainerInvocationModel(
                        name=c["name"], status="skipped", latency_ms=0, mutations=[],
                    ))

            all_mutations: list[Mutation] = [m for inv in all_invocations for m in inv.mutations]

            elapsed_ms = (time.monotonic() - start) * 1000
            resp = RTBResponse(
                id=req.id, mutations=all_mutations,
                metadata=Metadata(
                    api_version="1.0",
                    model_version=f"orchestrator-v1 ({len(all_mutations)} mutations, {elapsed_ms:.1f}ms)",
                    containers=all_invocations,
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


async def gpu_status(request: Request) -> JSONResponse:
    """GET /v1/gpu/status — return current GPU node group scaling state + Triton readiness."""
    import boto3
    try:
        eks = boto3.client("eks", region_name=_AWS_REGION)
        ng = eks.describe_nodegroup(clusterName=_EKS_CLUSTER, nodegroupName=_GPU_NODEGROUP)
        scaling = ng["nodegroup"]["scalingConfig"]
        ng_status = ng["nodegroup"]["status"]

        # Also check if Triton is actually serving (pod ready)
        triton_ready = False
        if scaling["desiredSize"] > 0:
            triton_url = os.environ.get("TRITON_URL", "triton-inference-server:8000")
            try:
                async with httpx.AsyncClient(timeout=2.0) as tc:
                    resp = await tc.get(f"http://{triton_url}/v2/health/ready")
                    triton_ready = resp.status_code == 200
            except Exception:
                pass

        return JSONResponse({
            "status": ng_status,
            "desiredSize": scaling["desiredSize"],
            "minSize": scaling["minSize"],
            "maxSize": scaling["maxSize"],
            "running": scaling["desiredSize"] > 0 and triton_ready,
            "tritonReady": triton_ready,
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

_KINESIS_STREAM = os.environ.get("FEEDBACK_KINESIS_STREAM", "artf-bid-outcomes")
_SIGNAL_ASSOCIATOR: SignalAssociator | None = None


def _get_signal_associator() -> SignalAssociator:
    """Lazy-initialize the signal associator singleton."""
    global _SIGNAL_ASSOCIATOR
    if _SIGNAL_ASSOCIATOR is None:
        collector = FeedbackCollector(
            stream_name=_KINESIS_STREAM,
            region=_AWS_REGION,
        )
        _SIGNAL_ASSOCIATOR = SignalAssociator(feedback_collector=collector)
    return _SIGNAL_ASSOCIATOR


async def receive_signal(request: Request) -> JSONResponse:
    """POST /v1/signals — receive a downstream signal and associate it with an originating bid.

    Delegates to the signal_receiver module which validates the payload,
    emits a SignalEvent to Kinesis for ETL joining, and passes the signal
    to the in-memory SignalAssociator for immediate enrichment.

    Requirements: 1.4
    """
    associator = _get_signal_associator()
    return await _receive_signal_handler(
        request=request,
        signal_associator=associator,
        feedback_collector=_feedback_collector,
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
            compare_handler as gov_compare,
            promote_handler as gov_promote,
        )
    except ImportError:
        from governance_api import (  # noqa: E402
            training_estimate_handler as gov_training_estimate,
            train_handler as gov_train,
            eligible_runs_handler as gov_eligible_runs,
            trainable_runs_handler as gov_trainable_runs,
            compare_handler as gov_compare,
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
        Route(f"{prefix}/v1/governance/compare", gov_compare, methods=["POST"]),
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


routes = [
    Route("/v1/mutations", get_mutations, methods=["POST"]),
    Route("/v1/containers", list_containers),
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
    Route("/fabric/mcp", mcp_proxy, methods=["POST", "GET", "DELETE", "OPTIONS"]),
    Route("/fabric/health/ready", health),
]

# Closed-loop demo routes on both the direct ('/v1/...') and CloudFront ('/api/v1/...') prefixes.
routes += _closed_loop_routes("")
routes += _closed_loop_routes("/api")

# Governance training-trigger routes (Train-from-Load-Test feature, Unit 2).
routes += _governance_routes("")
routes += _governance_routes("/api")

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
