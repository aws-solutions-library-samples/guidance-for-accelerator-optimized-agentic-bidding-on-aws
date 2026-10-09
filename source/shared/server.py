"""Multi-protocol ARTF server — gRPC + MCP (JSON-RPC) + Health + Web UI.

Each ARTF container runs this server, which exposes:

- **gRPC** on port 50051 — ``RTBExtensionPoint.GetMutations`` per the ARTF
  proto service definition.

- **MCP** (JSON-RPC over Streamable HTTP) on port 8081 at ``/mcp`` — the
  ``extend_rtb`` tool, with session management via ``Mcp-Session-Id``
  headers per the MCP spec.  When co-hosted with the Web UI (port 8081),
  MCP is served on the same port at ``/mcp``.

- **Health** on port 8080 — ``/health/live`` and ``/health/ready``.

- **Web UI** on port 8081 at ``/`` — served alongside MCP.

The container provides a ``mutate(RTBRequest) -> RTBResponse`` function;
this module handles all protocol plumbing.

A container may also pass ``warmup`` to :func:`run_artf_server`. ``/health/ready``
then answers 503 until two consecutive warm-up ``mutate()`` calls have completed
under ``ARTF_WARMUP_BUDGET_MS`` (see :class:`WarmupGate`). Liveness is not gated.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import signal
import threading
import time
import uuid
from concurrent import futures
from typing import Any, Callable

import grpc
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from shared import hop_timing
from shared.artf_types import RTBRequest, RTBResponse
from shared.load_test_context import (
    HEADER_NAME,
    IS_LOAD_TEST_HEADER_NAME,
    load_test_scope,
    target_variant_scope,
)

logger = logging.getLogger("artf.server")

# Type alias for the container's mutation handler
MutateFunc = Callable[[RTBRequest], RTBResponse]


# ---------------------------------------------------------------------------
# mutate() offloading
# ---------------------------------------------------------------------------
#
# mutate() is a synchronous function that reaches blocking network I/O:
# dlrm_bid_shader/ncf_deal_manager call Triton through tritonclient.http, the
# SYNCHRONOUS Triton client. Calling it inline from an async handler blocks the
# single uvicorn event loop for the whole Triton round trip, so concurrent
# requests do not overlap -- they queue, and the Nth in flight waits N round
# trips. Measured effect: a model that infers in single-digit milliseconds
# reported hundreds of milliseconds as soon as load-test concurrency exceeded 1.
#
# Both HTTP entry points therefore run mutate() in this pool instead.
ARTF_MUTATE_WORKERS = max(1, int(os.environ.get("ARTF_MUTATE_WORKERS", "16")))

_mutate_executor = futures.ThreadPoolExecutor(
    max_workers=ARTF_MUTATE_WORKERS, thread_name_prefix="artf-mutate",
)


async def _await_mutate(mutate_fn: MutateFunc, req: RTBRequest) -> RTBResponse:
    """Run a container's synchronous mutate() off the event loop.

    The context is copied explicitly: run_in_executor does not propagate
    contextvars, and target_variant_scope/load_test_scope (shared/
    load_test_context.py) are ContextVars that mutate() reads. Without the
    copy, a load test's X-Load-Test-Target-Variant would silently stop
    reaching the Triton router. asyncio.to_thread would copy the context for
    us but runs on the default executor, shared with unrelated callers and
    sized from the host's CPU count rather than this pod's limit.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(_mutate_executor, lambda: ctx.run(mutate_fn, req))


def _timed_mutate(mutate_fn: MutateFunc, req: RTBRequest, timing: dict[str, float]) -> RTBResponse:
    """Run mutate_fn and fill the ``queue``/``mutate``/``triton`` segments.

    ``timing["_handler_start"]`` (perf_counter, set by the caller before the
    offload) gives the queue wait; it is popped so only segments reach the wire.
    Runs on the worker thread, so the Triton accumulator started by the caller is
    the one the inference clients append to (same list object across the copied
    context -- see hop_timing).
    """
    started = time.perf_counter()
    handler_start = timing.pop("_handler_start", started)
    timing["queue"] = hop_timing.ms(handler_start, started)
    calls = hop_timing.start_triton_accumulator()
    try:
        resp = mutate_fn(req)
    finally:
        timing["mutate"] = hop_timing.ms(started)
        timing["triton"] = hop_timing.triton_ms(calls)
        hop_timing.clear_triton_accumulator()
    return resp


def _attach_timing(resp: RTBResponse, timing: dict[str, float], handler_start: float) -> dict:
    """Serialise the response with ``metadata.timing`` filled in.

    ``build`` is the model_dump cost; ``total`` is handler entry to this point.
    Returns the dict ready for the wire so the JSON step is inside ``build``.
    """
    build_start = time.perf_counter()
    data = resp.model_dump()
    timing["build"] = hop_timing.ms(build_start)
    timing["total"] = hop_timing.ms(handler_start)
    data.setdefault("metadata", {})["timing"] = {k: timing[k] for k in hop_timing.CONTAINER_SEGMENTS if k in timing}
    return data


# ---------------------------------------------------------------------------
# Readiness warm-up
# ---------------------------------------------------------------------------
#
# The first mutate() after a pod starts pays for connection setup to Triton, the
# first inference on each engine and, in dlrm_bid_shader, the first parameter
# store read. The readiness probe used to pass as soon as the process listened,
# so that first request was a live bid. With a warm-up the probe stays 503 until
# a canned request has been through mutate() inside the latency budget twice in a row.

WarmupFunc = Callable[[], None]

#: Milliseconds each warm-up mutate() may take (Q18 A).
ARTF_WARMUP_BUDGET_MS = float(os.environ.get("ARTF_WARMUP_BUDGET_MS", "20"))
#: Consecutive in-budget calls that make the pod ready (Q18 A).
ARTF_WARMUP_CONSECUTIVE = 2
#: Attempts after which the gate logs ERROR and reports state ``failed``. The pod
#: stays unready and keeps retrying at the longest backoff: Triton loads its
#: repository from S3 and runs its own model_warmup before it serves, which can
#: take minutes, and only an in-budget pair ever makes the pod ready.
ARTF_WARMUP_MAX_ATTEMPTS = max(1, int(os.environ.get("ARTF_WARMUP_MAX_ATTEMPTS", "50")))
#: Backoff after a failed or over-budget attempt: doubles from the first value to the second.
ARTF_WARMUP_BACKOFF_S = (0.1, 5.0)

#: Request id prefix for warm-up requests, so they are recognisable in logs.
WARMUP_ID_PREFIX = "artf-warmup-"


class WarmupNotServed(RuntimeError):
    """A warm-up mutate() returned without the inference it was meant to exercise."""


def warmup_from_request(
    mutate_fn: MutateFunc,
    request: dict[str, Any],
    *,
    require_triton: bool,
    check: Callable[[RTBResponse], None] | None = None,
) -> WarmupFunc:
    """A warm-up that runs ``mutate_fn`` once on ``request``.

    ``request`` is validated as an RTBRequest here, so a canned request that no
    longer fits the model fails at startup rather than on every attempt. With
    ``require_triton`` the call must reach the Triton client at least once
    (hop_timing records each ``infer``); ``check`` may reject the response, for
    example a DLRM abstention.
    """
    req = RTBRequest(**request)
    if not req.id.startswith(WARMUP_ID_PREFIX):
        raise ValueError(f"warm-up request id {req.id!r} must start with {WARMUP_ID_PREFIX!r}")

    def _warmup() -> None:
        calls = hop_timing.start_triton_accumulator()
        try:
            resp = mutate_fn(req)
        finally:
            hop_timing.clear_triton_accumulator()
        if require_triton and not calls:
            raise WarmupNotServed("mutate() made no Triton call")
        if check is not None:
            check(resp)

    return _warmup


class WarmupGate:
    """Readiness state driven by a container's warm-up (Q18 A).

    ``state`` is ``warming`` until ``consecutive`` calls in a row complete within
    ``budget_ms``, then ``ready``. An attempt that raises or completes over budget
    resets the run and is retried after a backoff that doubles from
    ``backoff_s[0]`` to ``backoff_s[1]``. After ``max_attempts`` attempts without
    readiness the state becomes ``failed`` (logged as an error): still unready,
    still retrying. Only ``ready`` reports ready.
    """

    def __init__(
        self,
        warmup: WarmupFunc,
        *,
        budget_ms: float = ARTF_WARMUP_BUDGET_MS,
        consecutive: int = ARTF_WARMUP_CONSECUTIVE,
        max_attempts: int = ARTF_WARMUP_MAX_ATTEMPTS,
        backoff_s: tuple[float, float] = ARTF_WARMUP_BACKOFF_S,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._warmup = warmup
        self.budget_ms = budget_ms
        self.consecutive = max(1, consecutive)
        self.max_attempts = max(1, max_attempts)
        self.backoff_s = backoff_s
        self._clock = clock
        self._sleep = sleep
        self.state = "warming"
        self.attempts = 0
        self.in_budget_run = 0
        self.last_ms: float | None = None
        self.last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def describe(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "attempts": self.attempts,
            "in_budget_run": self.in_budget_run,
            "consecutive": self.consecutive,
            "last_ms": self.last_ms,
            "budget_ms": self.budget_ms,
            "last_error": self.last_error,
        }

    def run(self) -> None:
        """Attempt warm-ups until ready. Blocks."""
        delay = self.backoff_s[0]
        while not self.ready:
            self.attempts += 1
            started = self._clock()
            try:
                self._warmup()
            except Exception as exc:  # noqa: BLE001 - every failure is retried
                self.last_ms = None
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("warm-up attempt %d failed: %s", self.attempts, self.last_error)
            else:
                self.last_ms = hop_timing.ms(started, self._clock())
                self.last_error = None
                if self.last_ms <= self.budget_ms:
                    self.in_budget_run += 1
                    if self.in_budget_run >= self.consecutive:
                        self.state = "ready"
                        logger.info("warm-up attempt %d completed in %.3f ms, %d in a row within %.1f ms; ready",
                                    self.attempts, self.last_ms, self.in_budget_run, self.budget_ms)
                        return
                    logger.info("warm-up attempt %d completed in %.3f ms (budget %.1f ms, %d of %d in a row)",
                                self.attempts, self.last_ms, self.budget_ms, self.in_budget_run, self.consecutive)
                    continue
                logger.warning("warm-up attempt %d completed in %.3f ms, over the %.1f ms budget",
                               self.attempts, self.last_ms, self.budget_ms)
            self.in_budget_run = 0
            if self.attempts >= self.max_attempts and self.state == "warming":
                self.state = "failed"
                logger.error("warm-up not ready after %d attempts (last %s); the pod stays unready "
                             "and keeps retrying every %.1f s",
                             self.attempts,
                             f"{self.last_ms:.3f} ms" if self.last_ms is not None else self.last_error,
                             self.backoff_s[1])
            self._sleep(delay)
            delay = min(delay * 2, self.backoff_s[1])

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name="artf-warmup", daemon=True)
        thread.start()
        return thread


def _ready_response(gate: WarmupGate | None) -> JSONResponse:
    """``/health/ready``: unchanged without a gate, 503 until the gate is ready."""
    if gate is None:
        return JSONResponse({"status": "ok"})
    if not gate.ready:
        return JSONResponse({"status": gate.state, "warmup": gate.describe()}, status_code=503)
    return JSONResponse({"status": "ok", "warmup": gate.describe()})


def _build_health_app(gate: WarmupGate | None = None) -> Starlette:
    return Starlette(routes=[
        Route("/health/live", lambda r: JSONResponse({"status": "ok"})),
        Route("/health/ready", lambda r: _ready_response(gate)),
    ])


# ---------------------------------------------------------------------------
# MCP Session Store
# ---------------------------------------------------------------------------

class _MCPSessionStore:
    """Simple in-memory session store for MCP Mcp-Session-Id tracking."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        # UUID4 = 36 chars, well within AgentCore's 33-256 char range
        # for session identifiers.
        sid = str(uuid.uuid4())
        self._sessions[sid] = {"created": time.time()}
        return sid

    def exists(self, sid: str) -> bool:
        return sid in self._sessions

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid, None)


_sessions = _MCPSessionStore()


# ---------------------------------------------------------------------------
# gRPC service (JSON-over-gRPC, matching the proto contract shape)
# ---------------------------------------------------------------------------

# We implement the gRPC service using the generic handler approach so we
# don't need compiled proto stubs.  The service name and method match the
# ARTF proto exactly:
#   com.iabtechlab.bidstream.mutation.services.v1.RTBExtensionPoint/GetMutations

_SERVICE = "com.iabtechlab.bidstream.mutation.services.v1.RTBExtensionPoint"
_METHOD = f"/{_SERVICE}/GetMutations"


def _load_test_signals(metadata) -> tuple[str | None, bool]:
    """(target variant, is_load_test) from gRPC invocation metadata.

    Same rule as mutate_rest: only "stable"/"canary" count as a variant; the
    load-test flag is the literal "1". ``metadata`` is an iterable of
    (key, value) pairs or None.
    """
    variant = None
    is_load_test = False
    for key, value in metadata or ():
        if key == HEADER_NAME.lower() and value in ("stable", "canary"):
            variant = value
        elif key == IS_LOAD_TEST_HEADER_NAME.lower() and value == "1":
            is_load_test = True
    return variant, is_load_test


class _RTBExtensionPointServicer:
    """gRPC servicer that delegates to the container's mutate function.

    Accepts JSON-encoded RTBRequest (application/grpc+json) or raw JSON
    bytes, and returns JSON-encoded RTBResponse.  This is compatible with
    grpcurl, grpc-web, and any client that speaks the ARTF proto.
    """

    def __init__(self, mutate_fn: MutateFunc) -> None:
        self._mutate = mutate_fn

    def GetMutations(self, request_bytes: bytes, context: grpc.ServicerContext) -> bytes:
        handler_start = time.perf_counter()
        try:
            data = json.loads(request_bytes)
            req = RTBRequest(**data)
            timing: dict[str, float] = {"parse": hop_timing.ms(handler_start), "_handler_start": handler_start}
            # The same two out-of-band signals /mutate reads from HTTP headers
            # arrive here as gRPC metadata (the orchestrator lowercases the header
            # names). Without this the load test's variant targeting silently
            # stopped working the moment the fan-out moved to gRPC.
            variant, is_load_test = _load_test_signals(context.invocation_metadata())
            # No executor hop here: the gRPC server already runs this on one of
            # its own worker threads, so queue is the time to reach mutate().
            with target_variant_scope(variant), load_test_scope(is_load_test):
                resp = _timed_mutate(self._mutate, req, timing)
            return json.dumps(_attach_timing(resp, timing, handler_start)).encode()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return b"{}"


def _grpc_handler(mutate_fn: MutateFunc):
    """Build a generic gRPC method handler for GetMutations."""
    servicer = _RTBExtensionPointServicer(mutate_fn)

    def handler(request, context):
        return servicer.GetMutations(request, context)

    return grpc.unary_unary_rpc_method_handler(
        handler,
        request_deserializer=lambda x: x,  # raw bytes
        response_serializer=lambda x: x,   # raw bytes
    )


def _build_grpc_server(mutate_fn: MutateFunc, port: int) -> grpc.Server:
    # gRPC is the real-time bidding path and each mutate() blocks its worker
    # thread for a full Triton round trip, so the worker count is the hard
    # concurrency ceiling: at 4 workers the 5th concurrent bid request waits.
    # Shares ARTF_MUTATE_WORKERS with the HTTP pool so one env var tunes the
    # container's concurrency regardless of which protocol drives it.
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=ARTF_MUTATE_WORKERS, thread_name_prefix="artf-grpc",
    ))

    # Register the generic handler using the correct grpc API
    rpc_handler = grpc.unary_unary_rpc_method_handler(
        lambda request, context: _RTBExtensionPointServicer(mutate_fn).GetMutations(request, context),
        request_deserializer=lambda x: x,  # raw bytes
        response_serializer=lambda x: x,   # raw bytes
    )
    generic_handler = grpc.method_handlers_generic_handler(
        _SERVICE,
        {"GetMutations": rpc_handler},
    )
    server.add_generic_rpc_handlers([generic_handler])
    server.add_insecure_port(f"0.0.0.0:{port}")
    return server


# ---------------------------------------------------------------------------
# MCP (JSON-RPC over HTTP) — extend_rtb tool
# ---------------------------------------------------------------------------

def _build_mcp_app(
    mutate_fn: MutateFunc, agent_name: str, samples_dir: str | None = None,
    gate: WarmupGate | None = None,
) -> Starlette:
    """Build a Starlette app serving MCP JSON-RPC at /mcp and health.

    ``gate`` makes this app's ``/health/ready`` report the same warm-up state as
    the health port.
    """

    # JSON-RPC helpers
    def _ok(id: Any, result: Any, headers: dict | None = None) -> JSONResponse:
        resp = JSONResponse({"jsonrpc": "2.0", "id": id, "result": result})
        if headers:
            for k, v in headers.items():
                resp.headers[k] = v
        return resp

    def _err(id: Any, code: int, message: str) -> JSONResponse:
        return JSONResponse({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})

    # MCP tool definition — matches the ARTF 01-MCP.md spec exactly
    TOOLS = [{
        "name": "extend_rtb",
        "description": (
            "Process an OpenRTB bid request/response and return proposed mutations "
            "for segment activation, deal management, bid shading, and metrics"
        ),
        "inputSchema": {
            "type": "object",
            "required": ["id", "bid_request"],
            "properties": {
                "lifecycle": {
                    "type": "string",
                    "description": "Auction lifecycle stage",
                    "enum": ["LIFECYCLE_UNSPECIFIED", "LIFECYCLE_PUBLISHER_BID_REQUEST", "LIFECYCLE_DSP_BID_RESPONSE"],
                    "default": "LIFECYCLE_UNSPECIFIED",
                },
                "id": {"type": "string", "description": "Unique request ID"},
                "tmax": {"type": "integer", "description": "Maximum response time in milliseconds", "default": 100},
                "bid_request": {"type": "object", "description": "OpenRTB v2.6 BidRequest object"},
                "bid_response": {"type": "object", "description": "OpenRTB v2.6 BidResponse object (optional, required for BID_SHADE)"},
                "originator": {"type": "object", "description": "Business entity originator with 'type' and 'id'"},
                "applicable_intents": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Filter which intents are returned. If omitted, all intents are applicable.",
                },
                "ext": {
                    "type": "object",
                    "description": "ARTF extension object for nonstandard signaling (e.g. ext.model_params).",
                },
            },
        },
    }]

    SERVER_INFO = {
        "name": agent_name,
        "version": "0.1.0",
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": agent_name, "version": "0.1.0"},
    }

    async def mcp_endpoint(request: Request) -> JSONResponse:
        """Handle MCP JSON-RPC requests with session management."""
        # Handle DELETE — terminate session
        if request.method == "DELETE":
            sid = request.headers.get("mcp-session-id", "")
            _sessions.delete(sid)
            return JSONResponse({"jsonrpc": "2.0", "id": None, "result": {}})

        # Handle OPTIONS — CORS preflight
        if request.method == "OPTIONS":
            return Response(status_code=200, headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Mcp-Session-Id, Last-Event-ID",
                "Access-Control-Expose-Headers": "Mcp-Session-Id",
            })

        try:
            body = await request.json()
        except Exception:
            return _err(None, -32700, "Parse error")

        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        if method == "initialize":
            # Create a new session and return the ID in the header
            session_id = _sessions.create()
            return _ok(rpc_id, SERVER_INFO, headers={"Mcp-Session-Id": session_id})

        if method == "notifications/initialized":
            return _ok(rpc_id, {})

        # For all other methods, validate session
        session_id = request.headers.get("mcp-session-id", "")

        if method == "tools/list":
            return _ok(rpc_id, {"tools": TOOLS}, headers={"Mcp-Session-Id": session_id} if session_id else None)

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            if tool_name != "extend_rtb":
                return _err(rpc_id, -32601, f"Unknown tool: {tool_name}")
            try:
                req = RTBRequest(**arguments)
                resp = await _await_mutate(mutate_fn, req)
                return _ok(rpc_id, {
                    "content": [{"type": "text", "text": json.dumps(resp.model_dump())}],
                }, headers={"Mcp-Session-Id": session_id} if session_id else None)
            except Exception as exc:
                return _err(rpc_id, -32000, str(exc))

        return _err(rpc_id, -32601, f"Method not found: {method}")

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def ready(request: Request) -> JSONResponse:
        return _ready_response(gate)

    async def mutate_rest(request: Request) -> JSONResponse:
        """Simple REST endpoint — POST RTBRequest JSON, get RTBResponse JSON.

        Reads two out-of-band headers, both set ONLY by the orchestrator's
        load-test invocation path (see source/orchestrator/loadtest.py /
        loadtest_targeting.py), and scopes them to this request's
        mutate_fn call:
        - X-Load-Test-Target-Variant — forces a specific Triton router
          variant ("stable"/"canary"); any other value (including absent)
          leaves the container's normal routing unaffected.
        - X-Load-Test — a plain load-test-origin signal, present (value
          "1") on every container call a load test makes. Real bid-serving
          traffic never sends this header, so get_is_load_test() is always
          False on that path.
        """
        handler_start = time.perf_counter()
        try:
            body = await request.json()
            req = RTBRequest(**body)
            timing: dict[str, float] = {"parse": hop_timing.ms(handler_start), "_handler_start": handler_start}
            raw_variant = request.headers.get(HEADER_NAME.lower())
            variant = raw_variant if raw_variant in ("stable", "canary") else None
            is_load_test = request.headers.get(IS_LOAD_TEST_HEADER_NAME.lower()) == "1"
            # The scopes are entered BEFORE the offload so _await_mutate's
            # copy_context() captures them for the worker thread. The timing
            # dict is mutated in place on that thread, which is why it is a
            # dict and not a ContextVar value.
            with target_variant_scope(variant), load_test_scope(is_load_test):
                resp = await _await_mutate(lambda r: _timed_mutate(mutate_fn, r, timing), req)
            return JSONResponse(_attach_timing(resp, timing, handler_start))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    routes = [
        Route("/mutate", mutate_rest, methods=["POST"]),
        Route("/mcp", mcp_endpoint, methods=["POST", "GET", "DELETE", "OPTIONS"]),
        Route("/health/live", health),
        Route("/health/ready", ready),
    ]

    app = Starlette(routes=routes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Mcp-Session-Id", "Last-Event-ID"],
        expose_headers=["Mcp-Session-Id"],
    )
    return app


# ---------------------------------------------------------------------------
# Unified multi-protocol launcher
# ---------------------------------------------------------------------------

def run_artf_server(
    mutate_fn: MutateFunc,
    *,
    agent_name: str = "artf-agent",
    grpc_port: int = 50051,
    mcp_port: int = 8081,
    health_port: int = 8080,
    warmup: WarmupFunc | None = None,
) -> None:
    """Start gRPC, MCP, and health servers for an ARTF container.

    ``warmup`` (optional) runs on a background thread once the servers are
    started; ``/health/ready`` on both HTTP ports is 503 until two consecutive
    calls complete under budget (see WarmupGate). Without it readiness is 200 from
    the start, as before. This blocks until interrupted.
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # --- gRPC ---
    grpc_server = _build_grpc_server(mutate_fn, grpc_port)
    grpc_server.start()
    logger.info("gRPC RTBExtensionPoint listening on :%d", grpc_port)

    # --- Readiness warm-up ---
    gate = WarmupGate(warmup) if warmup is not None else None

    # --- Health (simple HTTP) ---
    health_app = _build_health_app(gate)

    def _run_health():
        uvicorn.run(health_app, host="0.0.0.0", port=health_port, log_level="warning")

    health_thread = threading.Thread(target=_run_health, daemon=True)
    health_thread.start()
    logger.info("Health probes on :%d", health_port)

    if gate is not None:
        gate.start()
        logger.info("Readiness waits for warm-up (budget %.1f ms)", gate.budget_ms)

    # --- MCP + Web (Starlette on mcp_port) ---
    mcp_app = _build_mcp_app(mutate_fn, agent_name, gate=gate)
    logger.info("MCP (JSON-RPC) on :%d/mcp", mcp_port)
    uvicorn.run(mcp_app, host="0.0.0.0", port=mcp_port, log_level="info")
