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
"""

from __future__ import annotations

import asyncio
import json
import logging
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

from shared.artf_types import RTBRequest, RTBResponse

logger = logging.getLogger("artf.server")

# Type alias for the container's mutation handler
MutateFunc = Callable[[RTBRequest], RTBResponse]


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


class _RTBExtensionPointServicer:
    """gRPC servicer that delegates to the container's mutate function.

    Accepts JSON-encoded RTBRequest (application/grpc+json) or raw JSON
    bytes, and returns JSON-encoded RTBResponse.  This is compatible with
    grpcurl, grpc-web, and any client that speaks the ARTF proto.
    """

    def __init__(self, mutate_fn: MutateFunc) -> None:
        self._mutate = mutate_fn

    def GetMutations(self, request_bytes: bytes, context: grpc.ServicerContext) -> bytes:
        try:
            data = json.loads(request_bytes)
            req = RTBRequest(**data)
            resp = self._mutate(req)
            return json.dumps(resp.model_dump()).encode()
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

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

def _build_mcp_app(mutate_fn: MutateFunc, agent_name: str, samples_dir: str | None = None) -> Starlette:
    """Build a Starlette app serving MCP JSON-RPC at /mcp and health."""

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
                resp = mutate_fn(req)
                return _ok(rpc_id, {
                    "content": [{"type": "text", "text": json.dumps(resp.model_dump())}],
                }, headers={"Mcp-Session-Id": session_id} if session_id else None)
            except Exception as exc:
                return _err(rpc_id, -32000, str(exc))

        return _err(rpc_id, -32601, f"Method not found: {method}")

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def mutate_rest(request: Request) -> JSONResponse:
        """Simple REST endpoint — POST RTBRequest JSON, get RTBResponse JSON."""
        try:
            body = await request.json()
            req = RTBRequest(**body)
            resp = mutate_fn(req)
            return JSONResponse(resp.model_dump())
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    routes = [
        Route("/mutate", mutate_rest, methods=["POST"]),
        Route("/mcp", mcp_endpoint, methods=["POST", "GET", "DELETE", "OPTIONS"]),
        Route("/health/live", health),
        Route("/health/ready", health),
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
) -> None:
    """Start gRPC, MCP, and health servers for an ARTF container.

    This blocks until interrupted.
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # --- gRPC ---
    grpc_server = _build_grpc_server(mutate_fn, grpc_port)
    grpc_server.start()
    logger.info("gRPC RTBExtensionPoint listening on :%d", grpc_port)

    # --- Health (simple HTTP) ---
    health_app = Starlette(routes=[
        Route("/health/live", lambda r: JSONResponse({"status": "ok"})),
        Route("/health/ready", lambda r: JSONResponse({"status": "ok"})),
    ])

    def _run_health():
        uvicorn.run(health_app, host="0.0.0.0", port=health_port, log_level="warning")

    health_thread = threading.Thread(target=_run_health, daemon=True)
    health_thread.start()
    logger.info("Health probes on :%d", health_port)

    # --- MCP + Web (Starlette on mcp_port) ---
    mcp_app = _build_mcp_app(mutate_fn, agent_name)
    logger.info("MCP (JSON-RPC) on :%d/mcp", mcp_port)
    uvicorn.run(mcp_app, host="0.0.0.0", port=mcp_port, log_level="info")
