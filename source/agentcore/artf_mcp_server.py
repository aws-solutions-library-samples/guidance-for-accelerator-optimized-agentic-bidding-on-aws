"""AgentCore MCP Server — wraps the ARTF orchestrator as a Bedrock AgentCore tool.

This is the entrypoint that runs inside AgentCore's microVM.  It exposes
the ``extend_rtb`` tool via MCP (JSON-RPC) on port 8000 at ``/mcp``, and
a ``/ping`` health endpoint on port 8080, matching AgentCore's protocol
contract for MCP-type runtimes.

The server delegates to the four ARTF containers (DLRM, Wide&Deep, NCF,
Metrics) which run as sidecar processes inside the same microVM, or as
external services reachable via environment variables.

For local development, the containers run in-process (imported directly).
For production, set ``ARTF_MODE=remote`` and provide container URLs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import asyncio
import time
from typing import Any

# Ensure shared/ and containers/ are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from shared.artf_types import (
    Intent,
    Metadata,
    Mutation,
    RTBRequest,
    RTBResponse,
)

logger = logging.getLogger("agentcore.artf")

# ---------------------------------------------------------------------------
# Mode: "local" runs containers in-process, "remote" calls external URLs
# ---------------------------------------------------------------------------
ARTF_MODE = os.environ.get("ARTF_MODE", "local")

# Remote container URLs (used when ARTF_MODE=remote)
REMOTE_CONTAINERS = [
    {"name": "dlrm-bid-shader", "url": os.environ.get("DLRM_URL", "http://localhost:50061/mutate")},
    {"name": "widedeep-segment-activator", "url": os.environ.get("WIDEDEEP_URL", "http://localhost:50062/mutate")},
    {"name": "ncf-deal-manager", "url": os.environ.get("NCF_URL", "http://localhost:50063/mutate")},
    {"name": "metrics-enricher", "url": os.environ.get("METRICS_URL", "http://localhost:50064/mutate")},
]

# Local mutate functions (loaded lazily)
_local_mutators: list[Any] | None = None


def _load_local_mutators():
    global _local_mutators
    if _local_mutators is not None:
        return _local_mutators
    _local_mutators = []
    try:
        from containers.dlrm_bid_shader.app import mutate as dlrm
        _local_mutators.append(("dlrm-bid-shader", dlrm))
    except ImportError as e:
        logger.warning("DLRM container not available locally: %s", e)
    try:
        from containers.widedeep_segment_activator.app import mutate as wd
        _local_mutators.append(("widedeep-segment-activator", wd))
    except ImportError as e:
        logger.warning("Wide&Deep container not available locally: %s", e)
    try:
        from containers.ncf_deal_manager.app import mutate as ncf
        _local_mutators.append(("ncf-deal-manager", ncf))
    except ImportError as e:
        logger.warning("NCF container not available locally: %s", e)
    try:
        from containers.metrics_enricher.app import mutate as metrics
        _local_mutators.append(("metrics-enricher", metrics))
    except ImportError as e:
        logger.warning("Metrics container not available locally: %s", e)
    return _local_mutators


def _run_local(req: RTBRequest) -> RTBResponse:
    """Run all containers in-process and merge mutations."""
    mutators = _load_local_mutators()
    all_mutations: list[Mutation] = []
    for name, fn in mutators:
        try:
            resp = fn(req)
            all_mutations.extend(resp.mutations)
        except Exception as exc:
            logger.error("Container %s error: %s", name, exc)
    return RTBResponse(
        id=req.id,
        mutations=all_mutations,
        metadata=Metadata(api_version="1.0", model_version="agentcore-local"),
    )


async def _run_remote(req: RTBRequest) -> RTBResponse:
    """Call remote containers via HTTP and merge mutations."""
    payload = req.model_dump()
    all_mutations: list[Mutation] = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for c in REMOTE_CONTAINERS:
            tasks.append(_call_remote(client, c, payload, req.tmax / 1000.0))
        results = await asyncio.gather(*tasks)
    for mutations in results:
        all_mutations.extend(mutations)
    return RTBResponse(
        id=req.id,
        mutations=all_mutations,
        metadata=Metadata(api_version="1.0", model_version="agentcore-remote"),
    )


async def _call_remote(client: httpx.AsyncClient, container: dict, payload: dict, timeout: float) -> list[Mutation]:
    try:
        resp = await client.post(container["url"], json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return [Mutation(**m) for m in data.get("mutations", [])]
    except Exception as exc:
        logger.error("Remote %s error: %s", container["name"], exc)
    return []


# ---------------------------------------------------------------------------
# MCP Tool Definition
# ---------------------------------------------------------------------------

TOOLS = [{
    "name": "extend_rtb",
    "description": (
        "Process an OpenRTB bid request/response through Accelerator-optimized Agentic Bidding "
        "containers (DLRM, Wide&Deep, NCF, Metrics) and return proposed mutations. "
        "Supports ARTF intents: ACTIVATE_SEGMENTS, ACTIVATE_DEALS, SUPPRESS_DEALS, "
        "BID_SHADE, ADD_METRICS."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["id", "bid_request"],
        "properties": {
            "id": {"type": "string", "description": "Unique request ID"},
            "tmax": {"type": "integer", "description": "Max response time in ms", "default": 100},
            "bid_request": {"type": "object", "description": "OpenRTB v2.6 BidRequest"},
            "bid_response": {"type": "object", "description": "OpenRTB v2.6 BidResponse (required for BID_SHADE)"},
            "lifecycle": {"type": "string", "description": "LIFECYCLE_PUBLISHER_BID_REQUEST or LIFECYCLE_DSP_BID_RESPONSE"},
            "originator": {"type": "object", "description": "Business entity originator"},
            "applicable_intents": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter which intents are returned. Empty = all.",
            },
            "ext": {
                "type": "object",
                "description": "ARTF extension object for nonstandard signaling (e.g. ext.model_params).",
            },
        },
    },
}]

SERVER_INFO = {
    "name": "nvidia-artf-recommenders",
    "version": "0.1.0",
    "protocolVersion": "2025-03-26",
    "capabilities": {"tools": {}},
}


# ---------------------------------------------------------------------------
# Starlette app — MCP on :8000, Health on :8080
# ---------------------------------------------------------------------------

async def mcp_endpoint(request: Request) -> JSONResponse:
    """MCP JSON-RPC handler at /mcp."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    if method == "initialize":
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": SERVER_INFO})

    if method == "notifications/initialized":
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": {}})

    if method == "tools/list":
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": TOOLS}})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name != "extend_rtb":
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}})
        try:
            req = RTBRequest(**arguments)
            if ARTF_MODE == "remote":
                resp = await _run_remote(req)
            else:
                resp = _run_local(req)
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": {
                "content": [{"type": "text", "text": json.dumps(resp.model_dump())}],
            }})
        except Exception as exc:
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32000, "message": str(exc)}})

    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})


async def ping(request: Request) -> JSONResponse:
    """AgentCore health check — must return {"status": "Healthy"}."""
    return JSONResponse({"status": "Healthy"})


# MCP app on port 8000
mcp_app = Starlette(routes=[
    Route("/mcp", mcp_endpoint, methods=["POST", "GET", "DELETE", "OPTIONS"]),
    Route("/ping", ping),
])
mcp_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["Mcp-Session-Id"])


# ---------------------------------------------------------------------------
# Main — run both MCP (:8000) and health (:8080) servers
# ---------------------------------------------------------------------------

def main():
    import threading
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("ARTF AgentCore MCP Server starting (mode=%s)", ARTF_MODE)

    # Health server on :8080
    health_app = Starlette(routes=[
        Route("/ping", ping),
        Route("/health/live", ping),
        Route("/health/ready", ping),
    ])

    def _run_health():
        uvicorn.run(health_app, host="0.0.0.0", port=8080, log_level="warning")

    health_thread = threading.Thread(target=_run_health, daemon=True)
    health_thread.start()
    logger.info("Health/ping on :8080")

    # MCP server on :8000
    logger.info("MCP (JSON-RPC) on :8000/mcp")
    uvicorn.run(mcp_app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
