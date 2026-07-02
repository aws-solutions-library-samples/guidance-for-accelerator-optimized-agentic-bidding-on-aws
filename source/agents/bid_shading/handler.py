"""AgentCore HTTP entrypoint for the Bid Shading Strategy Agent.

This is the handler that runs inside AgentCore's Firecracker microVM.

AgentCore HTTP protocol contract (from official docs):
- POST :8080/invocations  - invocation handler (JSON payload)
- GET  :8080/ping         - health check returning {"status": "Healthy"}

All on port 8080. ARM64 container on host 0.0.0.0.

Requirements: 7.1, 7.6, 11.1
"""

from __future__ import annotations

import json
import logging
import os
import sys
import asyncio
import time
from typing import Any

# Ensure shared/ and agents/ are importable from the AgentCore container context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.bid_shading.agent import BidShadingStrategyAgent, ParameterUpdate
from shared.parameter_store import ParameterStore

logger = logging.getLogger("agentcore.bid_shading")

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

PARAMETER_STORE_TABLE = os.environ.get("PARAMETER_STORE_TABLE", "parameter-store")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


# ---------------------------------------------------------------------------
# HTTP handlers — AgentCore contract
# ---------------------------------------------------------------------------


async def handle_invocation(request: Request) -> JSONResponse:
    """AgentCore HTTP invocation handler at POST /invocations.

    Initializes the BidShadingStrategyAgent, runs a single evaluation cycle,
    and returns the parameter updates made.
    """
    invocation_start = time.time()

    try:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # Initialize dependencies
        parameter_store = ParameterStore(
            table_name=PARAMETER_STORE_TABLE,
            region=AWS_REGION,
        )
        cloudwatch_client = boto3.client("cloudwatch", region_name=AWS_REGION)

        # Create agent instance
        agent = BidShadingStrategyAgent(
            parameter_store=parameter_store,
            cloudwatch_client=cloudwatch_client,
        )

        # Run the evaluation cycle
        updates = await agent.evaluate_and_adjust()

        # Serialize updates for the response
        serialized_updates = [
            {
                "parameter_name": u.parameter_name,
                "old_value": u.old_value,
                "new_value": u.new_value,
                "reason": u.reason,
                "confidence": u.confidence,
            }
            for u in updates
        ]

        duration_ms = (time.time() - invocation_start) * 1000.0

        return JSONResponse(
            {
                "status": "completed",
                "updates": serialized_updates,
                "updates_count": len(serialized_updates),
                "duration_ms": round(duration_ms, 2),
                "timestamp": invocation_start,
            },
            status_code=200,
        )

    except Exception as exc:
        logger.exception("Invocation failed: %s", exc)
        duration_ms = (time.time() - invocation_start) * 1000.0
        return JSONResponse(
            {
                "status": "error",
                "error": str(exc),
                "duration_ms": round(duration_ms, 2),
                "timestamp": invocation_start,
            },
            status_code=500,
        )


async def ping(request: Request) -> JSONResponse:
    """AgentCore health check — GET /ping must return {"status": "Healthy"}."""
    return JSONResponse({"status": "Healthy"})


# ---------------------------------------------------------------------------
# Starlette app — single server on port 8080 per AgentCore contract
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/invocations", handle_invocation, methods=["POST"]),
        Route("/ping", ping, methods=["GET"]),
    ]
)


# ---------------------------------------------------------------------------
# Main — run on port 8080 (AgentCore HTTP protocol requirement)
# ---------------------------------------------------------------------------


def main():
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info(
        "Bid Shading Strategy Agent starting (table=%s, region=%s)",
        PARAMETER_STORE_TABLE,
        AWS_REGION,
    )
    logger.info("Serving on :8080 (AgentCore HTTP contract: POST /invocations, GET /ping)")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
