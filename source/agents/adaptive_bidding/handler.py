"""AgentCore HTTP entrypoint for the Adaptive Bidding Strategy Agent.

This is the handler that runs inside AgentCore's Firecracker microVM.

AgentCore HTTP protocol contract (from official docs):
- POST :8080/invocations  - invocation handler (JSON payload)
- GET  :8080/ping         - health check returning {"status": "Healthy"}

The agent is a Strands + Amazon Bedrock reasoning agent: each invocation it reads the
real market state from CloudWatch, reasons about how shade_factor / conversion_value
should move, and writes the adjustments to the DynamoDB Parameter Store (which enforces
bounds and versioning). If the Bedrock model is unavailable, the invocation returns an
honest error and makes no change — no formula fallback, no fabricated adjustment.

Requirements: 7.1, 7.6, 11.1 (parent spec); Req 2.2, 2.7, 2.9 (this spec).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

# Ensure shared/ and agents/ are importable from the AgentCore container context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.adaptive_bidding.agent import AdaptiveBiddingStrategyAgent
from shared.parameter_store import ParameterStore

logger = logging.getLogger("agentcore.adaptive_bidding")

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

PARAMETER_STORE_TABLE = os.environ.get("PARAMETER_STORE_TABLE", "parameter-store")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
ADAPTIVE_BIDDING_MODEL_ID = os.environ.get("ADAPTIVE_BIDDING_MODEL_ID", "")


# ---------------------------------------------------------------------------
# HTTP handlers — AgentCore contract
# ---------------------------------------------------------------------------


async def handle_invocation(request: Request) -> JSONResponse:
    """AgentCore HTTP invocation handler at POST /invocations.

    Runs one reasoning cycle (read metrics -> reason -> write parameter updates) and
    returns the applied updates plus the model's rationale.
    """
    invocation_start = time.time()

    try:
        try:
            await request.json()
        except Exception:
            pass  # payload is not required for the scheduled cycle

        if not ADAPTIVE_BIDDING_MODEL_ID:
            # Honest configuration error — do not silently fall back to a formula.
            return JSONResponse(
                {
                    "status": "error",
                    "error": (
                        "ADAPTIVE_BIDDING_MODEL_ID is not configured; the reasoning agent "
                        "cannot run. Set the Bedrock model id on the runtime."
                    ),
                    "timestamp": invocation_start,
                },
                status_code=500,
            )

        parameter_store = ParameterStore(
            table_name=PARAMETER_STORE_TABLE,
            region=AWS_REGION,
        )
        cloudwatch_client = boto3.client("cloudwatch", region_name=AWS_REGION)

        agent = AdaptiveBiddingStrategyAgent(
            parameter_store=parameter_store,
            cloudwatch_client=cloudwatch_client,
            model_id=ADAPTIVE_BIDDING_MODEL_ID,
            region=AWS_REGION,
        )

        # The reasoning cycle is synchronous (Strands agent loop + tool calls that
        # bridge to the async store). Run it in a worker thread so it doesn't block
        # the event loop and so the tool bridge has no active loop to conflict with.
        result = await asyncio.to_thread(agent.run_cycle)

        # Emit real CloudWatch events for any applied changes.
        agent.emit_update_event()

        duration_ms = (time.time() - invocation_start) * 1000.0
        return JSONResponse(
            {
                "status": "completed",
                "updates": result["updates"],
                "updates_count": result["updates_count"],
                "rationale": result["rationale"],
                "market_state": result["market_state"],
                "model_id": ADAPTIVE_BIDDING_MODEL_ID,
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
        "Adaptive Bidding Strategy Agent starting (table=%s, region=%s, model=%s)",
        PARAMETER_STORE_TABLE,
        AWS_REGION,
        ADAPTIVE_BIDDING_MODEL_ID or "<unset>",
    )
    logger.info("Serving on :8080 (AgentCore HTTP contract: POST /invocations, GET /ping)")

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
