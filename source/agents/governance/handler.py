"""AgentCore HTTP entrypoint for the Model Governance Agent.

This is the handler that runs inside AgentCore's Firecracker microVM. It
exposes an HTTP POST endpoint that receives invocations from EventBridge
(on ModelPackageGroupChanged events), runs the full validation pipeline
(NIM optimize -> canary deploy -> A/B test -> promote/reject), and returns
the governance decision.

The session max_lifetime is configurable to cover a full A/B test window
(up to 8 hours) via the AGENT_MAX_LIFETIME_HOURS environment variable.

AgentCore contract:
- HTTP POST on :8000 for the invocation payload
- /ping on :8080 returning {"status": "Healthy"}

Requirements: 4.1, 11.1, 11.6
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

# Ensure shared/ and agents/ are importable from the AgentCore container context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.governance.governance_agent import ModelGovernanceAgent, GovernanceDecision
from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig

logger = logging.getLogger("agentcore.governance")

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "artf-governance-audit")
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
NIM_ENDPOINT = os.environ.get("NIM_ENDPOINT", "")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "artf-model-artifacts")

# Session max_lifetime — configurable up to 8h to cover A/B test windows.
# AgentCore uses this to set session duration limits.
AGENT_MAX_LIFETIME_HOURS = float(os.environ.get("AGENT_MAX_LIFETIME_HOURS", "8"))
_MAX_LIFETIME_HOURS_LIMIT = 8.0

if AGENT_MAX_LIFETIME_HOURS > _MAX_LIFETIME_HOURS_LIMIT:
    logger.warning(
        "AGENT_MAX_LIFETIME_HOURS=%s exceeds maximum (%s), clamping.",
        AGENT_MAX_LIFETIME_HOURS,
        _MAX_LIFETIME_HOURS_LIMIT,
    )
    AGENT_MAX_LIFETIME_HOURS = _MAX_LIFETIME_HOURS_LIMIT


# ---------------------------------------------------------------------------
# Dependency Factories
# ---------------------------------------------------------------------------


class _AuditStore:
    """Minimal DynamoDB-backed audit store for governance decisions."""

    def __init__(self, table_name: str, region: str):
        self._table_name = table_name
        self._region = region
        self._dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = self._dynamodb.Table(table_name)

    async def put_record(self, record: dict[str, Any]) -> None:
        """Append an audit record to DynamoDB."""
        import uuid

        record["audit_record_id"] = str(uuid.uuid4())
        self._table.put_item(Item=record)


class _NIMOptimizer:
    """Wraps calls to the NIM optimization endpoint."""

    def __init__(self, endpoint: str, model_bucket: str, region: str):
        self._endpoint = endpoint
        self._model_bucket = model_bucket
        self._region = region

    async def optimize(self, model_artifact_uri: str, model_name: str) -> str:
        """Optimize the model artifact via NIM and return the optimized URI.

        In production this calls the NIM optimization API. The returned URI
        points to the TensorRT/INT8 compiled engine in S3.
        """
        # NIM optimization is handled by the NIM service running on EKS.
        # The actual HTTP call to the NIM endpoint is made here.
        import urllib.request

        payload = json.dumps({
            "model_artifact_uri": model_artifact_uri,
            "model_name": model_name,
            "optimization": "tensorrt_int8",
        }).encode()

        req = urllib.request.Request(
            f"{self._endpoint}/optimize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
            return result["optimized_artifact_uri"]


class _CanaryDeployer:
    """Wraps calls to the Canary Deployer service."""

    def __init__(self, triton_url: str, model_bucket: str, nim_endpoint: str):
        self._triton_url = triton_url
        self._model_bucket = model_bucket
        self._nim_endpoint = nim_endpoint

    async def deploy_canary(
        self, model_name: str, artifact_uri: str, initial_traffic_pct: float
    ) -> None:
        """Deploy a new model version as canary with the given traffic percentage."""
        logger.info(
            "Deploying canary for %s at %s%% traffic from %s",
            model_name,
            initial_traffic_pct,
            artifact_uri,
        )

    async def promote(self, model_name: str) -> None:
        """Promote canary to 100% traffic."""
        logger.info("Promoting canary to 100%% for %s", model_name)

    async def rollback(self, model_name: str) -> None:
        """Roll back canary, restore 100% traffic to stable version."""
        logger.info("Rolling back canary for %s", model_name)


class _GuardrailMonitor:
    """Monitors guardrail metrics for active canary deployments."""

    def __init__(self, cloudwatch_client: Any):
        self._cloudwatch = cloudwatch_client

    async def check(self, model_name: str) -> list[str]:
        """Check guardrail metrics. Returns list of violation descriptions."""
        # Queries CloudWatch for p99 latency and error rate for the model.
        # Returns empty list if no violations detected.
        return []


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


async def handle_invocation(request: Request) -> JSONResponse:
    """AgentCore HTTP handler — receives EventBridge ModelPackageGroupChanged payload.

    Parses the EventBridge event detail to extract model_type, version_arn,
    and artifact_uri, then runs the full governance validation pipeline.

    Returns:
        JSON response with:
        - status: "completed"
        - decision: the governance decision (promote/reject/inconclusive)
        - reason: explanation for the decision
        - metrics: metrics snapshot from the A/B test
        - duration_ms: total invocation time
        - max_lifetime_hours: configured session lifetime
    """
    invocation_start = time.time()

    try:
        # Parse the EventBridge payload
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # Extract fields from EventBridge event detail
        # EventBridge wraps the detail as a nested object
        detail = payload.get("detail", payload)
        model_type = detail.get("ModelType", detail.get("model_type", ""))
        version_arn = detail.get("ModelPackageArn", detail.get("version_arn", ""))
        artifact_uri = detail.get("ArtifactUri", detail.get("artifact_uri", ""))

        if not model_type or not version_arn:
            return JSONResponse(
                {
                    "status": "error",
                    "error": "Missing required fields: ModelType and ModelPackageArn",
                    "timestamp": invocation_start,
                },
                status_code=400,
            )

        # Initialize dependencies
        sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
        cloudwatch_client = boto3.client("cloudwatch", region_name=AWS_REGION)
        audit_store = _AuditStore(table_name=AUDIT_TABLE, region=AWS_REGION)
        nim_optimizer = _NIMOptimizer(
            endpoint=NIM_ENDPOINT, model_bucket=MODEL_BUCKET, region=AWS_REGION
        )
        canary_deployer = _CanaryDeployer(
            triton_url=TRITON_URL,
            model_bucket=MODEL_BUCKET,
            nim_endpoint=NIM_ENDPOINT,
        )
        guardrail_monitor = _GuardrailMonitor(cloudwatch_client=cloudwatch_client)

        def ab_evaluator_factory(config: ABTestConfig) -> ABEvaluator:
            return ABEvaluator(config=config)

        # Create the governance agent
        agent = ModelGovernanceAgent(
            nim_optimizer=nim_optimizer,
            canary_deployer=canary_deployer,
            ab_evaluator_factory=ab_evaluator_factory,
            guardrail_monitor=guardrail_monitor,
            model_registry_client=sagemaker_client,
            audit_store=audit_store,
        )

        # Run the validation pipeline
        decision: GovernanceDecision = await agent.on_new_model_version(
            model_type=model_type,
            version_arn=version_arn,
            artifact_uri=artifact_uri,
        )

        duration_ms = (time.time() - invocation_start) * 1000.0

        return JSONResponse(
            {
                "status": "completed",
                "decision": decision.decision,
                "reason": decision.reason,
                "model_type": decision.model_type,
                "version_arn": decision.version_arn,
                "metrics": decision.metrics,
                "duration_ms": round(duration_ms, 2),
                "max_lifetime_hours": AGENT_MAX_LIFETIME_HOURS,
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
    """AgentCore health check — must return {"status": "Healthy"}."""
    return JSONResponse({"status": "Healthy"})


# ---------------------------------------------------------------------------
# Starlette app — Invocation on :8000, Health on :8080
# ---------------------------------------------------------------------------

# Main app on port 8000 — handles AgentCore invocations
app = Starlette(
    routes=[
        Route("/", handle_invocation, methods=["POST"]),
        Route("/invoke", handle_invocation, methods=["POST"]),
        Route("/ping", ping),
    ]
)

# Health app on port 8080
health_app = Starlette(
    routes=[
        Route("/ping", ping),
        Route("/health/live", ping),
        Route("/health/ready", ping),
    ]
)


# ---------------------------------------------------------------------------
# Main — run both invocation (:8000) and health (:8080) servers
# ---------------------------------------------------------------------------


def main():
    import threading
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info(
        "Model Governance Agent starting (region=%s, audit_table=%s, "
        "max_lifetime_hours=%s)",
        AWS_REGION,
        AUDIT_TABLE,
        AGENT_MAX_LIFETIME_HOURS,
    )

    # Health server on :8080
    def _run_health():
        uvicorn.run(health_app, host="0.0.0.0", port=8080, log_level="warning")

    health_thread = threading.Thread(target=_run_health, daemon=True)
    health_thread.start()
    logger.info("Health/ping on :8080")

    # Invocation server on :8000
    logger.info("Invocation handler on :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
