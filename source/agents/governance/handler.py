"""AgentCore HTTP entrypoint for the Model Promotion Governance Agent.

Runs inside AgentCore's Firecracker microVM. Three payload shapes on one endpoint:

- **Event mode** (EventBridge ModelPackageGroupChanged, forwarded by the Invocation
  Shim Lambda): runs the deterministic validation pipeline
  (model optimize -> canary deploy -> A/B test -> promote/reject/inconclusive). The
  statistical gate is authoritative. A Bedrock reasoning layer then writes a
  natural-language rationale into the (single, real) audit record.
- **Explain-scenario mode** (`{"mode": "explain_scenario", ...}`): a lightweight,
  side-effect-free mode for the closed-loop demo UI. The caller has already run the
  REAL ``ABEvaluator`` (Welch's t-test + SPRT) against a demo scenario's synthetic A/B
  samples in the orchestrator and passes that real decision + real metrics here. This
  mode calls the SAME Bedrock reasoning function the event pipeline uses
  (``reasoning.generate_decision_rationale``) to produce a genuine natural-language
  explanation of that real decision — a real invocation of the deployed agent, not a
  fabricated response. It does NOT touch TensorRT optimization, Triton canary
  deployment, the SageMaker Model Registry, or the governance audit table: those are
  reserved for genuine model-registration events, and writing demo data into the real
  audit trail would misrepresent it as production history.
- **Conversational mode** (`{"query": "..."}`): answers an operator's question strictly
  from the real DynamoDB audit records. Unknown version -> honest "no record".

AgentCore HTTP contract (all on port 8080):
- POST /invocations  - invocation handler (JSON payload)
- GET  /ping         - {"status": "Healthy"}

The statistics decide; the model explains. Nothing is fabricated: if Bedrock is
unavailable the deterministic decision still stands (rationale_source="unavailable"),
and conversational queries never invent a record.

Requirements: 4.1, 5.2-5.6, 6.1-6.4, 11.1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from typing import Any

# Ensure shared/ and agents/ are importable from the AgentCore container context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.governance.governance_agent import ModelPromotionGovernanceAgent, GovernanceDecision
from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig
from agents.governance import reasoning

# Real deployment components (NOT stubs) — model optimization (TensorRT), Triton
# multi-version loading, progressive canary rollout, and guardrail monitoring.
from deployment.model_deployer import ModelOptimizer, TritonModelLoader
from deployment.canary_deployer import CanaryDeployer
from deployment.guardrail_monitor import GuardrailMonitor
from agents.governance.integrations import (
    HttpxClient,
    LambdaProxyHttpClient,
    CloudWatchStatsAdapter,
    GuardrailCheckAdapter,
)

logger = logging.getLogger("agentcore.governance")

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "artf-governance-audit")
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
OPTIMIZER_ENDPOINT = os.environ.get("OPTIMIZER_ENDPOINT", "")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "artf-model-artifacts")
GOVERNANCE_MODEL_ID = os.environ.get("GOVERNANCE_MODEL_ID", "")
# When set, the runtime runs PUBLIC and reaches the cluster-internal Model Optimizer
# and Triton NLBs by invoking this VPC-attached proxy Lambda instead of calling them
# over HTTP directly (which PUBLIC cannot do). Empty -> direct HTTP (requires VPC mode).
VPC_PROXY_LAMBDA_ARN = os.environ.get("VPC_PROXY_LAMBDA_ARN", "")

# A/B metric convention (real CloudWatch). The canary path emits per-variant samples of
# the primary and guardrail metrics under this namespace with a Variant dimension.
ABTEST_NAMESPACE = os.environ.get("ABTEST_NAMESPACE", "ARTF/ABTest")

AGENT_MAX_LIFETIME_HOURS = float(os.environ.get("AGENT_MAX_LIFETIME_HOURS", "8"))
_MAX_LIFETIME_HOURS_LIMIT = 8.0
if AGENT_MAX_LIFETIME_HOURS > _MAX_LIFETIME_HOURS_LIMIT:
    AGENT_MAX_LIFETIME_HOURS = _MAX_LIFETIME_HOURS_LIMIT


# ---------------------------------------------------------------------------
# Dependency factories (deterministic pipeline collaborators)
# ---------------------------------------------------------------------------


def _build_pipeline_components(cloudwatch_client: Any):
    """Construct the REAL deployment components the governance pipeline drives.

    Returns (model_optimizer, canary_deployer, guardrail_check, http_client).

    - ModelOptimizer: real POST {OPTIMIZER_ENDPOINT}/v1/optimize (TensorRT engine build).
    - TritonModelLoader + CanaryDeployer: real Triton multi-version load and
      progressive hash-routed canary rollout with promote/rollback.
    - GuardrailMonitor (via GuardrailCheckAdapter): real CloudWatch p99/error
      checks that force rollback on breach.

    The optimizer and Triton are EKS cluster-internal services behind internal NLBs.
    Reachability is provided one of two ways:
      - VPC_PROXY_LAMBDA_ARN set (default deploy): the runtime runs PUBLIC and reaches
        them by invoking a VPC-attached proxy Lambda (LambdaProxyHttpClient).
      - VPC_PROXY_LAMBDA_ARN unset: the runtime must itself be in VPC network mode so
        HttpxClient can reach OPTIMIZER_ENDPOINT / TRITON_URL directly.
    """
    # PUBLIC runtime -> VPC proxy Lambda when configured; else direct HTTP (VPC mode).
    if VPC_PROXY_LAMBDA_ARN:
        http_client = LambdaProxyHttpClient(VPC_PROXY_LAMBDA_ARN, region=AWS_REGION)
    else:
        http_client = HttpxClient(region=AWS_REGION)
    model_optimizer = ModelOptimizer(
        optimizer_endpoint=OPTIMIZER_ENDPOINT,
        model_bucket=MODEL_BUCKET,
        region=AWS_REGION,
        http_client=http_client,
    )
    triton_loader = TritonModelLoader(
        triton_url=TRITON_URL,
        model_bucket=MODEL_BUCKET,
        http_client=http_client,
        region=AWS_REGION,
    )
    canary_deployer = CanaryDeployer(triton_loader=triton_loader, model_optimizer=model_optimizer)
    guardrail_monitor = GuardrailMonitor(
        canary_deployer=canary_deployer,
        cloudwatch_client=CloudWatchStatsAdapter(cloudwatch_client),
    )
    guardrail_check = GuardrailCheckAdapter(guardrail_monitor)
    return model_optimizer, canary_deployer, guardrail_check, http_client


def _make_cloudwatch_collector(cloudwatch_client: Any, namespace: str):
    """Return an async collect_metrics(model_name, config) reading REAL CloudWatch data.

    Reads per-variant samples of the primary metric (and guardrail metrics) emitted by
    the canary path under ``namespace`` with a ``Variant`` dimension (control|treatment).
    If no datapoints exist yet, returns empty lists so the deterministic gate honestly
    reports insufficient data / inconclusive — it never fabricates samples.
    """
    from datetime import datetime, timedelta, timezone

    def _query(metric_name: str, variant: str, model_name: str, start, end) -> list[float]:
        resp = cloudwatch_client.get_metric_data(
            MetricDataQueries=[{
                "Id": "m",
                "MetricStat": {
                    "Metric": {
                        "Namespace": namespace,
                        "MetricName": metric_name,
                        "Dimensions": [
                            {"Name": "ModelType", "Value": model_name},
                            {"Name": "Variant", "Value": variant},
                        ],
                    },
                    "Period": 60,
                    "Stat": "Average",
                },
                "ReturnData": True,
            }],
            StartTime=start,
            EndTime=end,
        )
        results = resp.get("MetricDataResults", [])
        return [float(v) for v in results[0].get("Values", [])] if results else []

    async def collect(model_name: str, config: ABTestConfig):
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(hours=max(0.1, config.max_duration_hours))
        control = _query(config.primary_metric, "control", model_name, start, end)
        treatment = _query(config.primary_metric, "treatment", model_name, start, end)
        guardrail_data: dict[str, tuple[list[float], list[float]]] = {}
        for g in config.guardrail_metrics:
            gc = _query(g, "control", model_name, start, end)
            gt = _query(g, "treatment", model_name, start, end)
            if gc or gt:
                guardrail_data[g] = (gc, gt)
        return control, treatment, (guardrail_data or None)

    return collect


def _write_enriched_audit(table: Any, record: dict[str, Any]) -> None:
    """Write a single real audit record to DynamoDB with the required composite key."""
    item = dict(record)
    item.setdefault("audit_record_id", str(uuid.uuid4()))
    item["timestamp_version"] = reasoning.make_timestamp_version(item)
    table.put_item(Item=item)


# ---------------------------------------------------------------------------
# Event mode — deterministic pipeline + reasoning enrichment
# ---------------------------------------------------------------------------


async def _handle_event(payload: dict, invocation_start: float) -> JSONResponse:
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

    sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
    cloudwatch_client = boto3.client("cloudwatch", region_name=AWS_REGION)
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    audit_table = dynamodb.Table(AUDIT_TABLE)

    model_optimizer, canary_deployer, guardrail_check, http_client = _build_pipeline_components(
        cloudwatch_client
    )
    collecting_audit = reasoning.CollectingAuditStore()

    def ab_evaluator_factory(config: ABTestConfig) -> ABEvaluator:
        return ABEvaluator(config=config)

    agent = ModelPromotionGovernanceAgent(
        model_optimizer=model_optimizer,
        canary_deployer=canary_deployer,
        ab_evaluator_factory=ab_evaluator_factory,
        guardrail_monitor=guardrail_check,
        model_registry_client=sagemaker_client,
        audit_store=collecting_audit,
    )

    # Real A/B metrics come from CloudWatch; no data -> honest inconclusive (not faked).
    collect_metrics = _make_cloudwatch_collector(cloudwatch_client, ABTEST_NAMESPACE)

    decision: GovernanceDecision = await agent.on_new_model_version(
        model_type=model_type,
        version_arn=version_arn,
        artifact_uri=artifact_uri,
        collect_metrics=collect_metrics,
    )

    # Reasoning layer: rationale from the REAL decision + metrics (never overrides it).
    rationale, rationale_source = await asyncio.to_thread(
        reasoning.generate_decision_rationale,
        model_id=GOVERNANCE_MODEL_ID,
        region=AWS_REGION,
        decision=decision.decision,
        reason=decision.reason,
        model_type=decision.model_type,
        version_arn=decision.version_arn,
        metrics=decision.metrics,
    ) if GOVERNANCE_MODEL_ID else ("", "unavailable")

    # Enrich the single record the deterministic pipeline produced, then write it once.
    if collecting_audit.records:
        record = collecting_audit.records[-1]
    else:
        record = {
            "timestamp": invocation_start,
            "actor": "model_promotion_governance_agent",
            "model_type": decision.model_type,
            "version_arn": decision.version_arn,
            "decision": decision.decision,
            "reason": decision.reason,
            "metrics": decision.metrics,
        }
    record["rationale"] = rationale
    record["rationale_source"] = rationale_source
    record["signals_considered"] = reasoning.signals_considered(decision.metrics)
    try:
        _write_enriched_audit(audit_table, record)
    except Exception as exc:  # noqa: BLE001 - audit write failure must be surfaced, not hidden
        logger.error("Failed to write enriched audit record: %s", exc)

    try:
        await http_client.aclose()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass

    duration_ms = (time.time() - invocation_start) * 1000.0
    return JSONResponse(
        {
            "status": "completed",
            "decision": decision.decision,
            "reason": decision.reason,
            "rationale": rationale,
            "rationale_source": rationale_source,
            "signals_considered": record["signals_considered"],
            "model_type": decision.model_type,
            "version_arn": decision.version_arn,
            "metrics": decision.metrics,
            "duration_ms": round(duration_ms, 2),
            "max_lifetime_hours": AGENT_MAX_LIFETIME_HOURS,
            "timestamp": invocation_start,
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Explain-scenario mode — real Bedrock rationale over a real ABEvaluator decision,
# no infra side effects (used by the closed-loop demo UI)
# ---------------------------------------------------------------------------

_VALID_DECISIONS = {"promote", "reject", "extend"}


async def _handle_explain_scenario(payload: dict, invocation_start: float) -> JSONResponse:
    """Explain a real A/B decision the caller already computed — no side effects.

    Body: ``{"mode": "explain_scenario", "decision": "promote"|"reject"|"extend",
    "metrics": {...}, "model_type": "...", "scenario": "..."}``.

    ``metrics`` MUST be the real ``ABEvaluator`` result fields (control_metric,
    treatment_metric, relative_lift, p_value, samples_control, samples_treatment,
    guardrail_violations) — typically the same dict the orchestrator's
    ``/v1/closed-loop/generate`` endpoint already returned to the caller, computed
    by the real statistical gate. This mode does not run TensorRT optimization,
    Triton canary deployment, SageMaker registry updates, or write to the
    governance audit table — those are reserved for genuine model-registration
    events. If Bedrock is unavailable, this returns an honest error rather than a
    fabricated rationale (there is no deterministic decision to "stand" on here,
    unlike event mode, since explaining IS this mode's only job).
    """
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in _VALID_DECISIONS:
        return JSONResponse(
            {
                "status": "error",
                "error": f"'decision' must be one of {sorted(_VALID_DECISIONS)}, got {decision!r}",
                "timestamp": invocation_start,
            },
            status_code=400,
        )

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return JSONResponse(
            {
                "status": "error",
                "error": "'metrics' (the real ABEvaluator result) is required and must be a non-empty object",
                "timestamp": invocation_start,
            },
            status_code=400,
        )

    if not GOVERNANCE_MODEL_ID:
        return JSONResponse(
            {
                "status": "error",
                "error": (
                    "GOVERNANCE_MODEL_ID is not configured; the reasoning layer cannot run. "
                    "Set the Bedrock model id on the runtime."
                ),
                "timestamp": invocation_start,
            },
            status_code=500,
        )

    model_type = str(payload.get("model_type") or "unknown")
    scenario_key = payload.get("scenario")
    reason = payload.get("reason") or reasoning.default_gate_reason(decision, metrics)
    version_arn = f"demo-scenario:{scenario_key}" if scenario_key else "demo-scenario:unspecified"

    rationale, rationale_source = await asyncio.to_thread(
        reasoning.generate_decision_rationale,
        model_id=GOVERNANCE_MODEL_ID,
        region=AWS_REGION,
        decision=decision,
        reason=reason,
        model_type=model_type,
        version_arn=version_arn,
        metrics=metrics,
    )

    if rationale_source == "unavailable":
        return JSONResponse(
            {
                "status": "error",
                "error": "The Bedrock reasoning model was unavailable — no rationale was generated.",
                "duration_ms": round((time.time() - invocation_start) * 1000.0, 2),
                "timestamp": invocation_start,
            },
            status_code=503,
        )

    duration_ms = (time.time() - invocation_start) * 1000.0
    return JSONResponse(
        {
            "status": "completed",
            "mode": "explain_scenario",
            "decision": decision,
            "rationale": rationale,
            "rationale_source": rationale_source,
            "signals_considered": reasoning.signals_considered(metrics),
            "model_type": model_type,
            "scenario": scenario_key,
            "duration_ms": round(duration_ms, 2),
            "note": (
                "Real Bedrock reasoning over the real ABEvaluator decision. No TensorRT/"
                "canary/registry/audit-table side effects were performed for this demo call."
            ),
            "timestamp": invocation_start,
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Conversational mode — answer from real audit records only
# ---------------------------------------------------------------------------


async def _handle_query(payload: dict, invocation_start: float) -> JSONResponse:
    query = (payload.get("query") or "").strip()
    if not query:
        return JSONResponse(
            {"status": "error", "error": "Empty query.", "timestamp": invocation_start},
            status_code=400,
        )
    if not GOVERNANCE_MODEL_ID:
        return JSONResponse(
            {
                "status": "error",
                "error": "GOVERNANCE_MODEL_ID is not configured; the review assistant cannot run.",
                "timestamp": invocation_start,
            },
            status_code=500,
        )

    model_type = payload.get("model_type")
    version_arn = payload.get("version_arn")
    limit = int(payload.get("limit", 20))

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    audit_table = dynamodb.Table(AUDIT_TABLE)

    records = await asyncio.to_thread(
        reasoning.read_audit_records,
        audit_table,
        model_type=model_type,
        version_arn=version_arn,
        limit=limit,
    )

    if not records:
        # Honest empty state — never fabricate a decision record.
        return JSONResponse(
            {
                "status": "completed",
                "answer": "No governance decision records were found for the requested criteria.",
                "records_found": 0,
                "timestamp": invocation_start,
            },
            status_code=200,
        )

    answer = await asyncio.to_thread(
        reasoning.answer_governance_query,
        model_id=GOVERNANCE_MODEL_ID,
        region=AWS_REGION,
        query=query,
        records=records,
    )

    return JSONResponse(
        {
            "status": "completed",
            "answer": answer,
            "records_found": len(records),
            "timestamp": invocation_start,
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# HTTP handlers — AgentCore contract
# ---------------------------------------------------------------------------


async def handle_invocation(request: Request) -> JSONResponse:
    """AgentCore HTTP handler at POST /invocations. Dispatches by payload shape."""
    invocation_start = time.time()
    try:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        if isinstance(payload, dict) and payload.get("mode") == "explain_scenario":
            return await _handle_explain_scenario(payload, invocation_start)
        if isinstance(payload, dict) and payload.get("query"):
            return await _handle_query(payload, invocation_start)
        return await _handle_event(payload, invocation_start)

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
# Starlette app — single server on port 8080 per AgentCore HTTP contract
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/invocations", handle_invocation, methods=["POST"]),
        Route("/ping", ping, methods=["GET"]),
    ]
)


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info(
        "Model Promotion Governance Agent starting (region=%s, audit_table=%s, model=%s, max_lifetime_hours=%s)",
        AWS_REGION,
        AUDIT_TABLE,
        GOVERNANCE_MODEL_ID or "<unset>",
        AGENT_MAX_LIFETIME_HOURS,
    )
    logger.info("Serving on :8080 (AgentCore HTTP contract: POST /invocations, GET /ping)")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
