"""Decision invoker — calls deployed AgentCore runtimes.

Agentic loop: Invokes the Bid Shading Strategy Agent via its Bedrock AgentCore
runtime. The agent runs in a Firecracker microVM, reads CloudWatch metrics and
DynamoDB parameters, computes adjustments, and writes updates — all inside the
runtime. The orchestrator reads the before/after state from DynamoDB to show the
full picture.

Governance loop: Runs the ABEvaluator locally (it's a pure statistical
computation with no infrastructure dependencies — no AgentCore runtime needed).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import boto3

from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig
from closed_loop_demo.scenarios import Scenario, LOOP_AGENTIC, LOOP_GOVERNANCE
from shared.parameter_store import (
    OptimisticLockError,
    ParameterBoundsError,
    ParameterState,
)

logger = logging.getLogger(__name__)

# AgentCore runtime ARN — MUST be set for agentic scenarios to work.
_BID_SHADING_RUNTIME_ARN = os.environ.get("BID_SHADING_RUNTIME_ARN", "")
_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# Lazy client
_agentcore_client: Any = None


def _get_agentcore_client():
    global _agentcore_client
    if _agentcore_client is None:
        _agentcore_client = boto3.client("bedrock-agentcore", region_name=_REGION)
    return _agentcore_client


# Initialization defaults so agent writes satisfy the store's per-write delta
# guard (conversion_value needs a larger max_delta than the 0.05 default).
_INIT_DEFAULTS = {
    "shade_factor": {"initial_value": 0.65, "max_delta_per_update": 0.05},
    "conversion_value": {"initial_value": 10.0, "max_delta_per_update": 2.5},
}


async def ensure_parameters_initialized(parameter_store: Any, model_type: str) -> None:
    """Idempotently ensure the bidding parameters exist with correct bounds.

    Safe to call repeatedly — an existing parameter is left untouched
    (``initialize_parameter`` uses a conditional put and raises
    ``OptimisticLockError`` if it already exists, which we swallow).
    """
    for name, cfg in _INIT_DEFAULTS.items():
        try:
            await parameter_store.initialize_parameter(
                model_type=model_type,
                parameter_name=name,
                initial_value=cfg["initial_value"],
                max_delta_per_update=cfg["max_delta_per_update"],
            )
        except OptimisticLockError:
            pass  # already initialized — leave as-is
        except Exception as exc:
            logger.warning("Could not initialize %s/%s: %s", model_type, name, exc)


def _params_snapshot(params: dict[str, ParameterState]) -> dict[str, dict]:
    return {
        name: {
            "current_value": round(p.current_value, 6),
            "previous_value": round(p.previous_value, 6),
            "version": p.version,
            "updated_by": p.updated_by,
            "reason": p.reason,
            "updated_at": p.updated_at,
        }
        for name, p in params.items()
    }


def _invoke_agentcore(runtime_arn: str, payload: dict) -> dict:
    """Invoke a Bedrock AgentCore HTTP-protocol runtime.

    Uses the bedrock-agentcore data-plane API ``invoke_agent_runtime``.
    The runtime's handler receives the payload as the POST body and returns
    a JSON response.

    Args:
        runtime_arn: Full ARN of the AgentCore runtime.
        payload: JSON-serializable dict to send as the invocation body.

    Returns:
        Parsed JSON response from the agent.
    """
    client = _get_agentcore_client()

    # Generate a unique session ID for this invocation (min 33 chars)
    session_id = f"cl-invoke-{int(time.time() * 1000)}-{os.urandom(8).hex()}"

    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        qualifier="DEFAULT",
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(payload).encode("utf-8"),
    )

    # Read the streaming response body
    status_code = response.get("statusCode", 200)
    body = response["response"].read()
    parsed = json.loads(body) if body else {}

    if status_code >= 400:
        raise RuntimeError(
            f"AgentCore runtime returned HTTP {status_code}: "
            f"{parsed.get('error', body.decode('utf-8', errors='replace'))}"
        )

    return parsed


async def run_agentic_decision(
    scenario: Scenario,
    parameter_store: Any,
    real_cloudwatch_client: Any | None = None,
    config: dict | None = None,
) -> dict:
    """Invoke the Bid Shading Strategy Agent via AgentCore runtime.

    The agent runs remotely in a Firecracker microVM. It reads the CloudWatch
    metrics (which the generator already wrote) and the DynamoDB parameters,
    computes its decision, and writes updates — all inside the runtime.

    This function reads the before/after DynamoDB state to surface the full
    picture in the UI.
    """
    if scenario.bid_metrics is None:
        raise ValueError(f"Scenario '{scenario.key}' has no bid metrics")

    if not _BID_SHADING_RUNTIME_ARN:
        raise RuntimeError(
            "BID_SHADING_RUNTIME_ARN environment variable not set. "
            "Deploy the AgentCore runtime and configure the orchestrator."
        )

    model_type = scenario.model_type

    # Ensure parameters exist before the agent tries to read/write them.
    await ensure_parameters_initialized(parameter_store, model_type)

    # Snapshot before
    before = await parameter_store.read_all_parameters(model_type)

    m = scenario.bid_metrics
    result: dict = {
        "loop": LOOP_AGENTIC,
        "scenario": scenario.key,
        "model_type": model_type,
        "invoked_via": "agentcore",
        "runtime_arn": _BID_SHADING_RUNTIME_ARN,
        "market_state": {
            "total_bids": m.total_bids,
            "wins": m.wins,
            "win_rate": round(m.win_rate, 4),
            "roi": round(m.roi, 4),
        },
        "before": _params_snapshot(before),
        "updates": [],
        "skipped": False,
        "error": None,
    }

    try:
        # Invoke the AgentCore runtime — the agent reads CloudWatch & DynamoDB
        # and writes parameter updates, all within the microVM.
        invoke_start = time.time()
        agent_response = _invoke_agentcore(
            _BID_SHADING_RUNTIME_ARN,
            {
                "scenario": scenario.key,
                "model_type": model_type,
            },
        )
        invoke_duration_ms = (time.time() - invoke_start) * 1000.0
        result["agentcore_duration_ms"] = round(invoke_duration_ms, 1)

        # Parse the agent's response
        if agent_response.get("status") == "error":
            result["error"] = agent_response.get("error", "Unknown agent error")
        else:
            updates = agent_response.get("updates", [])
            result["updates"] = [
                {
                    "parameter_name": u["parameter_name"],
                    "old_value": round(u["old_value"], 6),
                    "new_value": round(u["new_value"], 6),
                    "delta": round(u["new_value"] - u["old_value"], 6),
                    "reason": u.get("reason", ""),
                    "confidence": round(u.get("confidence", 0), 4),
                }
                for u in updates
            ]
            result["skipped"] = len(updates) == 0

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    # Snapshot after — reflects whatever the agent wrote in the runtime
    result["after"] = _params_snapshot(await parameter_store.read_all_parameters(model_type))
    return result


def run_governance_decision(scenario: Scenario) -> dict:
    """Run the A/B evaluator against the scenario's generated samples.

    This is a pure statistical computation (Welch's t-test + SPRT) with no
    infrastructure dependencies — runs locally, no AgentCore runtime needed.
    """
    if scenario.ab_samples is None:
        raise ValueError(f"Scenario '{scenario.key}' has no A/B samples")

    control, treatment = scenario.ab_samples.materialize()

    config = ABTestConfig(
        model_type=scenario.model_type,
        control_version="current",
        treatment_version="challenger",
        traffic_percentage=5.0,
        min_samples=min(100, len(control)),
        max_duration_hours=4.0,
        significance_level=0.05,
        primary_metric=scenario.ab_samples.primary_metric,
        guardrail_metrics=[],
    )
    evaluator = ABEvaluator(config)
    res = evaluator.evaluate(control_data=control, treatment_data=treatment)

    return {
        "loop": LOOP_GOVERNANCE,
        "scenario": scenario.key,
        "model_type": scenario.model_type,
        "recommendation": res.recommendation,
        "status": res.status.value if hasattr(res.status, "value") else str(res.status),
        "control_metric": round(res.control_metric, 6),
        "treatment_metric": round(res.treatment_metric, 6),
        "relative_lift": round(res.relative_lift, 6),
        "p_value": round(res.p_value, 6),
        "samples_control": res.samples_control,
        "samples_treatment": res.samples_treatment,
        "guardrail_violations": res.guardrail_violations,
    }
