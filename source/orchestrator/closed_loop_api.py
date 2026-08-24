"""Closed-loop demo API — orchestrator HTTP handlers.

Exposes endpoints for the Part 2 closed-loop demo:

- ``GET  /v1/closed-loop/scenarios``  — list controllable scenarios
- ``POST /v1/closed-loop/generate``   — emit synthetic input for a scenario to
   CloudWatch and return the "before" context (agentic) or the real A/B decision
   (governance). This endpoint does NOT invoke any agent runtime — the UI invokes
   the Adaptive Bidding agent directly (browser-direct, SigV4). The orchestrator is
   never in the agent-invocation path.
- ``GET  /v1/closed-loop/sample-outcomes`` — a subset of individual synthetic
   sample records for a scenario (bid-outcome records for agentic scenarios,
   control/treatment A/B values for governance scenarios)
- ``GET  /v1/closed-loop/parameters`` — current bidding parameters (real DynamoDB)
- ``GET  /v1/closed-loop/audit``      — audit trail (real DynamoDB)
- ``GET  /v1/closed-loop/models``     — model registry versions/status (real SageMaker)
- ``GET  /v1/closed-loop/metrics``    — recent ARTF/BidOutcome metrics (real CloudWatch)

All read endpoints surface real persisted state. If a backing resource is not
configured/reachable, the handler returns an honest error rather than fabricated
data.

Configuration (environment variables):
- ``PARAMETER_STORE_TABLE`` (default ``parameter-store``)
- ``AUDIT_TRAIL_TABLE``     (default ``audit-trail``)
- ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (default ``us-east-1``)
- ``CLOSED_LOOP_MODEL_GROUP_<MODELTYPE>`` optional per-model group overrides
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

# Ensure the source root is importable (closed_loop_demo, shared, agents).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from closed_loop_demo import generator, invoker, readers  # noqa: E402
from closed_loop_demo.scenarios import (  # noqa: E402
    LOOP_AGENTIC,
    LOOP_GOVERNANCE,
    get_scenario,
    list_scenarios,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
_PARAMETER_STORE_TABLE = os.environ.get("PARAMETER_STORE_TABLE", "parameter-store")
_AUDIT_TRAIL_TABLE = os.environ.get("AUDIT_TRAIL_TABLE", "audit-trail")

# Default SageMaker Model Package Group per model type (matches closed_loop_cfn.yaml).
_DEFAULT_MODEL_GROUPS = {
    "dlrm_bid_shader": "artf-dlrm-bid-shader",
    "ncf_deal_manager": "artf-ncf-deal-manager",
    "widedeep_segment_activator": "artf-widedeep-segment-activator",
    "deal_yield_manager_floor": "artf-deal-yield-manager-floor",
    "deal_yield_manager_margin": "artf-deal-yield-manager-margin",
}


def _model_group(model_type: str) -> str:
    env_key = f"CLOSED_LOOP_MODEL_GROUP_{model_type.upper()}"
    return os.environ.get(env_key, _DEFAULT_MODEL_GROUPS.get(model_type, model_type))


# ---------------------------------------------------------------------------
# Lazy singletons for AWS clients / stores
# ---------------------------------------------------------------------------

_parameter_store: Any = None
_cloudwatch_client: Any = None
_dynamodb_resource: Any = None
_sagemaker_client: Any = None


def _get_parameter_store() -> Any:
    global _parameter_store
    if _parameter_store is None:
        from shared.parameter_store import ParameterStore

        _parameter_store = ParameterStore(
            table_name=_PARAMETER_STORE_TABLE,
            region=_REGION,
            audit_table_name=_AUDIT_TRAIL_TABLE,
        )
    return _parameter_store


def _get_cloudwatch() -> Any:
    global _cloudwatch_client
    if _cloudwatch_client is None:
        import boto3

        _cloudwatch_client = boto3.client("cloudwatch", region_name=_REGION)
    return _cloudwatch_client


def _get_dynamodb() -> Any:
    global _dynamodb_resource
    if _dynamodb_resource is None:
        import boto3

        _dynamodb_resource = boto3.resource("dynamodb", region_name=_REGION)
    return _dynamodb_resource


def _get_sagemaker() -> Any:
    global _sagemaker_client
    if _sagemaker_client is None:
        import boto3

        _sagemaker_client = boto3.client("sagemaker", region_name=_REGION)
    return _sagemaker_client


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def list_scenarios_handler(request: Request) -> JSONResponse:
    """GET — list all controllable scenarios (optionally filtered by ?loop=)."""
    loop = request.query_params.get("loop")
    scenarios = list_scenarios(loop if loop in (LOOP_AGENTIC, LOOP_GOVERNANCE) else None)
    return JSONResponse(
        {
            "scenarios": [s.summary() for s in scenarios],
            "note": (
                "Scenarios generate synthetic market data as INPUT to the real "
                "Part 2 decision code. Decisions and persisted state are real."
            ),
        }
    )


async def generate_handler(request: Request) -> JSONResponse:
    """POST — emit synthetic input for a scenario; return before-context or A/B decision.

    Body: ``{"scenario": "<key>"}``

    For **agentic** scenarios: emits the synthetic market metrics to CloudWatch,
    ensures the bidding parameters exist, and returns the "before" parameter
    snapshot + emitted market_state under ``context``. It does NOT invoke the
    agent — the UI invokes the Adaptive Bidding AgentCore runtime directly
    (browser-direct, SigV4) and then reads the persisted "after" state via the
    parameters endpoint.

    For **governance** scenarios: returns the real A/B statistical decision
    (``ABEvaluator``; pure computation, not an agent invocation) under ``decision``.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    scenario_key = (body or {}).get("scenario")
    if not scenario_key:
        return JSONResponse({"error": "missing 'scenario' in request body"}, status_code=400)

    try:
        scenario = get_scenario(scenario_key)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    response: dict[str, Any] = {
        "scenario": scenario.summary(),
        "generated": [],
        "context": None,   # agentic: "before" snapshot + emitted market_state (no invocation)
        "decision": None,  # governance: real A/B decision (pure ABEvaluator)
        "errors": [],
    }

    # Step 1: emit the synthetic input metrics to real CloudWatch (best-effort).
    # Run in a worker thread: emit_for_scenario polls GetMetricData (blocking
    # time.sleep, up to ~12s) to confirm the just-written datapoint is actually
    # queryable before the UI invokes the agent — that polling must not block
    # this async event loop.
    try:
        cw = _get_cloudwatch()
        evidences = await asyncio.to_thread(generator.emit_for_scenario, cw, scenario)
        response["generated"] = [
            {
                "namespace": e.namespace,
                "ok": e.ok,
                "error": e.error,
                "timestamp": e.timestamp,
                "emitted": e.emitted,
                "visible": e.visible,
                "visible_wait_seconds": e.visible_wait_seconds,
            }
            for e in evidences
        ]
    except Exception as exc:
        logger.warning("Metric emission failed: %s", exc)
        response["errors"].append(f"emit: {type(exc).__name__}: {exc}")
        cw = None

    # Step 2: prepare context / run the pure statistical gate. NO agent invocation.
    ready = False
    try:
        if scenario.loop == LOOP_AGENTIC:
            store = _get_parameter_store()
            # Prepare-only: ensure params + snapshot "before" + echo emitted market_state.
            # The UI invokes the Adaptive Bidding runtime directly (SigV4).
            response["context"] = await invoker.prepare_agentic_context(scenario, store)
            ready = True
        elif scenario.loop == LOOP_GOVERNANCE:
            # Pure statistical A/B gate (not an agent invocation).
            response["decision"] = invoker.run_governance_decision(scenario)
            ready = True
        else:
            response["errors"].append(f"unknown loop '{scenario.loop}'")
    except Exception as exc:
        logger.exception("Prepare failed for scenario %s", scenario_key)
        response["errors"].append(f"prepare: {type(exc).__name__}: {exc}")

    status = 200 if ready else 502
    return JSONResponse(response, status_code=status)


async def sample_outcomes_handler(request: Request) -> JSONResponse:
    """GET — a subset of individual synthetic sample records for a scenario.

    Query params: ``scenario`` (required), ``n`` (optional, default 10).

    For agentic scenarios: illustrative individual bid-outcome records
    (won/price_paid/impression/click/conversion) consistent with the scenario's
    aggregate metrics. For governance scenarios: the first N control/treatment
    values from the deterministic A/B sample lists actually fed to the real
    ABEvaluator. Always deterministic and explicitly labelled as synthetic
    sample input, never a fabricated result.
    """
    scenario_key = request.query_params.get("scenario")
    if not scenario_key:
        return JSONResponse({"error": "missing 'scenario' query parameter"}, status_code=400)

    try:
        n = int(request.query_params.get("n", "10"))
    except ValueError:
        n = 10

    try:
        scenario = get_scenario(scenario_key)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return JSONResponse(
        {
            "scenario": scenario_key,
            "loop": scenario.loop,
            **scenario.sample_outcomes(n=n),
        }
    )


async def parameters_handler(request: Request) -> JSONResponse:
    """GET — current bidding parameters for a model type (real DynamoDB)."""
    model_type = request.query_params.get("model_type", "dlrm_bid_shader")
    try:
        store = _get_parameter_store()
        params = await readers.read_parameters(store, model_type)
        return JSONResponse({"model_type": model_type, "parameters": params})
    except Exception as exc:
        logger.warning("read parameters failed: %s", exc)
        return JSONResponse(
            {"model_type": model_type, "error": f"{type(exc).__name__}: {exc}"},
            status_code=503,
        )


async def audit_handler(request: Request) -> JSONResponse:
    """GET — audit trail records for a model type (real DynamoDB), newest first."""
    model_type = request.query_params.get("model_type", "dlrm_bid_shader")
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        limit = 50
    try:
        ddb = _get_dynamodb()
        records = readers.read_audit_trail(ddb, _AUDIT_TRAIL_TABLE, model_type, limit=limit)
        return JSONResponse({"model_type": model_type, "records": records})
    except Exception as exc:
        logger.warning("read audit failed: %s", exc)
        return JSONResponse(
            {"model_type": model_type, "error": f"{type(exc).__name__}: {exc}"},
            status_code=503,
        )


async def models_handler(request: Request) -> JSONResponse:
    """GET — model registry versions and approval status (real SageMaker)."""
    model_type = request.query_params.get("model_type", "dlrm_bid_shader")
    try:
        limit = int(request.query_params.get("limit", "20"))
    except ValueError:
        limit = 20
    group = _model_group(model_type)
    try:
        sm = _get_sagemaker()
        versions = readers.read_model_versions(sm, group, limit=limit)
        return JSONResponse(
            {"model_type": model_type, "model_package_group": group, "versions": versions}
        )
    except Exception as exc:
        logger.warning("read model versions failed: %s", exc)
        return JSONResponse(
            {
                "model_type": model_type,
                "model_package_group": group,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_code=503,
        )


async def metrics_handler(request: Request) -> JSONResponse:
    """GET — recent ARTF/BidOutcome metrics from real CloudWatch for display."""
    try:
        window = int(request.query_params.get("window_minutes", "15"))
    except ValueError:
        window = 15

    metric_specs = [
        ("total_bids", "TotalBids", "Sum"),
        ("wins", "Wins", "Sum"),
        ("avg_price_paid", "AvgPricePaid", "Average"),
        ("avg_shaded_price", "AvgShadedPrice", "Average"),
        ("total_revenue", "TotalRevenue", "Sum"),
        ("total_cost", "TotalCost", "Sum"),
    ]
    queries = [
        {
            "Id": mid,
            "MetricStat": {
                "Metric": {"Namespace": generator.BID_OUTCOME_NAMESPACE, "MetricName": name},
                "Period": 300,
                "Stat": stat,
            },
        }
        for mid, name, stat in metric_specs
    ]

    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(minutes=window)
    try:
        cw = _get_cloudwatch()
        resp = cw.get_metric_data(MetricDataQueries=queries, StartTime=start, EndTime=now)
        values: dict[str, float] = {}
        for result in resp.get("MetricDataResults", []):
            vals = result.get("Values", [])
            values[result.get("Id", "")] = float(vals[0]) if vals else 0.0
        total_bids = values.get("total_bids", 0.0)
        wins = values.get("wins", 0.0)
        total_cost = values.get("total_cost", 0.0)
        total_revenue = values.get("total_revenue", 0.0)
        return JSONResponse(
            {
                "namespace": generator.BID_OUTCOME_NAMESPACE,
                "window_minutes": window,
                "metrics": values,
                "derived": {
                    "win_rate": round(wins / total_bids, 4) if total_bids > 0 else None,
                    "roi": round((total_revenue - total_cost) / total_cost, 4) if total_cost > 0 else None,
                },
            }
        )
    except Exception as exc:
        logger.warning("read metrics failed: %s", exc)
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=503)


# ---------------------------------------------------------------------------
# Schedule toggle — enable/disable scheduled retraining
# ---------------------------------------------------------------------------

_SCHEDULE_PREFIX = os.environ.get("STACK_PREFIX", "artf")
_BID_SHADING_SCHEDULE_NAME = f"{_SCHEDULE_PREFIX}-bid-shading-scheduler"

_scheduler_client: Any = None


def _get_scheduler_client():
    global _scheduler_client
    if _scheduler_client is None:
        import boto3
        _scheduler_client = boto3.client("scheduler", region_name=_REGION)
    return _scheduler_client


async def schedule_status_handler(request: Request) -> JSONResponse:
    """GET — current state of the retraining schedule (ENABLED/DISABLED)."""
    try:
        client = _get_scheduler_client()
        resp = client.get_schedule(Name=_BID_SHADING_SCHEDULE_NAME)
        return JSONResponse({
            "schedule_name": _BID_SHADING_SCHEDULE_NAME,
            "state": resp.get("State", "UNKNOWN"),
            "schedule_expression": resp.get("ScheduleExpression", ""),
        })
    except Exception as exc:
        logger.warning("get schedule status failed: %s", exc)
        return JSONResponse(
            {"schedule_name": _BID_SHADING_SCHEDULE_NAME, "error": f"{type(exc).__name__}: {exc}"},
            status_code=503,
        )


async def schedule_toggle_handler(request: Request) -> JSONResponse:
    """POST — enable or disable the scheduled retraining.

    Body: {"enabled": true} or {"enabled": false}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    enabled = body.get("enabled")
    if enabled is None:
        return JSONResponse({"error": "missing 'enabled' field (true/false)"}, status_code=400)

    new_state = "ENABLED" if enabled else "DISABLED"

    try:
        client = _get_scheduler_client()

        # Get current schedule to preserve all fields (update requires full spec)
        current = client.get_schedule(Name=_BID_SHADING_SCHEDULE_NAME)

        client.update_schedule(
            Name=_BID_SHADING_SCHEDULE_NAME,
            ScheduleExpression=current["ScheduleExpression"],
            FlexibleTimeWindow=current["FlexibleTimeWindow"],
            Target=current["Target"],
            State=new_state,
        )

        return JSONResponse({
            "schedule_name": _BID_SHADING_SCHEDULE_NAME,
            "state": new_state,
            "message": f"Scheduled retraining {'enabled' if enabled else 'disabled'}",
        })
    except Exception as exc:
        logger.warning("toggle schedule failed: %s", exc)
        return JSONResponse(
            {"error": f"{type(exc).__name__}: {exc}"},
            status_code=503,
        )
