"""Closed-loop demo helpers (orchestrator side) — NO agent invocation.

ARCHITECTURE (corrected): the real-time bidding orchestrator MUST NOT invoke
closed-loop agent runtimes. The UI invokes the Adaptive Bidding agent
**directly** (browser-direct, SigV4 via the Cognito Identity Pool) — the same way
production invokes it (EventBridge). This module therefore provides only
**non-invocation** helpers used by the orchestrator's closed-loop *data-plane*
endpoints:

- ``prepare_agentic_context`` — ensure the bidding parameters exist and snapshot
  the current ("before") parameter state plus the synthetic ``market_state`` that
  was emitted to CloudWatch. Performs **no** agent invocation.
- ``run_governance_decision`` — run the A/B statistical gate (Welch's t-test +
  SPRT) via the real ``ABEvaluator``. This is a pure statistical computation with
  no infrastructure dependency and is **not** an agent invocation.

The Adaptive Bidding agent's actual decision (parameter updates + the model's
``rationale`` + the market state it read from CloudWatch) is produced by the
direct browser -> AgentCore invocation and read back by the UI; the orchestrator
later reads the persisted "after" state from DynamoDB via the parameters endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig
from closed_loop_demo.scenarios import Scenario, LOOP_AGENTIC, LOOP_GOVERNANCE
from shared.parameter_store import OptimisticLockError, ParameterState

logger = logging.getLogger(__name__)


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


async def prepare_agentic_context(
    scenario: Scenario,
    parameter_store: Any,
) -> dict:
    """Prepare the "before" context for an agentic scenario — NO agent invocation.

    Ensures the bidding parameters exist, snapshots the current (before) parameter
    state, and echoes the synthetic ``market_state`` that the generator emitted to
    CloudWatch. The browser invokes the Adaptive Bidding AgentCore runtime directly
    (SigV4) to produce the actual decision + rationale; the UI then reads the
    persisted "after" state via the parameters endpoint.

    Returns a dict the UI uses to render the "before" side of the loop and to label
    that the invocation is performed browser-direct against AgentCore.
    """
    if scenario.bid_metrics is None:
        raise ValueError(f"Scenario '{scenario.key}' has no bid metrics")

    model_type = scenario.model_type

    # Ensure parameters exist before the agent (invoked by the browser) reads/writes them.
    await ensure_parameters_initialized(parameter_store, model_type)

    before = await parameter_store.read_all_parameters(model_type)
    m = scenario.bid_metrics

    return {
        "loop": LOOP_AGENTIC,
        "scenario": scenario.key,
        "model_type": model_type,
        # The UI performs the agent invocation directly against AgentCore (SigV4).
        # The orchestrator never invokes the runtime (architecture rule).
        "invocation": "browser-direct-agentcore",
        # Synthetic input we emitted to CloudWatch for this scenario. The agent
        # reads the real CloudWatch metrics itself; this is shown as the emitted input.
        "market_state": {
            "total_bids": m.total_bids,
            "wins": m.wins,
            "win_rate": round(m.win_rate, 4),
            "roi": round(m.roi, 4),
        },
        "before": _params_snapshot(before),
    }


def run_governance_decision(scenario: Scenario) -> dict:
    """Run the A/B evaluator against the scenario's generated samples.

    This is a pure statistical computation (Welch's t-test + SPRT) with no
    infrastructure dependencies — it runs locally and is **not** an agent
    invocation. It is the same ``ABEvaluator`` the deployed Governance agent uses
    as its decision gate. Governance *reasoning* (the model's rationale) is shown
    separately via a direct browser -> Governance runtime call (see design R-2).
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
