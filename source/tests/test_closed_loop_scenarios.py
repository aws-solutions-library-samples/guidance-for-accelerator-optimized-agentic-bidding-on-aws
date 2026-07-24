"""Tests for the closed-loop demo scenario helpers (orchestrator data-plane).

ARCHITECTURE: the real-time bidding orchestrator MUST NOT invoke the closed-loop
agent runtimes. The browser invokes the Adaptive Bidding agent directly (SigV4).
So these tests verify the orchestrator-side helpers only:

- Agentic scenarios go through ``closed_loop_demo.invoker.prepare_agentic_context``,
  which ensures the parameters exist, snapshots the current ("before") state, and
  echoes the synthetic ``market_state`` emitted to CloudWatch. It performs **no**
  agent invocation and returns **no** decision — the real decision + rationale come
  from the direct browser -> AgentCore call. Only the DynamoDB boundary is replaced
  by an in-memory fake mirroring the real ParameterStore semantics.
- Governance scenarios go through the real ``ABEvaluator`` (Welch's t-test + SPRT)
  via ``run_governance_decision`` — a pure statistical computation, not an agent
  invocation.

No fabricated results: assertions check the real snapshot / computed decision. The
adaptive agent's own reasoning decision is intentionally NOT asserted here because
it is produced by a Bedrock reasoning agent invoked directly by the browser (there
is no deterministic formula to assert against, and the orchestrator never invokes it).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from closed_loop_demo import invoker
from closed_loop_demo.scenarios import (
    LOOP_AGENTIC,
    LOOP_GOVERNANCE,
    get_scenario,
    list_scenarios,
)
from shared.parameter_store import (
    PARAMETER_BOUNDS,
    OptimisticLockError,
    ParameterBoundsError,
    ParameterState,
)


# ---------------------------------------------------------------------------
# In-memory ParameterStore double (mirrors real semantics at the AWS boundary)
# ---------------------------------------------------------------------------


class FakeParameterStore:
    """Async in-memory stand-in mirroring shared.parameter_store.ParameterStore.

    Enforces the same bounds, per-write max-delta, and optimistic version
    checks the real store enforces — only the DynamoDB calls are replaced.
    """

    def __init__(self):
        self._items: dict[tuple[str, str], ParameterState] = {}
        self.audit: list[dict] = []

    async def initialize_parameter(
        self, model_type, parameter_name, initial_value, max_delta_per_update=0.05
    ):
        key = (model_type, parameter_name)
        if key in self._items:
            raise OptimisticLockError(model_type, parameter_name, -1)
        bounds = PARAMETER_BOUNDS[parameter_name]
        state = ParameterState(
            model_type=model_type,
            parameter_name=parameter_name,
            current_value=initial_value,
            previous_value=initial_value,
            updated_at=0.0,
            updated_by="system",
            version=0,
            min_value=bounds.min_value,
            max_value=bounds.max_value,
            max_delta_per_update=max_delta_per_update,
            reason="init",
            confidence=1.0,
        )
        self._items[key] = state
        return state

    async def read_parameter(self, model_type, parameter_name):
        return self._items[(model_type, parameter_name)]

    async def read_all_parameters(self, model_type):
        return {
            name: state
            for (mt, name), state in self._items.items()
            if mt == model_type
        }

    async def update_parameter(
        self, model_type, parameter_name, new_value, updated_by, reason,
        confidence, expected_version,
    ):
        key = (model_type, parameter_name)
        current = self._items[key]
        if current.version != expected_version:
            raise OptimisticLockError(model_type, parameter_name, expected_version)
        # Bounds + max-delta enforcement (mirrors real store)
        if new_value < current.min_value or new_value > current.max_value:
            raise ParameterBoundsError(
                "out of bounds", parameter_name, new_value, PARAMETER_BOUNDS[parameter_name]
            )
        if abs(new_value - current.current_value) > current.max_delta_per_update + 1e-9:
            raise ParameterBoundsError(
                "delta too large", parameter_name, new_value, PARAMETER_BOUNDS[parameter_name]
            )
        updated = ParameterState(
            model_type=model_type,
            parameter_name=parameter_name,
            current_value=new_value,
            previous_value=current.current_value,
            updated_at=current.updated_at + 1,
            updated_by=updated_by,
            version=current.version + 1,
            min_value=current.min_value,
            max_value=current.max_value,
            max_delta_per_update=current.max_delta_per_update,
            reason=reason,
            confidence=confidence,
        )
        self._items[key] = updated
        self.audit.append(
            {"parameter_name": parameter_name, "old": current.current_value, "new": new_value}
        )
        return updated


# ---------------------------------------------------------------------------
# Scenario registry sanity
# ---------------------------------------------------------------------------


def test_registry_has_expected_scenarios():
    keys = {s.key for s in list_scenarios()}
    assert {"underbidding", "overpaying", "healthy", "insufficient_data"} <= keys
    assert {"challenger_wins", "challenger_loses", "inconclusive"} <= keys
    assert {"challenger_wins_ncf", "challenger_loses_ncf", "inconclusive_ncf"} <= keys


def test_agentic_and_governance_loops_partitioned():
    agentic = {s.key for s in list_scenarios(LOOP_AGENTIC)}
    governance = {s.key for s in list_scenarios(LOOP_GOVERNANCE)}
    assert agentic == {"underbidding", "overpaying", "healthy", "insufficient_data"}
    assert governance == {
        "challenger_wins", "challenger_loses", "inconclusive",
        "challenger_wins_ncf", "challenger_loses_ncf", "inconclusive_ncf",
    }
    assert agentic.isdisjoint(governance)


def test_governance_scenarios_cover_both_canary_models():
    """Each governance outcome (promote/reject/extend) exists for both
    GPU-accelerated, canary-routed models — DLRM and NCF — not just DLRM.
    Selecting a model in the UI must actually change which real scenario runs.
    """
    governance = [s for s in list_scenarios(LOOP_GOVERNANCE)]
    model_types = {s.model_type for s in governance}
    assert model_types == {"dlrm_bid_shader", "ncf_deal_manager"}
    ncf_scenarios = {s.key for s in governance if s.model_type == "ncf_deal_manager"}
    assert ncf_scenarios == {"challenger_wins_ncf", "challenger_loses_ncf", "inconclusive_ncf"}


def test_bid_metrics_derived_values():
    under = get_scenario("underbidding").bid_metrics
    assert under.win_rate == pytest.approx(0.20)
    assert under.roi == pytest.approx(0.50)
    over = get_scenario("overpaying").bid_metrics
    assert over.win_rate == pytest.approx(0.55)
    assert over.roi == pytest.approx(-0.30)
    healthy = get_scenario("healthy").bid_metrics
    assert healthy.win_rate == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Agentic loop — orchestrator prepares context only (NO invocation, NO decision)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_agentic_context_snapshots_before_without_invoking():
    store = FakeParameterStore()
    ctx = await invoker.prepare_agentic_context(get_scenario("underbidding"), store)

    # Loop + emitted synthetic market input echoed for the "before" side of the UI.
    assert ctx["loop"] == LOOP_AGENTIC
    assert ctx["scenario"] == "underbidding"
    assert ctx["market_state"]["win_rate"] == pytest.approx(0.20)
    assert ctx["market_state"]["roi"] == pytest.approx(0.50)

    # Parameters were ensured and snapshotted at their initial versions.
    before = ctx["before"]
    assert set(before) == {"shade_factor", "conversion_value"}
    assert before["shade_factor"]["version"] == 0
    assert before["conversion_value"]["version"] == 0


@pytest.mark.asyncio
async def test_prepare_agentic_context_does_not_invoke_or_decide():
    """The orchestrator path must never invoke the agent or return a decision.

    Absence of these keys is the contract that keeps the EKS orchestrator out of the
    agent-invocation path (FR-6): the real decision comes from the browser-direct
    SigV4 call, not from here.
    """
    store = FakeParameterStore()
    ctx = await invoker.prepare_agentic_context(get_scenario("overpaying"), store)

    assert ctx["invocation"] == "browser-direct-agentcore"
    for decision_key in ("updates", "rationale", "recommendation", "decision", "after"):
        assert decision_key not in ctx
    # No parameter writes happened (no invocation) — only the idempotent init.
    assert store.audit == []


@pytest.mark.asyncio
async def test_prepare_agentic_context_is_idempotent_and_non_mutating():
    store = FakeParameterStore()
    first = await invoker.prepare_agentic_context(get_scenario("healthy"), store)
    # Re-running must not raise (params already initialized) and must not change state.
    second = await invoker.prepare_agentic_context(get_scenario("healthy"), store)

    assert first["before"] == second["before"]
    assert first["before"]["shade_factor"]["version"] == 0
    assert store.audit == []


@pytest.mark.asyncio
async def test_prepare_agentic_context_handles_thin_data_scenario():
    """The thin-data scenario carries real (small) metrics; the orchestrator still
    only snapshots — it does not apply any min-samples guard (that reasoning now
    lives in the agent, invoked directly by the browser)."""
    store = FakeParameterStore()
    ctx = await invoker.prepare_agentic_context(get_scenario("insufficient_data"), store)

    assert ctx["market_state"]["total_bids"] == 500
    assert "updates" not in ctx  # no decision on the orchestrator side


# ---------------------------------------------------------------------------
# Governance loop — real A/B evaluator (Welch's t-test + SPRT)
# ---------------------------------------------------------------------------


def test_challenger_wins_promotes():
    result = invoker.run_governance_decision(get_scenario("challenger_wins"))
    assert result["recommendation"] == "promote"
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["treatment_metric"] > result["control_metric"]


def test_challenger_loses_rejects():
    result = invoker.run_governance_decision(get_scenario("challenger_loses"))
    assert result["recommendation"] == "reject"
    assert 0.0 <= result["p_value"] <= 1.0


def test_inconclusive_extends():
    result = invoker.run_governance_decision(get_scenario("inconclusive"))
    assert result["recommendation"] == "extend"
    assert result["p_value"] > 0.05


def test_ncf_challenger_wins_promotes():
    result = invoker.run_governance_decision(get_scenario("challenger_wins_ncf"))
    assert result["recommendation"] == "promote"
    assert result["model_type"] == "ncf_deal_manager"
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["treatment_metric"] > result["control_metric"]


def test_ncf_challenger_loses_rejects():
    result = invoker.run_governance_decision(get_scenario("challenger_loses_ncf"))
    assert result["recommendation"] == "reject"
    assert result["model_type"] == "ncf_deal_manager"
    assert 0.0 <= result["p_value"] <= 1.0


def test_ncf_inconclusive_extends():
    result = invoker.run_governance_decision(get_scenario("inconclusive_ncf"))
    assert result["recommendation"] == "extend"
    assert result["model_type"] == "ncf_deal_manager"
    assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# Sample outcomes — individual synthetic records shown in the UI
# ---------------------------------------------------------------------------


def test_agentic_sample_outcomes_are_bid_outcome_records():
    out = get_scenario("underbidding").sample_outcomes(n=10)
    assert out["kind"] == "bid_outcome_records"
    assert out["loop"] == LOOP_AGENTIC
    records = out["records"]
    assert len(records) == 10
    for r in records:
        # Monotonic outcome contract mirrors shared.feedback_models.BidOutcomeEvent
        assert not (r["conversion"] and not r["click"])
        assert not (r["click"] and not r["impression"])
        assert not (r["impression"] and not r["won"])
        assert (r["price_paid"] is not None) == r["won"]
        assert (r["conversion_value"] is not None) == r["conversion"]


def test_agentic_sample_outcomes_win_count_matches_scenario_win_rate():
    scenario = get_scenario("overpaying")  # win_rate = 0.55
    records = scenario.sample_outcomes(n=20)["records"]
    wins = sum(1 for r in records if r["won"])
    assert wins == round(20 * scenario.bid_metrics.win_rate)


def test_agentic_sample_outcomes_deterministic():
    a = get_scenario("healthy").sample_outcomes(n=8)
    b = get_scenario("healthy").sample_outcomes(n=8)
    assert a == b


def test_governance_sample_outcomes_match_materialized_lists():
    scenario = get_scenario("challenger_wins")
    out = scenario.sample_outcomes(n=5)
    assert out["kind"] == "ab_samples"
    assert out["loop"] == LOOP_GOVERNANCE
    control, treatment = scenario.ab_samples.materialize()
    assert out["control"] == [round(v, 4) for v in control[:5]]
    assert out["treatment"] == [round(v, 4) for v in treatment[:5]]


def test_governance_sample_outcomes_deterministic():
    scenario = get_scenario("inconclusive")
    assert scenario.sample_outcomes(n=6) == scenario.sample_outcomes(n=6)
