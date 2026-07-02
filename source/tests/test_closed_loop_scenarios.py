"""Tests for the closed-loop demo scenario → decision mapping.

These verify that each controllable scenario drives the *specific* decision it
claims, running the REAL Part 2 decision code:

- Agentic scenarios run through ``closed_loop_demo.invoker.run_agentic_decision``,
  which builds the real ``BidShadingStrategyAgent`` and calls its real
  ``evaluate_and_adjust``. Only the DynamoDB boundary is replaced by an
  in-memory fake that mirrors the real ParameterStore semantics (bounds,
  optimistic version, audit) — no decision logic is stubbed.
- Governance scenarios run through the real ``ABEvaluator`` (Welch's t-test +
  SPRT) via ``run_governance_decision``.

No fabricated results: assertions check the real computed decision.
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
# Agentic loop — real agent decision through the invoker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_underbidding_raises_shade_and_conversion():
    store = FakeParameterStore()
    result = await invoker.run_agentic_decision(get_scenario("underbidding"), store)

    assert result["error"] is None
    assert result["skipped"] is False
    updates = {u["parameter_name"]: u for u in result["updates"]}
    assert "shade_factor" in updates and updates["shade_factor"]["delta"] > 0
    assert "conversion_value" in updates and updates["conversion_value"]["delta"] > 0
    # Persisted state reflects the increase
    assert result["after"]["shade_factor"]["current_value"] > result["before"]["shade_factor"]["current_value"]


@pytest.mark.asyncio
async def test_overpaying_lowers_shade_and_conversion():
    store = FakeParameterStore()
    result = await invoker.run_agentic_decision(get_scenario("overpaying"), store)

    assert result["error"] is None
    assert result["skipped"] is False
    updates = {u["parameter_name"]: u for u in result["updates"]}
    assert updates["shade_factor"]["delta"] < 0
    assert updates["conversion_value"]["delta"] < 0


@pytest.mark.asyncio
async def test_healthy_makes_no_change():
    store = FakeParameterStore()
    result = await invoker.run_agentic_decision(get_scenario("healthy"), store)

    assert result["error"] is None
    assert result["skipped"] is True
    assert result["updates"] == []


@pytest.mark.asyncio
async def test_insufficient_data_skips():
    store = FakeParameterStore()
    result = await invoker.run_agentic_decision(get_scenario("insufficient_data"), store)

    assert result["error"] is None
    assert result["skipped"] is True
    assert result["updates"] == []


@pytest.mark.asyncio
async def test_shade_delta_bounded_to_five_percent():
    store = FakeParameterStore()
    result = await invoker.run_agentic_decision(get_scenario("underbidding"), store)
    for u in result["updates"]:
        if u["parameter_name"] == "shade_factor":
            assert abs(u["delta"]) <= 0.05 + 1e-9


# ---------------------------------------------------------------------------
# Governance loop — real A/B evaluator
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
