"""Unit tests for orchestrator.loadtest -- focused on the deal_yield_manager
load-test targeting wiring added to close the Yield Optimizer's cold-start
gap (the genesis model never emits a mutation on its own, so without this
wiring neither live nor load-test traffic could ever produce real outcome
data to train on).

Covers:
- deal_yield_manager is a selectable target_model_type and maps to the
  correct CONTAINERS registry entry name.
- ALL_INTENTS includes ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN so
  deal_yield_manager is actually included in the fan-out (a container
  whose intents don't overlap applicable_intents is filtered out by
  _filter_containers -- see orchestrator/app.py).
- _run_load_test's target_model_type="deal_yield_manager" branch calls
  emit_load_test_deal_yield_outcome() (not the bid-shading
  emit_load_test_outcome()) and aggregates its returned samples into
  outcome_sample_count/outcome_samples correctly, including the 0/1/2
  samples-per-request case BR-5 requires.
- LoadTestRequest accepts "deal_yield_manager" as a valid target_model_type.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import loadtest
from orchestrator.loadtest import ALL_INTENTS, LoadTestRequest, _MODEL_TYPE_TO_CONTAINER_NAME, _TARGET_MODEL_TYPES
from shared.artf_types import ContainerInvocationModel


class TestTargetModelTypeRegistry:
    def test_deal_yield_manager_is_selectable(self):
        assert "deal_yield_manager" in _TARGET_MODEL_TYPES

    def test_deal_yield_manager_maps_to_correct_container_name(self):
        assert _MODEL_TYPE_TO_CONTAINER_NAME["deal_yield_manager"] == "deal-yield-manager"

    def test_load_test_request_accepts_deal_yield_manager(self):
        req = LoadTestRequest(preset="100", target_model_type="deal_yield_manager")
        assert req.target_model_type == "deal_yield_manager"

    def test_load_test_request_rejects_unknown_model_type(self):
        with pytest.raises(Exception):
            LoadTestRequest(preset="100", target_model_type="not_a_real_model")


class TestAllIntentsIncludesDealYield:
    def test_adjust_deal_floor_included(self):
        assert "ADJUST_DEAL_FLOOR" in ALL_INTENTS

    def test_adjust_deal_margin_included(self):
        assert "ADJUST_DEAL_MARGIN" in ALL_INTENTS

    def test_deal_yield_manager_container_would_be_included_in_fanout(self):
        """Regression guard: deal_yield_manager's own intents (from
        orchestrator/app.py's CONTAINERS registry) must overlap ALL_INTENTS,
        or _filter_containers(ALL_INTENTS) silently drops it from every
        load test's fan-out regardless of target_model_type targeting."""
        deal_yield_intents = {"ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"}
        assert deal_yield_intents & set(ALL_INTENTS)


def _make_invocation(name: str, mutations=None, model_version="v1", status="ok") -> ContainerInvocationModel:
    return ContainerInvocationModel(
        name=name, status=status, latency_ms=1.0, mutations=mutations or [], model_version=model_version,
    )


class TestRunLoadTestDealYieldWiring:
    """Exercises _run_load_test's target_model_type="deal_yield_manager"
    branch directly (not through the HTTP route), mocking only the
    external-service boundary (_call_container_timed / FeedbackCollector),
    per this project's testing convention."""

    def _payload(self):
        return {
            "id": "br-0",
            "tmax": 100,
            "applicable_intents": ALL_INTENTS,
            "bid_request": {
                "id": "br-0",
                "imp": [
                    {
                        "id": "imp-0",
                        "bidfloor": 3.0,
                        "pmp": {"deals": [{"id": "deal-a", "bidfloor": 3.0, "at": 1}]},
                    }
                ],
            },
        }

    def test_deal_yield_target_uses_deal_yield_emitter_not_bid_shading_emitter(self, monkeypatch):
        """The deal_yield_manager branch must call
        emit_load_test_deal_yield_outcome(), never the bid-shading
        emit_load_test_outcome() (which would construct the wrong event
        type)."""
        from shared.artf_types import AdjustDealPayload, Intent, Operation, Mutation

        floor_mutation = Mutation(
            intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
            path="/imp/imp-0/deals/deal-a",
            adjust_deal=AdjustDealPayload(bidfloor=2.5),
        )

        fake_containers = [{"name": "deal-yield-manager", "intents": {"ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"}}]

        async def fake_filter_containers(intents):
            return fake_containers

        # _filter_containers is a plain (sync) function in app.py; match that.
        def fake_filter_containers_sync(intents):
            return fake_containers

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("deal-yield-manager", mutations=[floor_mutation], model_version="deal-yield-xgboost-v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, fake_filter_containers_sync),
        )

        mock_deal_yield_emit = MagicMock(return_value=[1.5])
        mock_bid_shade_emit = MagicMock(return_value=0.0)
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", mock_deal_yield_emit)
        monkeypatch.setattr(loadtest, "emit_load_test_outcome", mock_bid_shade_emit)

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-1", "100", seed=1, duration_s=5,
                target_model_type="deal_yield_manager", target_variant="current",
            )
        )

        mock_deal_yield_emit.assert_called()
        mock_bid_shade_emit.assert_not_called()

        status = loadtest._active_tests["test-dy-1"]
        assert status.target_model_type == "deal_yield_manager"
        assert status.outcome_sample_count > 0
        assert status.outcome_samples == [1.5] * status.outcome_sample_count

    def test_zero_mutations_produces_zero_samples(self, monkeypatch):
        """When deal_yield_manager returns no mutations for a request (the
        genesis model's normal constant-output behavior), the run must
        record zero samples for that request rather than fabricating one."""
        fake_containers = [{"name": "deal-yield-manager", "intents": {"ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"}}]

        def fake_filter_containers_sync(intents):
            return fake_containers

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("deal-yield-manager", mutations=[], model_version="deal-yield-xgboost-v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, fake_filter_containers_sync),
        )

        mock_deal_yield_emit = MagicMock(return_value=[])
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", mock_deal_yield_emit)

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-2", "100", seed=1, duration_s=5,
                target_model_type="deal_yield_manager", target_variant="current",
            )
        )

        status = loadtest._active_tests["test-dy-2"]
        assert status.outcome_sample_count == 0
        assert status.outcome_samples == []

    def test_two_mutations_per_request_both_counted(self, monkeypatch):
        """BR-5: a request can carry both ADJUST_DEAL_FLOOR and
        ADJUST_DEAL_MARGIN mutations for the same deal (independent,
        atomic) -- both must be counted, not deduplicated to one."""
        fake_containers = [{"name": "deal-yield-manager", "intents": {"ADJUST_DEAL_FLOOR", "ADJUST_DEAL_MARGIN"}}]

        def fake_filter_containers_sync(intents):
            return fake_containers

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("deal-yield-manager", mutations=[], model_version="v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, fake_filter_containers_sync),
        )

        # Every targeted request "emits" exactly 2 samples (floor + margin).
        mock_deal_yield_emit = MagicMock(return_value=[1.0, 2.0])
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", mock_deal_yield_emit)

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-3", "100", seed=1, duration_s=5,
                target_model_type="deal_yield_manager", target_variant="current",
            )
        )

        status = loadtest._active_tests["test-dy-3"]
        # preset "100" == 100 requests, each contributing 2 samples.
        assert status.outcome_sample_count == 200
