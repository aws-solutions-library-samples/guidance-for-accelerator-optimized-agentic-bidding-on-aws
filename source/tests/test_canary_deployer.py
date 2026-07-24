"""Unit tests for the Option-D Canary Deployer: router-split control-plane rollout.

In Option D the CanaryDeployer performs only control-plane actions against the
Triton-side router model — it never routes per request (the Triton router does
that in-process on the GPU node). These tests drive a fake TritonModelLoader and
assert the deployer's state transitions and the loader calls it makes:

- deploy_canary stages the canary model, waits for readiness, sets the router split
- deploy_canary enforces one canary per model, validates traffic %, and cleans up
  (remove_canary) + raises on a canary that never becomes ready
- adjust_traffic re-sets the router split; control + canary always == 100%
- promote publishes a new stable version, zeroes the split, removes the canary
- promote surfaces a failed new-stable readiness without cutting over
- rollback zeroes the split and removes the canary
- route_request no longer exists (routing lives in the Triton router)

Requirements: 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deployment.canary_deployer import (
    CanaryAlreadyActiveError,
    CanaryDeployer,
    CanaryLoadError,
    DeploymentState,
    InvalidTrafficPercentageError,
    NoActiveCanaryError,
)


# ---------------------------------------------------------------------------
# Fake TritonModelLoader — records control-plane calls, simulates readiness
# ---------------------------------------------------------------------------


class FakeTritonLoader:
    """In-memory stand-in for TritonModelLoader (Option-D control-plane ops)."""

    def __init__(self, *, ready: bool = True):
        self.ready = ready
        self.calls: list[tuple] = []
        self.router_splits: list[tuple[str, float, str | None]] = []
        self.staged: list[str] = []
        self.removed: list[str] = []
        self.promoted: list[tuple[str, str, int]] = []
        self._next_version = 1

    @staticmethod
    def stable_name(base: str) -> str:
        return f"{base}_stable"

    @staticmethod
    def canary_name(base: str) -> str:
        return f"{base}_canary"

    async def stage_canary(self, base_model: str, engine_uri: str) -> str:
        self.calls.append(("stage_canary", base_model, engine_uri))
        name = self.canary_name(base_model)
        self.staged.append(name)
        return name

    async def wait_until_ready(self, model_name: str, timeout: float | None = None) -> bool:
        self.calls.append(("wait_until_ready", model_name))
        return self.ready

    async def set_router_split(
        self, router_model: str, canary_traffic_pct: float, canary_model: str | None = None
    ) -> None:
        self.calls.append(("set_router_split", router_model, canary_traffic_pct, canary_model))
        self.router_splits.append((router_model, canary_traffic_pct, canary_model))

    async def promote_engine(self, base_model: str, engine_uri: str) -> int:
        self.calls.append(("promote_engine", base_model, engine_uri))
        self._next_version += 1
        self.promoted.append((base_model, engine_uri, self._next_version))
        return self._next_version

    async def remove_canary(self, base_model: str) -> None:
        self.calls.append(("remove_canary", base_model))
        self.removed.append(self.canary_name(base_model))


class _DummyOptimizer:
    """CanaryDeployer requires a model_optimizer, but Option-D deploy_canary takes an
    already-optimized engine URI, so the deployer never calls it here."""


MODEL_NAME = "dlrm_bid_shader"
ENGINE_URI = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"


def _make_deployer(loader: FakeTritonLoader) -> CanaryDeployer:
    return CanaryDeployer(triton_loader=loader, model_optimizer=_DummyOptimizer())


def _split_calls(loader: FakeTritonLoader) -> list[tuple[str, float, str | None]]:
    return loader.router_splits


# ---------------------------------------------------------------------------
# deploy_canary
# ---------------------------------------------------------------------------


class TestDeployCanary:
    @pytest.mark.asyncio
    async def test_successful_deploy_sets_state_and_router_split(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)

        state = await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=5.0
        )

        assert state.model_name == MODEL_NAME
        assert state.canary_model == f"{MODEL_NAME}_canary"
        assert state.canary_engine_uri == ENGINE_URI
        assert state.canary_traffic_pct == 5.0
        assert state.control_traffic_pct == 95.0
        assert state.status == "canary_active"

        # The canary was staged, readiness confirmed, then the router split set.
        assert ("stage_canary", MODEL_NAME, ENGINE_URI) in loader.calls
        assert loader.router_splits == [(MODEL_NAME, 5.0, f"{MODEL_NAME}_canary")]

    @pytest.mark.asyncio
    async def test_raises_when_canary_already_active(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)

        await deployer.deploy_canary(model_name=MODEL_NAME, artifact_uri=ENGINE_URI)

        with pytest.raises(CanaryAlreadyActiveError) as exc_info:
            await deployer.deploy_canary(model_name=MODEL_NAME, artifact_uri=ENGINE_URI)
        assert exc_info.value.model_name == MODEL_NAME

    @pytest.mark.asyncio
    async def test_raises_and_cleans_up_when_canary_never_ready(self):
        loader = FakeTritonLoader(ready=False)
        deployer = _make_deployer(loader)

        with pytest.raises(CanaryLoadError) as exc_info:
            await deployer.deploy_canary(model_name=MODEL_NAME, artifact_uri=ENGINE_URI)
        assert exc_info.value.model_name == MODEL_NAME

        # Never routed traffic to an unhealthy canary, and cleaned it up.
        assert loader.router_splits == []
        assert ("remove_canary", MODEL_NAME) in loader.calls

        # State remains stable (no canary).
        state = deployer.get_state(MODEL_NAME)
        assert state.canary_model is None
        assert state.status == "stable"

    @pytest.mark.asyncio
    async def test_invalid_traffic_percentage_rejected(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)

        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.deploy_canary(
                model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=-5.0
            )
        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.deploy_canary(
                model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=101.0
            )
        # Nothing was staged for an invalid request.
        assert loader.staged == []


# ---------------------------------------------------------------------------
# adjust_traffic
# ---------------------------------------------------------------------------


class TestAdjustTraffic:
    @pytest.mark.asyncio
    async def test_adjust_updates_state_and_split(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=5.0
        )

        state = await deployer.adjust_traffic(MODEL_NAME, 25.0)

        assert state.canary_traffic_pct == 25.0
        assert state.control_traffic_pct == 75.0
        # Latest router split reflects the new percentage (canary_model unchanged here).
        assert loader.router_splits[-1] == (MODEL_NAME, 25.0, None)

    @pytest.mark.asyncio
    async def test_total_traffic_always_100(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=5.0
        )

        for pct in [5.0, 25.0, 50.0, 75.0, 100.0, 0.0]:
            state = await deployer.adjust_traffic(MODEL_NAME, pct)
            assert state.canary_traffic_pct + state.control_traffic_pct == 100.0

    @pytest.mark.asyncio
    async def test_raises_with_no_canary(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        with pytest.raises(NoActiveCanaryError):
            await deployer.adjust_traffic(MODEL_NAME, 25.0)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_percentage(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        await deployer.deploy_canary(model_name=MODEL_NAME, artifact_uri=ENGINE_URI)

        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.adjust_traffic(MODEL_NAME, -1.0)
        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.adjust_traffic(MODEL_NAME, 100.1)


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


class TestPromote:
    @pytest.mark.asyncio
    async def test_promote_publishes_stable_zeroes_split_removes_canary(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=5.0
        )

        state = await deployer.promote(MODEL_NAME)

        assert state.status == "stable"
        assert state.current_version == 2  # promote_engine returned next version
        assert state.canary_model is None
        assert state.canary_traffic_pct == 0.0
        assert state.control_traffic_pct == 100.0

        # Promoted the validated engine as a new stable version, then cut traffic
        # back to 0% canary and removed the canary model.
        assert ("promote_engine", MODEL_NAME, ENGINE_URI) in loader.calls
        assert loader.router_splits[-1] == (MODEL_NAME, 0.0, None)
        assert ("remove_canary", MODEL_NAME) in loader.calls

    @pytest.mark.asyncio
    async def test_promote_surfaces_failed_new_stable_without_cutover(self):
        # Deploy while ready, then flip readiness off so the promoted stable never
        # becomes ready — promote must NOT zero the split or remove the canary.
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=5.0
        )
        splits_before = len(loader.router_splits)
        loader.ready = False

        with pytest.raises(CanaryLoadError):
            await deployer.promote(MODEL_NAME)

        # No new split beyond what deploy already set, and canary still present.
        assert len(loader.router_splits) == splits_before
        assert ("remove_canary", MODEL_NAME) not in loader.calls
        state = deployer.get_state(MODEL_NAME)
        assert state.canary_model == f"{MODEL_NAME}_canary"
        assert state.status == "canary_active"

    @pytest.mark.asyncio
    async def test_promote_raises_with_no_canary(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        with pytest.raises(NoActiveCanaryError):
            await deployer.promote(MODEL_NAME)


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_zeroes_split_and_removes_canary(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ENGINE_URI, initial_traffic_pct=25.0
        )

        state = await deployer.rollback(MODEL_NAME)

        assert state.status == "stable"
        assert state.current_version == 1  # unchanged — rollback does not promote
        assert state.canary_model is None
        assert state.canary_traffic_pct == 0.0
        assert state.control_traffic_pct == 100.0

        assert loader.router_splits[-1] == (MODEL_NAME, 0.0, None)
        assert ("remove_canary", MODEL_NAME) in loader.calls
        # promote_engine must never be called on a rollback.
        assert all(c[0] != "promote_engine" for c in loader.calls)

    @pytest.mark.asyncio
    async def test_rollback_raises_with_no_canary(self):
        loader = FakeTritonLoader(ready=True)
        deployer = _make_deployer(loader)
        with pytest.raises(NoActiveCanaryError):
            await deployer.rollback(MODEL_NAME)


# ---------------------------------------------------------------------------
# Option-D no longer does per-request routing in the deployer
# ---------------------------------------------------------------------------


class TestNoPerRequestRouting:
    def test_route_request_removed(self):
        """Routing moved into the Triton router model; the deployer must not expose it."""
        assert not hasattr(CanaryDeployer, "route_request")

    def test_deployment_state_uses_canary_model_not_version(self):
        state = DeploymentState(model_name=MODEL_NAME)
        assert hasattr(state, "canary_model")
        assert not hasattr(state, "canary_version")
        assert state.control_traffic_pct == 100.0
