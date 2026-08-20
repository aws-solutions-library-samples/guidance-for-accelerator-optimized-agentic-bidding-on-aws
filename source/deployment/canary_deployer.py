"""Canary Deployer: progressive, zero-downtime model rollouts (Option D).

Drives a LIVE canary split via the Triton-side router model: the router (kept
under the name the ARTF container calls) hash/randomly splits live traffic
between ``<model>_stable`` and ``<model>_canary``. This deployer performs only
control-plane actions — stage the canary engine in the S3 repo, set the router's
in-memory split, promote by publishing a new stable version, and roll back — so
the ARTF bidstream never gains an external dependency. Supports progressive
traffic adjustment (5→25→100%), promote, and rollback, ensuring at least one
healthy model is always serving.

Requirements: 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from deployment.model_deployer import ModelOptimizer, TritonModelLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DeploymentState:
    """Represents the current canary deployment state for a (router) model.

    ``model_name`` is the router model the ARTF container calls (e.g.
    ``dlrm_bid_shader``). Traffic is split by the Triton router between
    ``<model>_stable`` and ``<model>_canary`` per the router's in-memory
    ``canary_traffic_pct`` (control-plane), so there is no per-request routing here.
    """

    model_name: str
    current_version: int = 1
    canary_model: Optional[str] = None
    canary_engine_uri: Optional[str] = None
    canary_traffic_pct: float = 0.0  # 0.0-100.0
    status: str = "stable"  # "stable" | "canary_active" | "promoting" | "rolling_back"
    canary_version_arn: Optional[str] = None
    stable_version_arn: Optional[str] = None

    @property
    def control_traffic_pct(self) -> float:
        """Traffic percentage routed to the current (stable) version."""
        return 100.0 - self.canary_traffic_pct


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CanaryAlreadyActiveError(Exception):
    """Raised when attempting to deploy a canary while one is already active."""

    def __init__(self, model_name: str):
        super().__init__(
            f"A canary is already active for model '{model_name}'. "
            "Only one canary per model is allowed."
        )
        self.model_name = model_name


class NoActiveCanaryError(Exception):
    """Raised when attempting to adjust/promote/rollback with no active canary."""

    def __init__(self, model_name: str):
        super().__init__(f"No active canary for model '{model_name}'.")
        self.model_name = model_name


class CanaryLoadError(Exception):
    """Raised when the canary version fails to load or pass health checks."""

    def __init__(self, model_name: str, version: int):
        super().__init__(
            f"Failed to load canary version {version} for model '{model_name}'."
        )
        self.model_name = model_name
        self.version = version


class InvalidTrafficPercentageError(Exception):
    """Raised when traffic percentage is out of valid range."""

    def __init__(self, pct: float):
        super().__init__(
            f"Traffic percentage must be between 0 and 100, got {pct}."
        )
        self.pct = pct


# ---------------------------------------------------------------------------
# Canary Deployer
# ---------------------------------------------------------------------------


class CanaryDeployer:
    """Manages progressive model rollouts on Triton Inference Server.

    Uses deterministic request-hash routing to split traffic between
    the stable (control) and canary (treatment) model versions. Ensures:
    - At most one canary per model_name (Req 5.5)
    - Total traffic always == 100% (Req 5.4)
    - Promote only after confirming canary is serving (Req 5.6)
    - At least one healthy version always loaded (Req 5.7)
    """

    def __init__(
        self,
        triton_loader: TritonModelLoader,
        model_optimizer: ModelOptimizer,
    ):
        self._triton_loader = triton_loader
        self._model_optimizer = model_optimizer
        self._states: dict[str, DeploymentState] = {}

    async def deploy_canary(
        self,
        model_name: str,
        artifact_uri: str,
        initial_traffic_pct: float = 5.0,
        *,
        canary_version_arn: str | None = None,
    ) -> DeploymentState:
        """Deploy a new model version as a canary with initial traffic split.

        Loads the new version alongside the current version via Triton.
        If the load or health check fails, no state change occurs.

        Args:
            model_name: Logical model name (e.g. "dlrm_bid_shader").
            artifact_uri: URI of the optimized model artifact.
            initial_traffic_pct: Initial percentage of traffic for the canary
                (default 5%). Must be in [0, 100].
            canary_version_arn: The SageMaker Model Package version/ARN this
                canary corresponds to. Written to the router's
                canary_version_arn config parameter so per-request
                served_model_version resolution (FR-2) can report it. Omitted
                (None) leaves the router's existing value unchanged — never
                fabricated when the caller doesn't have a real ARN yet.

        Returns:
            Updated DeploymentState with status="canary_active".

        Raises:
            CanaryAlreadyActiveError: If a canary is already active for this model.
            InvalidTrafficPercentageError: If initial_traffic_pct is out of range.
            CanaryLoadError: If the new version fails to load or health-check.
        """
        if initial_traffic_pct < 0 or initial_traffic_pct > 100:
            raise InvalidTrafficPercentageError(initial_traffic_pct)

        state = self._states.get(model_name)

        # Enforce one canary per model_name (Req 5.5)
        if state and state.canary_model is not None:
            raise CanaryAlreadyActiveError(model_name)

        if state is None:
            state = DeploymentState(model_name=model_name, status="stable")

        # Stage the canary MODEL in the S3 repo (Triton poll-loads it). artifact_uri
        # is the optimized engine produced by the Model Optimizer.
        canary_model = await self._triton_loader.stage_canary(model_name, artifact_uri)

        # Confirm Triton actually loaded and is serving the canary before routing
        # any live traffic to it (Req 5.6/5.7 — never route to an unhealthy model).
        if not await self._triton_loader.wait_until_ready(canary_model):
            await self._triton_loader.remove_canary(model_name)
            raise CanaryLoadError(model_name, 1)

        # Route the initial slice of live traffic via the router's in-memory split
        # (control-plane config change; no per-request external dependency).
        await self._triton_loader.set_router_split(
            model_name,
            initial_traffic_pct,
            canary_model=canary_model,
            canary_version_arn=canary_version_arn,
        )

        state.canary_model = canary_model
        state.canary_engine_uri = artifact_uri
        state.canary_traffic_pct = initial_traffic_pct
        state.canary_version_arn = canary_version_arn
        state.status = "canary_active"
        self._states[model_name] = state

        logger.info(
            "Canary deployed for %s: %s live at %.1f%% traffic",
            model_name,
            canary_model,
            initial_traffic_pct,
        )

        return state

    async def adjust_traffic(
        self,
        model_name: str,
        new_pct: float,
    ) -> DeploymentState:
        """Adjust the traffic percentage routed to the canary.

        Validates that the percentage is within [0, 100] and ensures
        control + treatment always equals 100%.

        Args:
            model_name: Logical model name.
            new_pct: New percentage of traffic for the canary [0, 100].

        Returns:
            Updated DeploymentState.

        Raises:
            NoActiveCanaryError: If no canary is active.
            InvalidTrafficPercentageError: If new_pct is out of range.
        """
        if new_pct < 0 or new_pct > 100:
            raise InvalidTrafficPercentageError(new_pct)

        state = self._states.get(model_name)
        if state is None or state.canary_model is None:
            raise NoActiveCanaryError(model_name)

        # Change the router's in-memory split (control-plane); Triton reloads it.
        await self._triton_loader.set_router_split(model_name, new_pct)
        state.canary_traffic_pct = new_pct

        logger.info(
            "Traffic adjusted for %s: canary %s now at %.1f%% (control at %.1f%%)",
            model_name,
            state.canary_model,
            state.canary_traffic_pct,
            state.control_traffic_pct,
        )

        return state

    async def promote(self, model_name: str, *, new_stable_version_arn: str | None = None) -> DeploymentState:
        """Promote the canary to become the new stable version.

        Steps:
        1. Route 100% traffic to the canary version.
        2. Confirm the canary is healthy (serving at 100%).
        3. Unload the previous stable version.
        4. Update state: canary becomes current, status="stable".

        This ensures Triton always has at least one healthy version loaded
        during the transition (Req 5.7). The old version is only unloaded
        AFTER the canary is confirmed serving at 100% (Req 5.6).

        Args:
            model_name: Logical model name.
            new_stable_version_arn: The SageMaker Model Package version/ARN
                the promoted engine corresponds to. Written to the router's
                stable_version_arn config parameter. Defaults to the
                canary's own version_arn (state.canary_version_arn) when
                omitted, since promoting a canary to stable means the
                canary's version IS the new stable version.

        Returns:
            Updated DeploymentState with status="stable".

        Raises:
            NoActiveCanaryError: If no canary is active.
        """
        state = self._states.get(model_name)
        if state is None or state.canary_model is None:
            raise NoActiveCanaryError(model_name)

        state.status = "promoting"
        resolved_stable_version_arn = new_stable_version_arn or state.canary_version_arn

        # Step 1: Publish the validated canary engine as a NEW stable version.
        # Triton serves the highest version by default, so once poll-loaded the
        # stable model serves the new engine (Req 5.7 — the old version stays
        # loaded until the new one is ready, so there is never a serving gap).
        new_version = await self._triton_loader.promote_engine(
            model_name, state.canary_engine_uri
        )
        stable_model = self._triton_loader.stable_name(model_name)
        if not await self._triton_loader.wait_until_ready(stable_model):
            # New stable version did not become ready — keep the canary split as-is
            # and surface the failure rather than cutting over to a broken version.
            state.status = "canary_active"
            raise CanaryLoadError(stable_model, new_version)

        # Step 2: Route 100% back to stable (which now serves the promoted engine)
        # and remove the canary model. Also update the version-ARN parameters
        # so served_model_version resolution (FR-2) reflects the promotion:
        # the promoted version becomes stable_version_arn; canary_version_arn
        # is cleared since there is no longer an active canary.
        await self._triton_loader.set_router_split(
            model_name,
            0.0,
            stable_version_arn=resolved_stable_version_arn,
            canary_version_arn="",
        )
        await self._triton_loader.remove_canary(model_name)

        state.current_version = new_version
        state.canary_model = None
        state.canary_engine_uri = None
        state.canary_traffic_pct = 0.0
        state.stable_version_arn = resolved_stable_version_arn
        state.canary_version_arn = None
        state.status = "stable"

        logger.info(
            "Promoted %s: stable now serves v%d (canary removed)",
            model_name,
            new_version,
        )

        return state

    async def rollback(self, model_name: str) -> DeploymentState:
        """Rollback: unload canary, restore 100% traffic to stable version.

        Steps:
        1. Route 100% traffic to the current stable version.
        2. Unload the canary version.
        3. Clear canary state, status="stable".

        Args:
            model_name: Logical model name.

        Returns:
            Updated DeploymentState with status="stable".

        Raises:
            NoActiveCanaryError: If no canary is active.
        """
        state = self._states.get(model_name)
        if state is None or state.canary_model is None:
            raise NoActiveCanaryError(model_name)

        state.status = "rolling_back"

        # Step 1: Route 100% back to stable (control-plane split -> 0).
        await self._triton_loader.set_router_split(model_name, 0.0)
        # Step 2: Remove the canary model from the repo (Triton poll-unloads).
        await self._triton_loader.remove_canary(model_name)

        canary_model = state.canary_model
        state.canary_model = None
        state.canary_engine_uri = None
        state.canary_traffic_pct = 0.0
        state.status = "stable"

        logger.info(
            "Rolled back %s: canary %s removed, stable at 100%%",
            model_name,
            canary_model,
        )

        return state

    def get_state(self, model_name: str) -> DeploymentState:
        """Return the current deployment state for a model.

        Args:
            model_name: Logical model name.

        Returns:
            Current DeploymentState, or a default "stable" state if the model
            has not been tracked yet.
        """
        state = self._states.get(model_name)
        if state is None:
            return DeploymentState(model_name=model_name, status="stable")
        return state

    # NOTE: Per-request traffic routing is NOT done here. In Option D the Triton
    # router model (source/triton/router/model.py) performs the stable-vs-canary
    # split in-process on the GPU node, driven by its in-memory canary_traffic_pct.
    # This CanaryDeployer only sets that split (control-plane) via
    # TritonModelLoader.set_router_split — keeping the ARTF bidstream dependency-free.
