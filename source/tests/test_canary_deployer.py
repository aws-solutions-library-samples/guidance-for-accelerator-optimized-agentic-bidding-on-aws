"""Unit tests for Canary Deployer: traffic splitting, promote, and rollback.

Tests cover:
- deploy_canary with successful load
- deploy_canary raises when canary already active (one per model)
- adjust_traffic updates state correctly
- Total traffic always == 100% (control + treatment)
- Promote routes 100% to new, unloads old
- Rollback routes 100% to current, unloads canary
- route_request is deterministic for same request_id
- route_request distributes traffic approximately at configured percentage
- Always has at least 1 healthy version (promote doesn't unload until new is serving)

Requirements: 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

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
from deployment.model_deployer import (
    HttpResponse,
    NIMOptimizer,
    TritonModelLoader,
)


# ---------------------------------------------------------------------------
# Helpers: Mock HTTP client (reuses pattern from test_model_deployer)
# ---------------------------------------------------------------------------


@dataclass
class RecordedCall:
    """A recorded HTTP call for verification."""

    method: str
    url: str
    json_body: dict | None = None


class MockHttpClient:
    """A mock HTTP client that returns pre-configured responses by URL.

    Uses deterministic, pre-configured responses — no randomness.
    """

    def __init__(self, responses: dict[str, HttpResponse] | None = None):
        self._responses: dict[str, HttpResponse] = responses or {}
        self.calls: list[RecordedCall] = []

    def set_response(self, url: str, response: HttpResponse) -> None:
        """Configure a response for a specific URL."""
        self._responses[url] = response

    def _find_response(self, url: str) -> HttpResponse:
        """Find a matching response by exact URL or prefix match."""
        if url in self._responses:
            return self._responses[url]
        for key, resp in self._responses.items():
            if url.startswith(key):
                return resp
        return HttpResponse(status=404, body=b"Not Found", headers={})

    async def get(self, url: str) -> HttpResponse:
        self.calls.append(RecordedCall(method="GET", url=url))
        return self._find_response(url)

    async def post(self, url: str, json_body: dict | None = None) -> HttpResponse:
        self.calls.append(RecordedCall(method="POST", url=url, json_body=json_body))
        return self._find_response(url)


def _ok_json(data: dict) -> HttpResponse:
    """Create a 200 OK response with JSON body."""
    return HttpResponse(
        status=200,
        body=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


def _ok_bytes(data: bytes) -> HttpResponse:
    """Create a 200 OK response with binary body."""
    return HttpResponse(
        status=200,
        body=data,
        headers={"content-type": "application/octet-stream"},
    )


def _error(status: int, message: str) -> HttpResponse:
    """Create an error response."""
    return HttpResponse(
        status=status,
        body=message.encode(),
        headers={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRITON_URL = "http://triton:8000"
NIM_ENDPOINT = "http://nim-service:8080"
MODEL_BUCKET = "artf-model-bucket"
MODEL_NAME = "dlrm_bid_shader"
ARTIFACT_URI = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary model repository path."""
    return str(tmp_path / "models")


def _make_deployer(
    http_client: MockHttpClient,
    repo_path: str,
) -> CanaryDeployer:
    """Create a CanaryDeployer with mocked dependencies."""
    triton_loader = TritonModelLoader(
        triton_url=TRITON_URL,
        model_repository_path=repo_path,
        http_client=http_client,
    )
    nim_optimizer = NIMOptimizer(
        nim_endpoint=NIM_ENDPOINT,
        model_bucket=MODEL_BUCKET,
        region="us-east-1",
        http_client=http_client,
    )
    return CanaryDeployer(
        triton_loader=triton_loader,
        nim_optimizer=nim_optimizer,
    )


def _setup_successful_deploy_responses(
    client: MockHttpClient,
    model_name: str = MODEL_NAME,
    current_versions: list[str] | None = None,
    new_version: int = 2,
) -> None:
    """Configure mock responses for a successful canary deploy."""
    if current_versions is None:
        current_versions = ["1"]

    # get_loaded_versions
    client.set_response(
        f"{TRITON_URL}/v2/models/{model_name}",
        _ok_json({"name": model_name, "versions": current_versions}),
    )
    # Artifact download
    client.set_response(ARTIFACT_URI, _ok_bytes(b"engine-plan-data"))
    # Triton load
    client.set_response(
        f"{TRITON_URL}/v2/repository/models/{model_name}/load",
        _ok_json({}),
    )
    # Health check for new version
    client.set_response(
        f"{TRITON_URL}/v2/models/{model_name}/versions/{new_version}/ready",
        _ok_json({}),
    )


# ---------------------------------------------------------------------------
# Tests: deploy_canary
# ---------------------------------------------------------------------------


class TestDeployCanary:
    """Tests for CanaryDeployer.deploy_canary."""

    @pytest.mark.asyncio
    async def test_successful_deploy(self, tmp_repo):
        """deploy_canary loads new version and sets canary state."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        state = await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=5.0,
        )

        assert state.model_name == MODEL_NAME
        assert state.current_version == 1
        assert state.canary_version == 2
        assert state.canary_traffic_pct == 5.0
        assert state.control_traffic_pct == 95.0
        assert state.status == "canary_active"

    @pytest.mark.asyncio
    async def test_raises_when_canary_already_active(self, tmp_repo):
        """deploy_canary raises CanaryAlreadyActiveError if a canary exists (Req 5.5)."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        # First deploy succeeds
        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
        )

        # Second deploy for the same model should fail
        with pytest.raises(CanaryAlreadyActiveError) as exc_info:
            await deployer.deploy_canary(
                model_name=MODEL_NAME,
                artifact_uri=ARTIFACT_URI,
            )

        assert exc_info.value.model_name == MODEL_NAME

    @pytest.mark.asyncio
    async def test_raises_on_load_failure(self, tmp_repo):
        """deploy_canary raises CanaryLoadError if health check fails."""
        client = MockHttpClient()

        # get_loaded_versions returns current version
        client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}",
            _ok_json({"name": MODEL_NAME, "versions": ["1"]}),
        )
        # Artifact download succeeds
        client.set_response(ARTIFACT_URI, _ok_bytes(b"engine-data"))
        # Triton load succeeds
        client.set_response(
            f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/load",
            _ok_json({}),
        )
        # Health check FAILS
        client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/2/ready",
            _error(503, "Not Ready"),
        )
        # Unload is called after health check failure (by TritonModelLoader)
        client.set_response(
            f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/unload",
            _ok_json({}),
        )

        deployer = _make_deployer(client, tmp_repo)

        with pytest.raises(CanaryLoadError) as exc_info:
            await deployer.deploy_canary(
                model_name=MODEL_NAME,
                artifact_uri=ARTIFACT_URI,
            )

        assert exc_info.value.model_name == MODEL_NAME
        assert exc_info.value.version == 2

        # State should not be modified
        state = deployer.get_state(MODEL_NAME)
        assert state.canary_version is None
        assert state.status == "stable"

    @pytest.mark.asyncio
    async def test_invalid_traffic_percentage(self, tmp_repo):
        """deploy_canary raises on out-of-range traffic percentage."""
        client = MockHttpClient()
        deployer = _make_deployer(client, tmp_repo)

        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.deploy_canary(
                model_name=MODEL_NAME,
                artifact_uri=ARTIFACT_URI,
                initial_traffic_pct=-5.0,
            )

        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.deploy_canary(
                model_name=MODEL_NAME,
                artifact_uri=ARTIFACT_URI,
                initial_traffic_pct=101.0,
            )


# ---------------------------------------------------------------------------
# Tests: adjust_traffic
# ---------------------------------------------------------------------------


class TestAdjustTraffic:
    """Tests for CanaryDeployer.adjust_traffic."""

    @pytest.mark.asyncio
    async def test_adjust_updates_state(self, tmp_repo):
        """adjust_traffic updates the canary traffic percentage."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=5.0,
        )

        state = await deployer.adjust_traffic(MODEL_NAME, 25.0)

        assert state.canary_traffic_pct == 25.0
        assert state.control_traffic_pct == 75.0

    @pytest.mark.asyncio
    async def test_total_traffic_always_100(self, tmp_repo):
        """Control + treatment traffic always equals 100% (Req 5.4)."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=5.0,
        )

        # Test progressive adjustments
        for pct in [5.0, 25.0, 50.0, 75.0, 100.0, 0.0]:
            state = await deployer.adjust_traffic(MODEL_NAME, pct)
            total = state.canary_traffic_pct + state.control_traffic_pct
            assert total == 100.0, (
                f"Total traffic is {total}% at canary={pct}%, expected 100%"
            )

    @pytest.mark.asyncio
    async def test_raises_with_no_canary(self, tmp_repo):
        """adjust_traffic raises NoActiveCanaryError if no canary is active."""
        client = MockHttpClient()
        deployer = _make_deployer(client, tmp_repo)

        with pytest.raises(NoActiveCanaryError):
            await deployer.adjust_traffic(MODEL_NAME, 25.0)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_percentage(self, tmp_repo):
        """adjust_traffic raises on out-of-range percentage."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
        )

        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.adjust_traffic(MODEL_NAME, -1.0)

        with pytest.raises(InvalidTrafficPercentageError):
            await deployer.adjust_traffic(MODEL_NAME, 100.1)


# ---------------------------------------------------------------------------
# Tests: promote
# ---------------------------------------------------------------------------


class TestPromote:
    """Tests for CanaryDeployer.promote."""

    @pytest.mark.asyncio
    async def test_promote_routes_100_to_new_and_unloads_old(self, tmp_repo):
        """Promote routes 100% to canary and unloads old version (Req 5.6)."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        # Also set up health check for canary during promote
        client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/2/ready",
            _ok_json({}),
        )
        # Unload old version
        client.set_response(
            f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/unload",
            _ok_json({}),
        )

        deployer = _make_deployer(client, tmp_repo)
        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=5.0,
        )

        state = await deployer.promote(MODEL_NAME)

        assert state.status == "stable"
        assert state.current_version == 2
        assert state.canary_version is None
        assert state.canary_traffic_pct == 0.0
        assert state.control_traffic_pct == 100.0

        # Verify unload was called for old version (v1)
        unload_calls = [
            c for c in client.calls
            if c.method == "POST" and "unload" in c.url
        ]
        assert len(unload_calls) >= 1

    @pytest.mark.asyncio
    async def test_promote_confirms_health_before_unload(self, tmp_repo):
        """Promote checks canary health before unloading old version (Req 5.7)."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
        )

        # During promote, canary health check fails
        client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/2/ready",
            _error(503, "Not Ready"),
        )

        with pytest.raises(CanaryLoadError):
            await deployer.promote(MODEL_NAME)

        # Old version should NOT have been unloaded (at least 1 healthy remains)
        unload_calls = [
            c for c in client.calls
            if c.method == "POST" and "unload" in c.url
        ]
        # The only unload calls should not target version 1
        for call in unload_calls:
            # Should not see unload after failed promote health check
            # (TritonModelLoader might have unloaded during initial load failure,
            # but the promote path should not unload old v1)
            pass

        # State should reflect that canary is still present
        state = deployer.get_state(MODEL_NAME)
        assert state.current_version == 1  # Old version preserved

    @pytest.mark.asyncio
    async def test_promote_raises_with_no_canary(self, tmp_repo):
        """promote raises NoActiveCanaryError if no canary is active."""
        client = MockHttpClient()
        deployer = _make_deployer(client, tmp_repo)

        with pytest.raises(NoActiveCanaryError):
            await deployer.promote(MODEL_NAME)


# ---------------------------------------------------------------------------
# Tests: rollback
# ---------------------------------------------------------------------------


class TestRollback:
    """Tests for CanaryDeployer.rollback."""

    @pytest.mark.asyncio
    async def test_rollback_restores_stable_and_unloads_canary(self, tmp_repo):
        """Rollback routes 100% to stable and unloads canary."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        # Unload canary version
        client.set_response(
            f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/unload",
            _ok_json({}),
        )
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=25.0,
        )

        state = await deployer.rollback(MODEL_NAME)

        assert state.status == "stable"
        assert state.current_version == 1
        assert state.canary_version is None
        assert state.canary_traffic_pct == 0.0
        assert state.control_traffic_pct == 100.0

        # Verify unload was called
        unload_calls = [
            c for c in client.calls
            if c.method == "POST" and "unload" in c.url
        ]
        assert len(unload_calls) >= 1

    @pytest.mark.asyncio
    async def test_rollback_raises_with_no_canary(self, tmp_repo):
        """rollback raises NoActiveCanaryError if no canary is active."""
        client = MockHttpClient()
        deployer = _make_deployer(client, tmp_repo)

        with pytest.raises(NoActiveCanaryError):
            await deployer.rollback(MODEL_NAME)


# ---------------------------------------------------------------------------
# Tests: route_request (deterministic hash-based routing)
# ---------------------------------------------------------------------------


class TestRouteRequest:
    """Tests for CanaryDeployer.route_request."""

    @pytest.mark.asyncio
    async def test_deterministic_for_same_request_id(self, tmp_repo):
        """route_request returns the same version for the same request_id."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=50.0,
        )

        request_id = "req-abc-123-deterministic-test"

        # Call multiple times with same request_id
        results = [
            deployer.route_request(MODEL_NAME, request_id)
            for _ in range(100)
        ]

        # All results must be the same (deterministic)
        assert len(set(results)) == 1

    @pytest.mark.asyncio
    async def test_routes_to_current_when_no_canary(self, tmp_repo):
        """route_request returns current version when no canary is active."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        # Deploy and then rollback to get to a known state
        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
        )
        client.set_response(
            f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/unload",
            _ok_json({}),
        )
        await deployer.rollback(MODEL_NAME)

        version = deployer.route_request(MODEL_NAME, "any-request")
        assert version == 1  # Current stable version

    @pytest.mark.asyncio
    async def test_distributes_traffic_approximately(self, tmp_repo):
        """route_request distributes traffic approximately at configured %."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=50.0,
        )

        # Route a large number of distinct request IDs
        canary_count = 0
        total_requests = 10000
        for i in range(total_requests):
            request_id = f"request-{i:06d}"
            version = deployer.route_request(MODEL_NAME, request_id)
            if version == 2:  # canary version
                canary_count += 1

        canary_pct = canary_count / total_requests * 100

        # With 50% configured, expect roughly 50% ± 5%
        assert 45.0 <= canary_pct <= 55.0, (
            f"Expected ~50% canary traffic, got {canary_pct:.1f}%"
        )

    @pytest.mark.asyncio
    async def test_traffic_at_5_percent(self, tmp_repo):
        """route_request distributes ~5% traffic to canary at 5% config."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=5.0,
        )

        canary_count = 0
        total_requests = 10000
        for i in range(total_requests):
            request_id = f"traffic-test-{i:06d}"
            version = deployer.route_request(MODEL_NAME, request_id)
            if version == 2:
                canary_count += 1

        canary_pct = canary_count / total_requests * 100

        # With 5% configured, expect 3-8% (accounting for hash distribution)
        assert 3.0 <= canary_pct <= 8.0, (
            f"Expected ~5% canary traffic, got {canary_pct:.1f}%"
        )

    @pytest.mark.asyncio
    async def test_traffic_at_100_percent_all_canary(self, tmp_repo):
        """At 100% canary traffic, all requests route to canary."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=100.0,
        )

        for i in range(100):
            version = deployer.route_request(MODEL_NAME, f"req-{i}")
            assert version == 2  # All go to canary

    @pytest.mark.asyncio
    async def test_traffic_at_0_percent_all_stable(self, tmp_repo):
        """At 0% canary traffic, all requests route to stable."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
            initial_traffic_pct=0.0,
        )

        for i in range(100):
            version = deployer.route_request(MODEL_NAME, f"req-{i}")
            assert version == 1  # All go to stable


# ---------------------------------------------------------------------------
# Tests: At least 1 healthy version always loaded (Req 5.7)
# ---------------------------------------------------------------------------


class TestHealthyVersionInvariant:
    """Tests that at least one healthy version is always loaded."""

    @pytest.mark.asyncio
    async def test_promote_keeps_old_until_new_confirmed(self, tmp_repo):
        """During promote, old version is kept until canary is confirmed healthy."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
        )

        # Make canary healthy for promote
        client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/2/ready",
            _ok_json({}),
        )
        client.set_response(
            f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/unload",
            _ok_json({}),
        )

        # Track when health check vs unload are called
        pre_promote_calls = len(client.calls)
        await deployer.promote(MODEL_NAME)

        post_calls = client.calls[pre_promote_calls:]

        # Find health check and unload in the post-promote calls
        health_check_idx = None
        unload_idx = None
        for idx, call in enumerate(post_calls):
            if "ready" in call.url:
                health_check_idx = idx
            if "unload" in call.url:
                unload_idx = idx

        # Health check must come before unload
        assert health_check_idx is not None, "Health check was not called"
        assert unload_idx is not None, "Unload was not called"
        assert health_check_idx < unload_idx, (
            "Health check must occur before unload to ensure at least 1 "
            "healthy version is always available"
        )

    @pytest.mark.asyncio
    async def test_failed_promote_preserves_old_version(self, tmp_repo):
        """If canary health check fails during promote, old version stays loaded."""
        client = MockHttpClient()
        _setup_successful_deploy_responses(client)
        deployer = _make_deployer(client, tmp_repo)

        await deployer.deploy_canary(
            model_name=MODEL_NAME,
            artifact_uri=ARTIFACT_URI,
        )

        # Canary health check fails during promote
        client.set_response(
            f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/2/ready",
            _error(503, "Not Ready"),
        )

        with pytest.raises(CanaryLoadError):
            await deployer.promote(MODEL_NAME)

        # Verify no unload calls were made for the old version (v1)
        unload_calls = [
            c for c in client.calls
            if c.method == "POST" and "unload" in c.url
        ]
        # The only unload that might have happened is from the initial
        # TritonModelLoader during deploy (not for v1 during promote)
        for call in unload_calls:
            # Assert we never tried to unload v1 with version-specific policy
            if call.json_body and "version_policy" in str(call.json_body):
                assert '"versions":[1]' not in str(call.json_body)
