"""Unit tests for NIM optimization and multi-version Triton model loading.

Tests cover:
- NIM optimization constructs correct API call
- Optimized artifact URI stays within same account/bucket
- Successful model load + health check returns True
- Failed health check returns False and unloads the version
- get_loaded_versions queries correct Triton endpoint
- Error handling for NIM API and Triton API failures

Requirements: 5.1, 5.2, 5.3, 12.4
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deployment.model_deployer import (
    HttpResponse,
    NIMOptimizationError,
    NIMOptimizer,
    TritonModelLoadError,
    TritonModelLoader,
)


# ---------------------------------------------------------------------------
# Helpers: Mock HTTP client
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
        # Try prefix matching for flexibility
        for key, resp in self._responses.items():
            if url.startswith(key):
                return resp
        # Default: return 404 with empty body
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
# Tests: NIMOptimizer
# ---------------------------------------------------------------------------


class TestNIMOptimizer:
    """Tests for NIMOptimizer."""

    def _make_optimizer(
        self, http_client: MockHttpClient | None = None
    ) -> NIMOptimizer:
        return NIMOptimizer(
            nim_endpoint="http://nim-service:8080",
            model_bucket="artf-model-bucket",
            region="us-east-1",
            http_client=http_client,
        )

    @pytest.mark.asyncio
    async def test_optimize_constructs_correct_api_call(self):
        """NIM optimization sends the correct payload to the NIM API."""
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _ok_json({
                "output_uri": "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"
            }),
        })
        optimizer = self._make_optimizer(client)

        await optimizer.optimize(
            model_artifact_uri="s3://artf-model-bucket/models/dlrm/model.onnx",
            model_name="dlrm_bid_shader",
        )

        # Verify the POST call
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call.method == "POST"
        assert call.url == "http://nim-service:8080/v1/optimize"

        payload = call.json_body
        assert payload["source_model_uri"] == "s3://artf-model-bucket/models/dlrm/model.onnx"
        assert payload["model_name"] == "dlrm_bid_shader"
        assert payload["precision"] == "int8"
        assert payload["max_batch_size"] == 64
        assert payload["max_workspace_size"] == 4_294_967_296
        assert payload["target_runtime"] == "tensorrt"

    @pytest.mark.asyncio
    async def test_optimize_custom_config(self):
        """Custom optimization config overrides defaults."""
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _ok_json({
                "output_uri": "s3://artf-model-bucket/optimized-models/ncf/model.engine"
            }),
        })
        optimizer = self._make_optimizer(client)

        await optimizer.optimize(
            model_artifact_uri="s3://artf-model-bucket/models/ncf/model.onnx",
            model_name="ncf_deal_manager",
            optimization_config={
                "precision": "fp16",
                "max_batch_size": 128,
            },
        )

        call = client.calls[0]
        payload = call.json_body
        assert payload["precision"] == "fp16"
        assert payload["max_batch_size"] == 128
        # max_workspace_size retains default
        assert payload["max_workspace_size"] == 4_294_967_296

    @pytest.mark.asyncio
    async def test_optimized_artifact_stays_in_same_bucket(self):
        """Optimized artifact URI uses the same bucket (Req 12.4)."""
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _ok_json({}),
        })
        optimizer = self._make_optimizer(client)

        result_uri = await optimizer.optimize(
            model_artifact_uri="s3://artf-model-bucket/models/dlrm/v2/model.onnx",
            model_name="dlrm_bid_shader",
        )

        # The optimized artifact must stay within the same bucket
        assert result_uri.startswith("s3://artf-model-bucket/")
        assert "optimized-models/dlrm_bid_shader/" in result_uri
        assert result_uri.endswith(".engine")

    @pytest.mark.asyncio
    async def test_optimized_uri_uses_output_from_api(self):
        """When NIM API returns an output_uri, that URI is used."""
        expected_uri = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/custom.engine"
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _ok_json({
                "output_uri": expected_uri,
            }),
        })
        optimizer = self._make_optimizer(client)

        result_uri = await optimizer.optimize(
            model_artifact_uri="s3://artf-model-bucket/models/dlrm/model.onnx",
            model_name="dlrm_bid_shader",
        )

        assert result_uri == expected_uri

    @pytest.mark.asyncio
    async def test_optimize_raises_on_api_error(self):
        """NIM API error raises NIMOptimizationError."""
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _error(
                500, "Internal Server Error: GPU OOM"
            ),
        })
        optimizer = self._make_optimizer(client)

        with pytest.raises(NIMOptimizationError) as exc_info:
            await optimizer.optimize(
                model_artifact_uri="s3://artf-model-bucket/models/dlrm/model.onnx",
                model_name="dlrm_bid_shader",
            )

        assert exc_info.value.status_code == 500
        assert "GPU OOM" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_optimize_raises_on_400_error(self):
        """NIM API validation error raises NIMOptimizationError."""
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _error(
                400, "Bad Request: unsupported model format"
            ),
        })
        optimizer = self._make_optimizer(client)

        with pytest.raises(NIMOptimizationError) as exc_info:
            await optimizer.optimize(
                model_artifact_uri="s3://artf-model-bucket/bad-model.txt",
                model_name="dlrm_bid_shader",
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_output_uri_constructed_from_source_filename(self):
        """The output URI key is derived from the source artifact filename."""
        client = MockHttpClient({
            "http://nim-service:8080/v1/optimize": _ok_json({}),
        })
        optimizer = self._make_optimizer(client)

        result_uri = await optimizer.optimize(
            model_artifact_uri="s3://artf-model-bucket/training/out/dlrm_v3.onnx",
            model_name="dlrm_bid_shader",
        )

        # Should use the stem of the source filename
        assert "dlrm_v3.engine" in result_uri
        # And be under the optimized-models prefix
        assert "optimized-models/dlrm_bid_shader/dlrm_v3.engine" in result_uri


# ---------------------------------------------------------------------------
# Tests: TritonModelLoader
# ---------------------------------------------------------------------------


class TestTritonModelLoader:
    """Tests for TritonModelLoader."""

    def _make_loader(
        self, http_client: MockHttpClient | None = None, repo_path: str = "/tmp/test-models"
    ) -> TritonModelLoader:
        return TritonModelLoader(
            triton_url="http://triton:8000",
            model_repository_path=repo_path,
            http_client=http_client,
        )

    @pytest.mark.asyncio
    async def test_successful_load_and_health_check(self, tmp_path):
        """Successful load + health check returns True."""
        repo_path = str(tmp_path / "models")

        artifact_url = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"
        client = MockHttpClient({
            # Artifact download
            artifact_url: _ok_bytes(b"fake-engine-plan-data"),
            # Triton load
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/load": _ok_json({}),
            # Health check
            "http://triton:8000/v2/models/dlrm_bid_shader/versions/2/ready": _ok_json({}),
        })
        loader = self._make_loader(client, repo_path)

        result = await loader.load_model_version(
            model_name="dlrm_bid_shader",
            version=2,
            artifact_uri=artifact_url,
        )

        assert result is True

        # Verify artifact was written to the model repository
        artifact_path = os.path.join(repo_path, "dlrm_bid_shader", "2", "model.plan")
        assert os.path.exists(artifact_path)
        with open(artifact_path, "rb") as f:
            assert f.read() == b"fake-engine-plan-data"

    @pytest.mark.asyncio
    async def test_failed_health_check_unloads_version(self, tmp_path):
        """Failed health check returns False and unloads the version."""
        repo_path = str(tmp_path / "models")

        artifact_url = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"
        client = MockHttpClient({
            # Artifact download succeeds
            artifact_url: _ok_bytes(b"fake-engine-plan-data"),
            # Triton load succeeds
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/load": _ok_json({}),
            # Health check FAILS (model not ready)
            "http://triton:8000/v2/models/dlrm_bid_shader/versions/3/ready": _error(503, "Not Ready"),
            # Unload is called due to failed health check
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/unload": _ok_json({}),
        })
        loader = self._make_loader(client, repo_path)

        result = await loader.load_model_version(
            model_name="dlrm_bid_shader",
            version=3,
            artifact_uri=artifact_url,
        )

        assert result is False

        # Verify unload was called
        unload_calls = [
            c for c in client.calls
            if c.method == "POST" and "unload" in c.url
        ]
        assert len(unload_calls) == 1

        # Verify version directory was cleaned up
        version_dir = os.path.join(repo_path, "dlrm_bid_shader", "3")
        assert not os.path.exists(version_dir)

    @pytest.mark.asyncio
    async def test_failed_artifact_download_returns_false(self, tmp_path):
        """If artifact download fails, load returns False without calling Triton."""
        repo_path = str(tmp_path / "models")

        artifact_url = "s3://artf-model-bucket/optimized-models/dlrm/model.engine"
        client = MockHttpClient({
            # Artifact download fails
            artifact_url: _error(404, "Not Found"),
        })
        loader = self._make_loader(client, repo_path)

        result = await loader.load_model_version(
            model_name="dlrm_bid_shader",
            version=4,
            artifact_uri=artifact_url,
        )

        assert result is False

        # Triton load should not have been called
        triton_calls = [
            c for c in client.calls if "repository" in c.url
        ]
        assert len(triton_calls) == 0

    @pytest.mark.asyncio
    async def test_failed_triton_load_returns_false(self, tmp_path):
        """If Triton load API fails, returns False and cleans up."""
        repo_path = str(tmp_path / "models")

        artifact_url = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"
        client = MockHttpClient({
            # Artifact download succeeds
            artifact_url: _ok_bytes(b"engine-data"),
            # Triton load FAILS
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/load": _error(
                500, "Triton internal error"
            ),
        })
        loader = self._make_loader(client, repo_path)

        result = await loader.load_model_version(
            model_name="dlrm_bid_shader",
            version=5,
            artifact_uri=artifact_url,
        )

        assert result is False

        # Version directory should be cleaned up
        version_dir = os.path.join(repo_path, "dlrm_bid_shader", "5")
        assert not os.path.exists(version_dir)

    @pytest.mark.asyncio
    async def test_get_loaded_versions(self):
        """get_loaded_versions queries correct Triton endpoint."""
        client = MockHttpClient({
            "http://triton:8000/v2/models/dlrm_bid_shader": _ok_json({
                "name": "dlrm_bid_shader",
                "versions": ["1", "2", "3"],
            }),
        })
        loader = self._make_loader(client)

        versions = await loader.get_loaded_versions("dlrm_bid_shader")

        assert versions == [1, 2, 3]

        # Verify correct endpoint was called
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call.method == "GET"
        assert call.url == "http://triton:8000/v2/models/dlrm_bid_shader"

    @pytest.mark.asyncio
    async def test_get_loaded_versions_returns_sorted(self):
        """get_loaded_versions returns versions in sorted order."""
        client = MockHttpClient({
            "http://triton:8000/v2/models/ncf_deal_manager": _ok_json({
                "name": "ncf_deal_manager",
                "versions": ["5", "2", "8", "1"],
            }),
        })
        loader = self._make_loader(client)

        versions = await loader.get_loaded_versions("ncf_deal_manager")

        assert versions == [1, 2, 5, 8]

    @pytest.mark.asyncio
    async def test_get_loaded_versions_empty_on_error(self):
        """get_loaded_versions returns empty list on API error."""
        client = MockHttpClient({
            "http://triton:8000/v2/models/unknown_model": _error(404, "Model not found"),
        })
        loader = self._make_loader(client)

        versions = await loader.get_loaded_versions("unknown_model")

        assert versions == []

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_200(self):
        """health_check returns True when Triton reports the version ready."""
        client = MockHttpClient({
            "http://triton:8000/v2/models/dlrm_bid_shader/versions/1/ready": _ok_json({}),
        })
        loader = self._make_loader(client)

        result = await loader.health_check("dlrm_bid_shader", 1)

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_non_200(self):
        """health_check returns False when Triton reports the version not ready."""
        client = MockHttpClient({
            "http://triton:8000/v2/models/dlrm_bid_shader/versions/2/ready": _error(
                503, "Model not ready"
            ),
        })
        loader = self._make_loader(client)

        result = await loader.health_check("dlrm_bid_shader", 2)

        assert result is False

    @pytest.mark.asyncio
    async def test_unload_model_version_calls_correct_endpoint(self, tmp_path):
        """unload_model_version calls the Triton unload API."""
        repo_path = str(tmp_path / "models")
        # Pre-create a version directory to verify cleanup
        version_dir = os.path.join(repo_path, "dlrm_bid_shader", "2")
        os.makedirs(version_dir, exist_ok=True)
        with open(os.path.join(version_dir, "model.plan"), "wb") as f:
            f.write(b"data")

        client = MockHttpClient({
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/unload": _ok_json({}),
        })
        loader = self._make_loader(client, repo_path)

        await loader.unload_model_version("dlrm_bid_shader", 2)

        # Verify unload was called
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call.method == "POST"
        assert call.url == "http://triton:8000/v2/repository/models/dlrm_bid_shader/unload"

        # Verify local artifact cleaned up
        assert not os.path.exists(version_dir)

    @pytest.mark.asyncio
    async def test_no_routing_change_on_health_check_failure(self, tmp_path):
        """On health check failure, no routing/traffic change is made.

        This is verified by ensuring that load_model_version only calls
        download, load, health_check, and unload — never any traffic
        routing endpoint. The traffic routing responsibility belongs
        to the Canary Deployer (Task 6.2).
        """
        repo_path = str(tmp_path / "models")

        artifact_url = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"
        client = MockHttpClient({
            artifact_url: _ok_bytes(b"engine-data"),
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/load": _ok_json({}),
            "http://triton:8000/v2/models/dlrm_bid_shader/versions/7/ready": _error(
                503, "Not Ready"
            ),
            "http://triton:8000/v2/repository/models/dlrm_bid_shader/unload": _ok_json({}),
        })
        loader = self._make_loader(client, repo_path)

        result = await loader.load_model_version(
            model_name="dlrm_bid_shader",
            version=7,
            artifact_uri=artifact_url,
        )

        assert result is False

        # Verify no traffic/routing calls were made — only artifact download,
        # load, health check, and unload
        all_urls = [c.url for c in client.calls]
        for url in all_urls:
            assert "traffic" not in url
            assert "route" not in url
