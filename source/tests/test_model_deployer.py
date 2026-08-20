"""Unit tests for the Model Optimizer client and the Option-D Triton model loader.

Tests cover:
- ModelOptimizer constructs the correct /v1/optimize API call (fp16 default,
  per-model batch profiles, same-bucket output URI)
- ModelOptimizer surfaces optimizer-service errors as ModelOptimizationError
- TritonModelLoader (S3 poll-repo) stage_canary / promote_engine / set_router_split /
  remove_canary / list_versions / readiness — the control-plane ops the Option-D
  canary uses (no local model-repository path, no per-request routing)

Requirements: 5.1, 5.2, 5.3, 12.4
"""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deployment.model_deployer import (
    HttpResponse,
    ModelOptimizationError,
    ModelOptimizer,
    TritonModelLoadError,
    TritonModelLoader,
)


# ---------------------------------------------------------------------------
# Helpers: Mock HTTP client (deterministic, no randomness)
# ---------------------------------------------------------------------------


@dataclass
class RecordedCall:
    method: str
    url: str
    json_body: dict | None = None


class MockHttpClient:
    """A mock HTTP client that returns pre-configured responses by URL."""

    def __init__(self, responses: dict[str, HttpResponse] | None = None):
        self._responses: dict[str, HttpResponse] = responses or {}
        self.calls: list[RecordedCall] = []

    def set_response(self, url: str, response: HttpResponse) -> None:
        self._responses[url] = response

    def _find_response(self, url: str) -> HttpResponse:
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
    return HttpResponse(
        status=200,
        body=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


def _error(status: int, message: str) -> HttpResponse:
    return HttpResponse(status=status, body=message.encode(), headers={})


# ---------------------------------------------------------------------------
# Helpers: in-memory fake S3 (the loader accepts an injected s3_client)
# ---------------------------------------------------------------------------


class _FakePaginator:
    def __init__(self, s3: "FakeS3"):
        self._s3 = s3

    def paginate(self, Bucket, Prefix, Delimiter=None):  # noqa: N803 - boto3 kwarg names
        keys = [k for k in self._s3.objects if k.startswith(Prefix)]
        if Delimiter:
            prefixes: set[str] = set()
            for k in keys:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
            yield {"CommonPrefixes": [{"Prefix": p} for p in sorted(prefixes)]}
        else:
            yield {"Contents": [{"Key": k} for k in sorted(keys)]}


class FakeS3:
    """Minimal in-memory S3 stand-in supporting the ops TritonModelLoader uses."""

    class _Exceptions:
        class NoSuchKey(Exception):
            pass

    exceptions = _Exceptions()

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects: dict[str, bytes] = dict(objects or {})
        self.copies: list[tuple[dict, str]] = []
        self.deleted: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise FakeS3.exceptions.NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def copy_object(self, CopySource, Bucket, Key):  # noqa: N803
        src_key = CopySource["Key"]
        self.copies.append((CopySource, Key))
        self.objects[Key] = self.objects.get(src_key, b"engine-bytes")

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise RuntimeError("404 Not Found")
        return {"ContentLength": len(self.objects[Key])}

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
            self.deleted.append(obj["Key"])
        return {"Deleted": [{"Key": o["Key"]} for o in Delete["Objects"]]}

    def get_paginator(self, _op):
        return _FakePaginator(self)


# Router config seed with the params the loader's set_router_split rewrites.
_ROUTER_CONFIG = (
    'name: "dlrm_bid_shader"\n'
    'backend: "python"\n'
    "parameters { key: \"stable_model\" value: { string_value: \"dlrm_bid_shader_stable\" } }\n"
    "parameters { key: \"canary_model\" value: { string_value: \"\" } }\n"
    "parameters { key: \"canary_traffic_pct\" value: { string_value: \"0\" } }\n"
)

_STABLE_CONFIG = 'name: "dlrm_bid_shader_stable"\nplatform: "tensorrt_plan"\nmax_batch_size: 64\n'


# ---------------------------------------------------------------------------
# Tests: ModelOptimizer
# ---------------------------------------------------------------------------

OPTIMIZER_ENDPOINT = "http://model-optimizer:8080"
MODEL_BUCKET = "artf-model-bucket"


class TestModelOptimizer:
    def _make_optimizer(self, http_client: MockHttpClient | None = None) -> ModelOptimizer:
        return ModelOptimizer(
            optimizer_endpoint=OPTIMIZER_ENDPOINT,
            model_bucket=MODEL_BUCKET,
            region="us-east-1",
            http_client=http_client,
        )

    @pytest.mark.asyncio
    async def test_optimize_constructs_correct_api_call(self):
        """Optimize sends fp16 by default with the per-model batch profile."""
        client = MockHttpClient({
            f"{OPTIMIZER_ENDPOINT}/v1/optimize": _ok_json({
                "output_uri": f"s3://{MODEL_BUCKET}/optimized-models/dlrm_bid_shader/model.engine"
            }),
        })
        optimizer = self._make_optimizer(client)

        await optimizer.optimize(
            model_artifact_uri=f"s3://{MODEL_BUCKET}/models/dlrm/model.onnx",
            model_name="dlrm_bid_shader",
        )

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call.method == "POST"
        assert call.url == f"{OPTIMIZER_ENDPOINT}/v1/optimize"

        payload = call.json_body
        assert payload["source_model_uri"] == f"s3://{MODEL_BUCKET}/models/dlrm/model.onnx"
        assert payload["model_name"] == "dlrm_bid_shader"
        # fp16 is the honest default (int8 would need a real calibration cache).
        assert payload["precision"] == "fp16"
        assert payload["max_batch_size"] == 64
        assert payload["target_runtime"] == "tensorrt"
        # The dlrm model has a registered batch profile, so it is carried in the
        # request (otherwise the engine would be batch-1 and Triton couldn't batch).
        assert "input_profiles" in payload
        assert "dense_features" in payload["input_profiles"]
        assert payload["input_profiles"]["dense_features"]["max"] == [64, 4]

    @pytest.mark.asyncio
    async def test_optimize_custom_config_overrides_defaults(self):
        client = MockHttpClient({
            f"{OPTIMIZER_ENDPOINT}/v1/optimize": _ok_json({}),
        })
        optimizer = self._make_optimizer(client)

        await optimizer.optimize(
            model_artifact_uri=f"s3://{MODEL_BUCKET}/models/ncf/model.onnx",
            model_name="ncf_deal_manager",
            optimization_config={"precision": "fp32", "max_batch_size": 128},
        )

        payload = client.calls[0].json_body
        assert payload["precision"] == "fp32"
        assert payload["max_batch_size"] == 128
        assert payload["max_workspace_size"] == 4_294_967_296  # default retained

    @pytest.mark.asyncio
    async def test_explicit_input_profiles_override_defaults(self):
        """An explicit input_profiles arg wins over the per-model default map."""
        client = MockHttpClient({f"{OPTIMIZER_ENDPOINT}/v1/optimize": _ok_json({})})
        optimizer = self._make_optimizer(client)

        custom = {"my_input": {"min": [1], "opt": [4], "max": [8]}}
        await optimizer.optimize(
            model_artifact_uri=f"s3://{MODEL_BUCKET}/models/dlrm/model.onnx",
            model_name="dlrm_bid_shader",
            input_profiles=custom,
        )

        assert client.calls[0].json_body["input_profiles"] == custom

    @pytest.mark.asyncio
    async def test_optimized_artifact_stays_in_same_bucket(self):
        client = MockHttpClient({f"{OPTIMIZER_ENDPOINT}/v1/optimize": _ok_json({})})
        optimizer = self._make_optimizer(client)

        result_uri = await optimizer.optimize(
            model_artifact_uri=f"s3://{MODEL_BUCKET}/models/dlrm/v2/model.onnx",
            model_name="dlrm_bid_shader",
        )

        assert result_uri.startswith(f"s3://{MODEL_BUCKET}/")
        assert "optimized-models/dlrm_bid_shader/" in result_uri
        assert result_uri.endswith(".engine")

    @pytest.mark.asyncio
    async def test_optimized_uri_uses_output_from_api(self):
        expected = f"s3://{MODEL_BUCKET}/optimized-models/dlrm_bid_shader/custom.engine"
        client = MockHttpClient({
            f"{OPTIMIZER_ENDPOINT}/v1/optimize": _ok_json({"output_uri": expected}),
        })
        optimizer = self._make_optimizer(client)

        result_uri = await optimizer.optimize(
            model_artifact_uri=f"s3://{MODEL_BUCKET}/models/dlrm/model.onnx",
            model_name="dlrm_bid_shader",
        )
        assert result_uri == expected

    @pytest.mark.asyncio
    async def test_output_uri_constructed_from_source_filename(self):
        client = MockHttpClient({f"{OPTIMIZER_ENDPOINT}/v1/optimize": _ok_json({})})
        optimizer = self._make_optimizer(client)

        result_uri = await optimizer.optimize(
            model_artifact_uri=f"s3://{MODEL_BUCKET}/training/out/dlrm_v3.onnx",
            model_name="dlrm_bid_shader",
        )
        assert "optimized-models/dlrm_bid_shader/dlrm_v3.engine" in result_uri

    @pytest.mark.asyncio
    async def test_optimize_raises_on_server_error(self):
        client = MockHttpClient({
            f"{OPTIMIZER_ENDPOINT}/v1/optimize": _error(500, "trtexec failed: GPU OOM"),
        })
        optimizer = self._make_optimizer(client)

        with pytest.raises(ModelOptimizationError) as exc_info:
            await optimizer.optimize(
                model_artifact_uri=f"s3://{MODEL_BUCKET}/models/dlrm/model.onnx",
                model_name="dlrm_bid_shader",
            )
        assert exc_info.value.status_code == 500
        assert "GPU OOM" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_optimize_raises_on_validation_error(self):
        """A 400 from the optimizer (e.g. honest int8-without-calibration) propagates."""
        client = MockHttpClient({
            f"{OPTIMIZER_ENDPOINT}/v1/optimize": _error(
                400, "int8 precision requires 'calibration_cache_uri'"
            ),
        })
        optimizer = self._make_optimizer(client)

        with pytest.raises(ModelOptimizationError) as exc_info:
            await optimizer.optimize(
                model_artifact_uri=f"s3://{MODEL_BUCKET}/models/dlrm/model.onnx",
                model_name="dlrm_bid_shader",
                optimization_config={"precision": "int8"},
            )
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Tests: TritonModelLoader (Option-D S3 poll-repo)
# ---------------------------------------------------------------------------

TRITON_URL = "http://triton-internal:8000"
BASE_MODEL = "dlrm_bid_shader"
ENGINE_URI = f"s3://{MODEL_BUCKET}/optimized-models/dlrm_bid_shader/model.engine"


class TestTritonModelLoader:
    def _make_loader(
        self,
        s3: FakeS3,
        http_client: MockHttpClient | None = None,
    ) -> TritonModelLoader:
        return TritonModelLoader(
            triton_url=TRITON_URL,
            model_bucket=MODEL_BUCKET,
            http_client=http_client,
            s3_client=s3,
            region="us-east-1",
            poll_interval_seconds=0.0,
            ready_timeout_seconds=0.0,
        )

    def _key(self, *parts: str) -> str:
        return "/".join(["triton-models", *parts])

    def test_canary_engine_uri_matches_stage_canary_write_path(self):
        """canary_engine_uri() must return exactly the path stage_canary()
        writes the engine to — this is what PromotionService relies on to
        reconstruct a staged canary's engine location cross-process,
        without depending on CanaryDeployer's in-memory DeploymentState."""
        s3 = FakeS3()
        loader = self._make_loader(s3)
        expected = f"s3://{MODEL_BUCKET}/{self._key(f'{BASE_MODEL}_canary', '1', 'model.plan')}"
        assert loader.canary_engine_uri(BASE_MODEL) == expected

    @pytest.mark.asyncio
    async def test_stage_canary_derives_config_and_copies_engine(self):
        s3 = FakeS3({self._key(f"{BASE_MODEL}_stable", "config.pbtxt"): _STABLE_CONFIG.encode()})
        loader = self._make_loader(s3)

        canary = await loader.stage_canary(BASE_MODEL, ENGINE_URI)

        assert canary == f"{BASE_MODEL}_canary"
        # Canary config derived from stable config with the model name swapped.
        canary_cfg = s3.objects[self._key(f"{BASE_MODEL}_canary", "config.pbtxt")].decode()
        assert 'name: "dlrm_bid_shader_canary"' in canary_cfg
        assert "dlrm_bid_shader_stable" not in canary_cfg
        # Engine copied to the canary version dir.
        assert self._key(f"{BASE_MODEL}_canary", "1", "model.plan") in s3.objects

    @pytest.mark.asyncio
    async def test_stage_canary_missing_stable_config_raises(self):
        s3 = FakeS3()  # no stable config present
        loader = self._make_loader(s3)

        with pytest.raises(TritonModelLoadError):
            await loader.stage_canary(BASE_MODEL, ENGINE_URI)

    @pytest.mark.asyncio
    async def test_remove_canary_deletes_prefix(self):
        s3 = FakeS3({
            self._key(f"{BASE_MODEL}_canary", "config.pbtxt"): b"cfg",
            self._key(f"{BASE_MODEL}_canary", "1", "model.plan"): b"engine",
        })
        loader = self._make_loader(s3)

        await loader.remove_canary(BASE_MODEL)

        remaining = [k for k in s3.objects if f"{BASE_MODEL}_canary" in k]
        assert remaining == []

    @pytest.mark.asyncio
    async def test_promote_engine_publishes_next_version(self):
        s3 = FakeS3({self._key(f"{BASE_MODEL}_stable", "1", "model.plan"): b"v1-engine"})
        loader = self._make_loader(s3)

        new_version = await loader.promote_engine(BASE_MODEL, ENGINE_URI)

        assert new_version == 2
        assert self._key(f"{BASE_MODEL}_stable", "2", "model.plan") in s3.objects

    @pytest.mark.asyncio
    async def test_promote_engine_first_version_when_none(self):
        s3 = FakeS3()  # no existing stable versions
        loader = self._make_loader(s3)

        new_version = await loader.promote_engine(BASE_MODEL, ENGINE_URI)

        assert new_version == 1
        assert self._key(f"{BASE_MODEL}_stable", "1", "model.plan") in s3.objects

    @pytest.mark.asyncio
    async def test_list_versions_returns_sorted_numeric_dirs(self):
        s3 = FakeS3({
            self._key(f"{BASE_MODEL}_stable", "1", "model.plan"): b"a",
            self._key(f"{BASE_MODEL}_stable", "3", "model.plan"): b"b",
            self._key(f"{BASE_MODEL}_stable", "2", "model.plan"): b"c",
            # a non-numeric dir must be ignored
            self._key(f"{BASE_MODEL}_stable", "config.pbtxt"): b"cfg",
        })
        loader = self._make_loader(s3)

        versions = await loader.list_versions(f"{BASE_MODEL}_stable")
        assert versions == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_set_router_split_rewrites_config_params(self):
        s3 = FakeS3({self._key(BASE_MODEL, "config.pbtxt"): _ROUTER_CONFIG.encode()})
        loader = self._make_loader(s3)

        await loader.set_router_split(BASE_MODEL, 25.0, canary_model=f"{BASE_MODEL}_canary")

        cfg = s3.objects[self._key(BASE_MODEL, "config.pbtxt")].decode()
        assert 'key: "canary_traffic_pct" value: { string_value: "25.0" }' in cfg
        assert 'key: "canary_model" value: { string_value: "dlrm_bid_shader_canary" }' in cfg

    @pytest.mark.asyncio
    async def test_set_router_split_missing_router_config_raises(self):
        s3 = FakeS3()
        loader = self._make_loader(s3)

        with pytest.raises(TritonModelLoadError):
            await loader.set_router_split(BASE_MODEL, 5.0)

    def test_set_param_raises_when_absent(self):
        with pytest.raises(ValueError):
            TritonModelLoader._set_param('name: "x"', "canary_traffic_pct", "5")

    @pytest.mark.asyncio
    async def test_is_ready_true_on_200(self):
        s3 = FakeS3()
        http = MockHttpClient({
            f"{TRITON_URL}/v2/models/{BASE_MODEL}_canary/ready": _ok_json({}),
        })
        loader = self._make_loader(s3, http)

        assert await loader.is_ready(f"{BASE_MODEL}_canary") is True

    @pytest.mark.asyncio
    async def test_is_ready_false_on_non_200(self):
        s3 = FakeS3()
        http = MockHttpClient({
            f"{TRITON_URL}/v2/models/{BASE_MODEL}_canary/ready": _error(503, "not ready"),
        })
        loader = self._make_loader(s3, http)

        assert await loader.is_ready(f"{BASE_MODEL}_canary") is False

    @pytest.mark.asyncio
    async def test_wait_until_ready_polls_then_true(self):
        s3 = FakeS3()
        http = MockHttpClient({
            f"{TRITON_URL}/v2/models/{BASE_MODEL}_canary/ready": _ok_json({}),
        })
        loader = self._make_loader(s3, http)

        assert await loader.wait_until_ready(f"{BASE_MODEL}_canary") is True
