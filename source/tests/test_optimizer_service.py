"""Unit tests for the Model Optimizer service (source/optimizer/app.py).

Covers the HONEST-refusal validation branches of POST /v1/optimize — the ones that
return before any trtexec/S3 work, so they need no GPU or network:

- int8 without a calibration cache → 400 (never emit an uncalibrated int8 engine)
- max_batch_size > 1 without input_profiles → 400 (never emit a batch-1 engine for
  a batched model)
- malformed input_profiles → 400
- missing required fields / invalid precision → 400

Plus the pure batch-profile helpers (_valid_input_profiles, _shapes_flag) that build
the trtexec --minShapes/--optShapes/--maxShapes flags.

These assert the no-fabrication contract: the service refuses rather than producing
a mislabeled or unbatchable engine.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optimizer.app import _shapes_flag, _valid_input_profiles, optimize


class _FakeRequest:
    """Minimal Starlette-request stand-in exposing async json()."""

    def __init__(self, body):
        self._body = body

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _body(resp) -> dict:
    return json.loads(bytes(resp.body).decode())


_DLRM_PROFILE = {
    "dense_features": {"min": [1, 4], "opt": [8, 4], "max": [64, 4]},
    "sparse_user": {"min": [1], "opt": [8], "max": [64]},
}


# ---------------------------------------------------------------------------
# Honest-refusal validation branches (no trtexec / no S3 reached)
# ---------------------------------------------------------------------------


class TestOptimizeValidation:
    @pytest.mark.asyncio
    async def test_int8_without_calibration_cache_is_400(self):
        resp = await optimize(_FakeRequest({
            "source_model_uri": "s3://b/model.onnx",
            "output_uri": "s3://b/optimized-models/dlrm/model.engine",
            "model_name": "dlrm_bid_shader",
            "precision": "int8",
            # no calibration_cache_uri, max_batch_size defaults to 1
        }))
        assert resp.status_code == 400
        assert "calibration" in _body(resp)["error"].lower()

    @pytest.mark.asyncio
    async def test_batched_without_input_profiles_is_400(self):
        resp = await optimize(_FakeRequest({
            "source_model_uri": "s3://b/model.onnx",
            "output_uri": "s3://b/optimized-models/dlrm/model.engine",
            "model_name": "dlrm_bid_shader",
            "precision": "fp16",
            "max_batch_size": 64,
            # no input_profiles for a batched model
        }))
        assert resp.status_code == 400
        assert "input_profiles" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_malformed_input_profiles_is_400(self):
        resp = await optimize(_FakeRequest({
            "source_model_uri": "s3://b/model.onnx",
            "output_uri": "s3://b/optimized-models/dlrm/model.engine",
            "model_name": "dlrm_bid_shader",
            "precision": "fp16",
            "max_batch_size": 1,
            "input_profiles": {"dense_features": {"min": [1, 4]}},  # missing opt/max
        }))
        assert resp.status_code == 400
        assert "input_profiles" in _body(resp)["error"]

    @pytest.mark.asyncio
    async def test_missing_required_fields_is_400(self):
        resp = await optimize(_FakeRequest({"precision": "fp16"}))
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_precision_is_400(self):
        resp = await optimize(_FakeRequest({
            "source_model_uri": "s3://b/model.onnx",
            "output_uri": "s3://b/optimized-models/dlrm/model.engine",
            "model_name": "dlrm_bid_shader",
            "precision": "bf8",
        }))
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_json_body_is_400(self):
        resp = await optimize(_FakeRequest(ValueError("bad json")))
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Batch-profile helpers (pure functions)
# ---------------------------------------------------------------------------


class TestBatchProfileHelpers:
    def test_valid_input_profiles_accepts_well_formed(self):
        assert _valid_input_profiles(_DLRM_PROFILE) is True

    def test_valid_input_profiles_rejects_empty(self):
        assert _valid_input_profiles({}) is False

    def test_valid_input_profiles_rejects_missing_key(self):
        assert _valid_input_profiles({"x": {"min": [1], "opt": [1]}}) is False

    def test_valid_input_profiles_rejects_non_list_dims(self):
        assert _valid_input_profiles({"x": {"min": 1, "opt": 1, "max": 1}}) is False

    def test_shapes_flag_builds_trtexec_value(self):
        assert _shapes_flag(_DLRM_PROFILE, "min") == "dense_features:1x4,sparse_user:1"
        assert _shapes_flag(_DLRM_PROFILE, "max") == "dense_features:64x4,sparse_user:64"
