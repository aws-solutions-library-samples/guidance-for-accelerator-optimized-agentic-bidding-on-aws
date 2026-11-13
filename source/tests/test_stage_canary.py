"""On-demand canary staging for the FIL-served yield models.

Nothing could create a yield canary before this, so the variant routing added in
the yield containers was unreachable end to end. Two things were in the way:

1. no endpoint existed to stage a registered version as a canary
2. ``TritonModelLoader.stage_canary_fil`` derived the canary config from
   ``<base>_stable``, which does not exist for the yield models -- they use the
   direct layout with no router. Verified live: ``deal_yield_manager_floor`` is
   ready while ``deal_yield_manager_floor_stable`` returns 400.

The TensorRT-served types are deliberately refused here rather than
half-supported: their ONNX must be compiled into a model.plan by the Model
Optimizer first, and that orchestration lives in the governance agent. Staging
without the compile would produce a canary Triton could never load.
"""

from __future__ import annotations

import json
import sys
import unittest.mock as mock
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import governance_api  # noqa: E402

VERSION_ARN = "arn:aws:sagemaker:us-east-1:1:model-package/g-deal-yield-manager-floor/2"
ARTIFACT = "s3://bucket/models/deal_yield_manager_floor/job-1/job-1/output/model.tar.gz"


def _client():
    app = Starlette(routes=[
        Route("/v1/governance/stage-canary", governance_api.stage_canary_handler, methods=["POST"])
    ])
    return TestClient(app)


def _sagemaker(*, metadata_uri=ARTIFACT, model_data_url=None):
    detail = {"CustomerMetadataProperties": {}}
    if metadata_uri:
        detail["CustomerMetadataProperties"]["model_artifact_uri"] = metadata_uri
    if model_data_url:
        detail["InferenceSpecification"] = {"Containers": [{"ModelDataUrl": model_data_url}]}
    client = mock.Mock()
    client.describe_model_package.return_value = detail
    return client


def _loader(staged="deal_yield_manager_floor_canary"):
    loader = mock.Mock()
    loader.stage_canary_fil = mock.AsyncMock(return_value=staged)
    return loader


# ---------------------------------------------------------------------------
# The FIL path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_type", ["deal_yield_manager_floor", "deal_yield_manager_margin"]
)
def test_stages_a_fil_canary(model_type):
    loader = _loader(f"{model_type}_canary")
    with mock.patch.object(governance_api, "_sagemaker_client", lambda: _sagemaker()), \
         mock.patch.object(governance_api, "_triton_loader", lambda: loader):
        resp = _client().post(
            "/v1/governance/stage-canary",
            json={"model_type": model_type, "version_arn": VERSION_ARN},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["canary_model"] == f"{model_type}_canary"
    assert body["artifact_uri"] == ARTIFACT
    loader.stage_canary_fil.assert_awaited_once_with(model_type, ARTIFACT)


def test_does_not_claim_the_canary_is_ready():
    """Triton poll-loads from the repo, so it is not servable when this returns.
    Reporting ready=True would let a challenger run start against a model that
    is not up yet."""
    with mock.patch.object(governance_api, "_sagemaker_client", lambda: _sagemaker()), \
         mock.patch.object(governance_api, "_triton_loader", lambda: _loader()):
        resp = _client().post(
            "/v1/governance/stage-canary",
            json={"model_type": "deal_yield_manager_floor", "version_arn": VERSION_ARN},
        )
    assert resp.json()["ready"] is False


def test_falls_back_to_model_data_url_when_metadata_is_absent():
    """SageMaker always populates ModelDataUrl; the metadata key is only written
    by our own registration Lambda."""
    loader = _loader()
    sm = _sagemaker(metadata_uri=None, model_data_url="s3://bucket/other/model.tar.gz")
    with mock.patch.object(governance_api, "_sagemaker_client", lambda: sm), \
         mock.patch.object(governance_api, "_triton_loader", lambda: loader):
        resp = _client().post(
            "/v1/governance/stage-canary",
            json={"model_type": "deal_yield_manager_floor", "version_arn": VERSION_ARN},
        )
    assert resp.status_code == 202
    loader.stage_canary_fil.assert_awaited_once_with(
        "deal_yield_manager_floor", "s3://bucket/other/model.tar.gz"
    )


def test_missing_artifact_is_refused_not_guessed():
    sm = _sagemaker(metadata_uri=None)
    with mock.patch.object(governance_api, "_sagemaker_client", lambda: sm), \
         mock.patch.object(governance_api, "_triton_loader", lambda: _loader()):
        resp = _client().post(
            "/v1/governance/stage-canary",
            json={"model_type": "deal_yield_manager_floor", "version_arn": VERSION_ARN},
        )
    assert resp.status_code == 422
    assert resp.json()["reason"] == "artifact_not_found"


# ---------------------------------------------------------------------------
# The TensorRT types must be refused with a reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["dlrm_bid_shader", "ncf_deal_manager"])
def test_tensorrt_types_are_refused_with_compile_required(model_type):
    loader = _loader()
    with mock.patch.object(governance_api, "_sagemaker_client", lambda: _sagemaker()), \
         mock.patch.object(governance_api, "_triton_loader", lambda: loader):
        resp = _client().post(
            "/v1/governance/stage-canary",
            json={"model_type": model_type, "version_arn": VERSION_ARN},
        )
    assert resp.status_code == 422
    assert resp.json()["reason"] == "compile_required"
    # Must not have attempted to stage an artifact Triton could not load.
    loader.stage_canary_fil.assert_not_awaited()


def test_missing_fields_are_rejected():
    resp = _client().post("/v1/governance/stage-canary", json={"model_type": ""})
    assert resp.status_code == 422
    assert resp.json()["reason"] == "bad_request"


# ---------------------------------------------------------------------------
# The config-source fallback that makes a yield canary possible at all
# ---------------------------------------------------------------------------

class _FakeLoader:
    """Exercises the real _canary_source_config against a fake repo."""

    def __init__(self, objects):
        from deployment.model_deployer import TritonModelLoader

        self._objects = objects
        self._real = TritonModelLoader.__new__(TritonModelLoader)
        self._real._model_bucket = "b"
        self._real._repo_prefix = "triton-models"

    def _key(self, *parts):
        return self._real._key(*parts)

    async def _get_text(self, key):
        return self._objects.get(key)

    stable_name = staticmethod(lambda m: f"{m}_stable")
    canary_name = staticmethod(lambda m: f"{m}_canary")

    async def source(self, base_model):
        from deployment.model_deployer import TritonModelLoader

        return await TritonModelLoader._canary_source_config(self, base_model)


@pytest.mark.asyncio
async def test_router_layout_derives_from_the_stable_config():
    loader = _FakeLoader({
        "triton-models/dlrm_bid_shader_stable/config.pbtxt": 'name: "dlrm_bid_shader_stable"',
    })
    cfg, name = await loader.source("dlrm_bid_shader")
    assert name == "dlrm_bid_shader_stable"
    assert "dlrm_bid_shader_stable" in cfg


@pytest.mark.asyncio
async def test_direct_layout_falls_back_to_the_base_config():
    """The yield case: no <base>_stable exists, so the base model's own config is
    the only real thing to derive from."""
    loader = _FakeLoader({
        "triton-models/deal_yield_manager_floor/config.pbtxt": 'name: "deal_yield_manager_floor"',
    })
    cfg, name = await loader.source("deal_yield_manager_floor")
    assert name == "deal_yield_manager_floor"
    assert "deal_yield_manager_floor" in cfg


@pytest.mark.asyncio
async def test_stable_config_wins_when_both_exist():
    loader = _FakeLoader({
        "triton-models/m_stable/config.pbtxt": 'name: "m_stable"',
        "triton-models/m/config.pbtxt": 'name: "m"',
    })
    _cfg, name = await loader.source("m")
    assert name == "m_stable"


@pytest.mark.asyncio
async def test_no_config_at_all_raises_rather_than_inventing_one():
    from deployment.model_deployer import TritonModelLoadError

    loader = _FakeLoader({})
    with pytest.raises(TritonModelLoadError):
        await loader.source("m")
