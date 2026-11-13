"""Learner-mode Triton model configs (source/triton/model_repository_learner/).

``deploy.sh --learner-mode`` overlays these files on the normal model repository
so Triton serves on CPU. Each one must stay identical to its GPU sibling in
everything but the execution fields (platform/backend and instance_group), or the
two modes would serve different contracts to the ARTF containers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
GPU_REPO = SOURCE_DIR / "triton" / "model_repository"
LEARNER_REPO = SOURCE_DIR / "triton" / "model_repository_learner"

LEARNER_MODELS = [
    "deal_yield_manager_floor",
    "deal_yield_manager_margin",
    "dlrm_bid_shader_canary",
    "dlrm_bid_shader_stable",
    "ncf_deal_manager_stable",
]
ONNX_MODELS = ["dlrm_bid_shader_canary", "dlrm_bid_shader_stable", "ncf_deal_manager_stable"]
FIL_MODELS = ["deal_yield_manager_floor", "deal_yield_manager_margin"]


def _parse(path: Path):
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2

    return text_format.Parse(path.read_text(), model_config_pb2.ModelConfig())


def _learner(name):
    return _parse(LEARNER_REPO / name / "config.pbtxt")


def _gpu(name):
    return _parse(GPU_REPO / name / "config.pbtxt")


def test_learner_repo_holds_exactly_the_gpu_models():
    present = sorted(p.parent.name for p in LEARNER_REPO.glob("*/config.pbtxt"))
    assert present == LEARNER_MODELS
    # Only config files: deploy.sh copies this directory over the staged repo and
    # syncs it to S3, so anything else here would be served too.
    assert sorted(p.relative_to(LEARNER_REPO).as_posix() for p in LEARNER_REPO.rglob("*") if p.is_file()) == [
        f"{m}/config.pbtxt" for m in LEARNER_MODELS
    ]


@pytest.mark.parametrize("name", LEARNER_MODELS)
def test_learner_config_matches_gpu_sibling_outside_execution_fields(name):
    learner, gpu = _learner(name), _gpu(name)
    for cfg in (learner, gpu):
        cfg.ClearField("platform")
        cfg.ClearField("backend")
        cfg.ClearField("instance_group")
    assert learner == gpu


@pytest.mark.parametrize("name", LEARNER_MODELS)
def test_learner_config_runs_on_cpu_only(name):
    cfg = _learner(name)
    from tritonclient.grpc import model_config_pb2

    assert len(cfg.instance_group) >= 1
    for group in cfg.instance_group:
        assert group.kind == model_config_pb2.ModelInstanceGroup.KIND_CPU
        assert list(group.gpus) == []
    assert cfg.platform != "tensorrt_plan"
    assert len(cfg.optimization.execution_accelerators.gpu_execution_accelerator) == 0
    text = (LEARNER_REPO / name / "config.pbtxt").read_text()
    assert "KIND_GPU" not in text
    assert "tensorrt_plan" not in text


@pytest.mark.parametrize("name", ONNX_MODELS)
def test_engine_slots_use_onnx_runtime(name):
    cfg = _learner(name)
    assert cfg.platform == "onnxruntime_onnx"
    assert cfg.backend == ""
    assert _gpu(name).platform == "tensorrt_plan"


@pytest.mark.parametrize("name", FIL_MODELS)
def test_yield_models_stay_on_fil_with_xgboost_json(name):
    cfg = _learner(name)
    assert cfg.backend == "fil"
    assert cfg.platform == ""
    assert cfg.parameters["model_type"].string_value == "xgboost_json"


@pytest.mark.parametrize("name", LEARNER_MODELS)
def test_learner_warmup_inputs_match_declared_inputs(name):
    cfg = _learner(name)
    assert cfg.name == name
    assert len(cfg.model_warmup) >= 1
    declared = {i.name: i for i in cfg.input}
    for sample in cfg.model_warmup:
        assert 1 <= sample.batch_size <= cfg.max_batch_size
        assert set(sample.inputs) == set(declared), sample.name
        for key, value in sample.inputs.items():
            assert value.data_type == declared[key].data_type, key
            assert list(value.dims) == list(declared[key].dims), key
            assert value.WhichOneof("input_data_type") == "zero_data", key


def test_learner_configs_carry_the_compliance_caveat_and_no_dashes():
    for name in LEARNER_MODELS:
        text = (LEARNER_REPO / name / "config.pbtxt").read_text()
        assert "learner-mode" in text
        assert "production workloads must not" in text
        assert "\u2014" not in text and "\u2013" not in text
