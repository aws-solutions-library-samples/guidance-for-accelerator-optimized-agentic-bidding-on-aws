"""The canary routers in learner mode.

The Python-backend routers (dlrm_bid_shader, ncf_deal_manager) are served unchanged
in learner mode; only the stable/canary models behind them move to ONNX Runtime on
CPU. These tests run source/triton/router/model.py against its real router configs
with a stub triton_python_backend_utils, and check that every model a router can
dispatch to in learner mode exists in the learner repository and runs on CPU.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
TRITON_DIR = SOURCE_DIR / "triton"
GPU_REPO = TRITON_DIR / "model_repository"
LEARNER_REPO = TRITON_DIR / "model_repository_learner"


class _Tensor:
    def __init__(self, name, value):
        self._name, self._value = name, value

    def name(self):
        return self._name

    def as_numpy(self):
        return self._value


class _Response:
    def __init__(self, output_tensors=None, error=None):
        self._outputs = list(output_tensors or [])
        self._error = error

    def has_error(self):
        return self._error is not None

    def error(self):
        return self._error

    def output_tensors(self):
        return self._outputs


def _stub_pb_utils(dispatched):
    mod = types.ModuleType("triton_python_backend_utils")

    class TritonModelException(Exception):
        pass

    class InferenceRequest:
        def __init__(self, model_name, requested_output_names, inputs):
            self.model_name = model_name
            self.requested_output_names = requested_output_names
            self.inputs = inputs

        def exec(self):
            dispatched.append(self.model_name)
            return _Response([_Tensor(n, np.zeros(1, dtype=np.float32)) for n in self.requested_output_names])

    mod.TritonModelException = TritonModelException
    mod.Tensor = _Tensor
    mod.InferenceRequest = InferenceRequest
    mod.InferenceResponse = _Response
    mod.TritonError = lambda msg: msg
    mod.get_input_tensor_by_name = lambda request, name: request.get(name)
    return mod


@pytest.fixture
def router(monkeypatch):
    dispatched = []
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", _stub_pb_utils(dispatched))
    spec = importlib.util.spec_from_file_location("canary_router_under_test", TRITON_DIR / "router" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def make(router_name, **param_overrides):
        from google.protobuf import json_format, text_format
        from tritonclient.grpc import model_config_pb2

        cfg = text_format.Parse((GPU_REPO / router_name / "config.pbtxt").read_text(),
                                model_config_pb2.ModelConfig())
        for key, value in param_overrides.items():
            cfg.parameters[key].string_value = value
        model = module.TritonPythonModel()
        model.initialize({"model_config": json_format.MessageToJson(cfg, preserving_proto_field_name=True)})
        return model

    make.dispatched = dispatched
    return make


def test_dlrm_router_at_zero_percent_always_picks_stable(router):
    model = router("dlrm_bid_shader")
    assert model.canary_traffic_pct == 0.0
    assert {model._pick_target() for _ in range(500)} == {("dlrm_bid_shader_stable", "stable")}


def test_dlrm_router_canary_override_reaches_the_canary_slot(router):
    # dlrm_bid_shader_canary is served in learner mode (same genesis ONNX as stable).
    model = router("dlrm_bid_shader")
    assert model._pick_target("canary") == ("dlrm_bid_shader_canary", "canary")
    assert model._pick_target("stable") == ("dlrm_bid_shader_stable", "stable")


def test_router_without_a_canary_model_falls_back_to_stable(router):
    model = router("dlrm_bid_shader", canary_model="")
    assert model._pick_target("canary") == ("dlrm_bid_shader_stable", "stable")


def test_ncf_router_dispatches_to_stable_and_echoes_the_variant(router):
    model = router("ncf_deal_manager")
    request = {"user_ids": _Tensor("user_ids", np.array([[1]], dtype=np.int64)),
               "item_ids": _Tensor("item_ids", np.array([[2]], dtype=np.int64))}
    [response] = model.execute([request])
    assert router.dispatched == ["ncf_deal_manager_stable"]
    assert not response.has_error()
    names = [t.name() for t in response.output_tensors()]
    assert names == ["relevance_scores", "served_variant", "served_model_version"]
    variant = dict(zip(names, response.output_tensors()))["served_variant"].as_numpy()[0]
    assert variant == b"stable"


def _overlay(tmp_path: Path) -> Path:
    """The served repository as deploy.sh stage_served_repo assembles it in learner mode."""
    repo = tmp_path / "served"
    shutil.copytree(GPU_REPO, repo)
    shutil.copytree(LEARNER_REPO, repo, dirs_exist_ok=True)
    return repo


def _params(repo: Path, router_name: str):
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2

    cfg = text_format.Parse((repo / router_name / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())
    return {k: v.string_value for k, v in cfg.parameters.items()}


@pytest.mark.parametrize("router_name,slots", [
    ("dlrm_bid_shader", ("stable_model", "canary_model")),
    ("ncf_deal_manager", ("stable_model",)),
])
def test_every_dispatch_target_in_the_learner_repo_runs_on_cpu(tmp_path, router_name, slots):
    repo = _overlay(tmp_path)
    params = _params(repo, router_name)
    for slot in slots:
        target = params[slot]
        text = (repo / target / "config.pbtxt").read_text()
        assert 'platform: "onnxruntime_onnx"' in text, target
        assert "KIND_GPU" not in text and "tensorrt_plan" not in text, target
    # The routers themselves are the unchanged Python-backend configs.
    assert (repo / router_name / "config.pbtxt").read_text() == (GPU_REPO / router_name / "config.pbtxt").read_text()


def test_ncf_canary_slot_is_absent_in_both_modes(tmp_path):
    # Pre-existing: the NCF router names ncf_deal_manager_canary, which neither mode
    # provisions. At 0% it is only reachable through the load-test override.
    repo = _overlay(tmp_path)
    target = _params(repo, "ncf_deal_manager")["canary_model"]
    assert not (repo / target).exists()
    assert not (GPU_REPO / target).exists()


def test_no_served_model_in_the_learner_repo_needs_a_gpu(tmp_path):
    repo = _overlay(tmp_path)
    for cfg in repo.glob("*/config.pbtxt"):
        text = cfg.read_text()
        assert "KIND_GPU" not in text, cfg.parent.name
        assert 'platform: "tensorrt_plan"' not in text, cfg.parent.name
    # Triton reads only config.pbtxt; the legacy config_tensorrt.pbtxt files beside
    # the routers are inert in both modes.
    assert sorted(p.parent.name for p in repo.glob("*/config_tensorrt.pbtxt")) == \
        ["dlrm_bid_shader", "ncf_deal_manager"]
