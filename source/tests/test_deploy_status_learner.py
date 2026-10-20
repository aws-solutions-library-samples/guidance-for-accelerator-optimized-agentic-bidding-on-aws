"""deploy_status.py in learner mode (deploy.sh --learner-mode).

A learner deployment has no GPU node group, no Model Optimizer image and no
TensorRT engine; the Deal Scorer is served from its ONNX. The probe must expect
exactly that in learner mode and nothing different by default.
"""
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

_SCRIPTS = Path(__file__).resolve().parents[2] / "deployment" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import deploy_status  # noqa: E402
from deploy_status import (  # noqa: E402
    CORE_IMAGES,
    MISSING,
    NCF_SERVED_ONNX_KEY,
    OK,
    STALE,
    UNKNOWN,
    Probe,
)

FP = "b" * 64
OTHER = "c" * 64
BUCKET = "t-nvidia-artf-recommenders-triton-models-0123abcd"


def _probe(*, learner=False, retraining=True, fp="", s3=None):
    p = Probe.__new__(Probe)  # no AWS clients, no kubeconfig
    p.stack = "t-nvidia-artf-recommenders"
    p.uid = "0123abcd"
    p.prefix = "t"
    p.learner = learner
    p.retraining = retraining
    p.expect_ncf_fingerprint = fp
    p._client = lambda service: s3 if service == "s3" else None
    return p


# --- Phase 2: images ----------------------------------------------------------

def test_default_expects_the_model_optimizer_image():
    assert "model-optimizer" in _probe()._expected_images()
    assert _probe()._expected_images()[:len(CORE_IMAGES)] == CORE_IMAGES


def test_learner_does_not_expect_the_model_optimizer_image():
    images = _probe(learner=True, retraining=False)._expected_images()
    assert "model-optimizer" not in images
    assert set(images) == set(CORE_IMAGES) - {"model-optimizer"}


# --- Phase 2: node groups -------------------------------------------------------

class FakeEks:
    def __init__(self, groups):
        self.groups = groups

    def list_nodegroups(self, clusterName):
        return {"nodegroups": list(self.groups)}

    def describe_nodegroup(self, clusterName, nodegroupName):
        desired = self.groups[nodegroupName]
        return {"nodegroup": {"status": "ACTIVE", "scalingConfig": {"desiredSize": desired},
                              "instanceTypes": ["c5.2xlarge"], "health": {"issues": []}}}


def _nodegroup_checks(groups, *, learner):
    p = _probe(learner=learner)
    p._registered_nodes = lambda: {name: n for name, n in groups.items()}
    p._running_instances = lambda ec2, cluster, ng: groups[ng]
    p._scheduled_action = lambda d: ""
    return {c["name"]: c for c in p._nodegroups(FakeEks(groups), "t-cluster")}


def test_gpu_group_at_zero_is_missing_by_default():
    checks = _nodegroup_checks({"gpu-inference": 0, "cpu-services": 3}, learner=False)
    assert checks["node group gpu-inference"]["status"] == MISSING
    assert checks["node group cpu-services"]["status"] == OK


def test_gpu_group_at_zero_is_ok_in_learner_mode():
    checks = _nodegroup_checks({"gpu-inference": 0, "cpu-services": 4}, learner=True)
    assert checks["node group gpu-inference"]["status"] == OK
    assert "learner mode" in checks["node group gpu-inference"]["detail"]
    assert checks["node group cpu-services"]["status"] == OK


def test_cpu_group_at_zero_is_still_missing_in_learner_mode():
    # Only the GPU group is expected to be empty; an empty cpu-services is real downtime.
    checks = _nodegroup_checks({"cpu-services": 0}, learner=True)
    assert checks["node group cpu-services"]["status"] == MISSING


# --- Phase 3: the served Deal Scorer ONNX ------------------------------------------

def _err(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "HeadObject")


class FakeS3:
    def __init__(self, objects=None, error=None):
        self.objects = objects or {}
        self.error = error
        self.calls = []

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Bucket, Key))
        if self.error:
            raise self.error
        if Key not in self.objects:
            raise _err("404")
        return {"Metadata": self.objects[Key]}


def test_served_onnx_with_the_local_build_is_ok():
    s3 = FakeS3({NCF_SERVED_ONNX_KEY: {"source-fingerprint": FP}})
    check = _probe(learner=True, fp=FP, s3=s3)._ncf_served_onnx()
    assert check["status"] == OK
    assert s3.calls == [("head_object", BUCKET, NCF_SERVED_ONNX_KEY)]


def test_served_onnx_from_another_build_is_stale():
    s3 = FakeS3({NCF_SERVED_ONNX_KEY: {"Source-Fingerprint": OTHER}})
    check = _probe(learner=True, fp=FP, s3=s3)._ncf_served_onnx()
    assert check["status"] == STALE
    assert "cccccccccccc" in check["detail"] and "bbbbbbbbbbbb" in check["detail"]


def test_missing_served_onnx_is_missing():
    check = _probe(learner=True, fp=FP, s3=FakeS3())._ncf_served_onnx()
    assert check["status"] == MISSING
    assert NCF_SERVED_ONNX_KEY in check["detail"]


def test_served_onnx_access_error_is_unknown():
    check = _probe(learner=True, fp=FP, s3=FakeS3(error=_err("403")))._ncf_served_onnx()
    assert check["status"] == UNKNOWN


def test_phase3_checks_the_served_onnx_in_learner_mode_and_the_engine_otherwise(monkeypatch):
    workloads = [{"name": "deployment x", "status": OK, "detail": ""}]
    p = _probe(learner=True, fp=FP, s3=FakeS3({NCF_SERVED_ONNX_KEY: {"source-fingerprint": FP}}))
    monkeypatch.setattr(p, "_workloads", lambda: list(workloads), raising=False)
    assert [c["name"] for c in p.phase3()] == ["deployment x", "Deal Scorer served ONNX"]

    p = _probe(learner=False, fp=FP, s3=FakeS3())
    monkeypatch.setattr(p, "_workloads", lambda: list(workloads), raising=False)
    assert [c["name"] for c in p.phase3()] == ["deployment x", "Deal Scorer engine"]


def test_served_onnx_key_matches_deploy_sh():
    deploy = (Path(__file__).resolve().parents[2] / "deployment" / "deploy.sh").read_text()
    assert "triton-models/ncf_deal_manager_stable/1/model.onnx" in deploy
    assert NCF_SERVED_ONNX_KEY == "triton-models/ncf_deal_manager_stable/1/model.onnx"


# --- CLI ------------------------------------------------------------------------

@pytest.mark.parametrize("argv,expected", [([], False), (["--learner-mode"], True)])
def test_cli_passes_the_learner_flag(monkeypatch, argv, expected):
    seen = {}

    class _Probe:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def run(self):
            return {"phases": {}, "kubeContext": "", "notes": []}

    monkeypatch.setattr(deploy_status, "Probe", _Probe)
    monkeypatch.setattr(deploy_status, "boto3", object())
    deploy_status.main(["--prefix", "t", "--shell", *argv])
    assert seen["learner"] is expected
