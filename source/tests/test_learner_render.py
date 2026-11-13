"""deployment/scripts/learner_render.py: the learner-mode manifest rewrite.

Renders each real template from deployment/eks/ with a placeholder substitution,
applies the learner transform, and checks the result: no GPU request, Triton on
the CPU node group, a real DEPLOYMENT_MODE env var on the orchestrator, and every
other line of the input left as it was.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import pytest
import yaml

_DEPLOYMENT = Path(__file__).resolve().parents[2] / "deployment"
_SCRIPTS = _DEPLOYMENT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import learner_render  # noqa: E402
from learner_render import AnchorError, transform  # noqa: E402

EKS = _DEPLOYMENT / "eks"


def _rendered(name: str) -> str:
    """Template text with every __PLACEHOLDER__ replaced, as deploy.sh's sed does."""
    text = (EKS / f"{name}.yaml").read_text()
    text = text.replace("__MAX_GPUS__", "3")
    return re.sub(r"__([A-Z_]+)__", lambda m: m.group(1).lower().replace("_", "-"), text)


def _docs(text: str):
    return [d for d in yaml.safe_load_all(text) if d is not None]


def _kinds_and_names(text: str):
    return [(d.get("kind"), (d.get("metadata") or {}).get("name")) for d in _docs(text)]


def _changed_lines(before: str, after: str):
    removed, added = [], []
    for line in difflib.ndiff(before.splitlines(), after.splitlines()):
        if line.startswith("- "):
            removed.append(line[2:])
        elif line.startswith("+ "):
            added.append(line[2:])
    return removed, added


@pytest.mark.parametrize("name", ["triton-deployment", "triton-hpa", "cluster-config"])
def test_learner_render_has_no_gpu_reference(name):
    out = transform(name, _rendered(name))
    assert "nvidia.com/gpu" not in out
    assert "gpu-inference" not in out
    assert "KIND_GPU" not in out


def test_default_templates_do_reference_gpus():
    # The transform is what removes them; the templates themselves are unchanged.
    for name in ("triton-deployment", "triton-hpa", "cluster-config", "artf-containers-deployment"):
        assert "nvidia.com/gpu" in _rendered(name)


def test_triton_runs_on_the_cpu_node_group_with_cpu_resources():
    out = transform("triton-deployment", _rendered("triton-deployment"))
    deploy = next(d for d in _docs(out) if d["kind"] == "Deployment")
    pod = deploy["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"role": "services"}
    assert "tolerations" not in pod
    [container] = pod["containers"]
    assert container["image"] == "nvcr.io/nvidia/tritonserver:24.08-py3"
    assert container["resources"] == {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "8Gi"},
    }
    # Same server flags, ports and probes as the GPU deployment.
    gpu = next(d for d in _docs(_rendered("triton-deployment")) if d["kind"] == "Deployment")
    gpu_container = gpu["spec"]["template"]["spec"]["containers"][0]
    for key in ("args", "ports", "readinessProbe", "livenessProbe", "startupProbe", "env"):
        assert container[key] == gpu_container[key], key


def test_triton_hpa_scales_on_cpu_only():
    out = transform("triton-hpa", _rendered("triton-hpa"))
    hpa = next(d for d in _docs(out) if d["kind"] == "HorizontalPodAutoscaler")
    assert hpa["spec"]["maxReplicas"] == 2
    assert [m["resource"]["name"] for m in hpa["spec"]["metrics"]] == ["cpu"]


def test_orchestrator_gets_exactly_one_learner_mode_env_var():
    before = _rendered("orchestrator-deployment")
    out = transform("orchestrator-deployment", before)
    assert out.count("DEPLOYMENT_MODE") == 1
    removed, added = _changed_lines(before, out)
    assert removed == []
    assert [a for a in added if not a.strip().startswith("#")] == [
        "            - name: DEPLOYMENT_MODE",
        '              value: "learner"',
    ]
    deploy = next(d for d in _docs(out) if d["kind"] == "Deployment")
    env = {e["name"]: e.get("value") for e in deploy["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["DEPLOYMENT_MODE"] == "learner"


def test_default_orchestrator_template_has_no_deployment_mode():
    # The default deploy renders no such variable, which the orchestrator reads as standard.
    assert "DEPLOYMENT_MODE" not in _rendered("orchestrator-deployment")


def test_cluster_config_drops_the_gpu_group_and_sizes_cpu_services():
    out = transform("cluster-config", _rendered("cluster-config"))
    [cfg] = _docs(out)
    groups = {g["name"]: g for g in cfg["managedNodeGroups"]}
    assert list(groups) == ["cpu-services"]
    cpu = groups["cpu-services"]
    assert cpu["desiredCapacity"] == 4
    assert cpu["volumeSize"] == 100
    assert cpu["instanceType"] == "c5.2xlarge"
    assert cpu["privateNetworking"] is True
    assert cpu["labels"] == {"role": "services"}
    [gpu_cfg] = _docs(_rendered("cluster-config"))
    for key in ("vpc", "iam", "addons", "availabilityZones", "metadata", "cloudWatch"):
        assert cfg[key] == gpu_cfg[key], key


def test_artf_containers_manifest_passes_through_unchanged():
    # The ARTF containers' EKS deployment is applied as rendered in both modes,
    # GPU tolerations included (they match no taint in learner mode).
    before = _rendered("artf-containers-deployment")
    assert transform("artf-containers-deployment", before) == before
    assert "artf-containers-deployment" not in learner_render._TRANSFORMS


@pytest.mark.parametrize("name", ["triton-deployment", "triton-hpa", "cluster-config",
                                  "orchestrator-deployment"])
def test_learner_render_keeps_document_kinds_and_names(name):
    before = _rendered(name)
    assert _kinds_and_names(transform(name, before)) == _kinds_and_names(before)


@pytest.mark.parametrize("name", ["triton-internal-nlb", "orchestrator-internal-nlb",
                                  "artf-containers-deployment"])
def test_other_manifests_pass_through_unchanged(name):
    text = _rendered(name)
    assert transform(name, text) == text


@pytest.mark.parametrize("name,anchor", [
    ("triton-deployment", '        nvidia.com/gpu: "present"\n'),
    ("triton-hpa", "  maxReplicas: 4\n"),
    ("orchestrator-deployment", "            - name: EKS_CLUSTER_NAME\n"),
    ("cluster-config", "  # GPU node group for Triton Inference Server.\n"),
])
def test_missing_anchor_stops_the_render_and_names_it(name, anchor):
    mutated = _rendered(name).replace(anchor, "")
    with pytest.raises(AnchorError) as err:
        transform(name, mutated)
    assert name in str(err.value)


def test_cli_exit_code_and_output(tmp_path):
    src = tmp_path / "in.yaml"
    dst = tmp_path / "out.yaml"
    src.write_text(_rendered("triton-hpa"))
    assert learner_render.main(["x", "triton-hpa", str(src), str(dst)]) == 0
    assert "nvidia.com/gpu" not in dst.read_text()
    src.write_text("kind: Nothing\n")
    assert learner_render.main(["x", "triton-hpa", str(src), str(tmp_path / "bad.yaml")]) == 1
    assert not (tmp_path / "bad.yaml").exists()
    assert learner_render.main(["x"]) == 2


def test_learner_render_has_no_dashes_or_emojis():
    text = (_SCRIPTS / "learner_render.py").read_text()
    assert "\u2014" not in text and "\u2013" not in text
    for name in ("triton-deployment", "triton-hpa", "cluster-config", "orchestrator-deployment"):
        added = _changed_lines(_rendered(name), transform(name, _rendered(name)))[1]
        assert all("\u2014" not in a and "\u2013" not in a for a in added)
