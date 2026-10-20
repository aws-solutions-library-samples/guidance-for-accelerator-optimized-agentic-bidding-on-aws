#!/usr/bin/env python3
"""Rewrite rendered deployment manifests for ``deploy.sh --learner-mode``.

Learner mode runs the stack with no GPUs: Triton Inference Server serves the same
models on the CPU node group (FIL on CPU for the Yield Optimizer floor and margin
models, ONNX Runtime on CPU for the Deal Scorer and Bid Pricer). The templates in
deployment/eks/ are not edited. deploy.sh renders them with its usual sed
substitutions and, only when LEARNER_MODE=1, passes the result through
``transform()`` here.

Each edit is an exact text replacement that must match a fixed number of times
(once, unless stated). A template that no longer contains an anchor stops the
deploy and names the anchor, so a template change cannot silently leave a GPU
request in a learner render.

Stdlib only: the deploy path does not install PyYAML.

Usage: learner_render.py <kind> <input> <output>
  kind is the template file name without .yaml (cluster-config,
  triton-deployment, triton-hpa, orchestrator-deployment, artf-containers-deployment);
  any other kind is copied through unchanged.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List

LEARNER_HEADER = (
    "# Rendered by deploy.sh --learner-mode: no GPU node group, Triton on CPU.\n"
    "# Learner mode is for learning; ARTF-compliant production workloads must not use it.\n"
)


class AnchorError(ValueError):
    """An expected anchor was not found the expected number of times."""


def _replace(text: str, old: str, new: str, kind: str, count: int = 1) -> str:
    found = text.count(old)
    if found != count:
        raise AnchorError(
            f"learner_render {kind}: expected {count} occurrence(s) of anchor "
            f"{old!r}, found {found}"
        )
    return text.replace(old, new)


def _cut_block(text: str, start: str, end: str, kind: str) -> str:
    """Remove from the line ``start`` up to, not including, the line ``end``."""
    for anchor in (start, end):
        if text.count(anchor) != 1:
            raise AnchorError(
                f"learner_render {kind}: expected 1 occurrence of anchor "
                f"{anchor!r}, found {text.count(anchor)}"
            )
    i = text.index(start)
    j = text.index(end)
    if j <= i:
        raise AnchorError(f"learner_render {kind}: anchor {end!r} precedes {start!r}")
    return text[:i] + text[j:]


def _cluster_config(text: str) -> str:
    kind = "cluster-config"
    text = _cut_block(
        text,
        "  # GPU node group for Triton Inference Server.\n",
        "  # CPU node group for orchestrator",
        kind,
    )
    text = _replace(text, "  - name: gpu-inference\n", "", kind, count=0)
    # Triton (2 vCPU request) now shares cpu-services with the ARTF containers and
    # the orchestrator, so one more node keeps their HPA headroom. The Triton image
    # is about 15 GB pulled, hence the larger root volume.
    text = _replace(text, "    desiredCapacity: 3\n", "    desiredCapacity: 4\n", kind)
    text = _replace(text, "    volumeSize: 50\n", "    volumeSize: 100\n", kind)
    text = _replace(
        text,
        "    privateNetworking: true   # see gpu-inference above\n",
        "    privateNetworking: true   # private subnets behind the NAT gateways\n",
        kind,
    )
    return LEARNER_HEADER + text


def _triton_deployment(text: str) -> str:
    kind = "triton-deployment"
    text = _replace(
        text,
        "      nodeSelector:\n"
        "        nvidia.com/gpu: \"present\"\n"
        "      tolerations:\n"
        "        - key: nvidia.com/gpu\n"
        "          operator: Exists\n"
        "          effect: NoSchedule\n",
        "      nodeSelector:\n"
        "        role: services\n",
        kind,
    )
    text = _replace(
        text,
        "          resources:\n"
        "            requests:\n"
        "              cpu: \"2\"\n"
        "              memory: \"8Gi\"\n"
        "              nvidia.com/gpu: \"1\"\n"
        "            limits:\n"
        "              cpu: \"4\"\n"
        "              memory: \"16Gi\"\n"
        "              nvidia.com/gpu: \"1\"\n",
        "          resources:\n"
        "            requests:\n"
        "              cpu: \"2\"\n"
        "              memory: \"4Gi\"\n"
        "            limits:\n"
        "              cpu: \"4\"\n"
        "              memory: \"8Gi\"\n",
        kind,
    )
    return LEARNER_HEADER + text


def _triton_hpa(text: str) -> str:
    kind = "triton-hpa"
    text = _replace(
        text,
        "    # Scale on GPU utilization\n"
        "    - type: Resource\n"
        "      resource:\n"
        "        name: nvidia.com/gpu\n"
        "        target:\n"
        "          type: Utilization\n"
        "          averageUtilization: 75\n",
        "",
        kind,
    )
    text = _replace(text, "  maxReplicas: 4\n", "  maxReplicas: 2\n", kind)
    return LEARNER_HEADER + text


def _orchestrator_deployment(text: str) -> str:
    kind = "orchestrator-deployment"
    anchor = "            - name: EKS_CLUSTER_NAME\n"
    return _replace(
        text,
        anchor,
        "            # Set only by deploy.sh --learner-mode. GET /v1/gpu/status reports it\n"
        "            # as deploymentMode, which the UI reads to show the learner notice.\n"
        "            - name: DEPLOYMENT_MODE\n"
        "              value: \"learner\"\n" + anchor,
        kind,
    )


def _artf_containers_deployment(text: str) -> str:
    # The ARTF containers hold no GPU; their only GPU reference is a toleration
    # that lets them schedule onto a tainted GPU node. Learner mode has no GPU
    # node, so the toleration is dropped from the rendered copy. Images, env,
    # ports, probes and node selector are untouched.
    kind = "artf-containers-deployment"
    return _replace(
        text,
        "      tolerations:\n"
        "        - key: nvidia.com/gpu\n"
        "          operator: Exists\n"
        "          effect: NoSchedule\n",
        "",
        kind,
        count=4,
    )


_TRANSFORMS: Dict[str, Callable[[str], str]] = {
    "cluster-config": _cluster_config,
    "triton-deployment": _triton_deployment,
    "triton-hpa": _triton_hpa,
    "orchestrator-deployment": _orchestrator_deployment,
    "artf-containers-deployment": _artf_containers_deployment,
}


def transform(kind: str, text: str) -> str:
    fn = _TRANSFORMS.get(kind)
    return fn(text) if fn else text


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        sys.stderr.write("usage: learner_render.py <kind> <input> <output>\n")
        return 2
    kind, src, dst = argv[1], argv[2], argv[3]
    try:
        with open(src, encoding="utf-8") as handle:
            text = handle.read()
        out = transform(kind, text)
    except (OSError, AnchorError) as exc:
        sys.stderr.write(f"{exc} (input: {src})\n")
        return 1
    with open(dst, "w", encoding="utf-8") as handle:
        handle.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
