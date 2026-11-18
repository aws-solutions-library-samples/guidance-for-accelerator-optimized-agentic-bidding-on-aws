"""Python base images come from the ECR Public mirror of the Docker official image.

Docker Hub rate-limits anonymous pulls, which fails CodeBuild and Workshop Studio
builds that share an egress IP. public.ecr.aws/docker/library/ serves the same
official image without that limit. The NVIDIA NGC bases (nvcr.io) are out of scope.
"""
from __future__ import annotations

import re
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1]
_FROM = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)", re.MULTILINE)


def _dockerfiles():
    return sorted(p for p in _SOURCE.rglob("Dockerfile*")
                  if p.is_file() and "node_modules" not in p.parts)


def test_python_bases_use_the_ecr_public_mirror():
    bare = []
    for path in _dockerfiles():
        for image in _FROM.findall(path.read_text()):
            if image.startswith("python:"):
                bare.append(f"{path.relative_to(_SOURCE)}: {image}")
    assert bare == []


def test_the_mirrored_python_base_is_used_where_expected():
    expected = {
        "Dockerfile", "Dockerfile.adaptive-bidding-agent", "Dockerfile.agentcore",
        "Dockerfile.demand", "Dockerfile.governance", "Dockerfile.orchestrator",
        "triton/Dockerfile.triton-artf",
    }
    found = {
        str(p.relative_to(_SOURCE)) for p in _dockerfiles()
        if "public.ecr.aws/docker/library/python:3.12-slim" in _FROM.findall(p.read_text())
    }
    assert expected <= found


def test_ngc_bases_are_unchanged():
    assert "nvcr.io/nvidia/tensorrt:24.08-py3" in _FROM.findall(
        (_SOURCE / "Dockerfile.optimizer").read_text())
    assert "nvcr.io/nvidia/nemo:24.07" in _FROM.findall(
        (_SOURCE / "training" / "container" / "Dockerfile").read_text())
