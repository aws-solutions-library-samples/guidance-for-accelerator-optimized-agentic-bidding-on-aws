"""CodeBuild image builds pull nothing from Docker Hub anonymously.

CodeBuild egresses from shared addresses, so an anonymous Docker Hub pull can
fail on the rate limit another tenant used up ("toomanyrequests: You have
reached your unauthenticated pull rate limit"). A fresh r2v deploy failed that
way. These tests pin the three places a build pulled from Docker Hub:

- base images: every FROM names its registry (the python base comes from the
  ECR Public mirror of the Docker Official image);
- the buildx builder: the docker-container driver, which pulls moby/buildkit,
  is created only for a cross-compile;
- QEMU: only a cross-compile registers binfmt handlers (pinned in
  deployment/tests/test_deploy_phases.sh and the buildspec's HAS_ARM_IMAGE check).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "source"
BUILDSPEC = REPO_ROOT / "deployment" / "codebuild" / "buildspec.yml"

# Dockerfiles the CodeBuild buildspec builds.
CODEBUILD_DOCKERFILES = [
    SOURCE / "Dockerfile",
    SOURCE / "Dockerfile.agentcore",
    SOURCE / "Dockerfile.adaptive-bidding-agent",
    SOURCE / "Dockerfile.governance",
    SOURCE / "Dockerfile.orchestrator",
    SOURCE / "Dockerfile.optimizer",
    SOURCE / "triton" / "Dockerfile.triton-artf",
    SOURCE / "training" / "container" / "Dockerfile",
]

_FROM = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)", re.MULTILINE)


def _base_images(path: Path) -> list[str]:
    return _FROM.findall(path.read_text(encoding="utf-8"))


def _registry(image: str) -> str | None:
    """The registry host of an image reference, or None for a Docker Hub short name.

    Docker treats the first path component as a registry only when it contains
    a dot or a colon, or is "localhost"; otherwise the reference is on Docker Hub.
    """
    first = image.split("/", 1)[0]
    if "/" in image and ("." in first or ":" in first or first == "localhost"):
        return first
    return None


@pytest.mark.parametrize("dockerfile", CODEBUILD_DOCKERFILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_base_image_names_a_registry(dockerfile: Path):
    images = _base_images(dockerfile)
    assert images, f"{dockerfile} has no FROM line"
    docker_hub = [image for image in images if _registry(image) is None]
    assert not docker_hub, f"{dockerfile} pulls {docker_hub} from Docker Hub"


@pytest.mark.parametrize("dockerfile", CODEBUILD_DOCKERFILES[:5] + CODEBUILD_DOCKERFILES[6:7],
                         ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_python_base_is_the_ecr_public_mirror(dockerfile: Path):
    assert _base_images(dockerfile) == ["public.ecr.aws/docker/library/python:3.12-slim"]


def test_registry_parser_reads_docker_hub_short_names_as_docker_hub():
    assert _registry("python:3.12-slim") is None
    assert _registry("moby/buildkit:buildx-stable-1") is None
    assert _registry("public.ecr.aws/docker/library/python:3.12-slim") == "public.ecr.aws"
    assert _registry("nvcr.io/nvidia/nemo:24.07") == "nvcr.io"


def _pre_build_commands() -> list[str]:
    spec = yaml.safe_load(BUILDSPEC.read_text(encoding="utf-8"))
    return spec["phases"]["pre_build"]["commands"]


def test_buildspec_parses_and_creates_the_container_builder_only_for_a_cross_compile():
    commands = _pre_build_commands()
    builder = [c for c in commands if "buildx create" in c]
    assert len(builder) == 1, "exactly one pre_build command creates a buildx builder"
    block = builder[0]
    # The docker-container builder sits inside the NEEDS_ARM=1 branch...
    cross, _, native = block.partition("else")
    assert 'if [ "${NEEDS_ARM}" = "1" ]' in cross
    assert "--driver docker-container" in cross
    # ...and a native build selects the daemon's own BuildKit, which pulls no image.
    assert "docker buildx use default" in native
    assert "docker-container" not in native
    assert "export BUILDER" in block


def test_no_unconditional_docker_container_builder():
    for command in _pre_build_commands():
        if "docker-container" in command:
            assert "NEEDS_ARM" in command, "a docker-container builder is created outside the cross-compile branch"


def test_nemo_step_restores_the_selected_builder():
    text = BUILDSPEC.read_text(encoding="utf-8")
    assert 'docker buildx use "${BUILDER:-default}"' in text
    assert "docker buildx use multiarch 2>/dev/null || true" not in text
