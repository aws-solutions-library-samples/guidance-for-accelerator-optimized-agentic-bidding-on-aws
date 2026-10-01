"""The training image must actually contain the shared feature spec.

Item 1.4 of the dlrm-shading-correctness task list moved the DLRM feature
vector out of train.py and into shared/dlrm_features.py, so the trainer and the
bid shader build the same vector from the same code. That only holds if the file
reaches the training container.

It does not reach it for free. The training image's Docker build context is
`source/training/container/`, so its Dockerfile cannot COPY from
`source/shared/`. `source/training/stage_shared.sh` copies the modules in, and
the two build paths run it before building. These tests pin every link in that
chain, because each one fails silently on its own:

  - a missing COPY in the Dockerfile fails at SageMaker runtime, not build time;
  - a staging step that runs AFTER the rebuild hash leaves the old image in
    place while reporting the source unchanged;
  - a staged copy that has drifted from source/shared/ is the original
    two-definitions defect, reintroduced.

The `shared/` directory under the build context is generated and gitignored, so
these tests stage it themselves rather than assume a previous build left it
behind.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_DIR.parent
CONTAINER_DIR = SOURCE_DIR / "training" / "container"
STAGE_SCRIPT = SOURCE_DIR / "training" / "stage_shared.sh"
STAGE_DIR = CONTAINER_DIR / "shared"
DOCKERFILE = CONTAINER_DIR / "Dockerfile"

# Modules the trainer imports from `shared`. Kept here as the test's own
# statement of the contract rather than parsed out of the script, so a silent
# edit to the script's list is a test failure and not a redefinition.
EXPECTED_STAGED = (
    "__init__.py",
    "dlrm_features.py",
    "onnx_compat.py",
    "shading_policy.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def staged() -> Path:
    """Run the staging script and return the staged directory."""
    result = subprocess.run(
        ["sh", str(STAGE_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stage_shared.sh failed (exit {result.returncode}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return STAGE_DIR


class TestStagingScript:
    def test_script_exists_and_is_executable(self) -> None:
        assert STAGE_SCRIPT.is_file(), f"missing {STAGE_SCRIPT}"
        # Invoked as `sh <script>` by both build paths, so the executable bit is
        # not load-bearing -- but a developer running it directly expects it.
        assert STAGE_SCRIPT.stat().st_mode & 0o111, (
            f"{STAGE_SCRIPT} is not executable"
        )

    def test_stages_every_expected_module(self, staged: Path) -> None:
        for module in EXPECTED_STAGED:
            assert (staged / module).is_file(), (
                f"stage_shared.sh did not stage {module}"
            )

    def test_staging_produces_byte_identical_copies(self, staged: Path) -> None:
        """What lands in the image is the file source/shared/ holds, unmodified."""
        for module in EXPECTED_STAGED:
            original = SOURCE_DIR / "shared" / module
            assert _sha256(staged / module) == _sha256(original), (
                f"staged {module} differs from source/shared/{module}. "
                "The staged tree is generated output; edit source/shared/."
            )

    def test_overwrites_a_drifted_copy(self, staged: Path) -> None:
        """A stale staged copy must not survive into the image.

        This is the property that makes the arrangement safe rather than merely
        convenient. The staged directory is gitignored, but nothing stops a
        checkout from carrying an old one -- an earlier build, a restored
        backup, a developer who edited the generated file by hand. Because both
        build paths re-stage unconditionally, whatever is there is replaced.
        Drop the staging call from a build path and that path starts shipping
        whatever copy happens to be on disk.
        """
        target = staged / "dlrm_features.py"
        original_digest = _sha256(SOURCE_DIR / "shared" / "dlrm_features.py")

        target.write_text("# a stale copy from an earlier feature spec\n")
        assert _sha256(target) != original_digest

        subprocess.run(["sh", str(STAGE_SCRIPT)], capture_output=True, check=True)
        assert _sha256(target) == original_digest

    def test_is_idempotent(self, staged: Path) -> None:
        before = {m: _sha256(staged / m) for m in EXPECTED_STAGED}
        subprocess.run(["sh", str(STAGE_SCRIPT)], capture_output=True, check=True)
        after = {m: _sha256(staged / m) for m in EXPECTED_STAGED}
        assert before == after

    def test_refuses_when_a_module_is_missing(self, tmp_path: Path) -> None:
        """Fail at build time rather than inside a SageMaker training job."""
        # A tree shaped like source/ whose shared/ holds everything EXCEPT the
        # feature spec, so the script's relative lookups all resolve and the
        # missing spec is the only thing that can fail.
        fake = tmp_path / "source"
        (fake / "shared").mkdir(parents=True)
        (fake / "shared" / "__init__.py").write_text('"""stub."""\n')
        (fake / "training" / "container").mkdir(parents=True)
        shim = fake / "training" / "stage_shared.sh"
        shim.write_bytes(STAGE_SCRIPT.read_bytes())

        result = subprocess.run(
            ["sh", str(shim)], capture_output=True, text=True, check=False
        )
        assert result.returncode != 0, (
            "staging a source/shared/ with no dlrm_features.py succeeded; the "
            "image would be built without the feature spec"
        )
        assert "dlrm_features.py" in result.stderr
        assert not (fake / "training" / "container" / "shared" / "dlrm_features.py").exists()


class TestDockerfile:
    def test_copies_the_staged_directory(self) -> None:
        text = DOCKERFILE.read_text()
        assert "COPY shared/" in text, (
            "Dockerfile does not COPY the staged shared/ directory; "
            "`from shared import dlrm_features` will fail at runtime"
        )

    def test_copies_into_the_import_root(self) -> None:
        """ENTRYPOINT runs /opt/ml/code/train.py, so sys.path[0] is /opt/ml/code."""
        text = DOCKERFILE.read_text()
        assert "COPY shared/ /opt/ml/code/shared/" in text
        assert "/opt/ml/code/train.py" in text


class TestBuildPathsStageBeforeBuilding:
    """Both paths must stage, and the local path must stage before it hashes."""

    def test_local_build_stages_before_hashing(self) -> None:
        script = (REPO_ROOT / "deployment" / "deploy_closed_loop.sh").read_text()
        lines = script.splitlines()

        stage_at = next(
            (i for i, ln in enumerate(lines) if "stage_shared.sh" in ln), None
        )
        assert stage_at is not None, (
            "deploy_closed_loop.sh never runs stage_shared.sh; a local build "
            "would produce a training image with no shared/ directory"
        )

        hash_at = next(
            (i for i, ln in enumerate(lines) if "_nemo_source_hash" in ln and "=" in ln),
            None,
        )
        assert hash_at is not None, "could not find the NEMO_SRC_HASH assignment"
        assert stage_at < hash_at, (
            "stage_shared.sh runs AFTER the rebuild hash is computed. The staged "
            "files live inside the hashed directory, so a change to "
            "source/shared/dlrm_features.py would not change src-<hash>, the "
            "rebuild would be skipped, and SageMaker would keep running the "
            "previous feature spec."
        )

    def test_remote_build_stages_before_docker_build(self) -> None:
        spec = (
            REPO_ROOT / "deployment" / "codebuild" / "buildspec.yml"
        ).read_text()
        lines = spec.splitlines()

        stage_at = next(
            (i for i, ln in enumerate(lines) if "stage_shared.sh" in ln), None
        )
        assert stage_at is not None, (
            "buildspec.yml never runs stage_shared.sh; a remote build would "
            "produce a training image with no shared/ directory"
        )

        build_at = next(
            (
                i
                for i, ln in enumerate(lines)
                if "docker build" in ln and "training/container" in ln
            ),
            None,
        )
        assert build_at is not None, "could not find the training image build"
        assert stage_at < build_at


class TestTrainerImportsTheSharedSpec:
    def test_train_py_imports_from_shared(self) -> None:
        text = (CONTAINER_DIR / "train.py").read_text()
        assert "from shared import dlrm_features" in text

    def test_train_py_defines_no_feature_vector_of_its_own(self) -> None:
        """The local copies removed in Item 1.4 must not come back."""
        text = (CONTAINER_DIR / "train.py").read_text()
        for gone in (
            "_DLRM_DENSE_COLUMNS",
            "_DLRM_SPARSE_COLUMNS",
            "_DLRM_VOCAB_SIZE",
            "def _hash_to_idx",
        ):
            assert gone not in text, (
                f"{gone} is back in train.py. The feature vector is defined "
                "once, in source/shared/dlrm_features.py."
            )

    def test_import_resolves_the_way_the_image_resolves_it(
        self, staged: Path
    ) -> None:
        """Import `shared` with only the container dir on the path.

        In the image, train.py is run as `python /opt/ml/code/train.py`, so
        sys.path[0] is /opt/ml/code -- a flat directory holding train.py,
        reward.py, models/ and the staged shared/. Nothing above it is
        importable. This reproduces that: if the staged package cannot be
        imported from the container directory alone, the training job fails on
        its first line, whatever the repo-root layout allows locally.

        train.py itself is not imported here -- it needs torch, pandas and
        numpy, which the NeMo base image supplies and this test environment
        need not.

        The `shared` binding is saved and restored by hand. Importing it from
        the container directory binds the package to a directory holding only
        dlrm_features.py, so leaving that binding in sys.modules would make
        every later `shared.feedback_models` import in the session fail.
        """
        import importlib

        prefix = ("shared", "shared.")
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name == prefix[0] or name.startswith(prefix[1])
        }
        saved_path = list(sys.path)
        try:
            for name in saved:
                del sys.modules[name]
            sys.path.insert(0, str(CONTAINER_DIR))

            shared_pkg = importlib.import_module("shared")
            resolved = Path(shared_pkg.__file__ or "").resolve()
            assert resolved == (staged / "__init__.py").resolve(), (
                f"`shared` resolved to {resolved}, not the staged copy. This "
                "test is only meaningful if it exercises the container layout."
            )

            spec_module = importlib.import_module("shared.dlrm_features")

            # A smoke call, so the assertion is "the trainer's feature builder
            # runs from the staged copy" rather than "the file is present".
            dense, categorical = spec_module.build_from_row(
                {
                    "bid_floor": 2.5,
                    "hour_of_day": 14,
                    "day_of_week": 5,
                    "has_video": True,
                    "site_domain": "example.com",
                    "device_type": 2,
                    "geo_country": "USA",
                }
            )
            flat = spec_module.flatten(dense, categorical)
            assert len(flat) == spec_module.FEATURE_WIDTH
        finally:
            for name in [
                n
                for n in sys.modules
                if n == prefix[0] or n.startswith(prefix[1])
            ]:
                del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = saved_path


class TestEveryImportedSharedModuleIsStaged:
    """The trainer must not import a `shared` module the image does not contain.

    This is the expensive failure mode, and it is invisible locally: the module
    resolves fine from `source/`, the image builds and pushes clean, and the
    ImportError only appears inside SageMaker after an instance provision and a
    multi-gigabyte image pull. It cost a ~25 minute round trip per missing module,
    twice in one session -- `onnx_compat` and then `shading_policy`.

    `stage_shared.sh` now refuses to build when this holds, which catches it at
    build time. This asserts the same property in CI, where it is cheaper still,
    and parses the imports with `ast` rather than repeating the script's grep.
    """

    @staticmethod
    def _imported_shared_modules() -> set[str]:
        found: set[str] = set()
        for path in CONTAINER_DIR.rglob("*.py"):
            # The staged copy is generated from the very list under test.
            if STAGE_DIR in path.parents or path == STAGE_DIR:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "shared":
                        # from shared import a, b
                        found.update(alias.name for alias in node.names)
                    elif node.module and node.module.startswith("shared."):
                        # from shared.x import y
                        found.add(node.module.split(".", 1)[1].split(".")[0])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("shared."):
                            found.add(alias.name.split(".", 1)[1].split(".")[0])
        return found

    def test_the_trainer_imports_at_least_one_shared_module(self) -> None:
        """Guards the guard: if the parse silently found nothing, the assertion
        below would pass for the wrong reason."""
        assert self._imported_shared_modules(), (
            "parsed no `shared` imports out of the training build context, which "
            "means this test can no longer detect an unstaged module"
        )

    def test_every_imported_module_is_staged(self) -> None:
        imported = self._imported_shared_modules()
        missing = sorted(
            name for name in imported if f"{name}.py" not in EXPECTED_STAGED
        )
        assert not missing, (
            "the trainer imports shared modules that are not staged into the "
            "image: %s. Add them to STAGED_MODULES in stage_shared.sh and to "
            "EXPECTED_STAGED here." % ", ".join(f"{m}.py" for m in missing)
        )

    def test_the_staging_script_lists_them_too(self) -> None:
        """EXPECTED_STAGED and the script's own list must agree.

        They are maintained separately on purpose, so this is the assertion that
        keeps the duplication honest.
        """
        text = STAGE_SCRIPT.read_text()
        line = next(
            (ln for ln in text.splitlines() if ln.startswith("STAGED_MODULES=")),
            None,
        )
        assert line is not None, "stage_shared.sh has no STAGED_MODULES assignment"
        listed = set(line.split("=", 1)[1].strip().strip('"').split())
        assert listed == set(EXPECTED_STAGED), (
            "stage_shared.sh stages %s but this test expects %s"
            % (sorted(listed), sorted(EXPECTED_STAGED))
        )
