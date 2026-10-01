"""Items 1.5 and 1.8: the feature-spec version is recorded, and enforced.

An ONNX graph whose inputs have the right names and the right widths is not
evidence that its producer and the container that will serve it mean the same
thing by each position. Change what position two carries, or change a vocabulary
size, and the engine compiles, loads, serves, and is wrong -- with no error
anywhere. The defect this repo already had was exactly that shape.

So each producer records the feature-spec version it built against in a manifest
beside the artifact, and the promotion path refuses an artifact it cannot
interpret. These tests cover both halves, plus the part that makes the refusal
useful: it has to reach the caller's response, not just a log line
(Item 1.8).

Where the gate lives, and why the optimizer does not import the spec: the Model
Optimizer compiles ONNX to TensorRT for every model in the repository and is not
the authority on any one model's feature contract. Its image is also built from
source/Dockerfile.optimizer plus source/optimizer alone, with no access to
source/shared. So the caller states the version it expects and the optimizer
enforces the manifest matches -- deploy.sh substitutes it into the bootstrap spec
from dlrm_features.FEATURE_SPEC_VERSION, so there is still one definition.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tarfile
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_DIR.parent

sys.path.insert(0, str(SOURCE_DIR))

from shared import dlrm_features  # noqa: E402

from optimizer.app import (  # noqa: E402
    MANIFEST_FILENAME,
    OptimizeRequestError,
    _read_manifest,
    _require_feature_spec_version,
    run_optimize,
)


# ---------------------------------------------------------------------------
# The contract in the spec module
# ---------------------------------------------------------------------------

class TestSpecModule:
    def test_the_optimizer_and_the_spec_agree_on_the_filename(self) -> None:
        """The optimizer restates the filename because it cannot import the spec —
        its image is built without source/shared. This is the assertion that keeps
        the two copies in step."""
        assert MANIFEST_FILENAME == dlrm_features.MANIFEST_FILENAME

    def test_supported_versions_contains_the_emitted_version(self) -> None:
        assert (
            dlrm_features.FEATURE_SPEC_VERSION
            in dlrm_features.SUPPORTED_FEATURE_SPEC_VERSIONS
        )

    def test_manifest_records_the_whole_vector_not_just_the_version(self) -> None:
        """A version number alone does not say what it means.

        Recording the columns and vocabularies makes a mismatch diagnosable from
        the artifact rather than from whichever commit produced it.
        """
        record = dlrm_features.manifest("dlrm_bid_shader")

        assert record["feature_spec_version"] == dlrm_features.FEATURE_SPEC_VERSION
        assert record["dense_columns"] == list(dlrm_features.DENSE_COLUMNS)
        assert record["categorical_columns"] == list(dlrm_features.CATEGORICAL_COLUMNS)
        assert record["vocab_sizes"] == dict(dlrm_features.VOCAB_SIZES)
        assert record["triton_input_names"] == [
            dlrm_features.TRITON_DENSE_INPUT,
            *dlrm_features.TRITON_CATEGORICAL_INPUTS,
        ]

    def test_manifest_is_json_serialisable(self) -> None:
        """It is written with json.dump by two producers."""
        json.dumps(dlrm_features.manifest("dlrm_bid_shader", producer="test"))

    def test_extra_fields_are_merged_not_nested(self) -> None:
        record = dlrm_features.manifest("dlrm_bid_shader", label_column="label")
        assert record["label_column"] == "label"


# ---------------------------------------------------------------------------
# Reading the manifest off an artifact
# ---------------------------------------------------------------------------

class _NotFound(Exception):
    """Shaped like botocore's 404 so the fetch treats it as a missing object."""

    response = {"Error": {"Code": "404"}}


def _write_artifact(directory: Path, manifest: dict | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    onnx = directory / "model.onnx"
    onnx.write_bytes(b"not-a-real-onnx")
    if manifest is not None:
        (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return onnx


class TestReadManifest:
    def test_reads_the_manifest_beside_the_onnx(self, tmp_path: Path) -> None:
        onnx = _write_artifact(tmp_path / "m", {"feature_spec_version": 1})

        assert _read_manifest(str(onnx), "s3://b/k")["feature_spec_version"] == 1

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        onnx = _write_artifact(tmp_path / "m", None)

        assert _read_manifest(str(onnx), "s3://b/k") is None

    def test_finds_a_nested_manifest(self, tmp_path: Path) -> None:
        """A SageMaker archive can nest the model inside a directory."""
        onnx = _write_artifact(tmp_path / "m", None)
        nested = tmp_path / "m" / "inner"
        nested.mkdir()
        (nested / MANIFEST_FILENAME).write_text(json.dumps({"feature_spec_version": 1}))

        assert _read_manifest(str(onnx), "s3://b/k")["feature_spec_version"] == 1

    def test_unparseable_manifest_is_refused_not_ignored(self, tmp_path: Path) -> None:
        """Treating unreadable provenance as absent provenance loses the signal."""
        onnx = _write_artifact(tmp_path / "m", None)
        (tmp_path / "m" / MANIFEST_FILENAME).write_text("{not json")

        with pytest.raises(OptimizeRequestError, match="could not be read"):
            _read_manifest(str(onnx), "s3://b/k")

    def test_non_object_manifest_is_refused(self, tmp_path: Path) -> None:
        onnx = _write_artifact(tmp_path / "m", None)
        (tmp_path / "m" / MANIFEST_FILENAME).write_text("[1, 2, 3]")

        with pytest.raises(OptimizeRequestError, match="not a JSON object"):
            _read_manifest(str(onnx), "s3://b/k")


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

class TestRequireFeatureSpecVersion:
    def test_accepts_a_matching_version_and_returns_it(self, tmp_path: Path) -> None:
        onnx = _write_artifact(
            tmp_path / "m", dlrm_features.manifest("dlrm_bid_shader")
        )

        declared = _require_feature_spec_version(
            str(onnx), "s3://b/k", dlrm_features.FEATURE_SPEC_VERSION
        )

        assert declared == dlrm_features.FEATURE_SPEC_VERSION

    def test_refuses_a_newer_version(self, tmp_path: Path) -> None:
        """The case the gate exists for: a model built against a vector this
        deployment cannot interpret."""
        onnx = _write_artifact(tmp_path / "m", {"feature_spec_version": 2})

        with pytest.raises(OptimizeRequestError) as exc:
            _require_feature_spec_version(str(onnx), "s3://b/k", 1)

        message = str(exc.value)
        assert "version 2" in message
        assert "version 1" in message
        # The refusal has to be actionable, not just a rejection.
        assert "Retrain" in message

    def test_refuses_an_older_version(self, tmp_path: Path) -> None:
        onnx = _write_artifact(tmp_path / "m", {"feature_spec_version": 1})

        with pytest.raises(OptimizeRequestError, match="version 1"):
            _require_feature_spec_version(str(onnx), "s3://b/k", 2)

    def test_refuses_an_artifact_with_no_manifest(self, tmp_path: Path) -> None:
        """Absent is refused, not waved through.

        An artifact with no manifest came from something whose feature contract is
        unknown -- which is the case this check exists for, not an exemption from
        it.
        """
        onnx = _write_artifact(tmp_path / "m", None)

        with pytest.raises(OptimizeRequestError) as exc:
            _require_feature_spec_version(str(onnx), "s3://b/k", 1)

        assert "no manifest.json" in str(exc.value)

    def test_refuses_a_manifest_with_no_version_field(self, tmp_path: Path) -> None:
        onnx = _write_artifact(tmp_path / "m", {"model_type": "dlrm_bid_shader"})

        with pytest.raises(OptimizeRequestError, match="declares no"):
            _require_feature_spec_version(str(onnx), "s3://b/k", 1)

    @pytest.mark.parametrize("bad", ["1", 1.0, None, True, [1]])
    def test_refuses_a_non_integer_version(self, tmp_path: Path, bad) -> None:
        """`True` matters: bool is an int subclass, so `declared == 1` is true for
        it and a JSON `true` would otherwise pass as version 1."""
        onnx = _write_artifact(tmp_path / "m", {"feature_spec_version": bad})

        with pytest.raises(OptimizeRequestError):
            _require_feature_spec_version(str(onnx), "s3://b/k", 1)


# ---------------------------------------------------------------------------
# Item 1.8 — the refusal reaches the response
# ---------------------------------------------------------------------------

class TestRefusalIsVisibleInTheResponse:
    """A refusal recorded only in a log line is a silent failure to the caller.

    `run_optimize` raises OptimizeRequestError, which the HTTP handler maps to a
    400 carrying the message, and which the --optimize-once Job entrypoint prints
    and exits non-zero on. These tests assert the refusal happens in run_optimize
    -- before the compile, and as a raised error rather than a returned result.
    """

    def _body(self, tmp_path: Path, manifest: dict | None, expected: int | None) -> dict:
        onnx = _write_artifact(tmp_path / "m", manifest)
        body = {
            "source_model_uri": onnx.as_uri().replace("file://", "s3://bucket"),
            "output_uri": "s3://bucket/out/model.plan",
            "model_name": "dlrm_bid_shader",
            "precision": "fp16",
        }
        if expected is not None:
            body["expected_feature_spec_version"] = expected
        return body, onnx

    def test_mismatch_raises_before_the_compile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body, onnx = self._body(tmp_path, {"feature_spec_version": 99}, 1)

        import optimizer.app as app

        built = []
        monkeypatch.setattr(app, "_resolve_onnx", lambda uri, workdir: str(onnx))
        monkeypatch.setattr(
            app, "_build_engine", lambda *a, **k: built.append(a)
        )
        monkeypatch.setattr(app, "_upload", lambda *a, **k: built.append("upload"))

        with pytest.raises(OptimizeRequestError) as exc:
            run_optimize(body)

        assert "version 99" in str(exc.value)
        assert built == [], "the engine was compiled despite the version mismatch"

    def test_a_matching_version_is_reported_in_the_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body, onnx = self._body(
            tmp_path,
            dlrm_features.manifest("dlrm_bid_shader"),
            dlrm_features.FEATURE_SPEC_VERSION,
        )

        import optimizer.app as app

        monkeypatch.setattr(app, "_resolve_onnx", lambda uri, workdir: str(onnx))
        monkeypatch.setattr(
            app,
            "_build_engine",
            lambda onnx_path, engine_path, *a, **k: Path(engine_path).write_bytes(b"x"),
        )
        monkeypatch.setattr(app, "_upload", lambda *a, **k: None)

        result = run_optimize(body)

        assert result["feature_spec_version"] == dlrm_features.FEATURE_SPEC_VERSION

    def test_no_expected_version_means_not_checked_not_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The result must distinguish "verified against 1" from "not checked".

        Reporting 1 for an unchecked compile would be a fabricated verification.
        """
        body, onnx = self._body(tmp_path, None, None)

        import optimizer.app as app

        monkeypatch.setattr(app, "_resolve_onnx", lambda uri, workdir: str(onnx))
        monkeypatch.setattr(
            app,
            "_build_engine",
            lambda onnx_path, engine_path, *a, **k: Path(engine_path).write_bytes(b"x"),
        )
        monkeypatch.setattr(app, "_upload", lambda *a, **k: None)

        result = run_optimize(body)

        assert result["feature_spec_version"] is None

    def test_a_non_integer_expected_version_is_refused(self, tmp_path: Path) -> None:
        body, _ = self._body(tmp_path, {"feature_spec_version": 1}, None)
        body["expected_feature_spec_version"] = "1"

        with pytest.raises(OptimizeRequestError, match="must be an integer"):
            run_optimize(body)

    def test_the_real_sagemaker_archive_shape_is_handled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrained model arrives as model.tar.gz, not a bare .onnx.

        This drives `_resolve_onnx`'s own extraction rather than stubbing it, so the
        gate is shown to work on the shape a promotion actually carries: SM_MODEL_DIR
        tarred up, with manifest.json and metrics.json beside model.onnx.
        """
        import optimizer.app as app

        staging = tmp_path / "opt-ml-model"
        staging.mkdir()
        (staging / "model.onnx").write_bytes(b"not-a-real-onnx")
        (staging / "metrics.json").write_text("{}")
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(dlrm_features.manifest("dlrm_bid_shader", producer="train.py"))
        )
        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for entry in sorted(staging.iterdir()):
                tar.add(entry, arcname=entry.name)

        monkeypatch.setattr(app, "_download", lambda uri, dest: __import__(
            "shutil"
        ).copyfile(archive, dest))
        monkeypatch.setattr(
            app,
            "_build_engine",
            lambda onnx_path, engine_path, *a, **k: Path(engine_path).write_bytes(b"x"),
        )
        monkeypatch.setattr(app, "_upload", lambda *a, **k: None)

        result = run_optimize(
            {
                "source_model_uri": "s3://bucket/training/model.tar.gz",
                "output_uri": "s3://bucket/out/model.plan",
                "model_name": "dlrm_bid_shader",
                "precision": "fp16",
                "expected_feature_spec_version": dlrm_features.FEATURE_SPEC_VERSION,
            }
        )

        assert result["feature_spec_version"] == dlrm_features.FEATURE_SPEC_VERSION

    def test_a_raw_onnx_source_fetches_its_sibling_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression a live bootstrap run found.

        A genesis source is a bare `onnx-source/<model>/model.onnx`, and its manifest
        is a SIBLING S3 OBJECT — there is no archive carrying it. The first version of
        this gate only looked on the local filesystem beside the downloaded file, so
        it refused every raw-.onnx source as unmanifested, including ones whose
        manifest was sitting next to them in the bucket. The real bootstrap Job failed
        with "has no manifest.json" against a prefix that did have one.

        This drives `_resolve_onnx` for real against a fake S3 that holds both objects.
        """
        import optimizer.app as app

        objects = {
            "onnx-source/dlrm_bid_shader/model.onnx": b"not-a-real-onnx",
            "onnx-source/dlrm_bid_shader/manifest.json": json.dumps(
                dlrm_features.manifest("dlrm_bid_shader")
            ).encode(),
        }

        class _FakeS3:
            def download_file(self, bucket, key, dest):
                if key not in objects:
                    raise _NotFound()
                Path(dest).write_bytes(objects[key])

        monkeypatch.setattr(app, "_s3", lambda: _FakeS3())
        monkeypatch.setattr(
            app,
            "_build_engine",
            lambda onnx_path, engine_path, *a, **k: Path(engine_path).write_bytes(b"x"),
        )
        monkeypatch.setattr(app, "_upload", lambda *a, **k: None)

        result = run_optimize(
            {
                "source_model_uri": "s3://bucket/onnx-source/dlrm_bid_shader/model.onnx",
                "output_uri": "s3://bucket/out/model.plan",
                "model_name": "dlrm_bid_shader",
                "precision": "fp16",
                "expected_feature_spec_version": dlrm_features.FEATURE_SPEC_VERSION,
            }
        )

        assert result["feature_spec_version"] == dlrm_features.FEATURE_SPEC_VERSION

    def test_a_raw_onnx_with_no_sibling_manifest_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely absent sibling still refuses — the fix must not accept blindly."""
        import optimizer.app as app

        class _FakeS3:
            def download_file(self, bucket, key, dest):
                if key.endswith(MANIFEST_FILENAME):
                    raise _NotFound()
                Path(dest).write_bytes(b"not-a-real-onnx")

        monkeypatch.setattr(app, "_s3", lambda: _FakeS3())
        compiled = []
        monkeypatch.setattr(app, "_build_engine", lambda *a, **k: compiled.append(a))

        with pytest.raises(OptimizeRequestError, match="no manifest.json"):
            run_optimize(
                {
                    "source_model_uri": "s3://bucket/onnx-source/dlrm_bid_shader/model.onnx",
                    "output_uri": "s3://bucket/out/model.plan",
                    "model_name": "dlrm_bid_shader",
                    "precision": "fp16",
                    "expected_feature_spec_version": 1,
                }
            )

        assert compiled == []

    def test_an_s3_error_that_is_not_a_404_is_not_reported_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AccessDenied on the manifest is a different fact from no manifest.

        Collapsing the two would turn a permissions problem into a false claim that
        the artifact was produced without provenance.
        """
        import optimizer.app as app

        class _Denied(Exception):
            response = {"Error": {"Code": "AccessDenied"}}

        class _FakeS3:
            def download_file(self, bucket, key, dest):
                if key.endswith(MANIFEST_FILENAME):
                    raise _Denied()
                Path(dest).write_bytes(b"not-a-real-onnx")

        monkeypatch.setattr(app, "_s3", lambda: _FakeS3())

        with pytest.raises(_Denied):
            run_optimize(
                {
                    "source_model_uri": "s3://bucket/onnx-source/dlrm_bid_shader/model.onnx",
                    "output_uri": "s3://bucket/out/model.plan",
                    "model_name": "dlrm_bid_shader",
                    "precision": "fp16",
                    "expected_feature_spec_version": 1,
                }
            )

    def test_an_archive_with_no_manifest_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case a training image built before manifests would produce."""
        import optimizer.app as app

        staging = tmp_path / "opt-ml-model"
        staging.mkdir()
        (staging / "model.onnx").write_bytes(b"not-a-real-onnx")
        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staging / "model.onnx", arcname="model.onnx")

        monkeypatch.setattr(app, "_download", lambda uri, dest: __import__(
            "shutil"
        ).copyfile(archive, dest))
        compiled = []
        monkeypatch.setattr(app, "_build_engine", lambda *a, **k: compiled.append(a))

        with pytest.raises(OptimizeRequestError, match="no manifest.json"):
            run_optimize(
                {
                    "source_model_uri": "s3://bucket/training/model.tar.gz",
                    "output_uri": "s3://bucket/out/model.plan",
                    "model_name": "dlrm_bid_shader",
                    "precision": "fp16",
                    "expected_feature_spec_version": 1,
                }
            )

        assert compiled == []


# ---------------------------------------------------------------------------
# Both producers write a manifest, and deploy.sh carries the version through
# ---------------------------------------------------------------------------

class TestProducersWriteAManifest:
    def test_the_genesis_exporter_writes_one_beside_the_onnx(
        self, tmp_path: Path
    ) -> None:
        """A real export, then read the file off disk."""
        sys.path.insert(0, str(SOURCE_DIR / "triton"))
        import importlib

        export_models = importlib.import_module("export_models")
        export_models.export_dlrm(str(tmp_path))

        written = json.loads(
            (tmp_path / "dlrm_bid_shader" / "1" / MANIFEST_FILENAME).read_text()
        )
        assert written["feature_spec_version"] == dlrm_features.FEATURE_SPEC_VERSION
        assert written["triton_input_names"] == [
            dlrm_features.TRITON_DENSE_INPUT,
            *dlrm_features.TRITON_CATEGORICAL_INPUTS,
        ]
        # These weights are seeded init, and the manifest says so rather than
        # leaving a reader to infer it from the path.
        assert written["weights"] == "seeded_initialisation"

    def test_the_trainer_writes_one(self) -> None:
        """Asserted against the source, because running train.py's main() needs a
        SageMaker channel, a GPU-shaped image and a Parquet dataset. The import
        path it uses is covered by tests/test_training_image_packaging.py.
        """
        text = (SOURCE_DIR / "training" / "container" / "train.py").read_text()
        assert "dlrm_features.MANIFEST_FILENAME" in text
        assert "dlrm_features.manifest(" in text
        # The objective and the label it selects both travel into the manifest, so a
        # reader can tell what a set of weights was trained to predict.
        assert "objective=objective," in text
        assert "label_column=label_column," in text

    def test_the_label_column_has_one_definition(self) -> None:
        """The manifest must not be able to claim a different label than the run
        actually trained on."""
        text = (SOURCE_DIR / "training" / "container" / "train.py").read_text()
        assert 'df["label"]' not in text
        assert "df[label_column]" in text


class TestBootstrapSpecAndDeployScript:
    def test_the_spec_declares_an_expected_version_for_both_dlrm_entries(self) -> None:
        raw = (REPO_ROOT / "deployment" / "optimizer-bootstrap.json").read_text()
        spec = json.loads(
            raw.replace("__FEATURE_SPEC_VERSION__", "1").replace(
                "__MODEL_BUCKET__", "bucket"
            )
        )
        by_name = {m["model_name"]: m for m in spec["models"]}

        for name in ("dlrm_bid_shader", "dlrm_bid_shader_canary"):
            assert by_name[name]["expected_feature_spec_version"] == 1, (
                f"{name} would be compiled with no feature-contract check"
            )

    def test_ncf_declares_none(self) -> None:
        """NCF's vector is not versioned; claiming a DLRM version for it would be
        a check that means nothing."""
        raw = (REPO_ROOT / "deployment" / "optimizer-bootstrap.json").read_text()
        spec = json.loads(
            raw.replace("__FEATURE_SPEC_VERSION__", "1").replace(
                "__MODEL_BUCKET__", "bucket"
            )
        )
        by_name = {m["model_name"]: m for m in spec["models"]}

        assert "expected_feature_spec_version" not in by_name["ncf_deal_manager"]

    def test_the_version_is_a_placeholder_not_a_literal(self) -> None:
        """Written into the JSON, it would be a second definition to drift."""
        raw = (REPO_ROOT / "deployment" / "optimizer-bootstrap.json").read_text()
        assert "__FEATURE_SPEC_VERSION__" in raw
        assert not re.search(r'"expected_feature_spec_version":\s*\d', raw)

    def test_deploy_sh_substitutes_it_from_the_spec_module(self) -> None:
        text = (REPO_ROOT / "deployment" / "deploy.sh").read_text()
        assert "__FEATURE_SPEC_VERSION__" in text
        assert "from shared import dlrm_features" in text
        assert "dlrm_features.FEATURE_SPEC_VERSION" in text

    def test_deploy_sh_refuses_an_unsubstituted_placeholder(self) -> None:
        """An unsubstituted placeholder would upload invalid JSON, and the optimizer
        would fail to parse its own spec at a point far from the cause."""
        text = (REPO_ROOT / "deployment" / "deploy.sh").read_text()
        assert "still has an unsubstituted __FEATURE_SPEC_VERSION__" in text

    def test_deploy_sh_uploads_the_manifest_next_to_the_onnx(self) -> None:
        """The gate reads it from there; uploading the ONNX alone would make every
        bootstrap refuse."""
        text = (REPO_ROOT / "deployment" / "deploy.sh").read_text()
        assert "onnx-source/${m}/manifest.json" in text
