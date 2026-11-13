"""Optimizer bootstrap rebuild rule for the Deal Scorer engine (source/optimizer/app.py).

The Deal Scorer is retrained at deploy time, so its bootstrap entry carries
rebuild_on_source_change. Engines carry S3 metadata source-fingerprint and
provisioned-by. The bootstrap rebuilds an engine whose source-fingerprint differs
from the fingerprint in the source's manifest.json when provisioned-by is
bootstrap or absent, and leaves an engine a promotion wrote alone.

S3 is an in-memory fake keyed by (bucket, key) holding bytes and user metadata;
trtexec is replaced by a function that writes a marker engine.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import optimizer.app as app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

BUCKET = "models"
SOURCE_KEY = "onnx-source/ncf_deal_manager/model.onnx"
MANIFEST_KEY = "onnx-source/ncf_deal_manager/manifest.json"
ENGINE_KEY = "triton-models/ncf_deal_manager_stable/1/model.plan"
OLD_FP = "a" * 64
NEW_FP = "b" * 64


class _NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class _Denied(Exception):
    response = {"Error": {"Code": "AccessDenied"}}


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.uploads: list[tuple[str, dict[str, str]]] = []
        self.deny_head: set[str] = set()

    def put(self, key: str, body: bytes, metadata: dict[str, str] | None = None) -> None:
        self.objects[(BUCKET, key)] = (body, dict(metadata or {}))

    def download_file(self, bucket, key, dest):
        if (bucket, key) not in self.objects:
            raise _NotFound()
        Path(dest).write_bytes(self.objects[(bucket, key)][0])

    def head_object(self, Bucket, Key):
        if Key in self.deny_head:
            raise _Denied()
        if (Bucket, Key) not in self.objects:
            raise _NotFound()
        body, meta = self.objects[(Bucket, Key)]
        return {"ContentLength": len(body), "Metadata": dict(meta)}

    def upload_file(self, src, bucket, key, ExtraArgs=None):
        meta = dict((ExtraArgs or {}).get("Metadata") or {})
        self.objects[(bucket, key)] = (Path(src).read_bytes(), meta)
        self.uploads.append((key, meta))


def _entry(rebuild: bool = True) -> dict:
    entry = {
        "model_name": "ncf_deal_manager",
        "source_model_uri": f"s3://{BUCKET}/{SOURCE_KEY}",
        "output_uri": f"s3://{BUCKET}/{ENGINE_KEY}",
        "precision": "fp16",
        "input_profiles": {
            "user_ids": {"min": [1], "opt": [16], "max": [128]},
            "item_ids": {"min": [1], "opt": [16], "max": [128]},
        },
    }
    if rebuild:
        entry["rebuild_on_source_change"] = True
    return entry


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    fake = FakeS3()
    fake.put(SOURCE_KEY, b"onnx-bytes")
    fake.put(MANIFEST_KEY, json.dumps({"model_type": "ncf_deal_manager", "fingerprint": NEW_FP}).encode())
    monkeypatch.setattr(app, "_s3", lambda: fake)
    builds: list[str] = []

    def _fake_build(onnx_path, engine_path, *a, **k):
        builds.append(onnx_path)
        Path(engine_path).write_bytes(b"engine-from-" + Path(onnx_path).read_bytes())

    monkeypatch.setattr(app, "_build_engine", _fake_build)
    fake.builds = builds  # type: ignore[attr-defined]
    return fake


# ---------------------------------------------------------------------------
# The four cases of the rule
# ---------------------------------------------------------------------------

def test_changed_fingerprint_on_a_bootstrap_engine_rebuilds(s3: FakeS3) -> None:
    s3.put(ENGINE_KEY, b"old-engine", {"source-fingerprint": OLD_FP, "provisioned-by": "bootstrap"})

    app._bootstrap_one(_entry())

    assert len(s3.builds) == 1
    body, meta = s3.objects[(BUCKET, ENGINE_KEY)]
    assert body == b"engine-from-onnx-bytes"
    assert meta == {"source-fingerprint": NEW_FP, "provisioned-by": "bootstrap"}


def test_an_engine_with_no_metadata_counts_as_bootstrap_and_rebuilds(s3: FakeS3) -> None:
    """An engine written before the metadata existed: only the bootstrap wrote engines then."""
    s3.put(ENGINE_KEY, b"old-engine", {})

    app._bootstrap_one(_entry())

    assert len(s3.builds) == 1
    assert s3.objects[(BUCKET, ENGINE_KEY)][1]["source-fingerprint"] == NEW_FP


def test_a_promoted_engine_is_never_rebuilt(s3: FakeS3, caplog: pytest.LogCaptureFixture) -> None:
    s3.put(ENGINE_KEY, b"promoted-engine", {"provisioned-by": "promotion"})

    with caplog.at_level("INFO", logger="optimizer"):
        app._bootstrap_one(_entry())

    assert s3.builds == []
    assert s3.uploads == []
    assert s3.objects[(BUCKET, ENGINE_KEY)][0] == b"promoted-engine"
    assert "provisioned by promotion; not rebuilding it" in caplog.text


def test_matching_fingerprints_skip_the_rebuild(s3: FakeS3) -> None:
    s3.put(ENGINE_KEY, b"current-engine", {"source-fingerprint": NEW_FP, "provisioned-by": "bootstrap"})

    app._bootstrap_one(_entry())

    assert s3.builds == []
    assert s3.uploads == []


# ---------------------------------------------------------------------------
# Around the rule
# ---------------------------------------------------------------------------

def test_a_missing_engine_is_built_with_its_provenance(s3: FakeS3) -> None:
    app._bootstrap_one(_entry())

    assert s3.uploads == [(ENGINE_KEY, {"source-fingerprint": NEW_FP, "provisioned-by": "bootstrap"})]


def test_an_unknown_writer_is_left_alone(s3: FakeS3) -> None:
    s3.put(ENGINE_KEY, b"other-engine", {"source-fingerprint": OLD_FP, "provisioned-by": "manual"})

    app._bootstrap_one(_entry())

    assert s3.builds == []


def test_a_source_with_no_fingerprint_keeps_the_engine(s3: FakeS3) -> None:
    s3.put(MANIFEST_KEY, json.dumps({"model_type": "ncf_deal_manager"}).encode())
    s3.put(ENGINE_KEY, b"old-engine", {"source-fingerprint": OLD_FP, "provisioned-by": "bootstrap"})

    app._bootstrap_one(_entry())

    assert s3.builds == []


def test_an_unreadable_engine_head_is_raised_not_rebuilt(s3: FakeS3) -> None:
    """AccessDenied is not "no engine": rebuilding on it could overwrite a promotion."""
    s3.put(ENGINE_KEY, b"promoted-engine", {"provisioned-by": "promotion"})
    s3.deny_head.add(ENGINE_KEY)

    with pytest.raises(_Denied):
        app._bootstrap_one(_entry())

    assert s3.builds == []


def test_entries_without_the_flag_keep_skip_if_present(s3: FakeS3) -> None:
    s3.put(ENGINE_KEY, b"old-engine", {"source-fingerprint": OLD_FP, "provisioned-by": "bootstrap"})

    app._bootstrap_one(_entry(rebuild=False))

    assert s3.builds == []


def test_a_promotion_marks_its_engine(s3: FakeS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"retrained")
    monkeypatch.setattr(app, "_resolve_onnx", lambda uri, workdir: str(onnx))

    app.run_optimize(
        {
            "source_model_uri": f"s3://{BUCKET}/training/model.tar.gz",
            "output_uri": f"s3://{BUCKET}/{ENGINE_KEY}",
            "model_name": "ncf_deal_manager",
            "precision": "fp16",
        }
    )

    assert s3.uploads == [(ENGINE_KEY, {"provisioned-by": "promotion"})]


def test_after_a_promotion_the_next_bootstrap_leaves_it(s3: FakeS3, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"retrained")
    real_resolve = app._resolve_onnx
    monkeypatch.setattr(app, "_resolve_onnx", lambda uri, workdir: str(onnx))
    app.run_optimize(
        {
            "source_model_uri": f"s3://{BUCKET}/training/model.tar.gz",
            "output_uri": f"s3://{BUCKET}/{ENGINE_KEY}",
            "model_name": "ncf_deal_manager",
            "precision": "fp16",
        }
    )
    monkeypatch.setattr(app, "_resolve_onnx", real_resolve)
    builds_before = len(s3.builds)

    app._bootstrap_one(_entry())

    assert len(s3.builds) == builds_before
    assert s3.objects[(BUCKET, ENGINE_KEY)][0] == b"engine-from-retrained"


# ---------------------------------------------------------------------------
# The bootstrap spec
# ---------------------------------------------------------------------------

def _spec() -> dict:
    raw = (REPO_ROOT / "deployment" / "optimizer-bootstrap.json").read_text()
    return json.loads(raw.replace("__FEATURE_SPEC_VERSION__", "1").replace("__MODEL_BUCKET__", BUCKET))


def test_only_the_ncf_entry_rebuilds_on_source_change() -> None:
    by_name = {m["model_name"]: m for m in _spec()["models"]}
    assert by_name["ncf_deal_manager"].get("rebuild_on_source_change") is True
    for name, entry in by_name.items():
        if name != "ncf_deal_manager":
            assert "rebuild_on_source_change" not in entry, name
