"""Model Optimizer service — HTTP microservice on :8080.

Implements the ``POST /v1/optimize`` contract that
``source/deployment/model_deployer.py::ModelOptimizer`` calls, using real NVIDIA
TensorRT (``trtexec``) to compile an ONNX model into an optimized engine plan.

Request body (JSON), as sent by ModelOptimizer.optimize():
    {
      "source_model_uri":   "s3://.../model.onnx" | "s3://.../model.tar.gz",
      "output_uri":         "s3://.../optimized-models/<model>/<file>.engine",
      "model_name":         "dlrm_bid_shader",
      "precision":          "fp16" | "fp32" | "int8",
      "max_batch_size":     64,
      "max_workspace_size": 4294967296,          # bytes
      "target_runtime":     "tensorrt",
      "calibration_cache_uri": "s3://.../calib.cache"   # REQUIRED for int8
    }

Response (200):
    {"output_uri": "s3://...", "precision": "fp16", "engine_bytes": 12345,
     "source_model_uri": "s3://...", "model_name": "...", "duration_s": 12.3}

Honesty contract (no fabrication):
- FP16 / FP32 are built directly from the ONNX — no calibration needed.
- INT8 is only produced when a real calibration cache is supplied. If int8 is
  requested without ``calibration_cache_uri`` the service returns HTTP 400 and
  builds nothing — it never emits a mislabeled "int8" engine from uncalibrated
  dynamic ranges.
- If ``trtexec`` fails, the real stderr is returned with HTTP 500. Nothing is
  faked or silently downgraded.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # nosemgrep: dangerous-subprocess-use — trtexec args are built from validated fields
import sys
import tarfile
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

import boto3
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger("optimizer")

AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# trtexec ships in the TensorRT container; allow an override for other layouts.
TRTEXEC = os.environ.get("TRTEXEC_PATH", "trtexec")

_VALID_PRECISIONS = {"fp32", "fp16", "int8"}

# Provenance file a producer writes beside an exported model, and this service reads
# before compiling it. Matches shared/dlrm_features.py's MANIFEST_FILENAME; that
# module is not importable here (this image is built from source/Dockerfile.optimizer
# plus source/optimizer alone), and tests/test_feature_spec_version_gate.py asserts
# the two agree.
MANIFEST_FILENAME = "manifest.json"

# S3 user-defined metadata on every engine this service writes (S3 stores the keys
# in lowercase and returns them without the x-amz-meta- prefix). source-fingerprint
# is the `fingerprint` field of the source's manifest.json, absent when the manifest
# has none; provisioned-by says which path wrote the engine. A bootstrap entry with
# rebuild_on_source_change rebuilds its engine when the two fingerprints differ, and
# never overwrites one a promotion wrote.
META_SOURCE_FINGERPRINT = "source-fingerprint"
META_PROVISIONED_BY = "provisioned-by"
PROVISIONED_BY_BOOTSTRAP = "bootstrap"
PROVISIONED_BY_PROMOTION = "promotion"


def _valid_input_profiles(profiles: Any) -> bool:
    """Validate the input_profiles structure: {name: {min:[...],opt:[...],max:[...]}}."""
    if not isinstance(profiles, dict) or not profiles:
        return False
    for prof in profiles.values():
        if not isinstance(prof, dict):
            return False
        for key in ("min", "opt", "max"):
            dims = prof.get(key)
            if not isinstance(dims, list) or not dims or not all(
                isinstance(d, (int, float)) for d in dims
            ):
                return False
    return True


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _s3():
    return boto3.client("s3", region_name=AWS_REGION)


def _parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download(uri: str, dest_path: str) -> None:
    bucket, key = _parse_s3(uri)
    _s3().download_file(bucket, key, dest_path)


def _upload(src_path: str, uri: str, metadata: dict[str, str] | None = None) -> None:
    bucket, key = _parse_s3(uri)
    if metadata:
        _s3().upload_file(src_path, bucket, key, ExtraArgs={"Metadata": metadata})
    else:
        _s3().upload_file(src_path, bucket, key)


def _engine_metadata(
    provisioned_by: str, source_fingerprint: str | None
) -> dict[str, str]:
    """The S3 metadata an engine is written with."""
    meta = {META_PROVISIONED_BY: provisioned_by}
    if source_fingerprint:
        meta[META_SOURCE_FINGERPRINT] = source_fingerprint
    return meta


def _head_metadata(uri: str) -> dict[str, str] | None:
    """User metadata of the object at `uri`, or None when the object does not exist.

    Only a 404 means absent. Any other error is raised: reading AccessDenied as
    "no engine" would rebuild over an engine whose provenance could not be read,
    which may be a promoted one.
    """
    bucket, key = _parse_s3(uri)
    try:
        head = _s3().head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a plain 404
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return {str(k).lower(): str(v) for k, v in (head.get("Metadata") or {}).items()}


class OptimizeRequestError(ValueError):
    """Raised for an invalid /v1/optimize request body (maps to HTTP 400)."""


def _resolve_onnx(source_model_uri: str, workdir: str) -> str:
    """Fetch the source artifact and return a local path to the .onnx file.

    Handles both a raw ``model.onnx`` and a SageMaker ``model.tar.gz`` that
    contains one (searches the archive for the first ``*.onnx``).
    """
    _, key = _parse_s3(source_model_uri)
    local_artifact = os.path.join(workdir, os.path.basename(key) or "artifact")
    _download(source_model_uri, local_artifact)

    if local_artifact.endswith((".tar.gz", ".tgz", ".tar")):
        extract_dir = os.path.join(workdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(local_artifact) as tar:
            # Guard against path traversal in archive members.
            for member in tar.getmembers():
                member_path = os.path.realpath(os.path.join(extract_dir, member.name))
                if not member_path.startswith(os.path.realpath(extract_dir) + os.sep):
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            tar.extractall(extract_dir, filter="data")  # nosemgrep: tarfile-extractall: members validated above
        for root, _dirs, files in os.walk(extract_dir):
            for fname in files:
                if fname.endswith(".onnx"):
                    return os.path.join(root, fname)
        raise ValueError(f"No .onnx file found inside archive {source_model_uri}")

    if local_artifact.endswith(".onnx"):
        # A raw .onnx has no archive to carry its manifest, so the manifest is a
        # SIBLING S3 OBJECT and has to be fetched separately. Without this the
        # provenance check below sees an empty workdir and refuses every raw-.onnx
        # source as unmanifested, including ones whose manifest is sitting next to
        # them in the bucket. Found by a live bootstrap run doing exactly that.
        _fetch_sibling_manifest(source_model_uri, os.path.dirname(local_artifact))
        return local_artifact

    raise ValueError(
        f"Unsupported source artifact (expected .onnx or .tar.gz): {source_model_uri}"
    )


def _fetch_sibling_manifest(source_model_uri: str, dest_dir: str) -> None:
    """Download `manifest.json` from the source's S3 prefix, if it is there.

    A genuinely absent object leaves `dest_dir` without a manifest, which the
    provenance check reports as "no manifest" — the correct outcome. Any OTHER S3
    error is raised: a manifest that exists but could not be read is a different
    fact from one that was never written, and reporting the second for the first
    would turn a permissions problem into a false claim about the artifact.
    """
    bucket, key = _parse_s3(source_model_uri)
    sibling_key = "/".join([*key.split("/")[:-1], MANIFEST_FILENAME])
    dest = os.path.join(dest_dir, MANIFEST_FILENAME)
    try:
        _s3().download_file(bucket, sibling_key, dest)
        logger.info("Fetched s3://%s/%s", bucket, sibling_key)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a plain 404
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey"):
            logger.info("No %s beside %s", MANIFEST_FILENAME, source_model_uri)
            return
        raise


# ---------------------------------------------------------------------------
# Feature-spec provenance
# ---------------------------------------------------------------------------
# A model's inputs having the right NAMES and WIDTHS is not evidence that the
# producer and the serving container mean the same thing by them. A vector whose
# position two changed meaning, or whose vocabulary sizes changed, compiles and
# serves cleanly and is simply wrong. The producer records what it built against
# in a manifest beside the artifact; this is where that record is enforced.
#
# The expected version is supplied BY THE CALLER, not imported here. This service
# compiles ONNX to TensorRT for every model in the repository and is not the
# authority on any one model's feature contract -- and its image is built from
# source/Dockerfile.optimizer plus source/optimizer alone, with no access to
# source/shared. The callers that care (the bootstrap spec, and the governance
# promotion path) hold the spec and state the version they expect.


def _read_manifest(onnx_path: str, source_model_uri: str) -> dict[str, Any] | None:
    """The manifest beside a resolved ONNX, or None if there is none.

    Looks next to the .onnx first (both producers write it there), then anywhere
    under the extraction directory, because a SageMaker archive may nest the model
    inside a directory.
    """
    directory = os.path.dirname(onnx_path)
    candidate = os.path.join(directory, MANIFEST_FILENAME)
    if not os.path.exists(candidate):
        for root, _dirs, files in os.walk(directory):
            if MANIFEST_FILENAME in files:
                candidate = os.path.join(root, MANIFEST_FILENAME)
                break
        else:
            return None
    try:
        with open(candidate, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizeRequestError(
            f"{MANIFEST_FILENAME} beside {source_model_uri} could not be read "
            f"({exc}). Refusing to compile an artifact whose provenance record is "
            "unreadable."
        ) from exc
    if not isinstance(loaded, dict):
        raise OptimizeRequestError(
            f"{MANIFEST_FILENAME} beside {source_model_uri} is not a JSON object."
        )
    return loaded


def _require_feature_spec_version(
    onnx_path: str, source_model_uri: str, expected: int
) -> int:
    """Refuse unless the artifact declares exactly `expected`.

    Returns the declared version so the caller can report it. Absent is refused,
    not waved through: an artifact with no manifest was produced by something whose
    feature contract is unknown, which is the case this check exists for.
    """
    manifest = _read_manifest(onnx_path, source_model_uri)
    if manifest is None:
        raise OptimizeRequestError(
            f"{source_model_uri} has no {MANIFEST_FILENAME}, so the feature-spec "
            f"version it was built against is unknown. Expected "
            f"{expected}. Refusing to compile an engine whose feature contract "
            "cannot be established — matching input names and widths do not "
            "establish it."
        )
    declared = manifest.get("feature_spec_version")
    if declared is None:
        raise OptimizeRequestError(
            f"{MANIFEST_FILENAME} beside {source_model_uri} declares no "
            f"'feature_spec_version'. Expected {expected}."
        )
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise OptimizeRequestError(
            f"{MANIFEST_FILENAME} beside {source_model_uri} declares "
            f"feature_spec_version {declared!r}, which is not an integer."
        )
    if declared != expected:
        raise OptimizeRequestError(
            f"{source_model_uri} was built against feature-spec version "
            f"{declared}; this deployment serves version {expected}. Refusing to "
            "compile it. A vector of the same width with different meaning per "
            "position produces a working engine and wrong predictions. Retrain "
            f"against version {expected}, or deploy a build that serves "
            f"version {declared}."
        )
    return declared


# ---------------------------------------------------------------------------
# trtexec build
# ---------------------------------------------------------------------------


def _shapes_flag(input_profiles: dict, key: str) -> str:
    """Build a trtexec shape flag value like 'dense_features:8x4,sparse_site_domain:8'.

    key is one of 'min' | 'opt' | 'max'. Each profile entry maps an input name to
    {"min":[...],"opt":[...],"max":[...]} dim lists.
    """
    parts = []
    for name, prof in input_profiles.items():
        dims = prof[key]
        parts.append(f"{name}:{'x'.join(str(int(d)) for d in dims)}")
    return ",".join(parts)


def _build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str,
    max_workspace_bytes: int,
    calib_cache_path: str | None,
    input_profiles: dict | None = None,
) -> None:
    """Run trtexec to compile ONNX -> TensorRT engine plan. Raises on failure."""
    workspace_mib = max(256, int(max_workspace_bytes) // (1024 * 1024))
    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}MiB",
    ]
    # Optimization profile for dynamic (batched) inputs so the engine supports
    # Triton dynamic_batching / max_batch_size. Without this, a dynamic-axis ONNX
    # yields a batch=1 engine that Triton cannot batch.
    if input_profiles:
        cmd.append(f"--minShapes={_shapes_flag(input_profiles, 'min')}")
        cmd.append(f"--optShapes={_shapes_flag(input_profiles, 'opt')}")
        cmd.append(f"--maxShapes={_shapes_flag(input_profiles, 'max')}")
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        # Calibration cache presence is validated by the caller; --int8 without
        # a calibration cache would emit an uncalibrated (dishonest) engine.
        cmd.append("--int8")
        cmd.append(f"--calib={calib_cache_path}")
    # fp32 => no precision flag (default).

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit fixed trtexec binary, args built from validated fields
        cmd,  # nosemgrep: dangerous-subprocess-use-tainted-env-args the only env input is TRTEXEC_PATH, set by the deployment manifest; the rest are paths this job created
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit {proc.returncode}):\n{proc.stdout[-4000:]}"
        )
    if not os.path.exists(engine_path) or os.path.getsize(engine_path) == 0:
        raise RuntimeError(
            f"trtexec reported success but produced no engine at {engine_path}:\n"
            f"{proc.stdout[-2000:]}"
        )


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


def run_optimize(body: dict[str, Any]) -> dict[str, Any]:
    """Core optimize logic: validate the request, build the TensorRT engine, and
    upload it to output_uri. Returns the result dict on success.

    Shared by the HTTP handler (``POST /v1/optimize``) and the ``--optimize-once``
    Job entrypoint (option A: the VPC proxy Lambda launches this as an on-demand
    K8s Job so no optimizer pod holds a GPU at steady state). Raises
    OptimizeRequestError for bad input (400) and RuntimeError/Exception for build
    failures (500) — never fabricates a result.
    """
    start = time.time()

    source_model_uri = body.get("source_model_uri", "")
    output_uri = body.get("output_uri", "")
    model_name = body.get("model_name", "")
    precision = str(body.get("precision", "fp16")).lower()
    max_workspace_size = int(body.get("max_workspace_size", 4 * 1024 * 1024 * 1024))
    calibration_cache_uri = body.get("calibration_cache_uri")
    max_batch_size = int(body.get("max_batch_size", 1))
    input_profiles = body.get("input_profiles")
    expected_feature_spec_version = body.get("expected_feature_spec_version")

    if not source_model_uri or not output_uri or not model_name:
        raise OptimizeRequestError(
            "source_model_uri, output_uri, and model_name are required"
        )
    if precision not in _VALID_PRECISIONS:
        raise OptimizeRequestError(
            f"precision must be one of {sorted(_VALID_PRECISIONS)}; got '{precision}'"
        )
    # A batched model (Triton max_batch_size>1) needs a TensorRT optimization
    # profile, or the engine is fixed at batch=1 and Triton can't batch it.
    # Refuse honestly rather than emit a silently batch-1 engine.
    if max_batch_size > 1 and not input_profiles:
        raise OptimizeRequestError(
            "max_batch_size>1 requires 'input_profiles' (per-input min/opt/max "
            "dim lists) so the TensorRT engine supports Triton dynamic batching. "
            "Refusing to emit a batch-1 engine for a batched model."
        )
    if input_profiles is not None and not _valid_input_profiles(input_profiles):
        raise OptimizeRequestError(
            "input_profiles must map each input name to an object with 'min', "
            "'opt', and 'max' dim lists, e.g. {\"dense_features\": {\"min\":[1,4],"
            "\"opt\":[8,4],\"max\":[64,4]}}"
        )
    if precision == "int8" and not calibration_cache_uri:
        # Honest refusal — INT8 without calibration would be a fabricated result.
        raise OptimizeRequestError(
            "int8 precision requires 'calibration_cache_uri' (a real TensorRT "
            "calibration cache). Refusing to emit an uncalibrated int8 engine. "
            "Use precision 'fp16' or supply a calibration cache."
        )

    if expected_feature_spec_version is not None and (
        not isinstance(expected_feature_spec_version, int)
        or isinstance(expected_feature_spec_version, bool)
    ):
        raise OptimizeRequestError(
            "expected_feature_spec_version must be an integer; got "
            f"{expected_feature_spec_version!r}"
        )

    workdir = tempfile.mkdtemp(prefix="trt-opt-")
    try:
        onnx_path = _resolve_onnx(source_model_uri, workdir)

        # Before spending a GPU on the compile, and before anything can serve the
        # result. Omitting expected_feature_spec_version skips the check, for the
        # models that have no feature-spec contract to check (the yield models are
        # tree-based and never come through here; NCF's vector is not versioned).
        declared_feature_spec_version = None
        if expected_feature_spec_version is not None:
            declared_feature_spec_version = _require_feature_spec_version(
                onnx_path, source_model_uri, expected_feature_spec_version
            )

        calib_cache_path = None
        if precision == "int8":
            calib_cache_path = os.path.join(workdir, "calibration.cache")
            _download(calibration_cache_uri, calib_cache_path)

        engine_path = os.path.join(workdir, "model.plan")
        _build_engine(
            onnx_path,
            engine_path,
            precision,
            max_workspace_size,
            calib_cache_path,
            input_profiles=input_profiles,
        )

        # Marked as a promotion so a later bootstrap never rebuilds over it.
        _upload(
            engine_path,
            output_uri,
            _engine_metadata(PROVISIONED_BY_PROMOTION, None),
        )
        engine_bytes = os.path.getsize(engine_path)

        duration = time.time() - start
        logger.info(
            "Optimized %s (%s) -> %s (%d bytes, %.1fs)",
            model_name, precision, output_uri, engine_bytes, duration,
        )
        return {
            "output_uri": output_uri,
            "precision": precision,
            "engine_bytes": engine_bytes,
            "source_model_uri": source_model_uri,
            "model_name": model_name,
            "duration_s": round(duration, 2),
            # None when the caller asked for no check, so a reader can tell
            # "verified against version 1" from "not checked".
            "feature_spec_version": declared_feature_spec_version,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def optimize(request: Request) -> JSONResponse:
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        return JSONResponse(run_optimize(body), status_code=200)
    except OptimizeRequestError as exc:
        logger.error("Bad optimize request: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    except ValueError as exc:
        logger.error("Bad optimize request: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:  # noqa: BLE001 — surface the real failure, never fake success
        logger.exception("Optimization failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "Healthy"})


# ---------------------------------------------------------------------------
# Startup bootstrap — build base (stable) engines the first time
# ---------------------------------------------------------------------------
# tensorrt_plan serving requires each base model compiled to an engine, and that
# must happen on this GPU node (the deploy host has no GPU and can't reach the
# internal NLB). On startup, if OPTIMIZER_BOOTSTRAP_URI points at a spec, build any
# base engine that is not already present in S3, then serve. Idempotent: existing
# engines are skipped, so restarts and re-deploys are cheap. The exception is an
# entry with rebuild_on_source_change: its engine is rebuilt when the source
# manifest's fingerprint no longer matches the engine's source-fingerprint, unless
# a promotion wrote the engine (see _engine_is_current).

OPTIMIZER_BOOTSTRAP_URI = os.environ.get("OPTIMIZER_BOOTSTRAP_URI", "")


def _s3_object_exists(uri: str) -> bool:
    bucket, key = _parse_s3(uri)
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001 - any error (404/NoSuchKey) => treat as absent
        return False


def _manifest_fingerprint(manifest: dict[str, Any] | None) -> str | None:
    """The `fingerprint` a manifest records, or None when it records none."""
    value = (manifest or {}).get("fingerprint")
    return value if isinstance(value, str) and value else None


def _source_fingerprint(source_model_uri: str) -> str | None:
    """The fingerprint in the manifest beside a raw-.onnx source, or None.

    Reads only the sibling manifest.json (the optimizer image has no source/shared
    to recompute it from), so deciding whether to rebuild costs no ONNX download.
    """
    workdir = tempfile.mkdtemp(prefix="trt-fingerprint-")
    try:
        _fetch_sibling_manifest(source_model_uri, workdir)
        path = os.path.join(workdir, MANIFEST_FILENAME)
        if not os.path.exists(path):
            return None
        return _manifest_fingerprint(_read_manifest(path, source_model_uri))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _engine_is_current(entry: dict) -> bool:
    """Whether the bootstrap should leave the entry's engine as it is.

    Without rebuild_on_source_change an existing engine is always kept. With it, an
    existing engine is kept when a promotion wrote it (or any writer other than the
    bootstrap), when the source records no fingerprint to compare, or when its
    source-fingerprint equals the source's. An engine with no provisioned-by
    metadata counts as a bootstrap engine: it predates the metadata, and only the
    bootstrap wrote engines then.
    """
    model_name = entry["model_name"]
    output_uri = entry["output_uri"]
    if entry.get("rebuild_on_source_change") is not True:
        if _s3_object_exists(output_uri):
            logger.info("Bootstrap: engine already present for %s (%s); skipping", model_name, output_uri)
            return True
        return False

    engine_meta = _head_metadata(output_uri)
    if engine_meta is None:
        return False
    provisioned_by = engine_meta.get(META_PROVISIONED_BY, "")
    if provisioned_by not in ("", PROVISIONED_BY_BOOTSTRAP):
        logger.info(
            "Bootstrap: %s engine at %s was provisioned by %s; not rebuilding it",
            model_name, output_uri, provisioned_by,
        )
        return True
    source_fp = _source_fingerprint(entry["source_model_uri"])
    engine_fp = engine_meta.get(META_SOURCE_FINGERPRINT, "")
    if source_fp is None:
        logger.warning(
            "Bootstrap: %s source %s records no fingerprint; keeping the engine at %s",
            model_name, entry["source_model_uri"], output_uri,
        )
        return True
    if engine_fp == source_fp:
        logger.info(
            "Bootstrap: %s engine matches source fingerprint %s; skipping",
            model_name, source_fp[:12],
        )
        return True
    logger.info(
        "Bootstrap: %s engine fingerprint %s differs from source %s (provisioned-by %s); rebuilding",
        model_name, engine_fp[:12] or "<none>", source_fp[:12], provisioned_by or "<none>",
    )
    return False


def _bootstrap_one(entry: dict) -> None:
    model_name = entry["model_name"]
    source_model_uri = entry["source_model_uri"]
    output_uri = entry["output_uri"]
    precision = str(entry.get("precision", "fp16")).lower()
    max_workspace_size = int(entry.get("max_workspace_size", 4 * 1024 * 1024 * 1024))
    input_profiles = entry.get("input_profiles")

    expected_feature_spec_version = entry.get("expected_feature_spec_version")

    if _engine_is_current(entry):
        return

    workdir = tempfile.mkdtemp(prefix="trt-bootstrap-")
    try:
        onnx_path = _resolve_onnx(source_model_uri, workdir)
        # Same gate as the promotion path. A genesis artifact is still an artifact
        # whose feature contract has to match what will serve it -- the exporter and
        # the container are built from one repo, but not necessarily at one commit.
        if expected_feature_spec_version is not None:
            declared = _require_feature_spec_version(
                onnx_path, source_model_uri, expected_feature_spec_version
            )
            logger.info(
                "Bootstrap: %s declares feature_spec_version %d", model_name, declared
            )
        engine_path = os.path.join(workdir, "model.plan")
        _build_engine(onnx_path, engine_path, precision, max_workspace_size, None, input_profiles=input_profiles)
        source_fp = _manifest_fingerprint(_read_manifest(onnx_path, source_model_uri))
        _upload(
            engine_path,
            output_uri,
            _engine_metadata(PROVISIONED_BY_BOOTSTRAP, source_fp),
        )
        logger.info("Bootstrap: built base engine for %s -> %s", model_name, output_uri)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_bootstrap() -> None:
    """Build any missing (or, for rebuild_on_source_change entries, out-of-date)
    base engines described by the bootstrap spec.

    The spec is JSON at OPTIMIZER_BOOTSTRAP_URI: {"models": [ {model_name,
    source_model_uri, output_uri, precision?, max_workspace_size?, input_profiles?,
    rebuild_on_source_change?}, ...]}.
    A failure to build one model is logged and re-raised (fail loudly — a missing
    base engine means Triton cannot serve that model; we do not hide it).
    """
    if not OPTIMIZER_BOOTSTRAP_URI:
        logger.info("No OPTIMIZER_BOOTSTRAP_URI set — skipping base-engine bootstrap")
        return
    logger.info("Bootstrap: reading spec from %s", OPTIMIZER_BOOTSTRAP_URI)
    bucket, key = _parse_s3(OPTIMIZER_BOOTSTRAP_URI)
    raw = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    spec = json.loads(raw)
    models = spec.get("models", [])
    logger.info("Bootstrap: %d model(s) in spec", len(models))
    for entry in models:
        _bootstrap_one(entry)
    logger.info("Bootstrap: complete")


app = Starlette(
    routes=[
        Route("/v1/optimize", optimize, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/ping", health, methods=["GET"]),
    ]
)


def _require_trtexec() -> None:
    """Fail loudly if trtexec is not on PATH — this component is useless without
    it and a silent absence would surface later as confusing errors."""
    if shutil.which(TRTEXEC) is None:
        logger.error(
            "trtexec not found on PATH (TRTEXEC_PATH=%s). This must run in a TensorRT image.",
            TRTEXEC,
        )
        sys.exit(1)


def _run_optimize_once() -> None:
    """On-demand single optimize (option A): read the request from the
    OPTIMIZE_REQUEST_JSON env var, build+upload the engine, print the result JSON
    to stdout, and exit 0. Exit 2 for a bad request, 1 for a build failure. This is
    the entrypoint the VPC proxy Lambda launches as a one-shot K8s Job so no
    optimizer pod holds a GPU at steady state."""
    _require_trtexec()
    raw = os.environ.get("OPTIMIZE_REQUEST_JSON", "")
    if not raw:
        logger.error("--optimize-once requires the OPTIMIZE_REQUEST_JSON env var (the /v1/optimize body)")
        sys.exit(2)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("OPTIMIZE_REQUEST_JSON is not valid JSON: %s", exc)
        sys.exit(2)
    try:
        result = run_optimize(body)
    except OptimizeRequestError as exc:
        logger.error("Bad optimize request: %s", exc)
        print(json.dumps({"error": str(exc)}))
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 — surface the real failure, never fake success
        logger.exception("Optimization failed: %s", exc)
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
    print(json.dumps(result))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    args = set(sys.argv[1:])

    # --bootstrap: build any missing base (stable) engines from the bootstrap spec,
    # then exit. Run as a one-shot K8s Job at deploy time so Triton has engines to
    # serve without a permanently GPU-resident optimizer.
    if "--bootstrap" in args:
        _require_trtexec()
        logger.info("Model Optimizer bootstrap (build base engines then exit) region=%s trtexec=%s", AWS_REGION, TRTEXEC)
        run_bootstrap()
        return

    # --optimize-once: on-demand single engine build (launched as a Job by the VPC
    # proxy Lambda for promotions), then exit.
    if "--optimize-once" in args:
        _run_optimize_once()
        return

    # Default: long-running HTTP service (local dev / backward compatible). The
    # production deploy no longer runs this as an always-on GPU Deployment — base
    # engines come from the --bootstrap Job and promotions from --optimize-once Jobs.
    import uvicorn

    _require_trtexec()
    logger.info("Model Optimizer service starting on :8080 (region=%s, trtexec=%s)", AWS_REGION, TRTEXEC)
    run_bootstrap()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
