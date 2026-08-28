"""Model optimization and multi-version Triton model loading.

Provides:
- ModelOptimizer: Calls the Model Optimizer service (a self-hosted NVIDIA TensorRT
  compilation + optional INT8 quantization microservice) and stores optimized
  artifacts within the same AWS account. This is a TensorRT-based optimizer, NOT a
  stock NVIDIA NIM container (no stock NIM exists for these custom recommender
  models); TensorRT is the same engine NIM is built on.
- TritonModelLoader: Loads model versions into Triton alongside existing versions,
  performs health checks, and supports unloading — all without disrupting live traffic.

Requirements: 5.1, 5.2, 5.3, 12.4
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP Client Protocol — injectable for testing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpResponse:
    """Simple HTTP response container."""

    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body)

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class HttpClient(Protocol):
    """Protocol for an HTTP client used by ModelOptimizer and TritonModelLoader."""

    async def get(self, url: str) -> HttpResponse: ...

    async def post(self, url: str, json_body: dict | None = None) -> HttpResponse: ...


# ---------------------------------------------------------------------------
# NIM Optimizer
# ---------------------------------------------------------------------------

# Default precision is fp16: a real, accurate speedup that needs no calibration.
# int8 is only honest with a real calibration cache (the optimizer service rejects
# int8 without one), and no calibration dataset ships for these models — so fp16
# is the correct default rather than a silently-uncalibrated int8.
_DEFAULT_OPTIMIZATION_CONFIG: dict[str, Any] = {
    "precision": "fp16",
    "max_batch_size": 64,
    "max_workspace_size": 4_294_967_296,  # 4 GiB
}

# Per-model TensorRT optimization profiles (min/opt/max dim lists per input),
# matching the ONNX exports in source/triton/export_models.py. The leading dim is
# the batch dimension, ranged 1..64 to match Triton max_batch_size + dynamic
# batching. Sent to the optimizer so it builds batch-capable engines (without a
# profile, a dynamic-axis ONNX yields a batch-1 engine Triton cannot batch).
#
# NOTE: widedeep_segment_activator is intentionally absent — segment activation
# is now rule-based (no Triton/TensorRT model). See
# source/containers/widedeep_segment_activator/app.py.
_MODEL_INPUT_PROFILES: dict[str, dict[str, dict[str, list[int]]]] = {
    "dlrm_bid_shader": {
        "dense_features": {"min": [1, 4], "opt": [8, 4], "max": [64, 4]},
        "sparse_user": {"min": [1], "opt": [8], "max": [64]},
        "sparse_domain": {"min": [1], "opt": [8], "max": [64]},
        "sparse_device": {"min": [1], "opt": [8], "max": [64]},
    },
    "ncf_deal_manager": {
        # ncf config.pbtxt uses max_batch_size 128, so the engine profile max must
        # be >= 128 or Triton cannot batch up to its configured max.
        "user_ids": {"min": [1], "opt": [16], "max": [128]},
        "item_ids": {"min": [1], "opt": [16], "max": [128]},
    },
}


class ModelOptimizationError(Exception):
    """Raised when model optimization fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ModelOptimizer:
    """Optimizes model artifacts via the Model Optimizer service (TensorRT).

    Calls a self-hosted TensorRT compilation + optional INT8 quantization
    microservice (the ``optimizer_endpoint``). This is a TensorRT-based optimizer,
    NOT a stock NVIDIA NIM container. Artifacts remain within the AWS account
    boundary — optimized engine plans are written back to the same S3 bucket under
    a separate prefix. The optimization process runs within the EKS cluster. (Req 12.4)
    """

    def __init__(
        self,
        optimizer_endpoint: str,
        model_bucket: str,
        region: str,
        *,
        http_client: HttpClient | None = None,
    ):
        self._optimizer_endpoint = optimizer_endpoint.rstrip("/")
        self._model_bucket = model_bucket
        self._region = region
        self._http_client = http_client

    def _build_optimized_uri(self, model_name: str, source_uri: str) -> str:
        """Build the S3 URI for the optimized artifact.

        The optimized artifact stays in the same bucket (same account) under
        the prefix ``optimized-models/<model_name>/``.
        """
        parsed = urlparse(source_uri)
        # Extract the filename from the source path
        source_filename = Path(parsed.path).stem
        optimized_key = (
            f"optimized-models/{model_name}/{source_filename}.engine"
        )
        return f"s3://{self._model_bucket}/{optimized_key}"

    async def optimize(
        self,
        model_artifact_uri: str,
        model_name: str,
        optimization_config: dict[str, Any] | None = None,
        input_profiles: dict[str, dict[str, list[int]]] | None = None,
    ) -> str:
        """Run model optimization (TensorRT compilation + optional quantization).

        Args:
            model_artifact_uri: S3 URI of the raw model artifact.
            model_name: Logical model name (e.g. "dlrm_bid_shader").
            optimization_config: Optional overrides for precision,
                max_batch_size, max_workspace_size.

        Returns:
            S3 URI of the optimized engine plan artifact.

        Raises:
            ModelOptimizationError: If the optimizer service returns an error.
        """
        if self._http_client is None:
            raise RuntimeError("http_client is required to call the optimizer service")

        config: dict[str, Any] = {**_DEFAULT_OPTIMIZATION_CONFIG}
        if optimization_config:
            config.update(optimization_config)

        optimized_uri = self._build_optimized_uri(model_name, model_artifact_uri)

        # Resolve the batch optimization profile: explicit arg wins, else the
        # per-model default. Required by the optimizer whenever max_batch_size > 1.
        profiles = input_profiles or _MODEL_INPUT_PROFILES.get(model_name)

        payload = {
            "source_model_uri": model_artifact_uri,
            "output_uri": optimized_uri,
            "model_name": model_name,
            "precision": config["precision"],
            "max_batch_size": config["max_batch_size"],
            "max_workspace_size": config["max_workspace_size"],
            "target_runtime": "tensorrt",
        }
        if profiles:
            payload["input_profiles"] = profiles

        url = f"{self._optimizer_endpoint}/v1/optimize"

        logger.info(
            "Starting model optimization for %s (precision=%s)",
            model_name,
            config["precision"],
        )

        response = await self._http_client.post(url, json_body=payload)

        if response.status != 200:
            raise ModelOptimizationError(
                f"Model optimization failed for {model_name}: "
                f"status={response.status}, body={response.text()}",
                status_code=response.status,
            )

        result_data = response.json()
        # The optimizer service returns the final output URI in the response
        final_uri = result_data.get("output_uri", optimized_uri)

        logger.info(
            "Model optimization complete for %s -> %s", model_name, final_uri
        )
        return final_uri


# ---------------------------------------------------------------------------
# Triton Model Loader
# ---------------------------------------------------------------------------


class TritonModelLoadError(Exception):
    """Raised when a Triton model load operation fails."""

    def __init__(self, message: str, model_name: str, version: int):
        super().__init__(message)
        self.model_name = model_name
        self.version = version


class TritonModelLoader:
    """Manages the S3-backed Triton model repository for the Option-D canary.

    Triton runs in poll mode (``--model-control-mode=poll``), so "loading" or
    "unloading" a model means writing/deleting files under
    ``s3://<bucket>/<repo_prefix>/<model>/`` and letting Triton's repository poll
    pick up the change. Readiness is confirmed via Triton's HTTP ``/v2`` health
    endpoints (reached through the internal NLB at ``triton_url``).

    Served-repo layout (per recommender model ``<m>``):
      ``<repo_prefix>/<m>/``               canary ROUTER (python)  — committed
      ``<repo_prefix>/<m>_stable/<v>/model.plan``  stable engine (latest v served)
      ``<repo_prefix>/<m>_canary/1/model.plan``    canary engine — created on canary

    The canary split is a router config parameter changed here (control-plane),
    never a per-request lookup — the ARTF bidstream stays dependency-free.
    """

    _MODEL_PLAN = "model.plan"
    _MODEL_XGBOOST_JSON = "xgboost.json"

    def __init__(
        self,
        triton_url: str,
        model_bucket: str,
        *,
        repo_prefix: str = "triton-models",
        http_client: HttpClient | None = None,
        s3_client: Any | None = None,
        region: str | None = None,
        ready_timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 5.0,
    ):
        self._triton_url = triton_url.rstrip("/")
        self._model_bucket = model_bucket
        self._repo_prefix = repo_prefix.strip("/")
        self._http_client = http_client
        self._s3 = s3_client
        self._region = region
        self._ready_timeout_seconds = ready_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    # ------------------------------------------------------------------
    # S3 helpers (sync boto3 wrapped in threads for the async API)
    # ------------------------------------------------------------------

    def _s3c(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3", region_name=self._region)
        return self._s3

    def _key(self, *parts: str) -> str:
        return "/".join([self._repo_prefix, *parts])

    async def _get_text(self, key: str) -> str | None:
        def _fetch():
            try:
                obj = self._s3c().get_object(Bucket=self._model_bucket, Key=key)
                return obj["Body"].read().decode("utf-8")
            except self._s3c().exceptions.NoSuchKey:
                return None
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    return None
                raise

        return await asyncio.to_thread(_fetch)

    async def _put_text(self, key: str, text: str) -> None:
        await asyncio.to_thread(
            lambda: self._s3c().put_object(
                Bucket=self._model_bucket, Key=key, Body=text.encode("utf-8")
            )
        )

    async def _copy_engine(self, engine_uri: str, dest_key: str) -> None:
        """Server-side copy an s3:// engine artifact to a repo key."""
        parsed = urlparse(engine_uri)
        if parsed.scheme != "s3":
            raise TritonModelLoadError(
                f"engine_uri must be an s3:// URI, got {engine_uri}",
                model_name=dest_key,
                version=0,
            )
        src = {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}
        await asyncio.to_thread(
            lambda: self._s3c().copy_object(
                CopySource=src, Bucket=self._model_bucket, Key=dest_key
            )
        )

    async def _delete_prefix(self, prefix: str) -> None:
        def _delete():
            s3 = self._s3c()
            paginator = s3.get_paginator("list_objects_v2")
            to_delete: list[dict] = []
            for page in paginator.paginate(Bucket=self._model_bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    to_delete.append({"Key": obj["Key"]})
                    if len(to_delete) == 1000:
                        s3.delete_objects(
                            Bucket=self._model_bucket, Delete={"Objects": to_delete}
                        )
                        to_delete = []
            if to_delete:
                s3.delete_objects(
                    Bucket=self._model_bucket, Delete={"Objects": to_delete}
                )

        await asyncio.to_thread(_delete)

    async def list_versions(self, model_name: str) -> list[int]:
        """List integer version dirs present in S3 for a model (numeric prefixes)."""
        prefix = self._key(model_name) + "/"

        def _list():
            s3 = self._s3c()
            versions: set[int] = set()
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(
                Bucket=self._model_bucket, Prefix=prefix, Delimiter="/"
            ):
                for cp in page.get("CommonPrefixes", []):
                    tail = cp["Prefix"][len(prefix):].strip("/")
                    if tail.isdigit():
                        versions.add(int(tail))
            return sorted(versions)

        return await asyncio.to_thread(_list)

    # ------------------------------------------------------------------
    # Triton readiness (HTTP /v2)
    # ------------------------------------------------------------------

    async def is_ready(self, model_name: str) -> bool:
        """True if Triton reports the model ready (GET /v2/models/<m>/ready)."""
        if self._http_client is None:
            raise RuntimeError("http_client is required for Triton readiness checks")
        url = f"{self._triton_url}/v2/models/{model_name}/ready"
        try:
            response = await self._http_client.get(url)
            return response.status == 200
        except Exception as exc:  # noqa: BLE001
            logger.error("Readiness check error for %s: %s", model_name, exc)
            return False

    async def wait_until_ready(self, model_name: str, timeout: float | None = None) -> bool:
        """Poll until Triton reports the model ready or the timeout elapses.

        Triton poll mode loads the model asynchronously after S3 changes, so we
        wait for it to appear rather than calling an explicit load API.
        """
        deadline = (timeout if timeout is not None else self._ready_timeout_seconds)
        waited = 0.0
        while waited < deadline:
            if await self.is_ready(model_name):
                return True
            await asyncio.sleep(self._poll_interval_seconds)
            waited += self._poll_interval_seconds
        return await self.is_ready(model_name)

    # ------------------------------------------------------------------
    # Canary / stable repo operations
    # ------------------------------------------------------------------

    @staticmethod
    def stable_name(base_model: str) -> str:
        return f"{base_model}_stable"

    @staticmethod
    def canary_name(base_model: str) -> str:
        return f"{base_model}_canary"

    def canary_engine_uri(self, base_model: str) -> str:
        """Return the deterministic S3 URI of a staged canary's engine file.

        Follows this class's own documented repo layout
        (``<repo_prefix>/<m>_canary/1/model.plan``) — the same path
        stage_canary() writes to. Useful for callers that need to reference
        an already-staged canary's engine without holding onto
        CanaryDeployer's in-process DeploymentState (e.g. a promotion flow
        running in a different process than the one that called
        deploy_canary()).
        """
        canary = self.canary_name(base_model)
        return f"s3://{self._model_bucket}/{self._key(canary, '1', self._MODEL_PLAN)}"

    async def stage_canary(self, base_model: str, engine_uri: str) -> str:
        """Create the ``<base>_canary`` model in the S3 repo from an engine.

        Copies the optimized engine to ``<canary>/1/model.plan`` and writes a
        ``<canary>/config.pbtxt`` derived from the stable model's config (only the
        model name changes). Triton poll mode then loads it. Returns the canary
        model name. Raises TritonModelLoadError if the stable config is missing.
        """
        stable = self.stable_name(base_model)
        canary = self.canary_name(base_model)

        stable_cfg = await self._get_text(self._key(stable, "config.pbtxt"))
        if not stable_cfg:
            raise TritonModelLoadError(
                f"stable config not found for {stable}; cannot derive canary config",
                model_name=canary,
                version=1,
            )
        canary_cfg = stable_cfg.replace(f'"{stable}"', f'"{canary}"')

        await self._put_text(self._key(canary, "config.pbtxt"), canary_cfg)
        await self._copy_engine(engine_uri, self._key(canary, "1", self._MODEL_PLAN))
        logger.info("Staged canary model %s from %s", canary, engine_uri)
        return canary

    async def remove_canary(self, base_model: str) -> None:
        """Delete the ``<base>_canary`` model from the repo (Triton poll-unloads)."""
        canary = self.canary_name(base_model)
        await self._delete_prefix(self._key(canary) + "/")
        logger.info("Removed canary model %s from repo", canary)

    async def promote_engine(self, base_model: str, engine_uri: str) -> int:
        """Publish the engine as a NEW version of ``<base>_stable`` (version swap).

        Triton serves the highest version by default, so writing version N+1 makes
        it the served stable engine once poll-loaded. Returns the new version.
        """
        stable = self.stable_name(base_model)
        versions = await self.list_versions(stable)
        next_version = (max(versions) + 1) if versions else 1
        await self._copy_engine(
            engine_uri, self._key(stable, str(next_version), self._MODEL_PLAN)
        )
        logger.info("Promoted %s: published stable v%d", stable, next_version)
        return next_version

    async def stage_canary_fil(self, base_model: str, artifact_uri: str) -> str:
        """FIL counterpart to stage_canary() for tree-model backends.

        Triton's FIL backend (used by the two yield models) loads a native
        XGBoost artifact directly -- ``<canary>/1/xgboost.json`` -- instead
        of a TensorRT ``model.plan`` engine. No ``ModelOptimizer.optimize()``
        call exists in this path: FIL reads XGBoost's native format as-is,
        there is no TensorRT compilation step for tree models. Otherwise
        mirrors stage_canary() exactly (same config-derivation and
        server-side-copy pattern). Raises TritonModelLoadError if the stable
        config is missing.
        """
        stable = self.stable_name(base_model)
        canary = self.canary_name(base_model)

        stable_cfg = await self._get_text(self._key(stable, "config.pbtxt"))
        if not stable_cfg:
            raise TritonModelLoadError(
                f"stable config not found for {stable}; cannot derive canary config",
                model_name=canary,
                version=1,
            )
        canary_cfg = stable_cfg.replace(f'"{stable}"', f'"{canary}"')

        await self._put_text(self._key(canary, "config.pbtxt"), canary_cfg)
        await self._copy_engine(artifact_uri, self._key(canary, "1", self._MODEL_XGBOOST_JSON))
        logger.info("Staged FIL canary model %s from %s", canary, artifact_uri)
        return canary

    async def promote_fil(self, base_model: str, artifact_uri: str) -> int:
        """FIL counterpart to promote_engine() for tree-model backends.

        Publishes the native XGBoost artifact as a NEW version of
        ``<base>_stable`` (``<stable>/<v>/xgboost.json``), same
        version-numbering scheme as promote_engine(). Returns the new
        version.
        """
        stable = self.stable_name(base_model)
        versions = await self.list_versions(stable)
        next_version = (max(versions) + 1) if versions else 1
        await self._copy_engine(
            artifact_uri, self._key(stable, str(next_version), self._MODEL_XGBOOST_JSON)
        )
        logger.info("Promoted %s: published stable v%d (FIL)", stable, next_version)
        return next_version

    async def set_router_split(
        self,
        router_model: str,
        canary_traffic_pct: float,
        canary_model: str | None = None,
        *,
        canary_version_arn: str | None = None,
        stable_version_arn: str | None = None,
    ) -> None:
        """Set the router's in-memory canary split by editing its config in S3.

        Rewrites the ``canary_traffic_pct`` (and optionally ``canary_model``) config
        parameter and writes the config back; Triton poll mode reloads the router.
        This is a control-plane change — never a per-request lookup.

        ``canary_version_arn``/``stable_version_arn``, when provided, also update
        the router's version-ARN parameters (read by the router at execute()
        time to populate the additive ``served_model_version`` output — see
        source/triton/router/model.py). Omitted (None) leaves the existing
        parameter unchanged; the router config gracefully tolerates the
        parameter being absent entirely on older-generated configs (routers
        without ``served_model_version`` declared simply never read it).
        """
        key = self._key(router_model, "config.pbtxt")
        cfg = await self._get_text(key)
        if cfg is None:
            raise TritonModelLoadError(
                f"router config not found for {router_model}",
                model_name=router_model,
                version=0,
            )
        cfg = self._set_param(cfg, "canary_traffic_pct", str(canary_traffic_pct))
        if canary_model is not None:
            cfg = self._set_param(cfg, "canary_model", canary_model)
        if canary_version_arn is not None:
            cfg = self._set_param(cfg, "canary_version_arn", canary_version_arn)
        if stable_version_arn is not None:
            cfg = self._set_param(cfg, "stable_version_arn", stable_version_arn)
        await self._put_text(key, cfg)
        logger.info(
            "Router %s split set to %.1f%% (canary_model=%s)",
            router_model,
            canary_traffic_pct,
            canary_model or "<unchanged>",
        )

    @staticmethod
    def _set_param(config_text: str, key: str, value: str) -> str:
        """Set a Triton config.pbtxt parameter's string_value by key.

        Matches `key: "<key>", value: { string_value: "<old>" }` (whitespace/comma
        tolerant) and replaces the old value. Raises if the parameter is absent.
        """
        pattern = re.compile(
            r'(key:\s*"' + re.escape(key) + r'"\s*,?\s*value:\s*\{\s*string_value:\s*")'
            r'[^"]*'
            r'(")',
            re.DOTALL,
        )
        new_text, n = pattern.subn(r"\g<1>" + value + r"\g<2>", config_text)
        if n == 0:
            raise ValueError(f"parameter '{key}' not found in router config")
        return new_text
