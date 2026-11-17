"""HTTP routes for the demo training datasets (FR1, FR1.11, NFR2).

Demonstration only: these routes create and describe the synthetic datasets in
``orchestrator/demo_dataset.py``. Training reads one only when the person turns
on "Include demo dataset" (``POST /v1/governance/train``).

Routes (registered in app.py's ``_governance_routes``, behind
``CognitoAuthMiddleware`` like every other governance route):

- ``GET  /v1/governance/demo-dataset?model_type=...`` returns whether a dataset
  exists, its stored metadata, and the parameter bounds and defaults.
- ``GET  /v1/governance/demo-dataset/estimate?model_type=...&<param>=...``
  returns the row count and expected positives for a parameter set.
- ``POST /v1/governance/demo-dataset`` with
  ``{"model_type", "params", "seed"?, "confirmed": true}`` generates, checks
  and stores a dataset, replacing the previous one. 201 on success.

Storage is ``TRAINING_DATA_BUCKET`` under the prefixes in
``demo_dataset.DEMO_PREFIXES``. An unset bucket is 503 and an S3 failure is a
JSON 500; neither falls back to anything else.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from orchestrator import demo_dataset
from orchestrator.training_trigger import TRAINABLE_MODEL_TYPES

logger = logging.getLogger(__name__)

_MAX_SEED = 2**31 - 1


class GenerateDemoDatasetBody(BaseModel):
    """Body of POST /v1/governance/demo-dataset. Unknown fields are refused."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_type: str = Field(min_length=1, max_length=64)
    params: dict[str, Union[StrictInt, StrictFloat]] = Field(default_factory=dict, max_length=8)
    seed: Optional[StrictInt] = Field(default=None, ge=0, le=_MAX_SEED)
    confirmed: StrictBool = False


def _error(message: str, reason: str, status: int, **extra) -> JSONResponse:
    return JSONResponse({"error": message, "reason": reason, **extra}, status_code=status)


def _model_type_error(model_type: str) -> JSONResponse | None:
    if model_type in TRAINABLE_MODEL_TYPES and model_type in demo_dataset.PARAM_SPECS:
        return None
    supported = sorted(TRAINABLE_MODEL_TYPES & set(demo_dataset.PARAM_SPECS))
    return _error(
        f"Model type '{model_type}' has no demo dataset.",
        "unsupported_model_type",
        422,
        supported_model_types=supported,
    )


def _bucket() -> str:
    return os.environ.get("TRAINING_DATA_BUCKET", "")


def _bucket_missing() -> JSONResponse:
    return _error(
        "Demo datasets are not configured on this deployment (TRAINING_DATA_BUCKET unset).",
        "not_configured",
        503,
    )


def _describe(model_type: str) -> dict:
    return {
        "model_type": model_type,
        "prefix": f"{demo_dataset.DEMO_PREFIXES[model_type]}/",
        "params": demo_dataset.param_bounds(model_type),
        "row_cap": demo_dataset.ROW_CAP,
        "notice": demo_dataset.DEMO_NOTICE,
        "demonstration_only": True,
    }


async def status_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/demo-dataset?model_type=... (FR1.11)."""
    model_type = request.query_params.get("model_type", "")
    invalid = _model_type_error(model_type)
    if invalid:
        return invalid
    bucket = _bucket()
    if not bucket:
        return _bucket_missing()
    try:
        metadata = await asyncio.to_thread(demo_dataset.read_demo_metadata, bucket, model_type)
    except demo_dataset.DemoDatasetStorageError as exc:
        logger.error("demo dataset status read failed for %s: %s", model_type, exc)
        return _error(str(exc), "storage_error", 500)
    return JSONResponse({**_describe(model_type), "exists": metadata is not None, "metadata": metadata})


async def estimate_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/demo-dataset/estimate?model_type=...&<param>=... (FR1.5)."""
    model_type = request.query_params.get("model_type", "")
    invalid = _model_type_error(model_type)
    if invalid:
        return invalid
    params: dict[str, float] = {}
    for name, raw in request.query_params.items():
        if name == "model_type":
            continue
        try:
            params[name] = float(raw)
        except ValueError:
            return _error(f"'{name}' must be a number, got {raw!r}.", "invalid_params", 422, field=name)
    try:
        resolved = demo_dataset.validate_params(model_type, params)
    except demo_dataset.DemoParamsError as exc:
        return _error(str(exc), "invalid_params", 422, field=exc.field)
    sizing = demo_dataset.size_dataset(model_type, resolved)
    return JSONResponse({**_describe(model_type), "resolved_params": resolved, "sizing": asdict(sizing)})


async def generate_handler(request: Request) -> JSONResponse:
    """POST /v1/governance/demo-dataset (FR1.1, FR1.2, FR1.6, FR1.9, FR1.10).

    Order: body, model type, parameters, confirmation, bucket, generate and
    check, write. Every refusal happens before anything is written.
    """
    try:
        raw = await request.json()
    except ValueError:
        return _error("Request body must be JSON.", "invalid_request", 422)
    try:
        body = GenerateDemoDatasetBody.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())) or None
        return _error(f"Invalid request: {first.get('msg', 'validation failed')}", "invalid_request", 422, field=field)

    invalid = _model_type_error(body.model_type)
    if invalid:
        return invalid
    try:
        demo_dataset.validate_params(body.model_type, body.params)
    except demo_dataset.DemoParamsError as exc:
        return _error(str(exc), "invalid_params", 422, field=exc.field)
    if not body.confirmed:
        return _error(
            "Generating a demo dataset replaces the current one and requires confirmed: true.",
            "not_confirmed",
            422,
        )
    bucket = _bucket()
    if not bucket:
        return _bucket_missing()

    seed = demo_dataset.DEFAULT_SEED if body.seed is None else body.seed
    try:
        dataset = await asyncio.to_thread(
            demo_dataset.generate_demo_dataset, body.model_type, body.params, seed
        )
    except demo_dataset.DemoDatasetUnfitError as exc:
        return _error(str(exc), "dataset_unfit", 422)
    except demo_dataset.DemoParamsError as exc:
        return _error(str(exc), "invalid_params", 422, field=exc.field)

    try:
        metadata = await asyncio.to_thread(demo_dataset.write_demo_dataset, bucket, dataset)
    except demo_dataset.DemoDatasetStorageError as exc:
        logger.error("demo dataset write failed for %s: %s", body.model_type, exc)
        return _error(str(exc), "storage_error", 500)

    logger.info(
        "demo dataset written: model_type=%s rows=%s seed=%s",
        body.model_type, metadata.get("rows_written"), seed,
    )
    return JSONResponse({**_describe(body.model_type), "exists": True, "metadata": metadata}, status_code=201)
