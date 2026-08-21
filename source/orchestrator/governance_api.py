"""Governance panel API — orchestrator HTTP handlers for the Train-from-
Load-Test + Governance Outcome Comparison feature (Units 2 and 3).

Exposes:
- ``GET  /v1/governance/training-estimate?model_type=...``  — real cost/
  duration estimate before confirming a training trigger.
- ``POST /v1/governance/train`` — starts a real, fire-and-forget SageMaker
  training job (requires {"model_type": ..., "confirmed": true}).
- ``GET  /v1/governance/eligible-runs?model_type=...&role=current|challenger``
  — eligible load-test runs for comparison, auto-selecting the most recent.
- ``POST /v1/governance/compare`` — real per-sample ABEvaluator comparison
  between a current-version run and a challenger-version run.
- ``POST /v1/governance/promote`` — the real Promote action (Triton
  publish + registry update + audit write), only for a promote-recommending
  comparison result.

Configuration (environment variables, matching the existing closed-loop
API's naming conventions):
- ``SAGEMAKER_TRAINING_ROLE_ARN`` — IAM role SageMaker assumes for the job.
- ``MODEL_BUCKET`` — S3 bucket for training output artifacts / Triton model repo.
- ``TRAINING_DATA_BUCKET`` — S3 bucket containing labeled training data.
- ``TRAINING_IMAGE_REGISTRY`` — ECR registry hosting the NeMo-RL training image.
- ``TRITON_URL`` — cluster-internal Triton endpoint (already used elsewhere
  in the orchestrator, e.g. app.py's health-check routes).
- ``AUDIT_TRAIL_TABLE`` — the real audit-trail DynamoDB table (already used
  by closed_loop_api.py).
"""

from __future__ import annotations

import logging
import os

from starlette.requests import Request
from starlette.responses import JSONResponse

from orchestrator.comparison_service import (
    ComparisonRequest,
    InsufficientSamplesError,
    compare as run_comparison,
)
from orchestrator.loadtest_eligibility import list_eligible_runs, most_recent_eligible
from orchestrator.promotion_service import PromotionNotRecommendedError, promote as run_promote
from orchestrator.training_trigger import (
    TRAINABLE_MODEL_TYPES,
    ModelTypeNotTrainableError,
    NoApprovedBaseVersionError,
    TrainingAlreadyInProgressError,
    TrainingNotConfirmedError,
    estimate_cost,
    trigger_training,
)

_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
_AUDIT_TRAIL_TABLE = os.environ.get("AUDIT_TRAIL_TABLE", "audit-trail")


def _get_loadtest_history_fn():
    """Lazy import to avoid a circular import with orchestrator.loadtest,
    which itself imports from orchestrator.loadtest_targeting/instrumentation
    (mirrors the existing lazy-import pattern in loadtest.py's _get_app_deps)."""
    try:
        from orchestrator.loadtest import _get_history_from_dynamodb
    except ImportError:
        from container.loadtest import _get_history_from_dynamodb
    return _get_history_from_dynamodb


def _triton_loader():
    """Construct a real TritonModelLoader against the cluster-internal
    Triton endpoint — reachable directly from the orchestrator pod (same
    TRITON_URL env var app.py's own health-check routes already use), no
    VPC proxy needed since the orchestrator itself runs inside the cluster
    (unlike the governance AgentCore runtime, which may run PUBLIC and
    needs a VPC-proxy Lambda — see agents/governance/handler.py).

    Reuses the same HttpxClient adapter the governance agent uses for its
    own TritonModelLoader (agents/governance/integrations.py) rather than
    building a second HTTP-client implementation.
    """
    from agents.governance.integrations import HttpxClient
    from deployment.model_deployer import TritonModelLoader

    triton_url = os.environ.get("TRITON_URL", "triton-inference-server:8000")
    model_bucket = os.environ.get("MODEL_BUCKET", "")
    http_client = HttpxClient(region=_REGION)
    return TritonModelLoader(
        triton_url=f"http://{triton_url}",
        model_bucket=model_bucket,
        http_client=http_client,
        region=_REGION,
    )


def _sagemaker_client():
    import boto3

    return boto3.client("sagemaker", region_name=_REGION)


def _audit_table():
    import boto3

    dynamodb = boto3.resource("dynamodb", region_name=_REGION)
    return dynamodb.Table(_AUDIT_TRAIL_TABLE)


async def training_estimate_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/training-estimate?model_type=dlrm_bid_shader

    Returns the real cost/duration estimate for training model_type, or a
    422 if model_type is missing/not trainable.
    """
    model_type = request.query_params.get("model_type", "")

    if model_type not in TRAINABLE_MODEL_TYPES:
        return JSONResponse(
            {
                "error": f"Model type '{model_type}' has no training infrastructure.",
                "trainable_model_types": sorted(TRAINABLE_MODEL_TYPES),
            },
            status_code=422,
        )

    estimate = estimate_cost(model_type)
    return JSONResponse({
        "model_type": model_type,
        "instance_type": estimate.instance_type,
        "hourly_rate_usd": estimate.hourly_rate_usd,
        "max_runtime_seconds": estimate.max_runtime_seconds,
        "estimated_max_cost_usd": estimate.estimated_max_cost_usd,
        "estimated_max_duration_seconds": estimate.estimated_max_duration_seconds,
    })


async def train_handler(request: Request) -> JSONResponse:
    """POST /v1/governance/train

    Body: {"model_type": str, "confirmed": bool}.
    Returns 202 with the started job's name/base_model_version on success.
    Returns 422 for a precondition failure (not trainable, not confirmed,
    already in progress, no approved base version) — reported plainly,
    never silently substituting a different outcome.
    """
    body = await request.json()
    model_type = body.get("model_type", "")
    confirmed = bool(body.get("confirmed", False))

    sagemaker_role_arn = os.environ.get("SAGEMAKER_TRAINING_ROLE_ARN", "")
    model_bucket = os.environ.get("MODEL_BUCKET", "")
    training_data_bucket = os.environ.get("TRAINING_DATA_BUCKET", "")
    training_image_registry = os.environ.get("TRAINING_IMAGE_REGISTRY", "")

    if not all([sagemaker_role_arn, model_bucket, training_data_bucket, training_image_registry]):
        return JSONResponse(
            {"error": "Training trigger is not configured on this deployment "
                      "(missing SAGEMAKER_TRAINING_ROLE_ARN/MODEL_BUCKET/"
                      "TRAINING_DATA_BUCKET/TRAINING_IMAGE_REGISTRY)."},
            status_code=503,
        )

    try:
        result = trigger_training(
            model_type,
            confirmed,
            sagemaker_role_arn=sagemaker_role_arn,
            model_bucket=model_bucket,
            training_data_bucket=training_data_bucket,
            training_image_registry=training_image_registry,
        )
    except ModelTypeNotTrainableError as exc:
        return JSONResponse({"error": str(exc), "reason": "not_trainable"}, status_code=422)
    except TrainingNotConfirmedError as exc:
        return JSONResponse({"error": str(exc), "reason": "not_confirmed"}, status_code=422)
    except TrainingAlreadyInProgressError as exc:
        return JSONResponse(
            {"error": str(exc), "reason": "already_in_progress", "job_name": exc.job_name},
            status_code=422,
        )
    except NoApprovedBaseVersionError as exc:
        return JSONResponse({"error": str(exc), "reason": "no_approved_base_version"}, status_code=422)
    except Exception as exc:
        # Fail-safe: any other error (e.g. a boto3 ClientError) must still
        # return valid JSON, never fall through to a plain-text 500 that
        # breaks the UI's resp.json() call.
        logging.getLogger(__name__).exception("train_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        "job_name": result.job_name,
        "model_type": result.model_type,
        "base_model_version": result.base_model_version,
        "instance_type": result.instance_type,
    }, status_code=202)


async def eligible_runs_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/eligible-runs?model_type=...&role=current|challenger

    Returns {"runs": [...], "most_recent": {...} | None}. "runs" lists all
    eligible past runs for the dropdown's manual-override option; the
    result is not paginated (matches the existing GET /v1/loadtest/history
    endpoint's own unpaginated shape).
    """
    model_type = request.query_params.get("model_type", "")
    role = request.query_params.get("role", "current")

    if role not in ("current", "challenger"):
        return JSONResponse({"error": "role must be 'current' or 'challenger'"}, status_code=422)

    try:
        history_fn = _get_loadtest_history_fn()
        history = history_fn(limit=200)

        eligible = list_eligible_runs(history, model_type, role)
        most_recent = most_recent_eligible(history, model_type, role)
    except Exception as exc:
        logging.getLogger(__name__).exception("eligible_runs_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({"runs": eligible, "most_recent": most_recent})


async def compare_handler(request: Request) -> JSONResponse:
    """POST /v1/governance/compare

    Body: {"model_type": str, "current_run_id": str, "challenger_run_id": str}.
    Loads both runs' real outcome_samples from history and calls the real
    ABEvaluator. Returns 422 if either run isn't found or has no captured
    samples — never silently substitutes a different run or fabricates data.
    """
    body = await request.json()
    model_type = body.get("model_type", "")
    current_run_id = body.get("current_run_id", "")
    challenger_run_id = body.get("challenger_run_id", "")

    try:
        history_fn = _get_loadtest_history_fn()
        history = history_fn(limit=200)
        by_id = {run.get("id"): run for run in history}

        current_run = by_id.get(current_run_id)
        challenger_run = by_id.get(challenger_run_id)
        if current_run is None:
            return JSONResponse({"error": f"Run '{current_run_id}' not found."}, status_code=422)
        if challenger_run is None:
            return JSONResponse({"error": f"Run '{challenger_run_id}' not found."}, status_code=422)

        request_obj = ComparisonRequest(
            model_type=model_type,
            current_run_id=current_run_id,
            current_samples=[float(v) for v in (current_run.get("outcome_samples") or [])],
            challenger_run_id=challenger_run_id,
            challenger_samples=[float(v) for v in (challenger_run.get("outcome_samples") or [])],
        )

        result = run_comparison(request_obj)
    except InsufficientSamplesError as exc:
        return JSONResponse({"error": str(exc), "reason": "insufficient_samples"}, status_code=422)
    except Exception as exc:
        logging.getLogger(__name__).exception("compare_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        # Source attribution (FR-10) — plain factual labeling.
        "source": "load_test_runs",
        "current_run_id": current_run_id,
        "current_run_timestamp": current_run.get("timestamp"),
        "current_run_model_version": current_run.get("model_version"),
        "challenger_run_id": challenger_run_id,
        "challenger_run_timestamp": challenger_run.get("timestamp"),
        "challenger_run_model_version": challenger_run.get("model_version"),
        "status": result.status.value,
        "control_metric": result.control_metric,
        "treatment_metric": result.treatment_metric,
        "relative_lift": result.relative_lift,
        "p_value": result.p_value,
        "samples_control": result.samples_control,
        "samples_treatment": result.samples_treatment,
        "guardrail_violations": result.guardrail_violations,
        "recommendation": result.recommendation,
    })


async def promote_handler(request: Request) -> JSONResponse:
    """POST /v1/governance/promote

    Body: {"model_type": str, "version_arn": str, "recommendation": str,
    "reason": str}. Only "promote" recommendations may proceed — a reject/
    inconclusive result returns 422 without any side effect.
    """
    body = await request.json()
    model_type = body.get("model_type", "")
    version_arn = body.get("version_arn", "")
    recommendation = body.get("recommendation", "")
    reason = body.get("reason", "")

    try:
        result = await run_promote(
            model_type=model_type,
            version_arn=version_arn,
            recommendation=recommendation,
            reason=reason,
            triton_loader=_triton_loader(),
            sagemaker_client=_sagemaker_client(),
            audit_table=_audit_table(),
        )
    except PromotionNotRecommendedError as exc:
        return JSONResponse(
            {"error": str(exc), "reason": "not_recommended"}, status_code=422
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("promote_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        "model_type": result.model_type,
        "version_arn": result.version_arn,
        "audit_record_id": result.audit_record_id,
    }, status_code=200)
