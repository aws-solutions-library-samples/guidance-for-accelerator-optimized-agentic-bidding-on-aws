"""Governance panel API — orchestrator HTTP handlers for the Train-from-
Load-Test + Governance Outcome Comparison feature (Units 2 and 3).

Exposes:
- ``GET  /v1/governance/training-estimate?model_type=...``  — real cost/
  duration estimate before confirming a training trigger.
- ``POST /v1/governance/train`` — starts a real, fire-and-forget SageMaker
  training job (requires {"model_type": ..., "confirmed": true}).
- ``GET  /v1/governance/eligible-runs?model_type=...&role=current|challenger``
  — eligible load-test runs for comparison, auto-selecting the most recent.
- ``GET  /v1/governance/sweep-status[?run_id=...]`` — where a load-test run
  sits in the outcome-to-training-data pipeline (Firehose flush -> Glue sweep
  -> selectable for training), defaulting to the most recent run.
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
from orchestrator.etl_trigger import SWEEP_DELAY_SECONDS, resolve_glue_job_name
from orchestrator.loadtest_eligibility import list_eligible_runs, list_trainable_runs, most_recent_eligible
from orchestrator.promotion_service import PromotionNotRecommendedError, promote as run_promote
from orchestrator.sweep_status import GLUE_FAILED_STATES, as_utc, build_sweep_status
from orchestrator.training_trigger import (
    TRAINABLE_MODEL_TYPES,
    ModelTypeNotTrainableError,
    NoApprovedBaseVersionError,
    TrainingAlreadyInProgressError,
    TrainingNotConfirmedError,
    XGBoostTrainingImageNotConfiguredError,
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


def _glue_job_runs(job_name: str) -> list[dict]:
    """Returns the recent JobRuns for job_name (unfiltered by state), or [] when
    no job is configured.

    Unfiltered because the sweep-status poller needs the in-flight and failed
    runs that the trainability gate discards -- those are exactly what tells a
    user their run is mid-sweep rather than merely absent.
    """
    import boto3

    if not job_name:
        return []
    glue = boto3.client("glue", region_name=_REGION)
    resp = glue.get_job_runs(JobName=job_name, MaxResults=20)
    return resp.get("JobRuns", []) or []


def _latest_successful_completion(job_runs: list[dict]):
    """Returns the CompletedOn timestamp (UTC datetime) of the most recent
    SUCCEEDED run in job_runs, or None. Pure -- no I/O."""
    from datetime import timezone

    completions = [
        run["CompletedOn"] for run in job_runs
        if run.get("JobRunState") == "SUCCEEDED" and run.get("CompletedOn")
    ]
    if not completions:
        return None
    latest = max(completions)
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest


def _latest_glue_completion(model_type: str):
    """Returns the CompletedOn timestamp (UTC datetime) of the most recent
    SUCCEEDED run of the Glue job that labels model_type's training data,
    or None if no job is configured or no run has ever succeeded.

    The model_type -> job-name resolution lives in etl_trigger, which is also
    what starts an on-demand sweep after a load test. Keeping one mapping means
    the "which job covers this model" answer cannot differ between the read
    path (this gate) and the write path (the sweep) -- an earlier duplicate
    keyed on a non-existent bare "deal_yield_manager" type made every yield
    trainable-runs lookup silently resolve to "no Glue job configured."
    """
    job_name = resolve_glue_job_name(model_type)
    if not job_name:
        return None
    return _latest_successful_completion(_glue_job_runs(job_name))


def _run_summary(run: dict) -> dict:
    """The subset of a load-test history record the run pickers need."""
    return {
        "id": run.get("id"),
        "timestamp": run.get("timestamp"),
        "target_model_type": run.get("target_model_type") or "",
        "target_variant": run.get("target_variant") or "",
        "outcome_sample_count": int(run.get("outcome_sample_count") or 0),
        "preset": run.get("preset") or "",
        "state": run.get("state") or "",
    }


async def sweep_status_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/sweep-status[?run_id=lt-abc123]

    Returns where one load-test run sits in the outcome-to-training-data
    pipeline, plus the recent run list the status picker offers.

    ``run_id`` is optional: omitted, the most recent recorded run is used, so
    the panel opens on the load test the user most likely just finished.

    The ``trainable`` verdict is taken from list_trainable_runs -- the same
    function that populates the "Train from load test" picker -- so this card
    cannot report a run as ready while the picker still omits it.
    """
    from datetime import datetime, timezone

    requested_run_id = request.query_params.get("run_id", "")

    try:
        history_fn = _get_loadtest_history_fn()
        history = history_fn(limit=200)
        runs = [r for r in history if r.get("id")]

        selected = None
        if requested_run_id:
            selected = next((r for r in runs if r.get("id") == requested_run_id), None)
            if selected is None:
                return JSONResponse(
                    {
                        "error": f"Load test run '{requested_run_id}' is not in the recorded history.",
                        "reason": "run_not_found",
                        "runs": [_run_summary(r) for r in runs],
                    },
                    status_code=404,
                )
        elif runs:
            # _get_history_from_dynamodb already sorts newest-first.
            selected = runs[0]

        status = None
        if selected is not None:
            model_type = selected.get("target_model_type") or ""
            job_name = resolve_glue_job_name(model_type)
            job_runs = _glue_job_runs(job_name)
            latest_completion = _latest_successful_completion(job_runs)
            trainable = bool(list_trainable_runs([selected], model_type, latest_completion))
            status = build_sweep_status(
                selected,
                job_runs,
                job_name,
                now=datetime.now(timezone.utc),
                sweep_delay_seconds=SWEEP_DELAY_SECONDS,
                trainable=trainable,
            )
    except Exception as exc:
        logging.getLogger(__name__).exception("sweep_status_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        "runs": [_run_summary(r) for r in runs],
        "selected_run_id": (selected or {}).get("id") or "",
        "status": status,
    })


async def trainable_runs_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/trainable-runs?model_type=dlrm_bid_shader

    Returns {"runs": [...]} — load-test runs for model_type whose outcome
    data has actually been swept into training-data/ by a completed Glue
    job run (see loadtest_eligibility.list_trainable_runs). Each run
    includes its id, timestamp, target_model_type, and target_variant so
    the UI can label which model/test type a run was for.

    Also returns an ``etl`` block and a ``pending`` list so the picker can say
    WHY a run is missing instead of just showing a short list. Without it an
    empty or stale picker is indistinguishable between three very different
    situations: no load test has captured outcomes for this model, a run was
    captured but the next sweep has not happened yet, or the Glue job that
    labels this model's data is failing every run and no data will ever arrive.
    """
    model_type = request.query_params.get("model_type", "")

    try:
        job_name = resolve_glue_job_name(model_type)
        job_runs = _glue_job_runs(job_name)
        latest_completion = _latest_successful_completion(job_runs)
        history_fn = _get_loadtest_history_fn()
        history = history_fn(limit=200)
        runs = list_trainable_runs(history, model_type, latest_completion)

        # Runs that captured outcomes for this model but are not yet offered.
        # Almost always "recorded after the last successful sweep", which is the
        # single most common reason a user cannot find the run they just made.
        trainable_ids = {r.get("id") for r in runs}
        pending = [
            r for r in history
            if r.get("target_model_type") == model_type
            and int(r.get("outcome_sample_count") or 0) > 0
            and r.get("id") not in trainable_ids
        ]
        etl = _etl_health(job_name, job_runs, latest_completion)
    except Exception as exc:
        logging.getLogger(__name__).exception("trainable_runs_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        "runs": [
            {
                "id": r.get("id"),
                "timestamp": r.get("timestamp"),
                "target_model_type": r.get("target_model_type"),
                "target_variant": r.get("target_variant"),
                "outcome_sample_count": r.get("outcome_sample_count"),
            }
            for r in runs
        ],
        "etl": etl,
        "pending": [
            {
                "id": r.get("id"),
                "timestamp": r.get("timestamp"),
                "outcome_sample_count": int(r.get("outcome_sample_count") or 0),
            }
            for r in pending
        ],
    })


def _etl_health(job_name: str, job_runs: list[dict], latest_completion) -> dict:  # noqa: C901
    """Health of the Glue job that labels a model type's training data.

    Reports the job's own state rather than only its last success, so a picker
    can distinguish "waiting for the next scheduled sweep" from "this job fails
    every run, so waiting will never help". The yield job is currently the second
    case: it has never succeeded, failing each run with "Unable to infer schema
    for Parquet" because its input prefix is empty.

    ``last_error`` is taken from the most recent FAILED run regardless of when it
    started -- unlike sweep_status._describe_glue_runs, which only attributes a
    failure to a specific load test when the Glue run started after it. Here the
    question is "is this job working at all", which an older failure still
    answers.
    """
    from datetime import datetime, timezone

    if not job_name:
        return {
            "job_name": "",
            "configured": False,
            "last_success": None,
            "never_succeeded": True,
            "last_error": None,
            "consecutive_failures": 0,
        }

    failed = [r for r in job_runs if (r.get("JobRunState") or "").upper() in GLUE_FAILED_STATES]
    latest_failed = max(
        failed,
        key=lambda r: as_utc(r.get("StartedOn")) or datetime.min.replace(tzinfo=timezone.utc),
        default=None,
    )

    # job_runs comes back newest-first from Glue; count the unbroken run of
    # failures at the head, which is what makes a job "broken" rather than
    # "occasionally flaky".
    consecutive = 0
    for raw in job_runs:
        state = (raw.get("JobRunState") or "").upper()
        if state in GLUE_FAILED_STATES:
            consecutive += 1
        elif state == "SUCCEEDED":
            break

    return {
        "job_name": job_name,
        "configured": True,
        "last_success": latest_completion.isoformat() if latest_completion else None,
        "never_succeeded": latest_completion is None,
        "last_error": (
            {
                "started_on": (
                    as_utc(latest_failed.get("StartedOn")).isoformat()
                    if as_utc(latest_failed.get("StartedOn")) else None
                ),
                "message": latest_failed.get("ErrorMessage") or "",
            }
            if latest_failed is not None else None
        ),
        "consecutive_failures": consecutive,
    }


# Model types Triton serves through its FIL backend. FIL loads a native XGBoost
# artifact as-is, so staging a canary for these needs no TensorRT compile --
# which is what makes on-demand staging possible from the orchestrator. The
# nemo-rl-shaped types must be compiled first (see stage_canary_handler).
_FIL_MODEL_TYPES = frozenset({"deal_yield_manager_floor", "deal_yield_manager_margin"})


def _model_artifact_uri(sagemaker_client, version_arn: str) -> str:
    """The trained artifact S3 URI for a registered model package version.

    Prefers the CustomerMetadataProperties entry the registration Lambda writes
    (governance_eventbridge_cfn.yaml), falling back to the
    InferenceSpecification's ModelDataUrl, which SageMaker always populates.
    Returns "" when neither is present rather than guessing a path.
    """
    detail = sagemaker_client.describe_model_package(ModelPackageName=version_arn)
    from_metadata = (detail.get("CustomerMetadataProperties") or {}).get(
        "model_artifact_uri", ""
    )
    if from_metadata:
        return from_metadata
    containers = (detail.get("InferenceSpecification") or {}).get("Containers") or []
    return containers[0].get("ModelDataUrl", "") if containers else ""


async def stage_canary_handler(request: Request) -> JSONResponse:
    """POST /v1/governance/stage-canary

    Body: {"model_type": str, "version_arn": str}. Stages a registered model
    version as ``<model>_canary`` in the Triton repo so a challenger-targeted
    load test can reach it.

    Only the FIL-served model types are staged here. Triton's FIL backend reads
    XGBoost's native format directly, so there is nothing to compile. The
    TensorRT-served types (dlrm_bid_shader, ncf_deal_manager) need their ONNX
    compiled into a model.plan by the Model Optimizer first; that orchestration
    already lives in the Model Promotion Governance Agent, and this returns 422
    pointing there rather than duplicating it -- and rather than appearing to
    stage a canary that would never load.
    """
    body = await request.json()
    model_type = body.get("model_type", "")
    version_arn = body.get("version_arn", "")

    if not model_type or not version_arn:
        return JSONResponse(
            {"error": "model_type and version_arn are required.", "reason": "bad_request"},
            status_code=422,
        )

    if model_type not in _FIL_MODEL_TYPES:
        return JSONResponse(
            {
                "error": (
                    f"'{model_type}' is served by TensorRT, so a canary must be "
                    "compiled from ONNX by the Model Optimizer before it can be "
                    "staged. That runs in the Model Promotion Governance Agent, "
                    "which is triggered automatically when a new model version is "
                    "registered."
                ),
                "reason": "compile_required",
                "fil_model_types": sorted(_FIL_MODEL_TYPES),
            },
            status_code=422,
        )

    try:
        sagemaker_client = _sagemaker_client()
        artifact_uri = _model_artifact_uri(sagemaker_client, version_arn)
        if not artifact_uri:
            return JSONResponse(
                {
                    "error": (
                        f"No model artifact URI recorded for '{version_arn}', so "
                        "there is nothing to stage."
                    ),
                    "reason": "artifact_not_found",
                },
                status_code=422,
            )
        canary_model = await _triton_loader().stage_canary_fil(model_type, artifact_uri)
    except Exception as exc:
        logging.getLogger(__name__).exception("stage_canary_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        "model_type": model_type,
        "version_arn": version_arn,
        "artifact_uri": artifact_uri,
        "canary_model": canary_model,
        # Triton poll-loads from the repo, so the model is not servable the
        # instant this returns. The caller should poll readiness (the load-test
        # panel already refuses a challenger run until the canary reports ready)
        # rather than assume it is live.
        "ready": False,
    }, status_code=202)


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

    Body: {"model_type": str, "confirmed": bool, "run_id"?: str}.
    Returns 202 with the started job's name/base_model_version on success.
    Returns 422 for a precondition failure (not trainable, not confirmed,
    already in progress, no approved base version) — reported plainly,
    never silently substituting a different outcome.

    ``run_id`` is the load-test run the operator selected in the picker. It is
    recorded as training provenance so a later comparison can use that run as its
    control. The UI collected this selection but never sent it, so the resulting
    model version had no link back to the run it was triggered from.

    Provenance only: the training input is an entire S3 prefix containing every
    swept run, so this identifies the run the job was triggered FROM, not the only
    data it learned from.
    """
    body = await request.json()
    model_type = body.get("model_type", "")
    confirmed = bool(body.get("confirmed", False))
    run_id = body.get("run_id", "") or ""

    sagemaker_role_arn = os.environ.get("SAGEMAKER_TRAINING_ROLE_ARN", "")
    model_bucket = os.environ.get("MODEL_BUCKET", "")
    training_data_bucket = os.environ.get("TRAINING_DATA_BUCKET", "")
    training_image_registry = os.environ.get("TRAINING_IMAGE_REGISTRY", "")
    # Only required for deal_yield_manager_floor/margin (xgboost-shaped
    # training -- see training_trigger._TRAINING_SHAPE); unset/empty is
    # fine for dlrm_bid_shader, which never reads it.
    xgboost_training_image_uri = os.environ.get("XGBOOST_TRAINING_IMAGE_URI", "") or None

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
            xgboost_training_image_uri=xgboost_training_image_uri,
            load_test_run_id=run_id,
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
    except XGBoostTrainingImageNotConfiguredError as exc:
        return JSONResponse({"error": str(exc), "reason": "xgboost_image_not_configured"}, status_code=503)
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


async def comparison_pair_handler(request: Request) -> JSONResponse:
    """GET /v1/governance/comparison-pair?model_type=dlrm_bid_shader

    Suggests which two runs to compare, so the operator does not have to
    remember which load test a challenger was trained from.

    The control it prefers is the run recorded as the challenger version's
    training trigger (``training_run_id``, written through the provenance chain in
    training_trigger/governance_eventbridge_cfn.yaml). That answers the question
    the comparison is really asking: did training on this data beat what that data
    itself produced?

    Reports ``control_source`` so the caller can label the pair truthfully rather
    than implying provenance it does not have:

    - ``training_provenance`` — the challenger version records its originating run
      AND that run is still eligible as a control.
    - ``most_recent`` — no usable provenance, so this falls back to the most
      recent eligible current-variant run. Honest but weaker: it is not
      necessarily the data the challenger learned from.
    - ``none`` — nothing eligible to suggest.

    Suggestion only. It performs no comparison, and the caller may override
    either side; POST /v1/governance/compare still takes both ids explicitly.
    """
    model_type = request.query_params.get("model_type", "")

    try:
        history_fn = _get_loadtest_history_fn()
        history = history_fn(limit=200)
        eligible_current = list_eligible_runs(history, model_type, "current")
        eligible_ids = {r.get("id") for r in eligible_current}

        provenance_run_id = ""
        challenger_version_arn = ""
        try:
            # Imported here, matching training_trigger._resolve_base_model_version's
            # own lazy import of _model_group, which exists to avoid a module-level
            # cycle between the two API modules.
            from closed_loop_demo import readers
            from orchestrator.closed_loop_api import _model_group

            versions = readers.read_model_versions(
                _sagemaker_client(), _model_group(model_type), limit=20
            )
        except Exception:  # noqa: BLE001 — a registry failure must not block a manual pick
            logging.getLogger(__name__).exception("comparison_pair_handler: registry read failed")
            versions = []
        for version in versions:
            if version.get("training_run_id"):
                provenance_run_id = version["training_run_id"]
                challenger_version_arn = version.get("model_package_arn") or ""
                break

        if provenance_run_id and provenance_run_id in eligible_ids:
            control_run_id = provenance_run_id
            control_source = "training_provenance"
        else:
            fallback = most_recent_eligible(history, model_type, "current")
            control_run_id = (fallback or {}).get("id") or ""
            control_source = "most_recent" if control_run_id else "none"

        challenger = most_recent_eligible(history, model_type, "challenger")
        challenger_run_id = (challenger or {}).get("id") or ""
    except Exception as exc:
        logging.getLogger(__name__).exception("comparison_pair_handler failed unexpectedly")
        return JSONResponse({"error": str(exc), "reason": "internal_error"}, status_code=500)

    return JSONResponse({
        "model_type": model_type,
        "control_run_id": control_run_id,
        "control_source": control_source,
        "challenger_run_id": challenger_run_id,
        "challenger_version_arn": challenger_version_arn,
        # Set when a version records provenance but that run is no longer an
        # eligible control (aged out of history, or captured no samples). Surfaced
        # rather than silently falling back, so the weaker pairing is visible.
        "provenance_run_unavailable": bool(
            provenance_run_id and provenance_run_id not in eligible_ids
        ),
        "provenance_run_id": provenance_run_id,
    })


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
