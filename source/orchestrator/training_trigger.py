"""On-demand training trigger for the Governance panel (TrainingTriggerService).

Provides a real cost/duration estimate, a live SageMaker-backed concurrency
guard, and a fire-and-forget CreateTrainingJob call — matching the existing
scheduled EventBridge Lambda's pattern (deployment/governance_eventbridge_cfn.yaml's
RetrainingTriggerFunction) rather than TrainingPipeline.trigger_retraining(),
which blocks polling to completion (unsuitable for a UI request/response
cycle given jobs run for hours).

Maps to: FR-4, FR-5 (Story 3, train-from-load-test unit).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Literal

import boto3

ModelTypeStr = Literal["dlrm_bid_shader", "ncf_deal_manager"]

# Real, current on-demand SageMaker Training pricing for the instance type
# the automated retraining pipeline already uses (verified via
# `aws pricing get-products --service-code AmazonSageMaker --filters
# usagetype=USE1-Train:ml.g5.2xlarge`, us-east-1, 2026-08-20). Not a
# fabricated number — refresh this if the automated pipeline's instance
# type or region changes.
_INSTANCE_TYPE = "ml.g5.2xlarge"
_HOURLY_RATE_USD = 1.515

# Matches deployment/governance_eventbridge_cfn.yaml's
# RetrainingTriggerFunction StoppingCondition.MaxRuntimeInSeconds — the same
# ceiling the automated pipeline uses for the same instance type.
_MAX_RUNTIME_SECONDS = 14400

# ncf_deal_manager is intentionally excluded from on-demand training. Its
# ACTIVATE_DEALS/SUPPRESS_DEALS mutations disambiguate deals via
# path + IDsPayload (a list of deal IDs per impression) per the real ARTF
# proto (agenticrtbframework.proto) -- verified directly against
# github.com/IABTechLab/agentic-real-time-framework's canonical proto and
# reference-implementation handler (ProcessDeals in internal/handlers/
# handlers.go), which builds exactly one Mutation per impression carrying a
# list of deal IDs. BidShadingOutcomeEvent/Record (shared/feedback_models.py)
# has no deal_id field and no per-deal fan-out, so there is no way to
# attribute a training outcome to one specific deal today. Enabling this
# requires a schema change (deal_id field + per-deal event fan-out in
# feedback_integration.py, using OpenRTB's real BidResponse.SeatBid.Bid.dealid
# to determine which activated deal actually won) -- tracked as a follow-up,
# not implemented here. train.py's build_features() already raises a clear
# ValueError for ncf_deal_manager rather than silently misbehaving; this
# set is the earlier, UI-facing enforcement point so the option is never
# offered in the first place.
TRAINABLE_MODEL_TYPES: frozenset[str] = frozenset({"dlrm_bid_shader"})

# Model types with real training infrastructure (container/pipeline
# wiring exists) but temporarily excluded from TRAINABLE_MODEL_TYPES above.
# Kept separate (rather than deleted) so the reason is documented in one
# place and the set is trivial to restore once the deal_id schema work
# lands.
_PARKED_MODEL_TYPES: frozenset[str] = frozenset({"ncf_deal_manager"})

# Same training image naming convention as source/training/pipeline.py's
# _TRAINING_IMAGE_MAP and governance_eventbridge_cfn.yaml's image_tag.
# Includes parked model types too -- this map describes image-naming
# convention, not what's currently offered (that's TRAINABLE_MODEL_TYPES).
_IMAGE_TAG = {"dlrm_bid_shader": "dlrm", "ncf_deal_manager": "ncf"}


class ModelTypeNotTrainableError(Exception):
    """Raised when a training trigger is requested for a model type with no
    training infrastructure (e.g. the two rule-based container types)."""

    def __init__(self, model_type: str):
        super().__init__(
            f"Model type '{model_type}' has no training infrastructure. "
            f"Trainable model types: {sorted(TRAINABLE_MODEL_TYPES)}."
        )
        self.model_type = model_type


class TrainingNotConfirmedError(Exception):
    """Raised when trigger_training() is called without confirmed=True."""

    def __init__(self, model_type: str):
        super().__init__(
            f"Training for '{model_type}' requires explicit confirmation "
            "(confirmed=True) after the cost/duration estimate is shown."
        )
        self.model_type = model_type


class TrainingAlreadyInProgressError(Exception):
    """Raised when a training job for this model type is already running,
    per the live SageMaker ListTrainingJobs check (the actual enforcement
    mechanism, per the resolved Application Design Follow-up Question A)."""

    def __init__(self, model_type: str, job_name: str):
        super().__init__(
            f"A training job ('{job_name}') is already running for model "
            f"type '{model_type}'. Only one concurrent job per model type "
            "is allowed."
        )
        self.model_type = model_type
        self.job_name = job_name


class NoApprovedBaseVersionError(Exception):
    """Raised when there is no Approved registry version to fine-tune from
    (mirrors the scheduled Lambda's honest skip — never fabricates a
    base_model_version)."""

    def __init__(self, model_type: str):
        super().__init__(
            f"No Approved SageMaker Model Registry version exists for "
            f"'{model_type}' yet — cannot resolve a base_model_version to "
            "fine-tune from."
        )
        self.model_type = model_type


@dataclass(frozen=True)
class TrainingCostEstimate:
    """Real cost/duration estimate for a training job, shown before confirm."""

    instance_type: str
    hourly_rate_usd: float
    max_runtime_seconds: int
    estimated_max_cost_usd: float
    estimated_max_duration_seconds: int


@dataclass(frozen=True)
class TrainingTriggerResult:
    """Result of successfully starting a training job (fire-and-forget)."""

    job_name: str
    model_type: str
    base_model_version: str
    instance_type: str


def estimate_cost(model_type: str) -> TrainingCostEstimate:
    """Pure function. Computes the cost/duration ceiling for model_type.

    Real numbers: the actual configured instance type/hourly rate (verified
    AWS pricing, not fabricated) and the same MaxRuntimeInSeconds ceiling the
    automated pipeline already uses for this instance type. This is a
    ceiling (worst case if the job runs the full allowed duration), not a
    prediction of actual duration — SageMaker training jobs are billed by
    the second, so actual cost is typically lower.
    """
    max_cost = round((_MAX_RUNTIME_SECONDS / 3600.0) * _HOURLY_RATE_USD, 2)
    return TrainingCostEstimate(
        instance_type=_INSTANCE_TYPE,
        hourly_rate_usd=_HOURLY_RATE_USD,
        max_runtime_seconds=_MAX_RUNTIME_SECONDS,
        estimated_max_cost_usd=max_cost,
        estimated_max_duration_seconds=_MAX_RUNTIME_SECONDS,
    )


def _sagemaker_client():
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    return boto3.client("sagemaker", region_name=region)


def _job_name_prefix(model_type: str) -> str:
    """SageMaker resource-name-safe prefix for model_type.

    ``TrainingJobName`` and ``ListTrainingJobs``'s ``NameContains`` only
    accept ``[a-zA-Z0-9\\-]+`` — model_type values contain underscores
    (e.g. "dlrm_bid_shader"), which SageMaker rejects with a
    ValidationException on both APIs. Translating to hyphens here is the
    only change; model_type itself (used for HyperParameters, Model
    Package Group lookup, etc.) is left untouched.
    """
    return model_type.replace("_", "-")


def is_training_in_progress(model_type: str) -> tuple[bool, str | None]:
    """Live SageMaker ListTrainingJobs check — the actual enforcement
    mechanism (per the resolved Application Design Follow-up Question A;
    TrainingPipeline's own internal in-memory guard remains a secondary,
    non-authoritative backstop for its own blocking trigger_retraining()
    call path, not used by this on-demand trigger).

    Returns (in_progress, job_name_if_any).
    """
    client = _sagemaker_client()
    resp = client.list_training_jobs(
        NameContains=f"{_job_name_prefix(model_type)}-",
        StatusEquals="InProgress",
        MaxResults=1,
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    jobs = resp.get("TrainingJobSummaries", [])
    if jobs:
        return True, jobs[0]["TrainingJobName"]
    return False, None


def _resolve_base_model_version(model_type: str) -> str:
    """Resolve base_model_version from the current Approved registry
    version — same resolution logic the scheduled Lambda already uses
    (governance_eventbridge_cfn.yaml's _latest_approved_version).

    Reuses closed_loop_api._model_group() for the Model Package Group name
    so this stays consistent with the same stack-prefix override mechanism
    (CLOSED_LOOP_MODEL_GROUP_<MODELTYPE>) the rest of the closed-loop API
    already uses — never duplicates that naming logic.
    """
    from orchestrator.closed_loop_api import _model_group

    client = _sagemaker_client()
    package_group = _model_group(model_type)
    resp = client.list_model_packages(
        ModelPackageGroupName=package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = resp.get("ModelPackageSummaryList", [])
    if not packages:
        raise NoApprovedBaseVersionError(model_type)
    return packages[0]["ModelPackageArn"]


def trigger_training(
    model_type: str,
    confirmed: bool,
    *,
    sagemaker_role_arn: str,
    model_bucket: str,
    training_data_bucket: str,
    training_image_registry: str,
) -> TrainingTriggerResult:
    """Start a real, fire-and-forget SageMaker training job.

    Raises ModelTypeNotTrainableError, TrainingNotConfirmedError,
    TrainingAlreadyInProgressError, or NoApprovedBaseVersionError before
    ever calling CreateTrainingJob — the caller (the new
    POST /v1/governance/train route) is expected to have already shown
    estimate_cost()'s result and gotten explicit user confirmation.
    """
    if model_type not in TRAINABLE_MODEL_TYPES:
        raise ModelTypeNotTrainableError(model_type)
    if not confirmed:
        raise TrainingNotConfirmedError(model_type)

    in_progress, existing_job = is_training_in_progress(model_type)
    if in_progress:
        raise TrainingAlreadyInProgressError(model_type, existing_job or "")

    base_model_version = _resolve_base_model_version(model_type)

    job_name = f"{_job_name_prefix(model_type)}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    training_image = f"{training_image_registry}/artf-nemo-rl-training:{_IMAGE_TAG[model_type]}"
    output_path = f"s3://{model_bucket}/models/{model_type}/{job_name}"

    client = _sagemaker_client()
    client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
        },
        RoleArn=sagemaker_role_arn,
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{training_data_bucket}/training-data/",
                "S3DataDistributionType": "FullyReplicated",
            }},
            "ContentType": "application/x-parquet",
        }],
        OutputDataConfig={"S3OutputPath": output_path},
        ResourceConfig={
            "InstanceType": _INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": _MAX_RUNTIME_SECONDS},
        HyperParameters={
            "model_type": model_type,
            "base_model_version": base_model_version,
            "window_days": "7",
            "cadence_hours": "6.0",
            "triggered_by": "governance_ui_on_demand",
        },
    )

    return TrainingTriggerResult(
        job_name=job_name,
        model_type=model_type,
        base_model_version=base_model_version,
        instance_type=_INSTANCE_TYPE,
    )
