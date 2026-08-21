"""XGBoost Training Pipeline for the Yield Optimizer (deal floor/margin) model.

Standalone, NOT a subclass of training.pipeline.TrainingPipeline -- that
class is architecturally NeMo-RL-specific (RL-shaped hyperparameters,
NeMo-RL training image), which has nothing to do with an XGBoost tree model.
Reuses TrainingPipeline's TrainingResult dataclass (generic enough to
describe any completed training job) but nothing else.

Key differences from TrainingPipeline:
- Training image: SageMaker's first-party built-in XGBoost algorithm
  container (sagemaker.image_uris.retrieve), not the custom NeMo-RL image.
- Hyperparameters: tree-model hyperparameters (max_depth, num_round, eta,
  objective), not RL-shaped ones (reward_function, rl_learning_rate).
- Concurrency: a live sagemaker.list_training_jobs() check, matching the
  approach the Train-from-Load-Test feature's on-demand trigger settled on
  (see source/orchestrator/training_trigger.py's
  is_training_in_progress()), not a new in-memory guard.

CORRECTION (found during Unit 3 implementation): Triton's FIL backend does
NOT support multi-output regression models (confirmed against NVIDIA's
official FIL backend docs). The originally-designed single
"deal_yield_manager" model/target has been split into two independent
single-target models -- TARGET_FLOOR ("floor_multiplier") and TARGET_MARGIN
("margin_value") -- each trained, registered, and promoted independently,
matching ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN being genuinely distinct
intents (see deal-yield-model/functional-design/business-rules.md BR-5).
XGBoostTrainingPipeline is now parametrized by `target` (one instance per
target); MODEL_TYPE/MODEL_PACKAGE_GROUP become per-target lookups.

Requirements: FR-12 (deal-floor-margin-requirements.md).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import boto3

from training.pipeline import TrainingResult

logger = logging.getLogger(__name__)

TARGET_FLOOR = "floor"
TARGET_MARGIN = "margin"
VALID_TARGETS = (TARGET_FLOOR, TARGET_MARGIN)

# target -> model_type (matches the Triton model name convention:
# deal_yield_manager_floor / deal_yield_manager_margin).
MODEL_TYPE_BY_TARGET: dict[str, str] = {
    TARGET_FLOOR: "deal_yield_manager_floor",
    TARGET_MARGIN: "deal_yield_manager_margin",
}

# target -> Model Package Group name -- matches closed_loop_cfn.yaml's
# DealYieldManagerFloorModelPackageGroup / ...Margin... naming (see
# infrastructure-design.md).
MODEL_PACKAGE_GROUP_BY_TARGET: dict[str, str] = {
    TARGET_FLOOR: "artf-deal-yield-manager-floor",
    TARGET_MARGIN: "artf-deal-yield-manager-margin",
}

# SageMaker's TrainingJobName/ListTrainingJobs NameContains only accept
# [a-zA-Z0-9\-]+ -- same constraint training_trigger.py's _job_name_prefix()
# documents for the NeMo-RL model types.
_JOB_NAME_PREFIX_BY_TARGET: dict[str, str] = {
    TARGET_FLOOR: "deal-yield-manager-floor",
    TARGET_MARGIN: "deal-yield-manager-margin",
}


def _validate_target(target: str) -> None:
    if target not in VALID_TARGETS:
        raise ValueError(
            f"Invalid target '{target}'. Must be one of {VALID_TARGETS} "
            f"(deal_yield_manager was split into independent floor/margin "
            f"models -- see module docstring correction note)."
        )


@dataclass(frozen=True)
class XGBoostTrainingJobConfig:
    """Configuration for an XGBoost SageMaker training job.

    Deliberately NOT training.pipeline.TrainingJobConfig -- that dataclass's
    fields (reward_function, rl_learning_rate, rl_epochs,
    use_reinforcement_learning) are RL-specific and meaningless for a tree
    model.
    """

    training_data_uri: str
    base_model_version: str
    validation_split: float
    max_depth: int
    num_round: int
    eta: float
    objective: str  # e.g. "reg:squarederror" for a floor-multiplier target
    instance_type: str
    instance_count: int
    max_runtime_seconds: int


class XGBoostTrainingJobAlreadyRunningError(Exception):
    """Raised when a training job is already in progress for this target's
    model, per the live ListTrainingJobs check (BR-3)."""

    def __init__(self, existing_job_name: str, model_type: str):
        self.existing_job_name = existing_job_name
        super().__init__(
            f"Training job '{existing_job_name}' is already running for "
            f"model type '{model_type}'. Only one concurrent job is allowed."
        )


class XGBoostTrainingPipeline:
    """Orchestrates XGBoost training jobs via SageMaker's built-in
    algorithm container for ONE Yield Optimizer target (floor or margin --
    see TARGET_FLOOR/TARGET_MARGIN). One instance per target; the two
    targets are trained, registered, and promoted completely independently
    (per the FIL multi-output-limitation correction -- see module
    docstring).
    """

    def __init__(
        self,
        target: str,
        sagemaker_role: str,
        model_bucket: str,
        region: str,
        training_data_bucket: str,
        sagemaker_client: Any | None = None,
        poll_interval_seconds: float = 30.0,
    ):
        _validate_target(target)
        self._target = target
        self._model_type = MODEL_TYPE_BY_TARGET[target]
        self._model_package_group = MODEL_PACKAGE_GROUP_BY_TARGET[target]
        self._job_name_prefix = _JOB_NAME_PREFIX_BY_TARGET[target]
        self._sagemaker_role = sagemaker_role
        self._model_bucket = model_bucket
        self._region = region
        self._training_data_bucket = training_data_bucket
        self._poll_interval_seconds = poll_interval_seconds
        self._sagemaker_client = sagemaker_client or boto3.client(
            "sagemaker", region_name=region
        )

    @property
    def target(self) -> str:
        return self._target

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_package_group(self) -> str:
        return self._model_package_group

    def _generate_job_name(self) -> str:
        short_id = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        return f"{self._job_name_prefix}-{timestamp}-{short_id}"

    def _training_image_uri(self) -> str:
        """Resolves SageMaker's first-party built-in XGBoost container URI.

        Uses the sagemaker SDK's image_uris.retrieve if available; falls
        back to a documented, version-pinned public ECR URI pattern if the
        sagemaker SDK isn't installed in this environment (the SDK is a
        build/deploy-time convenience, not a hard runtime dependency for
        this class -- boto3 alone is sufficient to call CreateTrainingJob).
        """
        try:
            from sagemaker import image_uris  # type: ignore

            return image_uris.retrieve(
                framework="xgboost", region=self._region, version="1.7-1"
            )
        except ImportError:
            logger.warning(
                "sagemaker SDK not installed; resolving XGBoost training "
                "image URI is required before calling trigger_training(). "
                "Install the 'sagemaker' package or pass a pre-resolved "
                "image URI."
            )
            raise

    def is_training_in_progress(self) -> tuple[bool, str | None]:
        """Live SageMaker ListTrainingJobs check -- the sole enforcement
        mechanism for at-most-one-concurrent-job (BR-3), matching the
        approach source/orchestrator/training_trigger.py already
        established for on-demand triggers.
        """
        resp = self._sagemaker_client.list_training_jobs(
            NameContains=f"{self._job_name_prefix}-",
            StatusEquals="InProgress",
            MaxResults=1,
            SortBy="CreationTime",
            SortOrder="Descending",
        )
        jobs = resp.get("TrainingJobSummaries", [])
        if jobs:
            return True, jobs[0]["TrainingJobName"]
        return False, None

    def _build_hyperparameters(self, config: XGBoostTrainingJobConfig) -> dict[str, str]:
        """Tree-model hyperparameters -- NOT the RL-shaped set
        TrainingPipeline._build_hyperparameters() builds."""
        return {
            "max_depth": str(config.max_depth),
            "num_round": str(config.num_round),
            "eta": str(config.eta),
            "objective": config.objective,
            "base_model_version": config.base_model_version,
            "validation_split": str(config.validation_split),
        }

    async def trigger_training(
        self, config: XGBoostTrainingJobConfig
    ) -> TrainingResult:
        """Launch a SageMaker training job using the built-in XGBoost
        algorithm container. Enforces at-most-one concurrent job via a
        live ListTrainingJobs check (BR-3).

        Raises XGBoostTrainingJobAlreadyRunningError if a job is already
        running for this model type.
        """
        in_progress, existing_job = self.is_training_in_progress()
        if in_progress:
            raise XGBoostTrainingJobAlreadyRunningError(existing_job or "", self._model_type)

        job_name = self._generate_job_name()
        output_path = f"s3://{self._model_bucket}/models/{self._model_type}/{job_name}"
        training_image = self._training_image_uri()

        create_params = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": training_image,
                "TrainingInputMode": "File",
            },
            "RoleArn": self._sagemaker_role,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": config.training_data_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "application/x-parquet",
                }
            ],
            "OutputDataConfig": {"S3OutputPath": output_path},
            "ResourceConfig": {
                "InstanceType": config.instance_type,
                "InstanceCount": config.instance_count,
                "VolumeSizeInGB": 100,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": config.max_runtime_seconds,
            },
            "HyperParameters": self._build_hyperparameters(config),
        }

        start_time = time.time()
        self._sagemaker_client.create_training_job(**create_params)

        logger.info("Started XGBoost training job '%s' for %s", job_name, self._model_type)

        final_status = await self._wait_for_job_completion(job_name)
        training_duration_s = time.time() - start_time

        if final_status["TrainingJobStatus"] == "Failed":
            failure_reason = final_status.get("FailureReason", "Unknown failure")
            raise RuntimeError(
                f"XGBoost training job '{job_name}' failed: {failure_reason}"
            )

        metrics = self._extract_metrics(final_status)
        model_artifact_uri = final_status.get("ModelArtifacts", {}).get(
            "S3ModelArtifacts", f"{output_path}/output/model.tar.gz"
        )

        return TrainingResult(
            job_name=job_name,
            model_artifact_uri=model_artifact_uri,
            metrics=metrics,
            training_duration_s=training_duration_s,
            model_version="",
            data_window_days=0,
            sample_count=0,
            base_model_version=config.base_model_version,
        )

    async def register_model(self, result: TrainingResult) -> str:
        """Register the trained XGBoost model in the Yield Optimizer's
        Model Package Group, mirroring TrainingPipeline.register_model()'s
        lineage-metadata pattern.
        """
        customer_metadata = {
            "training_job_name": result.job_name,
            "model_artifact_uri": result.model_artifact_uri,
            "training_duration_seconds": str(result.training_duration_s),
            "base_model_version": result.base_model_version,
        }
        for metric_name, metric_value in result.metrics.items():
            customer_metadata[f"metric_{metric_name}"] = str(metric_value)

        create_params = {
            "ModelPackageGroupName": self._model_package_group,
            "ModelPackageDescription": (
                f"XGBoost model trained by job {result.job_name} for {self._model_type}"
            ),
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": self._training_image_uri(),
                        "ModelDataUrl": result.model_artifact_uri,
                    }
                ],
                "SupportedContentTypes": ["application/octet-stream"],
                "SupportedResponseMIMETypes": ["application/octet-stream"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge"],
            },
            "ModelApprovalStatus": "PendingManualApproval",
            "CustomerMetadataProperties": customer_metadata,
        }

        response = self._sagemaker_client.create_model_package(**create_params)
        model_package_arn = response["ModelPackageArn"]
        logger.info(
            "Registered XGBoost model package '%s' for %s (training job: %s)",
            model_package_arn, self._model_type, result.job_name,
        )
        return model_package_arn

    async def _wait_for_job_completion(self, job_name: str) -> dict:
        import asyncio

        terminal_states = {"Completed", "Failed", "Stopped"}
        while True:
            response = self._sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            status = response.get("TrainingJobStatus", "")
            if status in terminal_states:
                return response
            await asyncio.sleep(self._poll_interval_seconds)

    def _extract_metrics(self, job_description: dict) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for metric in job_description.get("FinalMetricDataList", []):
            name = metric.get("MetricName", "")
            value = metric.get("Value")
            if name and value is not None:
                metrics[name] = float(value)
        return metrics
