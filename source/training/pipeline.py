"""Training Pipeline for the closed-loop learning system.

Orchestrates SageMaker training jobs with NeMo-RL and registers versioned
model artifacts in the SageMaker Model Registry.

Key behaviors:
- Configurable retraining cadence (default 6h) and data window (default 7d)
- Combined supervised + RL loss via NeMo-RL
- At-most-one concurrent training job per ModelType
- Full lineage metadata on model registration

Requirements: 3.1, 3.2, 3.4, 3.5
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import boto3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """Supported model types in the bidding platform.

    NOTE: widedeep_segment_activator (audience segment activation) is
    intentionally absent — it is no longer a trainable neural model. Its ONNX
    graph could not be compiled to a TensorRT engine, so it was replaced with
    deterministic rule-based logic and is slated for replacement by a partner
    ISV implementation. See
    source/containers/widedeep_segment_activator/app.py.
    """

    DLRM_BID_SHADER = "dlrm_bid_shader"
    NCF_DEAL_MANAGER = "ncf_deal_manager"


@dataclass(frozen=True)
class TrainingJobConfig:
    """Configuration for a SageMaker training job.

    Attributes:
        model_type: Which model to retrain.
        base_model_version: Checkpoint URI/version to fine-tune from.
        training_data_uri: S3 URI to labeled Parquet training data.
        validation_split: Fraction of data reserved for validation (0.0-1.0).
        use_reinforcement_learning: Whether to include NeMo-RL loss.
        reward_function: Reward signal type - "roi", "ctr", or "revenue".
        rl_learning_rate: Learning rate for the RL optimizer.
        rl_epochs: Number of RL training epochs.
        instance_type: SageMaker instance type (e.g. "ml.p4d.24xlarge").
        instance_count: Number of training instances.
        max_runtime_seconds: Maximum wall-clock time for the training job.
        cadence_hours: Retraining cadence in hours (default 6).
        window_days: Training data window in days (default 7).
    """

    model_type: ModelType
    base_model_version: str
    training_data_uri: str
    validation_split: float
    use_reinforcement_learning: bool
    reward_function: str
    rl_learning_rate: float
    rl_epochs: int
    instance_type: str
    instance_count: int
    max_runtime_seconds: int
    cadence_hours: float = 6.0
    window_days: int = 7


@dataclass(frozen=True)
class TrainingResult:
    """Result of a completed SageMaker training job.

    Attributes:
        job_name: SageMaker training job name.
        model_artifact_uri: S3 URI to the trained model artifact.
        metrics: Offline metrics from training (e.g. {"ctr_auc": 0.82}).
        training_duration_s: Wall-clock training duration in seconds.
        model_version: Registered version identifier in Model Registry.
        data_window_days: Number of days of training data used.
        sample_count: Number of training samples in the dataset.
        base_model_version: Base model checkpoint used for fine-tuning.
    """

    job_name: str
    model_artifact_uri: str
    metrics: dict[str, float]
    training_duration_s: float
    model_version: str
    data_window_days: int = 7
    sample_count: int = 0
    base_model_version: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TrainingJobAlreadyRunningError(Exception):
    """Raised when attempting to start a training job while one is already
    running for the same ModelType.

    Requirement 3.4: At-most-one concurrent job per Model_Type.
    """

    def __init__(self, model_type: ModelType, existing_job_name: str):
        self.model_type = model_type
        self.existing_job_name = existing_job_name
        super().__init__(
            f"Training job '{existing_job_name}' is already running for "
            f"model type '{model_type.value}'. Only one concurrent job "
            f"per model type is allowed."
        )


class TrainingJobFailedError(Exception):
    """Raised when a SageMaker training job fails."""

    def __init__(self, job_name: str, failure_reason: str):
        self.job_name = job_name
        self.failure_reason = failure_reason
        super().__init__(
            f"Training job '{job_name}' failed: {failure_reason}"
        )


# ---------------------------------------------------------------------------
# Training image map per model type
# ---------------------------------------------------------------------------

# These are the SageMaker training container images configured per model type.
# In production these would come from a configuration store; here they serve as
# the default mapping.
# Training container image — built from source/training/container/Dockerfile
# and pushed to ECR during deployment. Uses NVIDIA NeMo Framework (nvcr.io/nvidia/nemo:24.07)
# with PyTorch, NeMo-RL, and ONNX export capabilities.
_TRAINING_IMAGE_MAP: dict[ModelType, str] = {
    ModelType.DLRM_BID_SHADER: "{account}.dkr.ecr.{region}.amazonaws.com/artf-nemo-rl-training:dlrm",
    ModelType.NCF_DEAL_MANAGER: "{account}.dkr.ecr.{region}.amazonaws.com/artf-nemo-rl-training:ncf",
}

# Model Package Group names in SageMaker Model Registry
_MODEL_PACKAGE_GROUP_MAP: dict[ModelType, str] = {
    ModelType.DLRM_BID_SHADER: "artf-dlrm-bid-shader",
    ModelType.NCF_DEAL_MANAGER: "artf-ncf-deal-manager",
}


# ---------------------------------------------------------------------------
# TrainingPipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Orchestrates model retraining via SageMaker + NeMo-RL.

    Manages the lifecycle of training jobs: launching, polling for completion,
    and registering trained models in the SageMaker Model Registry.

    Requirements: 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8
    """

    def __init__(
        self,
        sagemaker_role: str,
        model_bucket: str,
        region: str,
        training_data_bucket: str,
        sagemaker_client: Any | None = None,
        poll_interval_seconds: float = 30.0,
        failure_handler: Any | None = None,
    ):
        """Initialize the TrainingPipeline.

        Args:
            sagemaker_role: IAM role ARN for SageMaker training jobs.
            model_bucket: S3 bucket for model artifacts output.
            region: AWS region.
            training_data_bucket: S3 bucket containing training data.
            sagemaker_client: Optional pre-configured boto3 SageMaker client
                (useful for testing). If None, creates a new client.
            poll_interval_seconds: How often to poll job status (seconds).
            failure_handler: Optional TrainingFailureHandler for tracking
                failures, SNS alerting, and automatic pause logic.
        """
        self._sagemaker_role = sagemaker_role
        self._model_bucket = model_bucket
        self._region = region
        self._training_data_bucket = training_data_bucket
        self._poll_interval_seconds = poll_interval_seconds
        self._failure_handler = failure_handler

        if sagemaker_client is not None:
            self._sagemaker_client = sagemaker_client
        else:
            self._sagemaker_client = boto3.client(
                "sagemaker", region_name=region
            )

        # Track active jobs per model type for at-most-one enforcement
        self._active_jobs: dict[ModelType, str] = {}

    def _generate_job_name(self, model_type: ModelType) -> str:
        """Generate a unique training job name."""
        short_id = uuid.uuid4().hex[:8]
        timestamp = int(time.time())
        return f"{model_type.value}-{timestamp}-{short_id}"

    def _build_hyperparameters(self, config: TrainingJobConfig) -> dict[str, str]:
        """Build hyperparameters dict for the SageMaker training job."""
        return {
            "model_type": config.model_type.value,
            "base_model_version": config.base_model_version,
            "validation_split": str(config.validation_split),
            "use_reinforcement_learning": str(config.use_reinforcement_learning),
            "reward_function": config.reward_function,
            "rl_learning_rate": str(config.rl_learning_rate),
            "rl_epochs": str(config.rl_epochs),
            "window_days": str(config.window_days),
            "cadence_hours": str(config.cadence_hours),
        }

    async def trigger_retraining(
        self, config: TrainingJobConfig
    ) -> TrainingResult:
        """Launch a SageMaker training job and wait for completion.

        Enforces at-most-one concurrent job per ModelType. If a job for
        the requested model type is already running, raises
        TrainingJobAlreadyRunningError. If the model type is paused due
        to consecutive failures, raises TrainingPausedError.

        Args:
            config: Training job configuration.

        Returns:
            TrainingResult on successful completion.

        Raises:
            TrainingPausedError: If automated retraining is paused for
                this model type due to consecutive failures.
            TrainingJobAlreadyRunningError: If a job is already running
                for this model type.
            TrainingJobFailedError: If the training job fails.

        Requirements: 3.1, 3.2, 3.4, 3.7, 3.8
        """
        # Requirement 3.8: Check if model type is paused
        if self._failure_handler is not None:
            if self._failure_handler.is_paused(config.model_type):
                from training.failure_handler import TrainingPausedError

                raise TrainingPausedError(
                    config.model_type,
                    self._failure_handler.get_failure_count(config.model_type),
                )

        # Requirement 3.4: At-most-one concurrent job per Model_Type
        if config.model_type in self._active_jobs:
            existing_job = self._active_jobs[config.model_type]
            # Verify the job is actually still running
            if self._is_job_running(existing_job):
                raise TrainingJobAlreadyRunningError(
                    config.model_type, existing_job
                )
            else:
                # Previous job finished; clean up stale tracking entry
                del self._active_jobs[config.model_type]

        job_name = self._generate_job_name(config.model_type)
        output_path = (
            f"s3://{self._model_bucket}/models/{config.model_type.value}/{job_name}"
        )

        training_image = _TRAINING_IMAGE_MAP[config.model_type]

        # Requirement 3.2: Launch SageMaker training job with combined
        # supervised + RL loss
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

        # Create the training job
        self._sagemaker_client.create_training_job(**create_params)

        # Track active job
        self._active_jobs[config.model_type] = job_name

        logger.info(
            "Started training job '%s' for model type '%s'",
            job_name,
            config.model_type.value,
        )

        # Poll for completion
        try:
            final_status = await self._wait_for_job_completion(job_name)
        except Exception:
            # On any error, remove from active tracking
            self._active_jobs.pop(config.model_type, None)
            raise

        # Remove from active tracking on completion
        self._active_jobs.pop(config.model_type, None)

        training_duration_s = time.time() - start_time

        if final_status["TrainingJobStatus"] == "Failed":
            failure_reason = final_status.get(
                "FailureReason", "Unknown failure"
            )
            # Requirement 3.7: On failure, alert via SNS and retain current model
            if self._failure_handler is not None:
                await self._failure_handler.handle_failure(
                    config.model_type, job_name, failure_reason
                )
            raise TrainingJobFailedError(job_name, failure_reason)

        # Requirement 3.7: On success, reset failure counter
        if self._failure_handler is not None:
            await self._failure_handler.handle_success(config.model_type)

        # Extract metrics from the completed job
        metrics = self._extract_metrics(final_status)
        model_artifact_uri = final_status.get("ModelArtifacts", {}).get(
            "S3ModelArtifacts", f"{output_path}/output/model.tar.gz"
        )

        return TrainingResult(
            job_name=job_name,
            model_artifact_uri=model_artifact_uri,
            metrics=metrics,
            training_duration_s=training_duration_s,
            model_version="",  # Populated after registration
            data_window_days=config.window_days,
            sample_count=0,  # Populated from job metadata if available
            base_model_version=config.base_model_version,
        )

    async def register_model(
        self, result: TrainingResult, model_type: ModelType
    ) -> str:
        """Register trained model in SageMaker Model Registry.

        Includes full lineage metadata: training job name, data window,
        sample count, base version, and offline metrics. After successful
        registration, emits a ModelPackageGroupChanged event via EventBridge.

        Args:
            result: The TrainingResult from a completed training job.
            model_type: The model type being registered.

        Returns:
            The Model Package Version ARN.

        Requirements: 3.5, 3.6
        """
        model_package_group = _MODEL_PACKAGE_GROUP_MAP[model_type]

        # Build lineage metadata as custom properties
        customer_metadata = {
            "training_job_name": result.job_name,
            "model_artifact_uri": result.model_artifact_uri,
            "training_duration_seconds": str(result.training_duration_s),
            "data_window_days": str(result.data_window_days),
            "sample_count": str(result.sample_count),
            "base_model_version": result.base_model_version,
        }

        # Add all offline metrics to metadata
        for metric_name, metric_value in result.metrics.items():
            customer_metadata[f"metric_{metric_name}"] = str(metric_value)

        create_params = {
            "ModelPackageGroupName": model_package_group,
            "ModelPackageDescription": (
                f"Model trained by job {result.job_name} for {model_type.value}"
            ),
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": _TRAINING_IMAGE_MAP[model_type],
                        "ModelDataUrl": result.model_artifact_uri,
                    }
                ],
                "SupportedContentTypes": ["application/octet-stream"],
                "SupportedResponseMIMETypes": ["application/octet-stream"],
                "SupportedRealtimeInferenceInstanceTypes": [
                    "ml.g5.xlarge",
                    "ml.p4d.24xlarge",
                ],
            },
            "ModelApprovalStatus": "PendingManualApproval",
            "CustomerMetadataProperties": customer_metadata,
        }

        response = self._sagemaker_client.create_model_package(**create_params)
        model_package_arn = response["ModelPackageArn"]

        logger.info(
            "Registered model package '%s' for model type '%s' "
            "(training job: %s)",
            model_package_arn,
            model_type.value,
            result.job_name,
        )

        # Requirement 3.6: Emit ModelPackageGroupChanged event
        if self._failure_handler is not None:
            await self._failure_handler.emit_registration_event(
                model_type, model_package_arn
            )

        return model_package_arn

    def is_training(self, model_type: ModelType) -> bool:
        """Check if a training job is currently running for this model type.

        Args:
            model_type: The model type to check.

        Returns:
            True if a training job is actively running for this model type.
        """
        if model_type not in self._active_jobs:
            return False
        return self._is_job_running(self._active_jobs[model_type])

    def _is_job_running(self, job_name: str) -> bool:
        """Check if a training job is still in progress."""
        try:
            response = self._sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            status = response.get("TrainingJobStatus", "")
            return status in ("InProgress", "Stopping")
        except Exception:
            # If we can't describe the job, assume it's not running
            return False

    async def _wait_for_job_completion(self, job_name: str) -> dict:
        """Poll SageMaker until the training job reaches a terminal state.

        Returns the final describe_training_job response.
        """
        terminal_states = {"Completed", "Failed", "Stopped"}

        while True:
            response = self._sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            status = response.get("TrainingJobStatus", "")

            if status in terminal_states:
                return response

            logger.debug(
                "Training job '%s' status: %s — polling again in %ss",
                job_name,
                status,
                self._poll_interval_seconds,
            )
            await asyncio.sleep(self._poll_interval_seconds)

    def _extract_metrics(self, job_description: dict) -> dict[str, float]:
        """Extract final metrics from a completed training job description."""
        metrics: dict[str, float] = {}
        final_metrics = job_description.get("FinalMetricDataList", [])
        for metric in final_metrics:
            name = metric.get("MetricName", "")
            value = metric.get("Value")
            if name and value is not None:
                metrics[name] = float(value)
        return metrics
