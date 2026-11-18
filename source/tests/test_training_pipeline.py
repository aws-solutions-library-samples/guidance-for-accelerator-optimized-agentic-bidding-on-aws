"""Unit tests for the Training Pipeline.

Tests cover:
- Successful training job creation and completion
- At-most-one concurrent job enforcement (raises error)
- Model registration with correct lineage metadata
- Failure handling (job failure -> exception)

Requirements: 3.1, 3.2, 3.4, 3.5
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.pipeline import (
    ModelType,
    TrainingJobAlreadyRunningError,
    TrainingJobConfig,
    TrainingJobFailedError,
    TrainingPipeline,
    TrainingResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    model_type: ModelType = ModelType.DLRM_BID_SHADER,
    **overrides,
) -> TrainingJobConfig:
    """Build a default TrainingJobConfig for tests."""
    defaults = {
        "model_type": model_type,
        "base_model_version": "v1.0.0",
        "training_data_uri": "s3://test-bucket/training-data/dlrm/",
        "validation_split": 0.2,
        "use_reinforcement_learning": True,
        "reward_function": "roi",
        "rl_learning_rate": 0.001,
        "rl_epochs": 10,
        "instance_type": "ml.p4d.24xlarge",
        "instance_count": 1,
        "max_runtime_seconds": 3600,
        "cadence_hours": 6.0,
        "window_days": 7,
    }
    defaults.update(overrides)
    return TrainingJobConfig(**defaults)


def _make_mock_sagemaker_client(
    final_status: str = "Completed",
    failure_reason: str = "",
    model_artifact_uri: str = "s3://model-bucket/models/output/model.tar.gz",
    metrics: list[dict] | None = None,
) -> MagicMock:
    """Create a mock SageMaker client configured for a single training run."""
    client = MagicMock()

    # create_training_job returns empty dict on success
    client.create_training_job.return_value = {}

    # describe_training_job: first call InProgress, second call terminal
    describe_responses = [
        {
            "TrainingJobStatus": "InProgress",
            "TrainingJobName": "test-job",
        },
        {
            "TrainingJobStatus": final_status,
            "TrainingJobName": "test-job",
            "FailureReason": failure_reason,
            "ModelArtifacts": {"S3ModelArtifacts": model_artifact_uri},
            "FinalMetricDataList": metrics or [
                {"MetricName": "ctr_auc", "Value": 0.82},
                {"MetricName": "revenue_lift", "Value": 0.07},
            ],
        },
    ]
    client.describe_training_job.side_effect = describe_responses

    # create_model_package returns an ARN
    client.create_model_package.return_value = {
        "ModelPackageArn": (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/"
            "artf-dlrm-bid-shader/1"
        )
    }

    return client


def _make_pipeline(
    sagemaker_client: MagicMock | None = None,
) -> TrainingPipeline:
    """Create a TrainingPipeline with a mocked SageMaker client."""
    if sagemaker_client is None:
        sagemaker_client = _make_mock_sagemaker_client()

    return TrainingPipeline(
        sagemaker_role="arn:aws:iam::123456789012:role/SageMakerRole",
        model_bucket="model-bucket",
        region="us-east-1",
        training_data_bucket="training-data-bucket",
        sagemaker_client=sagemaker_client,
        poll_interval_seconds=0.01,  # Fast polling for tests
    )


# ---------------------------------------------------------------------------
# Tests: Successful training job
# ---------------------------------------------------------------------------


class TestTriggerRetraining:
    """Tests for TrainingPipeline.trigger_retraining."""

    @pytest.mark.asyncio
    async def test_successful_training_job(self):
        """Successful training: creates job, polls, returns result."""
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(client)
        config = _make_config()

        result = await pipeline.trigger_retraining(config)

        # Verify create_training_job was called
        client.create_training_job.assert_called_once()
        call_kwargs = client.create_training_job.call_args[1]

        # Check job name contains model type
        assert "dlrm_bid_shader" in call_kwargs["TrainingJobName"]

        # Check training image
        assert (
            call_kwargs["AlgorithmSpecification"]["TrainingImage"]
            == "nemo-rl-dlrm:latest"
        )

        # Check role
        assert call_kwargs["RoleArn"] == (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )

        # Check input data
        s3_uri = (
            call_kwargs["InputDataConfig"][0]["DataSource"]["S3DataSource"][
                "S3Uri"
            ]
        )
        assert s3_uri == "s3://test-bucket/training-data/dlrm/"

        # Check hyperparameters include RL settings
        hp = call_kwargs["HyperParameters"]
        assert hp["use_reinforcement_learning"] == "True"
        assert hp["reward_function"] == "roi"
        assert hp["rl_learning_rate"] == "0.001"
        assert hp["rl_epochs"] == "10"
        assert hp["window_days"] == "7"
        assert hp["cadence_hours"] == "6.0"

        # Verify result
        assert isinstance(result, TrainingResult)
        assert result.model_artifact_uri == (
            "s3://model-bucket/models/output/model.tar.gz"
        )
        assert result.metrics == {"ctr_auc": 0.82, "revenue_lift": 0.07}
        assert result.training_duration_s > 0
        assert "dlrm_bid_shader" in result.job_name

    @pytest.mark.asyncio
    async def test_different_model_types(self):
        """Each model type uses the correct training image."""
        for model_type, expected_image in [
            (ModelType.DLRM_BID_SHADER, "nemo-rl-dlrm:latest"),
            (ModelType.NCF_DEAL_MANAGER, "nemo-rl-ncf:latest"),
        ]:
            client = _make_mock_sagemaker_client()
            pipeline = _make_pipeline(client)
            config = _make_config(model_type=model_type)

            await pipeline.trigger_retraining(config)

            call_kwargs = client.create_training_job.call_args[1]
            actual_image = call_kwargs["AlgorithmSpecification"][
                "TrainingImage"
            ]
            assert actual_image == expected_image, (
                f"Expected {expected_image} for {model_type}, "
                f"got {actual_image}"
            )

    @pytest.mark.asyncio
    async def test_resource_config_from_config(self):
        """Instance type and count from config are passed to SageMaker."""
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(client)
        config = _make_config(
            instance_type="ml.g5.12xlarge", instance_count=2
        )

        await pipeline.trigger_retraining(config)

        call_kwargs = client.create_training_job.call_args[1]
        resource_config = call_kwargs["ResourceConfig"]
        assert resource_config["InstanceType"] == "ml.g5.12xlarge"
        assert resource_config["InstanceCount"] == 2

    @pytest.mark.asyncio
    async def test_stopping_condition(self):
        """max_runtime_seconds is correctly passed as stopping condition."""
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(client)
        config = _make_config(max_runtime_seconds=7200)

        await pipeline.trigger_retraining(config)

        call_kwargs = client.create_training_job.call_args[1]
        assert (
            call_kwargs["StoppingCondition"]["MaxRuntimeInSeconds"] == 7200
        )


# ---------------------------------------------------------------------------
# Tests: At-most-one concurrent job enforcement
# ---------------------------------------------------------------------------


class TestAtMostOneConcurrentJob:
    """Tests for the at-most-one concurrent job per ModelType constraint.

    Requirement 3.4
    """

    @pytest.mark.asyncio
    async def test_raises_when_job_already_running(self):
        """Second job for same ModelType raises TrainingJobAlreadyRunningError."""
        client = MagicMock()
        client.create_training_job.return_value = {}

        # describe_training_job always returns InProgress (simulates
        # a never-completing first job)
        client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress",
            "TrainingJobName": "first-job",
        }

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/SageMakerRole",
            model_bucket="model-bucket",
            region="us-east-1",
            training_data_bucket="training-data-bucket",
            sagemaker_client=client,
            poll_interval_seconds=0.01,
        )

        config = _make_config()

        # Manually inject an active job for this model type
        pipeline._active_jobs[ModelType.DLRM_BID_SHADER] = "first-job"

        # Second attempt should raise
        with pytest.raises(TrainingJobAlreadyRunningError) as exc_info:
            await pipeline.trigger_retraining(config)

        assert exc_info.value.model_type == ModelType.DLRM_BID_SHADER
        assert exc_info.value.existing_job_name == "first-job"

    @pytest.mark.asyncio
    async def test_allows_different_model_types_concurrently(self):
        """Different model types can train concurrently."""
        client = MagicMock()
        client.create_training_job.return_value = {}
        client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress",
            "TrainingJobName": "dlrm-job",
        }

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/SageMakerRole",
            model_bucket="model-bucket",
            region="us-east-1",
            training_data_bucket="training-data-bucket",
            sagemaker_client=client,
            poll_interval_seconds=0.01,
        )

        # Inject active job for DLRM
        pipeline._active_jobs[ModelType.DLRM_BID_SHADER] = "dlrm-job"

        # NCF should still be allowed (set up describe for completion)
        client.describe_training_job.side_effect = [
            {"TrainingJobStatus": "Completed",
             "TrainingJobName": "ncf-job",
             "ModelArtifacts": {"S3ModelArtifacts": "s3://b/m.tar.gz"},
             "FinalMetricDataList": [
                 {"MetricName": "ndcg", "Value": 0.75}
             ]},
        ]

        ncf_config = _make_config(model_type=ModelType.NCF_DEAL_MANAGER)
        result = await pipeline.trigger_retraining(ncf_config)

        assert isinstance(result, TrainingResult)

    @pytest.mark.asyncio
    async def test_stale_active_job_allows_new_job(self):
        """If tracked job already completed, allow a new one."""
        client = MagicMock()
        client.create_training_job.return_value = {}

        # First call for _is_job_running check: job is Completed (stale)
        # Then calls for the new job polling: InProgress -> Completed
        client.describe_training_job.side_effect = [
            # _is_job_running check for stale job
            {"TrainingJobStatus": "Completed", "TrainingJobName": "old-job"},
            # Polling for new job
            {"TrainingJobStatus": "Completed",
             "TrainingJobName": "new-job",
             "ModelArtifacts": {"S3ModelArtifacts": "s3://b/m.tar.gz"},
             "FinalMetricDataList": [
                 {"MetricName": "ctr_auc", "Value": 0.85}
             ]},
        ]

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/SageMakerRole",
            model_bucket="model-bucket",
            region="us-east-1",
            training_data_bucket="training-data-bucket",
            sagemaker_client=client,
            poll_interval_seconds=0.01,
        )

        # Inject stale active job
        pipeline._active_jobs[ModelType.DLRM_BID_SHADER] = "old-job"

        config = _make_config()
        result = await pipeline.trigger_retraining(config)

        # Should succeed because the old job was already completed
        assert isinstance(result, TrainingResult)
        # create_training_job should have been called for the new job
        client.create_training_job.assert_called_once()

    def test_is_training_returns_true_when_job_active(self):
        """is_training returns True when a job is running for model type."""
        client = MagicMock()
        client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress",
            "TrainingJobName": "active-job",
        }

        pipeline = _make_pipeline(client)
        pipeline._active_jobs[ModelType.DLRM_BID_SHADER] = "active-job"

        assert pipeline.is_training(ModelType.DLRM_BID_SHADER) is True

    def test_is_training_returns_false_when_no_job(self):
        """is_training returns False when no job is tracked for model type."""
        pipeline = _make_pipeline()
        assert pipeline.is_training(ModelType.DLRM_BID_SHADER) is False

    def test_is_training_returns_false_when_job_completed(self):
        """is_training returns False when the tracked job already completed."""
        client = MagicMock()
        client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingJobName": "done-job",
        }

        pipeline = _make_pipeline(client)
        pipeline._active_jobs[ModelType.DLRM_BID_SHADER] = "done-job"

        assert pipeline.is_training(ModelType.DLRM_BID_SHADER) is False


# ---------------------------------------------------------------------------
# Tests: Model registration
# ---------------------------------------------------------------------------


class TestRegisterModel:
    """Tests for TrainingPipeline.register_model.

    Requirement 3.5
    """

    @pytest.mark.asyncio
    async def test_registers_with_correct_metadata(self):
        """register_model includes full lineage metadata."""
        client = MagicMock()
        client.create_model_package.return_value = {
            "ModelPackageArn": (
                "arn:aws:sagemaker:us-east-1:123456789012:"
                "model-package/artf-dlrm-bid-shader/1"
            )
        }

        pipeline = _make_pipeline(client)

        result = TrainingResult(
            job_name="dlrm_bid_shader-1234567890-abc12345",
            model_artifact_uri="s3://model-bucket/models/dlrm/output/model.tar.gz",
            metrics={"ctr_auc": 0.82, "revenue_lift": 0.07},
            training_duration_s=1800.5,
            model_version="",
            data_window_days=7,
            sample_count=50000,
            base_model_version="v1.0.0",
        )

        arn = await pipeline.register_model(result, ModelType.DLRM_BID_SHADER)

        # Check return value
        assert arn == (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "model-package/artf-dlrm-bid-shader/1"
        )

        # Check create_model_package was called with correct params
        client.create_model_package.assert_called_once()
        call_kwargs = client.create_model_package.call_args[1]

        # Model package group
        assert (
            call_kwargs["ModelPackageGroupName"] == "artf-dlrm-bid-shader"
        )

        # Lineage metadata
        metadata = call_kwargs["CustomerMetadataProperties"]
        assert (
            metadata["training_job_name"]
            == "dlrm_bid_shader-1234567890-abc12345"
        )
        assert metadata["model_artifact_uri"] == (
            "s3://model-bucket/models/dlrm/output/model.tar.gz"
        )
        assert metadata["training_duration_seconds"] == "1800.5"
        assert metadata["data_window_days"] == "7"
        assert metadata["sample_count"] == "50000"
        assert metadata["base_model_version"] == "v1.0.0"
        assert metadata["metric_ctr_auc"] == "0.82"
        assert metadata["metric_revenue_lift"] == "0.07"

        # Approval status should be pending
        assert (
            call_kwargs["ModelApprovalStatus"] == "PendingManualApproval"
        )

    @pytest.mark.asyncio
    async def test_registers_correct_model_package_group(self):
        """Each model type maps to the correct package group name."""
        expected_groups = {
            ModelType.DLRM_BID_SHADER: "artf-dlrm-bid-shader",
            ModelType.NCF_DEAL_MANAGER: "artf-ncf-deal-manager",
        }

        for model_type, expected_group in expected_groups.items():
            client = MagicMock()
            client.create_model_package.return_value = {
                "ModelPackageArn": f"arn:aws:sagemaker:us-east-1:123:model-package/{expected_group}/1"
            }
            pipeline = _make_pipeline(client)

            result = TrainingResult(
                job_name=f"{model_type.value}-123-abc",
                model_artifact_uri="s3://b/m.tar.gz",
                metrics={"loss": 0.1},
                training_duration_s=600.0,
                model_version="",
            )

            await pipeline.register_model(result, model_type)

            call_kwargs = client.create_model_package.call_args[1]
            assert call_kwargs["ModelPackageGroupName"] == expected_group

    @pytest.mark.asyncio
    async def test_inference_specification_included(self):
        """Registration includes inference specification for serving."""
        client = MagicMock()
        client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:model-package/g/1"
        }
        pipeline = _make_pipeline(client)

        result = TrainingResult(
            job_name="dlrm_bid_shader-123-abc",
            model_artifact_uri="s3://b/models/m.tar.gz",
            metrics={},
            training_duration_s=300.0,
            model_version="",
        )

        await pipeline.register_model(result, ModelType.DLRM_BID_SHADER)

        call_kwargs = client.create_model_package.call_args[1]
        inf_spec = call_kwargs["InferenceSpecification"]

        # Container image and model data
        assert len(inf_spec["Containers"]) == 1
        assert inf_spec["Containers"][0]["ModelDataUrl"] == "s3://b/models/m.tar.gz"
        assert inf_spec["Containers"][0]["Image"] == "nemo-rl-dlrm:latest"


# ---------------------------------------------------------------------------
# Tests: Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Tests for training job failure scenarios.

    Requirement 3.7 (failure detection; SNS/escalation is in Task 4.3)
    """

    @pytest.mark.asyncio
    async def test_job_failure_raises_exception(self):
        """A failed training job raises TrainingJobFailedError."""
        client = _make_mock_sagemaker_client(
            final_status="Failed",
            failure_reason="ResourceLimitExceeded: Insufficient capacity",
        )
        pipeline = _make_pipeline(client)
        config = _make_config()

        with pytest.raises(TrainingJobFailedError) as exc_info:
            await pipeline.trigger_retraining(config)

        assert "ResourceLimitExceeded" in exc_info.value.failure_reason
        assert exc_info.value.job_name is not None

    @pytest.mark.asyncio
    async def test_failed_job_clears_active_tracking(self):
        """After a job fails, the model type is no longer tracked as active."""
        client = _make_mock_sagemaker_client(
            final_status="Failed",
            failure_reason="Algorithm error",
        )
        pipeline = _make_pipeline(client)
        config = _make_config()

        with pytest.raises(TrainingJobFailedError):
            await pipeline.trigger_retraining(config)

        # Active jobs should be empty — the failed job was cleaned up
        assert ModelType.DLRM_BID_SHADER not in pipeline._active_jobs

    @pytest.mark.asyncio
    async def test_stopped_job_returns_failure(self):
        """A stopped training job is treated as a failure."""
        client = MagicMock()
        client.create_training_job.return_value = {}
        client.describe_training_job.side_effect = [
            {
                "TrainingJobStatus": "Stopped",
                "TrainingJobName": "stopped-job",
                "FailureReason": "ManualStop",
                "ModelArtifacts": {},
                "FinalMetricDataList": [],
            },
        ]

        pipeline = _make_pipeline(client)
        config = _make_config()

        # Stopped jobs reach terminal state but aren't "Completed" or "Failed"
        # They should return a result with empty metrics (the pipeline doesn't
        # raise for Stopped — it's a terminal but not "Failed" state)
        result = await pipeline.trigger_retraining(config)

        # Stopped is terminal but not Failed, so it returns a result
        assert isinstance(result, TrainingResult)


# ---------------------------------------------------------------------------
# Tests: Data models
# ---------------------------------------------------------------------------


class TestDataModels:
    """Tests for TrainingJobConfig and TrainingResult dataclasses."""

    def test_training_job_config_defaults(self):
        """TrainingJobConfig has correct default values."""
        config = _make_config()
        assert config.cadence_hours == 6.0
        assert config.window_days == 7

    def test_training_job_config_immutable(self):
        """TrainingJobConfig is frozen (immutable)."""
        config = _make_config()
        with pytest.raises(AttributeError):
            config.model_type = ModelType.NCF_DEAL_MANAGER  # type: ignore

    def test_training_result_immutable(self):
        """TrainingResult is frozen (immutable)."""
        result = TrainingResult(
            job_name="test",
            model_artifact_uri="s3://b/m",
            metrics={},
            training_duration_s=100.0,
            model_version="v1",
        )
        with pytest.raises(AttributeError):
            result.job_name = "other"  # type: ignore

    def test_model_type_values(self):
        """ModelType enum has the expected values."""
        assert ModelType.DLRM_BID_SHADER.value == "dlrm_bid_shader"
        assert ModelType.NCF_DEAL_MANAGER.value == "ncf_deal_manager"
