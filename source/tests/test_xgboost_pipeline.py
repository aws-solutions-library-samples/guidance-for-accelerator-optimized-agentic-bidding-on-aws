"""Unit tests for XGBoostTrainingPipeline (Unit 3: deal-yield-training-pipeline).

Mocks only the SageMaker client boundary (boto3), per this project's testing
convention. Mirrors test_training_pipeline.py's structure for the parallel
(not shared) XGBoost training path.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.pipeline import TrainingResult
from training.xgboost_pipeline import (
    MODEL_PACKAGE_GROUP_BY_TARGET,
    MODEL_TYPE_BY_TARGET,
    TARGET_FLOOR,
    TARGET_MARGIN,
    XGBoostTrainingJobAlreadyRunningError,
    XGBoostTrainingJobConfig,
    XGBoostTrainingPipeline,
)


def _make_config(**overrides) -> XGBoostTrainingJobConfig:
    defaults = {
        "training_data_uri": "s3://test-bucket/training-data/deal_yield_manager/",
        "base_model_version": "v1.0.0",
        "validation_split": 0.2,
        "max_depth": 6,
        "num_round": 100,
        "eta": 0.3,
        "objective": "reg:squarederror",
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "max_runtime_seconds": 3600,
    }
    defaults.update(overrides)
    return XGBoostTrainingJobConfig(**defaults)


def _make_mock_sagemaker_client(training_status: str = "Completed") -> MagicMock:
    client = MagicMock()
    client.create_training_job.return_value = {}
    client.describe_training_job.return_value = {
        "TrainingJobStatus": training_status,
        "ModelArtifacts": {"S3ModelArtifacts": "s3://test-bucket/models/deal_yield_manager_floor/job-1/output/model.tar.gz"},
        "FinalMetricDataList": [{"MetricName": "rmse", "Value": 0.05}],
    }
    client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
    client.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/artf-deal-yield-manager-floor/1"
    }
    return client


def _make_pipeline(sagemaker_client=None, target: str = TARGET_FLOOR, **kwargs) -> XGBoostTrainingPipeline:
    defaults = {
        "sagemaker_role": "arn:aws:iam::123456789012:role/sagemaker-training",
        "model_bucket": "test-model-bucket",
        "region": "us-east-1",
        "training_data_bucket": "test-training-data-bucket",
        "poll_interval_seconds": 0.001,
    }
    defaults.update(kwargs)
    return XGBoostTrainingPipeline(target=target, sagemaker_client=sagemaker_client, **defaults)


class TestModelTypeAndPackageGroup:
    def test_floor_target_model_type(self):
        pipeline = _make_pipeline(target=TARGET_FLOOR)
        assert pipeline.model_type == "deal_yield_manager_floor"
        assert pipeline.model_type == MODEL_TYPE_BY_TARGET[TARGET_FLOOR]

    def test_margin_target_model_type(self):
        pipeline = _make_pipeline(target=TARGET_MARGIN)
        assert pipeline.model_type == "deal_yield_manager_margin"
        assert pipeline.model_type == MODEL_TYPE_BY_TARGET[TARGET_MARGIN]

    def test_floor_package_group_matches_infrastructure_design(self):
        pipeline = _make_pipeline(target=TARGET_FLOOR)
        assert pipeline.model_package_group == "artf-deal-yield-manager-floor"
        assert pipeline.model_package_group == MODEL_PACKAGE_GROUP_BY_TARGET[TARGET_FLOOR]

    def test_margin_package_group_matches_infrastructure_design(self):
        pipeline = _make_pipeline(target=TARGET_MARGIN)
        assert pipeline.model_package_group == "artf-deal-yield-manager-margin"
        assert pipeline.model_package_group == MODEL_PACKAGE_GROUP_BY_TARGET[TARGET_MARGIN]

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            _make_pipeline(target="not-a-real-target")


class TestConcurrencyEnforcement:
    def test_is_training_in_progress_false_when_no_jobs(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client)
        in_progress, job_name = pipeline.is_training_in_progress()
        assert in_progress is False
        assert job_name is None

    def test_is_training_in_progress_true_when_job_exists(self):
        client = _make_mock_sagemaker_client()
        client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "deal-yield-manager-floor-123-abcd1234"}]
        }
        pipeline = _make_pipeline(sagemaker_client=client)
        in_progress, job_name = pipeline.is_training_in_progress()
        assert in_progress is True
        assert job_name == "deal-yield-manager-floor-123-abcd1234"

    def test_uses_job_name_prefix_matching_target(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_FLOOR)
        pipeline.is_training_in_progress()
        call_kwargs = client.list_training_jobs.call_args[1]
        assert call_kwargs["NameContains"] == "deal-yield-manager-floor-"
        assert call_kwargs["StatusEquals"] == "InProgress"

    def test_margin_target_uses_its_own_job_name_prefix(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_MARGIN)
        pipeline.is_training_in_progress()
        call_kwargs = client.list_training_jobs.call_args[1]
        assert call_kwargs["NameContains"] == "deal-yield-manager-margin-"

    def test_floor_and_margin_concurrency_is_independent(self):
        """A running floor job must not block a margin trigger_training()
        call -- the two targets are trained completely independently."""
        client = _make_mock_sagemaker_client()
        client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "deal-yield-manager-floor-123-abcd"}]
        }
        margin_pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_MARGIN)
        # The margin pipeline's own ListTrainingJobs call is scoped to its
        # own prefix, so it never even queries into the floor job's namespace.
        in_progress, _ = margin_pipeline.is_training_in_progress()
        call_kwargs = client.list_training_jobs.call_args[1]
        assert call_kwargs["NameContains"] == "deal-yield-manager-margin-"


class TestTriggerTraining:
    def test_raises_when_job_already_in_progress(self):
        client = _make_mock_sagemaker_client()
        client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "deal-yield-manager-floor-existing"}]
        }
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_FLOOR)

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            with pytest.raises(XGBoostTrainingJobAlreadyRunningError) as exc_info:
                asyncio.run(pipeline.trigger_training(_make_config()))
            assert "deal_yield_manager_floor" in str(exc_info.value)
        client.create_training_job.assert_not_called()

    def test_successful_training_job(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client)

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            result = asyncio.run(pipeline.trigger_training(_make_config()))

        assert isinstance(result, TrainingResult)
        assert result.base_model_version == "v1.0.0"
        client.create_training_job.assert_called_once()

    def test_job_name_uses_hyphenated_prefix(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_FLOOR)

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            result = asyncio.run(pipeline.trigger_training(_make_config()))

        assert result.job_name.startswith("deal-yield-manager-floor-")
        assert "_" not in result.job_name

    def test_margin_job_name_uses_its_own_prefix(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_MARGIN)

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            result = asyncio.run(pipeline.trigger_training(_make_config()))

        assert result.job_name.startswith("deal-yield-manager-margin-")

    def test_hyperparameters_are_tree_shaped_not_rl_shaped(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client)

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            asyncio.run(pipeline.trigger_training(_make_config(max_depth=8, eta=0.1)))

        call_kwargs = client.create_training_job.call_args[1]
        hp = call_kwargs["HyperParameters"]
        assert hp["max_depth"] == "8"
        assert hp["eta"] == "0.1"
        assert "reward_function" not in hp
        assert "rl_learning_rate" not in hp

    def test_raises_runtime_error_on_job_failure(self):
        client = _make_mock_sagemaker_client(training_status="Failed")
        client.describe_training_job.return_value["FailureReason"] = "OOM"
        pipeline = _make_pipeline(sagemaker_client=client)

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            with pytest.raises(RuntimeError, match="OOM"):
                asyncio.run(pipeline.trigger_training(_make_config()))


class TestRegisterModel:
    def test_registers_floor_into_its_own_package_group(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_FLOOR)
        result = TrainingResult(
            job_name="deal-yield-manager-floor-123-abcd",
            model_artifact_uri="s3://bucket/model.tar.gz",
            metrics={"rmse": 0.05},
            training_duration_s=120.0,
            model_version="",
            base_model_version="v1.0.0",
        )

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            asyncio.run(pipeline.register_model(result))

        call_kwargs = client.create_model_package.call_args[1]
        assert call_kwargs["ModelPackageGroupName"] == "artf-deal-yield-manager-floor"
        assert call_kwargs["ModelApprovalStatus"] == "PendingManualApproval"
        assert call_kwargs["CustomerMetadataProperties"]["metric_rmse"] == "0.05"

    def test_registers_margin_into_its_own_package_group(self):
        client = _make_mock_sagemaker_client()
        pipeline = _make_pipeline(sagemaker_client=client, target=TARGET_MARGIN)
        result = TrainingResult(
            job_name="deal-yield-manager-margin-123-abcd",
            model_artifact_uri="s3://bucket/model.tar.gz",
            metrics={"rmse": 0.03},
            training_duration_s=90.0,
            model_version="",
            base_model_version="v1.0.0",
        )

        with patch.object(
            XGBoostTrainingPipeline, "_training_image_uri",
            return_value="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
        ):
            asyncio.run(pipeline.register_model(result))

        call_kwargs = client.create_model_package.call_args[1]
        assert call_kwargs["ModelPackageGroupName"] == "artf-deal-yield-manager-margin"
