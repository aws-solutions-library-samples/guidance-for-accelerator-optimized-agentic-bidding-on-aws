"""Unit tests for orchestrator.training_trigger — TrainingTriggerService.

Validates:
- estimate_cost() (pure function) computes real numbers from the actual
  configured instance type/hourly rate and MaxRuntimeInSeconds ceiling —
  never a fabricated number (NFR-2).
- is_training_in_progress() uses a real ListTrainingJobs call, the
  concurrency-enforcement mechanism per the resolved Application Design
  decision.
- trigger_training() raises the correct error for each precondition
  failure (untrainable model type, unconfirmed, already in progress, no
  approved base version) BEFORE ever calling CreateTrainingJob.
- A confirmed, unblocked trigger calls CreateTrainingJob with the real
  resolved base_model_version and returns a TrainingTriggerResult.

Maps to: FR-4, FR-5 (Story 3, train-from-load-test unit).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.training_trigger import (
    TRAINABLE_MODEL_TYPES,
    ModelTypeNotTrainableError,
    NoApprovedBaseVersionError,
    TrainingAlreadyInProgressError,
    TrainingCostEstimate,
    TrainingNotConfirmedError,
    TrainingTriggerResult,
    estimate_cost,
    is_training_in_progress,
    trigger_training,
)


class TestEstimateCost:
    def test_returns_real_instance_type_and_rate(self):
        estimate = estimate_cost("dlrm_bid_shader")
        assert estimate.instance_type == "ml.g5.2xlarge"
        assert estimate.hourly_rate_usd == 1.515

    def test_max_runtime_matches_automated_pipeline(self):
        """MaxRuntimeInSeconds must match governance_eventbridge_cfn.yaml's
        RetrainingTriggerFunction StoppingCondition for the same instance
        type — same ceiling, not a separately invented number."""
        estimate = estimate_cost("dlrm_bid_shader")
        assert estimate.max_runtime_seconds == 14400

    def test_cost_computed_from_rate_and_runtime(self):
        """The cost estimate is a real computation, not a hardcoded value."""
        estimate = estimate_cost("dlrm_bid_shader")
        expected = round((14400 / 3600.0) * 1.515, 2)
        assert estimate.estimated_max_cost_usd == expected

    @given(st.sampled_from(sorted(TRAINABLE_MODEL_TYPES)))
    def test_same_estimate_for_all_trainable_model_types(self, model_type):
        """Property: both trainable model types use the same instance
        type/rate/runtime today (both fine-tune on the same infra)."""
        estimate = estimate_cost(model_type)
        assert estimate.instance_type == "ml.g5.2xlarge"
        assert estimate.max_runtime_seconds == 14400

    def test_returns_frozen_dataclass(self):
        estimate = estimate_cost("dlrm_bid_shader")
        assert isinstance(estimate, TrainingCostEstimate)
        with pytest.raises(Exception):
            estimate.hourly_rate_usd = 99.0


class TestIsTrainingInProgress:
    def test_in_progress_when_job_found(self):
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "dlrm_bid_shader-123-abc"}]
        }
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            in_progress, job_name = is_training_in_progress("dlrm_bid_shader")
        assert in_progress is True
        assert job_name == "dlrm_bid_shader-123-abc"

    def test_not_in_progress_when_no_jobs(self):
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            in_progress, job_name = is_training_in_progress("dlrm_bid_shader")
        assert in_progress is False
        assert job_name is None

    def test_filters_by_in_progress_status(self):
        """The real enforcement mechanism must filter on InProgress status,
        not just name — a completed job with a matching name prefix must
        not block a new trigger."""
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            is_training_in_progress("dlrm_bid_shader")
        call_kwargs = mock_client.list_training_jobs.call_args[1]
        assert call_kwargs["StatusEquals"] == "InProgress"


class TestTriggerTraining:
    _COMMON_KWARGS = dict(
        sagemaker_role_arn="arn:aws:iam::123456789012:role/sagemaker-training",
        model_bucket="artf-model-bucket",
        training_data_bucket="artf-training-data-bucket",
        training_image_registry="123456789012.dkr.ecr.us-east-1.amazonaws.com",
    )

    def test_rejects_untrainable_model_type(self):
        with pytest.raises(ModelTypeNotTrainableError):
            trigger_training(
                "widedeep_segment_activator", confirmed=True, **self._COMMON_KWARGS
            )

    def test_rejects_unconfirmed_request(self):
        with pytest.raises(TrainingNotConfirmedError):
            trigger_training("dlrm_bid_shader", confirmed=False, **self._COMMON_KWARGS)

    def test_rejects_when_already_in_progress(self):
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {
            "TrainingJobSummaries": [{"TrainingJobName": "dlrm_bid_shader-999-xyz"}]
        }
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            with pytest.raises(TrainingAlreadyInProgressError) as exc_info:
                trigger_training("dlrm_bid_shader", confirmed=True, **self._COMMON_KWARGS)
        assert exc_info.value.job_name == "dlrm_bid_shader-999-xyz"
        # CreateTrainingJob must never be called when blocked
        mock_client.create_training_job.assert_not_called()

    def test_rejects_when_no_approved_base_version(self):
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        mock_client.list_model_packages.return_value = {"ModelPackageSummaryList": []}
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            with pytest.raises(NoApprovedBaseVersionError):
                trigger_training("dlrm_bid_shader", confirmed=True, **self._COMMON_KWARGS)
        mock_client.create_training_job.assert_not_called()

    def test_successful_trigger_calls_create_training_job(self):
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        mock_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:model-package/artf-dlrm-bid-shader/3"}
            ]
        }
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            result = trigger_training("dlrm_bid_shader", confirmed=True, **self._COMMON_KWARGS)

        assert isinstance(result, TrainingTriggerResult)
        assert result.model_type == "dlrm_bid_shader"
        assert result.base_model_version == (
            "arn:aws:sagemaker:us-east-1:123:model-package/artf-dlrm-bid-shader/3"
        )
        assert result.instance_type == "ml.g5.2xlarge"
        mock_client.create_training_job.assert_called_once()
        call_kwargs = mock_client.create_training_job.call_args[1]
        assert call_kwargs["ResourceConfig"]["InstanceType"] == "ml.g5.2xlarge"
        assert call_kwargs["StoppingCondition"]["MaxRuntimeInSeconds"] == 14400
        assert call_kwargs["HyperParameters"]["base_model_version"] == result.base_model_version
        assert call_kwargs["HyperParameters"]["triggered_by"] == "governance_ui_on_demand"
