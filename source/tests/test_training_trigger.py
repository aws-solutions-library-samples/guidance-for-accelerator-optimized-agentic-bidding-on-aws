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
- ncf_deal_manager is parked (TRAINABLE_MODEL_TYPES == {"dlrm_bid_shader"}):
  its ACTIVATE_DEALS/SUPPRESS_DEALS mutations disambiguate deals via
  path + a list of deal IDs (verified against the real ARTF proto/
  reference implementation), but BidShadingOutcomeEvent/Record has no
  deal_id field or per-deal fan-out yet, so there's no way to attribute a
  training outcome to one specific deal today.

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


class TestTrainableModelTypes:
    def test_dlrm_and_yield_submodels_are_trainable_today(self):
        """ncf_deal_manager is parked pending a deal_id schema change (see
        module docstring) -- must not silently become trainable again.
        dlrm_bid_shader and the two Yield Optimizer sub-models
        (deal_yield_manager_floor/margin) are the three real trainable
        model types."""
        assert TRAINABLE_MODEL_TYPES == frozenset({
            "dlrm_bid_shader", "deal_yield_manager_floor", "deal_yield_manager_margin",
        })

    def test_ncf_deal_manager_is_not_trainable(self):
        assert "ncf_deal_manager" not in TRAINABLE_MODEL_TYPES


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
        """Property: every currently-trainable model type uses the same
        instance type/rate/runtime (holds trivially for today's single
        trainable model type, and stays correct if a second one is
        re-enabled later)."""
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

    def test_name_contains_has_no_underscore(self):
        """Regression test: SageMaker's ListTrainingJobs NameContains param
        only accepts [a-zA-Z0-9\\-]+ (verified live — a real deployment hit
        botocore.exceptions.ClientError: ValidationException on
        NameContains='dlrm_bid_shader-' because model_type contains an
        underscore). NameContains must use the hyphenated form."""
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            is_training_in_progress("dlrm_bid_shader")
        call_kwargs = mock_client.list_training_jobs.call_args[1]
        assert call_kwargs["NameContains"] == "dlrm-bid-shader-"
        assert "_" not in call_kwargs["NameContains"]


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

    def test_rejects_ncf_deal_manager_as_parked(self):
        """ncf_deal_manager training is parked (see module docstring) --
        must be rejected the same way as a model type with no training
        infrastructure at all, and must never call CreateTrainingJob."""
        mock_client = MagicMock()
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            with pytest.raises(ModelTypeNotTrainableError) as exc_info:
                trigger_training("ncf_deal_manager", confirmed=True, **self._COMMON_KWARGS)
        assert exc_info.value.model_type == "ncf_deal_manager"
        mock_client.create_training_job.assert_not_called()

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
        # HyperParameters.model_type must stay the real model_type (used for
        # downstream resolution), unlike the job name itself.
        assert call_kwargs["HyperParameters"]["model_type"] == "dlrm_bid_shader"

    def test_training_job_name_has_no_underscore(self):
        """Regression test: CreateTrainingJob's TrainingJobName only accepts
        [a-zA-Z0-9\\-]+ — same live failure mode as NameContains above. The
        job name must be hyphenated even though model_type (with
        underscores) is still used everywhere else (HyperParameters, S3
        paths, Model Package Group lookup)."""
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        mock_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:model-package/artf-dlrm-bid-shader/3"}
            ]
        }
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            result = trigger_training("dlrm_bid_shader", confirmed=True, **self._COMMON_KWARGS)

        call_kwargs = mock_client.create_training_job.call_args[1]
        assert "_" not in call_kwargs["TrainingJobName"]
        assert call_kwargs["TrainingJobName"].startswith("dlrm-bid-shader-")
        assert "_" not in result.job_name


class TestTriggerTrainingXGBoostShape:
    """deal_yield_manager_floor/margin use the xgboost training shape (see
    _TRAINING_SHAPE) -- SageMaker's built-in XGBoost container, tree
    hyperparameters, and a per-target training-data S3 prefix, rather than
    the NeMo-RL shape dlrm_bid_shader/ncf_deal_manager use."""

    _COMMON_KWARGS = dict(
        sagemaker_role_arn="arn:aws:iam::123456789012:role/sagemaker-training",
        model_bucket="artf-model-bucket",
        training_data_bucket="artf-training-data-bucket",
        training_image_registry="123456789012.dkr.ecr.us-east-1.amazonaws.com",
    )

    def _mock_client_ready(self, base_version_arn):
        mock_client = MagicMock()
        mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        mock_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [{"ModelPackageArn": base_version_arn}]
        }
        return mock_client

    def test_rejects_when_xgboost_image_uri_not_configured(self):
        from orchestrator.training_trigger import XGBoostTrainingImageNotConfiguredError

        mock_client = self._mock_client_ready(
            "arn:aws:sagemaker:us-east-1:123:model-package/artf-deal-yield-manager-floor/1"
        )
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            with pytest.raises(XGBoostTrainingImageNotConfiguredError):
                trigger_training(
                    "deal_yield_manager_floor", confirmed=True,
                    xgboost_training_image_uri=None, **self._COMMON_KWARGS,
                )
        mock_client.create_training_job.assert_not_called()

    def test_floor_target_uses_xgboost_image_and_floor_prefix(self):
        mock_client = self._mock_client_ready(
            "arn:aws:sagemaker:us-east-1:123:model-package/artf-deal-yield-manager-floor/1"
        )
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            result = trigger_training(
                "deal_yield_manager_floor", confirmed=True,
                xgboost_training_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
                **self._COMMON_KWARGS,
            )

        assert result.model_type == "deal_yield_manager_floor"
        call_kwargs = mock_client.create_training_job.call_args[1]
        assert call_kwargs["AlgorithmSpecification"]["TrainingImage"] == (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1"
        )
        s3_uri = call_kwargs["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri == "s3://artf-training-data-bucket/training-data-deal-yield-floor/"
        # Tree hyperparameters, not the NeMo-RL shape.
        hp = call_kwargs["HyperParameters"]
        assert hp["max_depth"] == "6"
        assert hp["objective"] == "reg:squarederror"
        assert "window_days" not in hp
        assert "model_type" not in hp

    def test_margin_target_uses_margin_prefix(self):
        mock_client = self._mock_client_ready(
            "arn:aws:sagemaker:us-east-1:123:model-package/artf-deal-yield-manager-margin/1"
        )
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            trigger_training(
                "deal_yield_manager_margin", confirmed=True,
                xgboost_training_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
                **self._COMMON_KWARGS,
            )

        call_kwargs = mock_client.create_training_job.call_args[1]
        s3_uri = call_kwargs["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri == "s3://artf-training-data-bucket/training-data-deal-yield-margin/"

    def test_floor_job_name_has_no_underscore(self):
        mock_client = self._mock_client_ready(
            "arn:aws:sagemaker:us-east-1:123:model-package/artf-deal-yield-manager-floor/1"
        )
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            result = trigger_training(
                "deal_yield_manager_floor", confirmed=True,
                xgboost_training_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1",
                **self._COMMON_KWARGS,
            )
        assert "_" not in result.job_name
        assert result.job_name.startswith("deal-yield-manager-floor-")

    def test_dlrm_still_uses_nemo_rl_shape_unaffected(self):
        """Sanity check that adding the xgboost branch didn't change
        dlrm_bid_shader's existing behavior."""
        mock_client = self._mock_client_ready(
            "arn:aws:sagemaker:us-east-1:123:model-package/artf-dlrm-bid-shader/3"
        )
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=mock_client):
            trigger_training("dlrm_bid_shader", confirmed=True, **self._COMMON_KWARGS)

        call_kwargs = mock_client.create_training_job.call_args[1]
        assert call_kwargs["AlgorithmSpecification"]["TrainingImage"] == (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/artf-nemo-rl-training:dlrm"
        )
        s3_uri = call_kwargs["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri == "s3://artf-training-data-bucket/training-data/"
