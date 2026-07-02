"""Unit tests for the Training Failure Handler.

Tests cover:
- Failure counter increments correctly
- SNS alert sent on each failure
- Paused after max_consecutive_failures (default 3) consecutive failures
- Escalation alert sent when pausing
- Success resets the failure counter
- is_paused returns correct state
- EventBridge event emitted on registration

Requirements: 3.6, 3.7, 3.8
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.failure_handler import (
    TrainingFailureHandler,
    TrainingPausedError,
)
from training.pipeline import (
    ModelType,
    TrainingJobConfig,
    TrainingJobFailedError,
    TrainingPipeline,
    TrainingResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:123456789012:training-alerts"
REGION = "us-east-1"


def _make_handler(
    max_consecutive_failures: int = 3,
    sns_client: MagicMock | None = None,
    events_client: MagicMock | None = None,
) -> TrainingFailureHandler:
    """Create a TrainingFailureHandler with mocked AWS clients."""
    if sns_client is None:
        sns_client = MagicMock()
    if events_client is None:
        events_client = MagicMock()

    return TrainingFailureHandler(
        sns_topic_arn=SNS_TOPIC_ARN,
        region=REGION,
        max_consecutive_failures=max_consecutive_failures,
        sns_client=sns_client,
        events_client=events_client,
    )


# ---------------------------------------------------------------------------
# Tests: Failure counter
# ---------------------------------------------------------------------------


class TestFailureCounter:
    """Tests for consecutive failure counting logic."""

    @pytest.mark.asyncio
    async def test_failure_increments_counter(self):
        """Each failure increments the counter for the model type."""
        handler = _make_handler()

        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 0

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "OutOfMemory"
        )
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 1

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Timeout"
        )
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 2

    @pytest.mark.asyncio
    async def test_failures_tracked_independently_per_model_type(self):
        """Failures for different model types are tracked independently."""
        handler = _make_handler()

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "dlrm-job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "dlrm-job-2", "Error"
        )
        await handler.handle_failure(
            ModelType.NCF_DEAL_MANAGER, "ncf-job-1", "Error"
        )

        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 2
        assert handler.get_failure_count(ModelType.NCF_DEAL_MANAGER) == 1
        assert (
            handler.get_failure_count(ModelType.WIDEDEEP_SEGMENT_ACTIVATOR)
            == 0
        )

    @pytest.mark.asyncio
    async def test_success_resets_counter(self):
        """Success resets the consecutive failure counter to 0."""
        handler = _make_handler()

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Error"
        )
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 2

        await handler.handle_success(ModelType.DLRM_BID_SHADER)
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 0

    @pytest.mark.asyncio
    async def test_success_does_not_affect_other_model_types(self):
        """Success for one model type does not reset others."""
        handler = _make_handler()

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.NCF_DEAL_MANAGER, "job-2", "Error"
        )

        await handler.handle_success(ModelType.DLRM_BID_SHADER)

        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 0
        assert handler.get_failure_count(ModelType.NCF_DEAL_MANAGER) == 1


# ---------------------------------------------------------------------------
# Tests: SNS alerting
# ---------------------------------------------------------------------------


class TestSNSAlerting:
    """Tests for SNS alert emission on failures."""

    @pytest.mark.asyncio
    async def test_sns_alert_sent_on_each_failure(self):
        """An SNS alert is published for every failure."""
        sns_client = MagicMock()
        handler = _make_handler(sns_client=sns_client)

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "ResourceLimitExceeded"
        )

        sns_client.publish.assert_called_once()
        call_kwargs = sns_client.publish.call_args[1]

        assert call_kwargs["TopicArn"] == SNS_TOPIC_ARN
        assert "Training Failed" in call_kwargs["Subject"]
        assert "dlrm_bid_shader" in call_kwargs["Subject"]

        message_body = json.loads(call_kwargs["Message"])
        assert message_body["event"] == "training_job_failed"
        assert message_body["model_type"] == "dlrm_bid_shader"
        assert message_body["job_name"] == "job-1"
        assert message_body["failure_reason"] == "ResourceLimitExceeded"
        assert message_body["consecutive_failures"] == 1

    @pytest.mark.asyncio
    async def test_escalation_alert_sent_when_pausing(self):
        """An escalation alert is sent when the model type is paused."""
        sns_client = MagicMock()
        handler = _make_handler(
            max_consecutive_failures=3, sns_client=sns_client
        )

        # First two failures: regular alerts only
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error1"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Error2"
        )
        assert sns_client.publish.call_count == 2

        # Third failure triggers escalation (2 calls: regular + escalation)
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-3", "Error3"
        )
        assert sns_client.publish.call_count == 4  # 2 prior + 2 for third

        # Check the escalation call (last publish call)
        escalation_call = sns_client.publish.call_args_list[3]
        escalation_kwargs = escalation_call[1]

        assert "ESCALATION" in escalation_kwargs["Subject"]
        assert "Retraining paused" in escalation_kwargs["Subject"]

        escalation_body = json.loads(escalation_kwargs["Message"])
        assert escalation_body["event"] == "retraining_paused"
        assert escalation_body["consecutive_failures"] == 3

    @pytest.mark.asyncio
    async def test_failure_alert_includes_correct_count(self):
        """Each alert correctly reports the consecutive failure count."""
        sns_client = MagicMock()
        handler = _make_handler(sns_client=sns_client)

        await handler.handle_failure(
            ModelType.NCF_DEAL_MANAGER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.NCF_DEAL_MANAGER, "job-2", "Error"
        )

        # Check second alert has count=2
        second_call = sns_client.publish.call_args_list[1]
        message_body = json.loads(second_call[1]["Message"])
        assert message_body["consecutive_failures"] == 2


# ---------------------------------------------------------------------------
# Tests: Pause logic
# ---------------------------------------------------------------------------


class TestPauseLogic:
    """Tests for automatic retraining pause after consecutive failures."""

    @pytest.mark.asyncio
    async def test_not_paused_below_threshold(self):
        """Model type is not paused below the failure threshold."""
        handler = _make_handler(max_consecutive_failures=3)

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Error"
        )

        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is False

    @pytest.mark.asyncio
    async def test_paused_at_threshold(self):
        """Model type is paused when reaching the failure threshold."""
        handler = _make_handler(max_consecutive_failures=3)

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-3", "Error"
        )

        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is True

    @pytest.mark.asyncio
    async def test_success_clears_paused_status(self):
        """A success clears the paused status."""
        handler = _make_handler(max_consecutive_failures=3)

        # Pause the model type
        for i in range(3):
            await handler.handle_failure(
                ModelType.DLRM_BID_SHADER, f"job-{i}", "Error"
            )
        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is True

        # Success clears pause
        await handler.handle_success(ModelType.DLRM_BID_SHADER)
        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is False

    @pytest.mark.asyncio
    async def test_pause_only_affects_specific_model_type(self):
        """Pausing one model type does not affect others."""
        handler = _make_handler(max_consecutive_failures=2)

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Error"
        )

        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is True
        assert handler.is_paused(ModelType.NCF_DEAL_MANAGER) is False
        assert (
            handler.is_paused(ModelType.WIDEDEEP_SEGMENT_ACTIVATOR) is False
        )

    @pytest.mark.asyncio
    async def test_custom_max_consecutive_failures(self):
        """Custom max_consecutive_failures threshold is respected."""
        handler = _make_handler(max_consecutive_failures=5)

        for i in range(4):
            await handler.handle_failure(
                ModelType.DLRM_BID_SHADER, f"job-{i}", "Error"
            )
        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is False

        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-4", "Error"
        )
        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is True

    def test_is_paused_returns_false_for_untouched_model(self):
        """is_paused returns False for a model type with no history."""
        handler = _make_handler()
        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is False


# ---------------------------------------------------------------------------
# Tests: EventBridge registration event
# ---------------------------------------------------------------------------


class TestRegistrationEvent:
    """Tests for EventBridge event emission on model registration."""

    @pytest.mark.asyncio
    async def test_emits_model_package_group_changed_event(self):
        """emit_registration_event sends the correct EventBridge event."""
        events_client = MagicMock()
        handler = _make_handler(events_client=events_client)

        model_package_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "model-package/artf-dlrm-bid-shader/1"
        )

        await handler.emit_registration_event(
            ModelType.DLRM_BID_SHADER, model_package_arn
        )

        events_client.put_events.assert_called_once()
        call_kwargs = events_client.put_events.call_args[1]

        entries = call_kwargs["Entries"]
        assert len(entries) == 1

        entry = entries[0]
        assert entry["Source"] == "artf.training-pipeline"
        assert entry["DetailType"] == "ModelPackageGroupChanged"
        assert model_package_arn in entry["Resources"]

        detail = json.loads(entry["Detail"])
        assert detail["ModelPackageGroupName"] == "artf-dlrm-bid-shader"
        assert detail["ModelPackageArn"] == model_package_arn
        assert detail["ModelType"] == "dlrm_bid_shader"

    @pytest.mark.asyncio
    async def test_event_uses_correct_package_group_per_model_type(self):
        """Each model type produces an event with the correct group name."""
        expected = {
            ModelType.DLRM_BID_SHADER: "artf-dlrm-bid-shader",
            ModelType.NCF_DEAL_MANAGER: "artf-ncf-deal-manager",
            ModelType.WIDEDEEP_SEGMENT_ACTIVATOR: "artf-widedeep-segment-activator",
        }

        for model_type, expected_group in expected.items():
            events_client = MagicMock()
            handler = _make_handler(events_client=events_client)

            arn = f"arn:aws:sagemaker:us-east-1:123:model-package/{expected_group}/1"
            await handler.emit_registration_event(model_type, arn)

            entry = events_client.put_events.call_args[1]["Entries"][0]
            detail = json.loads(entry["Detail"])
            assert detail["ModelPackageGroupName"] == expected_group


# ---------------------------------------------------------------------------
# Tests: TrainingPausedError exception
# ---------------------------------------------------------------------------


class TestTrainingPausedError:
    """Tests for the TrainingPausedError exception."""

    def test_exception_attributes(self):
        """TrainingPausedError stores model_type and failure count."""
        err = TrainingPausedError(ModelType.DLRM_BID_SHADER, 3)
        assert err.model_type == ModelType.DLRM_BID_SHADER
        assert err.consecutive_failures == 3
        assert "paused" in str(err).lower()
        assert "dlrm_bid_shader" in str(err)


# ---------------------------------------------------------------------------
# Tests: Pipeline integration with failure handler
# ---------------------------------------------------------------------------


class TestPipelineFailureHandlerIntegration:
    """Tests for TrainingPipeline integration with TrainingFailureHandler."""

    @pytest.mark.asyncio
    async def test_paused_model_type_raises_training_paused_error(self):
        """trigger_retraining raises TrainingPausedError for paused types."""
        handler = _make_handler(max_consecutive_failures=2)

        # Pause the model type
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-1", "Error"
        )
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "job-2", "Error"
        )
        assert handler.is_paused(ModelType.DLRM_BID_SHADER) is True

        # Create pipeline with failure handler
        sm_client = MagicMock()
        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/R",
            model_bucket="bucket",
            region="us-east-1",
            training_data_bucket="data-bucket",
            sagemaker_client=sm_client,
            poll_interval_seconds=0.01,
            failure_handler=handler,
        )

        config = TrainingJobConfig(
            model_type=ModelType.DLRM_BID_SHADER,
            base_model_version="v1.0",
            training_data_uri="s3://bucket/data/",
            validation_split=0.2,
            use_reinforcement_learning=True,
            reward_function="roi",
            rl_learning_rate=0.001,
            rl_epochs=10,
            instance_type="ml.p4d.24xlarge",
            instance_count=1,
            max_runtime_seconds=3600,
        )

        with pytest.raises(TrainingPausedError) as exc_info:
            await pipeline.trigger_retraining(config)

        assert exc_info.value.model_type == ModelType.DLRM_BID_SHADER
        # SageMaker should never have been called
        sm_client.create_training_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_handler_called_on_job_failure(self):
        """handle_failure is called when a training job fails."""
        sns_client = MagicMock()
        handler = _make_handler(sns_client=sns_client)

        sm_client = MagicMock()
        sm_client.create_training_job.return_value = {}
        sm_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed",
            "TrainingJobName": "fail-job",
            "FailureReason": "AlgorithmError",
            "ModelArtifacts": {},
            "FinalMetricDataList": [],
        }

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/R",
            model_bucket="bucket",
            region="us-east-1",
            training_data_bucket="data-bucket",
            sagemaker_client=sm_client,
            poll_interval_seconds=0.01,
            failure_handler=handler,
        )

        config = TrainingJobConfig(
            model_type=ModelType.DLRM_BID_SHADER,
            base_model_version="v1.0",
            training_data_uri="s3://bucket/data/",
            validation_split=0.2,
            use_reinforcement_learning=True,
            reward_function="roi",
            rl_learning_rate=0.001,
            rl_epochs=10,
            instance_type="ml.p4d.24xlarge",
            instance_count=1,
            max_runtime_seconds=3600,
        )

        with pytest.raises(TrainingJobFailedError):
            await pipeline.trigger_retraining(config)

        # Verify failure was tracked
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 1
        # Verify SNS was called
        sns_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_success_resets_handler_on_completion(self):
        """handle_success is called when a training job completes."""
        handler = _make_handler()

        # Pre-populate a failure count
        await handler.handle_failure(
            ModelType.DLRM_BID_SHADER, "old-job", "OldError"
        )
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 1

        sm_client = MagicMock()
        sm_client.create_training_job.return_value = {}
        sm_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingJobName": "success-job",
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://bucket/model.tar.gz"
            },
            "FinalMetricDataList": [
                {"MetricName": "auc", "Value": 0.9}
            ],
        }

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/R",
            model_bucket="bucket",
            region="us-east-1",
            training_data_bucket="data-bucket",
            sagemaker_client=sm_client,
            poll_interval_seconds=0.01,
            failure_handler=handler,
        )

        config = TrainingJobConfig(
            model_type=ModelType.DLRM_BID_SHADER,
            base_model_version="v1.0",
            training_data_uri="s3://bucket/data/",
            validation_split=0.2,
            use_reinforcement_learning=True,
            reward_function="roi",
            rl_learning_rate=0.001,
            rl_epochs=10,
            instance_type="ml.p4d.24xlarge",
            instance_count=1,
            max_runtime_seconds=3600,
        )

        result = await pipeline.trigger_retraining(config)

        assert isinstance(result, TrainingResult)
        # Failure count should be reset to 0
        assert handler.get_failure_count(ModelType.DLRM_BID_SHADER) == 0

    @pytest.mark.asyncio
    async def test_register_model_emits_eventbridge_event(self):
        """register_model emits ModelPackageGroupChanged via failure handler."""
        events_client = MagicMock()
        handler = _make_handler(events_client=events_client)

        sm_client = MagicMock()
        model_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:"
            "model-package/artf-dlrm-bid-shader/1"
        )
        sm_client.create_model_package.return_value = {
            "ModelPackageArn": model_arn
        }

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/R",
            model_bucket="bucket",
            region="us-east-1",
            training_data_bucket="data-bucket",
            sagemaker_client=sm_client,
            poll_interval_seconds=0.01,
            failure_handler=handler,
        )

        result = TrainingResult(
            job_name="dlrm_bid_shader-123-abc",
            model_artifact_uri="s3://bucket/model.tar.gz",
            metrics={"auc": 0.85},
            training_duration_s=1200.0,
            model_version="",
            data_window_days=7,
            sample_count=10000,
            base_model_version="v1.0",
        )

        arn = await pipeline.register_model(result, ModelType.DLRM_BID_SHADER)

        assert arn == model_arn
        # Verify EventBridge event was emitted
        events_client.put_events.assert_called_once()
        entry = events_client.put_events.call_args[1]["Entries"][0]
        assert entry["DetailType"] == "ModelPackageGroupChanged"
        detail = json.loads(entry["Detail"])
        assert detail["ModelPackageArn"] == model_arn

    @pytest.mark.asyncio
    async def test_pipeline_works_without_failure_handler(self):
        """Pipeline still works when no failure_handler is provided."""
        sm_client = MagicMock()
        sm_client.create_training_job.return_value = {}
        sm_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingJobName": "job",
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://bucket/model.tar.gz"
            },
            "FinalMetricDataList": [],
        }

        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123456789012:role/R",
            model_bucket="bucket",
            region="us-east-1",
            training_data_bucket="data-bucket",
            sagemaker_client=sm_client,
            poll_interval_seconds=0.01,
            # No failure_handler
        )

        config = TrainingJobConfig(
            model_type=ModelType.DLRM_BID_SHADER,
            base_model_version="v1.0",
            training_data_uri="s3://bucket/data/",
            validation_split=0.2,
            use_reinforcement_learning=True,
            reward_function="roi",
            rl_learning_rate=0.001,
            rl_epochs=10,
            instance_type="ml.p4d.24xlarge",
            instance_count=1,
            max_runtime_seconds=3600,
        )

        result = await pipeline.trigger_retraining(config)
        assert isinstance(result, TrainingResult)
