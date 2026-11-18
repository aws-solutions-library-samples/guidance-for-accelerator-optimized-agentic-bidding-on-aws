"""Training Failure Handler for the closed-loop learning system.

Tracks consecutive training failures per ModelType and manages alerting
and automatic pause/escalation logic.

Key behaviors:
- Counts consecutive failures per ModelType (in-memory)
- Emits SNS alerts on each failure
- Pauses automated retraining after N consecutive failures (default 3)
- Escalates via SNS with a distinct subject when pausing
- Resets counters on success

Requirements: 3.6, 3.7, 3.8
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3

from training.pipeline import ModelType

logger = logging.getLogger(__name__)


class TrainingPausedError(Exception):
    """Raised when retraining is attempted on a paused ModelType.

    Requirement 3.8: After 3 consecutive failures, automated retraining
    is paused for the affected Model_Type.
    """

    def __init__(self, model_type: ModelType, consecutive_failures: int):
        self.model_type = model_type
        self.consecutive_failures = consecutive_failures
        super().__init__(
            f"Automated retraining is paused for '{model_type.value}' "
            f"after {consecutive_failures} consecutive failures. "
            f"Manual review required."
        )


class TrainingFailureHandler:
    """Handles training job failures: counting, alerting, and pausing.

    Tracks consecutive failures per ModelType in memory and emits SNS
    alerts on each failure. After max_consecutive_failures consecutive
    failures for a ModelType, the handler pauses automated retraining
    and sends an escalation alert.

    Requirements: 3.7, 3.8
    """

    def __init__(
        self,
        sns_topic_arn: str,
        region: str,
        max_consecutive_failures: int = 3,
        sns_client: Any | None = None,
        events_client: Any | None = None,
    ):
        """Initialize the TrainingFailureHandler.

        Args:
            sns_topic_arn: ARN of the SNS topic for failure alerts.
            region: AWS region.
            max_consecutive_failures: Number of consecutive failures
                before pausing retraining (default 3).
            sns_client: Optional pre-configured boto3 SNS client
                (useful for testing). If None, creates a new client.
            events_client: Optional pre-configured boto3 EventBridge client
                (useful for testing). If None, creates a new client.
        """
        self._sns_topic_arn = sns_topic_arn
        self._region = region
        self._max_consecutive_failures = max_consecutive_failures

        if sns_client is not None:
            self._sns_client = sns_client
        else:
            self._sns_client = boto3.client("sns", region_name=region)

        if events_client is not None:
            self._events_client = events_client
        else:
            self._events_client = boto3.client("events", region_name=region)

        # In-memory state: consecutive failure count per ModelType
        self._failure_counts: dict[ModelType, int] = {}
        # Paused model types
        self._paused: set[ModelType] = set()

    async def handle_failure(
        self, model_type: ModelType, job_name: str, failure_reason: str
    ) -> None:
        """Handle a training job failure.

        Increments the consecutive failure counter, sends an SNS alert,
        and pauses + escalates if the threshold is reached.

        Args:
            model_type: The model type that failed.
            job_name: SageMaker training job name that failed.
            failure_reason: The failure reason from SageMaker.

        Requirements: 3.7, 3.8
        """
        # Increment failure counter
        current_count = self._failure_counts.get(model_type, 0) + 1
        self._failure_counts[model_type] = current_count

        logger.warning(
            "Training job '%s' failed for model type '%s' "
            "(consecutive failures: %d). Reason: %s",
            job_name,
            model_type.value,
            current_count,
            failure_reason,
        )

        # Emit SNS alert for the failure
        alert_message = json.dumps(
            {
                "event": "training_job_failed",
                "model_type": model_type.value,
                "job_name": job_name,
                "failure_reason": failure_reason,
                "consecutive_failures": current_count,
                "max_before_pause": self._max_consecutive_failures,
            }
        )

        self._sns_client.publish(
            TopicArn=self._sns_topic_arn,
            Subject=f"Training Failed: {model_type.value} ({current_count}/{self._max_consecutive_failures})",
            Message=alert_message,
        )

        # Check if we should pause
        if current_count >= self._max_consecutive_failures:
            self._paused.add(model_type)

            logger.error(
                "Pausing automated retraining for '%s' after %d "
                "consecutive failures. Escalating for manual review.",
                model_type.value,
                current_count,
            )

            # Emit escalation alert via SNS (different subject)
            escalation_message = json.dumps(
                {
                    "event": "retraining_paused",
                    "model_type": model_type.value,
                    "consecutive_failures": current_count,
                    "action_required": "Manual review and intervention required",
                    "last_job_name": job_name,
                    "last_failure_reason": failure_reason,
                }
            )

            self._sns_client.publish(
                TopicArn=self._sns_topic_arn,
                Subject=f"ESCALATION: Retraining paused for {model_type.value}",
                Message=escalation_message,
            )

    async def handle_success(self, model_type: ModelType) -> None:
        """Handle a training job success.

        Resets the consecutive failure counter and clears paused status.

        Args:
            model_type: The model type that succeeded.
        """
        self._failure_counts[model_type] = 0
        self._paused.discard(model_type)

        logger.info(
            "Training succeeded for '%s'. Failure counter reset.",
            model_type.value,
        )

    def is_paused(self, model_type: ModelType) -> bool:
        """Check if automated retraining is paused for a model type.

        Args:
            model_type: The model type to check.

        Returns:
            True if the model type has been paused due to consecutive
            failures reaching the threshold.
        """
        return model_type in self._paused

    def get_failure_count(self, model_type: ModelType) -> int:
        """Get the current consecutive failure count for a model type.

        Args:
            model_type: The model type to check.

        Returns:
            The number of consecutive failures (0 if none).
        """
        return self._failure_counts.get(model_type, 0)

    async def emit_registration_event(
        self, model_type: ModelType, model_package_arn: str
    ) -> None:
        """Emit a ModelPackageGroupChanged event to EventBridge.

        Called after successful model registration to notify downstream
        consumers (e.g., the Governance Agent).

        Args:
            model_type: The model type that was registered.
            model_package_arn: The ARN of the newly registered model package.

        Requirements: 3.6
        """
        from training.pipeline import _MODEL_PACKAGE_GROUP_MAP

        model_package_group = _MODEL_PACKAGE_GROUP_MAP[model_type]

        event_detail = json.dumps(
            {
                "ModelPackageGroupName": model_package_group,
                "ModelPackageArn": model_package_arn,
                "ModelType": model_type.value,
            }
        )

        self._events_client.put_events(
            Entries=[
                {
                    "Source": "artf.training-pipeline",
                    "DetailType": "ModelPackageGroupChanged",
                    "Detail": event_detail,
                    "Resources": [model_package_arn],
                }
            ]
        )

        logger.info(
            "Emitted ModelPackageGroupChanged event for model type '%s' "
            "(ARN: %s)",
            model_type.value,
            model_package_arn,
        )
