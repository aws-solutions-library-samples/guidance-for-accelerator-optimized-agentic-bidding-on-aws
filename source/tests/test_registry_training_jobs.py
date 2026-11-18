"""Training jobs shown next to registry versions in the Governance tab.

A running SageMaker training job has no model package version yet, and a failed
one never gets one, so the registry table was silent for the whole training
duration and after a failure. ``training_trigger.list_training_jobs_for_display``
returns the active jobs plus the most recent failure from the last 24 hours, and
``closed_loop_api.models_handler`` attaches them to the versions response as
``training_jobs`` without letting a failure of that lookup hide the versions.

ListTrainingJobs is called WITHOUT ``StatusEquals``: SageMaker applies
``MaxResults`` before the status filter, so filtering client side is the only
way an active job cannot be hidden behind newer terminal jobs.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import closed_loop_api  # noqa: E402
from orchestrator.training_trigger import list_training_jobs_for_display  # noqa: E402

NOW = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)


def _summary(name, status, *, created_ago_h=1.0, ended_ago_h=None, secondary=None):
    s = {
        "TrainingJobName": name,
        "TrainingJobArn": f"arn:aws:sagemaker:us-east-1:123456789012:training-job/{name}",
        "TrainingJobStatus": status,
        "CreationTime": NOW - timedelta(hours=created_ago_h),
        "LastModifiedTime": NOW - timedelta(hours=ended_ago_h if ended_ago_h is not None else created_ago_h),
    }
    if ended_ago_h is not None:
        s["TrainingEndTime"] = NOW - timedelta(hours=ended_ago_h)
    if secondary:
        s["SecondaryStatus"] = secondary
    return s


def _client_with(summaries, failure_reason="AlgorithmError: boom"):
    client = MagicMock()
    client.list_training_jobs.return_value = {"TrainingJobSummaries": summaries}
    client.describe_training_job.return_value = {"FailureReason": failure_reason}
    return client


class TestListTrainingJobsForDisplay:
    def test_in_progress_job_is_returned_with_fields(self):
        client = _client_with([
            _summary("dlrm-bid-shader-1-abc", "InProgress", secondary="Training"),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            jobs = list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp())
        assert len(jobs) == 1
        job = jobs[0]
        assert job["job_name"] == "dlrm-bid-shader-1-abc"
        assert job["status"] == "InProgress"
        assert job["secondary_status"] == "Training"
        assert job["model_type"] == "dlrm_bid_shader"
        assert job["created_at"] == (NOW - timedelta(hours=1)).isoformat()
        assert job["ended_at"] is None
        assert "failure_reason" not in job

    def test_query_has_no_status_filter_and_uses_hyphenated_prefix(self):
        client = _client_with([])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            list_training_jobs_for_display("deal_yield_manager_floor", now=NOW.timestamp())
        kwargs = client.list_training_jobs.call_args[1]
        assert "StatusEquals" not in kwargs
        assert kwargs["NameContains"] == "deal-yield-manager-floor-"
        assert kwargs["SortBy"] == "CreationTime"
        assert kwargs["SortOrder"] == "Descending"

    def test_active_job_not_hidden_behind_newer_completed_jobs(self):
        client = _client_with([
            _summary("dlrm-bid-shader-3", "Completed", created_ago_h=0.5, ended_ago_h=0.2),
            _summary("dlrm-bid-shader-2", "Completed", created_ago_h=0.8, ended_ago_h=0.6),
            _summary("dlrm-bid-shader-1", "InProgress", created_ago_h=2.0),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            jobs = list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp())
        assert [j["job_name"] for j in jobs] == ["dlrm-bid-shader-1"]

    def test_completed_jobs_are_not_listed(self):
        client = _client_with([
            _summary("dlrm-bid-shader-1", "Completed", created_ago_h=3, ended_ago_h=1),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            assert list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp()) == []

    def test_recent_failure_included_with_reason(self):
        client = _client_with([
            _summary("dlrm-bid-shader-2", "Failed", created_ago_h=3, ended_ago_h=2),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            jobs = list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp())
        assert len(jobs) == 1
        assert jobs[0]["status"] == "Failed"
        assert jobs[0]["failure_reason"] == "AlgorithmError: boom"
        assert jobs[0]["ended_at"] == (NOW - timedelta(hours=2)).isoformat()
        client.describe_training_job.assert_called_once_with(TrainingJobName="dlrm-bid-shader-2")

    def test_only_most_recent_failure_and_only_within_24h(self):
        client = _client_with([
            _summary("dlrm-bid-shader-3", "Failed", created_ago_h=3, ended_ago_h=2),
            _summary("dlrm-bid-shader-2", "Failed", created_ago_h=6, ended_ago_h=5),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            jobs = list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp())
        assert [j["job_name"] for j in jobs] == ["dlrm-bid-shader-3"]

        old = _client_with([
            _summary("dlrm-bid-shader-1", "Failed", created_ago_h=30, ended_ago_h=25),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=old):
            assert list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp()) == []
        old.describe_training_job.assert_not_called()

    def test_active_jobs_come_before_the_failure(self):
        client = _client_with([
            _summary("dlrm-bid-shader-2", "Failed", created_ago_h=1, ended_ago_h=0.5),
            _summary("dlrm-bid-shader-1", "Stopping", created_ago_h=2),
        ])
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            jobs = list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp())
        assert [j["status"] for j in jobs] == ["Stopping", "Failed"]

    def test_describe_failure_leaves_row_without_reason(self):
        client = _client_with([
            _summary("dlrm-bid-shader-2", "Stopped", created_ago_h=3, ended_ago_h=2),
        ])
        client.describe_training_job.side_effect = RuntimeError("AccessDenied")
        with patch("orchestrator.training_trigger._sagemaker_client", return_value=client):
            jobs = list_training_jobs_for_display("dlrm_bid_shader", now=NOW.timestamp())
        assert jobs[0]["status"] == "Stopped"
        assert jobs[0]["failure_reason"] is None


def _models_client():
    app = Starlette(routes=[
        Route("/v1/closed-loop/models", closed_loop_api.models_handler, methods=["GET"]),
    ])
    return TestClient(app)


VERSIONS = [{
    "model_package_arn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/g/1",
    "version": 1, "approval_status": "Approved", "status": "Completed",
    "created_at": "2026-10-04T20:07:09+00:00",
}]


class TestModelsHandlerTrainingJobs:
    def test_response_carries_versions_and_training_jobs(self):
        jobs = [{"job_name": "dlrm-bid-shader-9", "status": "InProgress"}]
        with patch.object(closed_loop_api, "_get_sagemaker", return_value=MagicMock()), \
             patch.object(closed_loop_api.readers, "read_model_versions", return_value=VERSIONS), \
             patch("orchestrator.training_trigger.list_training_jobs_for_display",
                   return_value=jobs) as lister:
            resp = _models_client().get("/v1/closed-loop/models?model_type=dlrm_bid_shader")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_type"] == "dlrm_bid_shader"
        assert body["versions"] == VERSIONS
        assert body["training_jobs"] == jobs
        assert "training_jobs_error" not in body
        lister.assert_called_once_with("dlrm_bid_shader")

    def test_training_jobs_failure_does_not_hide_versions(self):
        with patch.object(closed_loop_api, "_get_sagemaker", return_value=MagicMock()), \
             patch.object(closed_loop_api.readers, "read_model_versions", return_value=VERSIONS), \
             patch("orchestrator.training_trigger.list_training_jobs_for_display",
                   side_effect=RuntimeError("AccessDeniedException")):
            resp = _models_client().get("/v1/closed-loop/models?model_type=dlrm_bid_shader")
        assert resp.status_code == 200
        body = resp.json()
        assert body["versions"] == VERSIONS
        assert body["training_jobs"] == []
        assert "AccessDeniedException" in body["training_jobs_error"]

    def test_registry_failure_still_503(self):
        with patch.object(closed_loop_api, "_get_sagemaker", return_value=MagicMock()), \
             patch.object(closed_loop_api.readers, "read_model_versions",
                          side_effect=RuntimeError("ResourceNotFound")):
            resp = _models_client().get("/v1/closed-loop/models?model_type=dlrm_bid_shader")
        assert resp.status_code == 503
        assert "ResourceNotFound" in resp.json()["error"]
