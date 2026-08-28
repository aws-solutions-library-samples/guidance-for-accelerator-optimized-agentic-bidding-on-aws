"""Unit tests for orchestrator.governance_api's HTTP handlers.

Focused on the two handlers touched by the Yield Optimizer on-demand
training fix (real bugs, both confirmed live on a deployed stack):

1. trainable_runs_handler — a prior version of the model_type -> Glue job
   mapping (then a private dict in governance_api, now
   etl_trigger.GLUE_JOB_ENV_VAR_BY_MODEL_TYPE) was keyed on a
   non-existent model type ("deal_yield_manager" instead of the two real
   sub-model names), so every yield trainable-runs lookup silently
   resolved to "no Glue job configured" and always returned an empty
   list, even with real completed Glue runs and real captured outcome
   data.
2. train_handler — verifies the new XGBOOST_TRAINING_IMAGE_URI env var is
   read and passed through to trigger_training() for the xgboost-shaped
   model types (deal_yield_manager_floor/margin), and that dlrm_bid_shader
   is unaffected by its absence.

Uses a minimal Starlette TestClient app, matching the pattern already
established in test_sse_streaming.py.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from orchestrator.governance_api import (
    sweep_status_handler,
    trainable_runs_handler,
    train_handler,
)

test_app = Starlette(routes=[
    Route("/v1/governance/trainable-runs", trainable_runs_handler, methods=["GET"]),
    Route("/v1/governance/sweep-status", sweep_status_handler, methods=["GET"]),
    Route("/v1/governance/train", train_handler, methods=["POST"]),
])
client = TestClient(test_app)


class TestTrainableRunsHandler:
    """Reproduces the exact live bug: deal_yield_manager_floor/margin must
    resolve DEAL_YIELD_GLUE_JOB_NAME and return real trainable runs, not
    silently empty results due to a model-type key mismatch."""

    def test_floor_model_type_resolves_glue_job_and_returns_runs(self, monkeypatch):
        monkeypatch.setenv("DEAL_YIELD_GLUE_JOB_NAME", "nv5-deal-yield-feature-engineering-etl")

        history = [{
            "id": "lt-1",
            "target_model_type": "deal_yield_manager_floor",
            "target_variant": "current",
            "outcome_sample_count": 50,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]

        mock_glue = MagicMock()
        mock_glue.get_job_runs.return_value = {
            "JobRuns": [{
                "JobRunState": "SUCCEEDED",
                "CompletedOn": datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc),
            }]
        }

        with patch("boto3.client", return_value=mock_glue), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: history):
            resp = client.get("/v1/governance/trainable-runs?model_type=deal_yield_manager_floor")

        assert resp.status_code == 200
        data = resp.json()
        assert [r["id"] for r in data["runs"]] == ["lt-1"]
        # The Glue job actually queried must be the yield job, not the DLRM one.
        mock_glue.get_job_runs.assert_called_once_with(
            JobName="nv5-deal-yield-feature-engineering-etl", MaxResults=20
        )

    def test_margin_model_type_uses_same_glue_job_as_floor(self, monkeypatch):
        """Both models are labeled by the SAME Glue job run (it writes both
        output prefixes in one pass) -- the margin model type must resolve to
        that same job name, not a distinct/nonexistent one. Only the Glue job
        resolution is shared; each model type matches its own runs."""
        monkeypatch.setenv("DEAL_YIELD_GLUE_JOB_NAME", "nv5-deal-yield-feature-engineering-etl")

        history = [{
            "id": "lt-1",
            "target_model_type": "deal_yield_manager_margin",
            "target_variant": "current",
            "outcome_sample_count": 50,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        mock_glue = MagicMock()
        mock_glue.get_job_runs.return_value = {
            "JobRuns": [{
                "JobRunState": "SUCCEEDED",
                "CompletedOn": datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc),
            }]
        }

        with patch("boto3.client", return_value=mock_glue), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: history):
            resp = client.get("/v1/governance/trainable-runs?model_type=deal_yield_manager_margin")

        assert resp.status_code == 200
        assert [r["id"] for r in resp.json()["runs"]] == ["lt-1"]
        mock_glue.get_job_runs.assert_called_once_with(
            JobName="nv5-deal-yield-feature-engineering-etl", MaxResults=20
        )

    def test_dlrm_model_type_uses_glue_job_name_not_deal_yield(self, monkeypatch):
        """Sanity check that dlrm_bid_shader still resolves GLUE_JOB_NAME,
        unaffected by the deal-yield key fix."""
        monkeypatch.setenv("GLUE_JOB_NAME", "nv5-feature-engineering-etl")

        history = [{
            "id": "lt-2",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 100,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        mock_glue = MagicMock()
        mock_glue.get_job_runs.return_value = {
            "JobRuns": [{
                "JobRunState": "SUCCEEDED",
                "CompletedOn": datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc),
            }]
        }

        with patch("boto3.client", return_value=mock_glue), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: history):
            resp = client.get("/v1/governance/trainable-runs?model_type=dlrm_bid_shader")

        assert resp.status_code == 200
        assert [r["id"] for r in resp.json()["runs"]] == ["lt-2"]
        mock_glue.get_job_runs.assert_called_once_with(
            JobName="nv5-feature-engineering-etl", MaxResults=20
        )

    def test_no_glue_job_configured_returns_empty_not_an_error(self, monkeypatch):
        """The run below DOES target the queried model type, so an empty result
        can only come from the unconfigured Glue job -- a mismatched fixture
        would make this pass for the wrong reason."""
        monkeypatch.delenv("DEAL_YIELD_GLUE_JOB_NAME", raising=False)

        history = [{
            "id": "lt-1",
            "target_model_type": "deal_yield_manager_floor",
            "target_variant": "current",
            "outcome_sample_count": 50,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]

        with patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: history):
            resp = client.get("/v1/governance/trainable-runs?model_type=deal_yield_manager_floor")

        assert resp.status_code == 200
        assert resp.json()["runs"] == []

    def test_unexpected_error_returns_500_with_valid_json(self):
        with patch(
            "orchestrator.governance_api._get_loadtest_history_fn",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/v1/governance/trainable-runs?model_type=dlrm_bid_shader")

        assert resp.status_code == 500
        assert resp.json()["reason"] == "internal_error"


class TestSweepStatusHandler:
    """The sweep-status poller that shows where a load test's outcomes are
    between the run finishing and appearing in the "Train from load test"
    picker. Its trainable verdict must come from the same list_trainable_runs
    gate the picker uses, so the card cannot say "ready" while the picker
    still omits the run."""

    _HISTORY = [
        {
            "id": "lt-newest",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 900,
            "timestamp": "2026-08-22T09:00:00+00:00",
        },
        {
            "id": "lt-older",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 400,
            "timestamp": "2026-08-22T05:00:00+00:00",
        },
    ]

    def _mock_glue(self, job_runs):
        mock_glue = MagicMock()
        mock_glue.get_job_runs.return_value = {"JobRuns": job_runs}
        return mock_glue

    def test_defaults_to_the_most_recent_run(self, monkeypatch):
        monkeypatch.setenv("GLUE_JOB_NAME", "nv5-feature-engineering-etl")

        with patch("boto3.client", return_value=self._mock_glue([])), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: self._HISTORY):
            resp = client.get("/v1/governance/sweep-status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["selected_run_id"] == "lt-newest"
        assert body["status"]["run_id"] == "lt-newest"
        assert [r["id"] for r in body["runs"]] == ["lt-newest", "lt-older"]

    def test_explicit_run_id_is_honored(self, monkeypatch):
        monkeypatch.setenv("GLUE_JOB_NAME", "nv5-feature-engineering-etl")

        with patch("boto3.client", return_value=self._mock_glue([])), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: self._HISTORY):
            resp = client.get("/v1/governance/sweep-status?run_id=lt-older")

        assert resp.status_code == 200
        assert resp.json()["status"]["run_id"] == "lt-older"

    def test_unknown_run_id_returns_404_with_the_run_list(self, monkeypatch):
        """The UI pins a user's chosen run, which can age out of the history
        window; it needs the current list back to recover."""
        monkeypatch.setenv("GLUE_JOB_NAME", "nv5-feature-engineering-etl")

        with patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: self._HISTORY):
            resp = client.get("/v1/governance/sweep-status?run_id=lt-gone")

        assert resp.status_code == 404
        body = resp.json()
        assert body["reason"] == "run_not_found"
        assert [r["id"] for r in body["runs"]] == ["lt-newest", "lt-older"]

    def test_trainable_verdict_matches_the_trainable_runs_picker(self, monkeypatch):
        """A Glue run that succeeded after the load test makes the run
        trainable in BOTH endpoints, driven off the same fixture."""
        monkeypatch.setenv("GLUE_JOB_NAME", "nv5-feature-engineering-etl")
        job_runs = [{
            "Id": "jr_1",
            "JobRunState": "SUCCEEDED",
            "StartedOn": datetime(2026, 8, 22, 9, 10, 0, tzinfo=timezone.utc),
            "CompletedOn": datetime(2026, 8, 22, 9, 30, 0, tzinfo=timezone.utc),
        }]

        with patch("boto3.client", return_value=self._mock_glue(job_runs)), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: self._HISTORY):
            sweep = client.get("/v1/governance/sweep-status?run_id=lt-newest")
            picker = client.get("/v1/governance/trainable-runs?model_type=dlrm_bid_shader")

        assert sweep.json()["status"]["trainable"] is True
        assert sweep.json()["status"]["summary"] == "trainable"
        assert "lt-newest" in [r["id"] for r in picker.json()["runs"]]

    def test_in_flight_sweep_is_reported_as_in_progress_not_trainable(self, monkeypatch):
        monkeypatch.setenv("GLUE_JOB_NAME", "nv5-feature-engineering-etl")
        job_runs = [{
            "Id": "jr_2",
            "JobRunState": "RUNNING",
            "StartedOn": datetime(2026, 8, 22, 9, 10, 0, tzinfo=timezone.utc),
        }]

        with patch("boto3.client", return_value=self._mock_glue(job_runs)), \
             patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: self._HISTORY):
            resp = client.get("/v1/governance/sweep-status?run_id=lt-newest")

        body = resp.json()["status"]
        assert body["trainable"] is False
        assert body["summary"] == "in_progress"
        assert body["glue_run"]["id"] == "jr_2"

    def test_empty_history_returns_no_status_rather_than_an_error(self):
        with patch("orchestrator.governance_api._get_loadtest_history_fn", return_value=lambda limit: []):
            resp = client.get("/v1/governance/sweep-status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["runs"] == []
        assert body["selected_run_id"] == ""
        assert body["status"] is None

    def test_unexpected_error_returns_500_with_valid_json(self):
        with patch(
            "orchestrator.governance_api._get_loadtest_history_fn",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/v1/governance/sweep-status")

        assert resp.status_code == 500
        assert resp.json()["reason"] == "internal_error"


class TestTrainHandlerXGBoostImageWiring:
    """Verifies XGBOOST_TRAINING_IMAGE_URI is read from the environment and
    threaded through to trigger_training() -- the new wiring this session
    added for the on-demand Yield Optimizer training trigger."""

    _ENV = {
        "SAGEMAKER_TRAINING_ROLE_ARN": "arn:aws:iam::123456789012:role/sagemaker-training",
        "MODEL_BUCKET": "artf-model-bucket",
        "TRAINING_DATA_BUCKET": "artf-training-data-bucket",
        "TRAINING_IMAGE_REGISTRY": "123456789012.dkr.ecr.us-east-1.amazonaws.com",
    }

    def test_xgboost_image_uri_passed_through_for_floor_target(self, monkeypatch):
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("XGBOOST_TRAINING_IMAGE_URI", "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1")

        with patch("orchestrator.governance_api.trigger_training") as mock_trigger:
            mock_trigger.return_value = MagicMock(
                job_name="deal-yield-manager-floor-123-abc",
                model_type="deal_yield_manager_floor",
                base_model_version="arn:...:1",
                instance_type="ml.g5.2xlarge",
            )
            resp = client.post(
                "/v1/governance/train",
                json={"model_type": "deal_yield_manager_floor", "confirmed": True},
            )

        assert resp.status_code == 202
        call_kwargs = mock_trigger.call_args[1]
        assert call_kwargs["xgboost_training_image_uri"] == (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1"
        )

    def test_missing_xgboost_image_uri_passes_none_not_empty_string(self, monkeypatch):
        """An unset env var must resolve to None (matching
        trigger_training()'s xgboost_training_image_uri: str | None = None
        default), not an empty string -- an empty string would be a falsy-
        but-distinct value that could mask a real "not configured" state
        differently than the function's own default."""
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("XGBOOST_TRAINING_IMAGE_URI", raising=False)

        with patch("orchestrator.governance_api.trigger_training") as mock_trigger:
            mock_trigger.return_value = MagicMock(
                job_name="dlrm-bid-shader-123-abc",
                model_type="dlrm_bid_shader",
                base_model_version="arn:...:1",
                instance_type="ml.g5.2xlarge",
            )
            resp = client.post(
                "/v1/governance/train",
                json={"model_type": "dlrm_bid_shader", "confirmed": True},
            )

        assert resp.status_code == 202
        call_kwargs = mock_trigger.call_args[1]
        assert call_kwargs["xgboost_training_image_uri"] is None

    def test_xgboost_image_not_configured_error_returns_503(self, monkeypatch):
        for k, v in self._ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("XGBOOST_TRAINING_IMAGE_URI", raising=False)

        from orchestrator.training_trigger import XGBoostTrainingImageNotConfiguredError

        with patch(
            "orchestrator.governance_api.trigger_training",
            side_effect=XGBoostTrainingImageNotConfiguredError("deal_yield_manager_floor"),
        ):
            resp = client.post(
                "/v1/governance/train",
                json={"model_type": "deal_yield_manager_floor", "confirmed": True},
            )

        assert resp.status_code == 503
        assert resp.json()["reason"] == "xgboost_image_not_configured"

    def test_missing_deployment_config_returns_503_before_calling_trigger(self, monkeypatch):
        for k in self._ENV:
            monkeypatch.delenv(k, raising=False)

        with patch("orchestrator.governance_api.trigger_training") as mock_trigger:
            resp = client.post(
                "/v1/governance/train",
                json={"model_type": "dlrm_bid_shader", "confirmed": True},
            )

        assert resp.status_code == 503
        mock_trigger.assert_not_called()
