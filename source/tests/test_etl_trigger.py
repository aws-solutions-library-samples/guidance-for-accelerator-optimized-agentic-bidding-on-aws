"""Tests for the on-demand Glue ETL sweep trigger (orchestrator/etl_trigger.py).

Covers the reason this module exists: a load-test run only becomes trainable
once a Glue sweep covering its timestamp has completed, and those sweeps are
otherwise only scheduled every 6 hours, so a run recorded just after one stays
untrainable for hours.

The correctness-critical property here is the delay. Outcomes reach S3 through
Firehose with a 300s buffering interval, so a sweep started immediately would
complete without the run's data present -- and its CompletedOn would then
satisfy the trainability gate anyway, offering a run whose training input is
genuinely empty. That is the failure the gate was added to prevent, so a
regression shortening the delay below the buffer window is a real defect and is
asserted against directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import etl_trigger  # noqa: E402


# ---------------------------------------------------------------------------
# Job-name resolution
# ---------------------------------------------------------------------------

def test_resolves_dlrm_to_bid_outcome_etl(monkeypatch):
    monkeypatch.setenv("GLUE_JOB_NAME", "p-feature-engineering-etl")
    assert etl_trigger.resolve_glue_job_name("dlrm_bid_shader") == "p-feature-engineering-etl"


@pytest.mark.parametrize(
    "model_type", ["deal_yield_manager_floor", "deal_yield_manager_margin"]
)
def test_both_yield_models_share_one_etl_job(monkeypatch, model_type):
    """glue_deal_yield_feature_engineering.py writes both the floor and margin
    output prefixes in a single run, so both models map to the same job."""
    monkeypatch.setenv("DEAL_YIELD_GLUE_JOB_NAME", "p-deal-yield-feature-engineering-etl")
    assert etl_trigger.resolve_glue_job_name(model_type) == "p-deal-yield-feature-engineering-etl"


def test_unknown_model_type_resolves_to_empty():
    assert etl_trigger.resolve_glue_job_name("no_such_model") == ""


def test_mapped_type_with_unset_env_var_resolves_to_empty(monkeypatch):
    """An unset env var must read as "not configured" rather than falling back
    to a guessed job name."""
    monkeypatch.delenv("GLUE_JOB_NAME", raising=False)
    assert etl_trigger.resolve_glue_job_name("dlrm_bid_shader") == ""


def test_mapping_covers_exactly_the_trainable_model_types():
    """Guards against the historical bug where the mapping was keyed on a
    model type that does not exist, silently disabling yield lookups."""
    assert set(etl_trigger.GLUE_JOB_ENV_VAR_BY_MODEL_TYPE) == {
        "dlrm_bid_shader",
        "deal_yield_manager_floor",
        "deal_yield_manager_margin",
    }


# ---------------------------------------------------------------------------
# The delay must clear the Firehose buffer window
# ---------------------------------------------------------------------------

def test_sweep_delay_exceeds_firehose_buffer_interval():
    """feedback_pipeline_cfn.yaml sets BufferingHints.IntervalInSeconds=300 on
    both delivery streams. Sweeping at or before that boundary can mark a run
    trainable with no data behind it."""
    assert etl_trigger.SWEEP_DELAY_SECONDS > 300


# ---------------------------------------------------------------------------
# Fire-and-forget contract: never raises
# ---------------------------------------------------------------------------

def test_unknown_model_type_returns_none_without_calling_aws(monkeypatch):
    def _boom(*a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError("must not construct a boto3 client for an unmapped type")

    monkeypatch.setattr("boto3.client", _boom, raising=False)
    assert etl_trigger.trigger_etl_sweep("no_such_model") is None


def test_returns_run_id_on_success(monkeypatch):
    monkeypatch.setenv("GLUE_JOB_NAME", "p-feature-engineering-etl")
    calls = {}

    class _Glue:
        def start_job_run(self, JobName):
            calls["job"] = JobName
            return {"JobRunId": "jr_abc123"}

    monkeypatch.setattr("boto3.client", lambda *a, **kw: _Glue())
    assert etl_trigger.trigger_etl_sweep("dlrm_bid_shader") == "jr_abc123"
    assert calls["job"] == "p-feature-engineering-etl"


def test_concurrent_run_is_swallowed_and_reported_as_not_started(monkeypatch):
    """The Glue jobs are MaxConcurrentRuns: 1. A concurrent start must not
    raise, and must return None rather than implying coverage -- the in-flight
    sweep may predate this run's outcomes reaching S3."""
    monkeypatch.setenv("GLUE_JOB_NAME", "p-feature-engineering-etl")

    class ConcurrentRunsExceededException(Exception):
        pass

    class _Glue:
        def start_job_run(self, JobName):
            raise ConcurrentRunsExceededException("already running")

    monkeypatch.setattr("boto3.client", lambda *a, **kw: _Glue())
    assert etl_trigger.trigger_etl_sweep("dlrm_bid_shader") is None


def test_arbitrary_api_error_is_swallowed(monkeypatch):
    """Runs on the load-test completion path, which must not fail because a
    best-effort sweep could not start."""
    monkeypatch.setenv("GLUE_JOB_NAME", "p-feature-engineering-etl")

    class _Glue:
        def start_job_run(self, JobName):
            raise RuntimeError("AccessDeniedException: glue:StartJobRun")

    monkeypatch.setattr("boto3.client", lambda *a, **kw: _Glue())
    assert etl_trigger.trigger_etl_sweep("dlrm_bid_shader") is None
