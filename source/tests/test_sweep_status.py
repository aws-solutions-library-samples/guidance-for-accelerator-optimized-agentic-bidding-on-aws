"""Tests for orchestrator.sweep_status — the load-test outcome sweep lifecycle.

The stage logic is pure (no boto3, no DynamoDB), so every case below is driven
by literal run records and literal glue:GetJobRuns payloads. The cases that
matter most are the ones the previous UI could not express at all:

- a Glue run that is RUNNING but started BEFORE the load test finished, so it
  will not cover that run's outcomes (the MaxConcurrentRuns: 1 /
  ConcurrentRunsExceededException case in etl_trigger.trigger_etl_sweep);
- a run that captured 0 outcome samples, which no sweep will ever make
  trainable, versus one that is merely waiting;
- a Glue failure that predates the load test, which belongs to an earlier sweep
  window and must not be reported as this run's failure.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.sweep_status import (
    SUMMARY_BLOCKED,
    SUMMARY_FAILED,
    SUMMARY_IN_PROGRESS,
    SUMMARY_TRAINABLE,
    SUMMARY_WAITING,
    as_utc,
    build_sweep_status,
)

DELAY = 360
RUN_TS = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def _run(**overrides) -> dict:
    run = {
        "id": "lt-f79654718b4c",
        "timestamp": RUN_TS.isoformat(),
        "target_model_type": "dlrm_bid_shader",
        "target_variant": "current",
        "outcome_sample_count": 1200,
    }
    run.update(overrides)
    return run


def _glue_run(state: str, *, started_offset_s: int, completed_offset_s: int | None = None, **extra) -> dict:
    entry = {
        "Id": "jr_abc123",
        "JobRunState": state,
        "StartedOn": RUN_TS + timedelta(seconds=started_offset_s),
    }
    if completed_offset_s is not None:
        entry["CompletedOn"] = RUN_TS + timedelta(seconds=completed_offset_s)
    entry.update(extra)
    return entry


def _stage(status: dict, stage_id: str) -> dict:
    return next(s for s in status["stages"] if s["id"] == stage_id)


def _build(run, glue_runs, *, now_offset_s, job_name="bid-shading-etl", trainable=False):
    return build_sweep_status(
        run,
        glue_runs,
        job_name,
        now=RUN_TS + timedelta(seconds=now_offset_s),
        sweep_delay_seconds=DELAY,
        trainable=trainable,
    )


class TestAsUtc:
    def test_naive_string_is_assumed_utc(self):
        assert as_utc("2026-08-28T12:00:00") == RUN_TS

    def test_aware_string_round_trips(self):
        assert as_utc("2026-08-28T12:00:00+00:00") == RUN_TS

    def test_naive_datetime_is_assumed_utc(self):
        assert as_utc(datetime(2026, 8, 28, 12, 0, 0)) == RUN_TS

    def test_unparseable_values_are_none(self):
        assert as_utc("not a timestamp") is None
        assert as_utc(None) is None
        assert as_utc(1234) is None


class TestFirehoseWindow:
    def test_inside_buffer_window_is_processing(self):
        status = _build(_run(), [], now_offset_s=100)
        flushed = _stage(status, "flushed")
        assert flushed["state"] == "processing"
        assert "260s" in flushed["detail"]
        assert status["summary"] == SUMMARY_IN_PROGRESS

    def test_after_buffer_window_is_ok(self):
        status = _build(_run(), [], now_offset_s=400)
        assert _stage(status, "flushed")["state"] == "ok"

    def test_recorded_stage_reports_sample_count_and_target(self):
        recorded = _stage(_build(_run(), [], now_offset_s=400), "recorded")
        assert recorded["state"] == "ok"
        assert "1200 outcome samples" in recorded["detail"]
        assert "dlrm_bid_shader" in recorded["detail"]


class TestGlueSweepStage:
    def test_succeeded_run_after_load_test_covers_it(self):
        status = _build(
            _run(),
            [_glue_run("SUCCEEDED", started_offset_s=380, completed_offset_s=500)],
            now_offset_s=600,
            trainable=True,
        )
        swept = _stage(status, "swept")
        assert swept["state"] == "ok"
        assert "jr_abc123" in swept["detail"]
        assert _stage(status, "trainable")["state"] == "ok"
        assert status["summary"] == SUMMARY_TRAINABLE

    def test_succeeded_run_before_load_test_does_not_cover_it(self):
        status = _build(
            _run(),
            [_glue_run("SUCCEEDED", started_offset_s=-900, completed_offset_s=-600)],
            now_offset_s=400,
        )
        assert _stage(status, "swept")["state"] == "pending"
        assert status["summary"] == SUMMARY_WAITING

    def test_in_flight_run_started_after_load_test_will_cover_it(self):
        status = _build(
            _run(),
            [_glue_run("RUNNING", started_offset_s=380)],
            now_offset_s=500,
        )
        swept = _stage(status, "swept")
        assert swept["state"] == "processing"
        assert "will cover this run" in swept["detail"]
        assert status["summary"] == SUMMARY_IN_PROGRESS

    def test_in_flight_run_started_before_load_test_will_not_cover_it(self):
        """The MaxConcurrentRuns: 1 case — a sweep is genuinely running, but it
        began before this run's outcomes reached S3, so it is not progress
        toward this run becoming trainable."""
        status = _build(
            _run(),
            [_glue_run("RUNNING", started_offset_s=-120)],
            now_offset_s=500,
        )
        swept = _stage(status, "swept")
        assert swept["state"] == "processing"
        assert "will not cover them" in swept["detail"]
        assert "one concurrent run" in swept["detail"]
        assert _stage(status, "trainable")["state"] == "pending"

    def test_waiting_state_is_used_when_no_run_has_started(self):
        status = _build(_run(), [], now_offset_s=500)
        assert _stage(status, "swept")["state"] == "pending"
        assert status["summary"] == SUMMARY_WAITING

    def test_queued_waiting_run_counts_as_in_flight(self):
        status = _build(
            _run(),
            [_glue_run("WAITING", started_offset_s=380)],
            now_offset_s=500,
        )
        assert _stage(status, "swept")["state"] == "processing"


class TestGlueFailures:
    def test_failure_after_load_test_is_reported_with_its_error(self):
        status = _build(
            _run(),
            [_glue_run(
                "FAILED", started_offset_s=380, completed_offset_s=420,
                ErrorMessage="AnalysisException: table not found",
            )],
            now_offset_s=600,
        )
        swept = _stage(status, "swept")
        assert swept["state"] == "error"
        assert "AnalysisException" in swept["detail"]
        assert status["summary"] == SUMMARY_FAILED

    def test_failure_before_load_test_is_not_attributed_to_this_run(self):
        status = _build(
            _run(),
            [_glue_run("FAILED", started_offset_s=-900, completed_offset_s=-800)],
            now_offset_s=500,
        )
        assert _stage(status, "swept")["state"] == "pending"
        assert status["summary"] == SUMMARY_WAITING

    def test_timeout_is_treated_as_a_failure(self):
        status = _build(
            _run(),
            [_glue_run("TIMEOUT", started_offset_s=380, completed_offset_s=3000)],
            now_offset_s=4000,
        )
        assert _stage(status, "swept")["state"] == "error"


class TestBlockedRuns:
    def test_zero_outcome_samples_blocks_rather_than_waits(self):
        status = _build(_run(outcome_sample_count=0), [], now_offset_s=500)
        assert _stage(status, "flushed")["state"] == "blocked"
        assert _stage(status, "swept")["state"] == "blocked"
        assert _stage(status, "trainable")["state"] == "blocked"
        assert status["summary"] == SUMMARY_BLOCKED

    def test_untargeted_run_blocks(self):
        status = _build(_run(target_model_type=""), [], now_offset_s=500, job_name="")
        assert _stage(status, "flushed")["state"] == "blocked"
        assert "did not target a model" in _stage(status, "flushed")["detail"]
        assert status["summary"] == SUMMARY_BLOCKED

    def test_unconfigured_glue_job_blocks_rather_than_waits(self):
        """An unset GLUE_JOB_NAME means the sweep will never run — reported as
        blocked, not as a run that is still on its way."""
        status = _build(_run(), [], now_offset_s=500, job_name="")
        swept = _stage(status, "swept")
        assert swept["state"] == "blocked"
        assert "No Glue job is configured" in swept["detail"]
        assert status["summary"] == SUMMARY_BLOCKED

    def test_missing_timestamp_is_an_error_not_a_silent_pass(self):
        status = _build(_run(timestamp=""), [], now_offset_s=500)
        assert _stage(status, "recorded")["state"] == "error"
        assert _stage(status, "flushed")["state"] == "pending"


class TestPayloadShape:
    def test_metadata_is_echoed_for_the_ui(self):
        status = _build(
            _run(),
            [_glue_run("RUNNING", started_offset_s=380)],
            now_offset_s=500,
        )
        assert status["run_id"] == "lt-f79654718b4c"
        assert status["target_model_type"] == "dlrm_bid_shader"
        assert status["target_variant"] == "current"
        assert status["outcome_sample_count"] == 1200
        assert status["glue_job_name"] == "bid-shading-etl"
        assert status["sweep_delay_seconds"] == DELAY
        assert status["trainable"] is False
        assert [s["id"] for s in status["stages"]] == ["recorded", "flushed", "swept", "trainable"]

    def test_glue_run_is_json_serializable(self):
        import json

        status = _build(
            _run(),
            [_glue_run("RUNNING", started_offset_s=380)],
            now_offset_s=500,
        )
        # No raw datetimes or private keys leak into the response payload.
        json.dumps(status)
        assert status["glue_run"]["state"] == "RUNNING"
        assert isinstance(status["glue_run"]["started_on"], str)
        assert not any(k.startswith("_") for k in status["glue_run"])

    def test_decimal_like_sample_count_is_coerced(self):
        """DynamoDB round-trips numbers through Decimal, so a float can arrive."""
        status = _build(_run(outcome_sample_count=1200.0), [], now_offset_s=500)
        assert status["outcome_sample_count"] == 1200
