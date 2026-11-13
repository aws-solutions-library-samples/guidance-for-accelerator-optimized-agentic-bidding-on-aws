"""Derives where a load-test run sits in the outcome-to-training-data pipeline.

A load test's outcomes travel: DynamoDB run record -> Kinesis Firehose buffer
-> S3 raw outcomes -> Glue ETL sweep -> training-data/ prefix. Only after the
final hop does the run pass ``loadtest_eligibility._is_trainable`` and appear in
the Governance panel's "Train from load test" picker. Before that the run is
simply absent from the picker, with no indication of which hop it is waiting on.

This module turns already-fetched data (a run's DynamoDB history record plus an
unfiltered ``glue:GetJobRuns`` response) into four stages the UI can render:

  1. ``recorded``  — the run's DynamoDB record exists, carrying its timestamp.
  2. ``flushed``   — Firehose's buffer window (ETL_SWEEP_DELAY_SECONDS) elapsed.
  3. ``swept``     — a Glue run completed after the load test's timestamp.
  4. ``trainable`` — the run passes the picker's trainability gate.

It performs no I/O, mirroring ``loadtest_eligibility``'s pure-filter design so
the stage logic is directly testable without AWS.

The ``trainable`` argument to :func:`build_sweep_status` must be computed by the
caller from ``loadtest_eligibility.list_trainable_runs`` — the same function
that populates the picker. Recomputing that verdict here would let the status
card and the picker disagree about whether a run is selectable.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Glue JobRunStates that mean a sweep is still in flight. WAITING appears when a
# run is queued behind the jobs' MaxConcurrentRuns: 1 limit (glue_etl_cfn.yaml).
GLUE_IN_FLIGHT_STATES = frozenset({"STARTING", "RUNNING", "STOPPING", "WAITING"})
GLUE_FAILED_STATES = frozenset({"FAILED", "TIMEOUT", "STOPPED", "ERROR"})

STAGE_RECORDED = "recorded"
STAGE_FLUSHED = "flushed"
STAGE_SWEPT = "swept"
STAGE_TRAINABLE = "trainable"

# Stage states the frontend renders. "processing"/"ok"/"error" match the classes
# PipelineBar already understands; "pending" and "blocked" render as inactive.
_OK = "ok"
_PENDING = "pending"
_PROCESSING = "processing"
_ERROR = "error"
_BLOCKED = "blocked"

# Summary codes, ordered by precedence in _summarize().
SUMMARY_TRAINABLE = "trainable"
SUMMARY_BLOCKED = "blocked"
SUMMARY_FAILED = "failed"
SUMMARY_IN_PROGRESS = "in_progress"
SUMMARY_WAITING = "waiting"


def as_utc(value) -> datetime | None:
    """Coerce a boto3 datetime or an ISO-8601 string to an aware UTC datetime.

    Naive values are assumed UTC, matching ``loadtest_eligibility._is_trainable``
    (load-test timestamps are written by ``datetime.now(timezone.utc)`` but
    round-trip through DynamoDB as strings).
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value) -> str | None:
    dt = as_utc(value)
    return dt.isoformat() if dt else None


def _stage(stage_id: str, label: str, service: str, state: str, detail: str) -> dict:
    return {"id": stage_id, "label": label, "service": service, "state": state, "detail": detail}


def _describe_glue_runs(glue_job_runs: list[dict], run_ts: datetime | None) -> dict:
    """Classify a job's recent runs relative to a load test's timestamp.

    Returns the most recent covering (SUCCEEDED after ``run_ts``), in-flight, and
    failed run. Failures are only attributed to this load test when the Glue run
    started after it; an older failure belongs to an earlier sweep window.
    """
    covering = None
    in_flight = None
    failed = None

    for raw in glue_job_runs or []:
        state = (raw.get("JobRunState") or "").upper()
        started = as_utc(raw.get("StartedOn"))
        completed = as_utc(raw.get("CompletedOn"))
        entry = {
            "id": raw.get("Id"),
            "state": state,
            "started_on": _iso(started),
            "completed_on": _iso(completed),
            "error_message": raw.get("ErrorMessage"),
            "_started": started,
            "_completed": completed,
        }

        if state == "SUCCEEDED" and completed and run_ts and completed >= run_ts:
            if covering is None or (covering["_completed"] and completed > covering["_completed"]):
                covering = entry
        elif state in GLUE_IN_FLIGHT_STATES:
            if in_flight is None or (started and in_flight["_started"] and started > in_flight["_started"]):
                in_flight = entry
        elif state in GLUE_FAILED_STATES and started and run_ts and started >= run_ts:
            if failed is None or (failed["_started"] and started > failed["_started"]):
                failed = entry

    return {"covering": covering, "in_flight": in_flight, "failed": failed}


def _public(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _summarize(stages: list[dict], trainable: bool) -> str:
    states = {s["state"] for s in stages}
    if trainable:
        return SUMMARY_TRAINABLE
    if _BLOCKED in states:
        return SUMMARY_BLOCKED
    if _ERROR in states:
        return SUMMARY_FAILED
    if _PROCESSING in states:
        return SUMMARY_IN_PROGRESS
    return SUMMARY_WAITING


def build_sweep_status(
    run: dict,
    glue_job_runs: list[dict],
    glue_job_name: str,
    *,
    now: datetime,
    sweep_delay_seconds: int,
    trainable: bool,
) -> dict:
    """Build the four-stage sweep status for one load-test run.

    ``run`` is a record from ``loadtest._get_history_from_dynamodb``.
    ``glue_job_runs`` is the unfiltered ``JobRuns`` list from
    ``glue:GetJobRuns`` for ``glue_job_name`` (``""`` when the model type has no
    configured job). ``trainable`` comes from
    ``loadtest_eligibility.list_trainable_runs`` — see the module docstring.
    """
    now = as_utc(now) or datetime.now(timezone.utc)
    run_ts = as_utc(run.get("timestamp"))
    target_model_type = run.get("target_model_type") or ""
    sample_count = int(run.get("outcome_sample_count") or 0)
    glue = _describe_glue_runs(glue_job_runs, run_ts)

    stages: list[dict] = []

    # ── 1. Run recorded ──────────────────────────────────────────────────────
    if run_ts is None:
        stages.append(_stage(
            STAGE_RECORDED, "Run recorded", "DynamoDB", _ERROR,
            "This run's record has no usable timestamp, so its position in the "
            "sweep window cannot be determined.",
        ))
    else:
        stages.append(_stage(
            STAGE_RECORDED, "Run recorded", "DynamoDB", _OK,
            f"Completed {run_ts.isoformat()} with {sample_count} outcome "
            f"sample{'' if sample_count == 1 else 's'} captured"
            + (f" for {target_model_type}." if target_model_type else "."),
        ))

    # ── 2. Outcomes flushed to S3 by Firehose ────────────────────────────────
    if not target_model_type:
        flushed = _stage(
            STAGE_FLUSHED, "Outcomes to S3", "Kinesis Firehose", _BLOCKED,
            "This load test did not target a model, so it emitted no outcome "
            "events and no sweep was scheduled for it.",
        )
    elif sample_count <= 0:
        flushed = _stage(
            STAGE_FLUSHED, "Outcomes to S3", "Kinesis Firehose", _BLOCKED,
            "This run captured 0 outcome samples, so there is nothing for the "
            "ETL sweep to label.",
        )
    elif run_ts is None:
        flushed = _stage(
            STAGE_FLUSHED, "Outcomes to S3", "Kinesis Firehose", _PENDING,
            "Cannot tell whether the Firehose buffer window has elapsed without "
            "a run timestamp.",
        )
    else:
        elapsed = (now - run_ts).total_seconds()
        if elapsed < sweep_delay_seconds:
            remaining = int(sweep_delay_seconds - elapsed)
            flushed = _stage(
                STAGE_FLUSHED, "Outcomes to S3", "Kinesis Firehose", _PROCESSING,
                f"Firehose buffers up to {sweep_delay_seconds}s before writing "
                f"this run's outcomes to S3. An on-demand sweep starts in about "
                f"{remaining}s.",
            )
        else:
            flushed = _stage(
                STAGE_FLUSHED, "Outcomes to S3", "Kinesis Firehose", _OK,
                f"The {sweep_delay_seconds}s Firehose buffer window elapsed "
                f"{int(elapsed - sweep_delay_seconds)}s ago.",
            )
    stages.append(flushed)

    # ── 3. Glue ETL sweep ────────────────────────────────────────────────────
    if not glue_job_name:
        swept = _stage(
            STAGE_SWEPT, "ETL sweep", "AWS Glue", _BLOCKED,
            f"No Glue job is configured for {target_model_type or 'this run'}, "
            "so its outcomes will not be swept into the training bucket.",
        )
    elif sample_count <= 0 or not target_model_type:
        swept = _stage(
            STAGE_SWEPT, "ETL sweep", "AWS Glue", _BLOCKED,
            f"{glue_job_name} has nothing to sweep for this run.",
        )
    elif glue["covering"]:
        swept = _stage(
            STAGE_SWEPT, "ETL sweep", "AWS Glue", _OK,
            f"{glue_job_name} run {glue['covering']['id']} succeeded at "
            f"{glue['covering']['completed_on']}, after this load test.",
        )
    elif glue["in_flight"]:
        started = glue["in_flight"]["_started"]
        covers = bool(started and run_ts and started >= run_ts)
        if covers:
            swept = _stage(
                STAGE_SWEPT, "ETL sweep", "AWS Glue", _PROCESSING,
                f"{glue_job_name} run {glue['in_flight']['id']} is "
                f"{glue['in_flight']['state']}, started "
                f"{glue['in_flight']['started_on']} — after this load test, so "
                "it will cover this run's outcomes.",
            )
        else:
            swept = _stage(
                STAGE_SWEPT, "ETL sweep", "AWS Glue", _PROCESSING,
                f"{glue_job_name} run {glue['in_flight']['id']} is "
                f"{glue['in_flight']['state']}, but it started "
                f"{glue['in_flight']['started_on']} — before this load test's "
                "outcomes reached S3, so it will not cover them. The job allows "
                "one concurrent run, so this run waits for the following sweep "
                "(scheduled every 6 hours).",
            )
    elif glue["failed"]:
        detail = (
            f"{glue_job_name} run {glue['failed']['id']} ended "
            f"{glue['failed']['state']}"
        )
        if glue["failed"]["error_message"]:
            detail += f": {glue['failed']['error_message']}"
        swept = _stage(
            STAGE_SWEPT, "ETL sweep", "AWS Glue", _ERROR,
            detail + ". This run stays unavailable for training until a sweep succeeds.",
        )
    else:
        swept = _stage(
            STAGE_SWEPT, "ETL sweep", "AWS Glue", _PENDING,
            f"No {glue_job_name} run has started since this load test. An "
            "on-demand sweep is scheduled after the Firehose buffer window, and "
            "the job also runs on a 6-hour schedule.",
        )
    stages.append(swept)

    # ── 4. Available in the training picker ──────────────────────────────────
    if trainable:
        trainable_stage = _stage(
            STAGE_TRAINABLE, "Ready to train", "Governance", _OK,
            "This run is selectable in the Load test run picker below.",
        )
    elif swept["state"] == _BLOCKED or flushed["state"] == _BLOCKED:
        trainable_stage = _stage(
            STAGE_TRAINABLE, "Ready to train", "Governance", _BLOCKED,
            "This run will not become selectable for training.",
        )
    else:
        trainable_stage = _stage(
            STAGE_TRAINABLE, "Ready to train", "Governance", _PENDING,
            "Appears in the Load test run picker below once a sweep covering "
            "this run succeeds.",
        )
    stages.append(trainable_stage)

    return {
        "run_id": run.get("id"),
        "run_timestamp": _iso(run_ts),
        "target_model_type": target_model_type,
        "target_variant": run.get("target_variant") or "",
        "outcome_sample_count": sample_count,
        "glue_job_name": glue_job_name,
        "sweep_delay_seconds": sweep_delay_seconds,
        "trainable": trainable,
        "summary": _summarize(stages, trainable),
        "stages": stages,
        "glue_run": _public(glue["covering"] or glue["in_flight"] or glue["failed"]),
        "checked_at": now.isoformat(),
    }
