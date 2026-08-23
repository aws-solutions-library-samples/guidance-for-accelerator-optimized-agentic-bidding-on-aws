"""Load-test run eligibility filtering (LoadTestRunEligibilityService).

Filters the EXISTING DynamoDB-backed load-test run history
(source/orchestrator/loadtest.py's _get_history_from_dynamodb) to runs
eligible for governance comparison, and auto-selects the most recent
eligible run per role. Introduces no new persistence — the durable history
already exists (Question 4's original "in-memory/ephemeral" premise was
corrected during Application Design after a context-gatherer investigation).

Maps to: FR-7 (Story 5, governance-comparison-promotion unit).

Also filters load-test runs eligible as a "Train from load test" input
(list_trainable_runs) — a run only qualifies once the Glue ETL job that
sweeps raw outcomes into training-data/ has actually completed a run
covering it, so the training job's S3 input isn't empty (see
governance_api.trainable_runs_handler).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

RoleStr = Literal["current", "challenger"]


def _is_eligible(run: dict, model_type: str, role: RoleStr) -> bool:
    """A run is eligible for comparison if it captured outcome samples
    (outcome_sample_count > 0), targeted the requested model_type, and its
    target_variant matches the requested role."""
    if run.get("target_model_type") != model_type:
        return False
    if int(run.get("outcome_sample_count") or 0) <= 0:
        return False
    expected_variant = "stable" if role == "current" else "canary"
    run_variant = "canary" if run.get("target_variant") == "challenger" else "stable"
    return run_variant == expected_variant


def list_eligible_runs(
    history: list[dict], model_type: str, role: RoleStr
) -> list[dict]:
    """Filter a load-test run history list to those eligible for role.

    ``history`` is the list returned by loadtest.py's
    _get_history_from_dynamodb() / GET /v1/loadtest/history — this function
    performs no I/O itself, keeping it a pure, directly-testable filter over
    already-fetched data.
    """
    return [run for run in history if _is_eligible(run, model_type, role)]


def most_recent_eligible(
    history: list[dict], model_type: str, role: RoleStr
) -> dict | None:
    """Return the most recent eligible run for role, or None if none exist.

    ``history`` is assumed already sorted newest-first (matching
    _get_history_from_dynamodb()'s existing sort-by-timestamp-descending
    behavior) — this function does not re-sort, it just takes the first
    eligible match.
    """
    eligible = list_eligible_runs(history, model_type, role)
    return eligible[0] if eligible else None


def _is_trainable(run: dict, model_type: str, latest_glue_completion: datetime | None) -> bool:
    """A run is trainable if it targeted model_type, captured at least one
    outcome sample, and a Glue job run covering its timestamp has already
    completed — otherwise its data may not exist under training-data/ yet
    (see this module's docstring)."""
    if run.get("target_model_type") != model_type:
        return False
    if int(run.get("outcome_sample_count") or 0) <= 0:
        return False
    if latest_glue_completion is None:
        return False
    run_ts = run.get("timestamp")
    if not run_ts:
        return False
    try:
        run_dt = datetime.fromisoformat(run_ts)
    except ValueError:
        return False
    if run_dt.tzinfo is None:
        run_dt = run_dt.replace(tzinfo=timezone.utc)
    return run_dt <= latest_glue_completion


def list_trainable_runs(
    history: list[dict], model_type: str, latest_glue_completion: datetime | None
) -> list[dict]:
    """Filter a load-test run history list to those whose outcome data has
    actually been processed by a completed Glue job run, for model_type.

    ``latest_glue_completion`` is the CompletedOn timestamp of the most
    recent SUCCEEDED run of the Glue job that labels model_type's training
    data (resolved via a live glue:GetJobRuns call in governance_api.py —
    this function performs no I/O itself). A run started AFTER that
    timestamp is excluded, since no Glue run has swept its data yet.
    """
    return [
        run for run in history
        if _is_trainable(run, model_type, latest_glue_completion)
    ]
