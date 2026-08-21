"""Load-test run eligibility filtering (LoadTestRunEligibilityService).

Filters the EXISTING DynamoDB-backed load-test run history
(source/orchestrator/loadtest.py's _get_history_from_dynamodb) to runs
eligible for governance comparison, and auto-selects the most recent
eligible run per role. Introduces no new persistence — the durable history
already exists (Question 4's original "in-memory/ephemeral" premise was
corrected during Application Design after a context-gatherer investigation).

Maps to: FR-7 (Story 5, governance-comparison-promotion unit).
"""

from __future__ import annotations

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
