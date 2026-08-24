"""Unit tests for orchestrator.loadtest_eligibility — LoadTestRunEligibilityService.

Validates:
- list_eligible_runs()/most_recent_eligible() correctly filter by
  target_model_type, outcome_sample_count > 0, and target_variant/role
  mapping ("current" -> stable, "challenger" -> canary).
- Runs targeting a different model type, with zero captured samples, or
  with the wrong variant are excluded.
- most_recent_eligible() returns the first eligible entry (history is
  assumed pre-sorted newest-first, matching loadtest.py's existing
  DynamoDB scan + timestamp-descending sort).

Maps to: FR-7 (Story 5, governance-comparison-promotion unit).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.loadtest_eligibility import list_eligible_runs, most_recent_eligible


def _run(
    *,
    id_="lt-1",
    target_model_type="dlrm_bid_shader",
    target_variant="current",
    outcome_sample_count=100,
):
    return {
        "id": id_,
        "target_model_type": target_model_type,
        "target_variant": target_variant,
        "outcome_sample_count": outcome_sample_count,
    }


class TestListEligibleRuns:
    def test_matching_model_type_and_role_included(self):
        history = [_run(id_="lt-1")]
        result = list_eligible_runs(history, "dlrm_bid_shader", "current")
        assert [r["id"] for r in result] == ["lt-1"]

    def test_different_model_type_excluded(self):
        history = [_run(id_="lt-1", target_model_type="ncf_deal_manager")]
        result = list_eligible_runs(history, "dlrm_bid_shader", "current")
        assert result == []

    def test_zero_outcome_samples_excluded(self):
        history = [_run(id_="lt-1", outcome_sample_count=0)]
        result = list_eligible_runs(history, "dlrm_bid_shader", "current")
        assert result == []

    def test_missing_outcome_sample_count_treated_as_zero(self):
        """A run predating this feature (no outcome_sample_count field at
        all) must not be treated as eligible."""
        history = [{
            "id": "lt-old",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
        }]
        result = list_eligible_runs(history, "dlrm_bid_shader", "current")
        assert result == []

    def test_challenger_role_requires_canary_variant(self):
        current_run = _run(id_="lt-current", target_variant="current")
        challenger_run = _run(id_="lt-challenger", target_variant="challenger")
        history = [current_run, challenger_run]

        assert [r["id"] for r in list_eligible_runs(history, "dlrm_bid_shader", "current")] == ["lt-current"]
        assert [r["id"] for r in list_eligible_runs(history, "dlrm_bid_shader", "challenger")] == ["lt-challenger"]

    def test_current_role_excludes_challenger_run(self):
        history = [_run(id_="lt-challenger", target_variant="challenger")]
        result = list_eligible_runs(history, "dlrm_bid_shader", "current")
        assert result == []

    def test_empty_history_returns_empty(self):
        assert list_eligible_runs([], "dlrm_bid_shader", "current") == []


class TestMostRecentEligible:
    def test_returns_first_eligible_entry(self):
        """history is pre-sorted newest-first; the first eligible match wins."""
        history = [
            _run(id_="lt-newest"),
            _run(id_="lt-older"),
        ]
        result = most_recent_eligible(history, "dlrm_bid_shader", "current")
        assert result["id"] == "lt-newest"

    def test_skips_ineligible_newer_entries(self):
        history = [
            _run(id_="lt-newest-wrong-model", target_model_type="ncf_deal_manager"),
            _run(id_="lt-older-correct-model"),
        ]
        result = most_recent_eligible(history, "dlrm_bid_shader", "current")
        assert result["id"] == "lt-older-correct-model"

    def test_none_when_no_eligible_runs(self):
        history = [_run(id_="lt-1", outcome_sample_count=0)]
        assert most_recent_eligible(history, "dlrm_bid_shader", "current") is None

    def test_none_for_empty_history(self):
        assert most_recent_eligible([], "dlrm_bid_shader", "current") is None


class TestListTrainableRuns:
    """list_trainable_runs() — only lists runs whose data has actually been
    swept into training-data/ by a completed Glue job run (this fix)."""

    def test_run_before_glue_completion_is_trainable(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 100,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "dlrm_bid_shader", latest_completion)
        assert [r["id"] for r in result] == ["lt-1"]

    def test_run_after_glue_completion_is_excluded(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 100,
            "timestamp": "2026-08-22T07:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "dlrm_bid_shader", latest_completion)
        assert result == []

    def test_no_glue_completion_yet_excludes_everything(self):
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 100,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        result = list_trainable_runs(history, "dlrm_bid_shader", None)
        assert result == []

    def test_different_model_type_excluded(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "ncf_deal_manager",
            "target_variant": "current",
            "outcome_sample_count": 100,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "dlrm_bid_shader", latest_completion)
        assert result == []

    def test_zero_outcome_samples_excluded(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 0,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "dlrm_bid_shader", latest_completion)
        assert result == []

    def test_missing_timestamp_excluded(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "dlrm_bid_shader",
            "target_variant": "current",
            "outcome_sample_count": 100,
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "dlrm_bid_shader", latest_completion)
        assert result == []

    def test_empty_history_returns_empty(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        assert list_trainable_runs([], "dlrm_bid_shader", latest_completion) == []


class TestListTrainableRunsYieldSubModelMapping:
    """deal_yield_manager_floor/margin training targets must match load
    test runs recorded under the shared container-level identifier
    "deal_yield_manager" (see _LOAD_TEST_TARGET_BY_TRAINING_MODEL_TYPE) --
    there is no per-sub-model load-test target. Reproduces a real bug: a
    prior version of this filter required an exact match against
    "deal_yield_manager_floor"/"deal_yield_manager_margin", which no load
    test run ever records, so trainable-runs always returned empty for
    both yield sub-models even with real captured outcome data."""

    def test_floor_target_matches_deal_yield_manager_run(self):
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "deal_yield_manager",
            "target_variant": "current",
            "outcome_sample_count": 50,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "deal_yield_manager_floor", latest_completion)
        assert [r["id"] for r in result] == ["lt-1"]

    def test_margin_target_matches_same_deal_yield_manager_run(self):
        """The SAME run qualifies for both training targets -- floor and
        margin outcomes are emitted independently from one load test run,
        not from separate runs."""
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "deal_yield_manager",
            "target_variant": "current",
            "outcome_sample_count": 50,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "deal_yield_manager_margin", latest_completion)
        assert [r["id"] for r in result] == ["lt-1"]

    def test_unrelated_model_type_still_excluded(self):
        """Sanity check that the remap table doesn't accidentally make
        every run trainable for every model type -- an unrelated model
        type must still be excluded."""
        from datetime import datetime, timezone
        from orchestrator.loadtest_eligibility import list_trainable_runs

        history = [{
            "id": "lt-1",
            "target_model_type": "deal_yield_manager",
            "target_variant": "current",
            "outcome_sample_count": 50,
            "timestamp": "2026-08-22T06:00:00+00:00",
        }]
        latest_completion = datetime(2026, 8, 22, 6, 30, tzinfo=timezone.utc)
        result = list_trainable_runs(history, "dlrm_bid_shader", latest_completion)
        assert result == []
