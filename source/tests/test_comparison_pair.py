"""Auto-pairing the comparison: control = the run the challenger was trained from.

The operator previously had to remember which load test a challenger came from,
and the UI just preselected the most recent eligible run — which is not
necessarily the data the challenger learned from. With provenance now recorded on
the model version, the pair can be suggested.

The important property is honesty about WHICH baseline was chosen.
`control_source` distinguishes a provenance-backed control from a most-recent
fallback, so the UI never implies the baseline is the training data when it is
not.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import governance_api  # noqa: E402

MODEL_TYPE = "dlrm_bid_shader"


def _run(run_id, *, variant="current", samples=10, ts="2026-08-28T10:00:00+00:00"):
    return {
        "id": run_id,
        "timestamp": ts,
        "target_model_type": MODEL_TYPE,
        "target_variant": variant,
        "outcome_sample_count": samples,
    }


def _client():
    app = Starlette(routes=[
        Route(
            "/v1/governance/comparison-pair",
            governance_api.comparison_pair_handler,
            methods=["GET"],
        )
    ])
    return TestClient(app)


def _call(history, versions, *, registry_raises=False):
    def _read_versions(*a, **kw):
        if registry_raises:
            raise RuntimeError("AccessDenied")
        return versions

    with mock.patch.object(governance_api, "_get_loadtest_history_fn",
                           lambda: (lambda limit=200: history)), \
         mock.patch.object(governance_api, "_sagemaker_client", lambda: mock.Mock()), \
         mock.patch.dict(sys.modules, {}, clear=False), \
         mock.patch("closed_loop_demo.readers.read_model_versions", _read_versions), \
         mock.patch("orchestrator.closed_loop_api._model_group", lambda mt: "group"):
        return _client().get(f"/v1/governance/comparison-pair?model_type={MODEL_TYPE}")


# ---------------------------------------------------------------------------
# Provenance-backed pairing
# ---------------------------------------------------------------------------

def test_prefers_the_run_the_challenger_was_trained_from():
    history = [
        _run("lt-newest", ts="2026-08-28T12:00:00+00:00"),
        _run("lt-trained-from", ts="2026-08-28T09:00:00+00:00"),
        _run("lt-chal", variant="challenger", ts="2026-08-28T13:00:00+00:00"),
    ]
    versions = [{"training_run_id": "lt-trained-from", "model_package_arn": "arn:v2"}]
    body = _call(history, versions).json()

    assert body["control_run_id"] == "lt-trained-from"
    assert body["control_source"] == "training_provenance"
    assert body["challenger_run_id"] == "lt-chal"
    assert body["challenger_version_arn"] == "arn:v2"
    assert body["provenance_run_unavailable"] is False


def test_uses_the_newest_version_that_records_provenance():
    """Older versions may predate provenance being recorded."""
    history = [_run("lt-a"), _run("lt-b")]
    versions = [
        {"training_run_id": None, "model_package_arn": "arn:v3"},
        {"training_run_id": "lt-b", "model_package_arn": "arn:v2"},
    ]
    body = _call(history, versions).json()
    assert body["control_run_id"] == "lt-b"
    assert body["challenger_version_arn"] == "arn:v2"


# ---------------------------------------------------------------------------
# Honest fallback
# ---------------------------------------------------------------------------

def test_falls_back_to_most_recent_and_says_so():
    history = [
        _run("lt-newest", ts="2026-08-28T12:00:00+00:00"),
        _run("lt-older", ts="2026-08-28T09:00:00+00:00"),
    ]
    body = _call(history, [{"training_run_id": None}]).json()

    assert body["control_run_id"] == "lt-newest"
    assert body["control_source"] == "most_recent"
    assert body["provenance_run_unavailable"] is False


def test_flags_when_the_provenance_run_is_no_longer_eligible():
    """A recorded run can age out of history or have captured no samples.
    Falling back silently would present a weaker baseline as if it were the
    training data."""
    history = [_run("lt-newest")]
    versions = [{"training_run_id": "lt-long-gone", "model_package_arn": "arn:v2"}]
    body = _call(history, versions).json()

    assert body["control_source"] == "most_recent"
    assert body["provenance_run_unavailable"] is True
    assert body["provenance_run_id"] == "lt-long-gone"


def test_run_with_no_samples_is_not_a_usable_control():
    """list_eligible_runs excludes zero-sample runs, so provenance pointing at
    one must not be honoured."""
    history = [_run("lt-empty", samples=0), _run("lt-good")]
    versions = [{"training_run_id": "lt-empty"}]
    body = _call(history, versions).json()

    assert body["control_run_id"] == "lt-good"
    assert body["control_source"] == "most_recent"
    assert body["provenance_run_unavailable"] is True


def test_no_eligible_runs_reports_none_rather_than_a_guess():
    body = _call([], [{"training_run_id": None}]).json()
    assert body["control_run_id"] == ""
    assert body["control_source"] == "none"
    assert body["challenger_run_id"] == ""


def test_registry_failure_still_returns_a_usable_manual_pair():
    """A registry read error must not block the operator from comparing."""
    history = [_run("lt-newest"), _run("lt-chal", variant="challenger")]
    resp = _call(history, [], registry_raises=True)

    assert resp.status_code == 200
    body = resp.json()
    assert body["control_run_id"] == "lt-newest"
    assert body["control_source"] == "most_recent"
    assert body["challenger_run_id"] == "lt-chal"


def test_suggestion_only_performs_no_comparison():
    """This endpoint must not compute a verdict — compare_handler owns that, and
    still takes both ids explicitly so either side can be overridden."""
    history = [_run("lt-a"), _run("lt-chal", variant="challenger")]
    body = _call(history, [{"training_run_id": "lt-a"}]).json()
    for verdict_field in ("status", "relative_lift", "p_value", "recommendation"):
        assert verdict_field not in body


def test_frontend_labels_the_control_source():
    """A provenance-backed control and a most-recent fallback must not read the
    same in the UI."""
    panel = (
        Path(__file__).resolve().parents[2]
        / "source" / "frontend-react" / "src" / "components" / "GovernancePanel.jsx"
    ).read_text(encoding="utf-8")
    assert 'controlSource === "training_provenance"' in panel
    assert 'controlSource === "most_recent"' in panel
    assert "not necessarily the data the challenger" in panel
