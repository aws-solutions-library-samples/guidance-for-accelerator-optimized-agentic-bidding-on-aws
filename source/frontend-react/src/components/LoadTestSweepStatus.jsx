import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { authFetch } from "../authFetch.js";
import {
  PipelineBar,
  IconDynamo, IconCloudWatch, IconEtl, IconRegistry,
} from "./closedLoopUi.jsx";

// Stage ids come from orchestrator/sweep_status.py. Icons are the existing
// closed-loop glyph set so this bar matches the one at the top of the panel.
const STAGE_ICONS = {
  recorded: IconDynamo,
  flushed: IconCloudWatch,
  swept: IconEtl,
  trainable: IconRegistry,
};

// Poll faster while the pipeline is still moving, matching GpuControl's
// transitional-state cadence (15s steady / 10s while scaling).
const POLL_MS_ACTIVE = 10000;
const POLL_MS_SETTLED = 30000;

const SUMMARY_TEXT = {
  trainable: "Swept into the training bucket — this run is selectable in the picker below.",
  in_progress: "In progress — the outcomes are still on their way to the training bucket.",
  waiting: "Waiting for an ETL sweep to pick up this run's outcomes.",
  failed: "The ETL sweep for this run did not succeed.",
  blocked: "This run will not become available for training.",
};

const SUMMARY_CLASS = {
  trainable: "cl-sweep-summary--ok",
  in_progress: "cl-sweep-summary--active",
  waiting: "cl-sweep-summary--active",
  failed: "cl-sweep-summary--error",
  blocked: "cl-sweep-summary--error",
};

function runOptionLabel(run) {
  const when = run.timestamp ? ` (${new Date(run.timestamp).toLocaleString()})` : "";
  const target = run.target_model_type
    ? `${run.target_model_type}${run.target_variant ? ` / ${run.target_variant}` : ""}`
    : "no model targeted";
  const samples = run.outcome_sample_count > 0 ? "" : " — 0 outcomes";
  return `${run.id} — ${target}${samples}${when}`;
}

/**
 * LoadTestSweepStatus — where a load test's outcomes are between finishing the
 * run and becoming selectable for training.
 *
 * The "Train from load test" picker below only lists runs a Glue ETL sweep has
 * already swept into the training bucket, so a run that just finished is simply
 * absent from it with no indication of why. This card polls
 * GET /v1/governance/sweep-status for the four real pipeline stages (DynamoDB
 * record -> Firehose flush -> Glue sweep -> selectable), listing every recorded
 * run and defaulting to the most recent one.
 *
 * `onRunBecameTrainable` fires when a polled run crosses into trainable, so the
 * parent can refresh its picker immediately rather than on its own interval.
 */
export default function LoadTestSweepStatus({ onRunBecameTrainable }) {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  // Once the user picks a run explicitly, stop following the newest run so a
  // poll cannot move the selection out from under them.
  const pinnedRef = useRef(false);
  const prevTrainableRef = useRef({});

  // Held in a ref so a new callback identity from the parent does not change
  // fetchStatus's identity and trigger an extra poll.
  const onTrainableRef = useRef(onRunBecameTrainable);
  useEffect(() => { onTrainableRef.current = onRunBecameTrainable; }, [onRunBecameTrainable]);

  const fetchStatus = useCallback(async () => {
    setChecking(true);
    try {
      const qs = pinnedRef.current && selectedRunId ? `?run_id=${encodeURIComponent(selectedRunId)}` : "";
      const resp = await authFetch(`/api/v1/governance/sweep-status${qs}`);

      // Read as text first. An orchestrator build that predates this endpoint
      // has no such route, and Starlette answers an unmatched route with the
      // plain-text body "Not Found" — calling resp.json() on that throws a
      // JSON syntax error, which says nothing about the actual cause.
      const raw = await resp.text();
      let data = null;
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch (_) {
        data = null;
      }

      if (data === null) {
        setStatus(null);
        setError(resp.status === 404
          ? "The deployed orchestrator does not serve /v1/governance/sweep-status yet. Redeploy the orchestrator to enable this card."
          : `HTTP ${resp.status} from the orchestrator: ${raw.slice(0, 160)}`);
        return;
      }

      // A pinned run can age out of the 200-record history window; fall back to
      // following the newest run rather than showing a dead selection.
      if (resp.status === 404 && data.reason === "run_not_found") {
        pinnedRef.current = false;
        setRuns(data.runs || []);
        setError(null);
        return;
      }
      if (!resp.ok) {
        setStatus(null);
        setError(data.error || `HTTP ${resp.status}`);
        return;
      }

      setError(null);
      setRuns(data.runs || []);
      setStatus(data.status || null);
      if (data.selected_run_id) setSelectedRunId(data.selected_run_id);

      // Fire only on a real false -> true transition, so the parent refreshes
      // its picker the moment a sweep lands. An undefined prior value means
      // this is the first poll for that run, which is not a transition.
      const st = data.status;
      if (st?.run_id) {
        const was = prevTrainableRef.current[st.run_id];
        if (st.trainable && was === false && typeof onTrainableRef.current === "function") {
          onTrainableRef.current(st);
        }
        prevTrainableRef.current[st.run_id] = !!st.trainable;
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setChecking(false);
    }
  }, [selectedRunId]);

  // Initial load, and again whenever the user changes the selected run.
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const pollMs = status && (status.summary === "in_progress" || status.summary === "waiting")
    ? POLL_MS_ACTIVE
    : POLL_MS_SETTLED;

  useEffect(() => {
    const timer = setInterval(fetchStatus, pollMs);
    return () => clearInterval(timer);
  }, [fetchStatus, pollMs]);

  const handleSelect = useCallback((e) => {
    pinnedRef.current = true;
    setSelectedRunId(e.target.value);
  }, []);

  const nodes = useMemo(() => (status?.stages || []).map((s) => ({
    id: s.id,
    label: s.label,
    service: s.service,
    icon: STAGE_ICONS[s.id],
    state: s.state,
  })), [status]);

  return (
    <div className="cl-card cl-sweep-card sg-elevated" data-testid="loadtest-sweep-status-card">
      <div className="cl-control-group">
        <label htmlFor="cl-sweep-run">Load test:</label>
        <select
          id="cl-sweep-run"
          className="cl-select sg-interactive"
          data-testid="sweep-status-run-select"
          value={selectedRunId}
          onChange={handleSelect}
          disabled={runs.length === 0}
        >
          {runs.length === 0 ? (
            // An error means the run list was never read, which is not the same
            // as there being no runs -- the Train picker below may well be
            // listing some.
            <option value="">{error ? "Run list unavailable" : "No load test runs recorded yet"}</option>
          ) : (
            runs.map((r) => (
              <option key={r.id} value={r.id}>{runOptionLabel(r)}</option>
            ))
          )}
        </select>
        <button
          className="btn-secondary sg-interactive"
          data-testid="sweep-status-refresh-button"
          onClick={fetchStatus}
          disabled={checking}
        >
          {checking ? <><span className="spinner" /> Checking…</> : "Check now"}
        </button>
      </div>

      {error && (
        <div className="cl-honest cl-honest-block" data-testid="sweep-status-error">
          {error}
        </div>
      )}

      {!error && runs.length === 0 && (
        <div className="cl-honest cl-honest-block">
          No load test has been recorded yet. Run one from the Load Test panel and its
          progress toward the training bucket will appear here.
        </div>
      )}

      {status && (
        <>
          <div
            className={`cl-sweep-summary ${SUMMARY_CLASS[status.summary] || ""}`}
            data-testid="sweep-status-summary"
            aria-live="polite"
          >
            {SUMMARY_TEXT[status.summary] || status.summary}
          </div>

          <PipelineBar
            nodes={nodes}
            revealed={nodes.length}
            running={false}
            ariaLabel="Load test outcome sweep stages"
          />

          <div className="cl-sweep-stagelog" data-testid="sweep-status-stage-log">
            {(status.stages || []).map((s) => (
              <div key={s.id} className={`cl-sweep-stageline cl-sweep-stageline--${s.state}`}>
                <strong>{s.label}:</strong> {s.detail}
              </div>
            ))}
          </div>

          <div className="cl-sweep-meta">
            <span>Glue job: {status.glue_job_name || "not configured"}</span>
            {status.glue_run?.id && <span>Latest run: {status.glue_run.id} ({status.glue_run.state})</span>}
            {status.checked_at && (
              <span>Checked {new Date(status.checked_at).toLocaleTimeString()}</span>
            )}
            <span>Auto-refreshing every {pollMs / 1000}s</span>
          </div>
        </>
      )}
    </div>
  );
}
