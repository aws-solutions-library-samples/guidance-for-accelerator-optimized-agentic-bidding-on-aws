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
export const POLL_MS_ACTIVE = 10000;
export const POLL_MS_SETTLED = 30000;

const SUMMARY_TEXT = {
  trainable: "Swept into the training bucket. This run is ready to train.",
  in_progress: "In progress. The outcomes are still on their way to the training bucket.",
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

/**
 * LoadTestSweepStatus — where one load test's outcomes are between finishing
 * the run and becoming selectable for training.
 *
 * Controlled by the parent: `runId` is the run selected in the Train step's
 * picker, and this component draws the four real pipeline stages (DynamoDB
 * record -> Firehose flush -> Glue sweep -> trainable) for that run by polling
 * GET /v1/governance/sweep-status?run_id=. It owns no picker of its own, so
 * there is only one run selection on the page.
 *
 * `onRunBecameTrainable` fires when the polled run crosses into trainable, so
 * the parent can refresh its trainable set (and enable the Train button) at
 * once rather than on its own interval.
 */
export default function LoadTestSweepStatus({ runId, onRunBecameTrainable }) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  const prevTrainableRef = useRef({});

  // Held in a ref so a new callback identity from the parent does not change
  // fetchStatus's identity and trigger an extra poll.
  const onTrainableRef = useRef(onRunBecameTrainable);
  useEffect(() => { onTrainableRef.current = onRunBecameTrainable; }, [onRunBecameTrainable]);

  const fetchStatus = useCallback(async () => {
    if (!runId) return;
    setChecking(true);
    try {
      const resp = await authFetch(`/api/v1/governance/sweep-status?run_id=${encodeURIComponent(runId)}`);

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
          ? "The deployed orchestrator does not serve /v1/governance/sweep-status yet. Redeploy the orchestrator to enable the pipeline view."
          : `HTTP ${resp.status} from the orchestrator: ${raw.slice(0, 160)}`);
        return;
      }

      if (!resp.ok) {
        setStatus(null);
        // run_not_found: the run aged out of the 200-record history window.
        setError(data.error || `HTTP ${resp.status}`);
        return;
      }

      setError(null);
      setStatus(data.status || null);

      // Fire only on a real false -> true transition, so the parent refreshes
      // its trainable set the moment a sweep lands. An undefined prior value
      // means this is the first poll for that run, which is not a transition.
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
  }, [runId]);

  // Initial load, and again whenever the parent changes the selected run. A
  // stale status from the previous run must not sit under the new selection
  // while the first poll is in flight.
  useEffect(() => {
    setStatus(null);
    setError(null);
    fetchStatus();
  }, [fetchStatus]);

  const pollMs = status && (status.summary === "in_progress" || status.summary === "waiting")
    ? POLL_MS_ACTIVE
    : POLL_MS_SETTLED;

  useEffect(() => {
    if (!runId) return undefined;
    const timer = setInterval(fetchStatus, pollMs);
    return () => clearInterval(timer);
  }, [fetchStatus, pollMs, runId]);

  const nodes = useMemo(() => (status?.stages || []).map((s) => ({
    id: s.id,
    label: s.label,
    service: s.service,
    icon: STAGE_ICONS[s.id],
    state: s.state,
  })), [status]);

  if (!runId) return null;

  return (
    <div className="cl-sweep-body" data-testid="loadtest-sweep-status">
      {error && (
        <div className="cl-honest cl-honest-block" data-testid="sweep-status-error">
          {error}
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
        </>
      )}

      <div className="cl-sweep-meta">
        {status && <span>Glue job: {status.glue_job_name || "not configured"}</span>}
        {status?.glue_run?.id && <span>Latest run: {status.glue_run.id} ({status.glue_run.state})</span>}
        {status?.checked_at && (
          <span>Checked {new Date(status.checked_at).toLocaleTimeString()}</span>
        )}
        {status && <span>Auto-refreshing every {pollMs / 1000}s</span>}
        <button
          type="button"
          className="btn-secondary sg-interactive"
          data-testid="sweep-status-refresh-button"
          onClick={fetchStatus}
          disabled={checking}
        >
          {checking ? <><span className="spinner" /> Checking…</> : "Check now"}
        </button>
      </div>
    </div>
  );
}
