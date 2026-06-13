import { useState, useEffect, useRef, useCallback } from "react";
import DeltaLabel from "./DeltaLabel";
import nvidiaLogo from "../logos/Nvidia_logo.svg";
import { authFetch } from "../authFetch.js";

const PRESETS = [
  { value: "100", label: "100", requests: 100 },
  { value: "1k", label: "1K", requests: 1000 },
  { value: "10k", label: "10K", requests: 10000 },
  { value: "100k", label: "100K", requests: 100000 },
];

const DURATIONS = [
  { value: 10, label: "10s" },
  { value: 30, label: "30s" },
  { value: 60, label: "60s" },
  { value: 120, label: "2m" },
];

/**
 * LoadTestPanel — UI for running server-side load tests against the ARTF pipeline.
 *
 * Connects to POST /v1/loadtest to start a test, then opens an EventSource on
 * GET /v1/loadtest/{id}/stream to receive real-time progress events (every 500ms).
 *
 * Props:
 *   onRunningChange(isRunning: boolean) — called when load test starts/stops,
 *     allowing parent to disable scenario buttons during active test.
 */
export default function LoadTestPanel({ onRunningChange, onResultChange }) {
  const [preset, setPreset] = useState("100");
  const [duration, setDuration] = useState(10);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedDisplay, setElapsedDisplay] = useState(0);

  // Notify parent of result/progress changes for main area display
  useEffect(() => {
    if (onResultChange) onResultChange({ progress, result, running, error });
  }, [progress, result, running, error, onResultChange]);

  const eventSourceRef = useRef(null);
  const streamAbortRef = useRef(null);
  const timerRef = useRef(null);
  const pollRef = useRef(null);
  const startTimeRef = useRef(null);
  const testIdRef = useRef(null);

  // Notify parent of running state changes
  useEffect(() => {
    if (onRunningChange) onRunningChange(running);
  }, [running, onRunningChange]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
      if (streamAbortRef.current) streamAbortRef.current.abort();
      if (timerRef.current) clearInterval(timerRef.current);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Helper: fully reset to idle state
  const resetToIdle = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRunning(false);
    testIdRef.current = null;
  }, []);

  const stopTest = useCallback(async () => {
    const tid = testIdRef.current;
    resetToIdle();
    if (tid) {
      try { await authFetch(`/api/v1/loadtest/${tid}`, { method: "DELETE" }); } catch (_) {}
    }
  }, [resetToIdle]);

  // Polling fallback: fetch test status every 2s when SSE is unavailable
  const startPolling = useCallback((testId) => {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const r = await authFetch(`/api/v1/loadtest/${testId}`);
        if (!r.ok) {
          // Test gone (404) — completed and cleaned up
          resetToIdle();
          return;
        }
        const data = await r.json();
        if (data.state === "complete" || data.state === "error" || data.state === "cancelled") {
          setResult(data);
          resetToIdle();
        } else if (data.state === "running") {
          // Update progress from poll
          setProgress({
            completed: data.completed ?? 0,
            total: data.total_requests ?? 0,
            rps: data.rps ?? 0,
            elapsed_ms: data.elapsed_ms ?? 0,
            latency_p50: data.latency_p50,
            latency_p95: data.latency_p95,
            latency_p99: data.latency_p99,
            errors: data.errors ?? 0,
          });
        }
      } catch (_) {
        // Network error — keep polling
      }
    }, 2000);
  }, [resetToIdle]);

  const startTest = useCallback(async () => {
    setRunning(true);
    setProgress(null);
    setResult(null);
    setError(null);
    setElapsedDisplay(0);
    startTimeRef.current = Date.now();

    // Elapsed time display timer
    timerRef.current = setInterval(() => {
      setElapsedDisplay(((Date.now() - startTimeRef.current) / 1000).toFixed(1));
      if (Date.now() - startTimeRef.current > (duration + 30) * 1000) {
        stopTest();
      }
    }, 100);

    try {
      const resp = await authFetch("/api/v1/loadtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset, seed: 42, duration_s: duration }),
      });

      if (!resp.ok) {
        const body = await resp.text();
        throw new Error(resp.status === 409 ? "A load test is already running" : `HTTP ${resp.status}: ${body}`);
      }

      const { id } = await resp.json();
      testIdRef.current = id;

      // Open SSE stream using fetch (supports Authorization header, unlike EventSource)
      const abortController = new AbortController();
      streamAbortRef.current = abortController;

      const streamWithAuth = async (testId) => {
        try {
          const streamResp = await authFetch(`/api/v1/loadtest/${testId}/stream`, {
            headers: { "Accept": "text/event-stream" },
            signal: abortController.signal,
          });
          if (!streamResp.ok || !streamResp.body) {
            // Fall back to polling if streaming not available
            startPolling(testId);
            return;
          }
          const reader = streamResp.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split("\n");
            buffer = lines.pop() || ""; // Keep incomplete line in buffer

            let eventType = "";
            for (const line of lines) {
              if (line.startsWith("event: ")) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith("data: ")) {
                const data = line.slice(6);
                try {
                  const parsed = JSON.parse(data);
                  if (eventType === "progress") {
                    setProgress(parsed);
                  } else if (eventType === "complete") {
                    setResult(parsed);
                    resetToIdle();
                    return;
                  }
                } catch (_) {}
                eventType = "";
              } else if (line === "") {
                eventType = "";
              }
            }
          }
          // Stream ended naturally — poll for final result if we didn't get "complete"
          if (testIdRef.current) startPolling(testIdRef.current);
        } catch (err) {
          // Network error or aborted — fall back to polling only if not intentionally aborted
          if (err?.name !== "AbortError" && testIdRef.current) startPolling(testIdRef.current);
        }
      };

      streamWithAuth(id);

    } catch (err) {
      setError(err.message);
      resetToIdle();
    }
  }, [preset, duration, stopTest, resetToIdle, startPolling]);

  const totalRequests = PRESETS.find((p) => p.value === preset)?.requests || 0;
  const completedPct = progress ? ((progress.completed / progress.total) * 100).toFixed(1) : 0;

  return (
    <div className="loadtest-panel">
      <div className="loadtest-header">
        <h3>Load Test</h3>
        {running && (
          <span className="loadtest-elapsed">{elapsedDisplay}s</span>
        )}
      </div>

      {/* Preset selector */}
      <div className="loadtest-presets" role="radiogroup" aria-label="Load test preset">
        {PRESETS.map((p) => (
          <label
            key={p.value}
            className={`loadtest-preset ${preset === p.value ? "active" : ""}`}
          >
            <input
              type="radio"
              name="loadtest-preset"
              value={p.value}
              checked={preset === p.value}
              onChange={() => setPreset(p.value)}
              disabled={running}
              aria-label={`${p.label} requests`}
            />
            <span className="loadtest-preset-label">{p.label}</span>
            <span className="loadtest-preset-detail">
              {p.requests.toLocaleString()} req
            </span>
          </label>
        ))}
      </div>

      {/* Duration selector */}
      <div className="loadtest-duration" role="radiogroup" aria-label="Test duration">
        <span className="loadtest-duration-label">Duration:</span>
        {DURATIONS.map((d) => (
          <label
            key={d.value}
            className={`loadtest-duration-option ${duration === d.value ? "active" : ""}`}
          >
            <input
              type="radio"
              name="loadtest-duration"
              value={d.value}
              checked={duration === d.value}
              onChange={() => setDuration(d.value)}
              disabled={running}
            />
            {d.label}
          </label>
        ))}
      </div>

      {/* Run / Stop buttons */}
      <div className="loadtest-actions">
        {!running ? (
          <button
            className="btn btn-primary loadtest-run"
            onClick={startTest}
            aria-label="Run load test"
          >
            ▶ Run
          </button>
        ) : (
          <button
            className="btn loadtest-stop"
            onClick={stopTest}
            aria-label="Stop load test"
          >
            ■ Stop
          </button>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="loadtest-error" role="alert">
          {error}
        </div>
      )}

      {/* Progress indicator */}
      {running && progress && (
        <div className="loadtest-progress">
          <div className="loadtest-progress-bar-container">
            <div
              className="loadtest-progress-bar-fill"
              style={{ width: `${completedPct}%` }}
              role="progressbar"
              aria-valuenow={progress.completed}
              aria-valuemin={0}
              aria-valuemax={progress.total}
            />
          </div>
          <div className="loadtest-progress-stats">
            <span>{progress.completed.toLocaleString()} / {progress.total.toLocaleString()}</span>
            <span>{progress.rps?.toLocaleString()} RPS</span>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * LoadTestResults — renders in the main content area (where timeline normally shows).
 * Exported separately so App.jsx can place it in the main area.
 *
 * The history table is always visible. Users can click a row to set it as the
 * baseline for delta comparison against new test results.
 */
export function LoadTestResults({ progress, result, running, error: loadError }) {
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [baseline, setBaseline] = useState(null);

  // Fetch history on mount and after each test completes
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const resp = await authFetch("/api/v1/loadtest/history?limit=10");
      if (resp.ok) {
        const data = await resp.json();
        setHistory(data.history || []);
      } else {
        // Show empty state on failure (no error message)
        setHistory([]);
      }
    } catch (_) {
      // Show empty state on failure (no error message)
      setHistory([]);
    }
    setHistoryLoading(false);
  }, []);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  // Refresh history when a test completes (within 3 seconds)
  useEffect(() => {
    if (result && result.state === "complete") {
      const timer = setTimeout(fetchHistory, 2000);
      return () => clearTimeout(timer);
    }
  }, [result, fetchHistory]);

  // Also refresh when running transitions to false (covers SSE disconnect cases)
  const prevRunningRef = useRef(running);
  useEffect(() => {
    if (prevRunningRef.current && !running) {
      // Was running, now stopped — refresh history after a short delay
      const timer = setTimeout(fetchHistory, 2500);
      prevRunningRef.current = running;
      return () => clearTimeout(timer);
    }
    prevRunningRef.current = running;
  }, [running, fetchHistory]);

  // Handle clicking a history row to set/toggle baseline
  const handleRowClick = useCallback((row) => {
    setBaseline((prev) => (prev && prev.id === row.id) ? null : row);
  }, []);

  // Clear baseline selection
  const clearBaseline = useCallback(() => {
    setBaseline(null);
  }, []);

  const data = result || progress;

  return (
    <div id="flow-canvas">
      <div className="loadtest-results-main">
        {loadError && (
          <div className="loadtest-error" role="alert">{loadError}</div>
        )}

        {/* Real-time metrics — always visible, shows dashes when no data */}
        <MetricsDisplay data={data} isComplete={!!result} baseline={baseline} />

        {/* Progress bar when running */}
        {running && progress && (
          <div className="loadtest-progress" style={{ margin: "12px 0" }}>
            <div className="loadtest-progress-bar-container">
              <div
                className="loadtest-progress-bar-fill"
                style={{ width: `${((progress.completed / progress.total) * 100).toFixed(1)}%` }}
              />
            </div>
            <div className="loadtest-progress-stats">
              <span>{progress.completed?.toLocaleString()} / {progress.total?.toLocaleString()}</span>
              <span>{progress.rps?.toLocaleString()} RPS</span>
            </div>
          </div>
        )}

        {running && !data && (
          <div className="placeholder"><span className="spinner" /> Starting load test…</div>
        )}

        {/* Per-container breakdown — always visible with dashes */}
        <ContainerBreakdown containers={data?.per_container} />

        {/* Latency summary — always visible with dashes */}
        <LatencyTimeline result={result} />

        {/* History — always visible */}
        <div className="loadtest-history">
          <h4 className="loadtest-history-title">
            History
            {historyLoading && <span className="spinner" style={{ marginLeft: 8 }} />}
            {baseline && (
              <button
                className="btn btn-secondary loadtest-clear-baseline"
                onClick={clearBaseline}
                aria-label="Clear baseline selection"
              >
                Clear baseline
              </button>
            )}
          </h4>
          {history.length === 0 ? (
            <p className="loadtest-history-empty">No past results available.</p>
          ) : (
            <table className="loadtest-breakdown-table" aria-label="Load test history">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Preset</th>
                  <th>Requests</th>
                  <th>RPS</th>
                  <th>p50</th>
                  <th>p99</th>
                  <th>Errors</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => {
                  const isBaseline = baseline && baseline.id === h.id;
                  return (
                    <tr
                      key={h.id}
                      className={isBaseline ? "loadtest-history-row loadtest-history-row--baseline" : "loadtest-history-row"}
                      onClick={() => handleRowClick(h)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleRowClick(h); } }}
                      aria-pressed={isBaseline}
                      aria-label={`Select ${h.preset} test from ${h.timestamp ? new Date(h.timestamp).toLocaleString() : "unknown time"} as baseline`}
                    >
                      <td style={{ fontSize: "0.7rem" }}>{h.timestamp ? new Date(h.timestamp).toLocaleString() : "—"}</td>
                      <td>{h.preset}</td>
                      <td>{Number(h.completed || 0).toLocaleString()}</td>
                      <td>{Number(h.rps || 0).toFixed(0)}</td>
                      <td>{Number(h.latency_p50 || 0).toFixed(0)}ms</td>
                      <td>{Number(h.latency_p99 || 0).toFixed(0)}ms</td>
                      <td>{Number(h.errors || 0).toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricsDisplay({ data, isComplete, baseline }) {
  return (
    <div className="loadtest-metrics">
      <div className="loadtest-metrics-grid">
        <MetricItem label="Sent" value={data?.completed?.toLocaleString() || "—"} />
        <MetricItem label="Received" value={data ? (data.completed - (data.errors || 0))?.toLocaleString() : "—"} />
        <MetricItem
          label="Errors"
          value={data?.errors?.toLocaleString() || "—"}
          highlight={data && data.errors > 0 ? "error" : null}
          delta={baseline && data && <DeltaLabel current={data.errors} baseline={baseline.errors} unit="" lowerIsBetter={true} />}
        />
        <MetricItem label="Error Rate" value={data && data.completed > 0 ? `${((data.errors || 0) / data.completed * 100).toFixed(2)}%` : "—"} highlight={data && data.errors > 0 ? "error" : null} />
        <MetricItem
          label="RPS"
          value={data?.rps?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || "—"}
          delta={baseline && data && <DeltaLabel current={data.rps} baseline={baseline.rps} unit="" lowerIsBetter={false} />}
        />
        <MetricItem
          label="p50"
          value={data?.latency_p50 != null ? `${data.latency_p50.toFixed(1)}ms` : "—"}
          delta={baseline && data && <DeltaLabel current={data.latency_p50} baseline={baseline.latency_p50} unit="ms" lowerIsBetter={true} />}
        />
        <MetricItem
          label="p95"
          value={data?.latency_p95 != null ? `${data.latency_p95.toFixed(1)}ms` : "—"}
          delta={baseline && data && <DeltaLabel current={data.latency_p95} baseline={baseline.latency_p95} unit="ms" lowerIsBetter={true} />}
        />
        <MetricItem
          label="p99"
          value={data?.latency_p99 != null ? `${data.latency_p99.toFixed(1)}ms` : "—"}
          delta={baseline && data && <DeltaLabel current={data.latency_p99} baseline={baseline.latency_p99} unit="ms" lowerIsBetter={true} />}
        />
      </div>
    </div>
  );
}

function MetricItem({ label, value, highlight, delta }) {
  return (
    <div className={`loadtest-metric ${highlight ? `loadtest-metric--${highlight}` : ""}`}>
      <span className="loadtest-metric-value">{value}</span>
      {delta && <span className="loadtest-metric-delta">{delta}</span>}
      <span className="loadtest-metric-label">{label}</span>
    </div>
  );
}

function ContainerBreakdown({ containers }) {
  const CONTAINER_NAMES = [
    "dlrm-bid-shader",
    "widedeep-segment-activator",
    "ncf-deal-manager",
    "metrics-enricher",
  ];

  const CONTAINER_LOGOS = {
    "dlrm-bid-shader": nvidiaLogo,
    "widedeep-segment-activator": nvidiaLogo,
    "ncf-deal-manager": nvidiaLogo,
    "metrics-enricher": nvidiaLogo,
  };

  const rows = containers && containers.length > 0
    ? containers
    : CONTAINER_NAMES.map((name) => ({ name, avg_latency_ms: null, total_mutations: null }));

  return (
    <div className="loadtest-breakdown">
      <h4>Per-Container Breakdown</h4>
      <table className="loadtest-breakdown-table" aria-label="Per-container load test results">
        <thead>
          <tr>
            <th></th>
            <th>Container</th>
            <th>Avg Latency</th>
            <th>Mutations</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.name}>
              <td className="loadtest-breakdown-logo">
                <img src={CONTAINER_LOGOS[c.name]} alt="" className="loadtest-container-logo" />
              </td>
              <td className="loadtest-breakdown-name">{c.name}</td>
              <td>{c.avg_latency_ms != null ? `${c.avg_latency_ms.toFixed(1)}ms` : "—"}</td>
              <td>{c.total_mutations != null ? c.total_mutations.toLocaleString() : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * LatencyTimeline — Post-completion summary showing:
 * 1. Timeline highlights: best (min), average (mean), worst (max) latency
 * 2. Histogram: 3 buckets (<10ms, 10–30ms, >50ms) as labeled bars with count + percentage
 *
 * Validates: Requirements 10.10, 10.11
 */
function LatencyTimeline({ result }) {
  const latency_min = result?.latency_min;
  const latency_avg = result?.latency_avg;
  const latency_max = result?.latency_max;
  const histogram = result?.histogram;
  const completed = result?.completed || 0;
  const warmup_avg_ms = result?.warmup_avg_ms;
  const steady_state_avg_ms = result?.steady_state_avg_ms;
  const scaled_replicas = result?.scaled_replicas;
  const total_mutations = result?.total_mutations;

  const hasData = latency_min != null && latency_avg != null && latency_max != null;
  const total = completed || 0;

  // Histogram buckets from server data
  const buckets = [
    { label: "< 10ms", count: histogram?.lt_10ms ?? 0 },
    { label: "10–30ms", count: histogram?.["10_30ms"] ?? 0 },
    { label: "> 50ms", count: histogram?.gt_50ms ?? 0 },
  ];

  // Find max count for bar scaling
  const maxCount = Math.max(...buckets.map((b) => b.count), 1);

  return (
    <div className="loadtest-timeline" aria-label="Latency summary">
      <h4 className="loadtest-timeline-title">Latency Summary</h4>

      {/* Timeline highlights: best / avg / worst */}
      <div className="loadtest-timeline-highlights">
        <TimelineHighlight label="Best" value={latency_min} variant="best" />
        <TimelineHighlight label="Average" value={latency_avg} variant="avg" />
        <TimelineHighlight label="Worst" value={latency_max} variant="worst" />
      </div>

      {/* Visual timeline bar showing min/avg/max positions */}
      {hasData && (
        <div className="loadtest-timeline-bar" aria-hidden="true">
          <div className="loadtest-timeline-bar-track">
            <TimelineMarker value={latency_min} max={latency_max} variant="best" />
            <TimelineMarker value={latency_avg} max={latency_max} variant="avg" />
            <TimelineMarker value={latency_max} max={latency_max} variant="worst" />
          </div>
          <div className="loadtest-timeline-bar-labels">
            <span>{latency_min.toFixed(1)}ms</span>
            <span>{latency_max.toFixed(1)}ms</span>
          </div>
        </div>
      )}

      {/* Histogram */}
      <div className="loadtest-histogram" aria-label="Latency distribution histogram">
        <h5 className="loadtest-histogram-title">Distribution</h5>
        {buckets.map((bucket) => {
          const pct = total > 0 ? ((bucket.count / total) * 100).toFixed(1) : "0.0";
          const barWidth = hasData ? (bucket.count / maxCount) * 100 : 0;
          return (
            <div className="loadtest-histogram-row" key={bucket.label}>
              <span className="loadtest-histogram-label">{bucket.label}</span>
              <div className="loadtest-histogram-bar-track">
                <div
                  className="loadtest-histogram-bar-fill"
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <span className="loadtest-histogram-count">
                {hasData ? bucket.count.toLocaleString() : "—"}
              </span>
              <span className="loadtest-histogram-pct">{hasData ? `${pct}%` : ""}</span>
            </div>
          );
        })}
      </div>

      {/* Scaling & Performance Stats — always visible */}
      <div className="loadtest-extra-stats">
        <h4 className="loadtest-histogram-title">Performance Details</h4>
        <div className="loadtest-stats-grid">
          <div className="loadtest-stat-item">
            <span className="loadtest-stat-value">{warmup_avg_ms != null && warmup_avg_ms > 0 ? `${warmup_avg_ms.toFixed(1)}ms` : "—"}</span>
            <span className="loadtest-stat-label">Warm-up (first 10%)</span>
          </div>
          <div className="loadtest-stat-item">
            <span className="loadtest-stat-value">{steady_state_avg_ms != null && steady_state_avg_ms > 0 ? `${steady_state_avg_ms.toFixed(1)}ms` : "—"}</span>
            <span className="loadtest-stat-label">Steady-state (last 50%)</span>
          </div>
          <div className="loadtest-stat-item">
            <span className="loadtest-stat-value">{scaled_replicas != null && scaled_replicas > 1 ? `${scaled_replicas}×` : "—"}</span>
            <span className="loadtest-stat-label">Replicas scaled</span>
          </div>
          <div className="loadtest-stat-item">
            <span className="loadtest-stat-value">{total_mutations != null && total_mutations > 0 ? total_mutations.toLocaleString() : "—"}</span>
            <span className="loadtest-stat-label">Total mutations</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function TimelineHighlight({ label, value, variant }) {
  return (
    <div className={`loadtest-timeline-highlight loadtest-timeline-highlight--${variant}`}>
      <span className="loadtest-timeline-highlight-value">
        {value != null ? `${value.toFixed(1)}ms` : "—"}
      </span>
      <span className="loadtest-timeline-highlight-label">{label}</span>
    </div>
  );
}

function TimelineMarker({ value, max, variant }) {
  // Position as percentage along the track (0% = 0ms, 100% = max)
  const position = max > 0 ? (value / max) * 100 : 0;
  return (
    <div
      className={`loadtest-timeline-marker loadtest-timeline-marker--${variant}`}
      style={{ left: `${position}%` }}
      aria-hidden="true"
    />
  );
}
