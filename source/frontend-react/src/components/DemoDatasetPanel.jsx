import React, { useState, useEffect, useCallback, useRef } from "react";
import { authFetch } from "../authFetch.js";

// Demo training dataset for one trainable model type (FR1). The orchestrator
// generates a seeded synthetic dataset from the rates set here, checks it with
// the trainer's rules, and stores it in its own S3 prefix
// (orchestrator/demo_dataset_api.py). Training reads it only when "Include demo
// dataset" is on. Bounds, defaults and sizing come from the API, so this panel
// does not restate them.

const NOTICE =
  "Demonstration only. This dataset is synthetic: its wins, clicks, conversions " +
  "and deal labels are generated from the rates set here, not observed from real " +
  "auctions or advertisers. A model trained with it shows the training loop " +
  "working; its accuracy says nothing about real traffic.";

const toPercent = (fraction) => Number((fraction * 100).toFixed(4));
const formatCount = (n) => (typeof n === "number" ? Math.round(n).toLocaleString("en-US") : "\u2014");

function positiveLabel(kind) {
  return kind === "conversions" ? "conversions" : "labelled deals";
}

function paramsFromPercents(specs, percents) {
  const out = {};
  for (const spec of specs) {
    const value = Number(percents[spec.name]);
    if (!Number.isFinite(value)) return null;
    const fraction = value / 100;
    if (fraction < spec.minimum || fraction > spec.maximum) return null;
    out[spec.name] = fraction;
  }
  return out;
}

export function DemoDatasetStatusLine({ status }) {
  if (!status) return null;
  if (!status.exists || !status.metadata) {
    return (
      <div className="cl-demo-dataset-status" data-testid="demo-dataset-status">
        No demo dataset
      </div>
    );
  }
  const m = status.metadata;
  const sizing = m.sizing || {};
  const params = Object.entries(m.params || {})
    .map(([name, value]) => `${name} ${toPercent(value)}%`)
    .join(", ");
  return (
    <div className="cl-demo-dataset-status" data-testid="demo-dataset-status">
      Demo dataset generated {m.generated_at}: {formatCount(m.rows_written)} rows,{" "}
      {formatCount(m.actual_positives)} {positiveLabel(sizing.positive_kind)} (expected{" "}
      {formatCount(sizing.expected_positives)}). Parameters: {params || "\u2014"}. Seed {m.seed}.
    </div>
  );
}

export default function DemoDatasetPanel({ modelType, onStatusChange }) {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);
  const [percents, setPercents] = useState({});
  const [estimate, setEstimate] = useState(null);
  const [estimateError, setEstimateError] = useState(null);
  const [confirming, setConfirming] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const onStatusChangeRef = useRef(onStatusChange);
  onStatusChangeRef.current = onStatusChange;

  const specs = status?.params || [];

  const publish = useCallback((next) => {
    setStatus(next);
    if (onStatusChangeRef.current) onStatusChangeRef.current(next);
  }, []);

  const loadStatus = useCallback(async () => {
    setStatusError(null);
    try {
      const resp = await authFetch(`/api/v1/governance/demo-dataset?model_type=${encodeURIComponent(modelType)}`);
      const data = await resp.json().catch(() => ({}));
      if (resp.ok) {
        publish(data);
        const initial = {};
        for (const spec of data.params || []) initial[spec.name] = toPercent(spec.default);
        setPercents(initial);
      } else {
        publish(null);
        setStatusError(data.error || `HTTP ${resp.status}`);
      }
    } catch (e) {
      publish(null);
      setStatusError(String(e));
    }
  }, [modelType, publish]);

  useEffect(() => {
    setConfirming(false);
    setResult(null);
    setError(null);
    setEstimate(null);
    loadStatus();
  }, [loadStatus]);

  const params = specs.length ? paramsFromPercents(specs, percents) : null;
  const paramsKey = params ? JSON.stringify(params) : "";

  useEffect(() => {
    if (!paramsKey) { setEstimate(null); return undefined; }
    let cancelled = false;
    (async () => {
      setEstimateError(null);
      const query = new URLSearchParams({ model_type: modelType, ...JSON.parse(paramsKey) });
      try {
        const resp = await authFetch(`/api/v1/governance/demo-dataset/estimate?${query.toString()}`);
        const data = await resp.json().catch(() => ({}));
        if (cancelled) return;
        if (resp.ok) setEstimate(data.sizing || null);
        else { setEstimate(null); setEstimateError(data.error || `HTTP ${resp.status}`); }
      } catch (e) {
        if (!cancelled) { setEstimate(null); setEstimateError(String(e)); }
      }
    })();
    return () => { cancelled = true; };
  }, [modelType, paramsKey]);

  const generate = useCallback(async () => {
    if (!params) return;
    setGenerating(true);
    setError(null);
    setResult(null);
    try {
      const resp = await authFetch("/api/v1/governance/demo-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: modelType, params, confirmed: true }),
      });
      const data = await resp.json().catch(() => ({}));
      if (resp.ok) {
        setResult(data.metadata || null);
        publish({ ...(status || {}), ...data, exists: true });
      } else {
        setError(data.error || `HTTP ${resp.status}`);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setGenerating(false);
      setConfirming(false);
    }
  }, [modelType, params, status, publish]);

  return (
    <div className="cl-demo-dataset" data-testid="demo-dataset-panel">
      <div className="cl-demo-dataset-title">Demo dataset · {modelType}</div>
      <div className="cl-honest cl-honest-block" data-testid="demo-dataset-notice">{NOTICE}</div>

      {statusError && (
        <div className="cl-honest cl-honest-block" data-testid="demo-dataset-status-error">
          Demo dataset status unavailable: {statusError}
        </div>
      )}
      <DemoDatasetStatusLine status={status} />

      {specs.length > 0 && (
        <div className="cl-demo-dataset-params">
          {specs.map((spec) => (
            <label key={spec.name} className="cl-demo-dataset-param">
              {spec.label} (%)
              <input
                type="number"
                className="sg-interactive"
                data-testid={`demo-dataset-param-${spec.name.replace(/_/g, "-")}`}
                min={toPercent(spec.minimum)}
                max={toPercent(spec.maximum)}
                step="any"
                value={percents[spec.name] ?? ""}
                onChange={(e) => setPercents((cur) => ({ ...cur, [spec.name]: e.target.value }))}
                disabled={generating}
              />
              <span className="cl-demo-dataset-bounds">
                {toPercent(spec.minimum)}% to {toPercent(spec.maximum)}%
              </span>
            </label>
          ))}
        </div>
      )}

      {specs.length > 0 && !params && (
        <div className="cl-honest cl-honest-block" data-testid="demo-dataset-params-invalid">
          Each rate must be a number inside the range shown under it.
        </div>
      )}
      {estimateError && (
        <div className="cl-honest cl-honest-block">Estimate unavailable: {estimateError}</div>
      )}
      {estimate && (
        <div className="cl-demo-dataset-estimate" data-testid="demo-dataset-estimate">
          {estimate.positive_kind === "conversions"
            ? `${formatCount(estimate.rows)} rows: expected ${formatCount(estimate.expected_clicks)} clicks and ${formatCount(estimate.expected_positives)} conversions.`
            : `${formatCount(estimate.rows)} deals offered: expected ${formatCount(estimate.expected_positives)} labelled deals.`}
          {estimate.capped
            ? ` Capped at ${formatCount(estimate.row_cap)} rows, so fewer than the target of ${formatCount(estimate.target_positives)} are expected.`
            : ""}
        </div>
      )}

      {result && (
        <div className="cl-train-result" data-testid="demo-dataset-result">
          Demo dataset written: {formatCount(result.rows_written)} rows,{" "}
          {formatCount(result.actual_positives)} {positiveLabel(result.sizing?.positive_kind)}. Synthetic
          demonstration data, stored under {status?.prefix}.
        </div>
      )}
      {error && (
        <div className="cl-honest cl-honest-block" data-testid="demo-dataset-error">{error}</div>
      )}

      {!confirming ? (
        <button
          className="btn-secondary sg-interactive"
          data-testid="demo-dataset-generate-button"
          onClick={() => setConfirming(true)}
          disabled={!params || generating}
        >
          Generate demo dataset
        </button>
      ) : (
        <div className={`cl-demo-dataset-actions${generating ? " state-processing" : ""}`}>
          <span>
            Generate synthetic demonstration data for {modelType}? It replaces the current demo dataset.
          </span>
          <button
            className="btn btn-primary sg-interactive"
            data-testid="demo-dataset-confirm-button"
            onClick={generate}
            disabled={generating}
          >
            {generating ? "Generating\u2026" : "Confirm"}
          </button>
          <button
            className="btn-secondary sg-interactive"
            data-testid="demo-dataset-cancel-button"
            onClick={() => setConfirming(false)}
            disabled={generating}
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
