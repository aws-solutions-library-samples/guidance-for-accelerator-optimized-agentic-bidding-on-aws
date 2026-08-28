import React, { useState, useEffect, useRef } from "react";
import { authFetch } from "../authFetch.js";

// ── Shared helpers, icons, and small components used by both the Adaptive
// Bidding and Governance closed-loop panels. Split out of the former
// ClosedLoopPanel.jsx so the two loops can live on separate tabs without
// duplicating ~300 lines of icons/markdown/table helpers. ───────────────────

export const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };
export const MONO = { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" };

export function fmt(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

// ── Minimal, safe markdown for agent rationale (LLM output uses **bold**,
// `code`, *italic*, line breaks, and simple -/1. lists). No HTML injection:
// everything is built as React nodes, never dangerouslySetInnerHTML. ─────────
function mdInline(text, kp = "") {
  const nodes = [];
  const re = /(\*\*([^*]+)\*\*|`([^`]+)`|\*([^*\n]+)\*)/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    if (m[2] !== undefined) nodes.push(<strong key={`${kp}b${i}`}>{m[2]}</strong>);
    else if (m[3] !== undefined) nodes.push(<code key={`${kp}c${i}`}>{m[3]}</code>);
    else if (m[4] !== undefined) nodes.push(<em key={`${kp}i${i}`}>{m[4]}</em>);
    last = m.index + m[0].length;
    i += 1;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

export function MarkdownLite({ text }) {
  if (!text) return null;
  const lines = String(text).replace(/\r/g, "").split("\n");
  const blocks = [];
  let list = null; // { type: "ul" | "ol", items: [] }
  const flush = () => {
    if (list) { blocks.push(list); list = null; }
  };
  lines.forEach((raw) => {
    const line = raw.replace(/\s+$/, "");
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    const num = /^\s*\d+\.\s+(.*)$/.exec(line);
    if (bullet) {
      if (!list || list.type !== "ul") { flush(); list = { type: "ul", items: [] }; }
      list.items.push(bullet[1]);
    } else if (num) {
      if (!list || list.type !== "ol") { flush(); list = { type: "ol", items: [] }; }
      list.items.push(num[1]);
    } else if (line.trim() === "") {
      flush();
    } else {
      flush();
      blocks.push({ type: "p", text: line });
    }
  });
  flush();
  return (
    <>
      {blocks.map((b, i) => {
        if (b.type === "p") return <p key={i}>{mdInline(b.text, `${i}-`)}</p>;
        const items = b.items.map((it, j) => <li key={j}>{mdInline(it, `${i}-${j}-`)}</li>);
        return b.type === "ul"
          ? <ul key={i}>{items}</ul>
          : <ol key={i}>{items}</ol>;
      })}
    </>
  );
}

// ── Icons (architecture components) ───────────────────────────────────────
export const IconCloudWatch = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
);
export const IconAgent = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="8" width="16" height="12" rx="2" /><path d="M12 8V4" /><circle cx="12" cy="3" r="1" /><path d="M9 14h.01M15 14h.01" /></svg>
);
export const IconBrain = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5a3 3 0 0 0-5.9-.6A3 3 0 0 0 4 9.5a3 3 0 0 0 1 5.8V17a3 3 0 0 0 6 0V5Z" /><path d="M12 5a3 3 0 0 1 5.9-.6A3 3 0 0 1 20 9.5a3 3 0 0 1-1 5.8V17a3 3 0 0 1-6 0" /></svg>
);
export const IconDynamo = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>
);
export const IconRegistry = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
);
export const IconChip = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2" /></svg>
);
export const IconSplit = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 3v12" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="6" r="3" /><path d="M18 9a9 9 0 0 1-9 9" /></svg>
);
export const IconGavel = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m14 13-7.5 7.5a2.12 2.12 0 0 1-3-3L11 10" /><path d="m16 16 6-6" /><path d="m8 8 6-6" /><path d="m9 7 8 8" /><path d="m21 11-8-8" /></svg>
);
export const IconEtl = (
  // Funnel — the Glue ETL sweep that filters/labels raw outcomes into training data.
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 4h18l-7 8v7l-4 2v-9z" /></svg>
);
export const IconAudit = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><path d="m9 15 2 2 4-4" /></svg>
);

// Derives the delta from old_value/new_value rather than trusting a separate
// `delta` field on the update record: an agent response that populates
// old_value/new_value correctly but omits/zeros the delta field would
// otherwise show a contradictory "oldValue → newValue → no change" label.
export function DeltaArrow({ delta, oldValue, newValue }) {
  const effectiveDelta =
    delta != null && !Number.isNaN(Number(delta))
      ? Number(delta)
      : (newValue != null && oldValue != null && !Number.isNaN(Number(newValue)) && !Number.isNaN(Number(oldValue)))
        ? Number(newValue) - Number(oldValue)
        : null;

  if (effectiveDelta == null || Math.abs(effectiveDelta) < 1e-9) {
    return <span style={{ color: "var(--text-muted)" }}>→ no change</span>;
  }
  const up = effectiveDelta > 0;
  return (
    <span style={{ color: up ? "#16a34a" : "#dc2626", fontWeight: 600 }}>
      {up ? "▲ +" : "▼ "}{fmt(effectiveDelta, 4)}
    </span>
  );
}

export function RecommendationBadge({ rec }) {
  const map = {
    promote: { bg: "#16a34a", label: "PROMOTE" },
    reject: { bg: "#dc2626", label: "REJECT" },
    extend: { bg: "#d97706", label: "EXTEND · keep testing" },
  };
  const s = map[rec] || { bg: "var(--text-muted)", label: String(rec || "—").toUpperCase() };
  return <span className="cl-rec-badge" style={{ background: s.bg }}>{s.label}</span>;
}

// Recognises the reasons the governance agent writes for a step that FAILED,
// as opposed to a model that was evaluated and lost. The distinction matters to
// a reader: "the pipeline broke" and "the challenger underperformed" are very
// different conclusions, and a bare "Rejected" badge conflates them.
//
// Matched against the ApprovalDescription text the agent produces (see
// agents/governance/governance_agent.py, which prefixes step failures with
// "Model optimization failed", "Canary deployment failed", "A/B test error", or
// "Guardrail breach"). Anything unmatched is treated as an evaluation verdict
// rather than asserting a category we cannot confirm.
const PIPELINE_FAILURE_PREFIXES = [
  "model optimization failed",
  "canary deployment failed",
  "canary load",
  "a/b test error",
  "guardrail breach",
];

export function isPipelineFailure(reason) {
  if (!reason) return false;
  const text = String(reason).toLowerCase();
  return PIPELINE_FAILURE_PREFIXES.some((p) => text.startsWith(p));
}

// First sentence / clause of the reason, for the inline cell. The full text goes
// in the title attribute so nothing is lost — some reasons carry an entire HTTP
// error body and would otherwise wreck the table layout.
export function shortReason(reason) {
  const text = String(reason).trim();
  const cut = text.indexOf(": ");
  const head = cut > 0 ? text.slice(0, cut) : text;
  return head.length > 60 ? `${head.slice(0, 57)}...` : head;
}

export function ApprovalReason({ status, reason }) {
  if (!reason) {
    // Pending versions have not been decided, so "no reason" is expected and
    // needs no explanation. A decided version with no recorded reason is worth
    // saying plainly rather than leaving the cell blank.
    if (status === "PendingManualApproval" || !status) return <span style={SUBTLE}>—</span>;
    return <span style={SUBTLE}>No reason recorded</span>;
  }

  const pipelineFailure = isPipelineFailure(reason);
  return (
    <span
      className="cl-approval-reason"
      title={reason}
      data-testid="approval-reason"
    >
      {pipelineFailure && (
        <span className="cl-approval-reason-tag" title={reason}>
          pipeline failure
        </span>
      )}
      <span style={SUBTLE}>{shortReason(reason)}</span>
    </span>
  );
}

export function ApprovalBadge({ status }) {
  const map = { Approved: "#16a34a", Rejected: "#dc2626", PendingManualApproval: "#d97706" };
  const bg = map[status] || "var(--text-muted)";
  return <span className="cl-approval-badge" style={{ background: bg }}>{status || "—"}</span>;
}

// ── Scenario card (reframed "Expected" → "What to watch for") ──────────────
export function ScenarioCardCL({ s, selected, onSelect, onViewSamples }) {
  return (
    <div className={`cl-scenario-card sg-elevated${selected ? " selected" : ""}`}>
      <button className="cl-scenario-card-btn sg-interactive" onClick={onSelect} type="button">
        <div className="cl-scenario-label">{s.label}</div>
        <div className="cl-scenario-desc">{s.description}</div>
        <div className="cl-scenario-hint">
          <span className="cl-hint-tag">What to watch for</span> {s.expected_decision}
        </div>
      </button>
      <button
        className="cl-samples-link sg-interactive"
        type="button"
        onClick={(e) => { e.stopPropagation(); onViewSamples(s); }}
        data-testid={`scenario-${s.key}-view-samples-button`}
      >
        View sample outcomes
      </button>
    </div>
  );
}

// ── Scenario detail "peek" card ─────────────────────────────────────────────
// Shows details for ONE already-selected scenario (selection happens via a
// dropdown elsewhere) rather than a grid of every scenario at once. Same
// visual language as ScenarioCardCL, minus the whole-card click-to-select
// affordance (only the "View sample outcomes" button is interactive).
export function ScenarioDetailCard({ s, onViewSamples }) {
  if (!s) return null;
  return (
    <div className="cl-scenario-card cl-scenario-peek sg-elevated">
      <div className="cl-scenario-card-body">
        <div className="cl-scenario-label">{s.label}</div>
        <div className="cl-scenario-desc">{s.description}</div>
        <div className="cl-scenario-hint">
          <span className="cl-hint-tag">What to watch for</span> {s.expected_decision}
        </div>
      </div>
      <button
        className="cl-samples-link sg-interactive"
        type="button"
        onClick={() => onViewSamples(s)}
        data-testid={`scenario-${s.key}-view-samples-button`}
      >
        View sample outcomes
      </button>
    </div>
  );
}

// ── Sample outcomes modal ───────────────────────────────────────────────────
// Shows a subset of the individual synthetic records behind a scenario: for
// agentic scenarios, illustrative bid-outcome records (won/price_paid/
// impression/click/conversion) consistent with the scenario's aggregate
// metrics; for governance scenarios, the first N control/treatment values
// actually fed to the real ABEvaluator. Data is fetched from the orchestrator
// on open — nothing here is fabricated client-side.
export function SampleOutcomesModal({ scenario, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    (async () => {
      try {
        const resp = await authFetch(
          `/api/v1/closed-loop/sample-outcomes?scenario=${encodeURIComponent(scenario.key)}&n=10`
        );
        const raw = await resp.text();
        let body;
        try {
          body = JSON.parse(raw);
        } catch (_) {
          // Non-JSON response (e.g. a plain-text 404 from a backend that hasn't
          // been redeployed with this route yet) — report honestly instead of
          // throwing a raw JSON.parse SyntaxError at the user.
          if (!cancelled) {
            setError(
              resp.ok
                ? `Unexpected non-JSON response: ${raw.slice(0, 120)}`
                : `HTTP ${resp.status}: ${raw.slice(0, 120) || "no body"}`
            );
          }
          return;
        }
        if (cancelled) return;
        if (resp.ok) setData(body);
        else setError(body.error || `HTTP ${resp.status}`);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [scenario.key]);

  return (
    <div className="cl-modal-overlay" onClick={onClose} data-testid="sample-outcomes-modal-overlay">
      <div className="cl-modal sg-elevated" onClick={(e) => e.stopPropagation()}>
        <div className="cl-modal-head">
          <div>
            <div className="cl-modal-title">Sample outcomes — {scenario.label}</div>
            <div style={SUBTLE}>{scenario.key} · {scenario.loop}</div>
          </div>
          <button className="cl-modal-close sg-interactive" onClick={onClose} data-testid="sample-outcomes-modal-close-button">✕</button>
        </div>

        {loading && <div style={SUBTLE}>Loading…</div>}
        {error && <div className="cl-honest">Unavailable: {error}</div>}

        {!loading && !error && data?.kind === "bid_outcome_records" && (
          <>
            <div className="cl-modal-note">{data.note}</div>
            <table className="cl-table">
              <thead>
                <tr>
                  <th>#</th><th>Won</th><th>Shaded price</th><th>Price paid</th>
                  <th>Impression</th><th>Click</th><th>Conversion</th><th>Conv. value</th>
                </tr>
              </thead>
              <tbody>
                {(data.records || []).map((r) => (
                  <tr key={r.index}>
                    <td>{r.index}</td>
                    <td>{r.won ? "yes" : "no"}</td>
                    <td>{fmt(r.shaded_price, 2)}</td>
                    <td>{r.price_paid != null ? fmt(r.price_paid, 2) : "—"}</td>
                    <td>{r.impression ? "yes" : "no"}</td>
                    <td>{r.click ? "yes" : "no"}</td>
                    <td>{r.conversion ? "yes" : "no"}</td>
                    <td>{r.conversion_value != null ? fmt(r.conversion_value, 2) : "—"}</td>
                  </tr>
                ))}
                {(data.records || []).length === 0 && (
                  <tr><td colSpan={8} style={SUBTLE}>No sample records for this scenario.</td></tr>
                )}
              </tbody>
            </table>
          </>
        )}

        {!loading && !error && data?.kind === "ab_samples" && (
          <>
            <div className="cl-modal-note">{data.note}</div>
            <div style={SUBTLE}>
              primary metric <b>{data.primary_metric}</b> · showing {data.n_shown} of {data.n_per_group} per group
            </div>
            <table className="cl-table">
              <thead><tr><th>#</th><th>Control</th><th>Treatment</th></tr></thead>
              <tbody>
                {(data.control || []).map((v, i) => (
                  <tr key={i}>
                    <td>{i}</td>
                    <td>{fmt(v, 4)}</td>
                    <td>{data.treatment?.[i] != null ? fmt(data.treatment[i], 4) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {!loading && !error && data && data.kind === "none" && (
          <div style={SUBTLE}>{data.note}</div>
        )}
      </div>
    </div>
  );
}

// ── Architecture flow (circle nodes, progressive reveal) ───────────────────
export function FlowTrack({ nodes, revealed, running }) {
  return (
    <div className="cl-flow sg-elevated">
      <div className="cl-flow-track">
        {nodes.map((node, i) => {
          const isRevealed = i < revealed;
          const processing = node.state === "processing" && running;
          return (
            <React.Fragment key={node.id}>
              {i > 0 && <div className={`cl-flow-connector${i < revealed ? " active" : ""}`} />}
              <div
                className={`cl-flow-node cl-flow-node--${node.state}${isRevealed ? " revealed" : ""}${processing ? " state-processing" : ""}`}
                title={node.service}
              >
                <div className="cl-flow-icon">{node.icon}</div>
                <div className="cl-flow-label">{node.label}</div>
                <div className="cl-flow-service">{node.service}</div>
              </div>
            </React.Fragment>
          );
        })}
      </div>

      <div className="cl-flow-details">
        {nodes.map((node, i) => (
          i < revealed && node.detail ? (
            <div key={node.id} className={`cl-flow-detail cl-flow-detail--${node.state}`}>
              <div className="cl-flow-detail-head">{node.label} <span>· {node.service}</span></div>
              {node.detail}
            </div>
          ) : null
        ))}
      </div>
    </div>
  );
}

// ── Compact pipeline bar (icons-only, top-of-page) ──────────────────────────
// A slim horizontal row of stage icons, matching reference_prototype.html's
// `.pipeline-bar` — small circular icons + thin connectors in a single
// bordered strip, rather than the wider FlowTrack used by AdaptiveBiddingPanel.
// Intended to sit at the TOP of the page (icon steps first, like the prototype)
// with a separate, simple step-by-step text log below (see StepByStepLog).
//
// Visual state per stage is derived from BOTH the progressive reveal position
// (revealed/running — paces disclosure of already-computed real data) and the
// node's real data state (ok/error/event/...): not-yet-revealed stages show
// "done" (grey, matching the prototype's default), the currently-revealing
// stage pulses "running", and once revealed a stage settles into "active"
// (green, real success) or "error" (red, real failure) — never fabricated.
function _pipelineStageClass(node, i, revealed, running) {
  if (i >= revealed) return "done";
  if (node.state === "processing" || (i === revealed - 1 && running)) return "running";
  if (node.state === "error") return "error";
  if (node.state === "ok") return "active";
  return "done";
}

export function PipelineBar({ nodes, revealed, running, ariaLabel = "Governance pipeline stages" }) {
  return (
    <div className="cl-pipeline-bar sg-elevated" role="list" aria-label={ariaLabel}>
      {nodes.map((node, i) => {
        const cls = _pipelineStageClass(node, i, revealed, running);
        const isRevealed = i < revealed;
        return (
          <React.Fragment key={node.id}>
            {i > 0 && <div className={`cl-pi-conn${i < revealed ? " active" : ""}`} aria-hidden="true" />}
            <div
              className={`cl-pi-stage cl-pi-stage--${cls}${isRevealed ? " revealed" : ""}`}
              role="listitem"
              aria-label={`${node.label} (${node.service}): ${cls}`}
            >
              <div className="cl-pi-icon">{node.icon}</div>
              <div className="cl-pi-name">{node.label}</div>
              <div className="cl-pi-service">{node.service}</div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

// ── Step-by-step log (simple text, per DESIGN_BRIEF.md section 5) ──────────
// A single card with one plain-text line per revealed stage — replaces the
// bulkier per-stage FlowTrack detail cards for the Governance panel. Each
// line is derived from the same real data already computed elsewhere (A/B
// decision, agent rationale); nothing here is scenario-scripted narration.
export function StepByStepLog({ nodes, revealed }) {
  const visible = nodes.slice(0, revealed).filter((n) => n.logText);
  return (
    <div className="cl-card cl-steplog-card sg-elevated" data-testid="step-by-step-log">
      <div className="cl-card-head">
        <span className="cl-card-title"><span className="cl-dot cl-dot--ink" /> Step-by-step governance flow</span>
      </div>
      <div className="cl-steplog" aria-live="polite">
        {visible.length === 0 ? (
          <div style={SUBTLE}>Select a scenario and click "Run governance scenario" to walk through the pipeline step by step.</div>
        ) : (
          visible.map((n) => (
            <div key={n.id} className="cl-steplog-line">
              <strong>{n.label}:</strong> {n.logText}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ── Persisted-state views (real DynamoDB / SageMaker reads) ────────────────
export function ParametersView({ params, error }) {
  return (
    <div className="cl-block sg-elevated">
      <div className="cl-block-title">Current parameters · DynamoDB</div>
      {error && <div className="cl-honest">Unavailable: {error}</div>}
      {!error && params && params.length === 0 && (
        <div style={SUBTLE}>No parameters initialized yet. Run an Adaptive Bidding scenario to create them.</div>
      )}
      {!error && params && params.length > 0 && (
        <table className="cl-table">
          <thead><tr><th>Parameter</th><th>Value</th><th>v</th><th>Updated by</th><th>Reason</th></tr></thead>
          <tbody>
            {params.map((p, i) => (
              <tr key={i}>
                <td style={MONO}>{p.parameter_name}</td>
                <td>{fmt(p.current_value)}</td>
                <td>{p.version}</td>
                <td style={SUBTLE}>{p.updated_by}</td>
                <td style={SUBTLE}>{p.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function AuditView({ records, error }) {
  return (
    <div className="cl-block sg-elevated">
      <div className="cl-block-title">Audit trail · newest first</div>
      {error && <div className="cl-honest">Unavailable: {error}</div>}
      {!error && records && records.length === 0 && <div style={SUBTLE}>No audit records yet.</div>}
      {!error && records && records.length > 0 && (
        <table className="cl-table">
          <thead><tr><th>When</th><th>Parameter</th><th>Old → New</th><th>By</th><th>Reason</th></tr></thead>
          <tbody>
            {records.map((r, i) => (
              <tr key={i}>
                <td style={SUBTLE}>{r.timestamp ? new Date(r.timestamp * 1000).toLocaleTimeString() : "—"}</td>
                <td style={MONO}>{r.parameter_name}</td>
                <td>{fmt(r.old_value)} → {fmt(r.new_value)}</td>
                <td style={SUBTLE}>{r.updated_by}</td>
                <td style={SUBTLE}>{r.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── ARTF mutation framing (Model Governance design brief) ──────────────────
// Static facts about what each container mutates on the bidstream and how it
// actually works, verified against the real implementations (container apps,
// Triton config.pbtxt) rather than copied from the design spec's prototype
// figures, which use illustrative numbers that don't all match this repo's
// real thresholds/serving config.
export const ARTF_MUTATION_INFO = {
  dlrm_bid_shader: {
    label: "Bid Price Optimization",
    intents: ["BID_SHADE"],
    purpose:
      "Predicts click-through rate (CTR) and computes an optimal shaded bid price. When a challenger model is promoted, all BID_SHADE mutations are computed with the new CTR model.",
    metricLabel: "revenue_per_bid",
    technical: {
      architecture:
        "DLRM with 3 EmbeddingBag tables (vocab=1000, dim=16), bottom MLP (4\u219232\u219216, ReLU), pairwise dot-product feature interaction, top MLP (64\u219232\u21921, sigmoid).",
      serving:
        "Triton tensorrt_plan backend (source/triton/model_repository/dlrm_bid_shader_stable), dynamic batching preferred sizes [8, 16, 32], 500\u03bcs max queue delay, 2 GPU instances.",
      decisionLogic:
        "shaded_price = min(bid, CTR \u00d7 conversion_value \u00d7 shade_factor), floored at bidfloor. conversion_value/shade_factor are tunable parameters the Adaptive Bidding Agent writes to DynamoDB.",
      canary:
        "The dlrm_bid_shader Triton router forwards each request to _stable or _canary based on an in-memory traffic-percentage parameter (control-plane only \u2014 no ARTF container change).",
      training:
        "SageMaker NeMo-RL retraining; reward = ROI (revenue \u2212 cost), computed per bid outcome (source/training/reward.py).",
    },
  },
  ncf_deal_manager: {
    label: "Deal Activation",
    intents: ["ACTIVATE_DEALS", "SUPPRESS_DEALS"],
    purpose:
      "Predicts user-deal relevance via collaborative filtering and activates or suppresses private marketplace (PMP) deals. When a challenger model is promoted, all deal-relevance mutations use the new model.",
    metricLabel: "deal_hit_rate",
    technical: {
      architecture:
        "Neural Collaborative Filtering (NeuMF): GMF + MLP branches (dim=64 embeddings) fused to a final sigmoid, predicting P(user engages with deal).",
      serving:
        "Triton tensorrt_plan backend (source/triton/model_repository/ncf_deal_manager_stable), dynamic batching preferred sizes [16, 32, 64], 500\u03bcs max queue delay, 2 GPU instances.",
      decisionLogic:
        "relevance \u2265 0.499 \u2192 emit ACTIVATE_DEALS; relevance < 0.497 \u2192 emit SUPPRESS_DEALS (source/containers/ncf_deal_manager/app.py thresholds).",
      canary:
        "The ncf_deal_manager Triton router forwards each request to _stable or _canary based on an in-memory traffic-percentage parameter.",
      training:
        "SageMaker NeMo-RL retraining; reward = deal_relevance \u00d7 conversion_probability.",
      businessMetric:
        "Per-variant revenue attribution requires the ADOT/CloudWatch integration (ARTF/ABTest Variant dimension). Serving guardrails (latency, error rate) come from Triton metrics directly.",
    },
  },
};

// ── Technical detail popover ────────────────────────────────────────────────
// Hover (desktop) / focus+click (keyboard, touch) detail overlay, per
// COMPONENT_SPEC.md. Dismiss on blur/mouse-leave/click-outside/Escape.
export function TechnicalPopover({ technical }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e) {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    }
    function onKeyDown(e) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div
      ref={rootRef}
      className="cl-has-detail"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="cl-detail-trigger"
        onFocus={() => setOpen(true)}
        onClick={() => setOpen((v) => !v)}
        data-testid="technical-popover-trigger"
      >
        Technical detail &#8599;
      </button>
      {open && (
        <div role="tooltip" className="cl-detail-pop" data-testid="technical-popover-content">
          <div><strong>Architecture:</strong> {technical.architecture}</div>
          <div><strong>Serving:</strong> {technical.serving}</div>
          <div><strong>Decision logic:</strong> <code>{technical.decisionLogic}</code></div>
          <div><strong>Canary:</strong> {technical.canary}</div>
          <div><strong>Training:</strong> {technical.training}</div>
          {technical.businessMetric && <div><strong>Business metric:</strong> {technical.businessMetric}</div>}
        </div>
      )}
    </div>
  );
}

// ── Mutation intent card ────────────────────────────────────────────────────
// Always shows both models (2-column, per DESIGN_BRIEF.md), with the current
// mutation outcome populated for whichever model the most recent real
// decision applies to; the other card shows an honest "run a scenario for
// this model" placeholder rather than inventing a number for it.
export function MutationIntentCard({ modelKey, decision, agentModelType }) {
  const info = ARTF_MUTATION_INFO[modelKey];
  if (!info) return null;
  const hasOutcome = decision && agentModelType === modelKey;

  return (
    <div className="cl-intent-card sg-elevated" data-testid={`mutation-intent-card-${modelKey}`}>
      <div className="cl-intent-head">
        <div className={`cl-intent-icon cl-intent-icon--${modelKey}`} aria-hidden="true">
          {modelKey === "dlrm_bid_shader" ? IconChip : IconSplit}
        </div>
        <div>
          <span className="cl-intent-name">{info.label}</span>
          {info.intents.map((i) => <span key={i} className="cl-intent-artf">{i}</span>)}
        </div>
      </div>
      <div className="cl-intent-desc">{info.purpose}</div>
      <div className="cl-intent-outcome">
        <div className="cl-outcome-label">Current mutation output</div>
        {hasOutcome ? (
          <>
            <div className="cl-outcome-val">
              {info.metricLabel}: control <b>{fmt(decision.control_metric)}</b> &#8594; treatment{" "}
              <b>{fmt(decision.treatment_metric)}</b>{" "}
              <RecommendationBadge rec={decision.recommendation} />
            </div>
            <div className="cl-outcome-impact">
              lift {fmt(decision.relative_lift)} &middot; p {fmt(decision.p_value, 5)} &middot; samples{" "}
              {decision.samples_control}/{decision.samples_treatment}
            </div>
          </>
        ) : (
          <div style={SUBTLE}>Run a scenario for this model to see its mutation output.</div>
        )}
      </div>
      <TechnicalPopover technical={info.technical} />
    </div>
  );
}

// Generic, mechanism-level explanation of what each real recommendation does
// to production bidstream mutations for the selected model's ARTF intent(s).
// This is a fixed description of the canary-router mechanism (not scenario-
// dependent numbers) — it explains the real decision returned by the
// ABEvaluator, it does not invent one.
export function BidstreamImpactCard({ modelType }) {
  const info = ARTF_MUTATION_INFO[modelType];
  const intentText = info ? info.intents.join(" / ") : "this model's";
  return (
    <div className="cl-card sg-elevated">
      <div className="cl-card-head">
        <span className="cl-card-title"><span className="cl-dot cl-dot--green" /> What changes in the bidstream</span>
      </div>
      <div className="cl-bidstream-impact">
        <p>
          <strong>Promote:</strong> the canary router shifts 100% of {intentText} mutations to
          the challenger version. All production traffic is now mutated by the new model.
        </p>
        <p>
          <strong>Reject:</strong> the canary router reverts to the stable version. The
          challenger's {intentText} outputs are discarded; no bidstream behavior changes.
        </p>
        <p>
          <strong>Extend:</strong> no change. The canary stays at its current traffic split
          while the evaluation window continues.
        </p>
      </div>
    </div>
  );
}

// ── Governance verdict card ─────────────────────────────────────────────────
// Persistent (not transient inside the flow track) badge + reasoning card,
// matching DESIGN_BRIEF.md section 4. All content is the real decision +
// real agent rationale already computed elsewhere in GovernancePanel.
export function GovernanceVerdictCard({ decision, rationale, unavailableReason }) {
  return (
    <div className="cl-card cl-verdict-card sg-elevated" data-testid="governance-verdict-card">
      <div className="cl-card-head">
        <span className="cl-card-title"><span className="cl-dot cl-dot--purple" /> Governance Agent Decision</span>
        <span className="cl-badge cl-badge--agentcore">AgentCore</span>
      </div>
      {!decision ? (
        <div style={SUBTLE}>Run a scenario to see the governance agent's decision.</div>
      ) : (
        <>
          <RecommendationBadge rec={decision.recommendation} />
          <div className="cl-verdict-text">
            {rationale
              ? <MarkdownLite text={rationale} />
              : unavailableReason
                ? <span className="cl-honest">{unavailableReason}</span>
                : (
                  <>
                    Welch's t-test: p={fmt(decision.p_value, 4)}. Lift {fmt(decision.relative_lift)}.{" "}
                    {decision.guardrail_violations?.length > 0
                      ? `Guardrail violation: ${decision.guardrail_violations.join("; ")}.`
                      : "Serving guardrails within bounds."}
                  </>
                )}
          </div>
        </>
      )}
    </div>
  );
}

// ── Session run history ─────────────────────────────────────────────────────
// A client-local list of real decisions computed during this browser session
// (never a write into the real DynamoDB GovernanceAudit table, and never
// shown as if it were that persisted table). Each entry is a real completed
// run's actual recommendation/rationale, not fabricated data.
export function SessionAuditTrail({ entries }) {
  return (
    <div className="cl-audit-list" data-testid="session-audit-trail">
      {entries.length === 0 ? (
        <span className="cl-audit-empty">Run a scenario to see this session's decision history.</span>
      ) : (
        entries.map((e, i) => (
          <div key={i} className="cl-audit-entry">
            [{e.timestamp}] {e.model}: {e.decision}. {e.rationale}
          </div>
        ))
      )}
    </div>
  );
}

export function ModelsView({ versions, error }) {
  return (
    <div className="cl-block sg-elevated">
      <div className="cl-block-title">Model registry versions · SageMaker</div>
      {error && <div className="cl-honest">Unavailable: {error}</div>}
      {!error && versions && versions.length === 0 && <div style={SUBTLE}>No registered model versions yet.</div>}
      {!error && versions && versions.length > 0 && (
        <table className="cl-table">
          <thead><tr><th>Version</th><th>Approval</th><th>Reason</th><th>Status</th><th>Created</th></tr></thead>
          <tbody>
            {versions.map((v, i) => (
              <tr key={i}>
                <td>{v.version}</td>
                <td><ApprovalBadge status={v.approval_status} /></td>
                <td><ApprovalReason status={v.approval_status} reason={v.approval_description} /></td>
                <td style={SUBTLE}>{v.status}</td>
                <td style={SUBTLE}>{v.created_at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
