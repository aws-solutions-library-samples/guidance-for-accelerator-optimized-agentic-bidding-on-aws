import React, { useState, useEffect, useCallback, useRef } from "react";
import { authFetch } from "../authFetch.js";
import {
  invokeAdaptive,
  invokeGovernance,
  isAdaptiveConfigured,
  isGovernanceConfigured,
  agentUnavailableReason,
} from "../agentCoreClient.js";

const MODEL_TYPES = [
  { key: "dlrm_bid_shader", label: "DLRM Bid Shader" },
  { key: "ncf_deal_manager", label: "NCF Deal Manager" },
  { key: "widedeep_segment_activator", label: "Wide & Deep Segment Activator" },
];

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };
const MONO = { fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" };

function fmt(n, digits = 4) {
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

function MarkdownLite({ text }) {
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
const IconCloudWatch = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
);
const IconAgent = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="8" width="16" height="12" rx="2" /><path d="M12 8V4" /><circle cx="12" cy="3" r="1" /><path d="M9 14h.01M15 14h.01" /></svg>
);
const IconBrain = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5a3 3 0 0 0-5.9-.6A3 3 0 0 0 4 9.5a3 3 0 0 0 1 5.8V17a3 3 0 0 0 6 0V5Z" /><path d="M12 5a3 3 0 0 1 5.9-.6A3 3 0 0 1 20 9.5a3 3 0 0 1-1 5.8V17a3 3 0 0 1-6 0" /></svg>
);
const IconDynamo = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>
);
const IconRegistry = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
);
const IconChip = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2" /></svg>
);
const IconSplit = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 3v12" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="6" r="3" /><path d="M18 9a9 9 0 0 1-9 9" /></svg>
);
const IconGavel = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m14 13-7.5 7.5a2.12 2.12 0 0 1-3-3L11 10" /><path d="m16 16 6-6" /><path d="m8 8 6-6" /><path d="m9 7 8 8" /><path d="m21 11-8-8" /></svg>
);
const IconAudit = (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><path d="m9 15 2 2 4-4" /></svg>
);

// Derives the delta from old_value/new_value rather than trusting a separate
// `delta` field on the update record: an agent response that populates
// old_value/new_value correctly but omits/zeros the delta field would
// otherwise show a contradictory "oldValue → newValue → no change" label.
function DeltaArrow({ delta, oldValue, newValue }) {
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

function RecommendationBadge({ rec }) {
  const map = {
    promote: { bg: "#16a34a", label: "PROMOTE" },
    reject: { bg: "#dc2626", label: "REJECT" },
    extend: { bg: "#d97706", label: "EXTEND · keep testing" },
  };
  const s = map[rec] || { bg: "var(--text-muted)", label: String(rec || "—").toUpperCase() };
  return <span className="cl-rec-badge" style={{ background: s.bg }}>{s.label}</span>;
}

/**
 * ClosedLoopPanel — Adaptive Bidding & Model Governance demo.
 *
 * Emits synthetic market input via the orchestrator (data-plane only), then
 * invokes the closed-loop agents DIRECTLY (SigV4 via the Cognito Identity Pool —
 * the orchestrator is never in the agent-invocation path). Surfaces the agent's
 * real reasoning and animates an architecture-mapped flow (CloudWatch → AgentCore
 * → Bedrock → DynamoDB for the Adaptive Bidding loop; the governance pipeline for
 * the Governance loop). No fabricated data — unknown/unreachable states are shown
 * honestly.
 */
export default function ClosedLoopPanel() {
  const [scenarios, setScenarios] = useState([]);
  const [selected, setSelected] = useState(null);
  const [modelType, setModelType] = useState("dlrm_bid_shader");
  const [running, setRunning] = useState(false);
  const [nodes, setNodes] = useState([]);      // ordered flow nodes (real data)
  const [revealed, setRevealed] = useState(0); // staged-reveal count
  const [runError, setRunError] = useState(null);

  const [params, setParams] = useState(null);
  const [audit, setAudit] = useState(null);
  const [models, setModels] = useState(null);
  const [stateError, setStateError] = useState({});
  const [samplesScenario, setSamplesScenario] = useState(null); // scenario obj shown in sample-outcomes modal
  const revealTimer = useRef(null);

  // Load scenarios once.
  useEffect(() => {
    (async () => {
      try {
        const resp = await authFetch("/api/v1/closed-loop/scenarios");
        if (resp.ok) {
          const data = await resp.json();
          setScenarios(data.scenarios || []);
        }
      } catch (_) {
        /* honest empty state */
      }
    })();
  }, []);

  // Auto-sync the model-type selector to the selected scenario (manual override retained).
  useEffect(() => {
    if (!selected) return;
    const s = scenarios.find((x) => x.key === selected);
    if (s?.model_type) setModelType(s.model_type);
  }, [selected, scenarios]);

  const refreshState = useCallback(async () => {
    const errs = {};
    try {
      const r = await authFetch(`/api/v1/closed-loop/parameters?model_type=${modelType}`);
      const d = await r.json();
      if (r.ok) setParams(d.parameters || []);
      else { setParams(null); errs.params = d.error || `HTTP ${r.status}`; }
    } catch (e) { setParams(null); errs.params = String(e); }
    try {
      const r = await authFetch(`/api/v1/closed-loop/audit?model_type=${modelType}&limit=25`);
      const d = await r.json();
      if (r.ok) setAudit(d.records || []);
      else { setAudit(null); errs.audit = d.error || `HTTP ${r.status}`; }
    } catch (e) { setAudit(null); errs.audit = String(e); }
    try {
      const r = await authFetch(`/api/v1/closed-loop/models?model_type=${modelType}`);
      const d = await r.json();
      if (r.ok) setModels(d.versions || []);
      else { setModels(null); errs.models = d.error || `HTTP ${r.status}`; }
    } catch (e) { setModels(null); errs.models = String(e); }
    setStateError(errs);
  }, [modelType]);

  useEffect(() => { refreshState(); }, [refreshState]);

  // Staged reveal: once nodes are set, reveal them one-by-one (already-returned real data).
  useEffect(() => {
    if (revealTimer.current) clearInterval(revealTimer.current);
    if (nodes.length === 0) { setRevealed(0); return; }
    setRevealed(1);
    revealTimer.current = setInterval(() => {
      setRevealed((r) => {
        if (r >= nodes.length) { clearInterval(revealTimer.current); return r; }
        return r + 1;
      });
    }, 420);
    return () => revealTimer.current && clearInterval(revealTimer.current);
  }, [nodes]);

  // Replace the node at id with new props (used to flip a "processing" node to ok/error).
  const patchNode = useCallback((id, patch) => {
    setNodes((cur) => cur.map((n) => (n.id === id ? { ...n, ...patch } : n)));
  }, []);

  const runScenario = useCallback(async () => {
    if (!selected || running) return;
    const scenario = scenarios.find((s) => s.key === selected);
    if (!scenario) return;

    setRunning(true);
    setRunError(null);
    setNodes([]);
    setRevealed(0);

    try {
      if (scenario.loop === "agentic") {
        await runAgentic(scenario);
      } else {
        await runGovernance(scenario);
      }
      await refreshState();
    } catch (e) {
      setRunError(String(e?.message || e));
    } finally {
      setRunning(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, running, scenarios, modelType, refreshState]);

  // ── Adaptive Bidding loop ────────────────────────────────────────────────
  async function runAgentic(scenario) {
    // Step 1 — emit synthetic input via orchestrator (data-plane; NO invocation).
    let ctx = {};
    let emitOk = false;
    let visibilityWarning = null;
    try {
      const resp = await authFetch("/api/v1/closed-loop/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: scenario.key }),
      });
      const data = await resp.json();
      ctx = data.context || {};
      const generated = data.generated || [];
      emitOk = generated.some((g) => g.ok);
      // The bid-outcome emission (if present) reports whether CloudWatch confirmed
      // the datapoint is actually queryable before we proceed to invoke the agent
      // (PutMetricData is eventually consistent — see closed_loop_demo.generator).
      const bidEmit = generated.find((g) => g.namespace === "ARTF/BidOutcome");
      if (bidEmit && bidEmit.visible === false) {
        visibilityWarning =
          `CloudWatch had not confirmed the emitted metrics as queryable after ` +
          `${bidEmit.visible_wait_seconds ?? "?"}s — the agent may read an empty window.`;
      }
    } catch (_) { /* honest node below */ }

    const before = ctx.before || {};
    const ms = ctx.market_state || {};

    // Honest gate: is the Adaptive Bidding agent invokable?
    const unavailable = agentUnavailableReason("adaptive");

    const baseNodes = [
      {
        id: "cw", label: "CloudWatch", service: "Amazon CloudWatch", icon: IconCloudWatch,
        state: emitOk ? (visibilityWarning ? "warn" : "ok") : "warn",
        detail: (
          <div>
            <div className="cl-detail-title">Synthetic market metrics emitted</div>
            <div className="cl-kv">
              <span>win_rate <b>{fmt(ms.win_rate, 4)}</b></span>
              <span>ROI <b>{fmt(ms.roi, 4)}</b></span>
              <span>bids <b>{ms.total_bids ?? "—"}</b></span>
            </div>
            {visibilityWarning && <div className="cl-honest">{visibilityWarning}</div>}
          </div>
        ),
      },
      {
        id: "agent", label: "Adaptive Bidding Agent", service: "Amazon Bedrock AgentCore", icon: IconAgent,
        state: unavailable ? "error" : "processing",
        detail: unavailable
          ? <div className="cl-honest">{unavailable}</div>
          : <div className="cl-detail-title">Invoking runtime (SigV4)…</div>,
      },
      {
        id: "reason", label: "Reasoning", service: "Amazon Bedrock (LLM)", icon: IconBrain,
        state: "pending", detail: null,
      },
      {
        id: "persist", label: "Parameter Store", service: "Amazon DynamoDB", icon: IconDynamo,
        state: "pending",
        detail: (
          <div>
            <div className="cl-detail-title">Before</div>
            <div className="cl-kv">
              {Object.entries(before).map(([n, p]) => (
                <span key={n}>{n} <b>{fmt(p.current_value)}</b> <small>v{p.version}</small></span>
              ))}
            </div>
          </div>
        ),
      },
    ];
    setNodes(baseNodes);

    if (unavailable) return; // honest stop — no invocation, no fabricated result

    // Step 2 — invoke the Adaptive Bidding runtime DIRECTLY (SigV4 via Identity Pool).
    let agentResp;
    try {
      agentResp = await invokeAdaptive({ scenario: scenario.key, model_type: scenario.model_type });
    } catch (err) {
      patchNode("agent", { state: "error", detail: <div className="cl-honest">{err.message}</div> });
      patchNode("reason", { state: "skip", detail: <div style={SUBTLE}>No reasoning — invocation failed.</div> });
      return;
    }

    if (agentResp?.status === "error") {
      patchNode("agent", { state: "error", detail: <div className="cl-honest">{agentResp.error || "Agent error"}</div> });
      patchNode("reason", { state: "skip", detail: null });
      return;
    }

    const updates = agentResp?.updates || [];
    const rationale = agentResp?.rationale || "";
    const agentMs = agentResp?.market_state || ms;
    const durationMs = agentResp?.duration_ms;

    patchNode("agent", {
      state: "ok",
      detail: (
        <div>
          <div className="cl-detail-title">
            Runtime invoked{durationMs != null && <span className="cl-dur">{Math.round(durationMs)}ms</span>}
          </div>
          <div className="cl-kv">
            <span>win_rate <b>{fmt(agentMs.win_rate, 4)}</b></span>
            <span>ROI <b>{fmt(agentMs.roi, 4)}</b></span>
          </div>
        </div>
      ),
    });

    patchNode("reason", {
      state: "ok",
      detail: (
        <div>
          <div className="cl-detail-title">
            Agent reasoning {updates.length === 0 && <span className="cl-chip cl-chip-skip">no change</span>}
            {updates.length > 0 && <span className="cl-chip cl-chip-applied">applied</span>}
          </div>
          {rationale
            ? <div className="cl-rationale"><MarkdownLite text={rationale} /></div>
            : <p style={SUBTLE}>No rationale returned by the model.</p>}
          {updates.length > 0 && (
            <div className="cl-updates">
              {updates.map((u, i) => (
                <div key={i} className="cl-update sg-elevated">
                  <div className="cl-update-head">
                    <span style={MONO}>{u.parameter_name}</span>
                    <span>{fmt(u.old_value)} → <b>{fmt(u.new_value)}</b> <DeltaArrow delta={u.delta} oldValue={u.old_value} newValue={u.new_value} /></span>
                  </div>
                  {u.reason && <div className="cl-update-reason">{u.reason}</div>}
                  {u.confidence != null && <div style={SUBTLE}>confidence {fmt(u.confidence, 2)}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      ),
    });

    // Step 3 — read persisted "after" state (orchestrator data-plane read).
    let after = {};
    try {
      const r = await authFetch(`/api/v1/closed-loop/parameters?model_type=${scenario.model_type}`);
      const d = await r.json();
      if (r.ok) {
        after = Object.fromEntries((d.parameters || []).map((p) => [p.parameter_name, p]));
      }
    } catch (_) { /* keep before-only */ }

    patchNode("persist", {
      state: updates.length > 0 ? "ok" : "skip",
      detail: (
        <div>
          <div className="cl-detail-title">{updates.length > 0 ? "After (persisted)" : "Unchanged"}</div>
          <div className="cl-kv">
            {Object.entries(before).map(([n, p]) => {
              const a = after[n];
              return (
                <span key={n}>
                  {n} <b>{fmt(a ? a.current_value : p.current_value)}</b>{" "}
                  <small>v{a ? a.version : p.version}</small>
                </span>
              );
            })}
          </div>
        </div>
      ),
    });
  }

  // ── Governance loop ────────────────────────────────────────────────────
  async function runGovernance(scenario) {
    let decision = null;
    try {
      const resp = await authFetch("/api/v1/closed-loop/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: scenario.key }),
      });
      const data = await resp.json();
      decision = data.decision;
    } catch (e) {
      setRunError(String(e));
    }

    // Honest pipeline: stages that only run on real model-registration events are
    // labelled as such (never faked as completed). The A/B gate is the REAL result.
    const eventNote = "Runs on real model-registration events";
    const gNodes = [
      { id: "reg", label: "Model Registry", service: "Amazon SageMaker", icon: IconRegistry, state: "event", detail: <div style={SUBTLE}>{eventNote}</div> },
      { id: "opt", label: "Model Optimizer", service: "NVIDIA TensorRT", icon: IconChip, state: "event", detail: <div style={SUBTLE}>ONNX → TensorRT · {eventNote}</div> },
      { id: "canary", label: "Canary", service: "NVIDIA Triton", icon: IconSplit, state: "event", detail: <div style={SUBTLE}>stable ⇄ canary split · {eventNote}</div> },
      {
        id: "ab", label: "A/B Evaluation", service: "Welch t-test + SPRT", icon: IconGavel,
        state: decision ? "ok" : "error",
        detail: decision ? (
          <div>
            <div className="cl-detail-title">A/B result <RecommendationBadge rec={decision.recommendation} /></div>
            <div className="cl-kv">
              <span>control <b>{fmt(decision.control_metric)}</b></span>
              <span>treatment <b>{fmt(decision.treatment_metric)}</b></span>
              <span>lift <b>{fmt(decision.relative_lift)}</b></span>
              <span>p <b>{fmt(decision.p_value, 5)}</b></span>
            </div>
            <div className="cl-kv"><span>samples {decision.samples_control} / {decision.samples_treatment}</span></div>
            {decision.guardrail_violations?.length > 0 && (
              <div className="cl-honest">Guardrail: {decision.guardrail_violations.join("; ")}</div>
            )}
          </div>
        ) : <div className="cl-honest">A/B evaluation unavailable.</div>,
      },
      {
        id: "audit", label: "Audit + Rationale", service: "Amazon DynamoDB / Bedrock", icon: IconAudit,
        state: "event",
        detail: (
          <div style={SUBTLE}>
            Promotion decisions + the governance agent's natural-language rationale are
            written to the audit trail on real model events. {isGovernanceConfigured()
              ? "The deployed Governance runtime records them."
              : "Deploy the Governance agent to record them."}
          </div>
        ),
      },
    ];
    setNodes(gNodes);
  }

  const agentic = scenarios.filter((s) => s.loop === "agentic");
  const governance = scenarios.filter((s) => s.loop === "governance");

  return (
    <div className="cl-page">
      <div className="cl-page-header">
        <h2>Adaptive Bidding &amp; Model Governance</h2>
        <p className="cl-page-desc">
          Run a scenario to emit synthetic market data, then watch the closed-loop agents act
          across the architecture. Agents are invoked directly (SigV4); the orchestrator is not
          in the invocation path. Decisions, reasoning, and persisted state are real.
        </p>
      </div>

      <div className="cl-layout">
        <div className="cl-layout-left">
          <div className="cl-section-title">Adaptive Bidding loop — parameter tuning</div>
          <div className="cl-scenario-grid">
            {agentic.map((s) => (
              <ScenarioCardCL
                key={s.key}
                s={s}
                selected={selected === s.key}
                onSelect={() => setSelected(s.key)}
                onViewSamples={setSamplesScenario}
              />
            ))}
          </div>

          <div className="cl-section-title">Governance loop — A/B model promotion</div>
          <div className="cl-scenario-grid">
            {governance.map((s) => (
              <ScenarioCardCL
                key={s.key}
                s={s}
                selected={selected === s.key}
                onSelect={() => setSelected(s.key)}
                onViewSamples={setSamplesScenario}
              />
            ))}
          </div>

          <div className="cl-run-row">
            <button className="btn btn-primary sg-interactive" onClick={runScenario} disabled={!selected || running}>
              {running ? <><span className="spinner" /> Running…</> : "Run scenario"}
            </button>
            {selected && <span style={SUBTLE}>Selected: {selected}</span>}
          </div>

          <div className="info-bar" style={{ margin: "16px 0 12px" }}>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span>Model:</span>
              <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="cl-select sg-interactive">
                {MODEL_TYPES.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </div>
            <button className="btn-secondary sg-interactive" onClick={refreshState} style={{ padding: "5px 12px" }}>Refresh</button>
          </div>

          <AuditView records={audit} error={stateError.audit} />
          <ModelsView versions={models} error={stateError.models} />
        </div>

        <div className="cl-layout-right">
          {nodes.length === 0 && !running && (
            <div className="cl-empty sg-elevated">
              Select a scenario and click <strong>Run scenario</strong> to watch the loop execute
              across the architecture.
            </div>
          )}
          {runError && <div className="cl-honest cl-honest-block">Run failed: {runError}</div>}
          {nodes.length > 0 && <FlowTrack nodes={nodes} revealed={revealed} running={running} />}

          {/* Current parameters live under the dynamic testing (flow) area so the
              persisted DynamoDB state sits directly below the run it reflects. */}
          <ParametersView params={params} error={stateError.params} />
        </div>
      </div>

      {samplesScenario && (
        <SampleOutcomesModal scenario={samplesScenario} onClose={() => setSamplesScenario(null)} />
      )}
    </div>
  );
}

// ── Scenario card (reframed "Expected" → "What to watch for") ──────────────
function ScenarioCardCL({ s, selected, onSelect, onViewSamples }) {
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

// ── Sample outcomes modal ───────────────────────────────────────────────────
// Shows a subset of the individual synthetic records behind a scenario: for
// agentic scenarios, illustrative bid-outcome records (won/price_paid/
// impression/click/conversion) consistent with the scenario's aggregate
// metrics; for governance scenarios, the first N control/treatment values
// actually fed to the real ABEvaluator. Data is fetched from the orchestrator
// on open — nothing here is fabricated client-side.
function SampleOutcomesModal({ scenario, onClose }) {
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
function FlowTrack({ nodes, revealed, running }) {
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

// ── Persisted-state views (real DynamoDB / SageMaker reads) ────────────────
function ParametersView({ params, error }) {
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

function AuditView({ records, error }) {
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

function ModelsView({ versions, error }) {
  return (
    <div className="cl-block sg-elevated">
      <div className="cl-block-title">Model registry versions · SageMaker</div>
      {error && <div className="cl-honest">Unavailable: {error}</div>}
      {!error && versions && versions.length === 0 && <div style={SUBTLE}>No registered model versions yet.</div>}
      {!error && versions && versions.length > 0 && (
        <table className="cl-table">
          <thead><tr><th>Version</th><th>Approval</th><th>Status</th><th>Created</th></tr></thead>
          <tbody>
            {versions.map((v, i) => (
              <tr key={i}>
                <td>{v.version}</td>
                <td><ApprovalBadge status={v.approval_status} /></td>
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

function ApprovalBadge({ status }) {
  const map = { Approved: "#16a34a", Rejected: "#dc2626", PendingManualApproval: "#d97706" };
  const bg = map[status] || "var(--text-muted)";
  return <span className="cl-approval-badge" style={{ background: bg }}>{status || "—"}</span>;
}
