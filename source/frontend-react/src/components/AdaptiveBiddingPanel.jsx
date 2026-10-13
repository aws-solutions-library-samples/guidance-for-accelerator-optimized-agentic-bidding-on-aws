import React, { useState, useEffect, useCallback, useRef } from "react";
import { authFetch } from "../authFetch.js";
import { invokeAdaptive, agentUnavailableReason } from "../agentCoreClient.js";
import {
  SUBTLE, MONO, fmt, MarkdownLite,
  IconCloudWatch, IconAgent, IconBrain, IconDynamo,
  DeltaArrow,
  ScenarioDetailCard, SampleOutcomesModal, FlowTrack,
  ParametersView, AuditView,
} from "./closedLoopUi.jsx";

const MODEL_TYPES = [
  { key: "dlrm_bid_shader", label: "Bid Pricer" },
  { key: "ncf_deal_manager", label: "Deal Scorer" },
  { key: "widedeep_segment_activator", label: "Audience Activator" },
];

/**
 * AdaptiveBiddingPanel — the parameter-tuning closed loop.
 *
 * Emits synthetic market input via the orchestrator (data-plane only), then
 * invokes the Adaptive Bidding Strategy Agent DIRECTLY (SigV4 via the Cognito
 * Identity Pool — the orchestrator is never in the agent-invocation path).
 * Surfaces the agent's real reasoning and animates an architecture-mapped flow
 * (CloudWatch → AgentCore → Bedrock → DynamoDB). No fabricated data —
 * unknown/unreachable states are shown honestly.
 */
export default function AdaptiveBiddingPanel() {
  const [scenarios, setScenarios] = useState([]);
  const [selected, setSelected] = useState(null);
  const [modelType, setModelType] = useState("dlrm_bid_shader");
  const [running, setRunning] = useState(false);
  const [nodes, setNodes] = useState([]);      // ordered flow nodes (real data)
  const [revealed, setRevealed] = useState(0); // staged-reveal count
  const [runError, setRunError] = useState(null);

  const [params, setParams] = useState(null);
  const [audit, setAudit] = useState(null);
  const [stateError, setStateError] = useState({});
  const [samplesScenario, setSamplesScenario] = useState(null); // scenario obj shown in sample-outcomes modal
  const revealTimer = useRef(null);

  // Load scenarios once (filtered to the agentic loop).
  useEffect(() => {
    (async () => {
      try {
        const resp = await authFetch("/api/v1/closed-loop/scenarios?loop=agentic");
        if (resp.ok) {
          const data = await resp.json();
          const list = data.scenarios || [];
          setScenarios(list);
          // Default to the first scenario so the controls row's dropdown and
          // detail card are populated immediately (matches Governance panel).
          if (list.length > 0) setSelected((cur) => cur ?? list[0].key);
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
      await runAgentic(scenario);
      await refreshState();
    } catch (e) {
      setRunError(String(e?.message || e));
    } finally {
      setRunning(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, running, scenarios, modelType, refreshState]);

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

  return (
    <div className="cl-page">
      <div className="cl-page-header">
        <h2>Adaptive Bidding</h2>
        <p className="cl-page-desc">
          Run a scenario to emit synthetic market data, then watch the Adaptive Bidding
          Strategy Agent act across the architecture. The agent is invoked directly (SigV4);
          the orchestrator is not in the invocation path. Decisions, reasoning, and persisted
          state are real.
        </p>
      </div>

      {/* Controls — single compact row: model select, scenario select, run
          button, matching the Governance panel's controls-bar. */}
      <div className="cl-controls-bar">
        <div className="cl-control-group">
          <label htmlFor="cl-adaptive-model">Model:</label>
          <select
            id="cl-adaptive-model"
            className="cl-select sg-interactive"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            disabled={running}
          >
            {MODEL_TYPES.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>
        <div className="cl-control-group">
          <label htmlFor="cl-adaptive-scenario">Scenario:</label>
          <select
            id="cl-adaptive-scenario"
            className="cl-select sg-interactive"
            value={selected || ""}
            onChange={(e) => setSelected(e.target.value)}
            disabled={running}
          >
            {scenarios.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </div>
        <button className="btn btn-primary sg-interactive" onClick={runScenario} disabled={!selected || running}>
          {running ? <><span className="spinner" /> Running…</> : "Run scenario"}
        </button>
        {selected && <span className="cl-run-status">Selected: {selected}</span>}
      </div>

      {runError && <div className="cl-honest cl-honest-block">Run failed: {runError}</div>}

      {/* Selected scenario's detail (left) next to the architecture-mapped live
          flow results (right) — same side-by-side pairing the Governance panel
          uses for its scenario detail + live mutation output. */}
      <div className="cl-mutation-grid">
        <ScenarioDetailCard
          s={scenarios.find((s) => s.key === selected)}
          onViewSamples={setSamplesScenario}
        />
        {nodes.length > 0 ? (
          <FlowTrack nodes={nodes} revealed={revealed} running={running} />
        ) : (
          <div className="cl-empty sg-elevated">
            Select a scenario and click <strong>Run scenario</strong> to watch the loop execute
            across the architecture.
          </div>
        )}
      </div>

      {/* Persisted state — parameters + audit trail side by side, matching the
          Governance panel's bottom data-grid layout. */}
      <div className="cl-section-title">Persisted state</div>
      <div className="info-bar" style={{ margin: "0 0 12px" }}>
        <span>DynamoDB parameter store &amp; audit trail</span>
        <button className="btn-secondary sg-interactive" onClick={refreshState} style={{ padding: "5px 12px" }}>Refresh</button>
      </div>
      <div className="cl-mutation-grid">
        <ParametersView params={params} error={stateError.params} />
        <AuditView records={audit} error={stateError.audit} />
      </div>

      {samplesScenario && (
        <SampleOutcomesModal scenario={samplesScenario} onClose={() => setSamplesScenario(null)} />
      )}
    </div>
  );
}
