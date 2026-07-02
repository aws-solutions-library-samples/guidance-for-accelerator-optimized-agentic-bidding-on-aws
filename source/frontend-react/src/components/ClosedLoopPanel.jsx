import React, { useState, useEffect, useCallback } from "react";
import { authFetch } from "../authFetch.js";
import { invokeAgentCore, isAgentCoreConfigured } from "../agentCoreClient.js";

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

function DeltaArrow({ delta }) {
  if (delta === 0 || delta === null || delta === undefined) {
    return <span style={{ color: "var(--text-muted)" }}>→ no change</span>;
  }
  const up = delta > 0;
  return (
    <span style={{ color: up ? "#16a34a" : "#dc2626", fontWeight: 600 }}>
      {up ? "▲" : "▼"} {up ? "+" : ""}{fmt(delta, 4)}
    </span>
  );
}

function RecommendationBadge({ rec }) {
  const map = {
    promote: { bg: "#16a34a", label: "PROMOTE" },
    reject: { bg: "#dc2626", label: "REJECT" },
    extend: { bg: "#d97706", label: "EXTEND (keep testing)" },
  };
  const s = map[rec] || { bg: "var(--text-muted)", label: String(rec || "—").toUpperCase() };
  return (
    <span style={{ background: s.bg, color: "#fff", padding: "2px 10px", borderRadius: "10px", fontSize: "11px", fontWeight: 700 }}>
      {s.label}
    </span>
  );
}

/**
 * ClosedLoopPanel — Model Tuning page.
 *
 * Runs scenarios with synthetic market data as INPUT to the adaptive bidding
 * agents and displays the resulting decisions and persisted state.
 */
export default function ClosedLoopPanel() {
  const [scenarios, setScenarios] = useState([]);
  const [selected, setSelected] = useState(null);
  const [modelType, setModelType] = useState("dlrm_bid_shader");
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState(null);
  const [runError, setRunError] = useState(null);

  const [params, setParams] = useState(null);
  const [audit, setAudit] = useState(null);
  const [models, setModels] = useState(null);
  const [stateError, setStateError] = useState({});

  // Load scenarios once
  useEffect(() => {
    (async () => {
      try {
        const resp = await authFetch("/api/v1/closed-loop/scenarios");
        if (resp.ok) {
          const data = await resp.json();
          setScenarios(data.scenarios || []);
        }
      } catch (_) {
        /* keep empty; UI shows honest empty state */
      }
    })();
  }, []);

  const refreshState = useCallback(async () => {
    const errs = {};
    // Parameters
    try {
      const r = await authFetch(`/api/v1/closed-loop/parameters?model_type=${modelType}`);
      const d = await r.json();
      if (r.ok) setParams(d.parameters || []);
      else { setParams(null); errs.params = d.error || `HTTP ${r.status}`; }
    } catch (e) { setParams(null); errs.params = String(e); }
    // Audit
    try {
      const r = await authFetch(`/api/v1/closed-loop/audit?model_type=${modelType}&limit=25`);
      const d = await r.json();
      if (r.ok) setAudit(d.records || []);
      else { setAudit(null); errs.audit = d.error || `HTTP ${r.status}`; }
    } catch (e) { setAudit(null); errs.audit = String(e); }
    // Models
    try {
      const r = await authFetch(`/api/v1/closed-loop/models?model_type=${modelType}`);
      const d = await r.json();
      if (r.ok) setModels(d.versions || []);
      else { setModels(null); errs.models = d.error || `HTTP ${r.status}`; }
    } catch (e) { setModels(null); errs.models = String(e); }
    setStateError(errs);
  }, [modelType]);

  useEffect(() => { refreshState(); }, [refreshState]);

  const runScenario = useCallback(async () => {
    if (!selected) return;
    setRunning(true);
    setRunError(null);
    setRunResult(null);

    const scenario = scenarios.find((s) => s.key === selected);

    try {
      if (scenario?.loop === "agentic" && isAgentCoreConfigured()) {
        // Invoke AgentCore directly from the browser — no orchestrator middleman
        const startTime = performance.now();

        // Step 1: Write synthetic metrics to CloudWatch via the orchestrator
        // (the agent reads these when it runs)
        const emitResp = await authFetch("/api/v1/closed-loop/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scenario: selected }),
        });
        const emitData = await emitResp.json();

        // Step 2: Invoke the agent directly via AgentCore
        const agentResponse = await invokeAgentCore({
          scenario: selected,
          model_type: modelType,
        });

        const duration = Math.round(performance.now() - startTime);

        // Build the result combining emit evidence + agent response
        setRunResult({
          generated: emitData.generated || [],
          decision: {
            loop: "agentic",
            invoked_via: "agentcore",
            agentcore_duration_ms: duration,
            scenario: selected,
            model_type: modelType,
            market_state: emitData.decision?.market_state || scenario?.bid_metrics || {},
            before: emitData.decision?.before || {},
            updates: agentResponse.updates || [],
            skipped: (agentResponse.updates || []).length === 0,
            after: emitData.decision?.after || {},
            error: agentResponse.status === "error" ? agentResponse.error : null,
          },
          errors: [],
        });
      } else {
        // Governance scenarios or AgentCore not configured — use orchestrator
        const resp = await authFetch("/api/v1/closed-loop/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scenario: selected }),
        });
        const data = await resp.json();
        setRunResult(data);
        if (!resp.ok && (!data.errors || data.errors.length === 0)) {
          setRunError(`HTTP ${resp.status}`);
        }
      }
      // Refresh persisted state after the decision runs
      await refreshState();
    } catch (e) {
      setRunError(String(e));
    } finally {
      setRunning(false);
    }
  }, [selected, scenarios, modelType, refreshState]);

  const agentic = scenarios.filter((s) => s.loop === "agentic");
  const governance = scenarios.filter((s) => s.loop === "governance");

  return (
    <div className="cl-page">
      <div className="cl-page-header">
        <h2>Adaptive Bidding & Model Retraining</h2>
        <p className="cl-page-desc">
          Run scenarios with synthetic market data to exercise the adaptive bidding agents.
          Decisions and persisted state are computed by the live agents.
        </p>
      </div>

      <div className="cl-layout">
        {/* Left: Scenario picker + controls + persisted state */}
        <div className="cl-layout-left">
          <div className="cl-section-title">Agentic parameter loop — Bid Shading Agent</div>
          <div className="cl-scenario-grid">
            {agentic.map((s) => (
              <ScenarioCardCL key={s.key} s={s} selected={selected === s.key} onSelect={() => setSelected(s.key)} />
            ))}
          </div>

          <div className="cl-section-title">Governance loop — A/B evaluation</div>
          <div className="cl-scenario-grid">
            {governance.map((s) => (
              <ScenarioCardCL key={s.key} s={s} selected={selected === s.key} onSelect={() => setSelected(s.key)} />
            ))}
          </div>

          <div style={{ display: "flex", gap: "10px", alignItems: "center", margin: "12px 0" }}>
            <button className="btn" onClick={runScenario} disabled={!selected || running}>
              {running ? <><span className="spinner" /> Running…</> : "Generate & Run"}
            </button>
            {selected && <span style={SUBTLE}>Selected: {selected}</span>}
          </div>

          {/* Persisted state */}
          <div className="info-bar" style={{ margin: "16px 0 12px" }}>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span>Model:</span>
              <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="cl-select">
                {MODEL_TYPES.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
              </select>
            </div>
            <button className="btn-secondary" onClick={refreshState} style={{ padding: "5px 12px" }}>Refresh</button>
          </div>

          <ParametersView params={params} error={stateError.params} />
          <AuditView records={audit} error={stateError.audit} />
          <ModelsView versions={models} error={stateError.models} />
        </div>

        {/* Right: Live execution results */}
        <div className="cl-layout-right">
          {!runResult && !runError && !running && (
            <div className="cl-empty-state">
              Select a scenario and click <strong>Generate & Run</strong> to see the feedback loop execute.
            </div>
          )}
          {running && (
            <div className="cl-running-state">
              <span className="spinner" /> Invoking agent…
            </div>
          )}
          {runError && (
            <div className="cl-error">Run failed: {runError}</div>
          )}
          {runResult && <RunResult result={runResult} />}
        </div>
      </div>
    </div>
  );
}

function ScenarioCardCL({ s, selected, onSelect }) {
  return (
    <button
      className={`cl-scenario-card${selected ? " selected" : ""}`}
      onClick={onSelect}
      type="button"
    >
      <div className="cl-scenario-label">{s.label}</div>
      <div className="cl-scenario-desc">{s.description}</div>
      <div className="cl-scenario-expected">Expected: {s.expected_decision}</div>
    </button>
  );
}

function RunResult({ result }) {
  const decision = result.decision;
  const gen = result.generated || [];
  const errors = result.errors || [];

  if (!decision && gen.length === 0) return null;

  // Build flow nodes based on loop type
  const nodes = buildFlowNodes(gen, decision, errors);

  return (
    <div className="cl-flow">
      {/* Horizontal flowchart nodes */}
      <div className="cl-flow-track">
        {nodes.map((node, i) => (
          <React.Fragment key={node.id}>
            {i > 0 && <div className={`cl-flow-connector ${node.active ? "active" : ""}`} style={{ animationDelay: `${i * 0.4}s` }} />}
            <div className={`cl-flow-node ${node.active ? "active" : ""} ${node.status}`} style={{ animationDelay: `${i * 0.4}s` }}>
              <div className="cl-flow-icon">{node.icon}</div>
              <div className="cl-flow-label">{node.label}</div>
            </div>
          </React.Fragment>
        ))}
      </div>

      {/* Detail panels below each completed node */}
      <div className="cl-flow-details">
        {nodes.map((node) => (
          node.active && node.detail && (
            <div key={node.id} className="cl-flow-detail" style={{ animationDelay: `${node.index * 0.4 + 0.2}s` }}>
              {node.detail}
            </div>
          )
        ))}
      </div>

      {errors.length > 0 && (
        <div className="cl-error" style={{ marginTop: "12px" }}>{errors.join("; ")}</div>
      )}
    </div>
  );
}

function buildFlowNodes(gen, decision, errors) {
  const nodes = [];
  let idx = 0;

  // Node 1: CloudWatch emit
  const cwOk = gen.length > 0 && gen[0]?.ok;
  nodes.push({
    id: "emit",
    index: idx++,
    icon: <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>,
    label: "CloudWatch",
    active: gen.length > 0,
    status: cwOk ? "ok" : gen.length > 0 ? "error" : "pending",
    detail: gen.length > 0 ? (
      <div>
        <div className="cl-flow-detail-title">Metrics Written</div>
        <div className="cl-metric-pills">
          {gen.flatMap((g) => (g.emitted || []).map((m, j) => (
            <span key={`${g.namespace}-${j}`} className="cl-metric-pill">{m.metric_name}={m.value}</span>
          )))}
        </div>
      </div>
    ) : null,
  });

  if (decision && decision.loop === "agentic") {
    const ms = decision.market_state || {};
    const before = decision.before || {};
    const after = decision.after || {};
    const updates = decision.updates || [];

    // Node 2: Agent reads state
    nodes.push({
      id: "read",
      index: idx++,
      icon: <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
      label: "Read State",
      active: !!decision,
      status: decision.error && !updates.length ? "error" : "ok",
      detail: (
        <div>
          <div className="cl-flow-detail-title">
            Market State
            {decision.invoked_via === "agentcore" && (
              <span className="cl-badge cl-badge--agentcore" style={{ marginLeft: "8px" }}>AgentCore</span>
            )}
            {decision.agentcore_duration_ms && (
              <span className="cl-flow-duration">{decision.agentcore_duration_ms}ms</span>
            )}
          </div>
          <div className="cl-kv-row">
            <span>win_rate: <strong>{fmt(ms.win_rate, 4)}</strong></span>
            <span>ROI: <strong>{fmt(ms.roi, 4)}</strong></span>
            <span>bids: <strong>{ms.total_bids}</strong></span>
          </div>
          <div className="cl-flow-detail-title" style={{ marginTop: "8px" }}>Current Parameters</div>
          <div className="cl-kv-row">
            {Object.entries(before).map(([name, p]) => (
              <span key={name}>{name}: <strong>{fmt(p.current_value)}</strong> <small>(v{p.version})</small></span>
            ))}
          </div>
        </div>
      ),
    });

    // Node 3: Agent decides
    const hasUpdates = updates.length > 0;
    nodes.push({
      id: "decide",
      index: idx++,
      icon: <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>,
      label: "Decide",
      active: !!decision,
      status: decision.error ? "error" : decision.skipped ? "skip" : "ok",
      detail: (
        <div>
          <div className="cl-flow-detail-title">
            Agent Reasoning
            {decision.skipped && <span className="cl-badge cl-badge--skip" style={{ marginLeft: "8px" }}>no change</span>}
            {decision.error && <span className="cl-badge cl-badge--error" style={{ marginLeft: "8px" }}>rejected</span>}
            {hasUpdates && <span className="cl-badge cl-badge--applied" style={{ marginLeft: "8px" }}>applied</span>}
          </div>
          {decision.error && <div className="cl-error">{decision.error}</div>}
          {decision.skipped && !decision.error && (
            <div className="cl-skip-reason">Within tolerance or below min-samples.</div>
          )}
          {hasUpdates && (
            <div className="cl-decisions">
              {updates.map((u, i) => (
                <div key={i} className="cl-decision-card">
                  <div className="cl-decision-param">
                    <span style={MONO}>{u.parameter_name}</span>
                    <span className="cl-decision-delta">
                      {fmt(u.old_value)} → <strong>{fmt(u.new_value)}</strong>{" "}
                      <DeltaArrow delta={u.delta} />
                    </span>
                  </div>
                  <div className="cl-decision-reason">{u.reason}</div>
                  <div className="cl-decision-confidence">Confidence: {fmt(u.confidence, 2)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      ),
    });

    // Node 4: Persist to DynamoDB
    if (hasUpdates && !decision.error) {
      nodes.push({
        id: "persist",
        index: idx++,
        icon: <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
        label: "DynamoDB",
        active: true,
        status: "ok",
        detail: (
          <div>
            <div className="cl-flow-detail-title">Updated Parameters</div>
            <div className="cl-kv-row">
              {Object.entries(after).map(([name, p]) => (
                <span key={name}>{name}: <strong>{fmt(p.current_value)}</strong> <small>(v{p.version})</small></span>
              ))}
            </div>
          </div>
        ),
      });
    }
  } else if (decision && decision.loop === "governance") {
    // Governance: single evaluation node
    nodes.push({
      id: "evaluate",
      index: idx++,
      icon: <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
      label: "A/B Evaluate",
      active: true,
      status: decision.recommendation === "promote" ? "ok" : decision.recommendation === "reject" ? "error" : "skip",
      detail: (
        <div>
          <div className="cl-flow-detail-title">
            A/B Evaluation
            <RecommendationBadge rec={decision.recommendation} />
          </div>
          <div className="cl-kv-row">
            <span>Control: <strong>{fmt(decision.control_metric)}</strong></span>
            <span>Treatment: <strong>{fmt(decision.treatment_metric)}</strong></span>
            <span>Lift: <strong>{fmt(decision.relative_lift)}</strong></span>
            <span>p-value: <strong>{fmt(decision.p_value, 5)}</strong></span>
          </div>
          <div className="cl-kv-row" style={{ marginTop: "4px" }}>
            <span>Samples: {decision.samples_control} / {decision.samples_treatment}</span>
          </div>
          {decision.guardrail_violations?.length > 0 && (
            <div className="cl-error">Guardrail: {decision.guardrail_violations.join("; ")}</div>
          )}
        </div>
      ),
    });
  }

  return nodes;
}

function ParametersView({ params, error }) {
  return (
    <div className="cl-block">
      <div className="cl-block-title">Current parameters (DynamoDB)</div>
      {error && <div className="cl-error">Unavailable: {error}</div>}
      {!error && params && params.length === 0 && (
        <div style={SUBTLE}>No parameters initialized yet. Run an agentic scenario to create them.</div>
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
    <div className="cl-block">
      <div className="cl-block-title">Audit trail (newest first)</div>
      {error && <div className="cl-error">Unavailable: {error}</div>}
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
    <div className="cl-block">
      <div className="cl-block-title">Model registry versions (SageMaker)</div>
      {error && <div className="cl-error">Unavailable: {error}</div>}
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
  const map = {
    Approved: "#16a34a",
    Rejected: "#dc2626",
    PendingManualApproval: "#d97706",
  };
  const bg = map[status] || "var(--text-muted)";
  return (
    <span style={{ background: bg, color: "#fff", padding: "1px 8px", borderRadius: "8px", fontSize: "10px", fontWeight: 600 }}>
      {status || "—"}
    </span>
  );
}
