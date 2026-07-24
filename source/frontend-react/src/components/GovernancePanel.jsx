import React, { useState, useEffect, useCallback, useRef } from "react";
import { authFetch } from "../authFetch.js";
import { invokeGovernance, agentUnavailableReason } from "../agentCoreClient.js";
import {
  fmt,
  IconRegistry, IconChip, IconSplit, IconGavel, IconAudit,
  ScenarioDetailCard, SampleOutcomesModal, PipelineBar, StepByStepLog,
  ModelsView, MutationIntentCard, BidstreamImpactCard,
  GovernanceVerdictCard, SessionAuditTrail,
} from "./closedLoopUi.jsx";

// Default pipeline stages shown (all "done"/grey) before any scenario has run,
// matching the prototype's always-visible pipeline bar at the top of the page.
const DEFAULT_PIPELINE_NODES = [
  { id: "reg", label: "Registry", service: "SageMaker", icon: IconRegistry, state: "event" },
  { id: "opt", label: "Optimize", service: "TensorRT FP16", icon: IconChip, state: "event" },
  { id: "canary", label: "Canary", service: "Triton router", icon: IconSplit, state: "event" },
  { id: "ab", label: "A/B Eval", service: "Welch t + SPRT", icon: IconGavel, state: "event" },
  { id: "audit", label: "Audit", service: "DynamoDB / Bedrock", icon: IconAudit, state: "event" },
];

// Governance scenarios only exist for the two GPU-accelerated, canary-routed
// models (per DESIGN_BRIEF.md — Wide&Deep is rule-based, no canary/A-B loop).
// The model selector filters the scenario list, so picking a model actually
// changes which real scenario/decision can run — not just a display label.
const MODEL_TYPES = [
  { key: "dlrm_bid_shader", label: "DLRM Bid Shader" },
  { key: "ncf_deal_manager", label: "NCF Deal Manager" },
];

/**
 * GovernancePanel — the A/B model-promotion closed loop.
 *
 * Runs the REAL statistical gate (Welch's t-test + SPRT, via ``ABEvaluator``) in
 * the orchestrator against a scenario's synthetic A/B samples, then calls the
 * deployed Model Promotion Governance Agent's lightweight "explain_scenario" mode
 * DIRECTLY (SigV4 via the Cognito Identity Pool) so the real agent's Bedrock
 * reasoning layer explains that real decision. This does NOT run the full
 * event-driven pipeline (TensorRT optimize / Triton canary / SageMaker registry
 * update / audit-trail write) — those only happen on genuine model-registration
 * events, not demo scenarios. Everything shown here is real: the statistics, the
 * decision, and the rationale. Unknown/unreachable states are shown honestly.
 */
export default function GovernancePanel() {
  const [scenarios, setScenarios] = useState([]);
  const [selected, setSelected] = useState(null);
  const [modelType, setModelType] = useState("dlrm_bid_shader");
  const [running, setRunning] = useState(false);
  const [nodes, setNodes] = useState([]);
  const [revealed, setRevealed] = useState(0);
  const [runError, setRunError] = useState(null);

  const [models, setModels] = useState(null);
  const [stateError, setStateError] = useState({});
  const [samplesScenario, setSamplesScenario] = useState(null);
  const revealTimer = useRef(null);

  // Real decision + rationale from the most recently completed run, and this
  // session's run history — both persistent (not cleared when the flow track
  // resets on the next run), matching the prototype's verdict/audit cards.
  const [lastDecision, setLastDecision] = useState(null);
  const [lastRationale, setLastRationale] = useState(null);
  const [lastModelType, setLastModelType] = useState(null);
  const [sessionAudit, setSessionAudit] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const resp = await authFetch("/api/v1/closed-loop/scenarios?loop=governance");
        if (resp.ok) {
          const data = await resp.json();
          const list = data.scenarios || [];
          setScenarios(list);
          // Default to the first scenario matching the current model so the
          // controls row and detail card are populated immediately.
          const firstForModel = list.find((s) => s.model_type === modelType) || list[0];
          if (firstForModel) setSelected((cur) => cur ?? firstForModel.key);
        }
      } catch (_) {
        /* honest empty state */
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Model is the primary selector: changing it filters the scenario list to
  // that model's real governance scenarios and re-selects the first one, so
  // "Model: NCF" always corresponds to a scenario that actually evaluates NCF.
  const scenariosForModel = scenarios.filter((s) => s.model_type === modelType);

  const handleModelChange = useCallback((nextModelType) => {
    setModelType(nextModelType);
    const list = scenarios.filter((s) => s.model_type === nextModelType);
    if (list.length > 0) setSelected(list[0].key);
  }, [scenarios]);

  const refreshState = useCallback(async () => {
    const errs = {};
    try {
      const r = await authFetch(`/api/v1/closed-loop/models?model_type=${modelType}`);
      const d = await r.json();
      if (r.ok) setModels(d.versions || []);
      else { setModels(null); errs.models = d.error || `HTTP ${r.status}`; }
    } catch (e) { setModels(null); errs.models = String(e); }
    setStateError(errs);
  }, [modelType]);

  useEffect(() => { refreshState(); }, [refreshState]);

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

  const runScenario = useCallback(async () => {
    if (!selected || running) return;
    const scenario = scenarios.find((s) => s.key === selected);
    if (!scenario) return;

    setRunning(true);
    setRunError(null);
    setNodes([]);
    setRevealed(0);

    try {
      await runGovernance(scenario);
      await refreshState();
    } catch (e) {
      setRunError(String(e?.message || e));
    } finally {
      setRunning(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, running, scenarios, modelType, refreshState]);

  async function runGovernance(scenario) {
    // Step 1 — run the REAL statistical gate (ABEvaluator) via the orchestrator.
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

    setLastDecision(decision);
    setLastRationale(null);
    setLastModelType(scenario.model_type);

    // Honest pipeline: stages that only run on real model-registration events are
    // labelled as such (never faked as completed). The A/B gate is the REAL result.
    // Exactly 5 stages, matching the reference prototype's pipeline bar (the
    // agent-reasoning result is surfaced separately in the persistent verdict
    // card below, not as an extra pipeline icon).
    const eventNote = "runs on real model-registration events";
    const abLog = decision
      ? `Welch's t-test — control ${fmt(decision.control_metric)}, treatment ${fmt(decision.treatment_metric)} `
        + `(lift ${fmt(decision.relative_lift)}, p=${fmt(decision.p_value, 5)}, n=${decision.samples_control}/${decision.samples_treatment}). `
        + (decision.guardrail_violations?.length > 0
          ? `Guardrail violation: ${decision.guardrail_violations.join("; ")}.`
          : "Guardrails within bounds.")
      : "A/B evaluation unavailable.";

    const gNodes = [
      { id: "reg", label: "Registry", service: "Amazon SageMaker", icon: IconRegistry, state: "event", logText: `New candidate version registered — ${eventNote}.` },
      { id: "opt", label: "Optimize", service: "NVIDIA TensorRT", icon: IconChip, state: "event", logText: `ONNX \u2192 TensorRT compilation — ${eventNote}.` },
      { id: "canary", label: "Canary", service: "NVIDIA Triton", icon: IconSplit, state: "event", logText: `Stable \u21c4 canary traffic split — ${eventNote}.` },
      { id: "ab", label: "A/B Eval", service: "Welch t + SPRT", icon: IconGavel, state: decision ? "ok" : "error", logText: abLog },
      { id: "audit", label: "Audit", service: "DynamoDB / Bedrock", icon: IconAudit, state: "event", logText: "Decision + rationale would be logged to DynamoDB on a real model-registration event; this demo run is not written there." },
    ];
    setNodes(gNodes);

    if (!decision) return; // honest stop — no fabricated rationale without a real decision

    // Step 2 — invoke the REAL Governance Agent's lightweight explain_scenario mode
    // (SigV4 via Identity Pool) to explain the REAL decision computed above. The
    // result feeds the persistent verdict card, not an extra pipeline stage.
    const unavailable = agentUnavailableReason("governance");
    if (unavailable) return;

    let agentResp;
    try {
      agentResp = await invokeGovernance({
        mode: "explain_scenario",
        decision: decision.recommendation,
        metrics: decision,
        model_type: scenario.model_type,
        scenario: scenario.key,
      });
    } catch (err) {
      setRunError(err.message || String(err));
      return;
    }

    if (agentResp?.status === "error") {
      setRunError(agentResp.error || "Agent error");
      return;
    }

    const rationale = agentResp?.rationale || "";
    setLastRationale(rationale);
    setSessionAudit((prev) => [
      {
        timestamp: new Date().toISOString().slice(0, 16).replace("T", " ") + " UTC",
        model: scenario.model_type,
        decision: (decision.recommendation || "").toUpperCase(),
        rationale: rationale
          ? rationale.replace(/[*`_]/g, "").slice(0, 200)
          : `p=${fmt(decision.p_value, 4)}, lift=${fmt(decision.relative_lift)}`,
      },
      ...prev,
    ]);
  }

  return (
    <div className="cl-page">
      <div className="cl-page-header">
        <h2>Model Governance — Bid Request Mutation Outcomes</h2>
        <p className="cl-page-desc">
          Each model in the pipeline drives a specific ARTF mutation on the bidstream. Governance decisions (promote, reject, extend) directly affect how bids are priced, which deals are activated, and what metrics are attached.
        </p>
      </div>

      {/* Pipeline bar — icon steps at the TOP of the page, matching the
          prototype's layout order. Always visible; reflects real state once
          a scenario runs (default = all "done"/grey before any run). */}
      <PipelineBar
        nodes={nodes.length > 0 ? nodes : DEFAULT_PIPELINE_NODES}
        revealed={nodes.length > 0 ? revealed : DEFAULT_PIPELINE_NODES.length}
        running={running}
      />

      {/* Controls — simple single row: model select first (drives which
          scenarios are selectable), scenario select, run button. */}
      <div className="cl-controls-bar">
        <div className="cl-control-group">
          <label htmlFor="cl-gov-model">Model:</label>
          <select
            id="cl-gov-model"
            className="cl-select sg-interactive"
            value={modelType}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={running}
          >
            {MODEL_TYPES.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>
        <div className="cl-control-group">
          <label htmlFor="cl-gov-scenario">Scenario:</label>
          <select
            id="cl-gov-scenario"
            className="cl-select sg-interactive"
            value={selected || ""}
            onChange={(e) => setSelected(e.target.value)}
            disabled={running}
          >
            {scenariosForModel.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </div>
        <button className="btn btn-primary sg-interactive" onClick={runScenario} disabled={!selected || running}>
          {running ? <><span className="spinner" /> Running…</> : "Run governance scenario"}
        </button>
        {selected && <span className="cl-run-status">Selected: {selected}</span>}
      </div>

      {/* Consolidated: selected scenario's detail next to only the selected
          model's mutation-intent card, side by side (not a grid of every
          scenario/model at once). */}
      <div className="cl-mutation-grid">
        <ScenarioDetailCard
          s={scenariosForModel.find((s) => s.key === selected)}
          onViewSamples={setSamplesScenario}
        />
        <MutationIntentCard
          modelKey={modelType}
          decision={lastDecision}
          agentModelType={lastModelType}
        />
      </div>

      {/* Governance result — verdict card + bidstream impact, matching
          DESIGN_BRIEF.md section 4. */}
      <div className="cl-mutation-grid">
        <GovernanceVerdictCard
          decision={lastDecision}
          rationale={lastRationale}
          unavailableReason={lastDecision ? agentUnavailableReason("governance") : null}
        />
        <BidstreamImpactCard modelType={lastModelType || modelType} />
      </div>

      {runError && <div className="cl-honest cl-honest-block">Run failed: {runError}</div>}

      {/* Step-by-step governance flow — simple text log per stage, replacing
          the bulkier per-stage detail cards. */}
      <StepByStepLog nodes={nodes} revealed={revealed} />

      {/* Session audit trail — real decisions from this browser session, matching
          DESIGN_BRIEF.md section 6 (never a write into the real audit table). */}
      <div className="cl-card sg-elevated">
        <div className="cl-card-head">
          <span className="cl-card-title">Session decision history</span>
        </div>
        <SessionAuditTrail entries={sessionAudit} />
      </div>

      <div className="cl-section-title">Model registry</div>
      <div className="info-bar" style={{ margin: "0 0 12px" }}>
        <span>Model registry versions</span>
        <button className="btn-secondary sg-interactive" onClick={refreshState} style={{ padding: "5px 12px" }}>Refresh</button>
      </div>
      <ModelsView versions={models} error={stateError.models} />

      {samplesScenario && (
        <SampleOutcomesModal scenario={samplesScenario} onClose={() => setSamplesScenario(null)} />
      )}
    </div>
  );
}
