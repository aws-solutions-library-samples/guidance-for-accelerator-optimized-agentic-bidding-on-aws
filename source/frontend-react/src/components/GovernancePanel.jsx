import React, { useState, useEffect, useCallback, useRef } from "react";
import { authFetch } from "../authFetch.js";
import { invokeGovernance, agentUnavailableReason } from "../agentCoreClient.js";
import {
  fmt,
  IconRegistry, IconChip, IconSplit, IconGavel, IconAudit,
  ScenarioDetailCard, SampleOutcomesModal, PipelineBar, StepByStepLog,
  ModelsView, MutationIntentCard, BidstreamImpactCard,
  GovernanceVerdictCard, SessionAuditTrail, RecommendationBadge,
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
// NCF still has full scenario/compare/promote support (no training data
// needed there), so it stays in this general selector even though it's
// parked for the training section below (see TRAINING_MODEL_TYPES).
const MODEL_TYPES = [
  { key: "dlrm_bid_shader", label: "DLRM Bid Shader" },
  { key: "ncf_deal_manager", label: "NCF Deal Manager" },
];

// Training-specific model selector for the "Train from load test" card,
// decoupled from MODEL_TYPES above (which also drives scenario/compare/
// promote — those work fine for NCF). Three real trainable model types
// exist today (matches orchestrator.training_trigger.TRAINABLE_MODEL_TYPES,
// the authoritative backend enforcement): dlrm_bid_shader (NeMo-RL) and
// the Yield Optimizer's two independently-trained sub-models,
// deal_yield_manager_floor/margin (SageMaker built-in XGBoost — see
// training/xgboost_pipeline.py's module docstring on why Triton's FIL
// backend requires floor/margin to be trained as separate single-output
// models, not one combined "deal_yield_manager" model/target).
//
// ncf_deal_manager training is parked: its ACTIVATE_DEALS/SUPPRESS_DEALS
// mutations disambiguate deals via path + a list of deal IDs (verified
// against the real ARTF proto/reference implementation —
// github.com/IABTechLab/agentic-real-time-framework), but
// BidShadingOutcomeEvent/Record has no deal_id field and no per-deal
// fan-out, so there's no way to attribute a training outcome to one
// specific deal yet. Shown here, disabled, rather than removed, so it's
// discoverable and trivial to re-enable once that schema work lands.
const TRAINING_MODEL_TYPES = [
  { key: "dlrm_bid_shader", label: "DLRM Bid Shader", trainable: true },
  { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor", trainable: true },
  { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin", trainable: true },
  { key: "ncf_deal_manager", label: "NCF Deal Manager (parked — coming in a future release)", trainable: false },
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

  // Train-from-load-test (FR-4/FR-5, Story 3): cost/duration estimate shown
  // before confirming, and a live "is a job already running" check that
  // disables the button while true. Uses its own model selector
  // (trainingModelType), decoupled from the general `modelType` above —
  // ncf_deal_manager training is parked (see TRAINING_MODEL_TYPES), but
  // NCF scenario/compare/promote above are unaffected.
  const [trainingModelType, setTrainingModelType] = useState("dlrm_bid_shader");
  const [trainingEstimate, setTrainingEstimate] = useState(null);
  const [trainingEstimateError, setTrainingEstimateError] = useState(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [confirmingTraining, setConfirmingTraining] = useState(false);
  const [trainingSubmitting, setTrainingSubmitting] = useState(false);
  const [trainingResult, setTrainingResult] = useState(null);
  const [trainingError, setTrainingError] = useState(null);

  // Trainable load-test runs (this fix): only runs whose outcome data has
  // actually been swept into training-data/ by a completed Glue job run —
  // see GET /v1/governance/trainable-runs. Shown so the user can see which
  // load test (and its target model/test type) they're about to train from,
  // instead of just a bare "Train from load test" button with no run context.
  const [trainableRuns, setTrainableRuns] = useState([]);
  const [selectedTrainingRun, setSelectedTrainingRun] = useState("");
  const [trainableRunsError, setTrainableRunsError] = useState(null);

  // Load-test-based comparison (FR-7/FR-8, Story 5) and Promote (FR-9/FR-10, Story 6).
  const [currentRuns, setCurrentRuns] = useState([]);
  const [challengerRuns, setChallengerRuns] = useState([]);
  const [selectedCurrentRun, setSelectedCurrentRun] = useState("");
  const [selectedChallengerRun, setSelectedChallengerRun] = useState("");
  const [eligibleRunsError, setEligibleRunsError] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparisonError, setComparisonError] = useState(null);
  const [promoting, setPromoting] = useState(false);
  const [promotionResult, setPromotionResult] = useState(null);
  const [promotionError, setPromotionError] = useState(null);

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

  // Fetch the real cost/duration estimate whenever the selected training
  // model type changes, and reset any prior training-in-progress/result
  // state (it was scoped to the previous model type).
  const fetchTrainingEstimate = useCallback(async () => {
    setTrainingEstimateError(null);
    try {
      const resp = await authFetch(`/api/v1/governance/training-estimate?model_type=${trainingModelType}`);
      const data = await resp.json();
      if (resp.ok) setTrainingEstimate(data);
      else { setTrainingEstimate(null); setTrainingEstimateError(data.error || `HTTP ${resp.status}`); }
    } catch (e) {
      setTrainingEstimate(null);
      setTrainingEstimateError(String(e));
    }
  }, [trainingModelType]);

  const fetchTrainableRuns = useCallback(async () => {
    setTrainableRunsError(null);
    try {
      const resp = await authFetch(`/api/v1/governance/trainable-runs?model_type=${trainingModelType}`);
      const data = await resp.json();
      if (resp.ok) {
        const runs = data.runs || [];
        setTrainableRuns(runs);
        setSelectedTrainingRun(runs[0]?.id || "");
      } else {
        setTrainableRuns([]);
        setSelectedTrainingRun("");
        setTrainableRunsError(data.error || `HTTP ${resp.status}`);
      }
    } catch (e) {
      setTrainableRuns([]);
      setSelectedTrainingRun("");
      setTrainableRunsError(String(e));
    }
  }, [trainingModelType]);

  useEffect(() => {
    fetchTrainingEstimate();
    fetchTrainableRuns();
    setTrainingInProgress(false);
    setConfirmingTraining(false);
    setTrainingResult(null);
    setTrainingError(null);
  }, [trainingModelType, fetchTrainingEstimate, fetchTrainableRuns]);

  const startTraining = useCallback(async () => {
    setTrainingSubmitting(true);
    setTrainingError(null);
    try {
      const resp = await authFetch("/api/v1/governance/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: trainingModelType, confirmed: true }),
      });
      const data = await resp.json();
      if (resp.ok) {
        setTrainingResult(data);
        setConfirmingTraining(false);
      } else if (data.reason === "already_in_progress") {
        setTrainingInProgress(true);
        setConfirmingTraining(false);
        setTrainingError(data.error);
      } else {
        setTrainingError(data.error || `HTTP ${resp.status}`);
        setConfirmingTraining(false);
      }
    } catch (e) {
      setTrainingError(String(e));
      setConfirmingTraining(false);
    } finally {
      setTrainingSubmitting(false);
    }
  }, [trainingModelType]);

  // Fetch eligible load-test runs whenever the model type changes, auto-
  // selecting the most recent per role (FR-7). Clears any prior
  // comparison/promotion result — it was scoped to the previous model type.
  const fetchEligibleRuns = useCallback(async () => {
    setEligibleRunsError(null);
    try {
      const [curResp, chalResp] = await Promise.all([
        authFetch(`/api/v1/governance/eligible-runs?model_type=${modelType}&role=current`),
        authFetch(`/api/v1/governance/eligible-runs?model_type=${modelType}&role=challenger`),
      ]);
      const curData = await curResp.json();
      const chalData = await chalResp.json();
      if (curResp.ok && chalResp.ok) {
        setCurrentRuns(curData.runs || []);
        setChallengerRuns(chalData.runs || []);
        setSelectedCurrentRun(curData.most_recent?.id || "");
        setSelectedChallengerRun(chalData.most_recent?.id || "");
      } else {
        setEligibleRunsError(curData.error || chalData.error || "Could not load eligible runs.");
      }
    } catch (e) {
      setEligibleRunsError(String(e));
    }
  }, [modelType]);

  useEffect(() => {
    fetchEligibleRuns();
    setComparisonResult(null);
    setComparisonError(null);
    setPromotionResult(null);
    setPromotionError(null);
  }, [modelType, fetchEligibleRuns]);

  const compareRuns = useCallback(async () => {
    if (!selectedCurrentRun || !selectedChallengerRun) return;
    setComparing(true);
    setComparisonError(null);
    setPromotionResult(null);
    setPromotionError(null);
    try {
      const resp = await authFetch("/api/v1/governance/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_type: modelType,
          current_run_id: selectedCurrentRun,
          challenger_run_id: selectedChallengerRun,
        }),
      });
      const data = await resp.json();
      if (resp.ok) setComparisonResult(data);
      else { setComparisonResult(null); setComparisonError(data.error || `HTTP ${resp.status}`); }
    } catch (e) {
      setComparisonResult(null);
      setComparisonError(String(e));
    } finally {
      setComparing(false);
    }
  }, [modelType, selectedCurrentRun, selectedChallengerRun]);

  const promoteChallenger = useCallback(async () => {
    if (!comparisonResult || comparisonResult.recommendation !== "promote") return;
    setPromoting(true);
    setPromotionError(null);
    try {
      const resp = await authFetch("/api/v1/governance/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_type: modelType,
          version_arn: comparisonResult.challenger_run_model_version,
          recommendation: comparisonResult.recommendation,
          reason: `Load-test comparison: lift=${fmt(comparisonResult.relative_lift)}, p=${fmt(comparisonResult.p_value, 5)}`,
        }),
      });
      const data = await resp.json();
      if (resp.ok) setPromotionResult(data);
      else setPromotionError(data.error || `HTTP ${resp.status}`);
    } catch (e) {
      setPromotionError(String(e));
    } finally {
      setPromoting(false);
    }
  }, [modelType, comparisonResult]);

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

      {/* Train from load test (FR-4/FR-5, Story 3): real cost/duration
          estimate + explicit confirmation + concurrency-guarded trigger.
          Has its own model selector (decoupled from the general one above)
          since ncf_deal_manager training is parked while scenario/compare/
          promote still work for it. */}
      <div className="cl-section-title">Train from load test</div>
      <div className="cl-card sg-elevated" data-testid="governance-train-card">
        <div className="cl-control-group">
          <label htmlFor="cl-gov-train-model">Model:</label>
          <select
            id="cl-gov-train-model"
            className="cl-select sg-interactive"
            data-testid="governance-train-model-select"
            value={trainingModelType}
            onChange={(e) => setTrainingModelType(e.target.value)}
            disabled={trainingSubmitting}
          >
            {TRAINING_MODEL_TYPES.map((m) => (
              <option key={m.key} value={m.key} disabled={!m.trainable}>{m.label}</option>
            ))}
          </select>
        </div>
        <div className="cl-control-group">
          <label htmlFor="cl-gov-train-run">Load test run:</label>
          <select
            id="cl-gov-train-run"
            className="cl-select sg-interactive"
            data-testid="governance-train-run-select"
            value={selectedTrainingRun}
            onChange={(e) => setSelectedTrainingRun(e.target.value)}
            disabled={trainingSubmitting || trainableRuns.length === 0}
          >
            {trainableRuns.length === 0 ? (
              <option value="">No trainable load test runs yet</option>
            ) : (
              trainableRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} — {r.target_model_type} / {r.target_variant}
                  {r.timestamp ? ` (${new Date(r.timestamp).toLocaleString()})` : ""}
                </option>
              ))
            )}
          </select>
        </div>
        {trainableRunsError && (
          <div className="cl-honest cl-honest-block">Could not load trainable runs: {trainableRunsError}</div>
        )}
        {trainableRuns.length === 0 && !trainableRunsError && (
          <div className="cl-honest cl-honest-block">
            No load test run for {trainingModelType} has been processed by a completed Glue ETL job yet
            — run a load test targeting this model, then wait for the next Glue run (or trigger it
            manually) before training.
          </div>
        )}
        {trainingEstimateError && (
          <div className="cl-honest cl-honest-block">Cost estimate unavailable: {trainingEstimateError}</div>
        )}
        {trainingEstimate && (
          <div className="cl-train-estimate">
            <span>Instance: {trainingEstimate.instance_type}</span>
            <span>Rate: ${trainingEstimate.hourly_rate_usd.toFixed(3)}/hr</span>
            <span>Max duration: {Math.round(trainingEstimate.max_runtime_seconds / 60)} min</span>
            <span>Max cost: ${trainingEstimate.estimated_max_cost_usd.toFixed(2)}</span>
          </div>
        )}
        {trainingResult && (
          <div className="cl-train-result" data-testid="governance-train-result">
            Training started — job {trainingResult.job_name}, base version {trainingResult.base_model_version}.
          </div>
        )}
        {trainingError && !confirmingTraining && (
          <div className="cl-honest cl-honest-block">{trainingError}</div>
        )}
        {!confirmingTraining ? (
          <button
            className="btn btn-primary sg-interactive"
            data-testid="governance-train-trigger-button"
            onClick={() => setConfirmingTraining(true)}
            disabled={!trainingEstimate || !selectedTrainingRun || trainingInProgress || trainingSubmitting}
          >
            {trainingInProgress ? "Training in progress\u2026" : "Train from load test"}
          </button>
        ) : (
          <div className="cl-train-confirm">
            <span>
              Start a real SageMaker training job for {trainingModelType}, using run {selectedTrainingRun}?
              Estimated max cost ${trainingEstimate?.estimated_max_cost_usd.toFixed(2)}.
            </span>
            <button
              className="btn btn-primary sg-interactive"
              data-testid="governance-train-confirm-button"
              onClick={startTraining}
              disabled={trainingSubmitting}
            >
              {trainingSubmitting ? <><span className="spinner" /> Starting…</> : "Confirm"}
            </button>
            <button
              className="btn-secondary sg-interactive"
              data-testid="governance-train-cancel-button"
              onClick={() => setConfirmingTraining(false)}
              disabled={trainingSubmitting}
            >
              Cancel
            </button>
          </div>
        )}
      </div>

      {/* Compare load-test outcomes (FR-7/FR-8, Story 5) + Promote
          (FR-9/FR-10, Story 6). Every result below is labeled by source —
          load-test-derived, distinct from the automated pipeline's
          live-canary-CloudWatch-derived decisions above. */}
      <div className="cl-section-title">Compare load-test outcomes</div>
      <div className="cl-card sg-elevated" data-testid="governance-compare-card">
        {eligibleRunsError && (
          <div className="cl-honest cl-honest-block">Could not load eligible runs: {eligibleRunsError}</div>
        )}
        <div className="cl-controls-bar">
          <div className="cl-control-group">
            <label htmlFor="cl-gov-current-run">Current version run:</label>
            <select
              id="cl-gov-current-run"
              className="cl-select sg-interactive"
              data-testid="governance-current-run-select"
              value={selectedCurrentRun}
              onChange={(e) => setSelectedCurrentRun(e.target.value)}
            >
              <option value="">— select a run —</option>
              {currentRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} ({r.timestamp ? new Date(r.timestamp).toLocaleString() : "unknown time"})
                </option>
              ))}
            </select>
          </div>
          <div className="cl-control-group">
            <label htmlFor="cl-gov-challenger-run">Challenger version run:</label>
            <select
              id="cl-gov-challenger-run"
              className="cl-select sg-interactive"
              data-testid="governance-challenger-run-select"
              value={selectedChallengerRun}
              onChange={(e) => setSelectedChallengerRun(e.target.value)}
            >
              <option value="">— select a run —</option>
              {challengerRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} ({r.timestamp ? new Date(r.timestamp).toLocaleString() : "unknown time"})
                </option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary sg-interactive"
            data-testid="governance-compare-button"
            onClick={compareRuns}
            disabled={!selectedCurrentRun || !selectedChallengerRun || comparing}
          >
            {comparing ? <><span className="spinner" /> Comparing…</> : "Compare & Assess"}
          </button>
        </div>

        {comparisonError && (
          <div className="cl-honest cl-honest-block">{comparisonError}</div>
        )}

        {comparisonResult && (
          <div className="cl-compare-result" data-testid="governance-compare-result">
            <div className="cl-compare-source">
              Source: load test runs — current {comparisonResult.current_run_id}
              {comparisonResult.current_run_timestamp && ` (${new Date(comparisonResult.current_run_timestamp).toLocaleString()})`},
              {" "}challenger {comparisonResult.challenger_run_id}
              {comparisonResult.challenger_run_timestamp && ` (${new Date(comparisonResult.challenger_run_timestamp).toLocaleString()})`}
            </div>
            <div className="cl-train-estimate">
              <span>Current: {fmt(comparisonResult.control_metric)}</span>
              <span>Challenger: {fmt(comparisonResult.treatment_metric)}</span>
              <span>Lift: {fmt(comparisonResult.relative_lift)}</span>
              <span>p-value: {fmt(comparisonResult.p_value, 5)}</span>
              <span>n: {comparisonResult.samples_control}/{comparisonResult.samples_treatment}</span>
            </div>
            <RecommendationBadge rec={comparisonResult.recommendation} />

            {comparisonResult.recommendation === "promote" && !promotionResult && (
              <button
                className="btn btn-primary sg-interactive"
                data-testid="governance-promote-button"
                onClick={promoteChallenger}
                disabled={promoting}
              >
                {promoting ? <><span className="spinner" /> Promoting…</> : "Promote"}
              </button>
            )}
            {promotionResult && (
              <div className="cl-train-result" data-testid="governance-promote-result">
                Promoted — version {promotionResult.version_arn}, audit record {promotionResult.audit_record_id}.
              </div>
            )}
            {promotionError && (
              <div className="cl-honest cl-honest-block">{promotionError}</div>
            )}
          </div>
        )}
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
