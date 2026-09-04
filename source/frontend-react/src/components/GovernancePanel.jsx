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
import LoadTestSweepStatus from "./LoadTestSweepStatus.jsx";
import { COMPARABLE_MODEL_TYPES, isFilModel } from "../utils/comparableModels.js";

// Default pipeline stages shown (all "done"/grey) before any scenario has run,
// matching the prototype's always-visible pipeline bar at the top of the page.
const DEFAULT_PIPELINE_NODES = [
  { id: "reg", label: "Registry", service: "SageMaker", icon: IconRegistry, state: "event" },
  { id: "opt", label: "Optimize", service: "TensorRT FP16", icon: IconChip, state: "event" },
  { id: "canary", label: "Canary", service: "Triton router", icon: IconSplit, state: "event" },
  { id: "ab", label: "A/B Eval", service: "Welch t + SPRT", icon: IconGavel, state: "event" },
  { id: "audit", label: "Audit", service: "DynamoDB / Bedrock", icon: IconAudit, state: "event" },
];

// The Governance panel's model selector is the shared comparison/canary model
// list (COMPARABLE_MODEL_TYPES): Bid Pricer (DLRM) + the two Yield Optimizer
// sub-models, with friendly names. This single selector drives the compare
// card, the model registry view, and the scenario testing harness. NCF was
// removed here (it is parked for training/canary); the yield models were added
// so they are comparable. The scenario harness only has scenarios for the
// canary-routed models, so yield selections show an honest empty scenario list.
const MODEL_TYPES = COMPARABLE_MODEL_TYPES;

// Training-specific model selector for the "Train from load test" card,
// decoupled from MODEL_TYPES above (which drives the scenario testing
// harness — that works fine for NCF). Three real trainable model types
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
  // Health of the Glue job that labels this model's training data, plus the runs
  // it has captured but not yet swept. Without these the picker cannot explain
  // itself: "nothing listed" looks identical whether no outcomes were ever
  // captured, a run is waiting on the next sweep, or the job fails every run.
  const [trainEtl, setTrainEtl] = useState(null);
  const [pendingRuns, setPendingRuns] = useState([]);

  // Load-test-based comparison (FR-7/FR-8, Story 5) and Promote (FR-9/FR-10, Story 6).
  const [currentRuns, setCurrentRuns] = useState([]);
  const [challengerRuns, setChallengerRuns] = useState([]);
  const [selectedCurrentRun, setSelectedCurrentRun] = useState("");
  const [selectedChallengerRun, setSelectedChallengerRun] = useState("");
  const [eligibleRunsError, setEligibleRunsError] = useState(null);
  // Where the preselected control came from — "training_provenance" (the run the
  // challenger was trained from) or "most_recent" (a weaker fallback). Shown so
  // the baseline is never implied to be the training data when it is not.
  const [controlSource, setControlSource] = useState("none");
  const [controlProvenance, setControlProvenance] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [comparisonError, setComparisonError] = useState(null);
  const [promoting, setPromoting] = useState(false);
  const [promotionResult, setPromotionResult] = useState(null);
  const [promotionError, setPromotionError] = useState(null);

  // One-click "compare current vs. retrained". Yield: stage the latest
  // registered version as the FIL canary (no compile). DLRM: the governance
  // agent already staged the retrained version as the canary when its
  // load-test-triggered training finished, so nothing is staged here. Then run
  // a real challenger load test at the control run's scenario+preset (seed is
  // fixed at 42 UI-wide, so control and challenger share the request stream)
  // and compare. Every step is gated on a real poll — no fabricated progress.
  const [retraining, setRetraining] = useState(false);
  const [retrainStep, setRetrainStep] = useState("");
  const [retrainError, setRetrainError] = useState(null);

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
        setTrainEtl(data.etl || null);
        setPendingRuns(data.pending || []);
        // This runs on an interval as well as on model change (see below), so
        // it keeps an existing selection when that run is still listed rather
        // than snapping back to the newest run under the user every refresh.
        setSelectedTrainingRun((cur) => (
          cur && runs.some((r) => r.id === cur) ? cur : (runs[0]?.id || "")
        ));
      } else {
        setTrainableRuns([]);
        setSelectedTrainingRun("");
        setTrainEtl(null);
        setPendingRuns([]);
        setTrainableRunsError(data.error || `HTTP ${resp.status}`);
      }
    } catch (e) {
      setTrainableRuns([]);
      setSelectedTrainingRun("");
      setTrainEtl(null);
      setPendingRuns([]);
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

  // A run becomes trainable when a Glue ETL sweep covering it succeeds, which
  // happens minutes after the load test ends and independently of this panel.
  // Without a refresh the picker stays stale until the browser is reloaded.
  useEffect(() => {
    const timer = setInterval(fetchTrainableRuns, 30000);
    return () => clearInterval(timer);
  }, [fetchTrainableRuns]);

  const startTraining = useCallback(async () => {
    setTrainingSubmitting(true);
    setTrainingError(null);
    try {
      const resp = await authFetch("/api/v1/governance/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // run_id is the run selected in the picker. It was collected in state but
        // never sent, so the resulting model version had no link back to the run
        // it was triggered from — which is what a later comparison needs to pick
        // its control.
        body: JSON.stringify({
          model_type: trainingModelType,
          confirmed: true,
          run_id: selectedTrainingRun,
        }),
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
        setSelectedChallengerRun(chalData.most_recent?.id || "");

        // Prefer the run the challenger was actually trained from over simply
        // the newest eligible run. That is the comparison worth making: did
        // training on this data beat what that data itself produced? The
        // endpoint reports which source it used so the label below stays honest
        // — it is a suggestion, and either side can still be overridden.
        let control = curData.most_recent?.id || "";
        let source = control ? "most_recent" : "none";
        try {
          const pairResp = await authFetch(
            `/api/v1/governance/comparison-pair?model_type=${modelType}`
          );
          if (pairResp.ok) {
            const pair = await pairResp.json();
            if (pair.control_run_id) {
              control = pair.control_run_id;
              source = pair.control_source || source;
            }
            setControlProvenance(pair);
          }
        } catch {
          // Suggestion is optional; the manual pickers still work without it.
          setControlProvenance(null);
        }
        setSelectedCurrentRun(control);
        setControlSource(source);
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

  const compareRetrained = useCallback(async () => {
    setRetrainError(null);
    setComparisonError(null);
    setComparisonResult(null);
    setPromotionResult(null);
    setPromotionError(null);

    // The current (control) run is the baseline we replay the challenger
    // against; its scenario+preset drive the challenger run so the two are
    // comparable on the same deterministic request stream.
    const controlRunId = selectedCurrentRun;
    const controlRun = currentRuns.find((r) => r.id === controlRunId);
    if (!controlRunId || !controlRun) {
      setRetrainError("Select a current-version run to compare the retrained model against.");
      return;
    }

    setRetraining(true);
    try {
      // 1. Stage the challenger canary. Yield (FIL) stages the latest registered
      //    version directly (no TensorRT compile). DLRM is already staged by the
      //    governance agent on its load-test-triggered training completion.
      if (isFilModel(modelType)) {
        const latest = (models || [])[0];
        if (!latest || !latest.model_package_arn) {
          setRetrainError("No registered version is available to stage as the challenger.");
          return;
        }
        setRetrainStep("Staging challenger canary…");
        const stageResp = await authFetch("/api/v1/governance/stage-canary", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_type: modelType, version_arn: latest.model_package_arn }),
        });
        if (!stageResp.ok) {
          const d = await stageResp.json().catch(() => ({}));
          setRetrainError(d.error || `Could not stage the challenger canary (HTTP ${stageResp.status}).`);
          return;
        }
      }

      // 2. Run the challenger load test against the canary at the control run's
      //    scenario+preset.
      setRetrainStep("Running challenger load test…");
      const startResp = await authFetch("/api/v1/loadtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset: controlRun.preset || "1k",
          seed: 42,
          duration_s: 30,
          scenario: controlRun.scenario || "baseline",
          target_model_type: modelType,
          target_variant: "challenger",
        }),
      });
      if (startResp.status === 409) {
        setRetrainError("A load test is already running. Wait for it to finish, then try again.");
        return;
      }
      if (!startResp.ok) {
        const d = await startResp.json().catch(() => ({}));
        // 422 no_canary_staged means the DLRM retrain is still compiling into the
        // canary slot (or a load-test-triggered training hasn't run yet).
        setRetrainError(
          d.error || `Could not start the challenger run (HTTP ${startResp.status}).`
        );
        return;
      }
      const { id: challengerRunId } = await startResp.json();

      // 3. Poll until the challenger run reaches a terminal state.
      setRetrainStep("Running challenger load test…");
      let terminal = null;
      for (let i = 0; i < 180; i++) {
        await new Promise((res) => setTimeout(res, 1000));
        const pollResp = await authFetch(`/api/v1/loadtest/${challengerRunId}`);
        if (!pollResp.ok) continue;
        const status = await pollResp.json();
        if (status.state && status.state !== "running") { terminal = status; break; }
      }
      if (!terminal) {
        setRetrainError("The challenger run did not finish in time — check the Load Test panel.");
        return;
      }
      if (terminal.state !== "complete") {
        setRetrainError(`The challenger run ended with state "${terminal.state}".`);
        return;
      }

      // 4. Compare the fresh challenger run against the control run.
      setRetrainStep("Comparing…");
      setSelectedChallengerRun(challengerRunId);
      const cmpResp = await authFetch("/api/v1/governance/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_type: modelType,
          current_run_id: controlRunId,
          challenger_run_id: challengerRunId,
        }),
      });
      const cmpData = await cmpResp.json();
      if (cmpResp.ok) setComparisonResult(cmpData);
      else setComparisonError(cmpData.error || `HTTP ${cmpResp.status}`);
    } catch (e) {
      setRetrainError(String(e));
    } finally {
      setRetrainStep("");
      setRetraining(false);
    }
  }, [modelType, selectedCurrentRun, currentRuns, models]);

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

      {/* Load-test outcome pipeline and Train-from-load-test sit side by side:
          the pipeline card shows which runs a Glue ETL sweep has moved into the
          training bucket, and the training card trains from one of them. Both
          are compact forms; collapses to one column under 980px. The step log,
          session history and model registry below stay full-width. */}
      <div className="cl-side-by-side">
      <section className="cl-side-by-side-col">
      <div className="cl-section-title">Load test outcome pipeline</div>
      <LoadTestSweepStatus onRunBecameTrainable={fetchTrainableRuns} />
      </section>

      {/* Train from load test (FR-4/FR-5, Story 3): real cost/duration
          estimate + explicit confirmation + concurrency-guarded trigger. Its
          own model selector (decoupled from the general one) since
          ncf_deal_manager training is parked. */}
      <section className="cl-side-by-side-col">
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
        {/* Timestamps are shown on every run above precisely so a run can be
            told apart from an older one. A run only becomes selectable after a
            Glue ETL pass has swept its outcomes into the training bucket, so
            the newest run in this list is NOT necessarily the load test you
            just finished. */}
        <div className="cl-train-latency-note" data-testid="governance-train-latency-note">
          Only runs already swept into the training bucket by a Glue ETL job are
          listed here, so a load test you just finished will be missing for a few
          minutes. The Load test outcome pipeline card above shows which stage
          each run is on. This list refreshes on its own — check the timestamp on
          the run you select so you are not retraining on data from an earlier
          load test.
        </div>
        {trainableRunsError && (
          <div className="cl-honest cl-honest-block">Could not load trainable runs: {trainableRunsError}</div>
        )}
        {/* The Glue job is what turns captured outcomes into training data, so
            its health decides whether waiting will ever help. A job that fails
            every run is not a "wait a few minutes" situation, and the picker
            previously looked identical in both cases. */}
        {trainEtl && trainEtl.configured && trainEtl.never_succeeded && (
          <div className="cl-honest cl-honest-block" data-testid="governance-train-etl-broken">
            <strong>The ETL job for {trainingModelType} has never completed successfully.</strong>{" "}
            No run of <code>{trainEtl.job_name}</code> has ever succeeded
            {trainEtl.consecutive_failures > 0
              ? ` (${trainEtl.consecutive_failures} consecutive failures)`
              : ""}
            , so no training data exists for this model and waiting will not
            change that.
            {trainEtl.last_error?.message && (
              <div
                className="cl-etl-error-detail"
                title={trainEtl.last_error.message}
                data-testid="governance-train-etl-error"
              >
                Latest failure: {trainEtl.last_error.message.slice(0, 140)}
                {trainEtl.last_error.message.length > 140 ? "..." : ""}
              </div>
            )}
          </div>
        )}
        {trainEtl && !trainEtl.configured && (
          <div className="cl-honest cl-honest-block">
            No Glue ETL job is configured for {trainingModelType}, so its load
            test outcomes are never labelled into training data.
          </div>
        )}
        {/* Recorded outcomes that simply have not been swept yet. This is the
            usual reason the run you just finished is not in the list, and it is
            a different situation from a broken job. */}
        {pendingRuns.length > 0 && (
          <div className="cl-honest cl-honest-block" data-testid="governance-train-pending">
            {pendingRuns.length} run{pendingRuns.length === 1 ? "" : "s"} for{" "}
            {trainingModelType} captured outcomes but{" "}
            {trainEtl?.never_succeeded
              ? "cannot be swept until that job succeeds"
              : "are waiting for the next ETL sweep"}
            {trainEtl?.last_success
              ? ` (last successful sweep ${new Date(trainEtl.last_success).toLocaleString()})`
              : ""}
            . Newest pending:{" "}
            {new Date(pendingRuns[0].timestamp).toLocaleString()} (
            {pendingRuns[0].outcome_sample_count} samples).
          </div>
        )}
        {trainableRuns.length === 0 && !trainableRunsError && !trainEtl?.never_succeeded && pendingRuns.length === 0 && (
          <div className="cl-honest cl-honest-block">
            No load test run for {trainingModelType} has captured outcomes yet
            — run a load test targeting this model, then wait for the next Glue
            sweep before training.
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
          (FR-9/FR-10, Story 6) — sits directly under "Train from load test" in
          the right column, so both fit beside the pipeline card on the left.
          Every result below is labeled by source — load-test-derived, distinct
          from the automated pipeline's live-canary-CloudWatch-derived decisions. */}
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
            {/* Says which baseline this actually is. Calling a most-recent run
                "the training data" when it is not would misstate what the
                comparison measured. */}
            {controlSource === "training_provenance" && (
              <div className="cl-control-note" data-testid="governance-control-provenance">
                Preselected the run this challenger was trained from.
              </div>
            )}
            {controlSource === "most_recent" && (
              <div className="cl-control-note" data-testid="governance-control-fallback">
                Most recent eligible run — not necessarily the data the challenger
                was trained from.
                {controlProvenance?.provenance_run_unavailable && (
                  <> Its training run ({controlProvenance.provenance_run_id}) is no
                  longer available as a control.</>
                )}
              </div>
            )}
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
            disabled={!selectedCurrentRun || !selectedChallengerRun || comparing || retraining}
          >
            {comparing ? <><span className="spinner" /> Comparing…</> : "Compare & Assess"}
          </button>
          {/* One-click: stage the retrained model as the challenger canary
              (yield) or use the agent-staged DLRM canary, run a challenger load
              test at the current run's scenario+preset, then compare. */}
          <button
            className="btn btn-secondary sg-interactive"
            data-testid="governance-compare-retrained-button"
            onClick={compareRetrained}
            disabled={!selectedCurrentRun || comparing || retraining}
            title="Run the retrained model against the selected current run and compare"
          >
            {retraining ? <><span className="spinner" /> {retrainStep || "Working…"}</> : "Compare current vs. retrained"}
          </button>
        </div>

        {retraining && retrainStep && (
          <div className="cl-control-note" data-testid="governance-retrain-progress">
            {retrainStep}
          </div>
        )}
        {retrainError && (
          <div className="cl-honest cl-honest-block" data-testid="governance-retrain-error">{retrainError}</div>
        )}

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

      </section>
      </div>

      <div className="cl-section-title">Model registry</div>
      <div className="info-bar" style={{ margin: "0 0 12px" }}>
        <span>Model registry versions</span>
        <button className="btn-secondary sg-interactive" onClick={refreshState} style={{ padding: "5px 12px" }}>Refresh</button>
      </div>
      <ModelsView versions={models} error={stateError.models} />

      {/* Governance Agent Testing — the synthetic A/B scenario harness. Demoted
          below the operational cards above and collapsed by default: this runs
          the REAL promotion gate (Welch's t-test + SPRT) and the deployed
          governance agent's explanation against a chosen A/B scenario. It is a
          decision demo and does NOT send traffic through the pipeline — the
          Load Test panel's traffic scenarios do that. The scenario selector
          lives here (rather than as a top-level control) because it only drives
          this testing harness. */}
      <details className="cl-collapsible" data-testid="governance-agent-testing">
        <summary className="cl-collapsible-summary">Governance Agent Testing</summary>
        <div className="cl-collapsible-body">
          <p className="cl-collapsible-note">
            Runs the real A/B promotion gate (Welch&apos;s t-test + SPRT) and the
            deployed governance agent&apos;s explanation against a synthetic A/B
            scenario. This is a decision demo — it does not send traffic through
            the pipeline. To shape synthetic <em>traffic</em>, use the Load Test
            panel&apos;s traffic scenarios.
          </p>

          {/* Pipeline bar — reflects real state once a scenario runs (default =
              all "done"/grey before any run). */}
          <PipelineBar
            nodes={nodes.length > 0 ? nodes : DEFAULT_PIPELINE_NODES}
            revealed={nodes.length > 0 ? revealed : DEFAULT_PIPELINE_NODES.length}
            running={running}
          />

          {/* Controls — model select first (drives which scenarios are
              selectable), scenario select, run button. */}
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

          {/* Selected scenario's detail next to the selected model's
              mutation-intent card. */}
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

          {/* Governance result — verdict card + bidstream impact. */}
          <div className="cl-mutation-grid">
            <GovernanceVerdictCard
              decision={lastDecision}
              rationale={lastRationale}
              unavailableReason={lastDecision ? agentUnavailableReason("governance") : null}
            />
            <BidstreamImpactCard modelType={lastModelType || modelType} />
          </div>

          {runError && <div className="cl-honest cl-honest-block">Run failed: {runError}</div>}

          {/* Step log + session decision history. */}
          <div className="cl-side-by-side">
            <StepByStepLog nodes={nodes} revealed={revealed} />

            <div className="cl-card sg-elevated">
              <div className="cl-card-head">
                <span className="cl-card-title">Session decision history</span>
              </div>
              <SessionAuditTrail entries={sessionAudit} />
            </div>
          </div>
        </div>
      </details>

      {samplesScenario && (
        <SampleOutcomesModal scenario={samplesScenario} onClose={() => setSamplesScenario(null)} />
      )}
    </div>
  );
}
