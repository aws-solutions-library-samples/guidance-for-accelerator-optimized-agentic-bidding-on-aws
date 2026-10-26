// Models that support the current-vs-retrained comparison flow (canary-capable
// + trainable). Job-oriented friendly names with the base model in parens.
//
// NCF is intentionally excluded: it is parked for training/canary (its
// ACTIVATE_DEALS/SUPPRESS_DEALS outcomes have no deal_id attribution yet — see
// GovernancePanel TRAINING_MODEL_TYPES). This list is the single source of the
// Governance panel's model selector, so removing NCF here removes it from the
// model registry view and the scenario harness too — intended, since those
// three are the trainable/canary-capable models.
export const COMPARABLE_MODEL_TYPES = [
  { key: "dlrm_bid_shader", label: "Bid Pricer (DLRM)" },
  { key: "deal_yield_manager_floor", label: "Yield Optimizer — Floor (XGBoost)" },
  { key: "deal_yield_manager_margin", label: "Yield Optimizer — Margin (XGBoost)" },
];

// The two Yield Optimizer sub-models are served by Triton's FIL backend
// (native XGBoost, no TensorRT compile), so their challenger canary is staged
// on demand from the registered artifact via the orchestrator's
// POST /v1/governance/stage-canary endpoint. DLRM is TensorRT and is instead
// auto-staged by the governance agent when a load-test-triggered training
// completes (source/agents/governance/governance_agent.py), so the UI does not
// stage a DLRM canary itself.
export function isFilModel(modelType) {
  return (
    modelType === "deal_yield_manager_floor" ||
    modelType === "deal_yield_manager_margin"
  );
}

// Friendly label for a model_type key, falling back to the raw key so an
// unknown type is shown honestly rather than hidden.
export function comparableModelLabel(modelType) {
  const found = COMPARABLE_MODEL_TYPES.find((m) => m.key === modelType);
  return found ? found.label : modelType;
}
