// intentMapping.js — canonical intent-to-container and stop-metadata tables
// Ported from the vanilla frontend's intent-mapping.js

export const INTENT_NAMES = Object.freeze({
  0: "UNSPECIFIED",
  1: "ACTIVATE_SEGMENTS",
  2: "ACTIVATE_DEALS",
  3: "SUPPRESS_DEALS",
  4: "ADJUST_DEAL_FLOOR",
  5: "ADJUST_DEAL_MARGIN",
  6: "BID_SHADE",
  7: "ADD_METRICS",
  8: "ADD_CIDS",
});

export const OP_NAMES = Object.freeze({
  0: "UNSPECIFIED",
  1: "ADD",
  2: "REMOVE",
  3: "REPLACE",
});

// ADJUST_DEAL_FLOOR (4) and ADJUST_DEAL_MARGIN (5) map to SEPARATE stops
// because they are served by two separate containers. They shared a single
// "yield" stop when one container served both; keeping that would make
// normalizer.js's buildExplicitContainerStops() collapse the two containers
// into one stop and silently discard the second one's status, latency and
// mutations (it keeps the first entry per stop id).
export const INTENT_TO_STOP = Object.freeze({
  0: "metrics",
  1: "widedeep",
  2: "ncf",
  3: "ncf",
  4: "yield-floor",
  5: "yield-margin",
  6: "dlrm",
  7: "metrics",
  8: "metrics",
});

export const STOP_MODEL_FAMILY = Object.freeze({
  ssp: "NONE",
  dlrm: "DLRM",
  widedeep: "WIDE_AND_DEEP",
  ncf: "NCF",
  metrics: "RULES",
  "yield-floor": "XGBOOST_FIL",
  "yield-margin": "XGBOOST_FIL",
  dsp: "NONE",
});

export const CONTAINER_NAME_TO_STOP_ID = Object.freeze({
  "dlrm-bid-shader": "dlrm",
  "widedeep-segment-activator": "widedeep",
  "ncf-deal-manager": "ncf",
  "metrics-enricher": "metrics",
  "yield-optimizer-floor": "yield-floor",
  "yield-optimizer-margin": "yield-margin",
});

// Job-oriented container labels, matching RawPanel.jsx's AGENT_LABELS,
// ContainersPanel.jsx's and LoadTestPanel.jsx's CONTAINER_LABELS, and
// RENAME_MAP.md. These four previously carried model-architecture names
// (DLRM/Wide & Deep/NCF/Metrics), which named the model rather than the job the
// container does and disagreed with every other lookup in the app. The model
// architectures are still documented per container; they are not the container's
// name. Stop ids themselves are unchanged internal keys.
export const DISPLAY_NAME_BY_STOP_ID = Object.freeze({
  ssp: "SSP",
  dlrm: "Bid Pricer",
  widedeep: "Audience Activator",
  ncf: "Deal Scorer",
  metrics: "Signals Enricher",
  "yield-floor": "Yield Optimizer — Floor",
  "yield-margin": "Yield Optimizer — Margin",
  dsp: "DSP",
});
