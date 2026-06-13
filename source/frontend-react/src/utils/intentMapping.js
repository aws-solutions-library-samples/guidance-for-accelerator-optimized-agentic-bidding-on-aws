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

export const INTENT_TO_STOP = Object.freeze({
  0: "metrics",
  1: "widedeep",
  2: "ncf",
  3: "ncf",
  4: "ncf",
  5: "ncf",
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
  dsp: "NONE",
});

export const CONTAINER_NAME_TO_STOP_ID = Object.freeze({
  "dlrm-bid-shader": "dlrm",
  "widedeep-segment-activator": "widedeep",
  "ncf-deal-manager": "ncf",
  "metrics-enricher": "metrics",
});

export const DISPLAY_NAME_BY_STOP_ID = Object.freeze({
  ssp: "SSP",
  dlrm: "DLRM Bid Shader",
  widedeep: "Wide & Deep",
  ncf: "NCF Deal Mgr",
  metrics: "Metrics Enricher",
  dsp: "DSP",
});
