// scenarioPayload.js — fetch a scenario's payload and apply the card's tuner values.
//
// Extracted from Sidebar.jsx because there are now TWO ways to run a scenario:
// `▶ Send` (the mutation timeline) and `Step through in Auction Theater`. Both must
// send the same bytes for the same tuner settings. Two copies of this is how they
// would drift, and the drift would be invisible — the Theater would narrate a run
// the timeline never made.

const AGE_YOB_MAP = [
  { yobMin: 2002, yobMax: 2008 },
  { yobMin: 1992, yobMax: 2001 },
  { yobMin: 1982, yobMax: 1991 },
  { yobMin: 1972, yobMax: 1981 },
  { yobMin: 1962, yobMax: 1971 },
  { yobMin: 1940, yobMax: 1961 },
];

/**
 * Apply a scenario card's tuner values to its payload.
 *
 * Every model parameter is gated on `scenario.controls` containing the control:
 * a value the card never showed must not travel, or a scenario would be running
 * with a parameter the reader had no way to see.
 */
export function applyTunerToPayload(payload, scenario, params = {}) {
  const patched = JSON.parse(JSON.stringify(payload));
  const br = patched.bid_request || patched;
  const imp0 = br?.imp?.[0];

  if (imp0 && params.bidFloor != null) {
    imp0.bidfloor = params.bidFloor;
  }

  if (br?.user && params.ageRange != null) {
    const range = AGE_YOB_MAP[params.ageRange] || AGE_YOB_MAP[1];
    br.user.yob = Math.round((range.yobMin + range.yobMax) / 2);
  }

  if (scenario.id === "video-deals" && imp0?.pmp?.deals && params.numDeals != null) {
    const original = imp0.pmp.deals;
    if (params.numDeals <= original.length) {
      imp0.pmp.deals = original.slice(0, params.numDeals);
    }
  }

  const modelParams = {};
  if (params.shadeFactor != null && scenario.controls?.includes("shadeFactor")) {
    modelParams.shade_factor = params.shadeFactor;
  }
  if (params.convValue != null && scenario.controls?.includes("convValue")) {
    modelParams.conversion_value = params.convValue;
  }
  if (params.segThreshold != null && scenario.controls?.includes("segThreshold")) {
    modelParams.segment_threshold = params.segThreshold;
  }
  if (params.explore != null && scenario.controls?.includes("explore")) {
    // Yield Optimizer's bounded exploration toggle (see
    // shared/yield_exploration.py's resolve_effective_epsilon) — explicit
    // True/False, read the same way every other demo-tunable parameter here is
    // (ext.model_params).
    modelParams.explore = params.explore;
  }
  if (Object.keys(modelParams).length > 0) {
    // Nonstandard signaling travels through the ARTF `ext` object, not as a
    // top-level field, per the spec's extension convention.
    patched.ext = { ...(patched.ext || {}), model_params: modelParams };
  }

  return patched;
}

/**
 * Fetch a scenario's payload from `public/samples/` and apply its tuner values.
 *
 * The sample is a static asset on the same origin, so plain fetch — not authFetch.
 * Cache-busted because a payload edited between deploys would otherwise be served
 * from the browser cache.
 */
export async function loadScenarioPayload(scenario, params = {}) {
  const resp = await fetch(`/samples/${scenario.file}?t=${Date.now()}`);
  if (!resp.ok) {
    throw new Error(`Could not load scenario ${scenario.file} (${resp.status})`);
  }
  const raw = await resp.json();
  return applyTunerToPayload(raw, scenario, params);
}
