// normalizer.js — pure transform from raw Orchestrator responses into
// NormalizedFlowResult. Ported from the vanilla frontend's flow-normalizer.js.

import {
  INTENT_NAMES,
  OP_NAMES,
  INTENT_TO_STOP,
  STOP_MODEL_FAMILY,
  CONTAINER_NAME_TO_STOP_ID,
  DISPLAY_NAME_BY_STOP_ID,
} from "./intentMapping.js";

// Two yield stops, one per container -- see intentMapping.js's INTENT_TO_STOP note.
const CONTAINER_STOP_IDS = Object.freeze([
  "dlrm", "widedeep", "ncf", "metrics", "yield-floor", "yield-margin",
]);

function isFiniteNumber(n) {
  return typeof n === "number" && Number.isFinite(n);
}

function resolveIntentName(code) {
  if (typeof code === "number" && INTENT_NAMES[code] !== undefined) {
    return INTENT_NAMES[code];
  }
  return `INTENT_${code}`;
}

function resolveOpName(code) {
  if (typeof code === "number" && OP_NAMES[code] !== undefined) {
    return OP_NAMES[code];
  }
  return `OP_${code}`;
}

export function toMutationModel(m) {
  const raw = m ?? {};
  const intentCode = raw.intent;
  const opCode = raw.op;

  let payload = null;
  if (raw.ids !== undefined && raw.ids !== null) payload = raw.ids;
  else if (raw.adjust_bid !== undefined && raw.adjust_bid !== null) payload = raw.adjust_bid;
  else if (raw.adjust_deal !== undefined && raw.adjust_deal !== null) payload = raw.adjust_deal;
  // ARTF standard field is `add_metrics`; tolerate the legacy `metrics` key so
  // the UI renders correctly across backend versions during the migration.
  else if (raw.add_metrics !== undefined && raw.add_metrics !== null) payload = raw.add_metrics;
  else if (raw.metrics !== undefined && raw.metrics !== null) payload = raw.metrics;

  return {
    intent: resolveIntentName(intentCode),
    intentCode: typeof intentCode === "number" ? intentCode : 0,
    op: resolveOpName(opCode),
    opCode: typeof opCode === "number" ? opCode : 0,
    path: typeof raw.path === "string" ? raw.path : "",
    payload,
    raw,
  };
}

function makeSspStop() {
  return {
    id: "ssp",
    displayName: DISPLAY_NAME_BY_STOP_ID.ssp,
    modelFamily: STOP_MODEL_FAMILY.ssp,
    status: "ok",
    latency: null,
    mutations: [],
  };
}

function makeDspStop() {
  return {
    id: "dsp",
    displayName: DISPLAY_NAME_BY_STOP_ID.dsp,
    modelFamily: STOP_MODEL_FAMILY.dsp,
    status: "ok",
    latency: null,
    mutations: [],
  };
}

function makePlaceholderContainerStop(id) {
  return {
    id,
    displayName: DISPLAY_NAME_BY_STOP_ID[id],
    modelFamily: STOP_MODEL_FAMILY[id],
    status: "unknown",
    latency: null,
    mutations: [],
    superseded: 0,
  };
}

// Prefix for stops synthesized from a container the frontend has no build-time
// entry for. Namespaced so it can never collide with one of the six fixed stop
// ids, and so downstream code can recognise a dynamic stop when it needs to.
export const DYNAMIC_STOP_PREFIX = "dynamic:";

export function isDynamicStopId(id) {
  return typeof id === "string" && id.startsWith(DYNAMIC_STOP_PREFIX);
}

function containerEntryToStop(entry) {
  const name = entry?.name;
  if (typeof name !== "string" || name === "") return null;

  const stopId = CONTAINER_NAME_TO_STOP_ID[name];
  const latency = isFiniteNumber(entry.latency_ms)
    ? { ms: entry.latency_ms, source: "orchestrator" }
    : null;
  const mutations = Array.isArray(entry.mutations)
    ? entry.mutations.map(toMutationModel)
    : [];
  const status = typeof entry.status === "string" ? entry.status : "unknown";

  // A container the frontend was not built with -- a store-defined one, e.g.
  // the ARTF template or a user's own container. This used to `return null`,
  // and buildExplicitContainerStops then dropped it, so such a container was
  // absent from the pipeline entirely rather than merely unlabelled. Synthesize
  // a stop instead, and take its label from the response since no build-time
  // lookup can know the name.
  // How many of this container's mutations lost a contest for a (path, intent)
  // to a higher-priority container. The mutations above still list them, because
  // the container really did compute them -- this count is what lets the UI say
  // "ran and was overridden" rather than showing a mutation that looks applied.
  const superseded = isFiniteNumber(entry.superseded) ? entry.superseded : 0;

  if (!stopId) {
    return {
      id: `${DYNAMIC_STOP_PREFIX}${name}`,
      // Falls back to the internal name, never to an invented label.
      displayName: (typeof entry.display_name === "string" && entry.display_name) || name,
      modelFamily: "CUSTOM",
      status,
      latency,
      mutations,
      superseded,
    };
  }

  return {
    id: stopId,
    // The response's display name wins so a renamed container shows its
    // configured name; the build-time table is the fallback.
    displayName:
      (typeof entry.display_name === "string" && entry.display_name) ||
      DISPLAY_NAME_BY_STOP_ID[stopId],
    modelFamily: STOP_MODEL_FAMILY[stopId],
    status,
    latency,
    mutations,
    superseded,
  };
}

export function summarizePacket(submittedPayload, raw) {
  const payload = submittedPayload ?? {};
  const br = payload.bid_request;
  const hasBidRequest = br && typeof br === "object";

  let impressionType = "unknown";
  const imp0 = hasBidRequest && Array.isArray(br.imp) ? br.imp[0] : undefined;
  if (imp0 && typeof imp0 === "object") {
    const kinds = [];
    if (imp0.banner) kinds.push("banner");
    if (imp0.video) kinds.push("video");
    if (imp0.audio) kinds.push("audio");
    if (imp0.native) kinds.push("native");
    if (kinds.length === 1) impressionType = kinds[0];
    else if (kinds.length > 1) impressionType = "mixed";
  }

  const siteDomain =
    hasBidRequest && br.site && typeof br.site.domain === "string"
      ? br.site.domain
      : null;
  const userId =
    hasBidRequest && br.user && typeof br.user.id === "string"
      ? br.user.id
      : null;
  const bidFloor = imp0 && isFiniteNumber(imp0.bidfloor) ? imp0.bidfloor : null;

  const payloadBid = payload?.bid_response?.seatbid?.[0]?.bid?.[0];
  const rawBid = raw?.bid_response?.seatbid?.[0]?.bid?.[0];
  const bid = payloadBid ?? rawBid ?? null;
  let bidResponse = null;
  if (bid && typeof bid === "object" && isFiniteNumber(bid.price)) {
    const creative =
      typeof bid.adm === "string" && bid.adm.length > 0
        ? bid.adm
        : typeof bid.crid === "string" && bid.crid.length > 0
          ? bid.crid
          : null;
    bidResponse = { price: bid.price, creative };
  }

  return { impressionType, siteDomain, userId, bidFloor, bidResponse };
}

const MODEL_VERSION_LATENCY_RE = /([\d.]+)\s*ms/;

export function extractTotalLatency(raw, browserObservedMs) {
  const metadata = raw?.metadata;
  if (metadata && isFiniteNumber(metadata.total_latency_ms) && metadata.total_latency_ms >= 0) {
    return metadata.total_latency_ms;
  }
  if (metadata && typeof metadata.model_version === "string") {
    const match = metadata.model_version.match(MODEL_VERSION_LATENCY_RE);
    if (match) {
      const parsed = Number.parseFloat(match[1]);
      if (Number.isFinite(parsed) && parsed >= 0) return parsed;
    }
  }
  if (isFiniteNumber(browserObservedMs) && browserObservedMs >= 0) return browserObservedMs;
  return 0;
}

function buildExplicitContainerStops(containers) {
  const byId = {};
  // Dynamic stops are collected separately and appended, so the six fixed stops
  // keep their positions and no existing layout moves when a store-defined
  // container appears or disappears.
  const dynamic = [];
  for (const entry of containers) {
    const stop = containerEntryToStop(entry);
    if (!stop) continue;
    if (isDynamicStopId(stop.id)) {
      if (!dynamic.some((s) => s.id === stop.id)) dynamic.push(stop);
      continue;
    }
    if (!byId[stop.id]) byId[stop.id] = stop;
  }
  const fixed = CONTAINER_STOP_IDS.map((id) => byId[id] ?? makePlaceholderContainerStop(id));
  // Registry order is the orchestrator's attribution order, so it is preserved
  // rather than re-sorted here.
  return [...fixed, ...dynamic];
}

function buildInferredContainerStops(mutations) {
  const buckets = {
    dlrm: [], widedeep: [], ncf: [], metrics: [],
    "yield-floor": [], "yield-margin": [],
  };
  for (const m of mutations) {
    const intentCode = m?.intent;
    const stopId = (typeof intentCode === "number" && INTENT_TO_STOP[intentCode]) || "metrics";
    buckets[stopId].push(toMutationModel(m));
  }
  return CONTAINER_STOP_IDS.map((id) => ({
    id,
    displayName: DISPLAY_NAME_BY_STOP_ID[id],
    modelFamily: STOP_MODEL_FAMILY[id],
    status: "unknown",
    latency: null,
    mutations: buckets[id],
    // The inferred path has no per-container metadata to read a count from, so
    // 0 here means "not known", not "nothing was overridden".
    superseded: 0,
  }));
}

/** A stable identity for one raw mutation, for matching the flattened list to a container's list. */
function mutationKey(raw) {
  return JSON.stringify([raw?.intent, raw?.op, raw?.path, raw?.ids, raw?.adjust_deal, raw?.adjust_bid, raw?.add_metrics, raw?.metrics]);
}

/**
 * The orchestrator's flattened mutation list, in its order, each stamped with the
 * stop that produced it.
 *
 * `raw.mutations` is already conflict-resolved and in application order; the
 * per-container lists on the stops say who produced what. A mutation that two
 * containers both produced byte-for-byte is attributed to the first in stop order,
 * which is the orchestrator's own tie-break. One that no stop lists (an inferred
 * attribution, or an older server) is attributed by intent.
 */
export function attributeOrderedMutations(raw, containerStops) {
  const list = Array.isArray(raw?.mutations) ? raw.mutations : [];
  const byKey = new Map();
  for (const stop of containerStops ?? []) {
    for (const m of stop.mutations ?? []) {
      const key = mutationKey(m.raw);
      if (!byKey.has(key)) byKey.set(key, stop.id);
    }
  }
  return list.map((m) => {
    const model = toMutationModel(m);
    const intentCode = m?.intent;
    const fallback = (typeof intentCode === "number" && INTENT_TO_STOP[intentCode]) || "metrics";
    return { ...model, sourceAgent: byKey.get(mutationKey(m)) ?? fallback };
  });
}

/** metadata.stages as the UI reads it. [] when absent. */
export function normalizeStages(rawStages) {
  if (!Array.isArray(rawStages)) return [];
  return rawStages
    .filter((s) => s && typeof s === "object")
    .map((s) => ({
      stage: isFiniteNumber(s.stage) ? s.stage : 0,
      name: typeof s.name === "string" ? s.name : "",
      containers: Array.isArray(s.containers) ? s.containers.filter((c) => typeof c === "string") : [],
      latencyMs: isFiniteNumber(s.latency_ms) ? s.latency_ms : null,
      budgetMs: isFiniteNumber(s.budget_ms) ? s.budget_ms : null,
      applied: isFiniteNumber(s.applied) ? s.applied : 0,
      rejected: Array.isArray(s.rejected) ? s.rejected : [],
    }));
}

/**
 * Normalize a raw Orchestrator response into the NormalizedFlowResult shape.
 */
export function normalize(raw, transport, submittedPayload) {
  const safePayload = submittedPayload ?? {};
  const lifecycle = safePayload?.lifecycle || raw?.lifecycle || "LIFECYCLE_SSP_BID_REQUEST";
  const isResponseLifecycle = lifecycle.includes("RESPONSE");

  const rpcError = raw && typeof raw === "object" ? raw.error : null;
  if (rpcError && typeof rpcError === "object") {
    const message = typeof rpcError.message === "string" ? rpcError.message : "Unknown error";
    return {
      id: typeof raw?.id === "string" ? raw.id : "",
      transport,
      totalLatencyMs: 0,
      stops: [
        makeSspStop(),
        ...CONTAINER_STOP_IDS.map((id) => makePlaceholderContainerStop(id)),
        makeDspStop(),
      ],
      packet: summarizePacket(safePayload, raw),
      mutations: [],
      stages: [],
      attribution: "inferred",
      lifecycle,
      isResponseLifecycle,
      error: { message, stage: "orchestrator" },
      raw,
      submittedPayload: safePayload,
    };
  }

  const hasExplicitContainers = raw && raw.metadata && Array.isArray(raw.metadata.containers);
  const containerStops = hasExplicitContainers
    ? buildExplicitContainerStops(raw.metadata.containers)
    : buildInferredContainerStops(Array.isArray(raw?.mutations) ? raw.mutations : []);

  const stops = [makeSspStop(), ...containerStops, makeDspStop()];

  return {
    id: typeof raw?.id === "string" ? raw.id : "",
    transport,
    totalLatencyMs: extractTotalLatency(raw),
    stops,
    // The orchestrator's returned list: stage-ordered and conflict-resolved, which
    // is the order the host applies. Each entry carries the stop id of the
    // container that produced it (resolved by matching the mutation against the
    // per-container lists; by intent when no container claims it). Stops above are
    // in display order and must not be used as an application order.
    mutations: attributeOrderedMutations(raw, containerStops),
    // The sequenced fan-out's own account: which containers ran in which stage
    // and how long each stage took. [] when the server did not report stages
    // (bypassed pass, or an orchestrator predating staging).
    stages: normalizeStages(raw?.metadata?.stages),
    packet: summarizePacket(safePayload, raw),
    attribution: hasExplicitContainers ? "explicit" : "inferred",
    // Contests for a (path, intent) claimed by more than one container. Always an
    // array: an orchestrator predating precedence omits the key, and for a
    // renderer "nothing to show" and "this build cannot tell you" both mean
    // render nothing.
    conflicts: Array.isArray(raw?.metadata?.conflicts) ? raw.metadata.conflicts : [],
    lifecycle,
    isResponseLifecycle,
    error: null,
    raw,
    submittedPayload: safePayload,
  };
}
