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

const CONTAINER_STOP_IDS = Object.freeze(["dlrm", "widedeep", "ncf", "metrics"]);

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
  };
}

function containerEntryToStop(entry) {
  const stopId = CONTAINER_NAME_TO_STOP_ID[entry?.name];
  if (!stopId) return null;
  return {
    id: stopId,
    displayName: DISPLAY_NAME_BY_STOP_ID[stopId],
    modelFamily: STOP_MODEL_FAMILY[stopId],
    status: typeof entry.status === "string" ? entry.status : "unknown",
    latency: isFiniteNumber(entry.latency_ms)
      ? { ms: entry.latency_ms, source: "orchestrator" }
      : null,
    mutations: Array.isArray(entry.mutations)
      ? entry.mutations.map(toMutationModel)
      : [],
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
  for (const entry of containers) {
    const stop = containerEntryToStop(entry);
    if (!stop) continue;
    if (!byId[stop.id]) byId[stop.id] = stop;
  }
  return CONTAINER_STOP_IDS.map((id) => byId[id] ?? makePlaceholderContainerStop(id));
}

function buildInferredContainerStops(mutations) {
  const buckets = { dlrm: [], widedeep: [], ncf: [], metrics: [] };
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
        makePlaceholderContainerStop("dlrm"),
        makePlaceholderContainerStop("widedeep"),
        makePlaceholderContainerStop("ncf"),
        makePlaceholderContainerStop("metrics"),
        makeDspStop(),
      ],
      packet: summarizePacket(safePayload, raw),
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
    packet: summarizePacket(safePayload, raw),
    attribution: hasExplicitContainers ? "explicit" : "inferred",
    lifecycle,
    isResponseLifecycle,
    error: null,
    raw,
    submittedPayload: safePayload,
  };
}
