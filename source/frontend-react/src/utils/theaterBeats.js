// theaterBeats.js — derive the Auction Theater's beat sequence from a real
// orchestrator response.
//
// Pure: no I/O, no clock, no randomness. Given the same submitted payload and
// normalized result it returns the same beats, which is what makes the stepper
// reproducible and this module property-testable.
//
// this implements (BR-1 through BR-12).

import { DISPLAY_NAME_BY_STOP_ID, CONTAINER_NAME_TO_STOP_ID } from "./intentMapping.js";
import { findOriginalDealBidfloor } from "./applyMutations.js";

/** Margin.calculation_type from the ARTF proto. */
const CALC_TYPE = Object.freeze({ 0: "CPM", 1: "PERCENT" });

const EXPLORE_SUFFIX = ":explore";

/** Beat kinds. */
export const BEAT_ORIGIN = "origin";
export const BEAT_CONTAINER = "container";
export const BEAT_RECAP = "recap";

/** Where attention travels for each beat kind (BR-20). */
const MOVEMENT_BY_KIND = Object.freeze({
  [BEAT_ORIGIN]: "sell-to-request",
  [BEAT_CONTAINER]: "none",
  [BEAT_RECAP]: "request-to-buy",
});

function isFiniteNumber(n) {
  return typeof n === "number" && Number.isFinite(n);
}

/**
 * True only when the container's real model_version carries the explore
 * suffix. Never inferred from the returned value: a value can look unusual
 * without exploration, and exploration can occur without it looking unusual
 * (BR-12).
 */
export function isExplored(modelVersion) {
  return typeof modelVersion === "string" && modelVersion.endsWith(EXPLORE_SUFFIX);
}

/** Parse the deal id out of an ARTF `/imp/{imp_id}/deals/{deal_id}` path. */
export function dealIdFromPath(path) {
  if (typeof path !== "string") return null;
  const match = /^\/imp\/([^/]+)\/deals\/([^/]+)$/.exec(path);
  return match ? match[2] : null;
}

/**
 * Map one mutation to its typed values (BR-6 through BR-10).
 *
 * An unrecognised payload yields an empty array. The caller still emits a beat
 * for it, so an intent we don't render yet degrades visibly rather than
 * silently (BR-5).
 */
export function mutationToValues(mutation, submittedPayload) {
  const intent = mutation?.intent;
  const payload = mutation?.payload;
  if (!payload) return [];

  if (intent === "ADD_METRICS") {
    const metrics = Array.isArray(payload.metric) ? payload.metric : [];
    return metrics
      .filter((m) => typeof m?.type === "string" && isFiniteNumber(m?.value))
      .map((m) => ({ kind: "metric", type: m.type, value: m.value }));
  }

  if (intent === "ACTIVATE_SEGMENTS" || intent === "ACTIVATE_DEALS" || intent === "SUPPRESS_DEALS") {
    const ids = Array.isArray(payload.id) ? payload.id : [];
    if (ids.length === 0) return [];
    const role =
      intent === "ACTIVATE_SEGMENTS"
        ? "segments"
        : intent === "ACTIVATE_DEALS"
          ? "deals-activated"
          : "deals-suppressed";
    return [{ kind: "ids", role, ids }];
  }

  if (intent === "ADJUST_DEAL_FLOOR") {
    if (!isFiniteNumber(payload.bidfloor)) return [];
    return [{
      kind: "floor",
      dealId: dealIdFromPath(mutation.path),
      // null is a real "unknown baseline", never defaulted to zero (BR-8).
      before: findOriginalDealBidfloor({ submittedPayload }, mutation.path),
      after: payload.bidfloor,
    }];
  }

  if (intent === "ADJUST_DEAL_MARGIN") {
    const margin = payload.margin;
    if (!margin || !isFiniteNumber(margin.value)) return [];
    return [{
      kind: "margin",
      dealId: dealIdFromPath(mutation.path),
      value: margin.value,
      // Read from the mutation, never assumed to be percent (BR-9).
      calculationType: CALC_TYPE[margin.calculation_type] ?? "CPM",
    }];
  }

  if (intent === "BID_SHADE") {
    if (!isFiniteNumber(payload.price)) return [];
    const before = submittedPayload?.bid_response?.seatbid?.[0]?.bid?.[0]?.price;
    return [{
      kind: "price",
      before: isFiniteNumber(before) ? before : null,
      after: payload.price,
    }];
  }

  return [];
}

/** Describe the impression, entirely from the submitted request (FR-13). */
export function buildScenarioContext(submittedPayload) {
  const payload = submittedPayload ?? {};
  const br = payload.bid_request ?? {};
  const site = br.site ?? br.app ?? {};
  const device = br.device ?? {};
  const user = br.user ?? {};
  const imps = Array.isArray(br.imp) ? br.imp : [];
  const imp0 = imps[0] ?? {};

  const kinds = [];
  if (imp0.banner) kinds.push("banner");
  if (imp0.video) kinds.push("video");
  if (imp0.audio) kinds.push("audio");
  if (imp0.native) kinds.push("native");

  const deals = [];
  for (const imp of imps) {
    for (const deal of imp?.pmp?.deals ?? []) {
      if (typeof deal?.id === "string") {
        deals.push({
          id: deal.id,
          bidFloor: isFiniteNumber(deal.bidfloor) ? deal.bidfloor : null,
          auctionType: Number.isInteger(deal.at) ? deal.at : null,
        });
      }
    }
  }

  // Request-borne user data. Kept separate from mutation values so it can only
  // ever render as "already sent by the exchange" (BR-23).
  const userSignals = [];
  if (Number.isInteger(user.yob)) {
    userSignals.push({ label: "year of birth", value: String(user.yob), provenance: "request" });
  }
  if (typeof user.gender === "string" && user.gender.length > 0) {
    userSignals.push({ label: "gender", value: user.gender, provenance: "request" });
  }
  for (const provider of user.data ?? []) {
    for (const seg of provider?.segment ?? []) {
      const name = seg?.name ?? seg?.id;
      if (typeof name === "string" && name.length > 0) {
        userSignals.push({ label: "data segment", value: name, provenance: "request" });
      }
    }
  }

  return {
    requestId: typeof payload.id === "string" ? payload.id : null,
    publisher: typeof site.domain === "string" ? site.domain : null,
    page: typeof site.page === "string" ? site.page : null,
    contentCategories: Array.isArray(site.cat) ? site.cat.filter((c) => typeof c === "string") : [],
    categoryTaxonomy: Number.isInteger(site.cattax) ? site.cattax : null,
    deviceType: typeof device.ua === "string" && device.ua.length > 0 ? device.ua : null,
    geo: typeof device?.geo?.region === "string"
      ? `${device.geo.country ?? ""}-${device.geo.region}`.replace(/^-/, "")
      : (typeof device?.geo?.country === "string" ? device.geo.country : null),
    impressionFormat: kinds.length === 1 ? kinds[0] : kinds.length > 1 ? "mixed" : "unknown",
    bidFloor: isFiniteNumber(imp0.bidfloor) ? imp0.bidfloor : null,
    deals,
    userSignals,
  };
}

function displayLabelFor(containerName) {
  const stopId = CONTAINER_NAME_TO_STOP_ID[containerName];
  return (stopId && DISPLAY_NAME_BY_STOP_ID[stopId]) || containerName || null;
}

function makeBeat(index, kind, extra) {
  return {
    index,
    kind,
    movement: MOVEMENT_BY_KIND[kind],
    containerName: null,
    displayLabel: null,
    intent: null,
    operation: null,
    modelVersion: null,
    explored: false,
    latencyMs: null,
    path: null,
    values: [],
    ...extra,
  };
}

/**
 * Derive the beat sequence: one origin beat, one beat per Mutation object in
 * stop order, one recap beat (BR-1 through BR-4).
 *
 * A stop with no mutations contributes no beats, so a container that was
 * invoked and deliberately recommended no change is absent from the
 * walkthrough (BR-4). The total is therefore only knowable after the response.
 */
export function buildBeats(submittedPayload, normalizedResult) {
  const stops = Array.isArray(normalizedResult?.stops) ? normalizedResult.stops : [];
  const beats = [];

  beats.push(makeBeat(0, BEAT_ORIGIN, {}));

  for (const stop of stops) {
    // Only container stops carry a resolvable container name; the ssp and dsp
    // bookends do not and contribute nothing.
    const containerName = stop?.id;
    if (containerName === "ssp" || containerName === "dsp") continue;
    const mutations = Array.isArray(stop?.mutations) ? stop.mutations : [];
    for (const mutation of mutations) {
      const modelVersion = typeof stop.modelVersion === "string" ? stop.modelVersion : null;
      beats.push(makeBeat(beats.length, BEAT_CONTAINER, {
        containerName,
        // The stop's own displayName wins. It is resolved by normalizer.js,
        // which prefers the label the orchestrator sent — the only source that
        // can name a store-defined container, since its name is not known when
        // this bundle is built. Falling straight through to the stop id would
        // print a synthesized id like "dynamic:artf-template" as the label.
        displayLabel:
          (typeof stop?.displayName === "string" && stop.displayName) ||
          DISPLAY_NAME_BY_STOP_ID[containerName] ||
          containerName,
        intent: mutation?.intent ?? null,
        operation: mutation?.op ?? null,
        modelVersion,
        explored: isExplored(modelVersion),
        latencyMs: isFiniteNumber(stop?.latency?.ms) ? stop.latency.ms : null,
        path: typeof mutation?.path === "string" ? mutation.path : null,
        values: mutationToValues(mutation, submittedPayload),
      }));
      // Attribution travels WITH the value. visibleValues flattens the beat
      // away, so a floor change reaching the sell-side column could otherwise
      // only render as "Floor · deal-x" — the container that decided it was lost
      // at the flatten. Stamped here rather than passed alongside because every
      // consumer that reads a value already has the value and nothing else.
      const beat = beats[beats.length - 1];
      for (const value of beat.values) {
        value.containerName = beat.containerName;
        value.displayLabel = beat.displayLabel;
        value.intent = beat.intent;
      }
    }
  }

  // The recap restates what actually changed. A container that produced nothing
  // is absent from it, consistent with having produced no beat (BR-9 of the
  // logic model).
  const contributors = [];
  for (const beat of beats) {
    if (beat.kind !== BEAT_CONTAINER) continue;
    const existing = contributors.find((c) => c.containerName === beat.containerName);
    if (existing) existing.beatIndexes.push(beat.index);
    else contributors.push({
      containerName: beat.containerName,
      displayLabel: beat.displayLabel,
      beatIndexes: [beat.index],
    });
  }
  beats.push(makeBeat(beats.length, BEAT_RECAP, { contributors }));

  return beats;
}

/**
 * Values from beats up to and including `index`, so what is on screen is
 * derived rather than accumulated across moves (BR-17, BR-24).
 */
export function visibleValues(beats, index) {
  if (!Array.isArray(beats)) return [];
  return beats.slice(0, index + 1).flatMap((b) => b.values ?? []);
}

/** Card state as a function of how far the walkthrough has progressed (BR-25). */
export function cardStateFor(beats, index) {
  if (!Array.isArray(beats) || beats.length === 0) return "neutral";
  const seen = beats.slice(0, index + 1);
  if (seen.some((b) => b.kind === BEAT_RECAP)) return "settled";
  if (seen.some((b) => b.kind === BEAT_CONTAINER)) return "enriching";
  return "neutral";
}

/** `displayLabelFor` is exported for the label-correction test in FR-42. */
export { displayLabelFor };
