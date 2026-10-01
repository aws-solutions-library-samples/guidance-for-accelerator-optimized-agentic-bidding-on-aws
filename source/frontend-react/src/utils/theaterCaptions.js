// theaterCaptions.js — a plain factual statement of a beat's real values.
//
// This is the floor of the caption feature, not a placeholder. The generated
// run summary is a separate surface now; this is what narrates every step, and
// it is strictly derived — a function of the beat rather than of a model's
// reading of it.
//
// It states values and makes no claim about how it was produced. The UI never
// labels either path (BR3-28).
//
// It lives here rather than in AuctionTheater.jsx because bedrockCaptionClient
// degrades to it, and importing it from the component that consumes the caption
// service would be circular. AuctionTheater.jsx re-exports it.

import { BEAT_BIDS, BEAT_RECAP } from "./theaterBeats.js";

/**
 * Human wording for an ARTF intent, for the pill beside the container name.
 *
 * Absent from this map means the pill shows the raw intent token. That is the
 * right failure: a new intent should read as an unfamiliar name rather than
 * silently borrowing prose written for a different one.
 */
export const INTENT_LABEL = Object.freeze({
  ACTIVATE_SEGMENTS: "ACTIVATE_SEGMENTS",
  ACTIVATE_DEALS: "ACTIVATE_DEALS",
  SUPPRESS_DEALS: "SUPPRESS_DEALS",
  ADD_METRICS: "ADD_METRICS",
  ADD_CIDS: "ADD_CIDS",
  ADJUST_DEAL_FLOOR: "ADJUST_DEAL_FLOOR",
  ADJUST_DEAL_MARGIN: "ADJUST_DEAL_MARGIN",
  BID_SHADE: "BID_SHADE",
});

/** One phrase per value, in the beat's own order. */
function valuePhrases(values) {
  return (values ?? []).map((v) => {
    switch (v.kind) {
      case "metric": return `${v.type.replace(/_/g, " ")} ${v.value}`;
      case "ids": return `${v.ids.length} ${v.role.replace(/-/g, " ")}`;
      case "floor": return `deal floor set to $${v.after.toFixed(2)}`;
      case "margin": return v.calculationType === "PERCENT"
        ? `margin set to ${(v.value * 100).toFixed(1)}%`
        : `margin set to $${v.value.toFixed(2)} CPM`;
      case "price": return `bid price set to $${v.after.toFixed(2)}`;
      default: return null;
    }
  }).filter(Boolean);
}

/**
 * The narration for one beat, as parts rather than a sentence.
 *
 * The centre-stage card needs the container and the intent as separate elements
 * so it can render the intent as a pill, and `factualCaption` needs the same
 * facts as prose. Both compose from this, so the pill and the sentence cannot
 * disagree about which container did what.
 *
 * @returns {{
 *   kind: string,
 *   containerLabel: string|null,
 *   containerName: string|null,
 *   intent: string|null,
 *   phrases: string[],
 *   latencyMs: number|null,
 *   explored: boolean,
 *   unrendered: boolean,
 *   text: string,
 * }}
 */
export function mutationNarration(beat, context) {
  const base = {
    kind: beat?.kind ?? null,
    containerLabel: null,
    containerName: null,
    intent: null,
    phrases: [],
    latencyMs: null,
    explored: false,
    unrendered: false,
    text: "",
  };
  if (!beat) return base;

  if (beat.kind === "origin") {
    const parts = [];
    if (context?.publisher) parts.push(context.publisher);
    if (context?.impressionFormat && context.impressionFormat !== "unknown") {
      parts.push(`${context.impressionFormat} impression`);
    }
    if (context?.bidFloor != null) parts.push(`floor $${context.bidFloor.toFixed(2)}`);
    return {
      ...base,
      text: parts.length
        ? `The bid request arrives from ${parts.join(", ")}.`
        : "The bid request arrives from the exchange.",
    };
  }

  if (beat.kind === BEAT_BIDS) {
    // No count here on purpose. This beat is built before the auction is fired, so
    // the number of bids is not known to it; the offers column is what states how
    // many arrived and from which seats. Naming a number here would either be
    // wrong or require the beat to wait on the auction.
    return {
      ...base,
      text: "The seats bid against the enriched request. No winner yet.",
    };
  }

  if (beat.kind === BEAT_RECAP) {
    const n = beat.contributors?.length ?? 0;
    return {
      ...base,
      // Unchanged wording: this exact string is the run's headline count and is
      // pinned by test.
      text: n === 0
        ? "No container mutated this request."
        : `${n} ${n === 1 ? "container" : "containers"} mutated this request.`,
    };
  }

  const containerLabel = beat.displayLabel ?? beat.containerName ?? "A container";
  const phrases = valuePhrases(beat.values);
  const latency = beat.latencyMs != null ? ` Responded in ${beat.latencyMs}ms.` : "";
  const explore = beat.explored ? " This response came from an exploration arm." : "";
  const intent = beat.intent ?? null;

  // The intent is named in the sentence, not only in the pill. The sentence is
  // what a screen reader gets and what the fallback path shows, so leaving the
  // intent to the pill alone would make it visual-only.
  const attribution = intent ? `${containerLabel} · ${intent}` : containerLabel;

  return {
    ...base,
    kind: beat.kind,
    containerLabel,
    containerName: beat.containerName ?? null,
    intent,
    phrases,
    latencyMs: beat.latencyMs ?? null,
    explored: beat.explored === true,
    unrendered: phrases.length === 0,
    text: phrases.length
      ? `${attribution}: ${phrases.join("; ")}.${latency}${explore}`
      : `${containerLabel} returned a ${intent ?? "mutation"} this view does not yet render.${latency}`,
  };
}

/**
 * @param {object|null} beat    a Beat from theaterBeats.js
 * @param {object|null} context a ScenarioContext from buildScenarioContext
 * @returns {string} plain prose, or "" for no beat
 */
export function factualCaption(beat, context) {
  return mutationNarration(beat, context).text;
}
