// theaterCaptions.js — a plain factual statement of a beat's real values.
//
// This is the floor of the caption feature, not a placeholder. The generated
// caption replaces it when one is obtained and passes validation; when one is
// not, this is what the reader sees, and it is the more strictly derived of the
// two because it is a function of the beat rather than of a model's reading of
// it.
//
// It states values and makes no claim about how it was produced. The UI never
// labels either path (BR3-28).
//
// It lives here rather than in AuctionTheater.jsx because bedrockCaptionClient
// degrades to it, and importing it from the component that consumes the caption
// service would be circular. AuctionTheater.jsx re-exports it.

import { BEAT_RECAP } from "./theaterBeats.js";

/**
 * @param {object|null} beat    a Beat from theaterBeats.js
 * @param {object|null} context a ScenarioContext from buildScenarioContext
 * @returns {string} plain prose, or "" for no beat
 */
export function factualCaption(beat, context) {
  if (!beat) return "";
  if (beat.kind === "origin") {
    const parts = [];
    if (context?.publisher) parts.push(context.publisher);
    if (context?.impressionFormat && context.impressionFormat !== "unknown") {
      parts.push(`${context.impressionFormat} impression`);
    }
    if (context?.bidFloor != null) parts.push(`floor $${context.bidFloor.toFixed(2)}`);
    return parts.length
      ? `The bid request arrives from ${parts.join(", ")}.`
      : "The bid request arrives from the exchange.";
  }
  if (beat.kind === BEAT_RECAP) {
    const n = beat.contributors?.length ?? 0;
    return n === 0
      ? "No container mutated this request."
      : `${n} ${n === 1 ? "container" : "containers"} mutated this request.`;
  }

  const label = beat.displayLabel ?? beat.containerName ?? "A container";
  const bits = (beat.values ?? []).map((v) => {
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

  const latency = beat.latencyMs != null ? ` Responded in ${beat.latencyMs}ms.` : "";
  const explore = beat.explored ? " This response came from an exploration arm." : "";
  return bits.length
    ? `${label}: ${bits.join("; ")}.${latency}${explore}`
    : `${label} returned a ${beat.intent ?? "mutation"} this view does not yet render.${latency}`;
}
