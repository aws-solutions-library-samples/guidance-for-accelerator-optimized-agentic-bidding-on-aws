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

import {
  BEAT_BIDS,
  BEAT_RECAP,
  BEAT_PASS,
  BEAT_BASELINE,
  PASS_BASELINE,
  PASS_ARTF,
} from "./theaterBeats.js";

/**
 * The words on a pass banner. One definition: the banner component renders
 * these, and `mutationNarration` reads the same ones out as a sentence, so the
 * visual and the spoken form cannot disagree.
 *
 * Pass 1 is the baseline -- the request as the publisher sent it, no container
 * consulted. Pass 2 is the same request with the ARTF containers mutating it
 * before the seats bid.
 *
 * Written as "First pass" / "Second pass" rather than "1 of 2": the caption
 * validator refuses any numeral that does not trace to a beat value (BR3-13),
 * and these words go through the same fallback path as every other caption.
 */
export function passBannerCopy(beat) {
  if (beat?.pass === PASS_BASELINE || beat?.artf === false) {
    return {
      title: "First pass",
      headline: "Without ARTF mutations",
      body: "The bid request goes to Prebid Server as the publisher sent it. No container is consulted.",
    };
  }
  return {
    title: "Second pass",
    headline: "With ARTF mutations",
    body: "The same request, mutated by the ARTF containers before the seats bid.",
  };
}

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

/**
 * The sentence appended to a mutation the exchange did not apply.
 *
 * Fixed wording with no numerals: the disposition reason can carry numbers (a
 * floor, a cap) that trace to no beat value, so the reason itself never enters a
 * caption and is rendered beside it instead (BR3-13).
 */
const NOT_APPLIED_SENTENCE = Object.freeze({
  rejected: "Not applied: rejected by the exchange.",
  superseded: "Not applied: superseded by a higher-priority container.",
  unknown: "The exchange did not report whether this was applied.",
});

/**
 * The recap's statement when the auction's own report has no mutations to show,
 * in words. Keyed by the hook's outcome; `unavailable` covers a report the
 * response did not carry.
 */
const HOOK_REPORT_SENTENCE = Object.freeze({
  unavailable: "The auction did not report the ARTF hook's mutations, so none are shown.",
  timeout: "The ARTF hook timed out, so this auction ran unmutated.",
  transport_failure: "The ARTF hook could not reach the orchestrator, so this auction ran unmutated.",
  skipped_insufficient_budget:
    "The ARTF hook had too little time left to call the orchestrator, so this auction ran unmutated.",
});

export function hookReportSentence(report) {
  if (!report) return null;
  if (report.available !== true) return HOOK_REPORT_SENTENCE.unavailable;
  return HOOK_REPORT_SENTENCE[report.outcome] ?? null;
}

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
 *   sampleTrained: boolean,
 *   unrendered: boolean,
 *   disposition: string|null,
 *   dispositionReason: string|null,
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
    sampleTrained: false,
    unrendered: false,
    disposition: null,
    dispositionReason: null,
    text: "",
  };
  if (!beat) return base;

  if (beat.kind === BEAT_PASS) {
    // The banner's own copy, as a sentence. This is what a screen reader gets and
    // what the fallback path shows; the banner component renders the same words.
    const copy = passBannerCopy(beat);
    return { ...base, text: `${copy.title}. ${copy.headline}. ${copy.body}` };
  }

  if (beat.kind === "origin") {
    const parts = [];
    if (context?.publisher) parts.push(context.publisher);
    if (context?.impressionFormat && context.impressionFormat !== "unknown") {
      parts.push(`${context.impressionFormat} impression`);
    }
    if (context?.bidFloor != null) parts.push(`floor $${context.bidFloor.toFixed(2)}`);
    const arrives = parts.length
      ? `The bid request arrives from ${parts.join(", ")}.`
      : "The bid request arrives from the exchange.";
    // The second arrival is the same request. Saying so is what tells the reader
    // the two passes are a controlled comparison and not two different requests.
    return {
      ...base,
      text: beat.pass === PASS_ARTF
        ? `${arrives} The same request, this time through the ARTF containers.`
        : arrives,
    };
  }

  if (beat.kind === BEAT_BIDS) {
    // No count here on purpose. The offers column states how many bids arrived and
    // from which seats; this beat is structural and carries no bid values.
    return {
      ...base,
      text: beat.pass === PASS_BASELINE
        ? "The seats bid against the request as the publisher sent it. No winner yet."
        : "The seats bid against the enriched request. No winner yet.",
    };
  }

  if (beat.kind === BEAT_BASELINE) {
    // No winner named here either: the offers column states it, from the response.
    return {
      ...base,
      text: "Prebid resolved the baseline auction. No ARTF container took part.",
    };
  }

  if (beat.kind === BEAT_RECAP) {
    const n = beat.contributors?.length ?? 0;
    // The hook's report said why nothing could be shown: say that, not "no
    // container mutated", which would claim the containers answered.
    const reportSentence = n === 0 ? hookReportSentence(beat.artfReport) : null;
    if (reportSentence) return { ...base, text: reportSentence };
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
  const disposition = typeof beat.disposition === "string" ? beat.disposition : null;
  const notApplied = disposition && disposition !== "applied"
    ? ` ${NOT_APPLIED_SENTENCE[disposition] ?? NOT_APPLIED_SENTENCE.unknown}`
    : "";

  return {
    ...base,
    kind: beat.kind,
    containerLabel,
    containerName: beat.containerName ?? null,
    intent,
    phrases,
    latencyMs: beat.latencyMs ?? null,
    explored: beat.explored === true,
    // Rendered beside the sentence, not in it: the sentence is what the caption
    // validator checks.
    sampleTrained: beat.sampleTrained === true,
    unrendered: phrases.length === 0,
    disposition,
    dispositionReason: notApplied && typeof beat.dispositionReason === "string"
      ? beat.dispositionReason
      : null,
    text: phrases.length
      ? `${attribution}: ${phrases.join("; ")}.${latency}${explore}${notApplied}`
      : `${containerLabel} returned a ${intent ?? "mutation"} this view does not yet render.${latency}${notApplied}`,
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
