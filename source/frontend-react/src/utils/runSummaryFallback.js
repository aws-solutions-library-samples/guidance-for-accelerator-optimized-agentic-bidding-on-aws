// runSummaryFallback.js — the run's outcome stated from the data, with no model.
//
// This is what the summary surface shows before a generated summary arrives, and
// what it keeps showing if generation is unavailable, throttled, or rejected by
// validation. It is not a placeholder: it is the strictly-derived version of the
// same four facts, so the surface is never empty and never waits.
//
// It reads exactly the fields that exist. Where a fact is genuinely unavailable
// the sentence for it is omitted rather than hedged, because a sentence saying
// "the audience is not reported" is noise in prose, and the visual column already
// shows that absence structurally.
//
// Pure. No I/O, no clock, no randomness.

import { BEAT_CONTAINER } from "./theaterBeats.js";

/** Segment names the REQUEST asserted. Never one a container contributed. */
function assertedSegments(context) {
  return (context?.userSignals ?? [])
    .filter((s) => s?.label === "data segment" && typeof s.value === "string" && s.value)
    .map((s) => s.value);
}

/**
 * @param {object[]} beats
 * @param {object}   context   ScenarioContext
 * @param {object}   viewModel offers view model
 * @returns {string}
 */
export function factualRunSummary(beats, context, viewModel) {
  const sentences = [];

  // 1. Publisher and page.
  if (context?.publisher) {
    const format = context.impressionFormat && context.impressionFormat !== "unknown"
      ? `${context.impressionFormat} impression`
      : "impression";
    sentences.push(`The ${format} came from ${context.publisher}.`);
  }

  // 2. Audience, as asserted by the exchange.
  const segments = assertedSegments(context);
  if (segments.length) {
    sentences.push(
      `The request described the audience as ${segments.join(", ")}.`
    );
  }

  // 3. What the containers did.
  const containers = (beats ?? []).filter((b) => b?.kind === BEAT_CONTAINER);
  const names = [];
  for (const b of containers) {
    const label = b.displayLabel ?? b.containerName;
    if (label && !names.includes(label)) names.push(label);
  }
  if (names.length) {
    sentences.push(
      `${names.join(", ")} ${names.length === 1 ? "changed" : "changed"} the request before it was auctioned.`
    );
  }

  // 4. The advertisement served, and the field it beat.
  const winner = viewModel?.winner;
  if (winner) {
    const campaign = winner.campaignName ?? winner.campaignId ?? "an unnamed campaign";
    const deal = winner.dealId ? ` on ${winner.dealId}` : "";
    const price = typeof winner.clearedPrice === "number"
      ? ` at $${winner.clearedPrice.toFixed(2)}`
      : "";
    sentences.push(`${campaign} won${deal}${price}.`);

    const others = (viewModel?.bidRows ?? []).filter((o) => o.key !== winner.offerKey);
    if (others.length) {
      sentences.push(
        `${others.length} other ${others.length === 1 ? "campaign" : "campaigns"} bid and did not win.`
      );
    }
    // The recorded reasons, as recorded. Nothing is inferred about causation.
    const decision = (viewModel?.groups ?? []).find((g) => g.id === "decision");
    if (decision) sentences.push(`${decision.label}.`);
  } else {
    sentences.push("This response records no winner.");
  }

  return sentences.join(" ");
}
