// runSummaryValidation.js — decide whether a generated run summary may be shown.
//
// It delegates to validateCaption rather than reimplementing it. There is one
// validator in this codebase and this module's whole job is to describe the run
// to it: an aggregate pseudo-beat carrying every value the run produced, plus the
// auction's own literals and numbers.
//
// Written as composition on purpose. A second copy of the number-extraction and
// register rules would be the obvious place for the two to drift, and a drift
// here means a summary containing an invented figure passes because the copy that
// would have caught it was not the copy that ran.
//
// Pure. No I/O, no clock, no randomness.

import { validateCaption, UNIT } from "./captionValidation.js";
import { BEAT_CONTAINER } from "./theaterBeats.js";
import { SUMMARY_WORD_CEILING, SUMMARY_CHAR_CEILING } from "./runSummaryPrompt.js";

/**
 * One beat-shaped object standing for the whole run.
 *
 * `latencyMs` is deliberately null: individual container latencies are real, but
 * no measured total exists, so allowing them would let the summary state a
 * per-container figure as if it described the run.
 */
export function aggregateBeat(beats) {
  const containers = (beats ?? []).filter((b) => b?.kind === BEAT_CONTAINER);
  const values = containers.flatMap((b) => b.values ?? []);
  const contributors = [];
  for (const b of containers) {
    if (!contributors.some((c) => c.containerName === b.containerName)) {
      contributors.push({ containerName: b.containerName, displayLabel: b.displayLabel });
    }
  }
  return {
    kind: "run",
    containerName: null,
    displayLabel: null,
    intent: null,
    modelVersion: null,
    path: null,
    latencyMs: null,
    explored: containers.some((b) => b.explored === true),
    values,
    contributors,
  };
}

/** Campaign names, deal ids and seats the auction really named. */
export function auctionLiterals(viewModel) {
  const out = [];
  const push = (s) => {
    if (typeof s === "string" && s.length > 0) out.push(s);
  };
  for (const offer of viewModel?.offers ?? []) {
    push(offer.campaignName);
    push(offer.campaignId);
    push(offer.dealId);
    push(offer.seat);
    push(offer.impId);
    push(offer.exclusionReason);
  }
  const winner = viewModel?.winner;
  if (winner) {
    push(winner.campaignName);
    push(winner.campaignId);
    push(winner.dealId);
  }
  push(viewModel?.currency);
  return out;
}

/** Prices and counts the auction really reported. */
export function auctionNumbers(viewModel) {
  const out = [];
  const add = (value, unit) => {
    if (typeof value === "number" && Number.isFinite(value)) out.push({ value, unit });
  };

  for (const offer of viewModel?.offers ?? []) {
    add(offer.price, UNIT.CURRENCY);
    add(offer.observed?.returnedPrice, UNIT.CURRENCY);
    add(offer.observed?.impFloor, UNIT.CURRENCY);
  }
  add(viewModel?.winner?.clearedPrice, UNIT.CURRENCY);

  // Group counts are stated in the summary lines the prompt hands over, so a
  // summary may legitimately repeat them.
  for (const group of viewModel?.groups ?? []) {
    add(group.count, UNIT.COUNT);
    for (const b of group.breakdown ?? []) add(b.count, UNIT.COUNT);
  }
  add((viewModel?.bidRows ?? []).length, UNIT.COUNT);
  add((viewModel?.offers ?? []).length, UNIT.COUNT);

  return out;
}

/**
 * Validate a generated run summary.
 *
 * @param {string} text
 * @param {object[]} beats
 * @param {object} context   ScenarioContext
 * @param {object} viewModel offers view model
 * @returns {{ok: true, text: string} | {ok: false, violations: object[]}}
 */
export function validateRunSummary(text, beats, context, viewModel) {
  return validateCaption(text, aggregateBeat(beats), context, {
    wordCeiling: SUMMARY_WORD_CEILING,
    charCeiling: SUMMARY_CHAR_CEILING,
    literals: auctionLiterals(viewModel),
    numbers: auctionNumbers(viewModel),
  });
}
