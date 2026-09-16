// outcomeClassifier.js — why an offer ended up as it did.
//
// FR-29 requires a candidate excluded by an ARTF *decision* to be
// distinguishable from one absent through *timeout or transport failure*.
//
// Three categories, not two. FR-29 was written when the hook's outcome type had
// four states; U3's NFR Design added SkippedInsufficientBudget, which is neither
// a decision nor a failure — nothing decided, nothing broke, the orchestrator was
// never asked. Calling it a decision would attribute a choice nobody made;
// calling it a failure would report a fault that did not occur (BR-9, BR-11).
//
// Classification is total: every input maps to exactly one category and nothing
// returns undefined. An unhandled code rendering blank would read to a viewer as
// "not involved", which is the misreading the whole column exists to prevent.

/**
 * Categories.
 *
 * FR-29's three are ARTF_DECISION, INFRASTRUCTURE_FAILURE and NOT_ATTEMPTED — the
 * three reasons a candidate did not transact. Two more exist for offers the auction
 * itself resolved:
 *
 *   WON             this offer transacted
 *   AUCTION_OUTCOME it bid and did not win
 *
 * AUCTION_OUTCOME is separate from ARTF_DECISION deliberately. A bid that lost on
 * price was not excluded by any container's decision — the auction decided. Folding
 * it into ARTF_DECISION would make `isDecision` answer "yes, an ARTF decision caused
 * this" about an ordinary price loss, which is exactly the question FR-29 needs
 * answered correctly.
 */
export const CATEGORY = Object.freeze({
  ARTF_DECISION: "ArtfDecision",
  INFRASTRUCTURE_FAILURE: "InfrastructureFailure",
  NOT_ATTEMPTED: "NotAttempted",
  WON: "Won",
  AUCTION_OUTCOME: "AuctionOutcome",
});

/** Presentation outcomes, per FR-27. */
export const OUTCOME = Object.freeze({
  WON: "won",
  LOST_ON_PRICE: "lost_on_price",
  REJECTED_BELOW_FLOOR: "rejected_below_floor",
  NOT_OFFERED: "not_offered",
  UNAVAILABLE: "unavailable",
});

/** Prebid seatnonbid status codes this surface understands. */
export const STATUS = Object.freeze({
  TIMEOUT: 101,
  BELOW_FLOOR: 301,
});

/** ARTF exclusion reasons emitted by the demand endpoint (U2). */
export const EXCLUSION = Object.freeze({
  DEAL_SUPPRESSED: "deal_suppressed",
  BELOW_FLOOR: "below_floor",
  NOT_TARGETED: "not_targeted",
  NO_DEAL_ON_IMPRESSION: "no_deal_on_impression",
});

/** Hook outcomes that mean enrichment never happened (U3). */
export const HOOK_OUTCOME = Object.freeze({
  TIMEOUT: "Timeout",
  TRANSPORT_FAILURE: "TransportFailure",
  SKIPPED_INSUFFICIENT_BUDGET: "SkippedInsufficientBudget",
});

// A Map, not an object literal, because the key comes off the wire.
//
// `exclusionReason` is a string from the demand endpoint's response. An object literal
// inherits from Object.prototype, so obj["valueOf"] returns a FUNCTION -- which passes
// a `!= null` guard and would be handed on as a human-readable reason. The same is true
// of "toString", "constructor" and "hasOwnProperty". A Map has no prototype chain to
// walk, so an unknown key is `undefined` and falls to the catch-all, which is what the
// unknown-key path is for.
//
// Found by the totality property test, not by inspection: fast-check drew the reason
// "valueOf" and the "always carries a reason" property returned false because `reason`
// was a function rather than a string.
const HUMAN_REASON = new Map([
  [EXCLUSION.DEAL_SUPPRESSED, "Deal suppressed by the Deal Scorer"],
  [EXCLUSION.BELOW_FLOOR, "CPM below the resolved floor"],
  [EXCLUSION.NOT_TARGETED, "Targeting did not match this impression"],
  [EXCLUSION.NO_DEAL_ON_IMPRESSION, "No deal for this campaign on the impression"],
  [HOOK_OUTCOME.TIMEOUT, "Enrichment timed out"],
  [HOOK_OUTCOME.TRANSPORT_FAILURE, "Enrichment could not be reached"],
  [
    HOOK_OUTCOME.SKIPPED_INSUFFICIENT_BUDGET,
    "Enrichment not attempted — too little time remained",
  ],
]);

/**
 * Map a status code and an exclusion reason to an outcome and its category.
 *
 * Total by construction: the final branch is a catch-all that yields
 * `UNAVAILABLE` in the `NOT_ATTEMPTED` category with an explicit reason, so an
 * unrecognised input is still legible rather than blank (BR-9).
 */
export function classify(seatNonBidCode, exclusionReason, opts = {}) {
  const { offered = false, markedWinner = false } = opts;

  if (markedWinner) {
    return { outcome: OUTCOME.WON, category: CATEGORY.WON, reason: null };
  }

  // Offered but not marked as winner: it competed and the auction decided. Not an
  // ARTF decision — no container excluded it.
  if (offered) {
    return { outcome: OUTCOME.LOST_ON_PRICE, category: CATEGORY.AUCTION_OUTCOME, reason: null };
  }

  // Hook-level outcomes mean enrichment never produced a decision.
  if (exclusionReason === HOOK_OUTCOME.SKIPPED_INSUFFICIENT_BUDGET) {
    return {
      outcome: OUTCOME.UNAVAILABLE,
      category: CATEGORY.NOT_ATTEMPTED,
      reason: HUMAN_REASON.get(HOOK_OUTCOME.SKIPPED_INSUFFICIENT_BUDGET),
    };
  }
  if (
    exclusionReason === HOOK_OUTCOME.TIMEOUT ||
    exclusionReason === HOOK_OUTCOME.TRANSPORT_FAILURE
  ) {
    return {
      outcome: OUTCOME.UNAVAILABLE,
      category: CATEGORY.INFRASTRUCTURE_FAILURE,
      reason: HUMAN_REASON.get(exclusionReason),
    };
  }

  // ARTF exclusion reasons are decisions.
  if (exclusionReason != null && HUMAN_REASON.has(exclusionReason)) {
    const outcome =
      exclusionReason === EXCLUSION.BELOW_FLOOR
        ? OUTCOME.REJECTED_BELOW_FLOOR
        : OUTCOME.NOT_OFFERED;
    return { outcome, category: CATEGORY.ARTF_DECISION, reason: HUMAN_REASON.get(exclusionReason) };
  }

  // Status codes, when no ARTF reason accompanied them.
  if (seatNonBidCode === STATUS.BELOW_FLOOR) {
    return {
      outcome: OUTCOME.REJECTED_BELOW_FLOOR,
      category: CATEGORY.ARTF_DECISION,
      reason: HUMAN_REASON.get(EXCLUSION.BELOW_FLOOR),
    };
  }
  if (seatNonBidCode === STATUS.TIMEOUT) {
    return {
      outcome: OUTCOME.UNAVAILABLE,
      category: CATEGORY.INFRASTRUCTURE_FAILURE,
      reason: HUMAN_REASON.get(HOOK_OUTCOME.TIMEOUT),
    };
  }

  // Catch-all. Not a decision and not a known failure, so it is reported as what
  // it is: no reason available.
  return {
    outcome: OUTCOME.UNAVAILABLE,
    category: CATEGORY.NOT_ATTEMPTED,
    reason:
      seatNonBidCode != null
        ? `No offer — unrecognised status ${seatNonBidCode}`
        : "No offer — no reason reported",
  };
}

/**
 * Did an ARTF decision cause this outcome?
 *
 * The single query point for FR-29's distinction, so it is never re-derived at a
 * call site (BR-12).
 */
export function isDecision(outcome) {
  return outcome?.category === CATEGORY.ARTF_DECISION;
}

/** Did infrastructure failure cause this, rather than a decision? */
export function isInfrastructureFailure(outcome) {
  return outcome?.category === CATEGORY.INFRASTRUCTURE_FAILURE;
}

/** Was enrichment never attempted for this candidate? */
export function isNotAttempted(outcome) {
  return outcome?.category === CATEGORY.NOT_ATTEMPTED;
}
