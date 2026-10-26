// offerPresentationService.js — a bid response becomes the offers column's view
// model.
//
// Sequences the two pure modules and chooses the notice. It holds no decision
// logic of its own: normalise, classify, resolve, then pick the notice.
//
// The notice is chosen from the SHAPE OF THE DATA, never from a prop default
// (FR-31, BR-19). That is the whole reason this module exists rather than the
// panel doing the wiring: a prop-driven notice can be wrong by omission — pass a
// fixture with the default prop and synthetic values render under a real-auction
// notice. Deriving it from the data makes that misrepresentation impossible
// rather than merely discouraged.

import { normalizeOffers, resolveWinner } from "./bidResponseNormalizer.js";
import { classify, CATEGORY, EXCLUSION } from "./outcomeClassifier.js";
import { isFixtureResponse } from "./bidResponseFixture.js";

export const NOTICE = Object.freeze({
  DECLARED_INVENTORY: "declared_inventory",
  ILLUSTRATIVE: "illustrative",
});

const NOTICE_TEXT = Object.freeze({
  [NOTICE.DECLARED_INVENTORY]:
    "Offers as returned by the auction. Prices are the campaigns' declared CPMs; " +
    "the winner and cleared price are what Prebid resolved.",
  [NOTICE.ILLUSTRATIVE]:
    "Illustrative example. These offers are fixture values captured for building " +
    "this view, not measured outcomes.",
});

/**
 * Which notice this data warrants.
 *
 * A response carrying the fixture marker is illustrative. Anything else is a
 * real response and gets the declared-inventory notice. There is no third case
 * and no default to fall back to.
 */
export function selectNotice(bidResponse) {
  return isFixtureResponse(bidResponse) ? NOTICE.ILLUSTRATIVE : NOTICE.DECLARED_INVENTORY;
}

/* ------------------------------------------------------------------ grouping */

/**
 * A row is either an OFFER, shown on its own, or it belongs to a GROUP.
 *
 * Every candidate still reaches the column — BR-15 — but not every candidate earns
 * a row of its own. Measured on the live stack, the six scenarios produced 18 real
 * bids, 3 sell-side decisions and 96 campaigns that were never candidates for the
 * impression at all. Four of the six had ZERO decisions, so their entire non-bid
 * section was catalog non-applicability: a banner campaign listed against a video
 * slot, once per impression.
 *
 * BR-14 requires a non-offering candidate to carry its reason rather than vanish,
 * and its stated rationale is that "a row reading 'suppressed by the Deal Scorer'
 * says the feature worked". A group preserves exactly that — the reason is still
 * rendered, still attributable, one disclosure away — while letting the bids and
 * the decisions be the content instead of 18% of it.
 *
 * REVISION TO BR-15, recorded rather than slipped in: it read "every candidate
 * present in the response appears in the column", written when appearing meant a
 * top-level row. Appearing inside a disclosure still satisfies it; a row omitted
 * entirely would not, which is why nothing is dropped.
 */
export const GROUP = Object.freeze({
  DECISION: "decision",
  INELIGIBLE: "ineligible",
  NO_BID: "no_bid",
});

/** Fixed order, so the same response always renders the same column (NFR-5). */
const GROUP_ORDER = Object.freeze([GROUP.DECISION, GROUP.INELIGIBLE, GROUP.NO_BID]);

const GROUP_LABEL = Object.freeze({
  [GROUP.DECISION]: {
    one: "1 campaign stopped by a sell-side decision",
    many: (n) => `${n} campaigns stopped by a sell-side decision`,
  },
  [GROUP.INELIGIBLE]: {
    one: "1 campaign not eligible for this impression",
    many: (n) => `${n} campaigns not eligible for this impression`,
  },
  [GROUP.NO_BID]: {
    one: "1 seat returned no bid",
    many: (n) => `${n} seats returned no bid`,
  },
});

/** Short wording for the summary line's breakdown. Long enough to be a reason. */
const BREAKDOWN_LABEL = new Map([
  [EXCLUSION.BELOW_FLOOR, "below floor"],
  [EXCLUSION.DEAL_SUPPRESSED, "deal suppressed"],
  [EXCLUSION.NO_DEAL_ON_IMPRESSION, "no deal on the impression"],
  [EXCLUSION.NOT_TARGETED, "targeting did not match"],
  [EXCLUSION.MEDIA_TYPE_UNSUPPORTED, "creative format"],
]);

/** A campaign that WAS in the running, and something decided against it. */
const DECISION_REASONS = new Set([EXCLUSION.BELOW_FLOOR, EXCLUSION.DEAL_SUPPRESSED]);

/** A campaign that was never a candidate for this impression. */
const INELIGIBLE_REASONS = new Set([
  EXCLUSION.NO_DEAL_ON_IMPRESSION,
  EXCLUSION.NOT_TARGETED,
  EXCLUSION.MEDIA_TYPE_UNSUPPORTED,
]);

/**
 * Which group a row belongs to, or null to render it on its own.
 *
 * Ungrouped: anything that BID, and any INFRASTRUCTURE_FAILURE. A fault is never
 * folded into a disclosure — a timeout hidden behind a summary line is a fault
 * nobody sees, which is the failure mode this column was built to avoid.
 */
export function groupOf(offer) {
  const { outcome } = offer;
  if (outcome?.category === CATEGORY.WON) return null;
  if (outcome?.category === CATEGORY.AUCTION_OUTCOME && offer.offered) return null;
  if (outcome?.category === CATEGORY.INFRASTRUCTURE_FAILURE) return null;

  if (DECISION_REASONS.has(offer.exclusionReason)) return GROUP.DECISION;
  if (INELIGIBLE_REASONS.has(offer.exclusionReason)) return GROUP.INELIGIBLE;
  return GROUP.NO_BID;
}

/**
 * The summary line's breakdown.
 *
 * Keyed on the exclusion reason for the campaign groups, and on the SEAT for the
 * no-bid group — a seat non-bid carries no exclusion reason, so reasons would make
 * that group read "3 no reason reported" about seats whose reason is on their own
 * rows. Naming the seats says what the group contains.
 */
function breakdownOf(rows, groupId) {
  const counts = new Map();
  for (const row of rows) {
    const label =
      groupId === GROUP.NO_BID
        ? (row.seat ?? "unnamed seat")
        : (BREAKDOWN_LABEL.get(row.exclusionReason) ?? "no reason reported");
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  // Descending by count, then alphabetical, so the order is total and stable.
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

/**
 * Build the offers column's view model.
 *
 * Pure: same response in, same view model out (NFR-5, BR-4). Reads no clock, no
 * random source and no ambient state.
 *
 * `offers` remains the COMPLETE set in response order. `bidRows` and `groups`
 * partition it for presentation; nothing is in neither and nothing is in both.
 */
export function buildOfferViewModel(bidResponse) {
  const rows = normalizeOffers(bidResponse);

  const offers = rows.map((row) => {
    const outcome = classify(row.statusCode, row.exclusionReason, {
      offered: row.offered,
      markedWinner: row.markedWinner,
      observed: row.observed,
    });
    return { ...row, outcome };
  });

  const bidRows = [];
  const byGroup = new Map(GROUP_ORDER.map((id) => [id, []]));
  for (const offer of offers) {
    const group = groupOf(offer);
    if (group == null) bidRows.push(offer);
    else byGroup.get(group).push(offer);
  }

  const groups = GROUP_ORDER.filter((id) => byGroup.get(id).length > 0).map((id) => {
    const rowsInGroup = byGroup.get(id);
    const label = GROUP_LABEL[id];
    return {
      id,
      count: rowsInGroup.length,
      label: rowsInGroup.length === 1 ? label.one : label.many(rowsInGroup.length),
      breakdown: breakdownOf(rowsInGroup, id),
      rows: rowsInGroup,
    };
  });

  const notice = selectNotice(bidResponse);

  return {
    offers,
    bidRows,
    groups,
    winner: resolveWinner(rows),
    notice,
    noticeText: NOTICE_TEXT[notice],
    currency: bidResponse?.cur ?? null,
  };
}

export { NOTICE_TEXT };
