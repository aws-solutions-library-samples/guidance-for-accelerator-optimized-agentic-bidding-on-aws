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
import { classify } from "./outcomeClassifier.js";
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

/**
 * Build the offers column's view model.
 *
 * Pure: same response in, same view model out (NFR-5, BR-4). Reads no clock, no
 * random source and no ambient state.
 */
export function buildOfferViewModel(bidResponse) {
  const rows = normalizeOffers(bidResponse);

  const offers = rows.map((row) => {
    const outcome = classify(row.statusCode, row.exclusionReason, {
      offered: row.offered,
      markedWinner: row.markedWinner,
    });
    return { ...row, outcome };
  });

  const notice = selectNotice(bidResponse);

  return {
    offers,
    winner: resolveWinner(rows),
    notice,
    noticeText: NOTICE_TEXT[notice],
    currency: bidResponse?.cur ?? null,
  };
}

export { NOTICE_TEXT };
