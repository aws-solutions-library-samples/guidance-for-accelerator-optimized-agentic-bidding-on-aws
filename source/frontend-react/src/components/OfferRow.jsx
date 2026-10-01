// OfferRow.jsx — one offer in the offers column.
//
// Shows campaign identity, the deal it transacts on, its price and its outcome
// (FR-27). A candidate that made no offer shows ITS REASON, never a blank
// (FR-30, BR-14) — an omitted or blank row reads as "this campaign was not
// involved", which is the opposite of what a suppression is meant to show.
//
// The three outcome categories are distinguishable WITHOUT RELYING ON COLOUR
// ALONE: each carries a text marker as well as a class, so the distinction
// survives colour-blindness and greyscale screenshots.

import { CATEGORY } from "../utils/outcomeClassifier.js";

const UNKNOWN = "unknown";

/** Text marker per category — the non-colour half of the distinction. */
const CATEGORY_MARK = Object.freeze({
  [CATEGORY.WON]: "won",
  [CATEGORY.AUCTION_OUTCOME]: "auction",
  [CATEGORY.ARTF_DECISION]: "decision",
  [CATEGORY.INFRASTRUCTURE_FAILURE]: "unavailable",
  [CATEGORY.NOT_ATTEMPTED]: "not attempted",
});

const OUTCOME_LABEL = Object.freeze({
  won: "Won",
  lost_on_price: "Lost on price",
  rejected_below_floor: "Rejected below floor",
  not_offered: "No offer",
  unavailable: "No offer",
  no_bid: "No bid returned",
});

/**
 * Who this row is about.
 *
 * A campaign name when there is one. Otherwise the SEAT, because a seat-level
 * non-bid has no campaign and the response names the seat plainly — rendering it
 * "unknown" discards an identity we were given. The suffix keeps the two readable
 * as different kinds of thing, so a seat is never mistaken for a campaign.
 */
function identityOf(offer) {
  if (offer.campaignName != null) return offer.campaignName;
  if (offer.seat != null) return `${offer.seat} (seat)`;
  return UNKNOWN;
}

/**
 * @param outcomeRevealed  has the auction been called? Defaults to true.
 *
 * When false the row shows identity, deal and price and NOTHING that betrays the
 * result: no won/lost label, no category class, no reason. Gating only `isWinner`
 * is not enough — the gold treatment comes from `th-offer-${category}` and the
 * words come from `outcome`, so both leak the winner a step early on their own.
 */
export function OfferRow({ offer, isWinner, outcomeRevealed = true }) {
  const { outcome } = offer;
  const realCategory = outcome?.category ?? CATEGORY.NOT_ATTEMPTED;
  // "pending" is not a CATEGORY value: it is the absence of one, and it must not
  // collide with a real category in CSS or in `data-category` assertions.
  const category = outcomeRevealed ? realCategory : "pending";
  const idForTest = offer.campaignId ?? offer.seat ?? "unknown";

  return (
    <div
      className={`th-offer th-offer-${category}${isWinner && outcomeRevealed ? " is-winner" : ""}`}
      data-testid={`offer-row-${idForTest}`}
      data-category={category}
    >
      <span className="th-offer-campaign">{identityOf(offer)}</span>

      <span className="th-offer-deal">{offer.dealId ?? "no deal"}</span>

      <span className="th-offer-price">
        {offer.price == null ? "no bid" : `$${offer.price.toFixed(2)}`}
      </span>

      <span className="th-offer-outcome" data-testid={`offer-row-outcome-${idForTest}`}>
        {outcomeRevealed ? (
          <>
            {OUTCOME_LABEL[outcome?.outcome] ?? "No offer"}
            <span className="th-offer-mark">{CATEGORY_MARK[category]}</span>
          </>
        ) : (
          <>
            In the auction
            <span className="th-offer-mark">pending</span>
          </>
        )}
      </span>

      {/* The reason explains an outcome, so it waits for the outcome. */}
      {outcomeRevealed && outcome?.reason ? (
        <span className="th-offer-reason" data-testid={`offer-row-reason-${idForTest}`}>
          {outcome.reason}
        </span>
      ) : null}
    </div>
  );
}
