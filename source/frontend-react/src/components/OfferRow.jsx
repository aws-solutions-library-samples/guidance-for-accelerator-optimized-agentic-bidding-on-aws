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

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };
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
});

export function OfferRow({ offer, isWinner }) {
  const { outcome } = offer;
  const category = outcome?.category ?? CATEGORY.NOT_ATTEMPTED;
  const idForTest = offer.campaignId ?? "unknown";

  return (
    <div
      className={`th-offer th-offer-${category}${isWinner ? " is-winner" : ""}`}
      data-testid={`offer-row-${idForTest}`}
      data-category={category}
    >
      <span className="th-offer-campaign">{offer.campaignName ?? UNKNOWN}</span>

      <span className="th-offer-deal" style={SUBTLE}>
        {offer.dealId ?? "no deal"}
      </span>

      <span className="th-offer-price">
        {offer.price == null ? "no bid" : `$${offer.price.toFixed(2)}`}
      </span>

      <span className="th-offer-outcome" data-testid={`offer-row-outcome-${idForTest}`}>
        {OUTCOME_LABEL[outcome?.outcome] ?? "No offer"}
        <span className="th-offer-mark" style={SUBTLE}>
          {CATEGORY_MARK[category]}
        </span>
      </span>

      {outcome?.reason ? (
        <span
          className="th-offer-reason"
          style={SUBTLE}
          data-testid={`offer-row-reason-${idForTest}`}
        >
          {outcome.reason}
        </span>
      ) : null}
    </div>
  );
}
