// OffersPanel.jsx — the sell-mode right column.
//
// The column is labelled OFFERS, not bidders (FR-26, BR-6). All offers arrive
// under one seat, so "bidders" would imply competing buyers where there is one
// seller's demand endpoint presenting several campaigns — a misrepresentation of
// what the auction actually contained.
//
// Sibling of TheaterBuySidePanel rather than a replacement: that panel remains the
// buy-mode column, and its illustrative fixture bidders are a different thing from
// these real offers.
//
// The notice comes from the view model, which chose it from the SHAPE OF THE DATA
// (FR-31, BR-19). This component cannot override it.

import { OfferRow } from "./OfferRow.jsx";

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };

export function OffersPanel({ viewModel, revealed }) {
  const offers = viewModel?.offers ?? [];
  const winner = viewModel?.winner ?? null;

  return (
    <div
      className={`th-col th-col-offers${revealed ? " is-live" : ""}`}
      data-testid="offers-panel"
    >
      <div className="th-col-head">
        <span className="th-col-title">Offers</span>
        <span className="th-col-sub">Candidate campaigns, one seat</span>
      </div>

      {viewModel?.noticeText ? (
        <p
          className={`th-offers-notice th-offers-notice-${viewModel.notice}`}
          style={SUBTLE}
          data-testid="offers-panel-notice"
        >
          {viewModel.noticeText}
        </p>
      ) : null}

      {offers.length === 0 ? (
        <div className="th-empty">No offers in this response</div>
      ) : (
        <div className="th-rows">
          {offers.map((offer) => (
            <OfferRow
              key={offer.key}
              offer={offer}
              isWinner={winner != null && winner.offerKey === offer.key}
            />
          ))}
        </div>
      )}

      {/*
        The outcome is the winning DEAL and the campaign holding it, with the
        cleared price as supporting detail (FR-28, BR-16). Leading with price would
        frame this as an ordinary price auction and bury the sell-side decision
        that produced the result.

        A null winner is an explicit unsold state, distinct from an empty or
        missing one (BR-17) — an empty render looks like a failed lookup.
      */}
      {winner != null ? (
        <div className="th-offers-winner" data-testid="offers-panel-winner">
          <span className="th-row-key">Winning deal</span>
          <span className="th-row-val">{winner.dealId ?? "unknown deal"}</span>
          <span className="th-offers-winner-campaign">{winner.campaignName ?? "unknown campaign"}</span>
          <span className="th-offers-winner-price" style={SUBTLE}>
            {winner.clearedPrice == null
              ? "cleared price not reported"
              : `cleared at $${winner.clearedPrice.toFixed(2)}`}
          </span>
        </div>
      ) : (
        <div className="th-offers-unsold" data-testid="offers-panel-unsold">
          <span className="th-row-key">No winning offer</span>
          <span className="th-row-val" style={SUBTLE}>
            This response records no winner
          </span>
        </div>
      )}
    </div>
  );
}
