// OffersPanel.jsx — the Theater's right column.
//
// The column is labelled OFFERS, not bidders (FR-26, BR-6). All offers arrive
// under one seat, so "bidders" would imply competing buyers where there is one
// seller's demand endpoint presenting several campaigns — a misrepresentation of
// what the auction actually contained.
//
// The column withholds its offers until `revealed`. The auction resolves against
// the ENRICHED request, so showing offers while the containers are still mutating
// it would put the consequence on screen before its cause, and would imply these
// campaigns were priced against the request as it arrived.
//
// The notice comes from the view model, which chose it from the SHAPE OF THE DATA
// (FR-31, BR-19). This component cannot override it.

import { OfferRow } from "./OfferRow.jsx";
import { AUCTION_FAILED } from "../hooks/useTheaterRun.js";

/**
 * Why the column is showing a fixture instead of the live auction.
 *
 * Reported separately from the view model's notice rather than folded into it. The
 * notice describes the DATA and is derived from its shape (FR-31); a fault
 * describes the RUN, which the data cannot know about. Deriving one from the other
 * would mean a fixture rendered for a good reason and a fixture rendered because a
 * request failed look identical — which is how the missing bearer token went
 * unnoticed through a whole feature.
 */
function AuctionFaultNotice({ fault }) {
  if (!fault) return null;
  const failed = fault.kind === AUCTION_FAILED;
  return (
    <p
      className={`th-offers-fault th-offers-fault-${fault.kind}`}
      data-testid="offers-panel-fault"
      data-fault-kind={fault.kind}
    >
      <strong>{failed ? "The live auction could not be read." : "No live auction."}</strong>{" "}
      {fault.detail}
      {failed
        ? " The offers below are the captured fixture, not this scenario's auction."
        : ""}
    </p>
  );
}

export function OffersPanel({ viewModel, revealed, auctionFault }) {
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

      {/*
        While pending the column keeps its place in the grid, so the layout does
        not reflow when the offers arrive. It states why it is empty rather than
        rendering nothing, which would read as a failed lookup.
      */}
      {!revealed ? (
        <div className="th-offers-pending" data-testid="offers-panel-pending">
          Offers resolve against the enriched request. They appear once the
          containers have finished mutating it.
        </div>
      ) : (
        <>
          <AuctionFaultNotice fault={auctionFault} />

          {viewModel?.noticeText ? (
            <p
              className={`th-offers-notice th-offers-notice-${viewModel.notice}`}
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
            cleared price as supporting detail (FR-28, BR-16). Leading with price
            would frame this as an ordinary price auction and bury the sell-side
            decision that produced the result.

            A null winner is an explicit unsold state, distinct from an empty or
            missing one (BR-17) — an empty render looks like a failed lookup.
          */}
          {winner != null ? (
            <div className="th-offers-winner" data-testid="offers-panel-winner">
              <span className="th-row-key">Winning deal</span>
              <span className="th-row-val">{winner.dealId ?? "unknown deal"}</span>
              <span className="th-offers-winner-campaign">
                {winner.campaignName ?? "unknown campaign"}
              </span>
              <span className="th-offers-winner-price">
                {winner.clearedPrice == null
                  ? "cleared price not reported"
                  : `cleared at $${winner.clearedPrice.toFixed(2)}`}
              </span>
            </div>
          ) : (
            <div className="th-offers-unsold" data-testid="offers-panel-unsold">
              <span className="th-row-key">No winning offer</span>
              <span className="th-row-val">This response records no winner</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
