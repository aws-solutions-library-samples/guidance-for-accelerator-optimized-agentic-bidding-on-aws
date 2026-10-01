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

/**
 * @param revealed        the bids have landed (the BIDS beat has been reached)
 * @param winnerRevealed  the auction has been called (the RECAP beat)
 *
 * Two gates, not one. Between them the column shows every bid with none of them
 * marked, which is the state the bids beat exists to show: the seats have
 * responded and the outcome is not yet claimed. `winnerRevealed` defaults to
 * `revealed` so a caller that passes only the old prop gets the old behaviour
 * rather than a column that never names a winner.
 */
export function OffersPanel({ viewModel, revealed, winnerRevealed, auctionFault }) {
  const offers = viewModel?.offers ?? [];
  const bidRows = viewModel?.bidRows ?? [];
  const groups = viewModel?.groups ?? [];
  const winner = viewModel?.winner ?? null;
  const showWinner = winnerRevealed === undefined ? revealed : winnerRevealed;

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
            <>
              {/*
                The bids, on their own rows. Nothing that did not bid appears
                between them: a summary line interleaved with real offers breaks
                the scan the column exists to support.
              */}
              {bidRows.length > 0 ? (
                <div className="th-rows" data-testid="offers-panel-bids">
                  {bidRows.map((offer) => (
                    <OfferRow
                      key={offer.key}
                      offer={offer}
                      // Unmarked until the auction is called, so the bids beat
                      // cannot give the winner away a step early.
                      isWinner={showWinner && winner != null && winner.offerKey === offer.key}
                      outcomeRevealed={showWinner}
                    />
                  ))}
                </div>
              ) : (
                <div className="th-empty" data-testid="offers-panel-no-bids">
                  No seat bid on this impression
                </div>
              )}

              {/*
                Everything that did not bid, grouped, AFTER all of the bids.
                `<details>` rather than component state: it is keyboard operable and
                screen-reader announced without any of that being implemented here,
                and the group's own count is visible while collapsed, so nothing is
                hidden — only folded.
              */}
              {groups.map((group) => (
                <details
                  key={group.id}
                  className={`th-offers-group th-offers-group-${group.id}`}
                  data-testid={`offers-group-${group.id}`}
                >
                  <summary className="th-offers-group-summary">
                    <span className="th-offers-group-label">{group.label}</span>
                    <span className="th-offers-group-breakdown">
                      {group.breakdown.map((b) => `${b.count} ${b.label}`).join(" · ")}
                    </span>
                  </summary>
                  <div className="th-rows th-offers-group-rows">
                    {/*
                      These keep their outcome even before the auction is called,
                      and that is not an inconsistency with the bid rows above: a
                      campaign that was suppressed or ruled ineligible was decided
                      by ARTF or the demand endpoint BEFORE the auction ran. Only
                      won/lost is an auction result, and nothing here won or lost.
                    */}
                    {group.rows.map((offer) => (
                      <OfferRow key={offer.key} offer={offer} isWinner={false} />
                    ))}
                  </div>
                </details>
              ))}
            </>
          )}

          {/*
            The outcome is the winning DEAL and the campaign holding it, with the
            cleared price as supporting detail (FR-28, BR-16). Leading with price
            would frame this as an ordinary price auction and bury the sell-side
            decision that produced the result.

            A null winner is an explicit unsold state, distinct from an empty or
            missing one (BR-17) — an empty render looks like a failed lookup.
          */}
          {!showWinner ? (
            /*
              Bids in, outcome not yet claimed. This states that the auction has
              not been called rather than rendering nothing in the winner's place,
              for the same reason the unsold state is explicit: an empty slot where
              an outcome belongs reads as a failed lookup.
            */
            <div className="th-offers-pending-winner" data-testid="offers-panel-pending-winner">
              <span className="th-row-key">Auction not yet called</span>
              <span className="th-row-val">
                {bidRows.length === 1
                  ? "1 bid is in — the winner is announced next"
                  : `${bidRows.length} bids are in — the winner is announced next`}
              </span>
            </div>
          ) : winner != null ? (
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
