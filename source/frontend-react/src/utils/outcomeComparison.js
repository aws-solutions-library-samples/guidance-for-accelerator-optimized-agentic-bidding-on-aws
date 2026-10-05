// outcomeComparison.js — the two auctions side by side: without ARTF, with ARTF.
//
// The Theater runs the same bid request through Prebid twice. Pass 1 has the ARTF
// extension point propose nothing; pass 2 lets the containers mutate the request
// before the seats bid. This module puts the two results next to each other.
//
// The boundary is the same one sellSideKpis.js keeps: every cell is read from one
// of the two Prebid responses, from the submitted request, or from a mutation
// value already on screen, or is arithmetic over those. A value a response does
// not carry is "not reported", never zero; a delta is stated only when both of
// its operands are numbers (FR-18, TP-9). No baseline is ever invented: when the
// pass-1 auction did not run, the comparison says so and shows nothing in its
// place (TP-7).
//
// Pure. No I/O, no clock, no randomness.

import { effectiveFloor, publisherTake } from "./sellSideKpis.js";

/** Rows shown in the left column at the recap: the headline set. */
export const SHORT_ROWS = Object.freeze(["mutations", "outcome", "cleared", "deal", "offers"]);

/** Rows shown under the run summary: everything. */
export const FULL_ROWS = Object.freeze([
  "mutations",
  "outcome",
  "campaign",
  "deal",
  "cleared",
  "offers",
  "stopped",
  "roundtrip",
  "floor",
  "margin",
]);

export const NOT_REPORTED = "not reported";

function isNum(n) {
  return typeof n === "number" && Number.isFinite(n);
}

function money(n) {
  return isNum(n) ? `$${n.toFixed(2)}` : null;
}

/**
 * A signed difference. The sign follows the DISPLAYED magnitude: a difference
 * that rounds to the formatter's zero is shown as ±, not as "+$0.00", which would
 * assert a direction the digits do not support.
 */
function signed(n, fmt) {
  const magnitude = fmt(Math.abs(n));
  const sign = magnitude === fmt(0) ? "±" : n > 0 ? "+" : "−";
  return `${sign}${magnitude}`;
}

/** The facts one auction yields for this comparison, each null when absent. */
function sideOf(viewModel) {
  if (!viewModel || typeof viewModel !== "object") return null;
  const winner = viewModel.winner && typeof viewModel.winner === "object" ? viewModel.winner : null;
  const bidRows = Array.isArray(viewModel.bidRows) ? viewModel.bidRows : [];
  const groups = Array.isArray(viewModel.groups) ? viewModel.groups : [];
  const offered = bidRows.filter((o) => o?.offered === true).length;
  const decision = groups.find((g) => g?.id === "decision");
  return {
    sold: winner != null,
    campaign: winner ? (winner.campaignName ?? winner.campaignId ?? null) : null,
    dealId: winner ? (winner.dealId ?? null) : null,
    clearedPrice: winner && isNum(winner.clearedPrice) ? winner.clearedPrice : null,
    offered,
    stopped: decision && Array.isArray(decision.rows) ? decision.rows.length : 0,
    illustrative: viewModel.notice === "illustrative",
  };
}

function hopMs(bidResponse) {
  const ms = bidResponse?.artf_meta?.hop_ms;
  return isNum(ms) ? ms : null;
}

/**
 * What the orchestrator says it asked the ARTF extension point to do for this
 * auction: "bypassed" (the request carried the marker; no container was consulted)
 * or "requested". Null when the response does not say, which an older orchestrator
 * would not.
 */
function mutationMode(bidResponse) {
  const mode = bidResponse?.artf_meta?.artf_mutations;
  return typeof mode === "string" && mode ? mode : null;
}

/**
 * The containers whose values are on screen for the with-ARTF pass. Values are
 * stamped with their producing container by theaterBeats, so this is a count of
 * distinct producers, not of values.
 */
function containersApplied(values) {
  const names = new Set();
  for (const v of values) {
    const name = v?.displayLabel ?? v?.containerName;
    if (typeof name === "string" && name) names.add(name);
  }
  return names;
}

/**
 * Why the baseline column is empty, in the orchestrator's words where it gave
 * them. Null when the baseline is present.
 */
function baselineReason(baselineVM, baselineFault) {
  if (baselineVM) return null;
  if (baselineFault?.detail) return `baseline unavailable: ${baselineFault.detail}`;
  if (baselineFault) return "baseline unavailable";
  return "baseline not yet read";
}

/**
 * @param {object}   args
 * @param {object|null} args.baselineVM      buildOfferViewModel(pass-1 response), or null
 * @param {object|null} args.baselineResponse the raw pass-1 response (for hop_ms), or null
 * @param {object|null} args.baselineFault   { kind, detail } when pass 1 did not run
 * @param {object|null} args.artfVM          buildOfferViewModel(pass-2 response or fixture)
 * @param {object|null} args.artfResponse    the raw pass-2 response (for hop_ms), or null
 * @param {object[]}    args.values          visible mutation values (stamped with container)
 * @param {object|null} args.context         ScenarioContext
 * @returns {{
 *   available: boolean,
 *   reason: string|null,
 *   illustrative: boolean,
 *   rows: {id: string, label: string, without: string, with: string, delta: string|null, detail: string|null}[],
 * }}
 */
export function compareOutcomes({
  baselineVM = null,
  baselineResponse = null,
  baselineFault = null,
  artfVM = null,
  artfResponse = null,
  values = [],
  context = null,
} = {}) {
  if (!Array.isArray(values)) values = [];
  const without = sideOf(baselineVM);
  const withArtf = sideOf(artfVM);
  const reason = baselineReason(baselineVM, baselineFault);
  const available = without != null && withArtf != null;
  // The with-ARTF side may be the labelled fixture when Prebid is absent. The
  // comparison then inherits the label; it never drops it.
  const illustrative = withArtf?.illustrative === true;

  const rows = [];
  const push = (id, label, w, a, delta = null, detail = null) => {
    rows.push({
      id,
      label,
      without: w ?? NOT_REPORTED,
      with: a ?? NOT_REPORTED,
      delta,
      detail,
    });
  };

  // Stated first, because every other row is read in its light: a difference
  // below is the mutations' doing, and an absence of difference with mutations
  // applied means the demand side did not react, not that nothing happened.
  const wMode = mutationMode(baselineResponse);
  const aMode = mutationMode(artfResponse);
  const applied = containersApplied(values);
  push(
    "mutations",
    "ARTF mutations",
    without
      ? (wMode === "bypassed"
        ? "bypassed (no container consulted)"
        : (wMode ?? NOT_REPORTED))
      : null,
    withArtf
      ? (applied.size > 0
        ? `applied by ${applied.size} container${applied.size === 1 ? "" : "s"}`
        : (aMode ?? NOT_REPORTED))
      : null,
    null,
    applied.size > 0 ? [...applied].join(", ") : null,
  );

  const soldText = (side) => (side == null ? null : side.sold ? "sold" : "unsold");
  push("outcome", "Outcome", soldText(without), soldText(withArtf));

  push("campaign", "Campaign served", without?.campaign ?? (without ? "none" : null), withArtf?.campaign ?? (withArtf ? "none" : null));

  push(
    "deal",
    "Deal transacted",
    without ? (without.dealId ?? "no deal on this win") : null,
    withArtf ? (withArtf.dealId ?? "no deal on this win") : null,
  );

  const wPrice = without?.clearedPrice ?? null;
  const aPrice = withArtf?.clearedPrice ?? null;
  push(
    "cleared",
    "Cleared at",
    without ? (money(wPrice) ?? (without.sold ? NOT_REPORTED : "—")) : null,
    withArtf ? (money(aPrice) ?? (withArtf.sold ? NOT_REPORTED : "—")) : null,
    isNum(wPrice) && isNum(aPrice) ? signed(aPrice - wPrice, money) : null,
    isNum(wPrice) && isNum(aPrice) ? null : "a difference needs a price on both sides",
  );

  push(
    "offers",
    "Offers received",
    without ? String(without.offered) : null,
    withArtf ? String(withArtf.offered) : null,
    without && withArtf ? signed(withArtf.offered - without.offered, (n) => String(n)) : null,
  );

  push(
    "stopped",
    "Campaigns stopped by a sell-side decision",
    without ? String(without.stopped) : null,
    withArtf ? String(withArtf.stopped) : null,
    without && withArtf ? signed(withArtf.stopped - without.stopped, (n) => String(n)) : null,
  );

  const wHop = hopMs(baselineResponse);
  const aHop = hopMs(artfResponse);
  push(
    "roundtrip",
    "Auction round trip",
    isNum(wHop) ? `${wHop} ms` : (without ? NOT_REPORTED : null),
    isNum(aHop) ? `${aHop} ms` : (withArtf ? NOT_REPORTED : null),
    isNum(wHop) && isNum(aHop) ? signed(aHop - wHop, (n) => `${n} ms`) : null,
    "orchestrator to Prebid and back, including the ARTF hook on the with side",
  );

  // Floor and margin exist only on the with-ARTF side: pass 1 consulted no
  // container, so its floor is the floor as sent, by construction. A deal the
  // request did not offer has no floor on the baseline side at all: the Deal
  // Scorer activated it, so the baseline never saw it.
  const dealId = withArtf?.dealId ?? null;
  const sent = (context?.deals ?? []).find((d) => d.id === dealId);
  const floor = effectiveFloor(values, context, dealId);
  push(
    "floor",
    "Floor on the transacted deal",
    without
      ? (money(sent?.bidFloor) ?? (dealId && !sent ? "deal not offered" : "as sent"))
      : null,
    withArtf
      ? (floor.source === "adjusted"
        ? `${money(floor.value)} set by ARTF`
        : (money(floor.value) ?? NOT_REPORTED))
      : null,
    floor.source === "adjusted" && isNum(sent?.bidFloor) && isNum(floor.value)
      ? signed(floor.value - sent.bidFloor, money)
      : null,
    floor.containerLabel ? `set by ${floor.containerLabel}` : null,
  );

  const take = publisherTake(values, context, dealId);
  push(
    "margin",
    "Publisher margin on this deal",
    without ? "none" : null,
    withArtf ? (take ? take.display : "none") : null,
  );

  return { available, reason, illustrative, rows };
}

/** The subset of rows for one surface, in the surface's order. */
export function selectRows(comparison, ids) {
  const byId = new Map((comparison?.rows ?? []).map((r) => [r.id, r]));
  return ids.map((id) => byId.get(id)).filter(Boolean);
}

/**
 * The baseline, reduced to what the run summary needs: who won the auction that
 * ran without ARTF and at what price, or why that is not known.
 *
 * Null when there is nothing to say yet (no response and no fault). Carries
 * `unavailable` with the reason when the pass did not run, so the prose can say
 * so instead of omitting the baseline silently.
 */
export function baselineFacts({ baselineVM = null, baselineFault = null } = {}) {
  const side = sideOf(baselineVM);
  if (side) {
    return {
      sold: side.sold,
      campaign: side.campaign,
      dealId: side.dealId,
      clearedPrice: side.clearedPrice,
    };
  }
  const reason = baselineReason(null, baselineFault);
  return baselineFault ? { unavailable: reason } : null;
}
