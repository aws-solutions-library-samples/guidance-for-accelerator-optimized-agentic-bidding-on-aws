// sellSideKpis.js — the sell-side result for the deal package that transacted.
//
// This is what the Prebid integration bought: before there was a real auction
// there was no cleared price, so there was no sell-side outcome to state at all.
//
// The boundary is narrow and deliberate. Every row below is read from the
// submitted request, a container mutation, or the resolved auction, or is
// arithmetic over those three. Nothing here is a RATE, an AGGREGATE or a TIME
// SERIES, because no counter, no aggregate store and no read path for one exists
// in this system — the one per-deal outcome record that is written
// (DealYieldOutcomeEvent) goes to Kinesis for training and carries won=False on
// live traffic. A fill rate or an eCPM on this panel would be invented.
//
// n=1. One impression cleared, so counts are stated as counts.
//
// Pure. No I/O, no clock, no randomness.

/** A value that was genuinely not available, distinct from zero. */
export const NOT_RECORDED = null;

function isNum(n) {
  return typeof n === "number" && Number.isFinite(n);
}

function money(n) {
  return isNum(n) ? `$${n.toFixed(2)}` : null;
}

/**
 * The floor decision a container made on this deal, if any.
 *
 * Values carry their producing container since theaterBeats stamps them, so the
 * row can name the container that moved the floor rather than reporting an
 * anonymous change.
 */
function floorDecisionFor(values, dealId) {
  if (!dealId) return null;
  return (values ?? []).find((v) => v.kind === "floor" && v.dealId === dealId) ?? null;
}

function marginDecisionFor(values, dealId) {
  if (!dealId) return null;
  return (values ?? []).find((v) => v.kind === "margin" && v.dealId === dealId) ?? null;
}

/**
 * The floor the auction actually faced on the winning deal, and where it came
 * from.
 *
 * A yield container's adjusted floor supersedes the one the exchange sent. Which
 * of the two was used is returned alongside the number, because "cleared $0.70
 * above the floor" means something different depending on whose floor that was.
 */
export function effectiveFloor(values, context, dealId) {
  const decision = floorDecisionFor(values, dealId);
  if (decision && isNum(decision.after)) {
    return {
      value: decision.after,
      source: "adjusted",
      containerLabel: decision.displayLabel ?? decision.containerName ?? null,
    };
  }
  const sent = (context?.deals ?? []).find((d) => d.id === dealId);
  if (sent && isNum(sent.bidFloor)) {
    return { value: sent.bidFloor, source: "as_sent", containerLabel: null };
  }
  return { value: NOT_RECORDED, source: "unknown", containerLabel: null };
}

/**
 * Publisher take on this deal, mirroring the server's own arithmetic in
 * deal_yield_feedback._effective_price_and_revenue: PERCENT scales the original
 * floor, CPM is an absolute amount. Margin is additive; floor is multiplicative.
 *
 * Returns null when no margin decision targeted this deal, or when a PERCENT
 * margin has no original floor to scale — a percentage of an unknown is unknown,
 * not zero.
 */
export function publisherTake(values, context, dealId) {
  const decision = marginDecisionFor(values, dealId);
  if (!decision || !isNum(decision.value)) return null;

  if (decision.calculationType === "PERCENT") {
    const sent = (context?.deals ?? []).find((d) => d.id === dealId);
    if (!sent || !isNum(sent.bidFloor)) {
      return { amount: NOT_RECORDED, display: `${(decision.value * 100).toFixed(1)}% of a floor that was not recorded` };
    }
    const amount = sent.bidFloor * decision.value;
    return { amount, display: `${money(amount)} (${(decision.value * 100).toFixed(1)}%)` };
  }
  return { amount: decision.value, display: `${money(decision.value)} CPM` };
}

/**
 * The sell-side result for this impression.
 *
 * @param {object}   args
 * @param {object[]} args.values    visible values (each stamped with its container)
 * @param {object}   args.context   ScenarioContext
 * @param {object}   args.viewModel offers view model
 * @returns {{
 *   sold: boolean,
 *   dealId: string|null,
 *   illustrative: boolean,
 *   rows: {id: string, label: string, value: string|null, detail: string|null}[],
 * }}
 */
export function deriveSellSideKpis({ values, context, viewModel }) {
  const winner = viewModel?.winner ?? null;
  const illustrative = viewModel?.notice === "illustrative";

  if (!winner) {
    return { sold: false, dealId: null, illustrative, rows: [] };
  }

  const dealId = winner.dealId ?? null;
  const rows = [];
  const push = (id, label, value, detail = null) => {
    rows.push({ id, label, value, detail });
  };

  push("campaign", "Campaign served", winner.campaignName ?? winner.campaignId ?? "not reported");
  push("deal", "Deal transacted", dealId ?? "no deal on this win");
  push(
    "cleared",
    "Cleared at",
    money(winner.clearedPrice) ?? "not reported",
    isNum(winner.clearedPrice) ? null : "the response carries no price for the winning bid",
  );

  const sent = (context?.deals ?? []).find((d) => d.id === dealId);
  push(
    "floor-sent",
    "Deal floor as sent",
    money(sent?.bidFloor) ?? "not recorded",
  );

  const floor = effectiveFloor(values, context, dealId);
  const decision = floorDecisionFor(values, dealId);

  if (decision) {
    push(
      "floor-adjusted",
      "Floor after the yield decision",
      money(decision.after) ?? "not recorded",
      floor.containerLabel ? `set by ${floor.containerLabel}` : null,
    );
    // A null baseline is a real unknown. Stating a movement against it would be
    // arithmetic on a value nothing measured.
    if (isNum(decision.before) && isNum(decision.after)) {
      const delta = decision.after - decision.before;
      push(
        "floor-move",
        "Floor movement",
        `${delta >= 0 ? "+" : "−"}${money(Math.abs(delta))}`,
        `from ${money(decision.before)}`,
      );
    } else {
      push("floor-move", "Floor movement", "not recorded", "no prior floor was recorded for this deal");
    }
  }

  if (isNum(winner.clearedPrice) && isNum(floor.value)) {
    const headroom = winner.clearedPrice - floor.value;
    push(
      "above-floor",
      "Cleared above the floor by",
      `${headroom >= 0 ? "+" : "−"}${money(Math.abs(headroom))}`,
      floor.source === "adjusted"
        ? "against the adjusted floor"
        : "against the floor as sent",
    );
  }

  const take = publisherTake(values, context, dealId);
  if (take) push("margin", "Publisher margin on this deal", take.display);

  // Counts, not rates. One impression cleared.
  const bidsOnDeal = (viewModel?.bidRows ?? []).filter(
    (o) => o.offered && dealId != null && o.dealId === dealId,
  ).length;
  if (dealId != null) {
    push("bids", "Bids on this deal", String(bidsOnDeal));
  }

  const stopped = (viewModel?.groups ?? []).find((g) => g.id === "decision");
  if (stopped && dealId != null) {
    const onThisDeal = stopped.rows.filter((r) => r.dealId === dealId).length;
    if (onThisDeal > 0) {
      push("stopped", "Campaigns stopped on this deal", String(onThisDeal), "by a sell-side decision");
    }
  }

  return { sold: true, dealId, illustrative, rows };
}
