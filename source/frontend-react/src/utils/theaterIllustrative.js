// theaterIllustrative.js — fixture values for the buy-side column.
//
// This system transforms bid requests. It has no auction, no bidder set and no
// clearing computation on the request path, so nothing here can be measured
// assessment of what building a real one would take).
//
// These values are therefore a labelled worked example, permitted only because
// the surface that shows them declares itself illustrative. They are:
//   - deterministic, so the same scenario always shows the same thing,
//   - inspectable, being written here rather than generated,
//   - never derived from, or mixed into, anything read from a real response.
//
// TheaterBuySidePanel renders its illustrative notice unconditionally.

/** Keyed by scenario id, with a default for scenarios that have no entry. */
const BY_SCENARIO = Object.freeze({
  "yield-optimizer": {
    bidders: [
      { name: "Northlake Home Goods", kind: "Performance · retail", bid: 3.10 },
      { name: "Vantage Motorsport", kind: "Brand · automotive", bid: null },
      { name: "Cedar & Co Furnishings", kind: "Brand · home, deal-matched", bid: 6.35 },
    ],
    winnerIndex: 2,
    clearingPrice: 6.35,
  },
  // Consistent with the fixture's real floors: imp floor 2.60, deals at 3.40 /
  // 2.10 / 0.85. The deal-matched bidder clears above the premium deal's floor,
  // as it would have to in order to win on that deal.
  "parenting-narrative": {
    bidders: [
      { name: "Harbour & Vale Nursery", kind: "Brand · family, deal-matched", bid: 4.15 },
      { name: "Northlake Home Goods", kind: "Performance · retail", bid: 2.85 },
      { name: "Vantage Motorsport", kind: "Brand · automotive", bid: null },
    ],
    winnerIndex: 0,
    clearingPrice: 3.40,
  },
  "bid-shading": {
    bidders: [
      { name: "Northlake Home Goods", kind: "Performance · retail", bid: 2.40 },
      { name: "Vantage Motorsport", kind: "Brand · automotive", bid: 2.95 },
    ],
    winnerIndex: 1,
    clearingPrice: 2.45,
  },
});

const DEFAULT_OUTCOME = Object.freeze({
  bidders: [
    { name: "Northlake Home Goods", kind: "Performance · retail", bid: 2.10 },
    { name: "Vantage Motorsport", kind: "Brand · automotive", bid: null },
    { name: "Cedar & Co Furnishings", kind: "Brand · home", bid: 2.80 },
  ],
  winnerIndex: 2,
  clearingPrice: 2.15,
});

export function illustrativeOutcomeFor(scenarioId) {
  const found = BY_SCENARIO[scenarioId] ?? DEFAULT_OUTCOME;
  return { ...found, isIllustrative: true };
}

export { BY_SCENARIO, DEFAULT_OUTCOME };
