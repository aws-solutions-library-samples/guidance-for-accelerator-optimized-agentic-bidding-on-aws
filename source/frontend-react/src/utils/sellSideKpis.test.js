import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  deriveSellSideKpis,
  effectiveFloor,
  publisherTake,
} from "./sellSideKpis.js";

const row = (id, label) => (rows) => rows.find((r) => r.id === id && (!label || r.label === label));
const valueOf = (kpis, id) => kpis.rows.find((r) => r.id === id)?.value ?? null;

const WINNER = {
  dealId: "deal-parenting-premium",
  campaignId: "c-1",
  campaignName: "brightstart.example",
  clearedPrice: 4.1,
  offerKey: "bid:amt:b1:deal-parenting-premium",
};

function viewModel(over = {}) {
  return {
    offers: [],
    bidRows: [],
    groups: [],
    winner: WINNER,
    notice: "declared_inventory",
    currency: "USD",
    ...over,
  };
}

const CONTEXT = {
  deals: [
    { id: "deal-parenting-premium", bidFloor: 3.4, auctionType: 1 },
    { id: "deal-family-network", bidFloor: 2.1, auctionType: 1 },
  ],
};

describe("effectiveFloor", () => {
  it("prefers a container's adjusted floor over the floor as sent, and names the container", () => {
    const values = [{
      kind: "floor", dealId: "deal-parenting-premium", before: 3.4, after: 3.9,
      displayLabel: "Yield Optimizer — Floor", containerName: "yield-optimizer-floor",
    }];
    const floor = effectiveFloor(values, CONTEXT, "deal-parenting-premium");
    expect(floor).toEqual({
      value: 3.9,
      source: "adjusted",
      containerLabel: "Yield Optimizer — Floor",
    });
  });

  it("falls back to the floor the exchange sent when no container adjusted it", () => {
    expect(effectiveFloor([], CONTEXT, "deal-parenting-premium")).toEqual({
      value: 3.4, source: "as_sent", containerLabel: null,
    });
  });

  it("reports an unknown floor as unknown rather than zero", () => {
    expect(effectiveFloor([], CONTEXT, "deal-that-was-never-sent")).toEqual({
      value: null, source: "unknown", containerLabel: null,
    });
  });
});

describe("publisherTake", () => {
  it("scales the original floor for a PERCENT margin", () => {
    const values = [{ kind: "margin", dealId: "deal-parenting-premium", value: 0.1, calculationType: "PERCENT" }];
    // Mirrors deal_yield_feedback._effective_price_and_revenue: PERCENT scales the
    // original floor. 3.4 * 0.1.
    expect(publisherTake(values, CONTEXT, "deal-parenting-premium").amount).toBeCloseTo(0.34, 10);
  });

  it("takes a CPM margin as an absolute amount", () => {
    const values = [{ kind: "margin", dealId: "deal-parenting-premium", value: 0.5, calculationType: "CPM" }];
    expect(publisherTake(values, CONTEXT, "deal-parenting-premium").amount).toBe(0.5);
  });

  it("does not compute a percentage of an unrecorded floor", () => {
    const values = [{ kind: "margin", dealId: "unknown-deal", value: 0.1, calculationType: "PERCENT" }];
    const take = publisherTake(values, CONTEXT, "unknown-deal");
    expect(take.amount).toBeNull();
    expect(take.display).toMatch(/not recorded/);
  });

  it("is null when no margin decision targeted the deal", () => {
    expect(publisherTake([], CONTEXT, "deal-parenting-premium")).toBeNull();
  });
});

describe("deriveSellSideKpis", () => {
  it("states the unsold outcome rather than rendering empty", () => {
    const kpis = deriveSellSideKpis({ values: [], context: CONTEXT, viewModel: viewModel({ winner: null }) });
    expect(kpis.sold).toBe(false);
    expect(kpis.rows).toEqual([]);
  });

  it("reports the campaign, the deal and the cleared price from the auction", () => {
    const kpis = deriveSellSideKpis({ values: [], context: CONTEXT, viewModel: viewModel() });
    expect(valueOf(kpis, "campaign")).toBe("brightstart.example");
    expect(valueOf(kpis, "deal")).toBe("deal-parenting-premium");
    expect(valueOf(kpis, "cleared")).toBe("$4.10");
  });

  it("measures headroom against the ADJUSTED floor when a container moved it", () => {
    const values = [{
      kind: "floor", dealId: "deal-parenting-premium", before: 3.4, after: 3.9,
      displayLabel: "Yield Optimizer — Floor",
    }];
    const kpis = deriveSellSideKpis({ values, context: CONTEXT, viewModel: viewModel() });
    // 4.10 - 3.90, not 4.10 - 3.40.
    expect(valueOf(kpis, "above-floor")).toBe("+$0.20");
    expect(row("above-floor")(kpis.rows).detail).toMatch(/adjusted floor/);
    expect(valueOf(kpis, "floor-move")).toBe("+$0.50");
  });

  it("measures headroom against the floor as sent when nothing moved it", () => {
    const kpis = deriveSellSideKpis({ values: [], context: CONTEXT, viewModel: viewModel() });
    expect(valueOf(kpis, "above-floor")).toBe("+$0.70");
    expect(row("above-floor")(kpis.rows).detail).toMatch(/as sent/);
  });

  it("does not compute a floor movement against an unrecorded baseline", () => {
    const values = [{ kind: "floor", dealId: "deal-parenting-premium", before: null, after: 3.9 }];
    const kpis = deriveSellSideKpis({ values, context: CONTEXT, viewModel: viewModel() });
    expect(valueOf(kpis, "floor-move")).toBe("not recorded");
  });

  it("omits headroom entirely when the cleared price is not reported", () => {
    const vm = viewModel({ winner: { ...WINNER, clearedPrice: null } });
    const kpis = deriveSellSideKpis({ values: [], context: CONTEXT, viewModel: vm });
    expect(valueOf(kpis, "cleared")).toBe("not reported");
    expect(kpis.rows.some((r) => r.id === "above-floor")).toBe(false);
  });

  it("counts bids on the winning deal, and only offers", () => {
    const vm = viewModel({
      bidRows: [
        { key: "a", offered: true, dealId: "deal-parenting-premium" },
        { key: "b", offered: true, dealId: "deal-parenting-premium" },
        { key: "c", offered: true, dealId: "deal-family-network" },
        { key: "d", offered: false, dealId: "deal-parenting-premium" },
      ],
    });
    expect(valueOf(deriveSellSideKpis({ values: [], context: CONTEXT, viewModel: vm }), "bids")).toBe("2");
  });

  it("carries the illustrative marker through from the view model", () => {
    const kpis = deriveSellSideKpis({
      values: [], context: CONTEXT, viewModel: viewModel({ notice: "illustrative" }),
    });
    expect(kpis.illustrative).toBe(true);
  });

  it("states no rate, aggregate or time-series KPI", () => {
    // No counter, aggregate store or read path for any of these exists in this
    // system, so any of them on this panel would be invented.
    const values = [
      { kind: "floor", dealId: "deal-parenting-premium", before: 3.4, after: 3.9 },
      { kind: "margin", dealId: "deal-parenting-premium", value: 0.1, calculationType: "PERCENT" },
    ];
    const kpis = deriveSellSideKpis({ values, context: CONTEXT, viewModel: viewModel() });
    const labels = kpis.rows.map((r) => r.label.toLowerCase()).join(" | ");
    for (const banned of ["fill rate", "sell-through", "win rate", "ecpm", "avails", "pacing", "impressions"]) {
      expect(labels).not.toContain(banned);
    }
  });
});

describe("deriveSellSideKpis properties", () => {
  // Invariant: a KPI row never states a numeric value the inputs cannot support.
  // Concretely, headroom is always clearedPrice minus the effective floor — so
  // recomputing it from the row must reproduce the inputs' arithmetic.
  it("headroom always equals cleared price minus the effective floor", () => {
    fc.assert(fc.property(
      fc.double({ min: 0.01, max: 50, noNaN: true }),
      fc.double({ min: 0.01, max: 50, noNaN: true }),
      fc.option(fc.double({ min: 0.01, max: 50, noNaN: true }), { nil: undefined }),
      (cleared, sentFloor, adjusted) => {
        const dealId = "deal-x";
        const context = { deals: [{ id: dealId, bidFloor: sentFloor, auctionType: 1 }] };
        const values = adjusted === undefined
          ? []
          : [{ kind: "floor", dealId, before: sentFloor, after: adjusted }];
        const vm = viewModel({ winner: { ...WINNER, dealId, clearedPrice: cleared } });

        const kpis = deriveSellSideKpis({ values, context, viewModel: vm });
        const shown = kpis.rows.find((r) => r.id === "above-floor")?.value;
        const expected = cleared - (adjusted === undefined ? sentFloor : adjusted);
        const sign = expected >= 0 ? "+" : "−";
        expect(shown).toBe(`${sign}$${Math.abs(expected).toFixed(2)}`);
      },
    ));
  });

  it("never emits a row whose value is undefined or an empty string", () => {
    fc.assert(fc.property(
      fc.option(fc.double({ min: 0.01, max: 50, noNaN: true }), { nil: null }),
      fc.option(fc.string({ minLength: 1, maxLength: 12 }), { nil: null }),
      (cleared, campaignName) => {
        const vm = viewModel({ winner: { ...WINNER, clearedPrice: cleared, campaignName } });
        const kpis = deriveSellSideKpis({ values: [], context: CONTEXT, viewModel: vm });
        for (const r of kpis.rows) {
          expect(typeof r.value).toBe("string");
          expect(r.value.length).toBeGreaterThan(0);
        }
      },
    ));
  });
});
