// outcomeComparison.test.js — the two auctions side by side, as derivation.
//
// Property tests (PBT-03/07): the comparison states a delta only when both sides
// carry a number, the delta's sign follows the ordering, a missing baseline is
// reported as missing with the fault's own words, and no cell is ever empty.
// Example tests pin the concrete rows a reader sees.

import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  compareOutcomes,
  baselineFacts,
  selectRows,
  SHORT_ROWS,
  FULL_ROWS,
  NOT_REPORTED,
} from "./outcomeComparison.js";

/* ------------------------------------------------------------------ helpers */

/** An offers view model with a winner at `price` and `n` other offers. */
function vm({ sold = true, price = 4.2, campaign = "Cedar & Co", dealId = "deal-a", offers = 2, stopped = 0, notice = "live" } = {}) {
  const bidRows = [];
  if (sold) {
    bidRows.push({ key: "w", offered: true, price, campaignName: campaign, dealId, markedWinner: true });
  }
  for (let i = 0; i < offers; i++) {
    bidRows.push({ key: `o${i}`, offered: true, price: 1 + i, campaignName: `Other ${i}`, dealId });
  }
  const groups = stopped > 0
    ? [{ id: "decision", rows: Array.from({ length: stopped }, (_, i) => ({ key: `s${i}`, dealId })) }]
    : [];
  return {
    bidRows,
    groups,
    winner: sold ? { offerKey: "w", campaignName: campaign, dealId, clearedPrice: price } : null,
    notice,
  };
}

const response = (hop) => ({ artf_meta: { hop_ms: hop } });

const CONTEXT = { deals: [{ id: "deal-a", bidFloor: 2.0 }] };
const FLOOR_VALUE = { kind: "floor", dealId: "deal-a", before: 2.0, after: 2.6, displayLabel: "Yield Optimizer" };
const MARGIN_VALUE = { kind: "margin", dealId: "deal-a", value: 0.1, calculationType: "PERCENT" };

const byId = (c) => Object.fromEntries(c.rows.map((r) => [r.id, r]));

/* ------------------------------------------------------------- examples */

describe("compareOutcomes — the rows a reader sees", () => {
  it("compares two sold auctions and states the price difference with its sign", () => {
    const c = compareOutcomes({
      baselineVM: vm({ price: 3.1, offers: 2 }),
      baselineResponse: response(40),
      artfVM: vm({ price: 4.2, offers: 3, stopped: 1 }),
      artfResponse: response(55),
      values: [FLOOR_VALUE, MARGIN_VALUE],
      context: CONTEXT,
    });
    expect(c.available).toBe(true);
    expect(c.reason).toBeNull();
    const rows = byId(c);
    expect(rows.outcome).toMatchObject({ without: "sold", with: "sold" });
    expect(rows.cleared).toMatchObject({ without: "$3.10", with: "$4.20", delta: "+$1.10" });
    expect(rows.offers).toMatchObject({ without: "3", with: "4", delta: "+1" });
    expect(rows.stopped).toMatchObject({ without: "0", with: "1", delta: "+1" });
    expect(rows.roundtrip).toMatchObject({ without: "40 ms", with: "55 ms", delta: "+15 ms" });
    expect(rows.floor).toMatchObject({ without: "$2.00", with: "$2.60 set by ARTF", delta: "+$0.60" });
    expect(rows.floor.detail).toBe("set by Yield Optimizer");
    expect(rows.margin.with).toBe("$0.20 (10.0%)");
    expect(rows.margin.without).toBe("none");
  });

  it("a lower price with ARTF is a negative delta, not an error", () => {
    const c = compareOutcomes({ baselineVM: vm({ price: 5 }), artfVM: vm({ price: 4.5 }) });
    expect(byId(c).cleared.delta).toBe("−$0.50");
  });

  it("an unsold side reads unsold, and no price delta is stated", () => {
    const c = compareOutcomes({ baselineVM: vm({ sold: false, offers: 0 }), artfVM: vm({ price: 4 }) });
    const rows = byId(c);
    expect(rows.outcome).toMatchObject({ without: "unsold", with: "sold" });
    expect(rows.cleared.delta).toBeNull();
    expect(rows.cleared.detail).toBe("a difference needs a price on both sides");
    expect(rows.deal.without).toBe("no deal on this win");
  });

  it("reports a missing baseline in the fault's own words and marks the comparison unavailable", () => {
    const c = compareOutcomes({
      baselineVM: null,
      baselineFault: { kind: "not_deployed", detail: "Prebid Server is not deployed." },
      artfVM: vm(),
    });
    expect(c.available).toBe(false);
    expect(c.reason).toBe("baseline unavailable: Prebid Server is not deployed.");
    // The with side still has its facts; the without side has none to show.
    expect(byId(c).cleared.with).toBe("$4.20");
    expect(byId(c).cleared.without).toBe(NOT_REPORTED);
    expect(byId(c).cleared.delta).toBeNull();
  });

  it("a baseline not yet read is distinguished from one that failed", () => {
    const c = compareOutcomes({ baselineVM: null, baselineFault: null, artfVM: vm() });
    expect(c.available).toBe(false);
    expect(c.reason).toBe("baseline not yet read");
  });

  it("inherits the fixture label from the with-ARTF side", () => {
    const c = compareOutcomes({ baselineVM: vm(), artfVM: vm({ notice: "illustrative" }) });
    expect(c.illustrative).toBe(true);
  });

  it("a price the response does not carry is 'not reported', not zero", () => {
    const sold = vm();
    sold.winner.clearedPrice = null;
    const c = compareOutcomes({ baselineVM: sold, artfVM: vm() });
    expect(byId(c).cleared.without).toBe(NOT_REPORTED);
    expect(byId(c).cleared.delta).toBeNull();
  });

  it("round trip is 'not reported' when artf_meta carries no hop_ms", () => {
    const c = compareOutcomes({ baselineVM: vm(), baselineResponse: {}, artfVM: vm(), artfResponse: response(12) });
    expect(byId(c).roundtrip).toMatchObject({ without: NOT_REPORTED, with: "12 ms", delta: null });
  });
});

describe("selectRows", () => {
  it("returns the short and full sets in order, and only ids that exist", () => {
    const c = compareOutcomes({ baselineVM: vm(), artfVM: vm() });
    expect(selectRows(c, SHORT_ROWS).map((r) => r.id)).toEqual([...SHORT_ROWS]);
    expect(selectRows(c, FULL_ROWS).map((r) => r.id)).toEqual([...FULL_ROWS]);
    expect(selectRows(c, ["outcome", "nope"]).map((r) => r.id)).toEqual(["outcome"]);
    expect(selectRows(null, SHORT_ROWS)).toEqual([]);
  });
});

describe("baselineFacts", () => {
  it("reduces a sold baseline to the four facts the summary needs", () => {
    expect(baselineFacts({ baselineVM: vm({ price: 3.1 }) })).toEqual({
      sold: true, campaign: "Cedar & Co", dealId: "deal-a", clearedPrice: 3.1,
    });
  });
  it("carries the fault's reason as `unavailable`, and is null with nothing to say", () => {
    expect(baselineFacts({ baselineFault: { kind: "failed", detail: "HTTP 502" } }))
      .toEqual({ unavailable: "baseline unavailable: HTTP 502" });
    expect(baselineFacts({})).toBeNull();
  });
});

/* ------------------------------------------------------------ properties */

const arbPrice = fc.double({ min: 0.01, max: 99.99, noNaN: true });
const arbSide = fc.record({
  sold: fc.boolean(),
  price: arbPrice,
  offers: fc.integer({ min: 0, max: 6 }),
  stopped: fc.integer({ min: 0, max: 4 }),
});
const arbHop = fc.option(fc.integer({ min: 0, max: 5000 }), { nil: null });

describe("compareOutcomes — properties", () => {
  it("states a cleared-price delta iff both sides sold, and its sign follows the ordering", () => {
    fc.assert(fc.property(arbSide, arbSide, (w, a) => {
      const c = compareOutcomes({ baselineVM: vm(w), artfVM: vm(a) });
      const row = byId(c).cleared;
      if (w.sold && a.sold) {
        expect(row.delta).not.toBeNull();
        const diff = a.price - w.price;
        // The sign follows the displayed two-decimal magnitude: a difference that
        // rounds to $0.00 is ±, whatever its sub-cent sign.
        const expectedSign = Math.abs(diff).toFixed(2) === "0.00" ? "±" : diff > 0 ? "+" : "−";
        expect(row.delta[0]).toBe(expectedSign);
      } else {
        expect(row.delta).toBeNull();
      }
    }), { numRuns: 300 });
  });

  it("offer and stopped counts always carry a delta when both sides exist, equal to the difference", () => {
    fc.assert(fc.property(arbSide, arbSide, (w, a) => {
      const c = compareOutcomes({ baselineVM: vm(w), artfVM: vm(a) });
      const rows = byId(c);
      const wOffers = w.offers + (w.sold ? 1 : 0);
      const aOffers = a.offers + (a.sold ? 1 : 0);
      expect(rows.offers.without).toBe(String(wOffers));
      expect(rows.offers.with).toBe(String(aOffers));
      const d = aOffers - wOffers;
      expect(rows.offers.delta).toBe(`${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d)}`);
      expect(rows.stopped.delta).toMatch(/^[+−±]\d+$/);
    }), { numRuns: 200 });
  });

  it("round-trip delta exists iff both responses carry hop_ms", () => {
    fc.assert(fc.property(arbHop, arbHop, (wh, ah) => {
      const c = compareOutcomes({
        baselineVM: vm(), baselineResponse: wh == null ? {} : response(wh),
        artfVM: vm(), artfResponse: ah == null ? {} : response(ah),
      });
      const row = byId(c).roundtrip;
      expect(row.delta != null).toBe(wh != null && ah != null);
    }), { numRuns: 200 });
  });

  it("no cell is ever an empty string, and every row id is one of the full set", () => {
    fc.assert(fc.property(arbSide, arbSide, fc.boolean(), (w, a, withBaseline) => {
      const c = compareOutcomes({
        baselineVM: withBaseline ? vm(w) : null,
        baselineFault: withBaseline ? null : { kind: "failed", detail: "x" },
        artfVM: vm(a),
      });
      for (const row of c.rows) {
        expect(FULL_ROWS).toContain(row.id);
        expect(typeof row.without).toBe("string");
        expect(row.without.length).toBeGreaterThan(0);
        expect(typeof row.with).toBe("string");
        expect(row.with.length).toBeGreaterThan(0);
      }
      expect(c.available).toBe(withBaseline);
    }), { numRuns: 200 });
  });

  it("is deterministic and never throws on junk", () => {
    fc.assert(fc.property(fc.anything(), (junk) => {
      expect(() => compareOutcomes(junk && typeof junk === "object" && !Array.isArray(junk) ? junk : {})).not.toThrow();
    }), { numRuns: 100 });
    const args = { baselineVM: vm(), artfVM: vm({ price: 1 }) };
    expect(compareOutcomes(args)).toEqual(compareOutcomes(args));
  });
});


/* ------------------------------------------------ the ARTF mutations row */

describe("compareOutcomes — the ARTF mutations row", () => {
  const bypassed = (hop) => ({ artf_meta: { hop_ms: hop, artf_mutations: "bypassed" } });
  const requested = (hop) => ({ artf_meta: { hop_ms: hop, artf_mutations: "requested" } });

  it("is the first row on both surfaces", () => {
    expect(SHORT_ROWS[0]).toBe("mutations");
    expect(FULL_ROWS[0]).toBe("mutations");
  });

  it("says the baseline was bypassed and names the containers that applied changes", () => {
    const c = compareOutcomes({
      baselineVM: vm({ price: 3.1 }),
      baselineResponse: bypassed(40),
      artfVM: vm({ price: 4.2 }),
      artfResponse: requested(55),
      values: [
        { ...FLOOR_VALUE, containerName: "ncf-deal-manager", displayLabel: "Deal Scorer" },
        { kind: "ids", role: "segments", ids: ["350"], containerName: "widedeep", displayLabel: "Audience Activator" },
        { kind: "ids", role: "deals-activated", ids: ["deal-a"], containerName: "ncf-deal-manager", displayLabel: "Deal Scorer" },
      ],
      context: CONTEXT,
    });
    const row = byId(c).mutations;
    expect(row).toMatchObject({
      label: "ARTF mutations",
      without: "bypassed (no container consulted)",
      with: "applied by 2 containers",
      delta: null,
    });
    expect(row.detail).toBe("Deal Scorer, Audience Activator");
  });

  it("falls back to the orchestrator's own word when no values are on screen", () => {
    const c = compareOutcomes({
      baselineVM: vm(), baselineResponse: bypassed(1),
      artfVM: vm(), artfResponse: requested(2),
      values: [], context: CONTEXT,
    });
    expect(byId(c).mutations).toMatchObject({ without: "bypassed (no container consulted)", with: "requested", detail: null });
  });

  it("surfaces a baseline that was NOT bypassed rather than assuming it was", () => {
    // If the marker had been lost in transit, the orchestrator would report
    // "requested" on the baseline. That is the one thing this row must never hide.
    const c = compareOutcomes({
      baselineVM: vm(), baselineResponse: requested(1),
      artfVM: vm(), artfResponse: requested(2),
      values: [], context: CONTEXT,
    });
    expect(byId(c).mutations.without).toBe("requested");
  });

  it("reports an older orchestrator that does not say as not reported", () => {
    const c = compareOutcomes({
      baselineVM: vm(), baselineResponse: response(1),
      artfVM: vm(), artfResponse: response(2),
      values: [], context: CONTEXT,
    });
    expect(byId(c).mutations).toMatchObject({ without: NOT_REPORTED, with: NOT_REPORTED });
  });

  it("uses one container label when every value came from the same container", () => {
    const c = compareOutcomes({
      baselineVM: vm(), baselineResponse: bypassed(1),
      artfVM: vm(), artfResponse: requested(2),
      values: [{ ...FLOOR_VALUE, displayLabel: "Deal Scorer" }, { ...MARGIN_VALUE, displayLabel: "Deal Scorer" }],
      context: CONTEXT,
    });
    expect(byId(c).mutations.with).toBe("applied by 1 container");
  });
});

describe("compareOutcomes — the floor row on an activated deal", () => {
  it("says the baseline never saw a deal the request did not offer", () => {
    const c = compareOutcomes({
      baselineVM: vm({ dealId: null, campaign: "Open Market" }),
      baselineResponse: response(1),
      artfVM: vm({ dealId: "deal-activated" }),
      artfResponse: response(2),
      values: [{ kind: "floor", dealId: "deal-activated", before: null, after: 3.4, displayLabel: "Deal Scorer" }],
      context: CONTEXT, // knows deal-a only
    });
    expect(byId(c).floor).toMatchObject({
      without: "deal not offered",
      with: "$3.40 set by ARTF",
      delta: null,
      detail: "set by Deal Scorer",
    });
  });
});
