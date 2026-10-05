// artfApplier.test.js — the frontend applier, pinned to the same vectors as
// shared/artf_applier.py (tests/test_artf_applier.py) and ArtfMutationApplier.java.
import { describe, expect, it } from "vitest";
import { applyMutationsToEnvelope, stringifyWithPointers } from "./artfApplier.js";

function envelope() {
  return {
    id: "home-001",
    bid_request: {
      id: "auction-1",
      cur: ["USD"],
      imp: [
        {
          id: "imp-1",
          banner: { w: 300, h: 250 },
          bidfloor: 2.5,
          bidfloorcur: "USD",
          pmp: { deals: [{ id: "deal-remnant-open", bidfloor: 1.0, at: 3 }] },
        },
      ],
      user: { id: "user-1", data: [{ id: "dmp", segment: [{ id: "seg-1" }] }] },
    },
  };
}

const activate = (ids, imp = "imp-1") => ({ intent: "ACTIVATE_DEALS", path: `/imp/${imp}`, payload: { id: ids } });
const suppress = (ids, imp = "imp-1") => ({ intent: "SUPPRESS_DEALS", path: `/imp/${imp}`, payload: { id: ids } });
const floor = (deal, bidfloor, imp = "imp-1") => ({ intent: "ADJUST_DEAL_FLOOR", path: `/imp/${imp}/deals/${deal}`, payload: { bidfloor } });
const segments = (ids) => ({ intent: "ACTIVATE_SEGMENTS", path: "/user/data/segment", payload: { id: ids } });
const metrics = (list, imp = "imp-1") => ({ intent: "ADD_METRICS", path: `/imp/${imp}/metric`, payload: { metric: list } });

describe("ACTIVATE_DEALS lands on imp.pmp.deals", () => {
  it("adds the activated deal and reports where it wrote", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [activate(["deal-home-premium"])]);
    expect(dispositions[0].applied).toBe(true);
    expect(out.bid_request.imp[0].pmp.deals.map((d) => d.id)).toEqual(["deal-remnant-open", "deal-home-premium"]);
    expect(dispositions[0].written).toEqual(["/bid_request/imp/0/pmp/deals/1"]);
    // The defect this replaces: the activated deal survives serialisation.
    expect(JSON.stringify(out)).toContain("deal-home-premium");
  });

  it("does not duplicate a deal already offered", () => {
    const { envelope: out } = applyMutationsToEnvelope(envelope(), [activate(["deal-remnant-open"])]);
    expect(out.bid_request.imp[0].pmp.deals).toHaveLength(1);
  });

  it("creates pmp on an impression without one", () => {
    const env = envelope();
    delete env.bid_request.imp[0].pmp;
    const { envelope: out } = applyMutationsToEnvelope(env, [activate(["deal-x"])]);
    expect(out.bid_request.imp[0].pmp.deals).toEqual([{ id: "deal-x" }]);
  });

  it("the Deal Scorer's activate-then-floor pair resolves onto the activated deal", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [
      activate(["deal-home-premium"]),
      floor("deal-home-premium", 3.0),
    ]);
    expect(dispositions.every((d) => d.applied)).toBe(true);
    expect(out.bid_request.imp[0].pmp.deals[1]).toEqual({ id: "deal-home-premium", bidfloor: 3.0, bidfloorcur: "USD" });
    expect(out.bid_request.imp[0].bidfloor).toBe(3.0);
  });

  it("a floor before its activation is rejected with a reason", () => {
    const { dispositions } = applyMutationsToEnvelope(envelope(), [floor("deal-home-premium", 3.0), activate(["deal-home-premium"])]);
    expect(dispositions[0].applied).toBe(false);
    expect(dispositions[0].reason).toContain("has no deal 'deal-home-premium'");
    expect(dispositions[1].applied).toBe(true);
  });
});

describe("Java vectors", () => {
  it("a deal floor mutation writes both the floor and the currency", () => {
    const env = envelope();
    delete env.bid_request.imp[0].bidfloorcur;
    const { envelope: out } = applyMutationsToEnvelope(env, [floor("deal-remnant-open", 4.0)]);
    const deal = out.bid_request.imp[0].pmp.deals[0];
    expect(deal.bidfloor).toBe(4.0);
    expect(deal.bidfloorcur).toBe("USD");
    expect(out.bid_request.imp[0].bidfloorcur).toBe("USD");
  });

  it("an impression floor is never lowered by a mutation", () => {
    const { envelope: out } = applyMutationsToEnvelope(envelope(), [floor("deal-remnant-open", 1.5)]);
    expect(out.bid_request.imp[0].pmp.deals[0].bidfloor).toBe(1.5);
    expect(out.bid_request.imp[0].bidfloor).toBe(2.5);
  });

  it("a non-positive floor is rejected rather than written and ignored", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [floor("deal-remnant-open", 0)]);
    expect(dispositions[0].applied).toBe(false);
    expect(dispositions[0].reason).toContain("not greater than zero");
    expect(out).toEqual(envelope());
  });

  it("suppression marks the deal rather than removing it", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [suppress(["deal-remnant-open"])]);
    expect(out.bid_request.imp[0].pmp.deals).toHaveLength(1);
    expect(out.bid_request.imp[0].pmp.deals[0].ext.artf.suppressed).toBe(true);
    expect(dispositions[0].written).toEqual(["/bid_request/imp/0/pmp/deals/0/ext"]);
  });

  it("a metric outside the OpenRTB range is rejected rather than clamped", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [metrics([{ type: "viewability", value: 1.5, vendor: "v" }])]);
    expect(dispositions[0].applied).toBe(false);
    expect(dispositions[0].reason).toContain("outside the [0, 1] range");
    expect(out.bid_request.imp[0].metric).toBeUndefined();
  });

  it("a metric inside the range is applied", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [metrics([{ type: "viewability", value: 0.8, vendor: "nvidia-artf" }])]);
    expect(out.bid_request.imp[0].metric).toEqual([{ type: "viewability", value: 0.8, vendor: "nvidia-artf" }]);
    expect(dispositions[0].written).toEqual(["/bid_request/imp/0/metric/0"]);
  });

  it("segment activation appends to user.data without discarding existing data", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [segments(["461", "460"])]);
    expect(out.bid_request.user.data[0].id).toBe("dmp");
    expect(out.bid_request.user.data[1]).toEqual({ name: "artf", segment: [{ id: "461" }, { id: "460" }] });
    expect(dispositions[0].written).toEqual(["/bid_request/user/data/1"]);
  });

  it("a mutation naming an unknown impression is rejected", () => {
    const { dispositions } = applyMutationsToEnvelope(envelope(), [activate(["d"], "nope")]);
    expect(dispositions[0].applied).toBe(false);
    expect(dispositions[0].reason).toContain("no impression with id 'nope'");
  });

  it("an unknown path is rejected with the permitted list", () => {
    const { dispositions } = applyMutationsToEnvelope(envelope(), [{ intent: "ADD_METRICS", path: "/site/cat", payload: {} }]);
    expect(dispositions[0].applied).toBe(false);
    expect(dispositions[0].reason).toContain("Permitted:");
  });

  it("an empty mutation list leaves the envelope untouched and produces no dispositions", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), []);
    expect(out).toEqual(envelope());
    expect(dispositions).toEqual([]);
  });

  it("inputs are not mutated", () => {
    const env = envelope();
    const snapshot = JSON.parse(JSON.stringify(env));
    applyMutationsToEnvelope(env, [activate(["deal-x"]), segments(["1"])]);
    expect(env).toEqual(snapshot);
  });
});

describe("margins", () => {
  const margin = (deal, value, calculation_type) => ({
    intent: "ADJUST_DEAL_MARGIN", path: `/imp/imp-1/deals/${deal}`, payload: { margin: { value, calculation_type } },
  });

  it("a PERCENT margin scales the deal floor by the fraction", () => {
    const { envelope: out } = applyMutationsToEnvelope(envelope(), [margin("deal-remnant-open", 0.5, 1)]);
    expect(out.bid_request.imp[0].pmp.deals[0].bidfloor).toBeCloseTo(1.5);
  });

  it("a CPM margin adds to the deal floor", () => {
    const { envelope: out } = applyMutationsToEnvelope(envelope(), [margin("deal-remnant-open", 0.75, 0)]);
    expect(out.bid_request.imp[0].pmp.deals[0].bidfloor).toBeCloseTo(1.75);
  });
});

describe("BID_SHADE on the response", () => {
  const withResponse = () => ({
    ...envelope(),
    bid_response: { seatbid: [{ seat: "artfhouse", bid: [{ id: "bid-1", impid: "imp-1", price: 6.35 }] }] },
  });
  const shade = (price, seat = "artfhouse", bid = "bid-1") => ({ intent: "BID_SHADE", path: `/seatbid/${seat}/bid/${bid}`, payload: { price } });

  it("replaces the price on the named bid", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(withResponse(), [shade(4.2)]);
    expect(dispositions[0].applied).toBe(true);
    expect(out.bid_response.seatbid[0].bid[0].price).toBe(4.2);
    expect(dispositions[0].written).toEqual(["/bid_response/seatbid/0/bid/0/price"]);
  });

  it("is rejected when the envelope carries no bid_response", () => {
    const { dispositions } = applyMutationsToEnvelope(envelope(), [shade(4.2)]);
    expect(dispositions[0].applied).toBe(false);
    expect(dispositions[0].reason).toContain("carries no bid_response");
  });

  it("is rejected for an unknown bid and leaves the response untouched", () => {
    const env = withResponse();
    const { envelope: out, dispositions } = applyMutationsToEnvelope(env, [shade(4.2, "artfhouse", "bid-9")]);
    expect(dispositions[0].applied).toBe(false);
    expect(out.bid_response).toEqual(env.bid_response);
  });
});

describe("stringifyWithPointers", () => {
  it("matches JSON.stringify(value, null, 2) byte for byte", () => {
    const { envelope: out } = applyMutationsToEnvelope(envelope(), [activate(["deal-home-premium"]), segments(["461"])]);
    const { text } = stringifyWithPointers(out);
    expect(text).toBe(JSON.stringify(out, null, 2));
  });

  it("maps a written pointer to the lines the node occupies", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [activate(["deal-home-premium"])]);
    const { text, ranges } = stringifyWithPointers(out);
    const range = ranges.get(dispositions[0].written[0]);
    const lines = text.split("\n").slice(range.start, range.end + 1);
    expect(lines.join("\n")).toContain('"id": "deal-home-premium"');
    expect(lines[0].trim()).toBe("{");
    expect(lines[lines.length - 1].trim()).toBe("}");
  });

  it("distinguishes two keys with the same name at different depths", () => {
    const { envelope: out, dispositions } = applyMutationsToEnvelope(envelope(), [segments(["461"])]);
    const { text, ranges } = stringifyWithPointers(out);
    const range = ranges.get(dispositions[0].written[0]); // /bid_request/user/data/1
    const block = text.split("\n").slice(range.start, range.end + 1).join("\n");
    expect(block).toContain('"name": "artf"');
    expect(block).not.toContain('"id": "dmp"');
  });

  it("renders empty arrays and objects inline as JSON.stringify does", () => {
    const v = { a: [], b: {}, c: [1, { d: null }] };
    expect(stringifyWithPointers(v).text).toBe(JSON.stringify(v, null, 2));
  });
});
