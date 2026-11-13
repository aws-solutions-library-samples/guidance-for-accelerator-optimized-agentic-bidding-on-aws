import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  buildBeats,
  mutationToValues,
  buildScenarioContext,
  isExplored,
  dealIdFromPath,
  visibleValues,
  cardStateFor,
  passOf,
  sawKindInPass,
  BEAT_ORIGIN,
  BEAT_CONTAINER,
  BEAT_BIDS,
  BEAT_RECAP,
  BEAT_PASS,
  BEAT_BASELINE,
  PASS_BASELINE,
  PASS_ARTF,
} from "./theaterBeats.js";

/**
 * The structural beats of pass 1 (banner, origin, bids, baseline) and the
 * structural beats of pass 2 (banner, origin, bids, recap). Container beats sit
 * between pass 2's origin and bids.
 */
const PASS1_STRUCTURE = [BEAT_PASS, BEAT_ORIGIN, BEAT_BIDS, BEAT_BASELINE];
const PASS2_HEAD = [BEAT_PASS, BEAT_ORIGIN];
const PASS2_TAIL = [BEAT_BIDS, BEAT_RECAP];
const STRUCTURAL_COUNT = PASS1_STRUCTURE.length + PASS2_HEAD.length + PASS2_TAIL.length;

/** The container beats, i.e. pass 2 with its structural beats removed. */
function containerSlice(beats) {
  return beats.slice(PASS1_STRUCTURE.length + PASS2_HEAD.length, beats.length - PASS2_TAIL.length);
}

/* ------------------------------------------------------------------ helpers */

function stop(id, mutations, extra = {}) {
  return { id, displayName: id, status: "ok", latency: { ms: 5 }, mutations, ...extra };
}

function result(containerStops) {
  return { stops: [stop("ssp", []), ...containerStops, stop("dsp", [])] };
}

const metricsMutation = {
  intent: "ADD_METRICS",
  op: "ADD",
  path: "/imp/imp-1/metric",
  payload: {
    metric: [
      { type: "viewability", value: 0.82 },
      { type: "brand_safety", value: 0.91 },
    ],
  },
};

const segmentsMutation = {
  intent: "ACTIVATE_SEGMENTS",
  op: "ADD",
  path: "/user/data/segment",
  payload: { id: ["int-sports", "demo-35-44"] },
};

const floorMutation = {
  intent: "ADJUST_DEAL_FLOOR",
  op: "REPLACE",
  path: "/imp/imp-1/deals/deal-a",
  payload: { bidfloor: 1.9 },
};

const marginPercentMutation = {
  intent: "ADJUST_DEAL_MARGIN",
  op: "REPLACE",
  path: "/imp/imp-1/deals/deal-a",
  payload: { margin: { value: 0.14, calculation_type: 1 } },
};

const marginCpmMutation = {
  intent: "ADJUST_DEAL_MARGIN",
  op: "REPLACE",
  path: "/imp/imp-1/deals/deal-b",
  payload: { margin: { value: 0.5, calculation_type: 0 } },
};

const shadeMutation = {
  intent: "BID_SHADE",
  op: "REPLACE",
  path: "/seatbid/0/bid/0/price",
  payload: { price: 3.2 },
};

const payloadWithDeal = {
  id: "req-1",
  bid_request: {
    imp: [{
      id: "imp-1",
      bidfloor: 0.9,
      banner: { w: 300, h: 250 },
      pmp: { deals: [{ id: "deal-a", bidfloor: 1.5, at: 2 }, { id: "deal-b", bidfloor: 3, at: 1 }] },
    }],
    site: { domain: "example.com", page: "/a", cat: ["192"], cattax: 9 },
    user: { yob: 1990, gender: "F" },
  },
  bid_response: { seatbid: [{ bid: [{ price: 7.5 }] }] },
};

/* ------------------------------------------------------ beat count, BR-1..4 */

describe("buildBeats sequence shape", () => {
  it("is pass 1 (banner, origin, bids, baseline) then pass 2 (banner, origin, one beat per mutation, bids, recap)", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("metrics", [metricsMutation]),
      stop("widedeep", [segmentsMutation]),
      stop("yield-floor", [floorMutation]),
    ]));
    expect(beats).toHaveLength(STRUCTURAL_COUNT + 3);
    expect(beats.slice(0, 4).map((b) => b.kind)).toEqual(PASS1_STRUCTURE);
    expect(beats.slice(4, 6).map((b) => b.kind)).toEqual(PASS2_HEAD);
    expect(beats.slice(-2).map((b) => b.kind)).toEqual(PASS2_TAIL);
    expect(containerSlice(beats).every((b) => b.kind === BEAT_CONTAINER)).toBe(true);
  });

  it("stamps every beat with its pass, and pass 1 precedes pass 2 entirely", () => {
    const beats = buildBeats(payloadWithDeal, result([stop("metrics", [metricsMutation])]));
    const passes = beats.map((b) => b.pass);
    expect(passes.slice(0, 4)).toEqual([PASS_BASELINE, PASS_BASELINE, PASS_BASELINE, PASS_BASELINE]);
    expect(passes.slice(4).every((p) => p === PASS_ARTF)).toBe(true);
  });

  it("marks the two banners: pass 1 without ARTF, pass 2 with", () => {
    const beats = buildBeats(payloadWithDeal, result([]));
    const banners = beats.filter((b) => b.kind === BEAT_PASS);
    expect(banners).toHaveLength(2);
    expect(banners[0]).toMatchObject({ pass: PASS_BASELINE, artf: false, movement: "none" });
    expect(banners[1]).toMatchObject({ pass: PASS_ARTF, artf: true, movement: "none" });
  });
  it("puts the bids beat AFTER every container and BEFORE the recap", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("metrics", [metricsMutation]),
      stop("yield-floor", [floorMutation]),
    ]));
    // Pass 2's bids beat: the LAST bids beat, since pass 1 has one too.
    const bidsAt = beats.map((b) => b.kind).lastIndexOf(BEAT_BIDS);
    const recapAt = beats.findIndex((b) => b.kind === BEAT_RECAP);
    const lastContainerAt = beats.reduce(
      (acc, b, i) => (b.kind === BEAT_CONTAINER ? i : acc), -1,
    );
    expect(lastContainerAt).toBeGreaterThan(-1);
    expect(bidsAt).toBeGreaterThan(lastContainerAt);
    expect(recapAt).toBe(bidsAt + 1);
  });
  it("emits one bids beat per pass, and neither carries values", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("metrics", [metricsMutation]),
      stop("widedeep", [segmentsMutation]),
    ]));
    const bids = beats.filter((b) => b.kind === BEAT_BIDS);
    expect(bids).toHaveLength(2);
    expect(bids.map((b) => b.pass)).toEqual([PASS_BASELINE, PASS_ARTF]);
    expect(bids.every((b) => b.values.length === 0)).toBe(true);
  });
  it("moves attention to the buy side on the bids beat, not the recap", () => {
    const beats = buildBeats(payloadWithDeal, result([stop("metrics", [metricsMutation])]));
    const bids = beats.filter((b) => b.kind === BEAT_BIDS);
    const recap = beats.find((b) => b.kind === BEAT_RECAP);
    expect(bids.every((b) => b.movement === "request-to-buy")).toBe(true);
    expect(recap.movement).toBe("none");
  });

  it("gives a metrics mutation ONE beat carrying TWO values", () => {
    const beats = buildBeats(payloadWithDeal, result([stop("metrics", [metricsMutation])]));
    const containerBeats = beats.filter((b) => b.kind === BEAT_CONTAINER);
    expect(containerBeats).toHaveLength(1);
    expect(containerBeats[0].values).toHaveLength(2);
  });

  it("contributes no beat for a container that returned no mutation", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("yield-floor", []),
      stop("metrics", [metricsMutation]),
    ]));
    expect(beats.filter((b) => b.kind === BEAT_CONTAINER)).toHaveLength(1);
  });

  it("ignores the ssp and dsp bookend stops", () => {
    const beats = buildBeats(payloadWithDeal, result([]));
    expect(beats.map((b) => b.kind)).toEqual([...PASS1_STRUCTURE, ...PASS2_HEAD, ...PASS2_TAIL]);
  });
  it("still emits the bids beat when NO container mutated", () => {
    // Structural, not derived: nothing was enriched, but the seats still bid.
    const beats = buildBeats(payloadWithDeal, result([stop("yield-floor", [])]));
    expect(beats.some((b) => b.kind === BEAT_BIDS)).toBe(true);
  });

  it("preserves stop order", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("widedeep", [segmentsMutation]),
      stop("yield-floor", [floorMutation]),
      stop("metrics", [metricsMutation]),
    ]));
    expect(beats.filter((b) => b.kind === BEAT_CONTAINER).map((b) => b.containerName))
      .toEqual(["widedeep", "yield-floor", "metrics"]);
  });

  it("still emits a beat for an unrecognised payload, with no values", () => {
    const unknown = { intent: "ADD_CIDS", op: "ADD", path: "/x", payload: { something: 1 } };
    const beats = buildBeats(payloadWithDeal, result([stop("metrics", [unknown])]));
    const b = beats.find((x) => x.kind === BEAT_CONTAINER);
    expect(b).toBeDefined();
    expect(b.intent).toBe("ADD_CIDS");
    expect(b.values).toEqual([]);
  });
});

/* --------------------------------------------------------- values, BR-6..10 */

describe("mutationToValues", () => {
  it("reads the real margin calculation type rather than assuming percent", () => {
    expect(mutationToValues(marginPercentMutation, payloadWithDeal)[0].calculationType).toBe("PERCENT");
    expect(mutationToValues(marginCpmMutation, payloadWithDeal)[0].calculationType).toBe("CPM");
  });

  it("resolves a floor's before from the deal in the submitted request", () => {
    const [v] = mutationToValues(floorMutation, payloadWithDeal);
    expect(v.dealId).toBe("deal-a");
    expect(v.before).toBe(1.5);
    expect(v.after).toBe(1.9);
  });

  it("leaves before null when the deal is absent from the request", () => {
    const orphan = { ...floorMutation, path: "/imp/imp-1/deals/not-there" };
    const [v] = mutationToValues(orphan, payloadWithDeal);
    expect(v.before).toBeNull();
  });

  it("resolves a shaded price's before from the submitted bid response", () => {
    const [v] = mutationToValues(shadeMutation, payloadWithDeal);
    expect(v.before).toBe(7.5);
    expect(v.after).toBe(3.2);
  });

  it("leaves a shaded price's before null when the request carried no bid response", () => {
    const [v] = mutationToValues(shadeMutation, { ...payloadWithDeal, bid_response: undefined });
    expect(v.before).toBeNull();
  });

  it("distinguishes activated from suppressed deals by intent", () => {
    const act = { intent: "ACTIVATE_DEALS", op: "ADD", path: "/imp/imp-1", payload: { id: ["d1"] } };
    const sup = { intent: "SUPPRESS_DEALS", op: "REMOVE", path: "/imp/imp-1", payload: { id: ["d2"] } };
    expect(mutationToValues(act, payloadWithDeal)[0].role).toBe("deals-activated");
    expect(mutationToValues(sup, payloadWithDeal)[0].role).toBe("deals-suppressed");
  });

  it("drops metric entries that are not real numbers rather than coercing them", () => {
    const bad = {
      intent: "ADD_METRICS", op: "ADD", path: "/imp/imp-1/metric",
      payload: { metric: [{ type: "viewability", value: null }, { type: "brand_safety", value: 0.5 }] },
    };
    const values = mutationToValues(bad, payloadWithDeal);
    expect(values).toHaveLength(1);
    expect(values[0].type).toBe("brand_safety");
  });
});

/* ---------------------------------------------------------- explore, BR-12 */

describe("isExplored", () => {
  it("is true only for a real explore suffix", () => {
    expect(isExplored("xgb-v1:explore")).toBe(true);
    expect(isExplored("xgb-v1")).toBe(false);
    expect(isExplored("")).toBe(false);
    expect(isExplored(null)).toBe(false);
    expect(isExplored("explore:xgb-v1")).toBe(false);
  });
});

describe("dealIdFromPath", () => {
  it("extracts the deal id from an ARTF deal path", () => {
    expect(dealIdFromPath("/imp/imp-1/deals/deal-a")).toBe("deal-a");
  });
  it("returns null for any other shape", () => {
    expect(dealIdFromPath("/imp/imp-1/metric")).toBeNull();
    expect(dealIdFromPath(undefined)).toBeNull();
  });
});

/* --------------------------------------------------------- context, FR-13/14 */

describe("buildScenarioContext", () => {
  it("reads real request fields", () => {
    const c = buildScenarioContext(payloadWithDeal);
    expect(c.publisher).toBe("example.com");
    expect(c.page).toBe("/a");
    expect(c.contentCategories).toEqual(["192"]);
    expect(c.categoryTaxonomy).toBe(9);
    expect(c.bidFloor).toBe(0.9);
    expect(c.impressionFormat).toBe("banner");
    expect(c.deals.map((d) => d.id)).toEqual(["deal-a", "deal-b"]);
  });

  it("puts request-borne user data in userSignals, marked as from the request", () => {
    const c = buildScenarioContext(payloadWithDeal);
    expect(c.userSignals.length).toBeGreaterThan(0);
    expect(c.userSignals.every((s) => s.provenance === "request")).toBe(true);
  });

  it("reports absent fields as null rather than defaulting them", () => {
    const c = buildScenarioContext({ id: "x", bid_request: {} });
    expect(c.publisher).toBeNull();
    expect(c.bidFloor).toBeNull();
    expect(c.impressionFormat).toBe("unknown");
    expect(c.deals).toEqual([]);
  });

  it("does not throw on a completely empty payload", () => {
    expect(() => buildScenarioContext(undefined)).not.toThrow();
  });
});

/* ------------------------------------------------------ derivation, BR-17/25 */

describe("derived rendering", () => {
  const beats = buildBeats(payloadWithDeal, result([
    stop("metrics", [metricsMutation]),
    stop("widedeep", [segmentsMutation]),
  ]));

  // The first container beat sits after pass 1's four beats and pass 2's banner
  // and origin.
  const firstContainerAt = PASS1_STRUCTURE.length + PASS2_HEAD.length;

  it("visibleValues is empty through the whole of pass 1 and accumulates from the first container beat", () => {
    for (let i = 0; i < firstContainerAt; i++) {
      expect(visibleValues(beats, i)).toHaveLength(0);
    }
    expect(visibleValues(beats, firstContainerAt)).toHaveLength(2);
    expect(visibleValues(beats, firstContainerAt + 1)).toHaveLength(3);
  });

  it("card state is neutral through pass 1, enriching from the first container beat, settled at the recap", () => {
    for (let i = 0; i < firstContainerAt; i++) {
      expect(cardStateFor(beats, i)).toBe("neutral");
    }
    expect(cardStateFor(beats, firstContainerAt)).toBe("enriching");
    expect(cardStateFor(beats, beats.length - 1)).toBe("settled");
  });

  it("passOf names the pass at each index, and null outside the sequence", () => {
    expect(passOf(beats, 0)).toBe(PASS_BASELINE);
    expect(passOf(beats, 3)).toBe(PASS_BASELINE);
    expect(passOf(beats, 4)).toBe(PASS_ARTF);
    expect(passOf(beats, beats.length - 1)).toBe(PASS_ARTF);
    expect(passOf(beats, -1)).toBeNull();
    expect(passOf(beats, beats.length)).toBeNull();
    expect(passOf(null, 0)).toBeNull();
  });

  it("sawKindInPass is scoped to the pass: reaching pass 1's bids says nothing about pass 2's", () => {
    const pass1Bids = 2;
    expect(sawKindInPass(beats, pass1Bids, BEAT_BIDS, PASS_BASELINE)).toBe(true);
    expect(sawKindInPass(beats, pass1Bids, BEAT_BIDS, PASS_ARTF)).toBe(false);
    // Standing on pass 2's origin: pass 2's bids has not been reached.
    expect(sawKindInPass(beats, firstContainerAt - 1, BEAT_BIDS, PASS_ARTF)).toBe(false);
    expect(sawKindInPass(beats, beats.length - 1, BEAT_BIDS, PASS_ARTF)).toBe(true);
  });
});

/* ------------------------------------------------ property tests, NFR-4 */

const arbMutation = fc.oneof(
  fc.record({
    intent: fc.constant("ADD_METRICS"),
    op: fc.constant("ADD"),
    path: fc.constant("/imp/imp-1/metric"),
    payload: fc.record({
      metric: fc.array(
        fc.record({
          type: fc.constantFrom("viewability", "brand_safety"),
          value: fc.double({ min: 0, max: 1, noNaN: true }),
        }),
        { minLength: 0, maxLength: 3 }
      ),
    }),
  }),
  fc.record({
    intent: fc.constantFrom("ACTIVATE_SEGMENTS", "ACTIVATE_DEALS", "SUPPRESS_DEALS"),
    op: fc.constant("ADD"),
    path: fc.constant("/user/data/segment"),
    payload: fc.record({ id: fc.array(fc.string({ minLength: 1 }), { maxLength: 4 }) }),
  }),
  fc.record({
    intent: fc.constant("ADJUST_DEAL_FLOOR"),
    op: fc.constant("REPLACE"),
    path: fc.constant("/imp/imp-1/deals/deal-a"),
    payload: fc.record({ bidfloor: fc.double({ min: 0, max: 50, noNaN: true }) }),
  }),
  fc.record({
    intent: fc.constant("ADJUST_DEAL_MARGIN"),
    op: fc.constant("REPLACE"),
    path: fc.constant("/imp/imp-1/deals/deal-a"),
    payload: fc.record({
      margin: fc.record({
        value: fc.double({ min: 0, max: 1, noNaN: true }),
        calculation_type: fc.constantFrom(0, 1),
      }),
    }),
  }),
  fc.record({
    intent: fc.constant("BID_SHADE"),
    op: fc.constant("REPLACE"),
    path: fc.constant("/seatbid/0/bid/0/price"),
    payload: fc.record({ price: fc.double({ min: 0, max: 100, noNaN: true }) }),
  })
);

const arbStops = fc.array(
  fc.record({
    id: fc.constantFrom("dlrm", "widedeep", "ncf", "metrics", "yield-floor", "yield-margin"),
    mutations: fc.array(arbMutation, { maxLength: 3 }),
  }),
  { maxLength: 6 }
).map((stops) => ({
  stops: [
    { id: "ssp", mutations: [] },
    ...stops.map((s) => ({ ...s, status: "ok", latency: { ms: 1 } })),
    { id: "dsp", mutations: [] },
  ],
}));

describe("buildBeats properties", () => {
  // Eight structural beats — two banners, two origins, two bids, a baseline and a
  // recap — regardless of what the containers did. Only the container beats
  // vary with the response.
  it("beat count is always 8 + the total number of mutations", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const mutationCount = res.stops
        .filter((s) => s.id !== "ssp" && s.id !== "dsp")
        .reduce((n, s) => n + s.mutations.length, 0);
      expect(buildBeats(payloadWithDeal, res)).toHaveLength(mutationCount + STRUCTURAL_COUNT);
    }), { numRuns: 200 });
  });
  it("always emits exactly two banners, two origins, two bids, one baseline and one recap", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const kinds = buildBeats(payloadWithDeal, res).map((b) => b.kind);
      expect(kinds.filter((k) => k === BEAT_PASS)).toHaveLength(2);
      expect(kinds.filter((k) => k === BEAT_ORIGIN)).toHaveLength(2);
      expect(kinds.filter((k) => k === BEAT_BIDS)).toHaveLength(2);
      expect(kinds.filter((k) => k === BEAT_BASELINE)).toHaveLength(1);
      expect(kinds.filter((k) => k === BEAT_RECAP)).toHaveLength(1);
      // Order is invariant: pass 1's four beats, then pass 2 opens with its
      // banner and origin, and closes with bids then recap.
      expect(kinds.slice(0, 4)).toEqual(PASS1_STRUCTURE);
      expect(kinds.slice(4, 6)).toEqual(PASS2_HEAD);
      expect(kinds.lastIndexOf(BEAT_BIDS)).toBe(kinds.length - 2);
      expect(kinds[kinds.length - 1]).toBe(BEAT_RECAP);
    }), { numRuns: 200 });
  });

  it("every pass-1 beat precedes every pass-2 beat, and no container beat is in pass 1", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const beats = buildBeats(payloadWithDeal, res);
      const lastPass1 = beats.map((b) => b.pass).lastIndexOf(PASS_BASELINE);
      const firstPass2 = beats.map((b) => b.pass).indexOf(PASS_ARTF);
      expect(lastPass1).toBeLessThan(firstPass2);
      expect(beats.every((b) => b.pass === PASS_BASELINE || b.pass === PASS_ARTF)).toBe(true);
      expect(beats.filter((b) => b.kind === BEAT_CONTAINER).every((b) => b.pass === PASS_ARTF)).toBe(true);
      // Nothing in pass 1 carries a value, so the request card is untouched
      // through the whole baseline pass.
      expect(visibleValues(beats, lastPass1)).toEqual([]);
    }), { numRuns: 200 });
  });

  it("indexes are always sequential from zero", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const beats = buildBeats(payloadWithDeal, res);
      expect(beats.map((b) => b.index)).toEqual(beats.map((_, i) => i));
    }), { numRuns: 200 });
  });

  it("is deterministic: the same input always yields the same beats", () => {
    fc.assert(fc.property(arbStops, (res) => {
      expect(buildBeats(payloadWithDeal, res)).toEqual(buildBeats(payloadWithDeal, res));
    }), { numRuns: 100 });
  });

  it("never invents a value: every value traces to a mutation payload", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const beats = buildBeats(payloadWithDeal, res);
      for (const beat of beats) {
        if (beat.kind !== BEAT_CONTAINER) {
          expect(beat.values).toEqual([]);
        }
        for (const v of beat.values) {
          expect(["metric", "ids", "floor", "margin", "price"]).toContain(v.kind);
        }
      }
    }), { numRuns: 200 });
  });

  it("visibleValues at the last index equals every value in order", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const beats = buildBeats(payloadWithDeal, res);
      const all = beats.flatMap((b) => b.values);
      expect(visibleValues(beats, beats.length - 1)).toEqual(all);
    }), { numRuns: 100 });
  });

  it("never throws, whatever the response shape", () => {
    fc.assert(fc.property(fc.anything(), (junk) => {
      expect(() => buildBeats(payloadWithDeal, junk)).not.toThrow();
    }), { numRuns: 200 });
  });
});

/* ------------------------------------------- store-defined (dynamic) stops */
//
// A store-defined container arrives from normalizer.js as a stop whose id is
// `dynamic:<name>`, because no build-time table can know its name. buildBeats
// used to label from DISPLAY_NAME_BY_STOP_ID[stop.id], which has no entry for a
// synthesized id, so the beat caption would have read "dynamic:artf-template".

describe("buildBeats with a store-defined container", () => {
  const cidsMutation = {
    intent: "ADD_CIDS",
    op: "ADD",
    path: "/imp/imp-1/ext/cids",
    payload: { id: ["cid-1"] },
  };

  it("labels the beat with the stop's resolved display name, not its synthesized id", () => {
    const beats = buildBeats(payloadWithDeal, result([
      { id: "dynamic:artf-template", displayName: "ARTF Template", status: "ok",
        latency: { ms: 4 }, mutations: [cidsMutation] },
    ]));
    const containerBeats = beats.filter((b) => b.kind === "container");
    expect(containerBeats).toHaveLength(1);
    expect(containerBeats[0].displayLabel).toBe("ARTF Template");
    expect(containerBeats[0].displayLabel).not.toContain("dynamic:");
  });

  it("produces a beat for a store-defined container's mutation like any other", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("metrics", [metricsMutation]),
      { id: "dynamic:artf-template", displayName: "ARTF Template", status: "ok",
        latency: { ms: 4 }, mutations: [cidsMutation] },
    ]));
    const labels = beats.filter((b) => b.kind === "container").map((b) => b.displayLabel);
    expect(labels).toContain("ARTF Template");
    expect(labels).toHaveLength(2);
  });

  it("falls back to the container name when the stop carries no display name", () => {
    const beats = buildBeats(payloadWithDeal, result([
      { id: "dynamic:my-container", status: "ok", latency: { ms: 4 }, mutations: [cidsMutation] },
    ]));
    const beat = beats.find((b) => b.kind === "container");
    // The id is the only thing left to name it by — still not an invented label.
    expect(beat.displayLabel).toBe("dynamic:my-container");
  });

  it("an inactive store container contributes no beat, having produced no mutation", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("metrics", [metricsMutation]),
      { id: "dynamic:artf-template", displayName: "ARTF Template", status: "disabled",
        latency: { ms: 0 }, mutations: [] },
    ]));
    const labels = beats.filter((b) => b.kind === "container").map((b) => b.displayLabel);
    expect(labels).not.toContain("ARTF Template");
    // Beats are derived from mutations, so a container that was never called is
    // absent from the walkthrough without the theater needing to know about
    // activation at all.
    expect(labels).toHaveLength(1);
  });
});
