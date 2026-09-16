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
  BEAT_ORIGIN,
  BEAT_CONTAINER,
  BEAT_RECAP,
} from "./theaterBeats.js";

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
  it("is origin + one beat per mutation + recap", () => {
    const beats = buildBeats(payloadWithDeal, result([
      stop("metrics", [metricsMutation]),
      stop("widedeep", [segmentsMutation]),
      stop("yield-floor", [floorMutation]),
    ]));
    expect(beats).toHaveLength(1 + 3 + 1);
    expect(beats[0].kind).toBe(BEAT_ORIGIN);
    expect(beats[beats.length - 1].kind).toBe(BEAT_RECAP);
    expect(beats.slice(1, -1).every((b) => b.kind === BEAT_CONTAINER)).toBe(true);
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
    expect(beats.map((b) => b.kind)).toEqual([BEAT_ORIGIN, BEAT_RECAP]);
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

  it("visibleValues accumulates only up to the index", () => {
    expect(visibleValues(beats, 0)).toHaveLength(0);
    expect(visibleValues(beats, 1)).toHaveLength(2);
    expect(visibleValues(beats, 2)).toHaveLength(3);
  });

  it("card state advances neutral to enriching to settled", () => {
    expect(cardStateFor(beats, 0)).toBe("neutral");
    expect(cardStateFor(beats, 1)).toBe("enriching");
    expect(cardStateFor(beats, beats.length - 1)).toBe("settled");
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
  it("beat count is always 2 + the total number of mutations", () => {
    fc.assert(fc.property(arbStops, (res) => {
      const mutationCount = res.stops
        .filter((s) => s.id !== "ssp" && s.id !== "dsp")
        .reduce((n, s) => n + s.mutations.length, 0);
      expect(buildBeats(payloadWithDeal, res)).toHaveLength(mutationCount + 2);
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
