import { describe, it, expect } from "vitest";
import { factualCaption } from "./theaterCaptions.js";
import { validateCaption, CHAR_CEILING } from "./captionValidation.js";
import { BEAT_ORIGIN, BEAT_CONTAINER, BEAT_RECAP, buildBeats } from "./theaterBeats.js";
import {
  SEGMENTS_BEAT, SEGMENTS_CONTEXT,
  FLOOR_BEAT, FLOOR_CONTEXT,
  METRICS_BEAT, METRICS_CONTEXT,
} from "./captionFixtures.js";

// BR3-31. The fallback is the floor of the feature, so it must clear the same bar
// the generated caption has to clear. If it could not, the failure path would be
// displaying prose the validator would have rejected.

const BEATS = [
  { kind: BEAT_ORIGIN, values: [], latencyMs: null },
  SEGMENTS_BEAT,
  FLOOR_BEAT,
  METRICS_BEAT,
  { kind: BEAT_CONTAINER, displayLabel: "Bid Pricer", intent: "BID_SHADE", latencyMs: 12,
    values: [{ kind: "price", before: 2.5, after: 2.1 }] },
  { kind: BEAT_CONTAINER, displayLabel: "Deal Scorer", intent: "ACTIVATE_DEALS", latencyMs: 15,
    values: [{ kind: "ids", role: "deals-activated", ids: ["d1", "d2"] }] },
  { kind: BEAT_CONTAINER, displayLabel: "Deal Scorer", intent: "SUPPRESS_DEALS", latencyMs: 15,
    values: [{ kind: "ids", role: "deals-suppressed", ids: ["d3"] }] },
  { kind: BEAT_CONTAINER, displayLabel: "Yield Optimizer", intent: "ADJUST_DEAL_MARGIN", latencyMs: 5,
    values: [{ kind: "margin", dealId: "d1", value: 0.14, calculationType: "PERCENT" }] },
  { kind: BEAT_CONTAINER, displayLabel: "Yield Optimizer", intent: "ADJUST_DEAL_MARGIN", latencyMs: 5,
    values: [{ kind: "margin", dealId: "d1", value: 0.5, calculationType: "CPM" }] },
  { kind: BEAT_CONTAINER, displayLabel: "Deal Scorer", intent: "SOMETHING_NEW", latencyMs: 9, values: [] },
  { kind: BEAT_CONTAINER, displayLabel: "Deal Scorer", intent: null, latencyMs: null, values: [] },
  { kind: BEAT_RECAP, contributors: [], values: [] },
  { kind: BEAT_RECAP, values: [],
    contributors: [{ containerName: "a", displayLabel: "Audience Activator", beatIndexes: [1] }] },
  { kind: BEAT_RECAP, values: [], contributors: [
    { containerName: "a", displayLabel: "A", beatIndexes: [1] },
    { containerName: "b", displayLabel: "B", beatIndexes: [2] },
  ] },
];

const CONTEXTS = [SEGMENTS_CONTEXT, FLOOR_CONTEXT, METRICS_CONTEXT, {}, null,
  { publisher: "pub.example", impressionFormat: "video", bidFloor: 0.85, deals: [], userSignals: [] }];

describe("BR3-31: the factual caption always passes validation", () => {
  for (const [i, beat] of BEATS.entries()) {
    it(`beat ${i} (${beat.kind}/${beat.intent ?? "-"}) passes for every context`, () => {
      for (const context of CONTEXTS) {
        const text = factualCaption(beat, context);
        if (text === "") continue;
        const result = validateCaption(text, beat, context);
        const detail = result.ok ? "" : JSON.stringify(result.violations);
        expect(result.ok, `"${text}" -> ${detail}`).toBe(true);
      }
    });
  }

  it("fits the caption strip for every beat and context", () => {
    for (const beat of BEATS) {
      for (const context of CONTEXTS) {
        expect(factualCaption(beat, context).length).toBeLessThanOrEqual(CHAR_CEILING);
      }
    }
  });
});

describe("BR3-31 over derived beats", () => {
  it("passes for every beat of a real multi-stop response", () => {
    const submitted = {
      id: "req-1",
      bid_request: {
        site: { domain: "pub.example", cat: ["192"], cattax: 9, page: "/a/b" },
        imp: [{ id: "1", bidfloor: 1.2, banner: {},
          pmp: { deals: [{ id: "d1", bidfloor: 1.2, at: 1 }] } }],
        user: { yob: 1989, data: [{ segment: [{ name: "Parents with Children" }] }] },
      },
      bid_response: { seatbid: [{ bid: [{ price: 3.75 }] }] },
    };
    const normalised = {
      stops: [
        { id: "signals-enricher", modelVersion: "m1", latency: { ms: 8 },
          mutations: [{ intent: "ADD_METRICS", op: "add", path: "/imp/1/metric",
            payload: { metric: [{ type: "viewability", value: 0.61 }] } }] },
        { id: "audience-activator", modelVersion: "m2", latency: { ms: 41 },
          mutations: [{ intent: "ACTIVATE_SEGMENTS", op: "add", path: "/user/data",
            payload: { id: ["350", "354", "98", "7"] } }] },
        { id: "yield-optimizer-floor", modelVersion: "m3:explore", latency: { ms: 19 },
          mutations: [{ intent: "ADJUST_DEAL_FLOOR", op: "replace", path: "/imp/1/deals/d1",
            payload: { bidfloor: 2.44 } }] },
        { id: "bid-pricer", modelVersion: "m4", latency: { ms: 31 },
          mutations: [{ intent: "BID_SHADE", op: "replace", path: "/seatbid/0/bid/0/price",
            payload: { price: 2.99 } }] },
      ],
    };
    const beats = buildBeats(submitted, normalised);
    expect(beats.length).toBe(7); // origin + 4 mutations + bids + recap

    for (const beat of beats) {
      const text = factualCaption(beat, {});
      if (text === "") continue;
      const result = validateCaption(text, beat, {});
      const detail = result.ok ? "" : JSON.stringify(result.violations);
      expect(result.ok, `"${text}" -> ${detail}`).toBe(true);
    }
  });
});

describe("the fallback makes no claim about how it was produced", () => {
  it("never mentions generation, a model, or an assistant", () => {
    for (const beat of BEATS) {
      for (const context of CONTEXTS) {
        const text = factualCaption(beat, context).toLowerCase();
        for (const word of ["generated", "model said", "ai ", "assistant", "unavailable",
                            "could not", "failed", "fallback"]) {
          expect(text).not.toContain(word);
        }
      }
    }
  });
});
