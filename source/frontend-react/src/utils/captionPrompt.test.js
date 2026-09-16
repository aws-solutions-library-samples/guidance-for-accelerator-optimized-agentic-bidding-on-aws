import { describe, it, expect } from "vitest";
import { buildCaptionPrompt, SYSTEM_PROMPT, MAX_TOKENS, TEMPERATURE } from "./captionPrompt.js";
import { resolveSegmentLabel } from "./segmentLabels.js";
import { BEAT_ORIGIN, BEAT_CONTAINER, BEAT_RECAP, buildBeats } from "./theaterBeats.js";
import {
  SEGMENTS_BEAT, SEGMENTS_CONTEXT,
  FLOOR_BEAT, FLOOR_CONTEXT,
  METRICS_BEAT, METRICS_CONTEXT,
} from "./captionFixtures.js";

const deps = { resolveSegmentLabel };

describe("the system prompt carries the constraints measurement proved necessary", () => {
  it("states a word budget, not a character budget", () => {
    // NFR3-5: instructed in characters the model complied 1 of 6; in words, 24 of 24.
    expect(SYSTEM_PROMPT).toMatch(/45 words/);
    expect(SYSTEM_PROMPT).not.toMatch(/\d+ characters/);
  });

  it("states the attribution rule and names the payload field", () => {
    // NFR3-8: without this the model credited the publisher for an exchange signal.
    expect(SYSTEM_PROMPT).toMatch(/signals_from_exchange/);
    expect(SYSTEM_PROMPT).toMatch(/Never attribute it to the publisher/);
  });

  it("forbids merging two segments into one phrase", () => {
    // BR3-20a / NFR3-27: 3 of 8 samples merged segments 98 and 7.
    expect(SYSTEM_PROMPT).toMatch(/Never merge two segments/);
  });

  it("forbids the register the validator also rejects", () => {
    for (const phrase of ["No emojis", "No em-dashes", "No exclamation marks", "No questions",
                          "No first person or second person"]) {
      expect(SYSTEM_PROMPT).toContain(phrase);
    }
  });

  it("does not ask the model to mention how the text was produced", () => {
    expect(SYSTEM_PROMPT).toMatch(/Never mention how this text was produced/);
  });
});

describe("inference parameters", () => {
  it("uses the measured values", () => {
    const prompt = buildCaptionPrompt(SEGMENTS_BEAT, SEGMENTS_CONTEXT, deps);
    expect(prompt.maxTokens).toBe(MAX_TOKENS);
    expect(prompt.temperature).toBe(TEMPERATURE);
    expect(MAX_TOKENS).toBe(120);
    expect(TEMPERATURE).toBe(0.3);
  });
});

describe("segment identifiers are accompanied by taxonomy names", () => {
  it("resolves the FR-32 parenting segments to real names", () => {
    const { message } = buildCaptionPrompt(SEGMENTS_BEAT, SEGMENTS_CONTEXT, deps);
    expect(message).toContain("Parenting");
    expect(message).toContain("Parents with Children");
    expect(message).toContain("35-39");
    // The identifiers are still present alongside the names.
    for (const id of ["350", "354", "98", "7"]) expect(message).toContain(`"${id}"`);
  });

  it("emits the identifier alone when no name is known", () => {
    const beat = { ...SEGMENTS_BEAT, values: [{ kind: "ids", role: "segments", ids: ["zzz-not-real"] }] };
    const { message } = buildCaptionPrompt(beat, SEGMENTS_CONTEXT, deps);
    expect(message).toContain("zzz-not-real");
    const parsed = JSON.parse(message);
    expect(parsed.step.segments[0].name).toBeUndefined();
  });
});

describe("the payload names exchange-sent data as such", () => {
  it("uses signals_from_exchange", () => {
    const { message } = buildCaptionPrompt(SEGMENTS_BEAT, SEGMENTS_CONTEXT, deps);
    const parsed = JSON.parse(message);
    expect(parsed.impression.signals_from_exchange).toEqual([
      { label: "year of birth", value: "1989" },
      { label: "data segment", value: "Parents with Children" },
    ]);
  });
});

describe("unknown baselines are passed as unknown, never as zero", () => {
  it("says a missing prior floor was not recorded", () => {
    const beat = { ...FLOOR_BEAT, values: [{ kind: "floor", dealId: "d1", before: null, after: 1.68 }] };
    const parsed = JSON.parse(buildCaptionPrompt(beat, FLOOR_CONTEXT, deps).message);
    expect(parsed.step.floor_before_usd).toBe("not recorded");
    expect(parsed.step.floor_before_usd).not.toBe(0);
  });

  it("says a missing prior price was not recorded", () => {
    const beat = { kind: BEAT_CONTAINER, displayLabel: "Bid Pricer", intent: "BID_SHADE", latencyMs: 12,
      values: [{ kind: "price", before: null, after: 2.1 }] };
    const parsed = JSON.parse(buildCaptionPrompt(beat, {}, deps).message);
    expect(parsed.step.price_before_usd).toBe("not recorded");
  });
});

describe("exploration is reported only when the beat says so", () => {
  it("includes the flag for an explored beat", () => {
    const parsed = JSON.parse(buildCaptionPrompt(FLOOR_BEAT, FLOOR_CONTEXT, deps).message);
    expect(parsed.step.from_exploration_arm).toBe(true);
  });

  it("omits the flag entirely when not explored", () => {
    const parsed = JSON.parse(buildCaptionPrompt(METRICS_BEAT, METRICS_CONTEXT, deps).message);
    expect("from_exploration_arm" in parsed.step).toBe(false);
  });
});

describe("beat kinds", () => {
  it("describes the origin beat without claiming anything changed", () => {
    const parsed = JSON.parse(buildCaptionPrompt({ kind: BEAT_ORIGIN, values: [] }, SEGMENTS_CONTEXT, deps).message);
    expect(parsed.step.nothing_has_been_changed_yet).toBe(true);
    expect(parsed.step.service).toBeUndefined();
  });

  it("gives the recap its contributor list", () => {
    const beat = { kind: BEAT_RECAP, values: [],
      contributors: [{ containerName: "a", displayLabel: "Audience Activator", beatIndexes: [1] }] };
    const parsed = JSON.parse(buildCaptionPrompt(beat, {}, deps).message);
    expect(parsed.step.services_that_changed_the_request).toEqual(["Audience Activator"]);
  });

  it("names an unrendered intent rather than inventing an action", () => {
    const beat = { kind: BEAT_CONTAINER, displayLabel: "Deal Scorer", intent: "SOMETHING_NEW", values: [] };
    const parsed = JSON.parse(buildCaptionPrompt(beat, {}, deps).message);
    expect(parsed.step.action).toContain("SOMETHING_NEW");
  });
});

describe("BR3-7: the prompt carries no other beat's values", () => {
  // The submitted payload and normalised result of a multi-stop run, so the
  // beats carry genuinely different numbers.
  const submitted = {
    id: "req-multi",
    bid_request: {
      site: { domain: "pub.example", cat: ["192"], cattax: 9 },
      imp: [{ id: "1", bidfloor: 1.2, banner: {}, pmp: { deals: [{ id: "d1", bidfloor: 1.2, at: 1 }] } }],
    },
    bid_response: { seatbid: [{ bid: [{ price: 3.75 }] }] },
  };
  const normalised = {
    stops: [
      { id: "signals-enricher", modelVersion: "m1", latency: { ms: 8 },
        mutations: [{ intent: "ADD_METRICS", op: "add", path: "/imp/1/metric",
          payload: { metric: [{ type: "viewability", value: 0.61 }] } }] },
      { id: "yield-optimizer-floor", modelVersion: "m2", latency: { ms: 19 },
        mutations: [{ intent: "ADJUST_DEAL_FLOOR", op: "replace", path: "/imp/1/deals/d1",
          payload: { bidfloor: 2.44 } }] },
      { id: "bid-pricer", modelVersion: "m3", latency: { ms: 31 },
        mutations: [{ intent: "BID_SHADE", op: "replace", path: "/seatbid/0/bid/0/price",
          payload: { price: 2.99 } }] },
    ],
  };

  const beats = buildBeats(submitted, normalised);
  // Values that belong to exactly one beat each.
  const exclusive = { 1: ["0.61", "8"], 2: ["2.44", "19"], 3: ["2.99", "31"] };

  for (const [indexStr, own] of Object.entries(exclusive)) {
    const index = Number(indexStr);
    it(`beat ${index} carries its own values and no other beat's`, () => {
      const { message } = buildCaptionPrompt(beats[index], {}, deps);
      for (const value of own) expect(message).toContain(value);
      for (const [otherStr, others] of Object.entries(exclusive)) {
        if (otherStr === indexStr) continue;
        for (const value of others) {
          expect(message, `beat ${index} must not mention ${value}`).not.toContain(value);
        }
      }
    });
  }

  it("no beat's prompt mentions the recap's contributor list", () => {
    for (const beat of beats.filter((b) => b.kind !== BEAT_RECAP)) {
      const parsed = JSON.parse(buildCaptionPrompt(beat, {}, deps).message);
      expect(parsed.step.services_that_changed_the_request).toBeUndefined();
    }
  });
});

describe("properties", () => {
  const ARBITRARY_BEATS = [
    null, undefined, {}, { kind: "container" }, { kind: "container", values: null },
    { kind: "container", values: [{ kind: "unknown-kind" }] },
    { kind: "container", values: [{ kind: "ids", role: "segments", ids: [] }] },
    { kind: BEAT_RECAP }, { kind: BEAT_ORIGIN },
    { kind: "container", intent: null, latencyMs: undefined, values: [{ kind: "metric" }] },
  ];
  const ARBITRARY_CONTEXTS = [
    null, undefined, {}, { deals: null, userSignals: null },
    { publisher: "", page: null, contentCategories: [], impressionFormat: "unknown" },
  ];

  it("never throws", () => {
    for (const beat of ARBITRARY_BEATS) {
      for (const context of ARBITRARY_CONTEXTS) {
        expect(() => buildCaptionPrompt(beat, context, deps)).not.toThrow();
        expect(() => buildCaptionPrompt(beat, context)).not.toThrow();
      }
    }
  });

  it("always produces parseable JSON with a step", () => {
    for (const beat of ARBITRARY_BEATS) {
      for (const context of ARBITRARY_CONTEXTS) {
        const { message } = buildCaptionPrompt(beat, context, deps);
        const parsed = JSON.parse(message);
        expect(parsed).toHaveProperty("step");
        expect(parsed).toHaveProperty("impression");
      }
    }
  });

  it("always returns the same static system prompt", () => {
    for (const beat of ARBITRARY_BEATS) {
      expect(buildCaptionPrompt(beat, {}, deps).system).toBe(SYSTEM_PROMPT);
    }
  });

  it("omits null and empty fields rather than emitting them", () => {
    const { message } = buildCaptionPrompt({ kind: "container", displayLabel: "X", values: [] }, {}, deps);
    expect(message).not.toContain("null");
    expect(message).not.toContain("undefined");
  });
});
