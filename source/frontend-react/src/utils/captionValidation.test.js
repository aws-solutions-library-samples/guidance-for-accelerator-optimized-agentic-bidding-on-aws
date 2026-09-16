import { describe, it, expect } from "vitest";
import {
  validateCaption,
  extractNumericClaims,
  allowedNumericClaims,
  sourcedLiterals,
  CHAR_CEILING,
  WORD_CEILING,
  countSentences,
} from "./captionValidation.js";
import {
  ACCEPTED,
  REJECTED,
  SYNTHETIC_REJECTIONS,
  SEGMENTS_BEAT,
  SEGMENTS_CONTEXT,
  FLOOR_BEAT,
  FLOOR_CONTEXT,
  METRICS_BEAT,
  METRICS_CONTEXT,
} from "./captionFixtures.js";
import { factualCaption } from "./theaterCaptions.js";
import { BEAT_ORIGIN, BEAT_CONTAINER, BEAT_RECAP } from "./theaterBeats.js";

function rules(result) {
  return result.ok ? [] : result.violations.map((v) => v.rule);
}

describe("measured captions the model actually produced", () => {
  for (const fixture of ACCEPTED) {
    it(`accepts: ${fixture.text.slice(0, 58)}...`, () => {
      const result = validateCaption(fixture.text, fixture.beat, fixture.context);
      expect(result.ok, JSON.stringify(rules(result))).toBe(true);
      expect(result.text).toBe(fixture.text);
    });
  }

  for (const fixture of REJECTED) {
    it(`rejects (${fixture.note}): ${fixture.text.slice(0, 44)}...`, () => {
      const result = validateCaption(fixture.text, fixture.beat, fixture.context);
      expect(result.ok).toBe(false);
      expect(rules(result)).toContain(fixture.expectRule);
    });
  }
});

describe("rule-specific rejections", () => {
  for (const fixture of SYNTHETIC_REJECTIONS) {
    it(`rejects for ${fixture.expectRule} (${fixture.note})`, () => {
      const result = validateCaption(fixture.text, fixture.beat, fixture.context);
      expect(result.ok).toBe(false);
      expect(rules(result)).toContain(fixture.expectRule);
    });
  }
});

describe("presentation tolerance the measurement showed is required", () => {
  it("accepts currency written as the model writes it, not as a test author imagines", () => {
    // The model writes "1.2 USD". A validator comparing raw strings against 1.2
    // would reject its own correct output.
    const result = validateCaption(
      "The Yield Optimizer raised the price floor for deal-premium-002 from 1.2 USD to 1.68 USD.",
      FLOOR_BEAT, FLOOR_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });

  it("accepts latency written as 41 milliseconds, not 41ms", () => {
    const result = validateCaption(
      "Audience Activator added four segments in 41 milliseconds.",
      SEGMENTS_BEAT, SEGMENTS_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });

  it("accepts a ratio left as a decimal", () => {
    const result = validateCaption(
      "Signals Enricher added viewability 0.72 and brand safety 0.95 to the request.",
      METRICS_BEAT, METRICS_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });

  it("accepts a ratio rendered as a percentage", () => {
    const result = validateCaption(
      "Signals Enricher recorded viewability of 72% for the impression.",
      METRICS_BEAT, METRICS_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });

  it("accepts a count written as a word", () => {
    const result = validateCaption(
      "Audience Activator added four segments to the request.",
      SEGMENTS_BEAT, SEGMENTS_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });
});

describe("units gate the comparison", () => {
  it("rejects a latency claim satisfied only by a currency value", () => {
    // A beat with a $0.47 floor and a 23ms latency must not accept "47
    // milliseconds" merely because 0.47 x 100 is 47.
    const beat = { ...FLOOR_BEAT, latencyMs: 23,
      values: [{ kind: "floor", dealId: "d1", before: null, after: 0.47 }] };
    const ctx = { ...FLOOR_CONTEXT, bidFloor: 0.47, deals: [{ id: "d1", bidFloor: 0.47, auctionType: 1 }] };
    const result = validateCaption("Yield Optimizer responded in 47 milliseconds.", beat, ctx);
    expect(result.ok).toBe(false);
    expect(rules(result)).toContain("BR3-13");
  });

  it("accepts cents phrasing for a real currency value", () => {
    const beat = { ...FLOOR_BEAT, latencyMs: null,
      values: [{ kind: "floor", dealId: "d1", before: null, after: 0.5 }] };
    const ctx = { ...FLOOR_CONTEXT, bidFloor: 0.5, deals: [] };
    const result = validateCaption("Yield Optimizer set the floor to 50 cents.", beat, ctx);
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });
});

describe("identifiers that contain digits are not read as invented numbers", () => {
  it("does not flag a real deal id", () => {
    expect(sourcedLiterals(FLOOR_BEAT, FLOOR_CONTEXT)).toContain("deal-premium-002");
    const result = validateCaption(
      "The Yield Optimizer raised the floor for deal-premium-002 to 1.68 USD.",
      FLOOR_BEAT, FLOOR_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });

  it("does not flag a real taxonomy name containing a range", () => {
    const result = validateCaption(
      "Audience Activator added a Demographic | Age Range | 35-39 segment.",
      SEGMENTS_BEAT, SEGMENTS_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });

  it("does flag a taxonomy-looking id the beat does not carry", () => {
    const result = validateCaption(
      "Audience Activator added segment 411 to the request.",
      SEGMENTS_BEAT, SEGMENTS_CONTEXT,
    );
    expect(result.ok).toBe(false);
    expect(rules(result)).toContain("BR3-13");
  });
});

describe("length rules", () => {
  it("rejects over the character ceiling", () => {
    const long = `Audience Activator added four segments in 41 milliseconds. ${"word ".repeat(60)}`;
    const result = validateCaption(long, SEGMENTS_BEAT, SEGMENTS_CONTEXT);
    expect(result.ok).toBe(false);
    expect(rules(result)).toContain("BR3-20");
  });

  it("counts sentences without treating a decimal or a domain as a boundary", () => {
    // The sentence count is advisory, enforced in the prompt rather than here
    // (the model honoured it 6 of 6), but the counter must still be correct.
    expect(countSentences("One thing. Two things.")).toBe(2);
    expect(countSentences("The floor moved from 1.2 USD to 1.68 USD.")).toBe(1);
    expect(countSentences("Served on parenting-weekly.example today.")).toBe(1);
  });

  it("does not read a decimal or a domain as a sentence boundary", () => {
    // "1.2 USD" and "parenting-weekly.example" each contain a period.
    const result = validateCaption(
      "The Yield Optimizer raised the floor from 1.2 USD to 1.68 USD on parenting-weekly.example.",
      FLOOR_BEAT, FLOOR_CONTEXT,
    );
    expect(result.ok, JSON.stringify(rules(result))).toBe(true);
  });
});

describe("every violation is reported, not just the first", () => {
  it("names both the invented number and the register breach", () => {
    const result = validateCaption(
      "Signals Enricher seamlessly added metric 0.44 to the request.",
      METRICS_BEAT, METRICS_CONTEXT,
    );
    expect(result.ok).toBe(false);
    expect(rules(result)).toContain("BR3-13");
    expect(rules(result)).toContain("BR3-19");
  });
});

describe("the caption is never repaired", () => {
  it("returns the input unchanged on success", () => {
    const text = ACCEPTED[0].text;
    const result = validateCaption(text, ACCEPTED[0].beat, ACCEPTED[0].context);
    expect(result.ok).toBe(true);
    expect(result.text).toBe(text);
  });

  it("returns no text at all on failure", () => {
    const result = validateCaption("Signals Enricher added metric 0.44.", METRICS_BEAT, METRICS_CONTEXT);
    expect(result.ok).toBe(false);
    expect(result.text).toBeUndefined();
  });
});

/* ------------------------------------------------------------- properties */

const ARBITRARY = [
  "", "   ", "\n\n", "0", "-1", "1e10", "NaN", "undefined", "null",
  "\u2014", "\u{1F600}", "?", "!", "a".repeat(1000),
  "1.2 USD 1.68 USD 41 milliseconds four", "%%%", "$", "$.", "0.0.0.0",
  "<script>alert(1)</script>", "SELECT * FROM x", "{}", "[]",
];

describe("properties", () => {
  it("never throws on arbitrary input", () => {
    for (const text of ARBITRARY) {
      for (const [beat, ctx] of [[SEGMENTS_BEAT, SEGMENTS_CONTEXT], [{}, {}], [null, null]]) {
        expect(() => validateCaption(text, beat, ctx)).not.toThrow();
      }
    }
  });

  it("never throws for non-string input", () => {
    for (const value of [null, undefined, 0, 1, {}, [], true, Symbol("x")]) {
      expect(() => validateCaption(value, SEGMENTS_BEAT, SEGMENTS_CONTEXT)).not.toThrow();
      expect(validateCaption(value, SEGMENTS_BEAT, SEGMENTS_CONTEXT).ok).toBe(false);
    }
  });

  it("always rejects a caption carrying a number absent from the beat", () => {
    // Numbers chosen to be absent from every fixture beat.
    for (const n of [7777, 411, 8888.5, 1234]) {
      for (const [beat, ctx] of [
        [SEGMENTS_BEAT, SEGMENTS_CONTEXT],
        [FLOOR_BEAT, FLOOR_CONTEXT],
        [METRICS_BEAT, METRICS_CONTEXT],
      ]) {
        const result = validateCaption(`A container recorded ${n} for the request.`, beat, ctx);
        expect(result.ok, `${n} should be rejected`).toBe(false);
        expect(rules(result)).toContain("BR3-13");
      }
    }
  });

  it("the factual caption always passes its own validation", () => {
    const beats = [
      { kind: BEAT_ORIGIN, values: [], latencyMs: null },
      SEGMENTS_BEAT,
      FLOOR_BEAT,
      METRICS_BEAT,
      { kind: BEAT_CONTAINER, displayLabel: "Bid Pricer", intent: "BID_SHADE", latencyMs: 12,
        values: [{ kind: "price", before: 2.5, after: 2.1 }] },
      { kind: BEAT_CONTAINER, displayLabel: "Yield Optimizer", intent: "ADJUST_DEAL_MARGIN", latencyMs: 5,
        values: [{ kind: "margin", dealId: "d1", value: 0.14, calculationType: "PERCENT" }] },
      { kind: BEAT_CONTAINER, displayLabel: "Yield Optimizer", intent: "ADJUST_DEAL_MARGIN", latencyMs: 5,
        values: [{ kind: "margin", dealId: "d1", value: 0.5, calculationType: "CPM" }] },
      { kind: BEAT_CONTAINER, displayLabel: "Deal Scorer", intent: "UNRENDERED", latencyMs: 9, values: [] },
      { kind: BEAT_RECAP, contributors: [], values: [] },
      { kind: BEAT_RECAP, values: [],
        contributors: [{ containerName: "a", displayLabel: "A", beatIndexes: [1] }] },
    ];
    const contexts = [SEGMENTS_CONTEXT, FLOOR_CONTEXT, METRICS_CONTEXT, {}];

    for (const beat of beats) {
      for (const context of contexts) {
        const text = factualCaption(beat, context);
        if (text === "") continue;
        const result = validateCaption(text, beat, context);
        expect(result.ok, `${text} -> ${JSON.stringify(rules(result))}`).toBe(true);
      }
    }
  });

  it("accepted fixtures all sit inside both ceilings", () => {
    for (const fixture of ACCEPTED) {
      expect(fixture.text.length).toBeLessThanOrEqual(CHAR_CEILING);
      expect(fixture.text.trim().split(/\s+/).length).toBeLessThanOrEqual(WORD_CEILING);
    }
  });
});

describe("extraction helpers", () => {
  it("tags units as written", () => {
    const claims = extractNumericClaims("1.2 USD and $1.68 and 41 milliseconds and 72% and 0.95 and 50 cents");
    const byRaw = Object.fromEntries(claims.map((c) => [c.raw, c.unit]));
    expect(byRaw["1.2 USD"]).toBe("currency");
    expect(byRaw["$1.68"]).toBe("currency");
    expect(byRaw["41 milliseconds"]).toBe("ms");
    expect(byRaw["72%"]).toBe("percent");
    expect(byRaw["0.95"]).toBe("bare");
    expect(byRaw["50 cents"]).toBe("cents");
  });

  it("derives the beat's supportable numbers", () => {
    const allowed = allowedNumericClaims(FLOOR_BEAT, FLOOR_CONTEXT);
    const values = allowed.map((a) => a.value);
    expect(values).toContain(1.68);
    expect(values).toContain(1.2);
    expect(values).toContain(23);
    expect(values).not.toContain(47);
  });
});
