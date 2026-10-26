import { describe, it, expect } from "vitest";
import {
  validateRunSummary,
  aggregateBeat,
  auctionLiterals,
  auctionNumbers,
} from "./runSummaryValidation.js";
import { BEAT_CONTAINER, BEAT_ORIGIN, BEAT_RECAP } from "./theaterBeats.js";

const BEATS = [
  { index: 0, kind: BEAT_ORIGIN, values: [] },
  {
    index: 1,
    kind: BEAT_CONTAINER,
    containerName: "widedeep-segment-activator",
    displayLabel: "Audience Activator",
    intent: "ACTIVATE_SEGMENTS",
    latencyMs: 6.8,
    explored: false,
    values: [{ kind: "ids", role: "segments", ids: ["350", "354"] }],
  },
  {
    index: 2,
    kind: BEAT_CONTAINER,
    containerName: "yield-optimizer-floor",
    displayLabel: "Yield Optimizer — Floor",
    intent: "ADJUST_DEAL_FLOOR",
    latencyMs: 10.9,
    explored: false,
    values: [{ kind: "floor", dealId: "deal-parenting-premium", before: 3.4, after: 3.9 }],
  },
  { index: 3, kind: BEAT_RECAP, values: [], contributors: [{ containerName: "a", displayLabel: "A" }] },
];

const CONTEXT = {
  publisher: "parenting-weekly.example",
  page: "https://parenting-weekly.example/g",
  bidFloor: 2.6,
  categoryTaxonomy: 9,
  contentCategories: ["192"],
  impressionFormat: "banner",
  deals: [{ id: "deal-parenting-premium", bidFloor: 3.4, auctionType: 1 }],
  userSignals: [{ label: "data segment", value: "Parents with Children", provenance: "request" }],
};

const VIEW_MODEL = {
  offers: [
    { key: "a", campaignName: "brightstart.example", dealId: "deal-parenting-premium", seat: "amt", price: 4.1, offered: true },
    { key: "b", campaignName: "familynetwork.example", dealId: "deal-family-network", seat: "amt", price: 3.05, offered: true },
  ],
  bidRows: [{ key: "a" }, { key: "b" }],
  groups: [{ id: "decision", count: 1, label: "1 campaign stopped by a sell-side decision", breakdown: [{ label: "below floor", count: 1 }], rows: [] }],
  winner: { dealId: "deal-parenting-premium", campaignName: "brightstart.example", clearedPrice: 4.1, offerKey: "a" },
  notice: "declared_inventory",
  currency: "USD",
};

describe("aggregateBeat", () => {
  it("collects every container beat's values into one pseudo-beat", () => {
    const agg = aggregateBeat(BEATS);
    expect(agg.values).toHaveLength(2);
    expect(agg.contributors.map((c) => c.displayLabel))
      .toEqual(["Audience Activator", "Yield Optimizer — Floor"]);
  });

  it("carries no latency, because no measured run total exists", () => {
    // Allowing one would let the summary state a single container's latency as if
    // it described the whole run.
    expect(aggregateBeat(BEATS).latencyMs).toBeNull();
  });

  it("returns an empty aggregate for no beats rather than throwing", () => {
    expect(aggregateBeat(null).values).toEqual([]);
  });
});

describe("auctionLiterals / auctionNumbers", () => {
  it("sources campaign names, deal ids and seats from the auction", () => {
    const lits = auctionLiterals(VIEW_MODEL);
    expect(lits).toContain("brightstart.example");
    expect(lits).toContain("deal-family-network");
    expect(lits).toContain("amt");
  });

  it("sources the cleared price and every declared price", () => {
    const nums = auctionNumbers(VIEW_MODEL).map((n) => n.value);
    expect(nums).toContain(4.1);
    expect(nums).toContain(3.05);
  });
});

describe("validateRunSummary", () => {
  it("accepts a summary whose numbers all trace to the run or the auction", () => {
    const text = [
      "The banner impression came from parenting-weekly.example.",
      "Audience Activator added two audience segments and Yield Optimizer raised the",
      "deal floor to $3.90 from $3.40.",
      "brightstart.example won deal-parenting-premium at $4.10.",
    ].join(" ");
    const verdict = validateRunSummary(text, BEATS, CONTEXT, VIEW_MODEL);
    expect(verdict.ok).toBe(true);
  });

  it("rejects a summary that invents a price", () => {
    const text = "brightstart.example won deal-parenting-premium at $9.99.";
    const verdict = validateRunSummary(text, BEATS, CONTEXT, VIEW_MODEL);
    expect(verdict.ok).toBe(false);
    expect(verdict.violations.some((v) => v.rule === "BR3-13")).toBe(true);
  });

  it("rejects a number that would only be legal in another unit", () => {
    // 10.9 is a real latency in ms on one beat. As a dollar figure it is invented,
    // and the aggregate carries no latency at all.
    const verdict = validateRunSummary(
      "brightstart.example won at $10.90.", BEATS, CONTEXT, VIEW_MODEL,
    );
    expect(verdict.ok).toBe(false);
  });

  it("allows more words than a single-beat caption", () => {
    // Four facts do not fit the per-beat 45-word budget. 60 words must pass here
    // and would fail validateCaption's default ceiling.
    const text = `${"word ".repeat(59)}end.`;
    const verdict = validateRunSummary(text, BEATS, CONTEXT, VIEW_MODEL);
    expect(verdict.ok).toBe(true);
  });

  it("still rejects promotional register", () => {
    const verdict = validateRunSummary(
      "This seamless auction was a testament to robust optimisation.",
      BEATS, CONTEXT, VIEW_MODEL,
    );
    expect(verdict.ok).toBe(false);
    expect(verdict.violations.some((v) => v.rule === "BR3-19")).toBe(true);
  });

  it("still rejects em-dashes and first person", () => {
    const verdict = validateRunSummary(
      "We saw the winner clear \u2014 it was brightstart.example.",
      BEATS, CONTEXT, VIEW_MODEL,
    );
    expect(verdict.ok).toBe(false);
    const rules = verdict.violations.map((v) => v.rule);
    expect(rules).toContain("BR3-17");
    expect(rules).toContain("BR3-18");
  });
});
