// theaterPass2Source.test.js — the with-ARTF pass built from the auction's own
// report (artf_meta.extension_point), with the exchange's disposition on every
// mutation.
//
// Example-based. One normalizer and one beat builder serve both sources, so these
// pin what the auction report adds: dispositions carried through, declined
// mutations marked and never counted as changes, and a recap that says why
// nothing is shown when the hook's report has nothing to show.
import { describe, it, expect } from "vitest";
import { normalize, normalizeExtensionPoint, toMutationModel } from "./normalizer.js";
import {
  buildBeats,
  visibleValues,
  isAppliedValue,
  appliedValues,
  BEAT_CONTAINER,
  BEAT_RECAP,
} from "./theaterBeats.js";
import { factualCaption, mutationNarration } from "./theaterCaptions.js";
import { validateCaption } from "./captionValidation.js";
import { buildCaptionPrompt } from "./captionPrompt.js";
import { deriveSellSideKpis, effectiveFloor } from "./sellSideKpis.js";
import { compareOutcomes } from "./outcomeComparison.js";

const PATH = "/imp/imp-1/deals/deal-a";
const SUBMITTED = {
  bid_request: {
    id: "req-1",
    imp: [{ id: "imp-1", bidfloor: 2.0, pmp: { deals: [{ id: "deal-a", bidfloor: 2.0 }] } }],
  },
};
const CONTEXT = {
  publisher: "example.com",
  impressionFormat: "banner",
  bidFloor: 2.0,
  deals: [{ id: "deal-a", bidFloor: 2.0, auctionType: 1 }],
  userSignals: [],
};

const floor = (bidfloor, disposition, reason = null) => ({
  intent: 4, op: 2, path: PATH, adjust_deal: { bidfloor },
  disposition, disposition_reason: reason,
});
const segments = { intent: 1, op: 1, path: "/user/data", ids: { id: ["seg-1"] }, disposition: "applied" };

/**
 * An extension point whose two floor containers contested deal-a's floor.
 * display_name is set as the orchestrator sends it; the build-time fallback
 * labels in intentMapping.js contain an em dash the caption validator rejects.
 */
function extensionPoint({ winnerDisposition = "applied", reason = null } = {}) {
  return {
    available: true,
    outcome: "mutations_returned",
    response: {
      id: "tid-1",
      mutations: [segments, floor(2.6, winnerDisposition, reason)],
      metadata: {
        containers: [
          { name: "widedeep-segment-activator", display_name: "Audience Activator", status: "ok", latency_ms: 2, mutations: [segments] },
          { name: "yield-optimizer-floor", display_name: "Yield Optimizer Floor", status: "ok", latency_ms: 3,
            mutations: [floor(2.4, "superseded", "superseded by ncf-deal-manager")] },
          { name: "ncf-deal-manager", display_name: "Deal Scorer", status: "ok", latency_ms: 4,
            mutations: [floor(2.6, winnerDisposition, reason)] },
        ],
        conflicts: [{ path: PATH, intent: 4, winner: "ncf-deal-manager", losers: ["yield-optimizer-floor"] }],
      },
    },
  };
}

function beatsFor(ep) {
  return buildBeats(SUBMITTED, normalizeExtensionPoint(ep, SUBMITTED));
}

describe("normalizer: the auction report goes through the same normalizer", () => {
  it("carries the disposition and its reason on every mutation", () => {
    const m = toMutationModel(floor(2.6, "rejected", "floor 2.60 exceeds cap"));
    expect(m.disposition).toBe("rejected");
    expect(m.dispositionReason).toBe("floor 2.60 exceeds cap");
  });

  it("a plain /v1/mutations answer has no disposition and no artfReport", () => {
    expect(toMutationModel({ intent: 4, op: 2, path: PATH, adjust_deal: { bidfloor: 1 } }).disposition).toBeNull();
    expect(normalize({ id: "x", mutations: [], metadata: { containers: [] } }, "REST", SUBMITTED).artfReport)
      .toBeUndefined();
  });

  it("an unknown disposition word is not trusted", () => {
    expect(toMutationModel({ intent: 1, disposition: "maybe" }).disposition).toBeNull();
  });

  it("a report with no response has no container mutations and says why", () => {
    const result = normalizeExtensionPoint(
      { available: true, outcome: "timeout", response: null }, SUBMITTED);
    expect(result.stops.every((s) => s.mutations.length === 0)).toBe(true);
    expect(result.artfReport).toEqual({ source: "auction", available: true, outcome: "timeout", reason: null });
  });
});

describe("buildBeats: dispositions on beats and values", () => {
  it("stamps each container beat and its values with the exchange's disposition", () => {
    const beats = beatsFor(extensionPoint());
    const byContainer = Object.fromEntries(
      beats.filter((b) => b.kind === BEAT_CONTAINER).map((b) => [b.containerName, b]));
    expect(byContainer["yield-floor"].disposition).toBe("superseded");
    expect(byContainer["yield-floor"].dispositionReason).toBe("superseded by ncf-deal-manager");
    expect(byContainer["yield-floor"].values[0].disposition).toBe("superseded");
    expect(byContainer.ncf.disposition).toBe("applied");
  });

  it("the recap does not credit a container whose only mutation was superseded", () => {
    const beats = beatsFor(extensionPoint());
    const recap = beats.find((b) => b.kind === BEAT_RECAP);
    const names = recap.contributors.map((c) => c.containerName);
    expect(names).toContain("ncf");
    expect(names).not.toContain("yield-floor");
    expect(recap.artfReport).toMatchObject({ source: "auction", available: true });
  });

  it("isAppliedValue treats a missing disposition as applied (the fallback source)", () => {
    expect(isAppliedValue({ kind: "floor" })).toBe(true);
    expect(isAppliedValue({ kind: "floor", disposition: "applied" })).toBe(true);
    expect(isAppliedValue({ kind: "floor", disposition: "rejected" })).toBe(false);
    expect(appliedValues([{ disposition: "superseded" }, {}])).toHaveLength(1);
  });
});

describe("narration and captions", () => {
  it("a declined mutation says so in fixed words, and its reason stays out of the caption", () => {
    const beats = beatsFor(extensionPoint({ winnerDisposition: "rejected", reason: "floor 2.60 exceeds cap 2.50" }));
    const rejected = beats.find((b) => b.kind === BEAT_CONTAINER && b.disposition === "rejected");
    const narration = mutationNarration(rejected, CONTEXT);
    expect(narration.text).toMatch(/Not applied: rejected by the exchange\./);
    expect(narration.text).not.toMatch(/2\.50/);
    expect(narration.dispositionReason).toBe("floor 2.60 exceeds cap 2.50");
    // BR3-13: the text that reaches factualCaption clears the validator.
    const result = validateCaption(factualCaption(rejected, CONTEXT), rejected, CONTEXT);
    expect(result.ok, JSON.stringify(result.violations)).toBe(true);
  });

  it("every beat of an auction-sourced run passes the caption validator", () => {
    for (const beat of beatsFor(extensionPoint())) {
      const result = validateCaption(factualCaption(beat, CONTEXT), beat, CONTEXT);
      expect(result.ok, `${beat.kind}: ${JSON.stringify(result.violations)}`).toBe(true);
    }
  });

  it.each([
    [{ available: false, reason: "no trace" }, /did not report the ARTF hook's mutations/],
    [{ available: true, outcome: "timeout", response: null }, /timed out, so this auction ran unmutated/],
    [{ available: true, outcome: "transport_failure", response: null }, /could not reach the orchestrator/],
    [{ available: true, outcome: "skipped_insufficient_budget", response: null }, /too little time/],
  ])("the recap states an empty hook report in words: %j", (ep, pattern) => {
    const recap = beatsFor(ep).find((b) => b.kind === BEAT_RECAP);
    const text = factualCaption(recap, CONTEXT);
    expect(text).toMatch(pattern);
    expect(text).not.toMatch(/\d/);
    expect(validateCaption(text, recap, CONTEXT).ok).toBe(true);
  });

  it("an answered hook with nothing applied keeps the pinned wording", () => {
    const recap = beatsFor({ available: true, outcome: "no_mutations",
      response: { id: "t", mutations: [], metadata: { containers: [] } } })
      .find((b) => b.kind === BEAT_RECAP);
    expect(factualCaption(recap, CONTEXT)).toBe("No container mutated this request.");
  });

  it("the caption prompt names a declined mutation without its reason", () => {
    const beats = beatsFor(extensionPoint({ winnerDisposition: "rejected", reason: "cap 2.50" }));
    const rejected = beats.find((b) => b.kind === BEAT_CONTAINER && b.disposition === "rejected");
    const { message } = buildCaptionPrompt(rejected, CONTEXT);
    expect(JSON.parse(message).step.not_applied).toBe("rejected");
    expect(message).not.toMatch(/cap 2\.50/);
  });
});

describe("sell side and comparison read the same values with the same rule", () => {
  const VM = {
    winner: { dealId: "deal-a", campaignName: "c", clearedPrice: 3.0 },
    bidRows: [], groups: [],
  };

  it("a superseded floor does not become the effective floor", () => {
    const beats = beatsFor(extensionPoint({ winnerDisposition: "rejected", reason: "cap" }));
    const values = visibleValues(beats, beats.length - 1);
    // Both floor values are on screen, marked; neither was applied.
    expect(values.filter((v) => v.kind === "floor")).toHaveLength(2);
    expect(effectiveFloor(values, CONTEXT, "deal-a")).toMatchObject({ source: "as_sent", value: 2.0 });
    const kpis = deriveSellSideKpis({ values, context: CONTEXT, viewModel: VM });
    expect(kpis.rows.find((r) => r.id === "floor-adjusted")).toBeUndefined();
  });

  it("the applied floor is the one the KPIs report", () => {
    const beats = beatsFor(extensionPoint());
    const values = visibleValues(beats, beats.length - 1);
    expect(effectiveFloor(values, CONTEXT, "deal-a")).toMatchObject({ source: "adjusted", value: 2.6 });
  });

  it("the comparison's mutations row counts only containers whose values were applied", () => {
    const beats = beatsFor(extensionPoint());
    const values = visibleValues(beats, beats.length - 1);
    const comparison = compareOutcomes({ baselineVM: VM, artfVM: VM, values, context: CONTEXT });
    const row = comparison.rows.find((r) => r.id === "mutations");
    expect(row.with).toBe("applied by 2 containers");
    expect(row.detail).not.toMatch(/Yield Optimizer Floor/);
  });
});
