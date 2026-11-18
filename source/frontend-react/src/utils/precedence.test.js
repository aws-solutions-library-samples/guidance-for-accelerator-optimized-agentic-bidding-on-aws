// precedence.test.js — the frontend side of mutation precedence.
//
// Two containers may claim the same intent. Both are called and both return a
// mutation, but the orchestrator now returns only the winner in the flattened
// list, and reports per container how many of its mutations were superseded.
//
// Two things this file is specifically protecting:
//
//  1. The normalizer builds its stops from `metadata.containers`, NOT from the
//     flattened `mutations` list. That is what lets a superseded mutation stay
//     visible in the pipeline and the Theater even though it was filtered out of
//     the response's top-level list. If a change ever switched the stop source
//     to `raw.mutations`, the overridden container would silently show nothing.
//  2. `precedenceNote` has to agree with the orchestrator's rule — higher
//     priority wins, equal priorities fall back to registry order where store
//     containers follow built-in ones. A UI that disagrees with the bid path is
//     worse than a UI that says nothing.

import { describe, it, expect } from "vitest";
import { normalize } from "./normalizer.js";
import { hasSharedIntent, precedenceNote } from "../components/ContainersPanel.jsx";

const FLOOR_INTENT = 4;
const DEAL_PATH = "/imp/imp-1/deals/deal-a";

function floorMutation() {
  return { intent: FLOOR_INTENT, op: 2, path: DEAL_PATH };
}

/** A response where the store container's floor won and the built-in's lost. */
function contestedRaw() {
  return {
    id: "req-1",
    // Only the winner is in the flattened list — this is the filtering the
    // orchestrator now does.
    mutations: [floorMutation()],
    metadata: {
      total_latency_ms: 40,
      containers: [
        {
          name: "yield-optimizer-floor",
          display_name: "Yield Optimizer",
          status: "ok",
          latency_ms: 8,
          // Still listed: the container really computed it.
          mutations: [floorMutation()],
          superseded: 1,
        },
        {
          name: "contextual-yield-agent",
          display_name: "Contextual Yield Agent",
          status: "ok",
          latency_ms: 11,
          mutations: [floorMutation()],
          superseded: 0,
        },
      ],
      conflicts: [
        {
          path: DEAL_PATH,
          intent: FLOOR_INTENT,
          winner: "contextual-yield-agent",
          losers: ["yield-optimizer-floor"],
        },
      ],
    },
  };
}

describe("normalizer — superseded passthrough", () => {
  it("keeps the superseded mutation visible on its own stop", () => {
    const result = normalize(contestedRaw(), "grpc", {});
    const floor = result.stops.find((s) => s.id === "yield-floor");
    expect(floor).toBeTruthy();
    // The mutation was filtered out of raw.mutations but is still on the stop,
    // because stops come from metadata.containers.
    expect(floor.mutations).toHaveLength(1);
    expect(floor.superseded).toBe(1);
  });

  it("reports zero superseded for the winner", () => {
    const result = normalize(contestedRaw(), "grpc", {});
    const external = result.stops.find((s) => s.id === "dynamic:contextual-yield-agent");
    expect(external).toBeTruthy();
    expect(external.superseded).toBe(0);
    expect(external.displayName).toBe("Contextual Yield Agent");
  });

  it("surfaces the conflicts block", () => {
    const result = normalize(contestedRaw(), "grpc", {});
    expect(result.conflicts).toHaveLength(1);
    expect(result.conflicts[0].winner).toBe("contextual-yield-agent");
    expect(result.conflicts[0].losers).toEqual(["yield-optimizer-floor"]);
  });

  it("defaults superseded to 0 when the orchestrator does not send it", () => {
    const raw = contestedRaw();
    for (const c of raw.metadata.containers) delete c.superseded;
    const result = normalize(raw, "grpc", {});
    for (const stop of result.stops) {
      if (stop.id === "ssp" || stop.id === "dsp") continue;
      expect(stop.superseded).toBe(0);
    }
  });

  it("normalises a missing conflicts key to an empty array", () => {
    const raw = contestedRaw();
    delete raw.metadata.conflicts;
    expect(normalize(raw, "grpc", {}).conflicts).toEqual([]);
  });

  it("ignores a non-numeric superseded value", () => {
    const raw = contestedRaw();
    raw.metadata.containers[0].superseded = "lots";
    const floor = normalize(raw, "grpc", {}).stops.find((s) => s.id === "yield-floor");
    expect(floor.superseded).toBe(0);
  });
});

describe("hasSharedIntent", () => {
  const shared = { ADJUST_DEAL_FLOOR: ["yield-optimizer-floor", "contextual-yield-agent"] };

  it("is true when one of the container's intents is contested", () => {
    expect(hasSharedIntent({ intents: ["ADJUST_DEAL_FLOOR"] }, shared)).toBe(true);
  });

  it("is false when nothing is contested", () => {
    expect(hasSharedIntent({ intents: ["ADD_CIDS"] }, shared)).toBe(false);
  });

  it("is false for a single claimant", () => {
    expect(hasSharedIntent({ intents: ["X"] }, { X: ["only-me"] })).toBe(false);
  });

  it("tolerates missing intents and a missing map", () => {
    expect(hasSharedIntent({}, shared)).toBe(false);
    expect(hasSharedIntent({ intents: ["ADJUST_DEAL_FLOOR"] }, undefined)).toBe(false);
  });
});

describe("precedenceNote — must agree with the orchestrator's rule", () => {
  const shared = { ADJUST_DEAL_FLOOR: ["yield-optimizer-floor", "contextual-yield-agent"] };

  // Registry order: code containers first, then store containers. So the store
  // container is LATER in this list, which is how it wins on a tie.
  const registryOrder = [
    { name: "yield-optimizer-floor", priority: 0, intents: ["ADJUST_DEAL_FLOOR"] },
    { name: "contextual-yield-agent", priority: 0, intents: ["ADJUST_DEAL_FLOOR"] },
  ];

  it("on a tie the later registry entry wins, matching last-write-wins", () => {
    const note = precedenceNote(registryOrder[1], shared, registryOrder);
    expect(note).toContain("applied");

    const builtinNote = precedenceNote(registryOrder[0], shared, registryOrder);
    expect(builtinNote).toContain("contextual-yield-agent outranks it");
    expect(builtinNote).toContain("discarded");
  });

  it("a higher priority beats registry order", () => {
    const withPriority = [
      { name: "yield-optimizer-floor", priority: 5, intents: ["ADJUST_DEAL_FLOOR"] },
      { name: "contextual-yield-agent", priority: 0, intents: ["ADJUST_DEAL_FLOOR"] },
    ];
    expect(precedenceNote(withPriority[0], shared, withPriority)).toContain("applied");
    expect(precedenceNote(withPriority[1], shared, withPriority)).toContain(
      "yield-optimizer-floor outranks it"
    );
  });

  it("a negative priority loses to the default", () => {
    const withNegative = [
      { name: "yield-optimizer-floor", priority: 0, intents: ["ADJUST_DEAL_FLOOR"] },
      { name: "contextual-yield-agent", priority: -1, intents: ["ADJUST_DEAL_FLOOR"] },
    ];
    expect(precedenceNote(withNegative[1], shared, withNegative)).toContain(
      "yield-optimizer-floor outranks it"
    );
  });

  it("says nothing contested when nothing is", () => {
    expect(
      precedenceNote({ name: "a", priority: 0, intents: ["ADD_CIDS"] }, shared, registryOrder)
    ).toBe("nothing contested.");
  });

  it("treats a missing priority as 0", () => {
    const noPriority = [
      { name: "yield-optimizer-floor", intents: ["ADJUST_DEAL_FLOOR"] },
      { name: "contextual-yield-agent", intents: ["ADJUST_DEAL_FLOOR"] },
    ];
    // Still resolves on registry order rather than throwing.
    expect(precedenceNote(noPriority[1], shared, noPriority)).toContain("applied");
  });

  it("names every outranking container when several intents are contested", () => {
    const multi = {
      ADJUST_DEAL_FLOOR: ["floor-builtin", "ext"],
      ADD_METRICS: ["metrics-builtin", "ext"],
    };
    const all = [
      { name: "floor-builtin", priority: 9, intents: ["ADJUST_DEAL_FLOOR"] },
      { name: "metrics-builtin", priority: 9, intents: ["ADD_METRICS"] },
      { name: "ext", priority: 0, intents: ["ADJUST_DEAL_FLOOR", "ADD_METRICS"] },
    ];
    const note = precedenceNote(all[2], multi, all);
    expect(note).toContain("floor-builtin");
    expect(note).toContain("metrics-builtin");
  });
});
