// normalizer.test.js — guards against dropping a registered container's
// stop from the normalized result. Regression coverage for a bug where
// CONTAINER_STOP_IDS omitted "yield" (deal-yield-manager): the orchestrator
// routed ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN correctly and returned real
// mutations, but the frontend normalizer silently dropped that stop before
// it ever reached the timeline or mutations panel.

import { describe, it, expect } from "vitest";
import { normalize, toMutationModel } from "./normalizer.js";

function makeRawWithContainers(containers) {
  return {
    id: "req-1",
    metadata: {
      total_latency_ms: 45,
      containers,
    },
    mutations: containers.flatMap((c) => c.mutations || []),
  };
}

describe("normalize() — explicit container stops", () => {
  it("includes a stop for every registered container, not just the original four", () => {
    const raw = makeRawWithContainers([
      { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [] },
      { name: "widedeep-segment-activator", status: "ok", latency_ms: 8, mutations: [] },
      { name: "ncf-deal-manager", status: "ok", latency_ms: 12, mutations: [] },
      { name: "metrics-enricher", status: "ok", latency_ms: 5, mutations: [] },
      { name: "deal-yield-manager", status: "ok", latency_ms: 21, mutations: [] },
    ]);

    const result = normalize(raw, "grpc", {});
    const stopIds = result.stops.map((s) => s.id);

    expect(stopIds).toEqual(["ssp", "dlrm", "widedeep", "ncf", "metrics", "yield", "dsp"]);
  });

  it("surfaces deal-yield-manager's real mutations on the yield stop, not dropped", () => {
    const floorMutation = {
      intent: 4, // ADJUST_DEAL_FLOOR
      op: 3, // REPLACE
      path: "/imp/imp-1/deals/deal-guaranteed-premium",
      adjust_deal: { bidfloor: 11.4 },
    };
    const marginMutation = {
      intent: 5, // ADJUST_DEAL_MARGIN
      op: 3,
      path: "/imp/imp-1/deals/deal-open-midtier",
      adjust_deal: { margin: { value: 0.12, calculation_type: 1 } },
    };

    const raw = makeRawWithContainers([
      { name: "deal-yield-manager", status: "ok", latency_ms: 21, mutations: [floorMutation, marginMutation] },
    ]);

    const result = normalize(raw, "grpc", {});
    const yieldStop = result.stops.find((s) => s.id === "yield");

    expect(yieldStop).toBeDefined();
    expect(yieldStop.status).toBe("ok");
    expect(yieldStop.latency).toEqual({ ms: 21, source: "orchestrator" });
    expect(yieldStop.mutations).toHaveLength(2);
    expect(yieldStop.mutations.map((m) => m.intent)).toEqual([
      "ADJUST_DEAL_FLOOR",
      "ADJUST_DEAL_MARGIN",
    ]);

    // These mutations must be reachable via the flattened stops list too,
    // since computeDiffRows/DemoSequenceOrchestrator both flatMap over
    // result.stops rather than reading a container-keyed map directly.
    const allMutations = result.stops.flatMap((s) => s.mutations || []);
    expect(allMutations.filter((m) => m.intent === "ADJUST_DEAL_FLOOR")).toHaveLength(1);
    expect(allMutations.filter((m) => m.intent === "ADJUST_DEAL_MARGIN")).toHaveLength(1);
  });

  it("gives yield a placeholder 'unknown' stop when the container wasn't invoked at all", () => {
    const raw = makeRawWithContainers([
      { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [] },
    ]);

    const result = normalize(raw, "grpc", {});
    const yieldStop = result.stops.find((s) => s.id === "yield");

    expect(yieldStop).toBeDefined();
    expect(yieldStop.status).toBe("unknown");
    expect(yieldStop.mutations).toEqual([]);
  });
});

describe("normalize() — inferred container stops (no metadata.containers)", () => {
  it("buckets ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN mutations into the yield stop", () => {
    const raw = {
      id: "req-2",
      mutations: [
        { intent: 4, op: 3, path: "/imp/imp-1/deals/deal-a", adjust_deal: { bidfloor: 9.0 } },
        { intent: 5, op: 3, path: "/imp/imp-1/deals/deal-b", adjust_deal: { margin: { value: 0.1, calculation_type: 1 } } },
      ],
    };

    const result = normalize(raw, "grpc", {});
    const stopIds = result.stops.map((s) => s.id);
    expect(stopIds).toContain("yield");

    const yieldStop = result.stops.find((s) => s.id === "yield");
    expect(yieldStop.mutations).toHaveLength(2);
  });

  it("does not throw when bucketing an ADJUST_DEAL_FLOOR mutation (regression: missing 'yield' bucket key)", () => {
    const raw = {
      id: "req-3",
      mutations: [{ intent: 4, op: 3, path: "/imp/imp-1/deals/deal-a", adjust_deal: { bidfloor: 5.0 } }],
    };

    expect(() => normalize(raw, "grpc", {})).not.toThrow();
  });
});

describe("normalize() — error fallback stops", () => {
  it("still includes a yield placeholder stop when the orchestrator returns an RPC error", () => {
    const raw = { id: "req-4", error: { message: "boom" } };

    const result = normalize(raw, "grpc", {});
    const stopIds = result.stops.map((s) => s.id);

    expect(stopIds).toEqual(["ssp", "dlrm", "widedeep", "ncf", "metrics", "yield", "dsp"]);
    expect(result.error).toEqual({ message: "boom", stage: "orchestrator" });
  });
});

describe("toMutationModel() — ADJUST_DEAL_FLOOR / ADJUST_DEAL_MARGIN payload extraction", () => {
  it("extracts the adjust_deal payload for ADJUST_DEAL_FLOOR", () => {
    const m = toMutationModel({
      intent: 4,
      op: 3,
      path: "/imp/imp-1/deals/deal-a",
      adjust_deal: { bidfloor: 7.25 },
    });

    expect(m.intent).toBe("ADJUST_DEAL_FLOOR");
    expect(m.op).toBe("REPLACE");
    expect(m.payload).toEqual({ bidfloor: 7.25 });
  });

  it("extracts the adjust_deal payload for ADJUST_DEAL_MARGIN", () => {
    const m = toMutationModel({
      intent: 5,
      op: 3,
      path: "/imp/imp-1/deals/deal-b",
      adjust_deal: { margin: { value: 0.15, calculation_type: 0 } },
    });

    expect(m.intent).toBe("ADJUST_DEAL_MARGIN");
    expect(m.payload).toEqual({ margin: { value: 0.15, calculation_type: 0 } });
  });
});
