// normalizer.test.js — guards against dropping a registered container's
// stop from the normalized result. Regression coverage for a bug where
// CONTAINER_STOP_IDS omitted the yield stop entirely: the orchestrator routed
// ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN correctly and returned real mutations,
// but the frontend normalizer silently dropped that stop before it ever
// reached the timeline or mutations panel.
//
// The yield containers are now two stops (yield-floor, yield-margin), one per
// container. buildExplicitContainerStops() keeps the FIRST entry per stop id,
// so mapping both containers onto one shared id would silently discard the
// second one's status, latency and mutations -- the same class of bug this
// file exists to catch. See the "keeps both yield containers separate" case.

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
      { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [] },
      { name: "yield-optimizer-margin", status: "ok", latency_ms: 19, mutations: [] },
    ]);

    const result = normalize(raw, "grpc", {});
    const stopIds = result.stops.map((s) => s.id);

    expect(stopIds).toEqual([
      "ssp", "dlrm", "widedeep", "ncf", "metrics", "yield-floor", "yield-margin", "dsp",
    ]);
  });

  it("keeps both yield containers separate, with their own status and latency", () => {
    const raw = makeRawWithContainers([
      { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [] },
      { name: "yield-optimizer-margin", status: "unreachable", latency_ms: 19, mutations: [] },
    ]);

    const result = normalize(raw, "grpc", {});
    const floor = result.stops.find((s) => s.id === "yield-floor");
    const margin = result.stops.find((s) => s.id === "yield-margin");

    // A shared stop id would have collapsed these into one entry, hiding the
    // fact that only one of the two models is actually down.
    expect(floor.status).toBe("ok");
    expect(floor.latency).toEqual({ ms: 21, source: "orchestrator" });
    expect(margin.status).toBe("unreachable");
    expect(margin.latency).toEqual({ ms: 19, source: "orchestrator" });
  });

  it("surfaces each yield container's real mutations on its own stop, not dropped", () => {
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
      { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [floorMutation] },
      { name: "yield-optimizer-margin", status: "ok", latency_ms: 19, mutations: [marginMutation] },
    ]);

    const result = normalize(raw, "grpc", {});
    const floorStop = result.stops.find((s) => s.id === "yield-floor");
    const marginStop = result.stops.find((s) => s.id === "yield-margin");

    expect(floorStop).toBeDefined();
    expect(floorStop.status).toBe("ok");
    expect(floorStop.latency).toEqual({ ms: 21, source: "orchestrator" });
    expect(floorStop.mutations.map((m) => m.intent)).toEqual(["ADJUST_DEAL_FLOOR"]);

    expect(marginStop).toBeDefined();
    expect(marginStop.mutations.map((m) => m.intent)).toEqual(["ADJUST_DEAL_MARGIN"]);

    // These mutations must be reachable via the flattened stops list too,
    // since computeDiffRows/DemoSequenceOrchestrator both flatMap over
    // result.stops rather than reading a container-keyed map directly.
    const allMutations = result.stops.flatMap((s) => s.mutations || []);
    expect(allMutations.filter((m) => m.intent === "ADJUST_DEAL_FLOOR")).toHaveLength(1);
    expect(allMutations.filter((m) => m.intent === "ADJUST_DEAL_MARGIN")).toHaveLength(1);
  });

  it.each(["yield-floor", "yield-margin"])(
    "gives %s a placeholder 'unknown' stop when that container wasn't invoked at all",
    (stopId) => {
      const raw = makeRawWithContainers([
        { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [] },
      ]);

      const result = normalize(raw, "grpc", {});
      const stop = result.stops.find((s) => s.id === stopId);

      expect(stop).toBeDefined();
      expect(stop.status).toBe("unknown");
      expect(stop.mutations).toEqual([]);
    },
  );
});

describe("normalize() — inferred container stops (no metadata.containers)", () => {
  it("buckets ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN into their own yield stops", () => {
    const raw = {
      id: "req-2",
      mutations: [
        { intent: 4, op: 3, path: "/imp/imp-1/deals/deal-a", adjust_deal: { bidfloor: 9.0 } },
        { intent: 5, op: 3, path: "/imp/imp-1/deals/deal-b", adjust_deal: { margin: { value: 0.1, calculation_type: 1 } } },
      ],
    };

    const result = normalize(raw, "grpc", {});
    const stopIds = result.stops.map((s) => s.id);
    expect(stopIds).toContain("yield-floor");
    expect(stopIds).toContain("yield-margin");

    // One mutation each -- not both piled onto a single stop.
    expect(result.stops.find((s) => s.id === "yield-floor").mutations).toHaveLength(1);
    expect(result.stops.find((s) => s.id === "yield-margin").mutations).toHaveLength(1);
  });

  it("does not throw when bucketing an ADJUST_DEAL_FLOOR mutation (regression: missing yield bucket key)", () => {
    const raw = {
      id: "req-3",
      mutations: [{ intent: 4, op: 3, path: "/imp/imp-1/deals/deal-a", adjust_deal: { bidfloor: 5.0 } }],
    };

    expect(() => normalize(raw, "grpc", {})).not.toThrow();
  });
});

describe("normalize() — error fallback stops", () => {
  it("still includes both yield placeholder stops when the orchestrator returns an RPC error", () => {
    const raw = { id: "req-4", error: { message: "boom" } };

    const result = normalize(raw, "grpc", {});
    const stopIds = result.stops.map((s) => s.id);

    expect(stopIds).toEqual([
      "ssp", "dlrm", "widedeep", "ncf", "metrics", "yield-floor", "yield-margin", "dsp",
    ]);
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
