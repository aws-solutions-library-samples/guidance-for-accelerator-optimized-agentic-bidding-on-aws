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

// ---------------------------------------------------------------------------
// Store-defined containers (the ARTF template, or a user's own).
//
// Same class of bug as the yield stop this file was written for, one step
// further out: containerEntryToStop() used to `return null` for any name absent
// from CONTAINER_NAME_TO_STOP_ID, and buildExplicitContainerStops() then skipped
// it, so a seventh container was missing from the pipeline entirely. A
// store-defined container's name is not knowable at build time, so there is no
// map entry to add — the normalizer has to synthesize the stop and take the
// label from the response.
// ---------------------------------------------------------------------------

describe("normalize() — store-defined (dynamic) containers", () => {
  const SIX = [
    { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [] },
    { name: "widedeep-segment-activator", status: "ok", latency_ms: 8, mutations: [] },
    { name: "ncf-deal-manager", status: "ok", latency_ms: 12, mutations: [] },
    { name: "metrics-enricher", status: "ok", latency_ms: 5, mutations: [] },
    { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [] },
    { name: "yield-optimizer-margin", status: "ok", latency_ms: 19, mutations: [] },
  ];

  it("synthesizes a stop for a container it has no build-time entry for", () => {
    const raw = makeRawWithContainers([
      ...SIX,
      { name: "artf-template", display_name: "ARTF Template", status: "no_mutations", latency_ms: 3, mutations: [] },
    ]);

    const result = normalize(raw, "grpc", {});
    const stop = result.stops.find((s) => s.id === "dynamic:artf-template");

    expect(stop).toBeDefined();
    expect(stop.displayName).toBe("ARTF Template");
    expect(stop.modelFamily).toBe("CUSTOM");
    expect(stop.status).toBe("no_mutations");
    expect(stop.latency).toEqual({ ms: 3, source: "orchestrator" });
  });

  it("appends dynamic stops after the six fixed ones so no existing position moves", () => {
    const raw = makeRawWithContainers([
      { name: "artf-template", display_name: "ARTF Template", status: "ok", latency_ms: 3, mutations: [] },
      ...SIX,
    ]);

    const result = normalize(raw, "grpc", {});
    // The template was FIRST in the response and still lands last but for dsp:
    // the six keep their fixed slots regardless of response order.
    expect(result.stops.map((s) => s.id)).toEqual([
      "ssp", "dlrm", "widedeep", "ncf", "metrics", "yield-floor", "yield-margin",
      "dynamic:artf-template", "dsp",
    ]);
  });

  it("falls back to the internal name when no display name was sent", () => {
    const raw = makeRawWithContainers([
      { name: "my-container", status: "ok", latency_ms: 1, mutations: [] },
    ]);

    const stop = normalize(raw, "grpc", {}).stops.find((s) => s.id === "dynamic:my-container");
    // The raw name, never an invented label.
    expect(stop.displayName).toBe("my-container");
  });

  it("surfaces a dynamic container's mutations rather than discarding them", () => {
    const cidsMutation = {
      intent: 8, // ADD_CIDS
      op: 1, // ADD
      path: "/imp/imp-1/ext/cids",
      ids: { id: ["cid-1", "cid-2"] },
    };
    const raw = makeRawWithContainers([
      { name: "artf-template", display_name: "ARTF Template", status: "ok", latency_ms: 3, mutations: [cidsMutation] },
    ]);

    const stop = normalize(raw, "grpc", {}).stops.find((s) => s.id === "dynamic:artf-template");
    expect(stop.mutations).toHaveLength(1);
    expect(stop.mutations[0].intent).toBe("ADD_CIDS");
    expect(stop.mutations[0].payload).toEqual({ id: ["cid-1", "cid-2"] });
  });

  it("prefers the response's display name over the build-time table for a known container", () => {
    const raw = makeRawWithContainers([
      { name: "dlrm-bid-shader", display_name: "Renamed Pricer", status: "ok", latency_ms: 10, mutations: [] },
    ]);

    const stop = normalize(raw, "grpc", {}).stops.find((s) => s.id === "dlrm");
    expect(stop.displayName).toBe("Renamed Pricer");
  });

  it("still uses the build-time label when the orchestrator sends no display name", () => {
    // An orchestrator build predating display_name must not lose its labels.
    const raw = makeRawWithContainers([
      { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [] },
    ]);

    const stop = normalize(raw, "grpc", {}).stops.find((s) => s.id === "dlrm");
    expect(stop.displayName).toBe("Bid Pricer");
  });

  it("ignores a container entry with no usable name instead of synthesizing a stop", () => {
    const raw = makeRawWithContainers([
      { name: "", status: "ok", latency_ms: 1, mutations: [] },
      { status: "ok", latency_ms: 1, mutations: [] },
    ]);

    const result = normalize(raw, "grpc", {});
    expect(result.stops.filter((s) => s.id.startsWith("dynamic:"))).toHaveLength(0);
  });

  it("does not duplicate a dynamic stop when the same name appears twice", () => {
    const raw = makeRawWithContainers([
      { name: "artf-template", display_name: "ARTF Template", status: "ok", latency_ms: 3, mutations: [] },
      { name: "artf-template", display_name: "ARTF Template", status: "unreachable", latency_ms: 9, mutations: [] },
    ]);

    const dynamic = normalize(raw, "grpc", {}).stops.filter((s) => s.id.startsWith("dynamic:"));
    expect(dynamic).toHaveLength(1);
    // First entry wins, matching buildExplicitContainerStops' behaviour for the
    // six fixed stops.
    expect(dynamic[0].status).toBe("ok");
  });

  it("carries the new statuses through verbatim without mapping them onto a guess", () => {
    for (const status of ["disabled", "no_mutations", "unreachable", "error", "timeout", "skipped"]) {
      const raw = makeRawWithContainers([
        { name: "artf-template", status, latency_ms: 0, mutations: [] },
      ]);
      const stop = normalize(raw, "grpc", {}).stops.find((s) => s.id === "dynamic:artf-template");
      expect(stop.status).toBe(status);
    }
  });
});

// The orchestrator's flattened `mutations` list is the only account of the order
// the containers' mutations were applied in. The stops above are in DISPLAY order
// (dlrm first) which is the reverse of the staging order (dlrm runs last), so a
// consumer that walks stops to rebuild the request gets the wrong sequence. These
// tests pin the normalized `mutations` and `stages` fields that RawPanel and the
// latency widgets read instead.
describe("normalize() — ordered mutations and stages", () => {
  const activate = { intent: 2, op: 1, path: "/imp/imp-1/deals", ids: ["deal-home-premium"] };
  const floor = { intent: 4, op: 3, path: "/imp/imp-1/deals/deal-home-premium", adjust_deal: { bidfloor: 6.5 } };
  const shade = { intent: 6, op: 3, path: "/imp/imp-1/bidfloor", adjust_bid: { bidfloor: 4.1 } };

  function stagedRaw() {
    return {
      id: "req-5",
      // Application order: deals stage, then yield, then price. Display order of
      // the stops is dlrm, widedeep, ncf, metrics, yield-floor, yield-margin.
      mutations: [activate, floor, shade],
      metadata: {
        total_latency_ms: 61,
        containers: [
          { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [shade] },
          { name: "ncf-deal-manager", status: "ok", latency_ms: 12, mutations: [activate] },
          { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [floor] },
        ],
        stages: [
          { stage: 1, name: "enrich", containers: ["metrics-enricher", "widedeep-segment-activator"], latency_ms: 9.5, budget_ms: 90, applied: 0, rejected: [] },
          { stage: 2, name: "deals", containers: ["ncf-deal-manager"], latency_ms: 12, budget_ms: 80, applied: 1, rejected: [] },
          { stage: 3, name: "yield", containers: ["yield-optimizer-floor", "yield-optimizer-margin"], latency_ms: 21, budget_ms: 68, applied: 1,
            rejected: [{ intent: "ADJUST_DEAL_MARGIN", path: "/imp/imp-1/deals/deal-x", reason: "deal not present" }] },
          { stage: 4, name: "price", containers: ["dlrm-bid-shader"], latency_ms: 10, budget_ms: 47, applied: 1, rejected: [] },
        ],
      },
    };
  }

  it("keeps the orchestrator's application order, not the stops' display order", () => {
    const result = normalize(stagedRaw(), "grpc", {});
    expect(result.mutations.map((m) => m.intent)).toEqual([
      "ACTIVATE_DEALS", "ADJUST_DEAL_FLOOR", "BID_SHADE",
    ]);
  });

  it("stamps each ordered mutation with the stop that produced it", () => {
    const result = normalize(stagedRaw(), "grpc", {});
    expect(result.mutations.map((m) => m.sourceAgent)).toEqual(["ncf", "yield-floor", "dlrm"]);
    // The model fields RawPanel's applier reads are present on every entry.
    expect(result.mutations[0].path).toBe("/imp/imp-1/deals");
    expect(result.mutations[0].payload).toEqual(["deal-home-premium"]);
  });

  it("falls back to intent attribution when no stop claims the mutation", () => {
    const raw = stagedRaw();
    raw.metadata.containers = []; // older server: no per-container lists
    const result = normalize(raw, "grpc", {});
    expect(result.mutations.map((m) => m.sourceAgent)).toEqual(["ncf", "yield-floor", "dlrm"]);
  });

  it("attributes a mutation two containers both produced to the first in stop order", () => {
    const raw = stagedRaw();
    raw.metadata.containers.push(
      { name: "yield-optimizer-margin", status: "ok", latency_ms: 19, mutations: [floor] },
    );
    const result = normalize(raw, "grpc", {});
    // yield-floor is listed before yield-margin in stop order and wins the tie.
    expect(result.mutations[1].sourceAgent).toBe("yield-floor");
  });

  it("normalizes metadata.stages into camelCase with rejections preserved", () => {
    const result = normalize(stagedRaw(), "grpc", {});
    expect(result.stages).toHaveLength(4);
    expect(result.stages.map((s) => s.name)).toEqual(["enrich", "deals", "yield", "price"]);
    expect(result.stages[2]).toEqual({
      stage: 3,
      name: "yield",
      containers: ["yield-optimizer-floor", "yield-optimizer-margin"],
      latencyMs: 21,
      budgetMs: 68,
      applied: 1,
      rejected: [{ intent: "ADJUST_DEAL_MARGIN", path: "/imp/imp-1/deals/deal-x", reason: "deal not present" }],
    });
  });

  it("returns [] for stages on a bypassed pass or an older server, and [] mutations on an RPC error", () => {
    const bypassed = normalize({ id: "req-6", mutations: [], metadata: { total_latency_ms: 2, containers: [] } }, "grpc", {});
    expect(bypassed.stages).toEqual([]);
    expect(bypassed.mutations).toEqual([]);

    const errored = normalize({ id: "req-7", error: { message: "boom" } }, "grpc", {});
    expect(errored.stages).toEqual([]);
    expect(errored.mutations).toEqual([]);
  });

  it("drops malformed stage entries and tolerates missing numeric fields", () => {
    const raw = stagedRaw();
    raw.metadata.stages = [null, "nope", { name: "deals", containers: ["ncf-deal-manager", 7] }];
    const result = normalize(raw, "grpc", {});
    expect(result.stages).toEqual([{
      stage: 0, name: "deals", containers: ["ncf-deal-manager"],
      latencyMs: null, budgetMs: null, applied: 0, rejected: [],
    }]);
  });
});
