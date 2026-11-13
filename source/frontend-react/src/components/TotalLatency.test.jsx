/**
 * @vitest-environment jsdom
 */
// TotalLatency.test.jsx -- agent time is the sum of the sequential stages'
// ceilings when the orchestrator reports stages, and the single parallel ceiling
// (max over stops) when it does not: the bypassed baseline pass of a Theater run,
// or an orchestrator that predates staging.
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import TotalLatency, { computeLatencyBreakdown } from "./TotalLatency.jsx";

const stops = [
  { id: "ssp", latency: null },
  { id: "dlrm", latency: { ms: 10 } },
  { id: "widedeep", latency: { ms: 8 } },
  { id: "ncf", latency: { ms: 12 } },
  { id: "metrics", latency: { ms: 9.5 } },
  { id: "yield-floor", latency: { ms: 21 } },
  { id: "yield-margin", latency: { ms: 19 } },
  { id: "dsp", latency: null },
];

const stages = [
  { stage: 1, name: "enrich", containers: ["metrics-enricher", "widedeep-segment-activator"], latencyMs: 9.5 },
  { stage: 2, name: "deals", containers: ["ncf-deal-manager"], latencyMs: 12 },
  { stage: 3, name: "yield", containers: ["yield-optimizer-floor", "yield-optimizer-margin"], latencyMs: 21 },
  { stage: 4, name: "price", containers: ["dlrm-bid-shader"], latencyMs: 10 },
];

describe("computeLatencyBreakdown()", () => {
  it("sums the per-stage ceilings when stages are reported", () => {
    const b = computeLatencyBreakdown(61, stops, stages);
    expect(b.mode).toBe("staged");
    // 9.5 + 12 + 21 + 10 = 52.5 -> 53; overhead 61 - 52.5 = 8.5 -> 9 (rounded separately)
    expect(b.agentMs).toBe(53);
    expect(b.orchestratorMs).toBe(9);
    expect(b.stageCount).toBe(4);
    expect(b.containerCount).toBe(6);
    expect(b.stages.map((s) => s.name)).toEqual(["enrich", "deals", "yield", "price"]);
  });

  it("falls back to the parallel ceiling (max over stops) when no stages are reported", () => {
    for (const noStages of [undefined, null, []]) {
      const b = computeLatencyBreakdown(45, stops, noStages);
      expect(b.mode).toBe("parallel");
      expect(b.agentMs).toBe(21);
      expect(b.orchestratorMs).toBe(24);
      expect(b.containerCount).toBe(6);
    }
  });

  it("never reports negative orchestrator overhead", () => {
    const b = computeLatencyBreakdown(40, stops, stages);
    expect(b.orchestratorMs).toBe(0);
  });

  it("ignores stage entries without a latency", () => {
    const b = computeLatencyBreakdown(30, stops, [{ stage: 1, name: "enrich", containers: [], latencyMs: null }, stages[1]]);
    expect(b.mode).toBe("staged");
    expect(b.stageCount).toBe(1);
    expect(b.agentMs).toBe(12);
  });

  it("returns null with no server latency", () => {
    expect(computeLatencyBreakdown(0, stops, stages)).toBeNull();
    expect(computeLatencyBreakdown(null, stops, stages)).toBeNull();
  });
});

describe("<TotalLatency />", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => { root.unmount(); });
    document.body.removeChild(container);
  });

  function render(jsx) {
    act(() => { root.render(jsx); });
  }

  it("labels the agent segment as sequential stages and shows one chip per stage", () => {
    render(<TotalLatency latencyMs={61} stops={stops} stages={stages} />);
    expect(container.querySelector('[data-testid="latency-agents-label"]').textContent)
      .toBe("Agents (4 sequential stages)");
    expect(container.querySelector('[data-testid="latency-agents-value"]').textContent).toBe("53ms");
    const chips = container.querySelectorAll(".latency-stage-chip");
    expect(chips).toHaveLength(4);
    expect(chips[0].textContent).toContain("1. enrich");
    expect(chips[3].textContent).toContain("4. price");
    expect(container.querySelector(".total-latency-breakdown").getAttribute("data-latency-mode")).toBe("staged");
  });

  it("keeps the parallel label and no stage strip on a bypassed pass", () => {
    render(<TotalLatency latencyMs={45} stops={stops} stages={[]} />);
    expect(container.querySelector('[data-testid="latency-agents-label"]').textContent)
      .toBe("Agents (parallel, max of 6)");
    expect(container.querySelector('[data-testid="latency-stage-strip"]')).toBeNull();
    expect(container.querySelector(".total-latency-breakdown").getAttribute("data-latency-mode")).toBe("parallel");
  });

  it("renders nothing without a server latency", () => {
    render(<TotalLatency latencyMs={0} stops={stops} stages={stages} />);
    expect(container.querySelector(".total-latency-breakdown")).toBeNull();
  });
});
