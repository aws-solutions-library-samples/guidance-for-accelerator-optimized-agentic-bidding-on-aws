/**
 * @vitest-environment jsdom
 */
// FlowPipeline.stages.test.jsx -- each agent row carries the number of the
// sequential stage the orchestrator ran it in, taken from metadata.stages. A
// result without stages (bypassed baseline pass, pre-staging orchestrator) shows
// no badges rather than inventing an order.
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import FlowPipeline from "./FlowPipeline.jsx";
import { normalize } from "../utils/normalizer.js";

const payload = { lifecycle: "LIFECYCLE_SSP_BID_REQUEST", bid_request: { id: "r", imp: [{ id: "imp-1" }] } };

function rawResult(withStages) {
  return {
    id: "r",
    mutations: [],
    metadata: {
      total_latency_ms: 61,
      containers: [
        { name: "dlrm-bid-shader", status: "ok", latency_ms: 10, mutations: [] },
        { name: "widedeep-segment-activator", status: "ok", latency_ms: 8, mutations: [] },
        { name: "ncf-deal-manager", status: "ok", latency_ms: 12, mutations: [] },
        { name: "metrics-enricher", status: "ok", latency_ms: 9.5, mutations: [] },
        { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [] },
        { name: "yield-optimizer-margin", status: "ok", latency_ms: 19, mutations: [] },
      ],
      ...(withStages
        ? {
            stages: [
              { stage: 1, name: "enrich", containers: ["metrics-enricher", "widedeep-segment-activator"], latency_ms: 9.5, budget_ms: 90, applied: 0, rejected: [] },
              { stage: 2, name: "deals", containers: ["ncf-deal-manager"], latency_ms: 12, budget_ms: 80, applied: 0, rejected: [] },
              { stage: 3, name: "yield", containers: ["yield-optimizer-floor", "yield-optimizer-margin"], latency_ms: 21, budget_ms: 68, applied: 0, rejected: [] },
              { stage: 4, name: "price", containers: ["dlrm-bid-shader"], latency_ms: 10, budget_ms: 47, applied: 0, rejected: [] },
            ],
          }
        : {}),
    },
  };
}

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

describe("FlowPipeline stage badges", () => {
  it("shows each agent's stage number in run order", () => {
    const result = normalize(rawResult(true), "grpc", payload);
    render(<FlowPipeline result={result} loading={false} error={null} />);
    const badge = (id) => container.querySelector(`[data-testid="flow-stage-${id}"]`)?.textContent;
    expect(badge("metrics")).toBe("1");
    expect(badge("widedeep")).toBe("1");
    expect(badge("ncf")).toBe("2");
    expect(badge("yield-floor")).toBe("3");
    expect(badge("yield-margin")).toBe("3");
    expect(badge("dlrm")).toBe("4");
    // Endpoints are not staged.
    expect(container.querySelector('[data-testid="flow-stage-ssp"]')).toBeNull();
    expect(container.querySelector('[data-testid="flow-stage-dsp"]')).toBeNull();
    // And the latency widget reads the same stages.
    expect(container.querySelector('[data-testid="latency-agents-label"]').textContent)
      .toBe("Agents (4 sequential stages)");
  });

  it("shows no badges when the orchestrator reported no stages", () => {
    const result = normalize(rawResult(false), "grpc", payload);
    render(<FlowPipeline result={result} loading={false} error={null} />);
    expect(container.querySelectorAll(".agent-stage-badge")).toHaveLength(0);
    expect(container.querySelector('[data-testid="latency-agents-label"]').textContent)
      .toBe("Agents (parallel, max of 6)");
  });
});
