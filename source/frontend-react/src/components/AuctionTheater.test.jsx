/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

// The orchestrator client reaches for Cognito credentials at import time in some
// builds; stub the submit path so this stays a rendering test.
vi.mock("../hooks/useOrchestratorClientWithBase.js", () => ({
  useOrchestratorClientWithBase: () => ({
    submit: vi.fn(),
    loading: false,
    error: null,
    result: null,
    setResult: vi.fn(),
    cancel: vi.fn(),
  }),
}));

import AuctionTheater, { factualCaption } from "./AuctionTheater.jsx";
import { BEAT_RECAP } from "../utils/theaterBeats.js";

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("AuctionTheater", () => {
  it("mounts and offers a scenario to choose", () => {
    act(() => root.render(<AuctionTheater onExit={() => {}} />));
    expect(container.querySelector('[data-testid="theater-scenario-select"]')).not.toBeNull();
    expect(container.querySelectorAll(".th-scenario").length).toBeGreaterThan(0);
  });

  it("does not claim a step count before a scenario has been submitted", () => {
    act(() => root.render(<AuctionTheater onExit={() => {}} />));
    const label = container.querySelector('[data-testid="theater-progress-label"]');
    expect(label.textContent).toBe("Preparing");
    expect(container.textContent).not.toMatch(/Step \d+ of \d+/);
  });

  it("offers an exit", () => {
    const onExit = vi.fn();
    act(() => root.render(<AuctionTheater onExit={onExit} />));
    const exit = container.querySelector('[data-testid="theater-exit"]');
    expect(exit).not.toBeNull();
    act(() => exit.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onExit).toHaveBeenCalled();
  });
});

describe("factualCaption", () => {
  it("states real values without claiming how it was produced", () => {
    const beat = {
      kind: "container",
      displayLabel: "Signals Enricher",
      latencyMs: 4.2,
      explored: false,
      values: [
        { kind: "metric", type: "viewability", value: 0.82 },
        { kind: "metric", type: "brand_safety", value: 0.91 },
      ],
    };
    const text = factualCaption(beat, {});
    expect(text).toContain("Signals Enricher");
    expect(text).toContain("viewability 0.82");
    expect(text).toContain("brand safety 0.91");
    expect(text).toContain("4.2ms");
  });

  it("discloses a real exploration arm", () => {
    const beat = {
      kind: "container",
      displayLabel: "Yield Optimizer — Floor",
      explored: true,
      latencyMs: null,
      values: [{ kind: "floor", dealId: "deal-a", before: 1.5, after: 1.9 }],
    };
    expect(factualCaption(beat, {})).toMatch(/exploration arm/);
  });

  it("renders a margin according to its real calculation type", () => {
    const pct = { kind: "container", displayLabel: "Y", explored: false, latencyMs: null,
      values: [{ kind: "margin", dealId: "d", value: 0.14, calculationType: "PERCENT" }] };
    const cpm = { kind: "container", displayLabel: "Y", explored: false, latencyMs: null,
      values: [{ kind: "margin", dealId: "d", value: 0.5, calculationType: "CPM" }] };
    expect(factualCaption(pct, {})).toContain("14.0%");
    expect(factualCaption(cpm, {})).toContain("$0.50 CPM");
  });

  it("says so plainly when a mutation is not yet rendered", () => {
    const beat = { kind: "container", displayLabel: "X", intent: "ADD_CIDS",
      explored: false, latencyMs: null, values: [] };
    expect(factualCaption(beat, {})).toMatch(/does not yet render/);
  });

  it("reports an empty recap as no container having mutated", () => {
    expect(factualCaption({ kind: BEAT_RECAP, contributors: [] }, {}))
      .toBe("No container mutated this request.");
  });

  it("returns an empty string for a missing beat rather than throwing", () => {
    expect(factualCaption(null, {})).toBe("");
  });
});
