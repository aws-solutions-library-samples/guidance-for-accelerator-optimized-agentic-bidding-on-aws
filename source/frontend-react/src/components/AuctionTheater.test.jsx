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

// The Theater fetches its scenario payload on mount. jsdom has no fetch here, so
// it is stubbed to a never-resolving promise: these tests are about the chrome the
// Theater renders while a run is in flight, not about the run.
const fetchMock = vi.fn(() => new Promise(() => {}));
vi.stubGlobal("fetch", fetchMock);

import AuctionTheater, { factualCaption } from "./AuctionTheater.jsx";
import { BEAT_RECAP } from "../utils/theaterBeats.js";
import { SCENARIOS } from "./ScenarioCard.jsx";

const SCENARIO = SCENARIOS[0];

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
  // Block body, not a concise arrow: `mockClear()` returns the mock, and Vitest
  // treats a function returned from a hook as a TEARDOWN callback. It would then
  // invoke the fetch mock after every test and await its never-resolving promise,
  // which times out the hook — with the failure reported against the hook, not
  // against anything this file is actually asserting.
  beforeEach(() => {
    fetchMock.mockClear();
  });

  it("names the scenario it was given", () => {
    act(() => root.render(
      <AuctionTheater scenario={SCENARIO} params={{}} onExit={() => {}} />
    ));
    expect(
      container.querySelector('[data-testid="theater-scenario-name"]').textContent
    ).toBe(SCENARIO.name);
  });

  it("has no scenario chooser of its own", () => {
    // It is opened from a scenario card and always arrives with one. A second
    // place to pick a scenario would be a second place for the two to disagree
    // about which tuner values were used.
    act(() => root.render(
      <AuctionTheater scenario={SCENARIO} params={{}} onExit={() => {}} />
    ));
    expect(container.querySelector('[data-testid="theater-scenario-select"]')).toBeNull();
  });

  it("submits the scenario it was given on mount", () => {
    act(() => root.render(
      <AuctionTheater scenario={SCENARIO} params={{}} onExit={() => {}} />
    ));
    expect(fetchMock).toHaveBeenCalled();
    expect(String(fetchMock.mock.calls[0][0])).toContain(SCENARIO.file);
  });

  it("does not claim a step count before the run is ready", () => {
    act(() => root.render(
      <AuctionTheater scenario={SCENARIO} params={{}} onExit={() => {}} />
    ));
    const label = container.querySelector('[data-testid="theater-progress-label"]');
    expect(label.textContent).toBe("Preparing");
    expect(container.textContent).not.toMatch(/Step \d+ of \d+/);
  });

  it("offers a close control", () => {
    const onExit = vi.fn();
    act(() => root.render(
      <AuctionTheater scenario={SCENARIO} params={{}} onExit={onExit} />
    ));
    const exit = container.querySelector('[data-testid="theater-exit"]');
    expect(exit).not.toBeNull();
    act(() => exit.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(onExit).toHaveBeenCalled();
  });

  it("does not submit anything when given no scenario", () => {
    act(() => root.render(<AuctionTheater scenario={null} params={{}} onExit={() => {}} />));
    expect(fetchMock).not.toHaveBeenCalled();
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
