/**
 * @vitest-environment jsdom
 *
 * The Deal Scorer is trained at deploy time on sample interactions generated from
 * the deal book. Every surface that shows its decisions says so: its theater
 * beats, the sell-side deal rows, and the scenario cards it scores. The registry
 * marks such a version with a "sample" badge.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import {
  DEAL_SCORER_TRAINING_DISCLOSURE,
  showsDealScorerDisclosure,
  scenarioUsesDealScorer,
} from "../utils/dealScorerTraining.js";
import { buildBeats, BEAT_CONTAINER } from "../utils/theaterBeats.js";
import { mutationNarration } from "../utils/theaterCaptions.js";
import { TheaterMutationStage } from "./TheaterMutationStage.jsx";
import { TheaterSellSidePanel } from "./TheaterPanels.jsx";
import ScenarioCard, { SCENARIOS, SURFACE_RESPONSE } from "./ScenarioCard.jsx";
import { ModelsView } from "./closedLoopUi.jsx";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const TEXT = "Deal Scorer trained on sample data";

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

function render(node) {
  act(() => { root.render(node); });
}

const q = (testid) => container.querySelector(`[data-testid="${testid}"]`);

const activate = {
  intent: "ACTIVATE_DEALS",
  op: "ADD",
  path: "/imp/imp-1/pmp/deals",
  payload: { id: ["deal-sports-premium-espn"] },
};
const metrics = {
  intent: "ADD_METRICS",
  op: "ADD",
  path: "/imp/imp-1/metric",
  payload: { metric: [{ type: "viewability", value: 0.8 }] },
};

function stop(id, mutations, extra = {}) {
  return { id, displayName: id === "ncf" ? "Deal Scorer" : id, status: "ok", latency: { ms: 4 }, mutations, ...extra };
}

function beatsFor(stops) {
  return buildBeats({ id: "r", bid_request: { imp: [{ id: "imp-1" }] } }, {
    stops: [stop("ssp", []), ...stops, stop("dsp", [])],
  });
}

describe("the disclosure text", () => {
  it("is one short line and says sample", () => {
    expect(DEAL_SCORER_TRAINING_DISCLOSURE).toBe(TEXT);
    expect(DEAL_SCORER_TRAINING_DISCLOSURE).toContain("sample");
  });
});

describe("showsDealScorerDisclosure", () => {
  it("is true on the Deal Scorer stop with no version or a sample build", () => {
    expect(showsDealScorerDisclosure("ncf", null)).toBe(true);
    expect(showsDealScorerDisclosure("ncf", "ncf-neumf-sample-9509e1edf027")).toBe(true);
  });

  it("is false for a Deal Scorer version without the sample prefix and for other stops", () => {
    expect(showsDealScorerDisclosure("ncf", "ncf-neumf-v1")).toBe(false);
    expect(showsDealScorerDisclosure("metrics", null)).toBe(false);
    expect(showsDealScorerDisclosure("yield-floor", "ncf-neumf-sample-9509e1edf027")).toBe(false);
  });

  it("names the request scenarios the Deal Scorer scores, and no response scenario", () => {
    for (const s of SCENARIOS) {
      const usesScorer = (s.models || []).some((m) => m.key === "ncf_deal_manager");
      expect(scenarioUsesDealScorer(s)).toBe(usesScorer);
      if (s.surface === SURFACE_RESPONSE) expect(scenarioUsesDealScorer(s)).toBe(false);
    }
  });
});

describe("theater beats", () => {
  it("flags the Deal Scorer beat and its values, not the others", () => {
    const beats = beatsFor([stop("ncf", [activate]), stop("metrics", [metrics])]);
    const containers = beats.filter((b) => b.kind === BEAT_CONTAINER);
    const scorer = containers.find((b) => b.containerName === "ncf");
    const enricher = containers.find((b) => b.containerName === "metrics");
    expect(scorer.sampleTrained).toBe(true);
    expect(scorer.values.every((v) => v.sampleTrained === true)).toBe(true);
    expect(enricher.sampleTrained).toBe(false);
    expect(beats.filter((b) => b.kind !== BEAT_CONTAINER).every((b) => b.sampleTrained === false)).toBe(true);
  });

  it("keeps the disclosure out of the narrated sentence", () => {
    const beat = beatsFor([stop("ncf", [activate])]).find((b) => b.kind === BEAT_CONTAINER);
    const narration = mutationNarration(beat, {});
    expect(narration.sampleTrained).toBe(true);
    expect(narration.text).not.toContain("trained on sample");
  });

  it("shows the disclosure on the Deal Scorer's mutation card", () => {
    const beat = beatsFor([stop("ncf", [activate])]).find((b) => b.kind === BEAT_CONTAINER);
    render(<TheaterMutationStage narration={mutationNarration(beat, {})} stepLabel="With ARTF" animate={false} />);
    expect(q("theater-deal-scorer-disclosure").textContent).toBe(TEXT);
  });

  it("does not show it on another container's card", () => {
    const beat = beatsFor([stop("metrics", [metrics])]).find((b) => b.kind === BEAT_CONTAINER);
    render(<TheaterMutationStage narration={mutationNarration(beat, {})} stepLabel="With ARTF" animate={false} />);
    expect(q("theater-mutation-stage")).not.toBeNull();
    expect(q("theater-deal-scorer-disclosure")).toBeNull();
  });

  it("shows it under the Deal Scorer's rows on the sell side", () => {
    const beat = beatsFor([stop("ncf", [activate])]).find((b) => b.kind === BEAT_CONTAINER);
    render(<TheaterSellSidePanel values={beat.values} revealed />);
    expect(container.textContent).toContain("Deals activated");
    expect(q("sell-side-deal-scorer-disclosure").textContent).toBe(TEXT);
  });

  it("leaves the sell side without it when no Deal Scorer value is on screen", () => {
    render(<TheaterSellSidePanel values={[]} revealed />);
    expect(q("sell-side-deal-scorer-disclosure")).toBeNull();
  });
});

describe("scenario cards", () => {
  const noop = () => {};

  it("discloses on every card the Deal Scorer scores", () => {
    for (const s of SCENARIOS.filter(scenarioUsesDealScorer)) {
      render(<ScenarioCard scenario={s} onSend={noop} onOpenTheater={noop} />);
      expect(q("scenario-deal-scorer-disclosure")?.textContent, s.id).toBe(TEXT);
    }
  });

  it("does not on a card it does not score", () => {
    const response = SCENARIOS.find((s) => !scenarioUsesDealScorer(s));
    render(<ScenarioCard scenario={response} onSend={noop} onOpenTheater={noop} />);
    expect(q("scenario-deal-scorer-disclosure")).toBeNull();
  });
});

describe("registry sample badge", () => {
  const base = { approval_status: "Approved", status: "Completed", created_at: "2026-10-07T10:00:00+00:00" };

  it("marks a version registered with training_data sample", () => {
    render(
      <ModelsView
        versions={[
          { ...base, version: 2, training_data: "sample", fingerprint: "9509e1edf027", description: TEXT },
          { ...base, version: 1, training_data: null, fingerprint: null },
        ]}
        modelLabel="Deal Scorer (NCF)"
      />,
    );
    const rows = container.querySelectorAll("tbody tr");
    const badge = rows[0].querySelector('[data-testid="model-version-sample-badge"]');
    expect(badge.textContent).toBe("sample");
    expect(badge.getAttribute("title")).toBe(TEXT);
    expect(rows[1].querySelector('[data-testid="model-version-sample-badge"]')).toBeNull();
  });
});
