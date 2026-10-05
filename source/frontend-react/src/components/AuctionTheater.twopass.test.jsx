/**
 * @vitest-environment jsdom
 *
 * The two-pass walkthrough: a banner opens each pass, pass 1 shows the baseline
 * auction and reveals its winner at the baseline beat, pass 2 shows the with-ARTF
 * auction and reveals its winner at the recap, and the recap puts the two side by
 * side -- the short set in the left column, the full set under the summary.
 *
 * The run hook is mocked with the shape buildBeats emits, so what is under test is
 * the component's derivation from (beats, index) and from the two responses. Same
 * mocking pattern as AuctionTheater.recap.test.jsx.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import {
  BEAT_PASS, BEAT_ORIGIN, BEAT_CONTAINER, BEAT_BIDS, BEAT_BASELINE, BEAT_RECAP,
  PASS_BASELINE, PASS_ARTF,
} from "../utils/theaterBeats.js";

vi.mock("../bedrockCaptionClient.js", () => ({
  generateRunSummary: vi.fn(() => new Promise(() => {})),
  generateCaption: vi.fn(() => new Promise(() => {})),
}));

const CONTEXT = {
  publisher: "example.com",
  impressionFormat: "video",
  bidFloor: 2.0,
  deals: [{ id: "deal-a", bidFloor: 2.0, auctionType: 1 }],
  userSignals: [],
};

const FLOOR_VALUE = {
  kind: "floor", dealId: "deal-a", before: 2.0, after: 2.6,
  containerName: "yield-optimizer-floor", displayLabel: "Yield Optimizer", intent: "ADJUST_DEAL_FLOOR",
};

// The sequence buildBeats emits for one mutation.
const BEATS = [
  { index: 0, kind: BEAT_PASS, pass: PASS_BASELINE, artf: false, movement: "none", values: [] },
  { index: 1, kind: BEAT_ORIGIN, pass: PASS_BASELINE, movement: "sell-to-request", values: [] },
  { index: 2, kind: BEAT_BIDS, pass: PASS_BASELINE, movement: "request-to-buy", values: [] },
  { index: 3, kind: BEAT_BASELINE, pass: PASS_BASELINE, movement: "none", values: [] },
  { index: 4, kind: BEAT_PASS, pass: PASS_ARTF, artf: true, movement: "none", values: [] },
  { index: 5, kind: BEAT_ORIGIN, pass: PASS_ARTF, movement: "sell-to-request", values: [] },
  {
    index: 6, kind: BEAT_CONTAINER, pass: PASS_ARTF, movement: "none",
    containerName: "yield-optimizer-floor", displayLabel: "Yield Optimizer",
    intent: "ADJUST_DEAL_FLOOR", latencyMs: 9, values: [FLOOR_VALUE],
  },
  { index: 7, kind: BEAT_BIDS, pass: PASS_ARTF, movement: "request-to-buy", values: [] },
  {
    index: 8, kind: BEAT_RECAP, pass: PASS_ARTF, movement: "none", values: [],
    contributors: [{ containerName: "yield-optimizer-floor", displayLabel: "Yield Optimizer", beatIndexes: [6] }],
  },
];

/** A live Prebid response with one marked winner at `price`. */
function liveResponse({ price, campaign, hop }) {
  return {
    id: "req-1",
    cur: "USD",
    seatbid: [{
      seat: "artfhouse",
      bid: [{
        id: "b1", impid: "imp-1", price, dealid: "deal-a",
        ext: { artf: { campaignId: "c1", campaignName: campaign }, prebid: { targeting: { hb_pb: String(price), hb_bidder: "artfhouse" } } },
      }],
    }],
    artf_meta: { source: "prebid", endpoint: "https://prebid/openrtb2/auction", hop_ms: hop },
  };
}

const BASELINE = liveResponse({ price: 3.1, campaign: "Baseline Co", hop: 40 });
const WITH_ARTF = liveResponse({ price: 4.2, campaign: "Cedar & Co", hop: 55 });

// Mutable so individual tests can take the baseline away.
const RUN = {
  status: "ready",
  beats: BEATS,
  context: CONTEXT,
  result: null,
  payload: {},
  bidResponse: WITH_ARTF,
  auctionFault: null,
  baselineResponse: BASELINE,
  baselineFault: null,
  error: null,
  start: vi.fn(() => Promise.resolve()),
};

vi.mock("../hooks/useTheaterRun.js", () => ({
  useTheaterRun: () => RUN,
  RUN_IDLE: "idle",
  RUN_SUBMITTING: "submitting",
  RUN_READY: "ready",
  RUN_FAILED: "failed",
  AUCTION_NOT_DEPLOYED: "not_deployed",
  AUCTION_FAILED: "failed",
}));

import AuctionTheater from "./AuctionTheater.jsx";

const SCENARIO = { id: "two-pass-probe", name: "Two-pass probe" };
let container;
let root;
const q = (testid) => container.querySelector(`[data-testid="${testid}"]`);
const text = (testid) => q(testid)?.textContent ?? "";

function clickNext(times = 1) {
  for (let i = 0; i < times; i++) {
    act(() => {
      q("theater-next").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  }
}

function render() {
  act(() => root.render(<AuctionTheater scenario={SCENARIO} params={{}} onExit={() => {}} />));
}

beforeEach(() => {
  RUN.baselineResponse = BASELINE;
  RUN.baselineFault = null;
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("AuctionTheater — two passes", () => {
  it("opens on the pass-1 banner, with no mutation stage over it", () => {
    render();
    expect(q("theater-pass-banner")).not.toBeNull();
    expect(q("theater-pass-banner").dataset.artf).toBe("without");
    expect(q("theater-mutation-stage")).toBeNull();
  });

  it("shows the baseline offers at pass 1's bids beat, winner withheld until the baseline beat", () => {
    render();
    clickNext(2); // origin, bids
    expect(q("offers-panel-pending")).toBeNull();
    expect(text("offers-panel")).toContain("Baseline Co");
    expect(q("offers-panel-pending-winner")).not.toBeNull();
    expect(q("offers-panel-winner")).toBeNull();

    clickNext(); // baseline
    expect(q("offers-panel-winner")).not.toBeNull();
    expect(text("offers-panel-winner")).toContain("3.10");
    // Nothing from the with-ARTF auction leaks into pass 1.
    expect(text("offers-panel")).not.toContain("Cedar & Co");
  });

  it("shows the pass-2 banner, then hides the offers again until pass 2's bids beat", () => {
    render();
    clickNext(4); // pass-2 banner
    expect(q("theater-pass-banner")).not.toBeNull();
    expect(q("theater-pass-banner").dataset.artf).toBe("with");

    clickNext(); // pass-2 origin: offers for THIS pass have not landed
    expect(q("offers-panel-pending")).not.toBeNull();
    expect(text("offers-panel")).not.toContain("Baseline Co");

    clickNext(2); // container, bids
    expect(text("offers-panel")).toContain("Cedar & Co");
    expect(q("offers-panel-winner")).toBeNull();
  });

  it("at the recap: with-ARTF winner, short comparison on the left, full comparison under the summary", () => {
    render();
    clickNext(8);
    expect(q("offers-panel-winner")).not.toBeNull();
    expect(text("offers-panel-winner")).toContain("4.20");

    // Left column: the short set, available, with the price delta.
    const left = q("sell-side-decisions-panel").querySelector('[data-testid="theater-comparison"]');
    expect(left).not.toBeNull();
    expect(left.dataset.available).toBe("true");
    expect(left.querySelector('[data-testid="theater-comparison-row-cleared"]').textContent).toContain("+$1.10");
    expect(left.querySelector('[data-testid="theater-comparison-row-roundtrip"]')).toBeNull();
    // The yield decision and the KPI block are still there beneath it.
    expect(text("sell-side-decisions-panel")).toContain("Floor");
    expect(q("sell-side-kpis")).not.toBeNull();

    // Summary modal: the full set.
    expect(q("theater-run-summary")).not.toBeNull();
    const full = q("theater-run-summary-comparison");
    expect(full).not.toBeNull();
    expect(full.querySelector('[data-testid="theater-comparison-row-roundtrip"]').textContent).toContain("+15 ms");
    expect(full.querySelector('[data-testid="theater-comparison-row-floor"]').textContent).toContain("$2.60 set by ARTF");
    // The prose names the baseline.
    expect(text("theater-run-summary-text")).toContain("Without ARTF");
    expect(text("theater-run-summary-text")).toContain("Baseline Co");
  });

  it("with no baseline, pass 1 states the fault and the comparison says the baseline is unavailable -- no fixture", () => {
    RUN.baselineResponse = null;
    RUN.baselineFault = { kind: "not_deployed", detail: "Prebid Server is not deployed." };
    render();
    clickNext(2);
    expect(q("offers-panel-fault")).not.toBeNull();
    expect(text("offers-panel-fault")).toContain("No baseline auction.");
    expect(q("offers-panel-no-response")).not.toBeNull();
    // The fixture campaigns never appear on the baseline pass.
    expect(text("offers-panel")).not.toContain("Cedar & Co");

    clickNext(6);
    const left = q("sell-side-decisions-panel").querySelector('[data-testid="theater-comparison"]');
    expect(left.dataset.available).toBe("false");
    expect(left.querySelector('[data-testid="theater-comparison-missing"]').textContent)
      .toContain("Prebid Server is not deployed.");
    expect(left.querySelector('[data-testid="theater-comparison-row-cleared"]').textContent).toContain("4.20");
    expect(text("theater-run-summary-text")).toContain("could not be compared");
  });

  it("stepping off a banner fades it out, and stepping back shows it again", () => {
    render();
    clickNext(5);
    // Off the pass-2 banner: it is either gone or on its way out, never shown.
    const leaving = q("theater-pass-banner");
    expect(leaving === null || leaving.className.includes("is-leaving")).toBe(true);
    act(() => {
      q("theater-back").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    const back = q("theater-pass-banner");
    expect(back).not.toBeNull();
    expect(back.className).not.toContain("is-leaving");
    expect(back.dataset.pass).toBe("2");
  });
});
