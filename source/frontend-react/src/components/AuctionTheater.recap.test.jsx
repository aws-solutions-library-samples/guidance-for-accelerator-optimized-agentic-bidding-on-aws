/**
 * @vitest-environment jsdom
 *
 * The recap beat's surface is the run summary, not the mutation stage.
 *
 * The mutation stage names which container changed the request under which
 * intent. The recap beat has neither, so it rendered a bare step label over a
 * headline count the summary already states. Worse, it is the LAST beat: Next is
 * disabled there and the card carries no dismiss control, so dismissing the
 * summary left an opaque card centred over the request card with no way to clear
 * it short of Back or Restart.
 *
 * Separate file from AuctionTheater.test.jsx because that file stubs fetch to a
 * never-resolving promise to hold the run in flight; these tests need a settled
 * run at RUN_READY.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { BEAT_ORIGIN, BEAT_CONTAINER, BEAT_RECAP } from "../utils/theaterBeats.js";

// Generation is a replacement for the factual summary, never a prerequisite for
// it. Stubbed so the summary resolves synchronously and this stays a rendering
// test rather than a network one.
vi.mock("../bedrockCaptionClient.js", () => ({
  generateRunSummary: vi.fn(() => new Promise(() => {})),
  generateCaption: vi.fn(() => new Promise(() => {})),
}));

const CONTEXT = {
  publisher: "example.com",
  impressionFormat: "video",
  bidFloor: 2.0,
  deals: [],
};

// One origin, one container, one recap: index 2 is the recap, and index 1 is a
// real mutation beat, which is the control case.
const BEATS = [
  { kind: BEAT_ORIGIN, values: [], latencyMs: null, movement: "none" },
  {
    kind: BEAT_CONTAINER,
    displayLabel: "Bid Pricer",
    containerName: "dlrm_bid_shader",
    intent: "BID_SHADE",
    latencyMs: 12,
    movement: "none",
    values: [{ key: "floor", label: "Floor", value: "$2.40", path: "imp.0.bidfloor" }],
  },
  {
    kind: BEAT_RECAP,
    // Shape as buildBeats emits it: the recap's contributor count comes from
    // beatIndexes, so omitting it is a fixture that could not occur.
    contributors: [
      { containerName: "dlrm_bid_shader", displayLabel: "Bid Pricer", beatIndexes: [1] },
    ],
    values: [],
    movement: "none",
  },
];

const RUN = {
  status: "ready",
  beats: BEATS,
  context: CONTEXT,
  result: null,
  payload: {},
  bidResponse: null,
  auctionFault: null,
  error: null,
  start: vi.fn(() => Promise.resolve()),
};

vi.mock("../hooks/useTheaterRun.js", () => ({
  useTheaterRun: () => RUN,
  RUN_IDLE: "idle",
  RUN_SUBMITTING: "submitting",
  RUN_READY: "ready",
  RUN_FAILED: "failed",
}));

import AuctionTheater from "./AuctionTheater.jsx";

const SCENARIO = { id: "recap-probe", name: "Recap probe" };

let container;
let root;

const q = (testid) => container.querySelector(`[data-testid="${testid}"]`);

/** Advance the stepper by clicking the footer control, as a reader would. */
function clickNext() {
  act(() => {
    q("theater-next").dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(
    <AuctionTheater scenario={SCENARIO} params={{}} onExit={() => {}} />
  ));
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("AuctionTheater — the recap beat", () => {
  it("shows the mutation stage on a mutation beat", () => {
    // The control case. If this fails the suppression is too broad and the
    // walkthrough has lost the card that attributes each change.
    clickNext();
    expect(q("theater-mutation-stage")).not.toBeNull();
    expect(q("mutation-container").textContent).toBe("Bid Pricer");
  });

  it("opens the summary and withholds the mutation stage on arrival at the recap", () => {
    clickNext();
    clickNext();
    expect(q("theater-run-summary")).not.toBeNull();
    expect(q("theater-mutation-stage")).toBeNull();
  });

  it("does not reveal the mutation stage when the summary is dismissed at the recap", () => {
    clickNext();
    clickNext();
    act(() => {
      q("theater-run-summary-close").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    // The summary is gone, and nothing has taken its place over the stage. The
    // regression was the mutation stage appearing here, pinned at the last beat.
    expect(q("theater-run-summary")).toBeNull();
    expect(q("theater-mutation-stage")).toBeNull();
    // What the reader came for is now uncovered.
    expect(q("theater-request-card")).not.toBeNull();
  });

  it("restores the mutation stage on stepping back off the recap", () => {
    clickNext();
    clickNext();
    act(() => {
      q("theater-run-summary-close").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    act(() => {
      q("theater-back").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(q("theater-mutation-stage")).not.toBeNull();
  });
});
