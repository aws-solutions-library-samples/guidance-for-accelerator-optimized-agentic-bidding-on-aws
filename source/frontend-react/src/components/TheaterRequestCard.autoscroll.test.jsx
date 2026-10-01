/**
 * @vitest-environment jsdom
 *
 * The visual body of the Bid Request card follows its contributed values down.
 *
 * The walkthrough's whole claim is that a named container changed a named value.
 * Values append to the foot of a scrolling body, so past four or five mutations
 * the one the step is narrating is below the fold and the claim is unverifiable
 * on screen.
 *
 * Layout is stubbed because jsdom reports every dimension as 0, and scrollTo is
 * absent in jsdom, which is what selects the hook's scrollTop fallback.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { TheaterRequestCard, CARD_VIEW } from "./TheaterPanels.jsx";

const CONTEXT = {
  requestId: "req-1",
  impressionFormat: "video",
  geo: "US",
  bidFloor: 2.0,
  deals: [],
};

/** Distinct deal ids so each renders as its own row. */
const floorValue = (n) => ({
  kind: "floor",
  dealId: `deal-${n}`,
  before: 2.0,
  after: 2.0 + n / 10,
});

let container;
let root;

const body = () => container.querySelector('[data-testid="theater-card-body-visual"]');

function render(visible, extra = {}) {
  act(() => {
    root.render(
      <TheaterRequestCard
        context={CONTEXT}
        visible={visible}
        landingValues={visible.slice(-1)}
        cardState="idle"
        contributors={null}
        view={CARD_VIEW.VISUAL}
        {...extra}
      />,
    );
  });
}

function stubOverflow(el, scrollHeight) {
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight, configurable: true });
  Object.defineProperty(el, "clientHeight", { value: 200, configurable: true });
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("TheaterRequestCard — following the contributed values down", () => {
  it("scrolls the visual body to its foot when a value lands", () => {
    render([floorValue(1)]);
    stubOverflow(body(), 600);

    render([floorValue(1), floorValue(2)]);
    expect(body().scrollTop).toBe(600);
  });

  it("scrolls when the contributions block appears at the recap", () => {
    const values = [floorValue(1)];
    render(values);
    stubOverflow(body(), 720);

    // The value count is unchanged; what grew is the contributions block, which
    // is appended below the values on the recap beat.
    render(values, {
      contributors: [
        { containerName: "dlrm_bid_shader", displayLabel: "Bid Pricer", beatIndexes: [1] },
      ],
    });
    expect(body().scrollTop).toBe(720);
  });

  it("does not fight a reader who scrolled up between steps", () => {
    render([floorValue(1)]);
    const el = body();
    stubOverflow(el, 600);
    render([floorValue(1), floorValue(2)]);

    el.scrollTop = 120;
    // A re-render that lands no new value: the same step, re-rendered.
    render([floorValue(1), floorValue(2)]);
    expect(el.scrollTop).toBe(120);
  });
});
