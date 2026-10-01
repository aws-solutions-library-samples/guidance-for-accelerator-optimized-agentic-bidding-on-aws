/**
 * @vitest-environment jsdom
 *
 * The mutation card does not stay on screen.
 *
 * It is centred over the request card it is describing, so leaving it up means
 * the reader cannot see the values the sentence just named. It dwells for three
 * seconds after the sentence is complete and then fades, and a reader who wants
 * the data sooner dismisses it.
 *
 * The dwell is measured from the END of the typed reveal rather than from mount.
 * At 18ms per character a long narration takes over three seconds to type, so a
 * dwell from mount would begin fading a sentence still arriving.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import {
  TheaterMutationStage,
  MUTATION_STAGE_DWELL_MS,
  MUTATION_STAGE_FADE_MS,
} from "./TheaterMutationStage.jsx";
import { TYPE_INTERVAL_MS } from "../hooks/useTypewriter.js";

const NARRATION = {
  text: "Bid Pricer raised the floor to $2.40.",
  containerLabel: "Bid Pricer",
  intent: "BID_SHADE",
  latencyMs: 12,
  explored: false,
};

let container;
let root;

const q = (testid) => container.querySelector(`[data-testid="${testid}"]`);
const card = () => q("theater-mutation-stage");

/** Render with the typed reveal off, so `done` is true on the first frame. */
function mount(props = {}) {
  act(() => {
    root.render(
      <TheaterMutationStage
        narration={NARRATION}
        stepLabel="Step 2 of 3"
        animate={false}
        {...props}
      />,
    );
  });
}

function advance(ms) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
});

describe("TheaterMutationStage — dwell and dismissal", () => {
  it("shows the narration and a dismiss control", () => {
    mount();
    expect(card()).not.toBeNull();
    expect(q("mutation-text").textContent).toBe(NARRATION.text);
    expect(q("theater-mutation-stage-close")).not.toBeNull();
  });

  it("holds for the full dwell before it begins to fade", () => {
    mount();
    advance(MUTATION_STAGE_DWELL_MS - 1);
    // Still fully opaque: a reader gets the whole three seconds, not most of it.
    expect(card().className).not.toContain("is-leaving");
  });

  it("fades and then unmounts once the dwell has elapsed", () => {
    mount();
    advance(MUTATION_STAGE_DWELL_MS);
    expect(card().className).toContain("is-leaving");
    // Present throughout the fade — removing it at full opacity would cut it from
    // the screen rather than fade it.
    advance(MUTATION_STAGE_FADE_MS - 1);
    expect(card()).not.toBeNull();
    advance(1);
    expect(card()).toBeNull();
  });

  it("does not start the dwell while the sentence is still being typed", () => {
    // Long enough that typing outlasts the dwell: the regression this guards is a
    // card that fades while the reader is still receiving the sentence.
    const text = "x".repeat(Math.ceil(MUTATION_STAGE_DWELL_MS / TYPE_INTERVAL_MS) + 40);
    mount({ narration: { ...NARRATION, text }, animate: true });

    advance(MUTATION_STAGE_DWELL_MS);
    expect(card()).not.toBeNull();
    expect(card().className).not.toContain("is-leaving");

    // Finish typing, then the dwell runs from there.
    advance(text.length * TYPE_INTERVAL_MS);
    expect(card().className).not.toContain("is-leaving");
    advance(MUTATION_STAGE_DWELL_MS);
    expect(card().className).toContain("is-leaving");
  });

  it("fades on dismissal without waiting for the dwell", () => {
    mount();
    act(() => {
      q("theater-mutation-stage-close").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(card().className).toContain("is-leaving");
    advance(MUTATION_STAGE_FADE_MS);
    expect(card()).toBeNull();
  });

  it("shows a fresh card for the next beat after the previous one has gone", () => {
    mount();
    // Two advances, not one: the fade timer is scheduled by the effect that runs
    // after the dwell's state change commits, so it does not exist yet while the
    // dwell timeout is being drained.
    advance(MUTATION_STAGE_DWELL_MS);
    advance(MUTATION_STAGE_FADE_MS);
    expect(card()).toBeNull();

    // A new beat carries new narration. Dismissal of the previous step must not
    // suppress it, and it must appear without a frame of absence first.
    mount({ narration: { ...NARRATION, text: "Segment Activator added 3 segments." } });
    expect(card()).not.toBeNull();
    expect(card().className).not.toContain("is-leaving");
    expect(q("mutation-text").textContent).toBe("Segment Activator added 3 segments.");
  });
});
