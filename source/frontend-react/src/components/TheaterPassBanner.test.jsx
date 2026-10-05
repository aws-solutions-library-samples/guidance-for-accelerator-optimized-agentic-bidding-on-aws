/**
 * @vitest-environment jsdom
 *
 * The pass banner fades in when the stepper lands on a pass beat and fades out
 * when it leaves, staying mounted for the length of the fade so it is never cut
 * from the screen at full opacity. With reduced motion both fades collapse to
 * nothing.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { TheaterPassBanner, PASS_BANNER_FADE_MS } from "./TheaterPassBanner.jsx";
import { BEAT_PASS, PASS_BASELINE, PASS_ARTF } from "../utils/theaterBeats.js";

/** Make the reader's motion preference whatever the test needs. */
function setReducedMotion(matches) {
  window.matchMedia = vi.fn(() => ({ matches, media: "", addListener() {}, removeListener() {} }));
}
let previousMatchMedia;

const PASS1 = { index: 0, kind: BEAT_PASS, pass: PASS_BASELINE, artf: false };
const PASS2 = { index: 4, kind: BEAT_PASS, pass: PASS_ARTF, artf: true };

let container;
let root;
const banner = () => container.querySelector('[data-testid="theater-pass-banner"]');

function mount(beat) {
  act(() => {
    root.render(<TheaterPassBanner beat={beat} />);
  });
}
function advance(ms) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  previousMatchMedia = window.matchMedia;
  setReducedMotion(false);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  window.matchMedia = previousMatchMedia;
  vi.useRealTimers();
});

describe("TheaterPassBanner", () => {
  it("renders nothing when the stepper is not on a pass beat", () => {
    mount(null);
    expect(banner()).toBeNull();
  });

  it("names the pass and whether ARTF is applied, in the words the narration uses", () => {
    mount(PASS1);
    expect(banner()).not.toBeNull();
    expect(banner().dataset.pass).toBe("1");
    expect(banner().dataset.artf).toBe("without");
    expect(banner().textContent).toContain("First pass");
    expect(banner().textContent).toContain("Without ARTF mutations");
    expect(banner().textContent).toContain("No container is consulted");

    mount(PASS2);
    expect(banner().dataset.pass).toBe("2");
    expect(banner().dataset.artf).toBe("with");
    expect(banner().textContent).toContain("Second pass");
    expect(banner().textContent).toContain("With ARTF mutations");
  });

  it("enters at zero opacity and becomes visible on the next frame", () => {
    mount(PASS1);
    expect(banner().className).not.toContain("is-visible");
    advance(16);
    expect(banner().className).toContain("is-visible");
    expect(banner().className).not.toContain("is-leaving");
  });

  it("stays mounted for the fade after the stepper leaves the beat, then goes", () => {
    mount(PASS1);
    advance(16);
    // The stepper moves to the next beat: the banner is told there is no pass beat.
    mount(null);
    expect(banner()).not.toBeNull();
    expect(banner().className).toContain("is-leaving");
    expect(banner().getAttribute("aria-hidden")).toBe("true");
    advance(PASS_BANNER_FADE_MS - 1);
    expect(banner()).not.toBeNull();
    advance(1);
    expect(banner()).toBeNull();
  });

  it("re-enters when the stepper comes back to a pass beat", () => {
    mount(PASS1);
    advance(16);
    mount(null);
    advance(PASS_BANNER_FADE_MS);
    expect(banner()).toBeNull();
    mount(PASS1);
    expect(banner()).not.toBeNull();
    expect(banner().dataset.pass).toBe("1");
  });

  it("coming back to the same beat mid-fade reverses the fade instead of finishing it", () => {
    mount(PASS2);
    advance(16);
    mount(null);
    expect(banner().className).toContain("is-leaving");
    advance(PASS_BANNER_FADE_MS / 2);
    // Back onto the same pass beat before the fade completed.
    mount(PASS2);
    expect(banner()).not.toBeNull();
    expect(banner().className).not.toContain("is-leaving");
    advance(16);
    expect(banner().className).toContain("is-visible");
    // The abandoned leave timer must not fire later and remove a shown banner.
    advance(PASS_BANNER_FADE_MS * 2);
    expect(banner()).not.toBeNull();
    expect(banner().className).toContain("is-visible");
  });

  it("with reduced motion, appears and disappears without a fade", () => {
    setReducedMotion(true);
    mount(PASS2);
    // Visible immediately: no entering frame.
    expect(banner().className).toContain("is-visible");
    mount(null);
    // Gone immediately: no leaving interval.
    expect(banner()).toBeNull();
  });
});
