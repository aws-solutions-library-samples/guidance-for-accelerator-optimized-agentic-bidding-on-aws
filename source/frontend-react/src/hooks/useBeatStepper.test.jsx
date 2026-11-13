/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { useBeatStepper, paceFor } from "./useBeatStepper.js";

let container;
let root;
let latest;

function Probe({ beats }) {
  latest = useBeatStepper({ beats });
  return null;
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  latest = null;
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
});

function render(beats) {
  act(() => root.render(<Probe beats={beats} />));
}

const beat = (index, movement = "none") => ({ index, movement, kind: "container", values: [] });
const THREE = [beat(0, "sell-to-request"), beat(1), beat(2, "request-to-buy")];

describe("useBeatStepper without beats", () => {
  it("reports the total as unknown rather than zero", () => {
    render(null);
    expect(latest.total).toBeNull();
    expect(latest.isIndeterminate).toBe(true);
  });

  it("offers no navigation", () => {
    render(null);
    expect(latest.canGoBack).toBe(false);
    expect(latest.canGoNext).toBe(false);
    expect(latest.currentBeat).toBeNull();
  });

  it("treats an empty array as still unknown, not as a total of zero", () => {
    render([]);
    expect(latest.total).toBeNull();
    expect(latest.isIndeterminate).toBe(true);
  });
});

describe("useBeatStepper with beats", () => {
  it("knows the total once beats exist", () => {
    render(THREE);
    expect(latest.total).toBe(3);
    expect(latest.isIndeterminate).toBe(false);
    expect(latest.index).toBe(0);
  });

  it("bounds the index at both ends", () => {
    render(THREE);
    expect(latest.canGoBack).toBe(false);
    act(() => latest.back());
    expect(latest.index).toBe(0);

    act(() => latest.next());
    act(() => latest.next());
    expect(latest.index).toBe(2);
    expect(latest.canGoNext).toBe(false);
    act(() => latest.next());
    expect(latest.index).toBe(2);
  });

  it("back then next returns to the same index", () => {
    render(THREE);
    act(() => latest.next());
    act(() => latest.next());
    const forward = latest.index;
    act(() => latest.back());
    act(() => latest.next());
    expect(latest.index).toBe(forward);
  });

  it("restart returns to the first beat and stops playing", () => {
    render(THREE);
    act(() => latest.next());
    act(() => latest.togglePlay());
    act(() => latest.restart());
    expect(latest.index).toBe(0);
    expect(latest.isPlaying).toBe(false);
  });

  it("back stops autoplay", () => {
    render(THREE);
    act(() => latest.next());
    act(() => latest.togglePlay());
    expect(latest.isPlaying).toBe(true);
    act(() => latest.back());
    expect(latest.isPlaying).toBe(false);
  });

  it("exposes the beat at the current index", () => {
    render(THREE);
    expect(latest.currentBeat.index).toBe(0);
    act(() => latest.next());
    expect(latest.currentBeat.index).toBe(1);
  });

  it("restarts when a new beat sequence arrives", () => {
    render(THREE);
    act(() => latest.next());
    act(() => latest.next());
    expect(latest.index).toBe(2);
    render([beat(0), beat(1)]);
    expect(latest.index).toBe(0);
    expect(latest.total).toBe(2);
  });
});

describe("useBeatStepper autoplay", () => {
  it("advances on a timer and stops at the last beat", () => {
    vi.useFakeTimers();
    render(THREE);
    act(() => latest.togglePlay());
    expect(latest.isPlaying).toBe(true);

    act(() => { vi.advanceTimersByTime(4000); });
    expect(latest.index).toBe(1);

    act(() => { vi.advanceTimersByTime(4000); });
    expect(latest.index).toBe(2);

    act(() => { vi.advanceTimersByTime(4000); });
    expect(latest.index).toBe(2);
    expect(latest.isPlaying).toBe(false);
  });
});

describe("paceFor", () => {
  it("dwells longer when attention crosses between columns", () => {
    expect(paceFor({ movement: "sell-to-request" })).toBeGreaterThan(paceFor({ movement: "none" }));
    expect(paceFor({ movement: "request-to-buy" })).toBeGreaterThan(paceFor({ movement: "none" }));
  });

  it("is defined for a missing beat", () => {
    expect(typeof paceFor(null)).toBe("number");
  });
});
