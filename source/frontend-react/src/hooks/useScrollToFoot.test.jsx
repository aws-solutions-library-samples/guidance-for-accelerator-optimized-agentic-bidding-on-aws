/**
 * @vitest-environment jsdom
 *
 * The request card's body is a fixed-height scroller and every mutation is
 * appended to its foot, so once the list outgrows the body a step's value lands
 * off screen. This hook follows it down.
 *
 * jsdom gives every element scrollHeight/clientHeight of 0 and does not implement
 * Element.prototype.scrollTo, so both are supplied here: the dimensions because
 * the hook's overflow guard reads them, and scrollTo because its absence is
 * precisely what selects the fallback branch.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { useScrollToFoot } from "./useScrollToFoot.js";

const OVERFLOWING = { scrollHeight: 500, clientHeight: 200 };
const FITS = { scrollHeight: 120, clientHeight: 200 };

/**
 * A minimal consumer: one scrolling div whose revision comes from a prop, which
 * is how TheaterRequestCard uses it.
 */
function Scroller({ revision, onRef }) {
  const ref = React.useRef(null);
  useScrollToFoot(ref, revision);
  React.useEffect(() => {
    onRef?.(ref.current);
  }, [onRef]);
  return <div ref={ref} data-testid="scroller" style={{ height: 200, overflowY: "auto" }} />;
}

let container;
let root;
let el;

/** Give the element a layout jsdom will not, and observe how it is scrolled. */
function stubLayout({ scrollHeight, clientHeight }) {
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight, configurable: true });
  Object.defineProperty(el, "clientHeight", { value: clientHeight, configurable: true });
}

function render(revision) {
  act(() => {
    root.render(<Scroller revision={revision} onRef={(node) => { el = node; }} />);
  });
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  render("0:0");
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("useScrollToFoot", () => {
  it("scrolls to the foot when the revision changes", () => {
    stubLayout(OVERFLOWING);
    render("1:0");
    expect(el.scrollTop).toBe(OVERFLOWING.scrollHeight);
  });

  it("prefers scrollTo, so a browser gets a smooth travel rather than a jump", () => {
    stubLayout(OVERFLOWING);
    const scrollTo = vi.fn();
    el.scrollTo = scrollTo;

    render("1:0");
    expect(scrollTo).toHaveBeenCalledWith({ top: OVERFLOWING.scrollHeight, behavior: "smooth" });
    // The fallback is not also taken — one scroll per revision, not two.
    expect(el.scrollTop).toBe(0);
  });

  it("leaves the reader alone when the revision has not changed", () => {
    stubLayout(OVERFLOWING);
    render("1:0");

    // The reader scrolls back up to re-read an earlier value, then something
    // re-renders that is not a new mutation. Being dragged back down here is the
    // behaviour the revision key exists to prevent.
    el.scrollTop = 40;
    render("1:0");
    expect(el.scrollTop).toBe(40);
  });

  it("does not scroll when the content fits", () => {
    stubLayout(FITS);
    const scrollTo = vi.fn();
    el.scrollTo = scrollTo;

    render("1:0");
    expect(scrollTo).not.toHaveBeenCalled();
    expect(el.scrollTop).toBe(0);
  });

  it("jumps rather than animates when the reader has asked for reduced motion", () => {
    const matchMedia = vi.fn(() => ({ matches: true, media: "", addListener() {}, removeListener() {} }));
    const previous = window.matchMedia;
    window.matchMedia = matchMedia;
    try {
      stubLayout(OVERFLOWING);
      const scrollTo = vi.fn();
      el.scrollTo = scrollTo;

      render("1:0");
      expect(scrollTo).toHaveBeenCalledWith({ top: OVERFLOWING.scrollHeight, behavior: "auto" });
    } finally {
      window.matchMedia = previous;
    }
  });
});
