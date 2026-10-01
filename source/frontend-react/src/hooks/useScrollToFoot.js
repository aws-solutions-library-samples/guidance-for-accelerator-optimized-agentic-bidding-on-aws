// useScrollToFoot.js — keep the foot of a scrolling panel in view as it grows.
//
// The request card's contributed group appends each mutation to the end of a
// list inside a fixed-height, scrolling body. Once the list is taller than the
// body, a step's value lands below the fold: the walkthrough says a container
// changed the request and the change it describes is off screen.
//
// Keyed on a caller-supplied revision rather than run on every render, so a
// reader who scrolls up to re-read an earlier value stays where they put
// themselves until the next revision actually arrives.

import { useEffect } from "react";
import { prefersReducedMotion } from "../utils/reducedMotion.js";

/**
 * @param {{current: HTMLElement|null}} ref  the scrolling element
 * @param {unknown} revision  scrolls when this changes; anything comparable by ===
 */
export function useScrollToFoot(ref, revision) {
  useEffect(() => {
    const el = ref?.current;
    if (!el) return;

    // Nothing overflows, so there is no foot to travel to. Guarded rather than
    // left to the browser because a smooth scroll to the current position still
    // emits scroll events.
    if (el.scrollHeight <= el.clientHeight) return;

    const behavior = prefersReducedMotion() ? "auto" : "smooth";
    if (typeof el.scrollTo === "function") {
      el.scrollTo({ top: el.scrollHeight, behavior });
    } else {
      // jsdom and older engines. Assigning scrollTop is the same destination
      // without the animation.
      el.scrollTop = el.scrollHeight;
    }
  }, [ref, revision]);
}
