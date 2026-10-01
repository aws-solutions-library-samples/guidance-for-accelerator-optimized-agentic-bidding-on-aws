// useTypewriter.js — reveal a string progressively.
//
// Two rules make this safe rather than decorative:
//
//   1. The FULL text is always returned as `full`, and the caller renders it in
//      the DOM. Only a visual clip is animated. A partially typed string is
//      never the only copy on screen, so assistive technology, a text-content
//      assertion and a user who reads faster than the animation all see the
//      whole sentence.
//   2. The timer is keyed on the text. A beat change restarts it, so a timer
//      scheduled for one mutation can never append characters to another's.
//
// Under `prefers-reduced-motion: reduce` it returns the complete string
// immediately and schedules nothing (NFR-3).

import { useEffect, useRef, useState } from "react";
import { prefersReducedMotion } from "../utils/reducedMotion.js";

/** Per-character dwell. 18ms is ~55 chars/second: readable, not sluggish. */
export const TYPE_INTERVAL_MS = 18;

/**
 * @param {string} text
 * @param {object} [opts]
 * @param {number} [opts.intervalMs]
 * @param {boolean} [opts.enabled] false holds the animation at the full string
 * @returns {{ shown: string, full: string, done: boolean }}
 */
export function useTypewriter(text, { intervalMs = TYPE_INTERVAL_MS, enabled = true } = {}) {
  const full = typeof text === "string" ? text : "";
  const reduced = prefersReducedMotion();
  const animate = enabled && !reduced && full.length > 0;

  const [count, setCount] = useState(() => (animate ? 0 : full.length));
  const timerRef = useRef(null);

  useEffect(() => {
    const clear = () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
    clear();

    if (!animate) {
      setCount(full.length);
      return clear;
    }

    setCount(0);
    timerRef.current = setInterval(() => {
      setCount((n) => {
        if (n >= full.length) {
          clear();
          return full.length;
        }
        return n + 1;
      });
    }, intervalMs);

    return clear;
  }, [full, animate, intervalMs]);

  const shown = animate ? full.slice(0, Math.min(count, full.length)) : full;
  return { shown, full, done: shown.length >= full.length };
}
