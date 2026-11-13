// reducedMotion.js — one reading of the reader's motion preference.
//
// Extracted from useTypewriter when a second hook needed it. Two copies of the
// jsdom fallback below would be two places to get the "a throw means no
// preference" decision wrong.

/**
 * @returns {boolean} true only when the reader has actively asked for reduced
 * motion. Absence of the API, and a failure to evaluate the query, both mean no
 * preference was expressed — which is not the same as asking for animation, but
 * is the case in which animating is allowed.
 */
export function prefersReducedMotion() {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    // Some jsdom configurations implement matchMedia without the media query
    // parser. A throw here means "no preference expressed", not "animate".
    return false;
  }
}
