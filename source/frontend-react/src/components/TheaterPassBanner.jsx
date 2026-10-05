// TheaterPassBanner.jsx — the banner that opens each pass of the walkthrough.
//
// The Theater runs the same bid request through Prebid twice: first with the
// ARTF extension point asked to propose nothing, then with the containers
// mutating the request. Each pass opens with this banner so a reader knows which
// of the two auctions they are about to watch. Without it the second origin beat
// would read as a replay.
//
// The banner is a BEAT, not a transition effect: it is on screen while the
// stepper's index is on a pass beat, and nowhere else. Stepping back to it shows
// it again, which keeps the screen a pure function of (beats, index). What this
// component adds is the fade: it enters over PASS_BANNER_FADE_MS, and when the
// stepper leaves the beat it stays mounted for the same interval at falling
// opacity before it goes, so the banner is never cut from the screen at full
// opacity.
//
// The words come from passBannerCopy(), which mutationNarration() also reads for
// the spoken form, so the two cannot disagree.

import { useEffect, useState } from "react";
import { passBannerCopy } from "../utils/theaterCaptions.js";
import { prefersReducedMotion } from "../utils/reducedMotion.js";

/** Length of each fade. Must match the opacity transition on .th-pass-banner-wrap. */
export const PASS_BANNER_FADE_MS = 240;

const PHASE_ENTERING = "entering";
const PHASE_SHOWN = "shown";
const PHASE_LEAVING = "leaving";
const PHASE_GONE = "gone";

/**
 * @param beat     the pass beat on screen, or null when the stepper is elsewhere
 */
export function TheaterPassBanner({ beat }) {
  const target = beat ?? null;
  const reduced = prefersReducedMotion();
  const fadeMs = reduced ? 0 : PASS_BANNER_FADE_MS;

  // The beat the banner is currently showing (or fading out). Kept separately
  // from `beat` so that when the stepper moves on, the banner still knows what to
  // display for the length of the fade.
  const [shown, setShown] = useState(target);
  const [phase, setPhase] = useState(target ? (reduced ? PHASE_SHOWN : PHASE_ENTERING) : PHASE_GONE);

  // Arrivals and departures, decided during render so no frame commits a stale
  // phase (the same reasoning as TheaterMutationStage's reset).
  if (target && target !== shown) {
    setShown(target);
    setPhase(reduced ? PHASE_SHOWN : PHASE_ENTERING);
  } else if (target && (phase === PHASE_LEAVING || phase === PHASE_GONE)) {
    // The stepper came back to the beat the banner was still fading out from.
    // Same beat, so `shown` is already right; the fade is simply reversed.
    setPhase(reduced ? PHASE_SHOWN : PHASE_ENTERING);
  } else if (!target && shown && phase !== PHASE_LEAVING && phase !== PHASE_GONE) {
    setPhase(reduced ? PHASE_GONE : PHASE_LEAVING);
  }

  useEffect(() => {
    if (phase !== PHASE_ENTERING) return undefined;
    // One frame at the start opacity, then the transition runs.
    const timer = setTimeout(() => setPhase(PHASE_SHOWN), 16);
    return () => clearTimeout(timer);
  }, [phase]);

  useEffect(() => {
    if (phase !== PHASE_LEAVING) return undefined;
    const timer = setTimeout(() => {
      setPhase(PHASE_GONE);
      setShown(null);
    }, fadeMs);
    return () => clearTimeout(timer);
  }, [phase, fadeMs]);

  if (!shown || phase === PHASE_GONE) return null;

  const copy = passBannerCopy(shown);
  const visible = phase === PHASE_SHOWN;

  return (
    <div
      className={`th-pass-banner-wrap${visible ? " is-visible" : ""}${phase === PHASE_LEAVING ? " is-leaving" : ""}`}
      data-testid="theater-pass-banner"
      data-pass={shown.pass}
      data-artf={shown.artf === true ? "with" : "without"}
      data-phase={phase}
      aria-hidden={phase === PHASE_LEAVING ? "true" : undefined}
    >
      <div className="th-pass-banner" role="status" aria-live="polite">
        <span className="th-pass-banner-step">{copy.title}</span>
        <h2 className="th-pass-banner-headline">{copy.headline}</h2>
        <p className="th-pass-banner-body">{copy.body}</p>
      </div>
    </div>
  );
}
