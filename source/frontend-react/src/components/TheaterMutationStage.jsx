// TheaterMutationStage.jsx — the current mutation, over the centre of the stage.
//
// Replaces the bottom caption strip. The strip sat above the controls and both
// sat over the columns, so the two things a reader needed at once — what changed,
// and the data it changed — competed for the same pixels.
//
// This card states WHO changed the request and UNDER WHICH INTENT before it
// states what changed, because "the floor moved to $2.40" is an anonymous fact
// and the point of the walkthrough is that a named container decided it.
//
// The typed reveal is a clip over text that is already complete in the DOM (see
// useTypewriter), so nothing here is hidden from a screen reader or a test.
//
// The card sits over the request card it is describing, so it does not stay.
// It dwells for MUTATION_STAGE_DWELL_MS once the sentence is complete, then
// fades, uncovering the data the sentence was about. A reader who wants that
// data sooner dismisses it.

import { useCallback, useEffect, useState } from "react";
import { useTypewriter } from "../hooks/useTypewriter.js";

/**
 * How long the completed card stays before it begins to fade. Measured from the
 * end of the typed reveal, not from mount: a long narration takes over two
 * seconds to type, so dwelling from mount would start fading a sentence the
 * reader had not finished receiving.
 */
export const MUTATION_STAGE_DWELL_MS = 3000;

/** Length of the fade. Must match the opacity transition on .th-mstage-wrap. */
export const MUTATION_STAGE_FADE_MS = 240;

const PHASE_SHOWING = "showing";
const PHASE_LEAVING = "leaving";
const PHASE_GONE = "gone";

/**
 * @param narration  from mutationNarration(beat, context)
 * @param stepLabel  "Step 3 of 6" or "Preparing"
 * @param animate    false while the reader is scrubbing rather than watching
 */
export function TheaterMutationStage({ narration, stepLabel, animate = true }) {
  const text = narration?.text ?? "";
  const { shown, done } = useTypewriter(text, { enabled: animate });

  const [phase, setPhase] = useState(PHASE_SHOWING);

  // A new beat is a new card: the dwell restarts and a dismissal of the previous
  // step's card does not carry over to it. Keyed on the text for the same reason
  // useTypewriter keys its timer on it — the text is what identifies the beat
  // being narrated, and stepping back to a beat should show its card again.
  //
  // Reset during render rather than in an effect. An effect would commit one
  // render in which the beat is new but the phase is still the previous beat's
  // `gone`, so a card that had faded would flicker absent on the step after it.
  const [narratedText, setNarratedText] = useState(text);
  if (narratedText !== text) {
    setNarratedText(text);
    setPhase(PHASE_SHOWING);
  }

  useEffect(() => {
    if (phase !== PHASE_SHOWING || !text || !done) return undefined;
    const timer = setTimeout(() => setPhase(PHASE_LEAVING), MUTATION_STAGE_DWELL_MS);
    return () => clearTimeout(timer);
  }, [phase, text, done]);

  // Unmounted only after the fade has run, so the card is not cut from the screen
  // at full opacity.
  useEffect(() => {
    if (phase !== PHASE_LEAVING) return undefined;
    const timer = setTimeout(() => setPhase(PHASE_GONE), MUTATION_STAGE_FADE_MS);
    return () => clearTimeout(timer);
  }, [phase]);

  // Dismissal takes the same fade as the timer, rather than vanishing: the two
  // are the same event, reached from a click instead of from elapsed time.
  const dismiss = useCallback(() => setPhase(PHASE_LEAVING), []);

  if (!text || phase === PHASE_GONE) return null;

  const { containerLabel, intent, latencyMs, explored } = narration;

  return (
    <div
      className={`th-mstage-wrap${phase === PHASE_LEAVING ? " is-leaving" : ""}`}
      data-testid="theater-mutation-stage"
    >
      <div className="th-mstage">
        <div className="th-mstage-head">
          <span className="th-mstage-step">{stepLabel}</span>
          <span className="th-mstage-head-right">
            {containerLabel ? (
              <span className="th-mstage-attr">
                <span className="th-mstage-container" data-testid="mutation-container">
                  {containerLabel}
                </span>
                {intent ? (
                  <span className="th-mstage-intent" data-testid="mutation-intent">
                    {intent}
                  </span>
                ) : null}
              </span>
            ) : null}
            <button
              type="button"
              className="th-mstage-close"
              onClick={dismiss}
              aria-label="Dismiss this step"
              data-testid="theater-mutation-stage-close"
            >
              ×
            </button>
          </span>
        </div>

        {/*
          aria-live on the text, not on the card: the card's heading does not
          change often enough to be worth announcing, and announcing the whole
          card would re-read the step label on every beat.

          The full string is the element's text content. `shown` drives a span
          that is visually revealed; the remainder is present but transparent, so
          the card does not reflow as characters arrive.
        */}
        <p className="th-mstage-text" data-testid="mutation-text" aria-live="polite">
          <span className="th-mstage-typed">{shown}</span>
          <span className="th-mstage-pending" aria-hidden="true">{text.slice(shown.length)}</span>
        </p>

        {done && (latencyMs != null || explored) ? (
          <div className="th-mstage-foot">
            {latencyMs != null ? (
              <span className="th-mstage-latency">{latencyMs}ms</span>
            ) : null}
            {/* A real exploration arm, disclosed. Never inferred from the value. */}
            {explored ? (
              <span className="th-mstage-explore">exploration arm</span>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
