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

import { useTypewriter } from "../hooks/useTypewriter.js";

/**
 * @param narration  from mutationNarration(beat, context)
 * @param stepLabel  "Step 3 of 6" or "Preparing"
 * @param animate    false while the reader is scrubbing rather than watching
 */
export function TheaterMutationStage({ narration, stepLabel, animate = true }) {
  const text = narration?.text ?? "";
  const { shown, done } = useTypewriter(text, { enabled: animate });

  if (!text) return null;

  const { containerLabel, intent, latencyMs, explored } = narration;

  return (
    <div className="th-mstage-wrap" data-testid="theater-mutation-stage">
      <div className="th-mstage">
        <div className="th-mstage-head">
          <span className="th-mstage-step">{stepLabel}</span>
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
