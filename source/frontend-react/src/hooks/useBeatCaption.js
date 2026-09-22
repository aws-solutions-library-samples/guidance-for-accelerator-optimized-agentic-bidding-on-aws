// useBeatCaption.js — the caption for the current beat.
//
// NOT CURRENTLY MOUNTED. The Theater generated one caption per beat until the
// narration moved to the centre-stage card, which states the beat's real values
// derived rather than generated. Generated prose is now one summary per run, in
// useRunSummary.js. Nothing in the application calls this hook, so no per-beat
// model invocation happens; the module and its tests are kept because the
// per-beat path is intact and correct, and reinstating it is an import away.
//
// The only module in this unit that knows about time.
//
// Fallback-first: the factual caption is written synchronously the moment a beat
// becomes current, before any request is issued. Generation is a REPLACEMENT, not
// a fulfilment. There is therefore no error path here and no loading state,
// because there is no moment at which the UI lacks a correct caption.
//
// Epoch guard: every request is stamped with the beat index it was issued for and
// a result is applied only if that beat is still current. This is the load-bearing
// rule of the unit. Without it, prose describing a deal floor can land under a
// bid-shading beat -- every word of it real, and the whole of it a fabricated
// claim about what a container did. Abort is also issued on beat change, but
// cancellation is cooperative and a response already queued still resolves, so the
// guard is checked at the point of application and the abort is an optimisation
// layered on top of it.

import { useEffect, useRef, useState } from "react";
import { factualCaption } from "../utils/theaterCaptions.js";
import { generateCaption } from "../bedrockCaptionClient.js";

/** NFR3-3. Under autoplay's 2.6s floor, with headroom over the measured ~1.0s. */
export const CAPTION_DEADLINE_MS = 2500;

/**
 * @param {object}    args
 * @param {object}    args.beat      the current beat
 * @param {object}    args.context   ScenarioContext
 * @param {Function} [args.generate] injected for tests
 * @param {number}   [args.deadlineMs]
 * @returns {{text: string, beatIndex: number|null}}
 */
export function useBeatCaption({ beat, context, generate = generateCaption, deadlineMs = CAPTION_DEADLINE_MS }) {
  const beatIndex = beat?.index ?? null;

  const [state, setState] = useState(() => ({
    text: factualCaption(beat, context),
    beatIndex,
  }));

  // What the reader is looking at right now, readable from an async callback
  // without making it a dependency of the effect.
  const currentIndexRef = useRef(beatIndex);
  currentIndexRef.current = beatIndex;

  // The factual caption is written on every beat change, synchronously with
  // respect to the render that follows it. No intermediate empty state.
  useEffect(() => {
    setState({ text: factualCaption(beat, context), beatIndex });
  }, [beat, context, beatIndex]);

  useEffect(() => {
    if (!beat) return undefined;

    let live = true;
    const controller = new AbortController();

    // A plain timer rather than AbortSignal.timeout. The latter is driven by an
    // internal clock that fake timers cannot advance, so the deadline would be
    // untestable -- and one abort path is simpler than two.
    const deadlineTimer = setTimeout(() => controller.abort(), deadlineMs);

    const issuedFor = beatIndex;

    // Invoked synchronously with the effect, not deferred into a microtask. The
    // request should be in flight by the time the effect returns, so a beat change
    // on the very next render has something to abort.
    let pending;
    try {
      pending = generate({ beat, context, signal: controller.signal });
    } catch {
      pending = Promise.resolve({ ok: false, kind: "unknown", detail: "generator threw" });
    }

    Promise.resolve(pending)
      .then((outcome) => {
        if (!live) return;
        // The epoch guard. A late result for a beat the reader has left is
        // discarded even though it succeeded.
        if (currentIndexRef.current !== issuedFor) return;
        if (outcome?.ok && typeof outcome.text === "string" && outcome.text.length > 0) {
          setState({ text: outcome.text, beatIndex: issuedFor });
        }
        // Any non-ok outcome leaves the factual caption in place. Nothing to do.
      })
      .catch(() => {
        // generateCaption resolves for every outcome; this is belt and braces so
        // an unexpected throw still leaves a correct caption on screen.
      });

    return () => {
      live = false;
      clearTimeout(deadlineTimer);
      controller.abort();
    };
  }, [beat, context, beatIndex, generate, deadlineMs]);

  return state;
}
