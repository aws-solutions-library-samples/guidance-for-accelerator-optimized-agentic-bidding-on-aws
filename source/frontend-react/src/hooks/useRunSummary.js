// useRunSummary.js — one generated summary per run, at the end of the run.
//
// Replaces useBeatCaption's per-beat generation. The reason is editorial rather
// than technical: a sentence under every step competed with the step's own data
// for attention, and six generated sentences about one impression say less than
// one that names the winner and why it won.
//
// Fallback-first, exactly as the per-beat hook was: the factual summary is set
// synchronously the moment the run settles, so the surface never has a loading
// state and never has nothing. Generation is a REPLACEMENT.
//
// Fires once per run. The guard is the run's own identity (the beats array, which
// is a fresh array per run), so a re-render cannot re-invoke and a second run
// cannot reuse the first run's prose.

import { useEffect, useRef, useState } from "react";
import { factualRunSummary } from "../utils/runSummaryFallback.js";
import { generateRunSummary } from "../bedrockCaptionClient.js";

/** Generous next to the per-beat 2.5s: this runs once and nothing waits on it. */
export const SUMMARY_DEADLINE_MS = 12000;

/**
 * @param {object}    args
 * @param {object[]}  args.beats
 * @param {object}    args.context   ScenarioContext
 * @param {object}    args.viewModel offers view model
 * @param {boolean}   args.ready     the run has settled, including its auction
 * @param {Function} [args.generate] injected for tests
 * @param {number}   [args.deadlineMs]
 * @returns {{text: string, generated: boolean}}
 */
export function useRunSummary({
  beats,
  context,
  viewModel,
  ready,
  generate = generateRunSummary,
  deadlineMs = SUMMARY_DEADLINE_MS,
}) {
  const [state, setState] = useState({ text: "", generated: false });

  // The run this hook has already asked about. Compared by identity: buildBeats
  // returns a new array per run, so a new run is a new object and a re-render is
  // not.
  const askedForRef = useRef(null);
  const liveRef = useRef(true);

  // The factual summary, written as soon as there is something to summarise.
  //
  // It must not overwrite a summary that has already been generated for THIS run.
  // The effect depends on `viewModel`, which is memoised on the bid response — so
  // anything that produces a new view-model object for the same run (a re-memo, a
  // second auction read) would otherwise silently replace generated prose with the
  // fallback, and the surface would appear to regress for no visible reason.
  useEffect(() => {
    if (!beats) {
      setState({ text: "", generated: false });
      return;
    }
    setState((prev) => (
      prev.generated && askedForRef.current === beats
        ? prev
        : { text: factualRunSummary(beats, context, viewModel), generated: false }
    ));
  }, [beats, context, viewModel]);

  useEffect(() => {
    if (!ready || !beats) return undefined;
    if (askedForRef.current === beats) return undefined;
    askedForRef.current = beats;

    liveRef.current = true;
    const controller = new AbortController();
    // A plain timer rather than AbortSignal.timeout, so fake timers can advance
    // it and the deadline is testable.
    const deadline = setTimeout(() => controller.abort(), deadlineMs);

    let pending;
    try {
      pending = generate({ beats, context, viewModel, signal: controller.signal });
    } catch {
      pending = Promise.resolve({ ok: false, kind: "unknown", detail: "generator threw" });
    }

    Promise.resolve(pending)
      .then((outcome) => {
        if (!liveRef.current) return;
        // Still the same run? A slow answer for a run the reader has left is
        // discarded even though it succeeded.
        if (askedForRef.current !== beats) return;
        if (outcome?.ok && typeof outcome.text === "string" && outcome.text.length > 0) {
          setState({ text: outcome.text, generated: true });
        }
        // Any non-ok outcome leaves the factual summary in place.
      })
      .catch(() => {
        // generateRunSummary resolves for every outcome; belt and braces so an
        // unexpected throw still leaves a correct summary on screen.
      });

    return () => {
      liveRef.current = false;
      clearTimeout(deadline);
      controller.abort();
    };
  }, [ready, beats, context, viewModel, generate, deadlineMs]);

  return state;
}
