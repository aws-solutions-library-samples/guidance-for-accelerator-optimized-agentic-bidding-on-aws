// useBeatStepper.js — owns the index into a beat sequence.
//
// Rendering derives from (beats, index), so moving backwards produces exactly
// the state that moving forwards to the same index produces. That is the
// mockup's determinism guarantee, obtained without its reset-and-replay
// indirection (BR-17).

import { useState, useEffect, useRef, useCallback, useMemo } from "react";

/**
 * Autoplay dwell. Longer when attention crosses between columns, because there
 * is travel to watch (BR-19).
 */
export function paceFor(beat) {
  return beat && beat.movement && beat.movement !== "none" ? 3200 : 2600;
}

export function useBeatStepper({ beats, paceForBeat = paceFor } = {}) {
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef(null);

  const hasBeats = Array.isArray(beats) && beats.length > 0;
  // null rather than 0 while beats are absent: zero would be a claim about the
  // scenario, null is the absence of one. The total genuinely cannot be known
  // before the response, because a container may decline to mutate (BR-13).
  const total = hasBeats ? beats.length : null;
  const isIndeterminate = total === null;

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // A new sequence restarts the walkthrough rather than leaving a stale index
  // pointing into a different array.
  useEffect(() => {
    setIndex(0);
    setIsPlaying(false);
    clearTimer();
  }, [beats, clearTimer]);

  useEffect(() => clearTimer, [clearTimer]);

  const canGoBack = hasBeats && index > 0;
  const canGoNext = hasBeats && index < beats.length - 1;

  const next = useCallback(() => {
    setIndex((i) => (hasBeats && i < beats.length - 1 ? i + 1 : i));
  }, [beats, hasBeats]);

  const back = useCallback(() => {
    clearTimer();
    setIsPlaying(false);
    setIndex((i) => (i > 0 ? i - 1 : i));
  }, [clearTimer]);

  const restart = useCallback(() => {
    clearTimer();
    setIsPlaying(false);
    setIndex(0);
  }, [clearTimer]);

  const togglePlay = useCallback(() => {
    setIsPlaying((playing) => {
      if (playing) clearTimer();
      return !playing;
    });
  }, [clearTimer]);

  // Autoplay: schedule one advance at a time, keyed on the beat currently on
  // screen. Stops at the last beat (BR-15).
  useEffect(() => {
    if (!isPlaying || !hasBeats) return undefined;
    if (index >= beats.length - 1) {
      setIsPlaying(false);
      return undefined;
    }
    const dwell = paceForBeat(beats[index]);
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      setIndex((i) => (i < beats.length - 1 ? i + 1 : i));
    }, dwell);
    return clearTimer;
  }, [isPlaying, index, beats, hasBeats, paceForBeat, clearTimer]);

  const currentBeat = hasBeats ? beats[Math.min(index, beats.length - 1)] : null;

  return useMemo(() => ({
    index,
    total,
    isIndeterminate,
    isPlaying,
    currentBeat,
    canGoBack,
    canGoNext,
    next,
    back,
    restart,
    togglePlay,
  }), [index, total, isIndeterminate, isPlaying, currentBeat, canGoBack, canGoNext,
    next, back, restart, togglePlay]);
}
