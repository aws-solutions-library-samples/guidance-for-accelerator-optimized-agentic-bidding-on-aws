import { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import AnimationEngine from './AnimationEngine.js';
import FocusAreaRegistry from './FocusAreaRegistry.js';

/**
 * React hook wrapping AnimationEngine for component lifecycle integration.
 *
 * Creates and memoizes a single AnimationEngine instance with a FocusAreaRegistry.
 * Provides reactive `isRunning` state that updates when the engine starts/stops.
 * Cleans up on unmount by calling engine.stop().
 *
 * @param {object} [options] - EngineOptions passed to AnimationEngine constructor
 * @param {number} [options.transitionDuration] - Default transition duration in seconds
 * @param {string} [options.easing] - GSAP easing function
 * @param {number} [options.annotationFadeMs] - Annotation fade duration in ms
 * @param {boolean} [options.autoScroll] - Whether to auto-scroll containers
 * @param {(text: string) => void} [options.onAnnotationShow] - Callback when annotation shows
 * @param {() => void} [options.onAnnotationHide] - Callback when annotation hides
 * @returns {{ isRunning: boolean, start: (sequence: AnimationStep[]) => Promise<void>, stop: () => Promise<void>, skipToNext: () => void, engine: AnimationEngine }}
 */
function useDemoAnimation(options = {}) {
  const [isRunning, setIsRunning] = useState(false);

  // Memoize the FocusAreaRegistry — created once per hook lifetime
  const registry = useMemo(() => new FocusAreaRegistry(), []);

  // Use a ref to hold the engine instance — created once, stable across renders
  const engineRef = useRef(null);

  if (!engineRef.current) {
    engineRef.current = new AnimationEngine(registry, options);
  }

  const engine = engineRef.current;

  /**
   * Start executing an animation sequence.
   * Updates isRunning state to true, then false when complete.
   * @param {Array<{ focusArea: string, zoomLevel: number, annotation: string|null, dwellMs: number, order: number }>} sequence
   * @returns {Promise<void>}
   */
  const start = useCallback(async (sequence) => {
    setIsRunning(true);
    try {
      await engine.execute(sequence);
    } finally {
      setIsRunning(false);
    }
  }, [engine]);

  /**
   * Stop the current animation sequence and restore UI.
   * @returns {Promise<void>}
   */
  const stop = useCallback(async () => {
    await engine.stop();
    setIsRunning(false);
  }, [engine]);

  /**
   * Skip to the next step in the current sequence.
   */
  const skipToNext = useCallback(() => {
    engine.skipToNext();
  }, [engine]);

  // Clean up on unmount — stop the engine to reverse transforms and release resources
  useEffect(() => {
    return () => {
      engineRef.current?.stop();
    };
  }, []);

  return { isRunning, start, stop, skipToNext, engine };
}

export default useDemoAnimation;
