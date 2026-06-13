/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import AnimationEngine from './AnimationEngine.js';
import FocusAreaRegistry from './FocusAreaRegistry.js';

// Mock GSAP — we don't want real animations in unit tests.
// The mock calls onComplete synchronously to avoid timing issues.
vi.mock('./gsapSetup.js', () => ({
  gsap: {
    to: vi.fn((target, vars) => {
      if (vars.onComplete) {
        vars.onComplete();
      }
      return { kill: vi.fn() };
    }),
  },
}));

/**
 * Helper: run engine.execute() with fake timers, advancing time for each dwell.
 * Since GSAP is mocked to complete instantly, only dwells use real timers.
 */
async function executeWithTimers(engine, sequence) {
  const promise = engine.execute(sequence);
  // Flush all pending timers (dwells) repeatedly until the promise resolves
  await flushDwells(promise);
  return promise;
}

/**
 * Flush microtasks and advance timers until the given promise resolves.
 */
async function flushDwells(promise, maxIterations = 50) {
  let resolved = false;
  promise.then(() => { resolved = true; });

  for (let i = 0; i < maxIterations && !resolved; i++) {
    // Advance timers by a large chunk to cover any dwell
    vi.advanceTimersByTime(100);
    // Flush microtasks
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  }
}

describe('AnimationEngine', () => {
  let registry;
  let engine;
  let mockElement;
  let mockContainer;

  beforeEach(() => {
    vi.useFakeTimers();

    registry = new FocusAreaRegistry();

    // Set up DOM elements for testing
    mockContainer = document.createElement('div');
    mockContainer.id = 'flow-canvas';
    document.body.appendChild(mockContainer);

    mockElement = document.createElement('div');
    mockElement.className = 'timeline-container';
    mockContainer.appendChild(mockElement);

    engine = new AnimationEngine(registry);
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  describe('constructor', () => {
    it('initializes with default options', () => {
      const eng = new AnimationEngine(registry);
      expect(eng.isRunning).toBe(false);
    });

    it('accepts custom options', () => {
      const onShow = vi.fn();
      const onHide = vi.fn();
      const eng = new AnimationEngine(registry, {
        transitionDuration: 1.0,
        easing: 'power3.out',
        annotationFadeMs: 500,
        autoScroll: false,
        onAnnotationShow: onShow,
        onAnnotationHide: onHide,
      });
      expect(eng.isRunning).toBe(false);
    });
  });

  describe('isRunning', () => {
    it('returns false when not executing', () => {
      expect(engine.isRunning).toBe(false);
    });

    it('returns true during execution and false after', async () => {
      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 500, order: 0 },
      ];

      const executePromise = engine.execute(sequence);

      // Engine should be running now (GSAP completes instantly, but dwell is pending)
      expect(engine.isRunning).toBe(true);

      // Advance past dwell and flush
      await flushDwells(executePromise);

      expect(engine.isRunning).toBe(false);
    });
  });

  describe('execute()', () => {
    it('executes a single step sequence', async () => {
      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(engine, sequence);
      expect(engine.isRunning).toBe(false);
    });

    it('executes steps in order (sorted by order field)', async () => {
      // Add a second DOM element for request-json
      const jsonEl = document.createElement('div');
      jsonEl.className = 'raw-section-standalone';
      const innerJson = document.createElement('div');
      innerJson.className = 'raw-json';
      jsonEl.appendChild(innerJson);
      document.body.appendChild(jsonEl);

      const executionOrder = [];
      const onShow = vi.fn((text) => executionOrder.push(text));

      const eng = new AnimationEngine(registry, { onAnnotationShow: onShow });

      // Steps provided out of order — engine should sort by `order`
      const sequence = [
        { focusArea: 'request-json', zoomLevel: 1.2, annotation: 'Second', dwellMs: 100, order: 1 },
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: 'First', dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(eng, sequence);

      expect(executionOrder).toEqual(['First', 'Second']);
    });

    it('skips steps with unregistered focus areas', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const onShow = vi.fn();
      const eng = new AnimationEngine(registry, { onAnnotationShow: onShow });

      const sequence = [
        { focusArea: 'nonexistent-area', zoomLevel: 1.0, annotation: 'Skipped', dwellMs: 100, order: 0 },
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: 'Valid', dwellMs: 100, order: 1 },
      ];

      await executeWithTimers(eng, sequence);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('nonexistent-area')
      );
      expect(onShow).toHaveBeenCalledWith('Valid', expect.any(Object));
      expect(onShow).not.toHaveBeenCalledWith('Skipped', expect.anything());

      warnSpy.mockRestore();
    });

    it('skips steps where DOM element is not found', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // mutations-panel is registered but no matching DOM element exists
      const sequence = [
        { focusArea: 'mutations-panel', zoomLevel: 1.2, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(engine, sequence);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('mutations-panel')
      );

      warnSpy.mockRestore();
    });

    it('calls onAnnotationShow and onAnnotationHide callbacks', async () => {
      const onShow = vi.fn();
      const onHide = vi.fn();
      const eng = new AnimationEngine(registry, {
        onAnnotationShow: onShow,
        onAnnotationHide: onHide,
      });

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: 'Test annotation', dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(eng, sequence);

      expect(onShow).toHaveBeenCalledWith('Test annotation', expect.any(Object));
      expect(onHide).toHaveBeenCalled();
    });

    it('does not call onAnnotationShow when annotation is null', async () => {
      const onShow = vi.fn();
      const eng = new AnimationEngine(registry, { onAnnotationShow: onShow });

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(eng, sequence);

      expect(onShow).not.toHaveBeenCalled();
    });

    it('handles empty sequence gracefully', async () => {
      await engine.execute([]);
      expect(engine.isRunning).toBe(false);
    });
  });

  describe('stop()', () => {
    it('stops a running sequence', async () => {
      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 50000, order: 0 },
      ];

      const executePromise = engine.execute(sequence);
      expect(engine.isRunning).toBe(true);

      await engine.stop();

      // Flush remaining
      await flushDwells(executePromise);

      expect(engine.isRunning).toBe(false);
    });

    it('calls onAnnotationHide when stopping', async () => {
      const onHide = vi.fn();
      const eng = new AnimationEngine(registry, { onAnnotationHide: onHide });

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: 'Visible', dwellMs: 50000, order: 0 },
      ];

      const executePromise = eng.execute(sequence);
      await eng.stop();
      await flushDwells(executePromise);

      expect(onHide).toHaveBeenCalled();
    });

    it('is safe to call when not running', async () => {
      // Should not throw
      await engine.stop();
      expect(engine.isRunning).toBe(false);
    });

    it('reverses transforms on stop', async () => {
      const { gsap } = await import('./gsapSetup.js');

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.5, annotation: null, dwellMs: 50000, order: 0 },
      ];

      const executePromise = engine.execute(sequence);

      // Flush microtasks so the step progresses through scroll/zoom to dwell
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();

      // The engine should now be in the dwell phase with a transform on the stack
      gsap.to.mockClear();
      await engine.stop();
      await flushDwells(executePromise);

      // After stop, gsap.to should have been called to reverse transforms.
      expect(gsap.to).toHaveBeenCalled();
      const allCalls = gsap.to.mock.calls;
      const hasScaleRestore = allCalls.some(
        (call) => call[1].scale === 1
      );
      expect(hasScaleRestore).toBe(true);
    });
  });

  describe('skipToNext()', () => {
    it('resolves current dwell early', async () => {
      const onShow = vi.fn();
      const eng = new AnimationEngine(registry, { onAnnotationShow: onShow });

      // Add second element for request-json
      const jsonEl = document.createElement('div');
      jsonEl.className = 'raw-section-standalone';
      const innerJson = document.createElement('div');
      innerJson.className = 'raw-json';
      jsonEl.appendChild(innerJson);
      document.body.appendChild(jsonEl);

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: 'Step 1', dwellMs: 999999, order: 0 },
        { focusArea: 'request-json', zoomLevel: 1.2, annotation: 'Step 2', dwellMs: 100, order: 1 },
      ];

      const executePromise = eng.execute(sequence);

      // Allow the first step to reach its dwell
      await Promise.resolve();
      await Promise.resolve();

      // Skip the first step's very long dwell
      eng.skipToNext();

      // Flush remaining — step 2 has a 100ms dwell
      await flushDwells(executePromise, 100);

      // Both annotations should have been shown
      expect(onShow).toHaveBeenCalledWith('Step 1', expect.any(Object));
      expect(onShow).toHaveBeenCalledWith('Step 2', expect.any(Object));
    });

    it('does nothing when not running', () => {
      // Should not throw
      engine.skipToNext();
    });
  });

  describe('GSAP integration', () => {
    it('calls gsap.to for zoom and reverse', async () => {
      const { gsap } = await import('./gsapSetup.js');
      gsap.to.mockClear();

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.5, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(engine, sequence);

      // Should have calls for: scroll, zoom, reverse zoom
      expect(gsap.to).toHaveBeenCalled();

      // Check that zoom was applied with correct scale
      const zoomCall = gsap.to.mock.calls.find(
        (call) => call[1].scale === 1.5
      );
      expect(zoomCall).toBeDefined();
      expect(zoomCall[1].transformOrigin).toBe('center center');
      expect(zoomCall[1].ease).toBe('power2.inOut');
    });
  });

  describe('mutation hover lifecycle', () => {
    let mutationEl;
    let mutationContainer;

    beforeEach(() => {
      // Set up DOM elements for mutation-line testing
      mutationContainer = document.createElement('div');
      mutationContainer.className = 'main-top-right';
      const rawJson = document.createElement('div');
      rawJson.className = 'raw-json';
      mutationContainer.appendChild(rawJson);
      document.body.appendChild(mutationContainer);

      mutationEl = document.createElement('div');
      mutationEl.className = 'raw-json-mutation-line';
      rawJson.appendChild(mutationEl);
    });

    it('dispatches mouseenter before dwell on mutation-line steps', async () => {
      const events = [];
      mutationEl.addEventListener('mouseenter', () => events.push('mouseenter'));
      mutationEl.addEventListener('mouseleave', () => events.push('mouseleave'));

      const sequence = [
        { focusArea: 'mutation-line-1', zoomLevel: 1.4, annotation: null, dwellMs: 2000, order: 0 },
      ];

      await executeWithTimers(engine, sequence);

      expect(events[0]).toBe('mouseenter');
    });

    it('dispatches mouseleave after dwell on mutation-line steps', async () => {
      const events = [];
      mutationEl.addEventListener('mouseenter', () => events.push('mouseenter'));
      mutationEl.addEventListener('mouseleave', () => events.push('mouseleave'));

      const sequence = [
        { focusArea: 'mutation-line-1', zoomLevel: 1.4, annotation: null, dwellMs: 2000, order: 0 },
      ];

      await executeWithTimers(engine, sequence);

      expect(events).toContain('mouseenter');
      expect(events).toContain('mouseleave');
      // mouseenter should come before mouseleave
      expect(events.indexOf('mouseenter')).toBeLessThan(events.indexOf('mouseleave'));
    });

    it('does not dispatch hover events for non-mutation-line steps', async () => {
      const events = [];
      mockElement.addEventListener('mouseenter', () => events.push('mouseenter'));
      mockElement.addEventListener('mouseleave', () => events.push('mouseleave'));

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(engine, sequence);

      expect(events).toEqual([]);
    });

    it('enforces minimum 2000ms dwell for mutation-line focus areas', async () => {
      // A mutation-line step with dwellMs < 2000 should still dwell for 2000ms
      const sequence = [
        { focusArea: 'mutation-line-1', zoomLevel: 1.4, annotation: null, dwellMs: 500, order: 0 },
      ];

      const executePromise = engine.execute(sequence);

      // Advance 500ms — should still be running (enforced to 2000ms)
      vi.advanceTimersByTime(500);
      await Promise.resolve();
      await Promise.resolve();
      expect(engine.isRunning).toBe(true);

      // Advance to 2000ms total — should complete the dwell
      vi.advanceTimersByTime(1500);
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await flushDwells(executePromise);

      expect(engine.isRunning).toBe(false);
    });

    it('uses specified dwell when it exceeds 2000ms for mutation-line steps', async () => {
      const sequence = [
        { focusArea: 'mutation-line-1', zoomLevel: 1.4, annotation: null, dwellMs: 5000, order: 0 },
      ];

      const executePromise = engine.execute(sequence);

      // Advance 2000ms — should still be running (dwell is 5000ms)
      vi.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();
      expect(engine.isRunning).toBe(true);

      // Advance remaining time
      await flushDwells(executePromise);
      expect(engine.isRunning).toBe(false);
    });

    it('clears hover state on stop()', async () => {
      const events = [];
      mutationEl.addEventListener('mouseleave', () => events.push('mouseleave'));

      const sequence = [
        { focusArea: 'mutation-line-1', zoomLevel: 1.4, annotation: null, dwellMs: 50000, order: 0 },
      ];

      const executePromise = engine.execute(sequence);

      // Let the step progress to the dwell phase (GSAP completes instantly)
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();

      // Stop during dwell — should clear hover state
      await engine.stop();
      await flushDwells(executePromise);

      // mouseleave should have been dispatched via _clearHoverState
      expect(events).toContain('mouseleave');
    });

    it('does not dispatch mouseleave on stop if no hover is active', async () => {
      const events = [];
      mockElement.addEventListener('mouseleave', () => events.push('mouseleave'));

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 50000, order: 0 },
      ];

      const executePromise = engine.execute(sequence);

      await Promise.resolve();
      await Promise.resolve();

      await engine.stop();
      await flushDwells(executePromise);

      // No mouseleave should be dispatched since timeline is not a mutation-line
      expect(events).toEqual([]);
    });
  });

  describe('autoScroll option', () => {
    it('skips scrolling when autoScroll is false', async () => {
      const { gsap } = await import('./gsapSetup.js');
      gsap.to.mockClear();

      const eng = new AnimationEngine(registry, { autoScroll: false });

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(eng, sequence);

      // Should only have zoom + reverse zoom calls (no scroll)
      const scrollCalls = gsap.to.mock.calls.filter((call) => call[1].scrollTo);
      expect(scrollCalls).toHaveLength(0);
    });

    it('scrolls container when autoScroll is true (default)', async () => {
      const { gsap } = await import('./gsapSetup.js');
      gsap.to.mockClear();

      const sequence = [
        { focusArea: 'timeline', zoomLevel: 1.3, annotation: null, dwellMs: 100, order: 0 },
      ];

      await executeWithTimers(engine, sequence);

      // Should have a scroll call with scrollTo
      const scrollCalls = gsap.to.mock.calls.filter((call) => call[1].scrollTo);
      expect(scrollCalls.length).toBeGreaterThan(0);
    });
  });
});
