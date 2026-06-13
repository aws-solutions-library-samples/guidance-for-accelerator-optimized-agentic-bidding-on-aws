/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import useDemoAnimation from './useDemoAnimation.js';

// Mock GSAP — synchronous completion for predictable tests
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
 * Helper: create a valid animation step
 */
function makeStep(overrides = {}) {
  return {
    focusArea: 'timeline',
    zoomLevel: 1.3,
    annotation: 'Test annotation',
    dwellMs: 100,
    order: 0,
    ...overrides,
  };
}

/**
 * Helper: render a component that uses the hook and captures its result.
 * Uses a callback ref pattern to capture the hook result synchronously.
 */
function setupHook(container) {
  let captured = {};
  let renderCount = 0;

  function TestComponent() {
    const hookResult = useDemoAnimation();
    // Capture on every render
    captured.isRunning = hookResult.isRunning;
    captured.start = hookResult.start;
    captured.stop = hookResult.stop;
    captured.skipToNext = hookResult.skipToNext;
    captured.engine = hookResult.engine;
    renderCount++;
    return null;
  }

  const root = createRoot(container);
  root.render(React.createElement(TestComponent));

  // Flush synchronous React render (React 18 createRoot batches, but in test env it's sync)
  // We need to wait for the initial render
  return {
    get result() { return captured; },
    get renderCount() { return renderCount; },
    root,
    TestComponent,
  };
}

describe('useDemoAnimation', () => {
  let container;
  let mockElement;

  beforeEach(() => {
    vi.useFakeTimers();

    container = document.createElement('div');
    document.body.appendChild(container);

    // Set up a DOM element that the registry can resolve
    mockElement = document.createElement('div');
    mockElement.className = 'timeline-container';
    document.body.appendChild(mockElement);
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('returns the expected API shape', async () => {
    const { result } = setupHook(container);

    // Flush React render
    await vi.advanceTimersByTimeAsync(0);

    expect(result).toHaveProperty('isRunning');
    expect(result).toHaveProperty('start');
    expect(result).toHaveProperty('stop');
    expect(result).toHaveProperty('skipToNext');
    expect(result).toHaveProperty('engine');
    expect(typeof result.start).toBe('function');
    expect(typeof result.stop).toBe('function');
    expect(typeof result.skipToNext).toBe('function');
  });

  it('isRunning is initially false', async () => {
    const { result } = setupHook(container);
    await vi.advanceTimersByTimeAsync(0);
    expect(result.isRunning).toBe(false);
  });

  it('isRunning becomes true during execution and false after completion', async () => {
    const { result } = setupHook(container);
    await vi.advanceTimersByTimeAsync(0);

    const startPromise = result.start([makeStep({ dwellMs: 200 })]);

    // After starting, isRunning should be true (state update is batched)
    await vi.advanceTimersByTimeAsync(0);
    expect(result.isRunning).toBe(true);

    // Advance timers to complete the dwell
    await vi.advanceTimersByTimeAsync(300);

    await startPromise;
    // Flush the setState(false) call
    await vi.advanceTimersByTimeAsync(0);

    expect(result.isRunning).toBe(false);
  });

  it('stop() sets isRunning to false', async () => {
    const { result } = setupHook(container);
    await vi.advanceTimersByTimeAsync(0);

    result.start([makeStep({ dwellMs: 5000 })]);
    await vi.advanceTimersByTimeAsync(0);

    expect(result.isRunning).toBe(true);

    await result.stop();
    await vi.advanceTimersByTimeAsync(0);

    expect(result.isRunning).toBe(false);
  });

  it('skipToNext() advances past the current dwell', async () => {
    const { result } = setupHook(container);
    await vi.advanceTimersByTimeAsync(0);

    const steps = [
      makeStep({ dwellMs: 5000, order: 0 }),
      makeStep({ focusArea: 'timeline', dwellMs: 100, order: 1, annotation: 'Step 2' }),
    ];

    const startPromise = result.start(steps);
    await vi.advanceTimersByTimeAsync(0);

    // Skip the first step's long dwell
    result.skipToNext();

    // Advance timers for the second step's dwell
    await vi.advanceTimersByTimeAsync(200);

    await startPromise;
    await vi.advanceTimersByTimeAsync(0);

    expect(result.isRunning).toBe(false);
  });

  it('engine instance is stable across re-renders', async () => {
    let hookResults = [];

    function TestComponent({ count }) {
      const hookResult = useDemoAnimation();
      hookResults.push(hookResult.engine);
      return React.createElement('span', null, count);
    }

    const root = createRoot(container);
    root.render(React.createElement(TestComponent, { count: 1 }));
    await vi.advanceTimersByTimeAsync(0);

    root.render(React.createElement(TestComponent, { count: 2 }));
    await vi.advanceTimersByTimeAsync(0);

    // Engine should be the same instance across renders
    expect(hookResults.length).toBeGreaterThanOrEqual(2);
    expect(hookResults[0]).toBe(hookResults[hookResults.length - 1]);
  });

  it('calls engine.stop() on unmount', async () => {
    const { result, root } = setupHook(container);
    await vi.advanceTimersByTimeAsync(0);

    const stopSpy = vi.spyOn(result.engine, 'stop');

    root.unmount();
    await vi.advanceTimersByTimeAsync(0);

    expect(stopSpy).toHaveBeenCalled();
  });

  it('engine has a FocusAreaRegistry with default areas registered', async () => {
    const { result } = setupHook(container);
    await vi.advanceTimersByTimeAsync(0);

    const engine = result.engine;
    expect(engine._registry.has('timeline')).toBe(true);
    expect(engine._registry.has('request-json')).toBe(true);
    expect(engine._registry.has('mutations-panel')).toBe(true);
  });
});
