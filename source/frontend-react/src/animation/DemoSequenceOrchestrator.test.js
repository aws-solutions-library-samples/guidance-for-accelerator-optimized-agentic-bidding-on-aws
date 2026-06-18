/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import DemoSequenceOrchestrator from './DemoSequenceOrchestrator.js';
import { SCENARIOS } from '../components/ScenarioCard.jsx';

// Mock the AnimationEngine
function createMockEngine() {
  return {
    execute: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn().mockResolvedValue(undefined),
    isRunning: false,
  };
}

// Mock result matching the normalized orchestrator response shape
function createMockResult(opts = {}) {
  const mutations = opts.mutations || [
    { agent: 'Wide & Deep', intent: 'ACTIVATE_SEGMENTS', path: 'segments', op: 1, payload: ['seg1'] },
    { agent: 'Metrics', intent: 'ADD_METRICS', path: 'metrics', op: 1, payload: { metric: [{ type: 'viewability', value: 0.85 }] } },
  ];

  return {
    id: opts.id || 'test-result',
    totalLatencyMs: opts.totalLatencyMs || 142,
    stops: opts.stops || [
      { id: 'ssp', displayName: 'SSP', status: 'ok', latency: null, mutations: [] },
      { id: 'widedeep', displayName: 'Wide & Deep', status: 'ok', latency: { ms: 45 }, mutations: [mutations[0]] },
      { id: 'metrics', displayName: 'Metrics Enricher', status: 'ok', latency: { ms: 30 }, mutations: [mutations[1]] },
      { id: 'dsp', displayName: 'DSP', status: 'ok', latency: null, mutations: [] },
    ],
  };
}

describe('DemoSequenceOrchestrator', () => {
  let engine;
  let submitFn;
  let orchestrator;

  beforeEach(() => {
    vi.useFakeTimers();
    engine = createMockEngine();
    submitFn = vi.fn().mockResolvedValue(createMockResult());
    orchestrator = new DemoSequenceOrchestrator(engine, submitFn, {
      interScenarioDelayMs: 100, // Short delay for tests
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('constructor', () => {
    it('initializes with default options', () => {
      expect(orchestrator.isRunning).toBe(false);
      expect(orchestrator.currentScenarioIndex).toBe(0);
    });

    it('accepts custom options', () => {
      const custom = new DemoSequenceOrchestrator(engine, submitFn, {
        maxConsecutiveFailures: 5,
        interScenarioDelayMs: 2000,
      });
      expect(custom.isRunning).toBe(false);
    });
  });

  describe('buildSequenceForResult()', () => {
    it('produces timeline step first with correct annotation', () => {
      const result = createMockResult({ totalLatencyMs: 200 });
      const scenario = SCENARIOS[0]; // banner-basic
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      expect(steps[0]).toEqual({
        focusArea: 'timeline',
        zoomLevel: 1.3,
        annotation: expect.stringContaining(scenario.name),
        dwellMs: 3000,
        order: 0,
      });
      expect(steps[0].annotation).toContain('200ms');
      expect(steps[0].annotation).toContain('2 agents');
    });

    it('produces request-json step second', () => {
      const result = createMockResult();
      const scenario = SCENARIOS[0];
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      expect(steps[1]).toEqual({
        focusArea: 'request-json',
        zoomLevel: 1.2,
        annotation: 'Request with 2 mutations applied',
        dwellMs: 2500,
        order: 1,
      });
    });

    it('produces mutation-line steps for each mutation', () => {
      const result = createMockResult();
      const scenario = SCENARIOS[0];
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      // 2 mutations → 2 mutation-line steps
      expect(steps[2]).toEqual({
        focusArea: 'mutation-line-1',
        zoomLevel: 1.4,
        annotation: 'Wide & Deep: ACTIVATE_SEGMENTS',
        dwellMs: 2000,
        order: 2,
      });
      expect(steps[3]).toEqual({
        focusArea: 'mutation-line-2',
        zoomLevel: 1.4,
        annotation: 'Metrics: ADD_METRICS',
        dwellMs: 2000,
        order: 3,
      });
    });

    it('handles result with no mutations', () => {
      const result = createMockResult({
        stops: [
          { id: 'ssp', status: 'ok', mutations: [] },
          { id: 'dsp', status: 'ok', mutations: [] },
        ],
      });
      const scenario = SCENARIOS[0];
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      expect(steps).toHaveLength(2); // timeline + request-json only
      expect(steps[1].annotation).toBe('Request with 0 mutations applied');
    });

    it('orders steps with ascending order values', () => {
      const result = createMockResult();
      const scenario = SCENARIOS[0];
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      for (let i = 0; i < steps.length - 1; i++) {
        expect(steps[i].order).toBeLessThan(steps[i + 1].order);
      }
    });

    it('handles missing stops gracefully', () => {
      const result = { totalLatencyMs: 0 };
      const scenario = SCENARIOS[0];
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      expect(steps).toHaveLength(2);
      expect(steps[0].annotation).toContain('0 agents');
    });

    it('uses mutation.source as fallback when agent is missing', () => {
      const result = createMockResult({
        stops: [
          { id: 'ssp', status: 'ok', mutations: [{ source: 'DLRM', intent: 'BID_SHADE' }] },
        ],
      });
      const scenario = SCENARIOS[0];
      const steps = orchestrator.buildSequenceForResult(result, scenario);

      expect(steps[2].annotation).toBe('DLRM: BID_SHADE');
    });
  });

  describe('start()', () => {
    it('sets isRunning to true while running', async () => {
      // Make submitFn hang until we stop
      let resolveSubmit;
      submitFn.mockImplementation(() => new Promise((r) => { resolveSubmit = r; }));

      const startPromise = orchestrator.start();
      await vi.advanceTimersByTimeAsync(0);

      expect(orchestrator.isRunning).toBe(true);

      // Stop to clean up
      await orchestrator.stop();
      if (resolveSubmit) resolveSubmit(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('submits the first scenario', async () => {
      // Make submitFn hang on first call so we can inspect
      let resolveSubmit;
      submitFn.mockImplementation(() => new Promise((r) => { resolveSubmit = r; }));

      const startPromise = orchestrator.start();
      await vi.advanceTimersByTimeAsync(0);

      expect(submitFn).toHaveBeenCalledWith(SCENARIOS[0]);

      // Clean up
      await orchestrator.stop();
      if (resolveSubmit) resolveSubmit(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('feeds built sequence to engine.execute()', async () => {
      // Let first scenario succeed, then hang on second
      let callCount = 0;
      let resolveSecond;
      submitFn.mockImplementation(async () => {
        callCount++;
        if (callCount === 1) return createMockResult();
        return new Promise((r) => { resolveSecond = r; });
      });

      const startPromise = orchestrator.start();
      // Flush microtasks for first submit + engine.execute
      await vi.advanceTimersByTimeAsync(0);
      // Advance past inter-scenario delay
      await vi.advanceTimersByTimeAsync(200);

      expect(engine.execute).toHaveBeenCalled();
      const sequence = engine.execute.mock.calls[0][0];
      expect(sequence[0].focusArea).toBe('timeline');
      expect(sequence[1].focusArea).toBe('request-json');

      // Clean up
      await orchestrator.stop();
      if (resolveSecond) resolveSecond(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('advances to next scenario after completion', async () => {
      let callCount = 0;
      let resolveThird;
      submitFn.mockImplementation(async () => {
        callCount++;
        if (callCount <= 2) return createMockResult();
        return new Promise((r) => { resolveThird = r; });
      });

      const startPromise = orchestrator.start();
      // Flush through first two scenarios + delays
      await vi.advanceTimersByTimeAsync(500);

      expect(submitFn).toHaveBeenCalledWith(SCENARIOS[0]);
      expect(submitFn).toHaveBeenCalledWith(SCENARIOS[1]);

      // Clean up
      await orchestrator.stop();
      if (resolveThird) resolveThird(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('loops back to first scenario after all complete', async () => {
      let callCount = 0;
      let resolveHang;
      submitFn.mockImplementation(async () => {
        callCount++;
        if (callCount <= SCENARIOS.length) return createMockResult();
        // On the 9th call (first scenario again), hang
        return new Promise((r) => { resolveHang = r; });
      });

      const startPromise = orchestrator.start();
      // Advance enough time for all 8 scenarios + delays
      await vi.advanceTimersByTimeAsync(5000);

      // Should have been called SCENARIOS.length + 1 times (looped back)
      expect(callCount).toBe(SCENARIOS.length + 1);
      // The last call should be for the first scenario again
      expect(submitFn.mock.calls[SCENARIOS.length][0]).toEqual(SCENARIOS[0]);

      // Clean up
      await orchestrator.stop();
      if (resolveHang) resolveHang(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });
  });

  describe('error handling', () => {
    it('skips failed scenario and continues to next', async () => {
      let callCount = 0;
      let resolveThird;
      submitFn.mockImplementation(async (scenario) => {
        callCount++;
        if (callCount === 1) throw new Error('Network error');
        if (callCount === 2) return createMockResult();
        return new Promise((r) => { resolveThird = r; });
      });

      const startPromise = orchestrator.start();
      await vi.advanceTimersByTimeAsync(1000);

      // First call fails, second succeeds
      expect(callCount).toBeGreaterThanOrEqual(2);
      expect(submitFn.mock.calls[1][0]).toEqual(SCENARIOS[1]);

      // Clean up
      await orchestrator.stop();
      if (resolveThird) resolveThird(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('shows error annotation on failure', async () => {
      let callCount = 0;
      let resolveHang;
      submitFn.mockImplementation(async () => {
        callCount++;
        if (callCount === 1) throw new Error('Timeout');
        return new Promise((r) => { resolveHang = r; });
      });

      const startPromise = orchestrator.start();
      await vi.advanceTimersByTimeAsync(500);

      // Engine should have been called with an error annotation sequence
      const errorCall = engine.execute.mock.calls.find(
        ([seq]) => seq[0]?.annotation?.includes('Failed')
      );
      expect(errorCall).toBeDefined();
      expect(errorCall[0][0].annotation).toContain(SCENARIOS[0].name);

      // Clean up
      await orchestrator.stop();
      if (resolveHang) resolveHang(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('stops after 3 consecutive failures', async () => {
      submitFn.mockRejectedValue(new Error('API unavailable'));

      const startPromise = orchestrator.start();
      // Advance enough time for 3 failures + error annotations + delays
      await vi.advanceTimersByTimeAsync(5000);
      await startPromise;

      // Should have attempted 3 scenarios then stopped
      expect(submitFn).toHaveBeenCalledTimes(3);
      expect(orchestrator.isRunning).toBe(false);
    });

    it('resets consecutive failure counter on success', async () => {
      let callCount = 0;
      let resolveHang;
      submitFn.mockImplementation(async () => {
        callCount++;
        if (callCount === 1) throw new Error('fail 1');
        if (callCount === 2) return createMockResult(); // success resets counter
        if (callCount === 3) throw new Error('fail 3');
        if (callCount === 4) throw new Error('fail 4');
        // 5th call — still running because counter was reset after call 2
        return new Promise((r) => { resolveHang = r; });
      });

      const startPromise = orchestrator.start();
      await vi.advanceTimersByTimeAsync(5000);

      // Should have gotten past 4 calls because the success at call 2 reset the counter
      expect(callCount).toBeGreaterThanOrEqual(5);

      // Clean up
      await orchestrator.stop();
      if (resolveHang) resolveHang(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('respects custom maxConsecutiveFailures', async () => {
      const custom = new DemoSequenceOrchestrator(engine, submitFn, {
        maxConsecutiveFailures: 1,
        interScenarioDelayMs: 100,
      });
      submitFn.mockRejectedValue(new Error('fail'));

      const startPromise = custom.start();
      await vi.advanceTimersByTimeAsync(5000);
      await startPromise;

      expect(submitFn).toHaveBeenCalledTimes(1);
      expect(custom.isRunning).toBe(false);
    });
  });

  describe('stop()', () => {
    it('stops the engine', async () => {
      let resolveSubmit;
      submitFn.mockImplementation(() => new Promise((r) => { resolveSubmit = r; }));

      const startPromise = orchestrator.start();
      await vi.advanceTimersByTimeAsync(0);

      await orchestrator.stop();

      expect(engine.stop).toHaveBeenCalled();
      expect(orchestrator.isRunning).toBe(false);

      // Clean up
      if (resolveSubmit) resolveSubmit(createMockResult());
      await vi.advanceTimersByTimeAsync(0);
    });

    it('clears pending inter-scenario delay', async () => {
      // Let first scenario succeed immediately
      let callCount = 0;
      let resolveSecond;
      submitFn.mockImplementation(async () => {
        callCount++;
        if (callCount === 1) return createMockResult();
        return new Promise((r) => { resolveSecond = r; });
      });

      const startPromise = orchestrator.start();
      // Let first scenario complete
      await vi.advanceTimersByTimeAsync(0);

      // Stop during the inter-scenario delay (before it fires)
      await orchestrator.stop();

      expect(orchestrator.isRunning).toBe(false);
      // Second scenario should not have been submitted
      expect(callCount).toBe(1);
    });

    it('is idempotent when not running', async () => {
      await orchestrator.stop();
      expect(engine.stop).toHaveBeenCalled();
      expect(orchestrator.isRunning).toBe(false);
    });
  });
});
