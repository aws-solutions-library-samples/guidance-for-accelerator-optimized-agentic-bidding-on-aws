/**
 * @vitest-environment jsdom
 *
 * One-click "Compare & Assess" flow in the Governance panel. The user picks only
 * the baseline load test; the challenger side is always a fresh load test of the
 * latest version on that baseline's inputs.
 *
 * Verifies the real orchestration entry + honest failure handling without a
 * live backend: for DLRM (the default model) the challenger canary is already
 * staged by the governance agent, so the UI must NOT call stage-canary — it
 * goes straight to a challenger load test at the selected current run's
 * scenario+preset. A 409 (a load test already running) is surfaced honestly
 * rather than swallowed.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

let fetchHandler;
vi.mock('../authFetch.js', () => ({
  authFetch: (...args) => fetchHandler(...args),
  isProxyTransport: () => false,
}));
vi.mock('../agentCoreClient.js', () => ({
  invokeGovernance: vi.fn(async () => ({})),
  agentUnavailableReason: () => null,
}));
// The sweep-status child does its own polling; stub it out so this test
// exercises only GovernancePanel's own one-click logic.
vi.mock('./LoadTestSweepStatus.jsx', () => ({ default: () => null }));

import GovernancePanel, { replayInputs } from './GovernancePanel.jsx';

// Enable React's act() environment so async state updates flush deterministically.
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

const CONTROL_RUN = {
  id: 'lt-cur',
  timestamp: '2026-01-01T00:00:00Z',
  scenario: 'nfl_sunday',
  preset: '10k',
  seed: 42,
  duration_s: 10,
};

let calls;

function defaultHandler(loadtestResponder, controlRun = CONTROL_RUN) {
  return (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });

    if (url.includes('/closed-loop/scenarios')) return Promise.resolve(jsonResp({ scenarios: [] }));
    if (url.includes('/closed-loop/models')) return Promise.resolve(jsonResp({ versions: [] }));
    if (url.includes('/governance/eligible-runs')) {
      if (url.includes('role=current')) {
        return Promise.resolve(jsonResp({ runs: [controlRun], most_recent: controlRun }));
      }
      return Promise.resolve(jsonResp({ runs: [], most_recent: null }));
    }
    if (url.includes('/governance/comparison-pair')) {
      return Promise.resolve(jsonResp({ control_run_id: 'lt-cur', control_source: 'training_provenance' }));
    }
    // Not-ok keeps trainingEstimate null so its optional render block (which
    // reads rate/duration fields) is skipped — this test is about the compare
    // flow, not the training-estimate card.
    if (url.includes('/governance/training-estimate')) {
      return Promise.resolve(jsonResp({ error: 'n/a' }, { ok: false, status: 404 }));
    }
    if (url.endsWith('/api/v1/loadtest') && method === 'POST') return Promise.resolve(loadtestResponder());
    return Promise.resolve(jsonResp({}));
  };
}

async function flush(times = 6) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  }
}

let container;
let root;

beforeEach(() => {
  calls = [];
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

async function openTestVersionsStep() {
  await act(async () => { root.render(<GovernancePanel />); });
  await flush();
  const step4 = container.querySelector('[data-testid="governance-step-test-versions"]');
  await act(async () => { step4.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
  await flush();
}

describe('one-click Compare & Assess (DLRM)', () => {
  it('offers one baseline picker and one compare button, with no challenger picker', async () => {
    fetchHandler = defaultHandler(() => jsonResp({}));
    await openTestVersionsStep();

    expect(container.querySelector('[data-testid="governance-current-run-select"]')).toBeTruthy();
    expect(container.querySelector('[data-testid="governance-challenger-run-select"]')).toBeNull();
    expect(container.querySelector('[data-testid="governance-compare-button"]')).toBeNull();
    const btn = container.querySelector('[data-testid="governance-compare-retrained-button"]');
    expect(btn.textContent).toContain('Compare & Assess');
    expect(calls.some((c) => c.url.includes('role=challenger'))).toBe(false);
  });

  it('flags a baseline recorded before seed and duration were stored', async () => {
    const legacy = { id: 'lt-cur', timestamp: '2026-01-01T00:00:00Z', scenario: 'nfl_sunday', preset: '10k' };
    fetchHandler = defaultHandler(() => jsonResp({}), legacy);
    await openTestVersionsStep();
    const note = container.querySelector('[data-testid="governance-replay-legacy"]');
    expect(note).toBeTruthy();
    expect(note.textContent).toContain('seed, duration_s');
  });

  it('surfaces a 409 honestly and does NOT stage a canary for DLRM', async () => {
    fetchHandler = defaultHandler(() =>
      jsonResp({ error: 'busy', reason: 'busy' }, { ok: false, status: 409 })
    );

    await act(async () => { root.render(<GovernancePanel />); });
    await flush();

    // The compare card lives in Step 4 now, and Step 1 is the landing step, so
    // the step has to be opened first — the same click a user makes.
    const step4 = container.querySelector('[data-testid="governance-step-test-versions"]');
    expect(step4).toBeTruthy();
    await act(async () => { step4.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    await flush();

    const btn = container.querySelector('[data-testid="governance-compare-retrained-button"]');
    expect(btn).toBeTruthy();
    // Eligible-runs auto-selected the control run, so the button is enabled.
    expect(btn.disabled).toBe(false);

    await act(async () => { btn.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
    await flush();

    // Honest 409 message.
    const err = container.querySelector('[data-testid="governance-retrain-error"]');
    expect(err).toBeTruthy();
    expect(err.textContent).toMatch(/already running/i);

    // DLRM canary is agent-staged: the UI must not call stage-canary.
    const stageCalls = calls.filter((c) => c.url.includes('/governance/stage-canary'));
    expect(stageCalls).toHaveLength(0);

    // It did start a challenger run at the control run's scenario+preset.
    const ltCalls = calls.filter((c) => c.url.endsWith('/api/v1/loadtest') && c.method === 'POST');
    expect(ltCalls).toHaveLength(1);
    expect(ltCalls[0].body).toMatchObject({
      target_model_type: 'dlrm_bid_shader',
      target_variant: 'challenger',
      scenario: 'nfl_sunday',
      preset: '10k',
      seed: 42,
      duration_s: 10,
    });
    expect(container.querySelector('[data-testid="governance-replay-legacy"]')).toBeNull();
  });
});

describe('replayInputs', () => {
  it('replays every recorded input of the baseline run', () => {
    const r = replayInputs({ preset: '100k', scenario: 'late_night', seed: 7, duration_s: 60 });
    expect(r.inputs).toEqual({ preset: '100k', scenario: 'late_night', seed: 7, duration_s: 60 });
    expect(r.missing).toEqual([]);
  });

  it('fills legacy defaults and names the fields an older record lacks', () => {
    const r = replayInputs({ preset: '1k', scenario: 'baseline' });
    expect(r.inputs).toEqual({ preset: '1k', scenario: 'baseline', seed: 42, duration_s: 30 });
    expect(r.missing).toEqual(['seed', 'duration_s']);
  });

  it('keeps a recorded seed of 0', () => {
    expect(replayInputs({ preset: '1k', scenario: 'baseline', seed: 0, duration_s: 10 }).inputs.seed).toBe(0);
  });
});
