/**
 * @vitest-environment jsdom
 *
 * One-click "Compare current vs. retrained" flow in the Governance panel.
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
}));
vi.mock('../agentCoreClient.js', () => ({
  invokeGovernance: vi.fn(async () => ({})),
  agentUnavailableReason: () => null,
}));
// The sweep-status child does its own polling; stub it out so this test
// exercises only GovernancePanel's own one-click logic.
vi.mock('./LoadTestSweepStatus.jsx', () => ({ default: () => null }));

import GovernancePanel from './GovernancePanel.jsx';

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
};

let calls;

function defaultHandler(loadtestResponder) {
  return (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });

    if (url.includes('/closed-loop/scenarios')) return Promise.resolve(jsonResp({ scenarios: [] }));
    if (url.includes('/closed-loop/models')) return Promise.resolve(jsonResp({ versions: [] }));
    if (url.includes('/governance/eligible-runs')) {
      if (url.includes('role=current')) {
        return Promise.resolve(jsonResp({ runs: [CONTROL_RUN], most_recent: CONTROL_RUN }));
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

describe('one-click Compare current vs. retrained (DLRM)', () => {
  it('surfaces a 409 honestly and does NOT stage a canary for DLRM', async () => {
    fetchHandler = defaultHandler(() =>
      jsonResp({ error: 'busy', reason: 'busy' }, { ok: false, status: 409 })
    );

    await act(async () => { root.render(<GovernancePanel />); });
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
    });
  });
});
