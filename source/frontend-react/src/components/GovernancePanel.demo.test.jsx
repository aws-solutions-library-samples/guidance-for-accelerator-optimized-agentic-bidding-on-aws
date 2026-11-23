/**
 * @vitest-environment jsdom
 *
 * FR2.1, FR2.2: the Train step's "Include demo dataset" option is off by
 * default, disabled while no demo dataset exists, and sends
 * include_demo_dataset with the train request. With it on, Train is allowed
 * before any load test run is ready, and run_id is sent empty.
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
vi.mock('./LoadTestSweepStatus.jsx', () => ({
  default: ({ runId }) => <div data-testid="sweep-stub" data-run-id={runId} />,
}));

import GovernancePanel from './GovernancePanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

const READY_RUN = {
  id: 'lt-ready', timestamp: '2026-11-15T09:00:00Z',
  target_model_type: 'dlrm_bid_shader', target_variant: 'current', outcome_sample_count: 40,
};
const PARAMS = [{ name: 'win_rate', label: 'Win rate', minimum: 0.1, maximum: 1.0, default: 0.9 }];

let calls;
let demoExists;
let trainable;
let container;
let root;

function handler(url, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();
  calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });
  if (url.includes('/closed-loop/scenarios')) return Promise.resolve(jsonResp({ scenarios: [] }));
  if (url.includes('/closed-loop/models')) return Promise.resolve(jsonResp({ versions: [], training_jobs: [] }));
  if (url.includes('/governance/eligible-runs')) return Promise.resolve(jsonResp({ runs: [], most_recent: null }));
  if (url.includes('/governance/training-estimate')) {
    return Promise.resolve(jsonResp({
      instance_type: 'ml.g6.2xlarge', hourly_rate_usd: 1.222,
      max_runtime_seconds: 14400, estimated_max_cost_usd: 4.89,
    }));
  }
  if (url.includes('/governance/trainable-runs')) {
    return Promise.resolve(jsonResp({ etl: { configured: true }, ...trainable }));
  }
  if (url.includes('/governance/demo-dataset/estimate')) {
    return Promise.resolve(jsonResp({ sizing: { rows: 55556, positive_kind: 'conversions', expected_positives: 50 } }));
  }
  if (url.includes('/governance/demo-dataset')) {
    return Promise.resolve(jsonResp({
      model_type: 'dlrm_bid_shader', prefix: 'demo-training-data/', params: PARAMS,
      exists: demoExists,
      metadata: demoExists
        ? { generated_at: '2026-11-15T10:00:00Z', rows_written: 55556, actual_positives: 48, seed: 1, params: { win_rate: 0.9 }, sizing: {} }
        : null,
    }));
  }
  if (url.endsWith('/api/v1/governance/train') && method === 'POST') {
    return Promise.resolve(jsonResp({ job_name: 'dlrm-bid-shader-1-ab', base_model_version: '1' }));
  }
  return Promise.resolve(jsonResp({}));
}

async function flush(times = 8) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  }
}

const q = (id) => container.querySelector(`[data-testid="${id}"]`);
const click = (el) => act(async () => { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); });

async function openTrainStep() {
  await act(async () => { root.render(<GovernancePanel />); });
  await flush();
  await click(q('governance-step-train'));
  await flush();
}

beforeEach(() => {
  calls = [];
  demoExists = false;
  trainable = { runs: [], pending: [] };
  fetchHandler = handler;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

describe('Train step demo dataset option', () => {
  it('is off and disabled when no demo dataset exists', async () => {
    await openTrainStep();
    const checkbox = q('governance-train-include-demo');
    expect(checkbox.checked).toBe(false);
    expect(checkbox.disabled).toBe(true);
    expect(q('demo-dataset-status').textContent).toBe('No demo dataset');
    expect(q('governance-train-trigger-button').disabled).toBe(true);
  });

  it('enables Train on demo data alone and sends include_demo_dataset with an empty run_id', async () => {
    demoExists = true;
    await openTrainStep();
    const checkbox = q('governance-train-include-demo');
    expect(checkbox.disabled).toBe(false);
    expect(checkbox.checked).toBe(false);

    await click(checkbox);
    await flush();
    expect(q('governance-train-include-demo-notice').textContent).toContain('Demonstration only');
    expect(q('governance-train-trigger-button').disabled).toBe(false);

    await click(q('governance-train-trigger-button'));
    expect(q('governance-train-confirm-text').textContent).toContain('synthetic demo dataset');
    await click(q('governance-train-confirm-button'));
    await flush();

    const train = calls.find((c) => c.url.endsWith('/api/v1/governance/train'));
    expect(train.body).toMatchObject({
      model_type: 'dlrm_bid_shader', confirmed: true, include_demo_dataset: true, run_id: '',
    });
  });

  it('sends include_demo_dataset false by default with a ready run', async () => {
    demoExists = true;
    trainable = { runs: [READY_RUN], pending: [] };
    await openTrainStep();
    await click(q('governance-train-trigger-button'));
    await click(q('governance-train-confirm-button'));
    await flush();
    const train = calls.find((c) => c.url.endsWith('/api/v1/governance/train'));
    expect(train.body).toMatchObject({ run_id: 'lt-ready', include_demo_dataset: false });
  });
});
