/**
 * @vitest-environment jsdom
 *
 * The merged Train step: one run picker covering swept and not-yet-swept runs,
 * the outcome pipeline for the selected run directly under it, and a Train
 * button that is enabled only once the selected run is in the training bucket.
 *
 * It replaces two separate steps that each had their own picker over the same
 * runs, so the two selections could disagree.
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
// Stand-in for the pipeline view. Records the runId the panel hands it and
// exposes the onRunBecameTrainable callback so a test can fire the transition.
const sweep = { runIds: [], onTrainable: null };
vi.mock('./LoadTestSweepStatus.jsx', () => ({
  default: ({ runId, onRunBecameTrainable }) => {
    sweep.runIds.push(runId);
    sweep.onTrainable = onRunBecameTrainable;
    return <div data-testid="sweep-stub" data-run-id={runId} />;
  },
}));

import GovernancePanel, { mergeTrainRunOptions } from './GovernancePanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

const OLD_READY = {
  id: 'lt-old', timestamp: '2026-10-05T15:00:00Z',
  target_model_type: 'dlrm_bid_shader', target_variant: 'current', outcome_sample_count: 40,
};
const NEW_PENDING = { id: 'lt-new', timestamp: '2026-10-05T19:00:00Z', outcome_sample_count: 36 };

let calls;
let trainable;

function handler(url, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();
  calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });

  if (url.includes('/closed-loop/scenarios')) return Promise.resolve(jsonResp({ scenarios: [] }));
  if (url.includes('/closed-loop/models')) return Promise.resolve(jsonResp({ versions: [], training_jobs: [] }));
  if (url.includes('/governance/eligible-runs')) return Promise.resolve(jsonResp({ runs: [], most_recent: null }));
  if (url.includes('/governance/comparison-pair')) return Promise.resolve(jsonResp({}));
  if (url.includes('/governance/training-estimate')) {
    return Promise.resolve(jsonResp({
      instance_type: 'ml.g5.2xlarge', hourly_rate_usd: 1.515,
      max_runtime_seconds: 14400, estimated_max_cost_usd: 6.06,
    }));
  }
  if (url.includes('/governance/trainable-runs')) {
    return Promise.resolve(jsonResp({ etl: { configured: true }, ...trainable }));
  }
  if (url.endsWith('/api/v1/governance/train') && method === 'POST') {
    return Promise.resolve(jsonResp({ job_name: 'dlrm-bid-shader-1-ab', base_model_version: '1' }));
  }
  return Promise.resolve(jsonResp({}));
}

async function flush(times = 6) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  }
}

function click(el) {
  return act(async () => { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
}

const q = (id) => container.querySelector(`[data-testid="${id}"]`);

let container;
let root;

async function openTrainStep() {
  await act(async () => { root.render(<GovernancePanel />); });
  await flush();
  await click(q('governance-step-train'));
  await flush();
}

beforeEach(() => {
  calls = [];
  sweep.runIds = [];
  sweep.onTrainable = null;
  trainable = { runs: [OLD_READY], pending: [NEW_PENDING] };
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

describe('mergeTrainRunOptions', () => {
  it('lists ready and pending runs newest first, flagged', () => {
    const merged = mergeTrainRunOptions([OLD_READY], [NEW_PENDING]);
    expect(merged.map((r) => [r.id, r.ready])).toEqual([['lt-new', false], ['lt-old', true]]);
  });

  it('never lists a run twice when it appears in both lists', () => {
    const merged = mergeTrainRunOptions([OLD_READY], [{ ...OLD_READY }]);
    expect(merged).toHaveLength(1);
    expect(merged[0].ready).toBe(true);
  });

  it('handles missing inputs', () => {
    expect(mergeTrainRunOptions(undefined, null)).toEqual([]);
  });
});

describe('Train step', () => {
  it('has no separate outcome-pipeline step', async () => {
    await act(async () => { root.render(<GovernancePanel />); });
    await flush();
    expect(q('governance-step-outcome-pipeline')).toBeNull();
    expect(q('governance-step-train')).not.toBeNull();
  });

  it('shows one picker with both runs and the pipeline for the selection', async () => {
    await openTrainStep();
    expect(container.querySelectorAll('select#cl-gov-train-run')).toHaveLength(1);
    const options = Array.from(q('governance-train-run-select').options).map((o) => o.value);
    expect(options).toEqual(['lt-new', 'lt-old']);
    expect(q('governance-train-run-select').options[0].textContent).toContain('in pipeline');
    expect(q('governance-train-run-select').options[1].textContent).toContain('ready');
    expect(q('sweep-stub').getAttribute('data-run-id')).toBe('lt-new');
  });

  it('defaults to the newest run even when it is not swept yet, with Train disabled', async () => {
    await openTrainStep();
    expect(q('governance-train-run-select').value).toBe('lt-new');
    expect(q('governance-train-trigger-button').disabled).toBe(true);
  });

  it('enables Train for a ready run and follows the selection into the pipeline', async () => {
    await openTrainStep();
    const select = q('governance-train-run-select');
    await act(async () => {
      select.value = 'lt-old';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await flush();
    expect(q('governance-train-trigger-button').disabled).toBe(false);
    expect(q('sweep-stub').getAttribute('data-run-id')).toBe('lt-old');
  });

  it('enables Train when the selected run lands in the training bucket', async () => {
    await openTrainStep();
    expect(q('governance-train-trigger-button').disabled).toBe(true);

    // The sweep finishes: the backend now lists lt-new as trainable, and the
    // pipeline view reports the transition.
    trainable = { runs: [{ ...NEW_PENDING, target_model_type: 'dlrm_bid_shader', target_variant: 'current' }, OLD_READY], pending: [] };
    const before = calls.filter((c) => c.url.includes('/governance/trainable-runs')).length;
    await act(async () => { sweep.onTrainable({ run_id: 'lt-new', trainable: true }); });
    await flush();

    expect(calls.filter((c) => c.url.includes('/governance/trainable-runs')).length).toBe(before + 1);
    expect(q('governance-train-run-select').value).toBe('lt-new');
    expect(q('governance-train-trigger-button').disabled).toBe(false);
  });

  it('sends the selected run on confirm', async () => {
    trainable = { runs: [OLD_READY], pending: [] };
    await openTrainStep();
    await click(q('governance-train-trigger-button'));
    await click(q('governance-train-confirm-button'));
    await flush();
    const train = calls.find((c) => c.url.endsWith('/api/v1/governance/train'));
    expect(train.body).toMatchObject({ model_type: 'dlrm_bid_shader', run_id: 'lt-old', confirmed: true });
  });

  it('says so when no run has captured outcomes for the model', async () => {
    trainable = { runs: [], pending: [] };
    await openTrainStep();
    expect(q('governance-train-run-select').disabled).toBe(true);
    expect(q('sweep-stub').getAttribute('data-run-id')).toBe('');
    expect(container.textContent).toContain('No load test run for dlrm_bid_shader has captured outcomes yet');
  });
});
