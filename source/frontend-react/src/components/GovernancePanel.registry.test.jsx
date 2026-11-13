/**
 * @vitest-environment jsdom
 *
 * Model registry card in the Governance panel: which model it reads, when it
 * re-reads, and how it reacts to a training trigger.
 *
 * The regression that prompted this: the registry table was fed by the
 * test-versions / scenario selector, so it silently showed whichever model that
 * selector was on. It now has its own selector, polls only while a training
 * job for that model is active, and re-reads right after a successful trigger
 * so the new job appears without a click.
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
vi.mock('./LoadTestSweepStatus.jsx', () => ({ default: () => null }));

import GovernancePanel, { REGISTRY_POLL_MS } from './GovernancePanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

const DLRM_VERSION = {
  version: 1, approval_status: 'Approved', status: 'Completed',
  created_at: '2026-10-04T20:07:09+00:00',
  model_package_arn: 'arn:aws:sagemaker:us-east-1:123456789012:model-package/dlrm/1',
};
const FLOOR_VERSION = {
  version: 3, approval_status: 'Approved', status: 'Completed',
  created_at: '2026-10-03T09:00:00+00:00',
  model_package_arn: 'arn:aws:sagemaker:us-east-1:123456789012:model-package/floor/3',
};
const RUNNING_JOB = {
  job_name: 'dlrm-bid-shader-1759690000-ab12cd34', status: 'InProgress',
  secondary_status: 'Training', created_at: '2026-10-05T18:01:00+00:00',
  model_type: 'dlrm_bid_shader',
};
const TRAINABLE_RUN = {
  id: 'lt-run-1', timestamp: '2026-10-05T17:00:00Z',
  target_model_type: 'dlrm_bid_shader', target_variant: 'current',
};

let calls;
// Per-model registry payloads the fake backend returns; tests mutate this to
// simulate a job appearing or finishing between reads.
let registry;

function handler(url, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();
  calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });

  if (url.includes('/closed-loop/scenarios')) return Promise.resolve(jsonResp({ scenarios: [] }));
  if (url.includes('/closed-loop/models')) {
    const modelType = new URL(url, 'http://x').searchParams.get('model_type');
    return Promise.resolve(jsonResp(registry[modelType] || { versions: [], training_jobs: [] }));
  }
  if (url.includes('/governance/eligible-runs')) return Promise.resolve(jsonResp({ runs: [], most_recent: null }));
  if (url.includes('/governance/comparison-pair')) return Promise.resolve(jsonResp({}));
  if (url.includes('/governance/training-estimate')) {
    return Promise.resolve(jsonResp({
      instance_type: 'ml.g5.2xlarge', hourly_rate_usd: 1.515,
      max_runtime_seconds: 14400, estimated_max_cost_usd: 6.06,
    }));
  }
  if (url.includes('/governance/trainable-runs')) {
    return Promise.resolve(jsonResp({ runs: [TRAINABLE_RUN], etl: { configured: true }, pending: [] }));
  }
  if (url.endsWith('/api/v1/governance/train') && method === 'POST') {
    return Promise.resolve(jsonResp({ job_name: RUNNING_JOB.job_name, base_model_version: '1' }));
  }
  return Promise.resolve(jsonResp({}));
}

async function flush(times = 6) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  }
}

function modelsCalls() {
  return calls
    .filter((c) => c.url.includes('/closed-loop/models'))
    .map((c) => new URL(c.url, 'http://x').searchParams.get('model_type'));
}

function click(el) {
  return act(async () => { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
}

let container;
let root;

beforeEach(() => {
  calls = [];
  registry = {
    dlrm_bid_shader: { versions: [DLRM_VERSION], training_jobs: [] },
    deal_yield_manager_floor: { versions: [FLOOR_VERSION], training_jobs: [] },
  };
  fetchHandler = handler;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('Governance model registry card', () => {
  it('loads on mount for its own selector and names the model in the title', async () => {
    await act(async () => { root.render(<GovernancePanel />); });
    await flush();

    expect(modelsCalls()).toEqual(['dlrm_bid_shader']);
    const title = container.querySelector('[data-testid="models-view-title"]');
    expect(title.textContent).toBe('Model registry versions · Bid Pricer (DLRM) · SageMaker');
    expect(container.querySelector('[data-testid="registry-model-select"]').value).toBe('dlrm_bid_shader');
  });

  it('re-reads for the newly selected model and leaves the other selectors alone', async () => {
    await act(async () => { root.render(<GovernancePanel />); });
    await flush();

    const select = container.querySelector('[data-testid="registry-model-select"]');
    await act(async () => {
      select.value = 'deal_yield_manager_floor';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await flush();

    expect(modelsCalls()).toEqual(['dlrm_bid_shader', 'deal_yield_manager_floor']);
    expect(container.querySelector('[data-testid="models-view-title"]').textContent)
      .toContain('Yield Optimizer — Floor (XGBoost)');
    expect(container.textContent).toContain('2026-10-03T09:00:00+00:00');
    // The registry selector must not drag the test-versions / scenario selector with it.
    expect(container.querySelector('#cl-gov-model').value).toBe('dlrm_bid_shader');
  });

  it('Refresh re-reads the same model', async () => {
    await act(async () => { root.render(<GovernancePanel />); });
    await flush();
    await click(container.querySelector('[data-testid="registry-refresh-button"]'));
    await flush();
    expect(modelsCalls()).toEqual(['dlrm_bid_shader', 'dlrm_bid_shader']);
  });

  it('shows an active job and polls every REGISTRY_POLL_MS until it is gone', async () => {
    vi.useFakeTimers();
    registry.dlrm_bid_shader = { versions: [DLRM_VERSION], training_jobs: [RUNNING_JOB] };

    await act(async () => { root.render(<GovernancePanel />); });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });

    expect(container.querySelector('[data-testid="training-job-row"]')).toBeTruthy();
    expect(container.querySelector('[data-testid="training-job-badge"]').textContent)
      .toBe('Training in progress');
    expect(modelsCalls()).toEqual(['dlrm_bid_shader']);

    // One minute later: one more read, the job is still running.
    await act(async () => { await vi.advanceTimersByTimeAsync(REGISTRY_POLL_MS); });
    expect(modelsCalls()).toHaveLength(2);

    // The job finishes and registers version 2. The next poll picks that up
    // and, with nothing active, the interval stops.
    registry.dlrm_bid_shader = {
      versions: [{ ...DLRM_VERSION, version: 2 }, DLRM_VERSION],
      training_jobs: [],
    };
    await act(async () => { await vi.advanceTimersByTimeAsync(REGISTRY_POLL_MS); });
    expect(modelsCalls()).toHaveLength(3);
    expect(container.querySelector('[data-testid="training-job-row"]')).toBeNull();

    await act(async () => { await vi.advanceTimersByTimeAsync(REGISTRY_POLL_MS * 3); });
    expect(modelsCalls()).toHaveLength(3);
  });

  it('does not poll when no job is active', async () => {
    vi.useFakeTimers();
    await act(async () => { root.render(<GovernancePanel />); });
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(modelsCalls()).toHaveLength(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(REGISTRY_POLL_MS * 2); });
    expect(modelsCalls()).toHaveLength(1);
  });

  it('re-reads the registry right after a successful training trigger', async () => {
    await act(async () => { root.render(<GovernancePanel />); });
    await flush();
    expect(modelsCalls()).toEqual(['dlrm_bid_shader']);

    await click(container.querySelector('[data-testid="governance-step-train"]'));
    await flush();
    const trigger = container.querySelector('[data-testid="governance-train-trigger-button"]');
    expect(trigger.disabled).toBe(false);
    await click(trigger);
    await flush();

    // The backend now reports the job the trigger just created.
    registry.dlrm_bid_shader = { versions: [DLRM_VERSION], training_jobs: [RUNNING_JOB] };
    await click(container.querySelector('[data-testid="governance-train-confirm-button"]'));
    await flush();

    const train = calls.find((c) => c.url.endsWith('/api/v1/governance/train'));
    expect(train.body).toMatchObject({ model_type: 'dlrm_bid_shader', run_id: 'lt-run-1', confirmed: true });
    expect(modelsCalls()).toEqual(['dlrm_bid_shader', 'dlrm_bid_shader']);
    expect(container.querySelector('[data-testid="training-job-row"]').textContent)
      .toContain(RUNNING_JOB.job_name);
  });
});
