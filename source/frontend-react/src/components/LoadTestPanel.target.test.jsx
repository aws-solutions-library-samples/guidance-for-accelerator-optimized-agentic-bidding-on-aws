/**
 * @vitest-environment jsdom
 *
 * The "Capture outcomes for" select must start on a REAL model value.
 *
 * Regression: the state defaulted to "" while the select's empty ("None")
 * option was commented out. With no option matching "", the browser paints the
 * first option ("Bid Pricer") — so the UI looked like Bid Pricer was selected
 * while target_model_type was never sent. Every run then captured no outcomes
 * and was silently ineligible for comparison/training.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

let fetchHandler;
vi.mock('../authFetch.js', () => ({
  authFetch: (...args) => fetchHandler(...args),
}));

import LoadTestPanel from './LoadTestPanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

let calls;
let container;
let root;

beforeEach(() => {
  calls = [];
  fetchHandler = (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });
    if (url.includes('/loadtest/history')) return Promise.resolve(jsonResp({ runs: [] }));
    if (url.endsWith('/api/v1/loadtest') && method === 'POST') {
      return Promise.resolve(jsonResp({ id: 'lt-test' }));
    }
    return Promise.resolve(jsonResp({}));
  };
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

async function flush(times = 4) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  }
}

describe('LoadTestPanel outcome-capture target default', () => {
  it('defaults the select value to a real trainable model, not empty', async () => {
    await act(async () => { root.render(<LoadTestPanel />); });
    await flush();

    const select = container.querySelector('[data-testid="loadtest-target-model-select"]');
    expect(select).toBeTruthy();
    // The displayed option and the underlying value must agree.
    expect(select.value).toBe('dlrm_bid_shader');
    expect(select.value).not.toBe('');
  });

  it('sends target_model_type on start so outcomes are actually captured', async () => {
    await act(async () => { root.render(<LoadTestPanel />); });
    await flush();

    const startBtn = container.querySelector('button.loadtest-run');
    expect(startBtn).toBeTruthy();

    await act(async () => {
      startBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await flush();

    const starts = calls.filter((c) => c.url.endsWith('/api/v1/loadtest') && c.method === 'POST');
    expect(starts).toHaveLength(1);
    expect(starts[0].body.target_model_type).toBe('dlrm_bid_shader');
    expect(starts[0].body.target_variant).toBe('current');
  });
});
