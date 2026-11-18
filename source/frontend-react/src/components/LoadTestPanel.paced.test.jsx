/**
 * @vitest-environment jsdom
 *
 * FR3.1, FR3.2, FR3.4: the 60K paced preset fixes the duration at 10 minutes,
 * states its rate before the run, and posts duration_s 600. The 10m duration is
 * available to every preset.
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
    if (url.endsWith('/api/v1/loadtest') && method === 'POST') return Promise.resolve(jsonResp({ id: 'lt-paced' }));
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

const q = (id) => container.querySelector(`[data-testid="${id}"]`);
const click = (el) => act(async () => { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); });

async function startRun() {
  await click(container.querySelector('button.loadtest-run'));
  await flush();
  return calls.filter((c) => c.url.endsWith('/api/v1/loadtest') && c.method === 'POST');
}

describe('LoadTestPanel paced preset', () => {
  it('forces 10 minutes, shows the rate line and posts duration_s 600', async () => {
    await act(async () => { root.render(<LoadTestPanel />); });
    await flush();
    expect(q('loadtest-paced-summary')).toBeNull();

    await click(q('loadtest-preset-60k-paced'));
    await flush();

    expect(q('loadtest-duration-600').checked).toBe(true);
    expect(q('loadtest-duration-10').disabled).toBe(true);
    expect(q('loadtest-duration-600').disabled).toBe(true);
    expect(q('loadtest-paced-summary').textContent).toContain('100 req/s for 10 min (60,000 requests)');

    const starts = await startRun();
    expect(starts).toHaveLength(1);
    expect(starts[0].body.preset).toBe('60k-paced');
    expect(starts[0].body.duration_s).toBe(600);
  });

  it('offers 10m to the other presets', async () => {
    await act(async () => { root.render(<LoadTestPanel />); });
    await flush();

    await click(q('loadtest-preset-1k'));
    await click(q('loadtest-duration-600'));
    await flush();
    expect(q('loadtest-duration-600').disabled).toBe(false);
    expect(q('loadtest-paced-summary')).toBeNull();

    const starts = await startRun();
    expect(starts[0].body.preset).toBe('1k');
    expect(starts[0].body.duration_s).toBe(600);
  });
});
