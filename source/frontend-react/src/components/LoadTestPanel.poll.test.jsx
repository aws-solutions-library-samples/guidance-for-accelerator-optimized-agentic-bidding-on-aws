/**
 * @vitest-environment jsdom
 *
 * Progress transport through the UI API proxy Lambda.
 *
 * The proxy is a synchronous Invoke and cannot carry the SSE stream (it answers
 * 406), so in proxy builds the panel must not request /stream at all and must
 * poll GET /api/v1/loadtest/{id} every POLL_INTERVAL_MS, forwarding the live
 * counters the orchestrator merges into that body. Regression: with the
 * 2 s fallback and a poll body that only ever said completed: 0, the stats
 * panel showed zeros for the whole run and no bid bubbles rose.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

let fetchHandler;
let proxyTransport = true;
vi.mock('../authFetch.js', () => ({
  authFetch: (...args) => fetchHandler(...args),
  isProxyTransport: () => proxyTransport,
}));

import LoadTestPanel, { POLL_INTERVAL_MS } from './LoadTestPanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data, body: null };
}

let calls;
let container;
let root;
let pollBodies;

beforeEach(() => {
  vi.useFakeTimers();
  proxyTransport = true;
  calls = [];
  // Successive poll replies: the orchestrator's live counters climbing.
  pollBodies = [
    { state: 'running', completed: 40, total_requests: 1000, rps: 38.5, elapsed_ms: 1040,
      latency_p50: 8.1, latency_p95: 20.2, latency_p99: 31.0, errors: 0 },
    { state: 'running', completed: 85, total_requests: 1000, rps: 41.0, elapsed_ms: 2070,
      latency_p50: 8.4, latency_p95: 21.0, latency_p99: 33.5, errors: 1 },
  ];
  fetchHandler = (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    calls.push({ url, method, headers: opts.headers || {} });
    if (url.endsWith('/api/v1/loadtest') && method === 'POST') {
      return Promise.resolve(jsonResp({ id: 'lt-poll' }));
    }
    if (url.endsWith('/stream')) {
      return Promise.resolve(jsonResp({ error: 'streaming_not_supported' }, { ok: false, status: 406 }));
    }
    if (url === '/api/v1/loadtest/lt-poll' && method === 'GET') {
      const body = pollBodies.shift() || { state: 'complete', completed: 1000, total_requests: 1000 };
      return Promise.resolve(jsonResp(body));
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
  vi.useRealTimers();
  vi.clearAllMocks();
});

async function flushMicrotasks(times = 4) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await Promise.resolve(); });
  }
}

async function tick(ms) {
  await act(async () => { vi.advanceTimersByTime(ms); });
  await flushMicrotasks();
}

async function startRun(onResultChange) {
  await act(async () => { root.render(<LoadTestPanel onResultChange={onResultChange} />); });
  await flushMicrotasks();
  const startBtn = container.querySelector('button.loadtest-run');
  expect(startBtn).toBeTruthy();
  await act(async () => {
    startBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
  await flushMicrotasks();
}

describe('LoadTestPanel progress transport behind the UI API proxy', () => {
  it('polls at POLL_INTERVAL_MS and never requests the SSE stream', async () => {
    await startRun();

    expect(calls.some((c) => c.url.endsWith('/stream'))).toBe(false);
    const pollsBefore = calls.filter((c) => c.url === '/api/v1/loadtest/lt-poll' && c.method === 'GET');
    expect(pollsBefore).toHaveLength(0);

    await tick(POLL_INTERVAL_MS);
    let polls = calls.filter((c) => c.url === '/api/v1/loadtest/lt-poll' && c.method === 'GET');
    expect(polls).toHaveLength(1);

    await tick(POLL_INTERVAL_MS);
    polls = calls.filter((c) => c.url === '/api/v1/loadtest/lt-poll' && c.method === 'GET');
    expect(polls).toHaveLength(2);
    expect(calls.some((c) => c.url.endsWith('/stream'))).toBe(false);
  });

  it('exposes a 1 s poll interval', () => {
    expect(POLL_INTERVAL_MS).toBe(1000);
  });

  it('forwards the live counters from each poll as progress', async () => {
    const seen = [];
    await startRun((s) => seen.push(s));

    await tick(POLL_INTERVAL_MS);
    let latest = seen[seen.length - 1];
    expect(latest.running).toBe(true);
    expect(latest.progress).toMatchObject({ completed: 40, total: 1000, rps: 38.5, errors: 0 });

    await tick(POLL_INTERVAL_MS);
    latest = seen[seen.length - 1];
    expect(latest.progress).toMatchObject({ completed: 85, total: 1000, rps: 41.0, errors: 1 });
    expect(latest.progress.latency_p99).toBe(33.5);
  });

  it('stops polling and surfaces the result when the body says complete', async () => {
    const seen = [];
    await startRun((s) => seen.push(s));
    await tick(POLL_INTERVAL_MS); // completed 40
    await tick(POLL_INTERVAL_MS); // completed 85
    await tick(POLL_INTERVAL_MS); // complete
    const latest = seen[seen.length - 1];
    expect(latest.running).toBe(false);
    expect(latest.result).toMatchObject({ state: 'complete', completed: 1000 });

    const pollsAtComplete = calls.filter((c) => c.url === '/api/v1/loadtest/lt-poll' && c.method === 'GET').length;
    await tick(POLL_INTERVAL_MS * 2);
    const pollsAfter = calls.filter((c) => c.url === '/api/v1/loadtest/lt-poll' && c.method === 'GET').length;
    expect(pollsAfter).toBe(pollsAtComplete);
  });

  it('still tries the SSE stream first when not behind the proxy', async () => {
    proxyTransport = false;
    await startRun();
    expect(calls.some((c) => c.url === '/api/v1/loadtest/lt-poll/stream')).toBe(true);
    // The 406 reply falls back to polling, as before.
    await tick(POLL_INTERVAL_MS);
    const polls = calls.filter((c) => c.url === '/api/v1/loadtest/lt-poll' && c.method === 'GET');
    expect(polls).toHaveLength(1);
  });
});
