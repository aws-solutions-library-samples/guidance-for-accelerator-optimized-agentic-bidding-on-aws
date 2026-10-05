/**
 * @vitest-environment jsdom
 *
 * A single non-OK poll must not end the live view.
 *
 * Behind the internal NLB the poll can reach an orchestrator replica that does
 * not own the test. The orchestrator now forwards to the owner, but a replica
 * that joined mid-run can still miss for a tick, and before this change one
 * 404 reset the panel to idle one second after Run with nothing on screen
 * on a two-replica deployment. The panel gives up only after POLL_MISS_LIMIT
 * consecutive misses, and a good poll resets the count.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

let fetchHandler;
vi.mock('../authFetch.js', () => ({
  authFetch: (...args) => fetchHandler(...args),
  isProxyTransport: () => true,
}));

import LoadTestPanel, { POLL_INTERVAL_MS, POLL_MISS_LIMIT } from './LoadTestPanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data, body: null };
}

const RUNNING = (completed) => ({
  state: 'running', completed, total_requests: 100, rps: 20, elapsed_ms: completed * 50,
  latency_p50: 8, latency_p95: 20, latency_p99: 30, errors: 0,
});
const MISS = () => jsonResp({ error: 'Load test not found' }, { ok: false, status: 404 });

let container;
let root;
let pollReplies;

beforeEach(() => {
  vi.useFakeTimers();
  fetchHandler = (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    if (url.endsWith('/api/v1/loadtest') && method === 'POST') {
      return Promise.resolve(jsonResp({ id: 'lt-miss' }));
    }
    if (url === '/api/v1/loadtest/lt-miss' && method === 'GET') {
      const next = pollReplies.shift();
      return Promise.resolve(next ? next() : jsonResp(RUNNING(99)));
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
  await act(async () => { startBtn.dispatchEvent(new MouseEvent('click', { bubbles: true })); });
  await flushMicrotasks();
}

describe('LoadTestPanel poll miss tolerance', () => {
  it('tolerates a single 404 and keeps the live view when the next poll is good', async () => {
    pollReplies = [
      () => jsonResp(RUNNING(10)),
      MISS,
      () => jsonResp(RUNNING(30)),
    ];
    const seen = [];
    await startRun((s) => seen.push(s));

    await tick(POLL_INTERVAL_MS); // completed 10
    expect(seen[seen.length - 1].progress.completed).toBe(10);

    await tick(POLL_INTERVAL_MS); // 404
    let latest = seen[seen.length - 1];
    expect(latest.running).toBe(true);
    expect(latest.progress.completed).toBe(10);
    expect(container.querySelector('button.loadtest-stop')).toBeTruthy();

    await tick(POLL_INTERVAL_MS); // completed 30
    latest = seen[seen.length - 1];
    expect(latest.running).toBe(true);
    expect(latest.progress.completed).toBe(30);
  });

  it(`resets to idle after ${POLL_MISS_LIMIT} consecutive misses`, async () => {
    pollReplies = [() => jsonResp(RUNNING(10))];
    for (let i = 0; i < POLL_MISS_LIMIT; i++) pollReplies.push(MISS);
    const seen = [];
    await startRun((s) => seen.push(s));

    await tick(POLL_INTERVAL_MS); // completed 10
    for (let i = 0; i < POLL_MISS_LIMIT - 1; i++) {
      // eslint-disable-next-line no-await-in-loop
      await tick(POLL_INTERVAL_MS);
      expect(seen[seen.length - 1].running).toBe(true);
    }
    await tick(POLL_INTERVAL_MS); // the limit-th miss
    expect(seen[seen.length - 1].running).toBe(false);
    expect(container.querySelector('button.loadtest-run')).toBeTruthy();
  });

  it('exposes the miss limit', () => {
    expect(POLL_MISS_LIMIT).toBe(3);
  });
});
