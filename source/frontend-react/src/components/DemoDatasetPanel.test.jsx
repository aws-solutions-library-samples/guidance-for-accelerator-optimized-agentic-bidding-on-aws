/**
 * @vitest-environment jsdom
 *
 * DemoDatasetPanel (FR1.1, FR1.5, FR1.6, FR1.11, NFR1): the demonstration-only
 * notice, the estimate from the API, the Confirm step before generation, and
 * the API's refusal message.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

let fetchHandler;
vi.mock('../authFetch.js', () => ({
  authFetch: (...args) => fetchHandler(...args),
  isProxyTransport: () => false,
}));

import DemoDatasetPanel from './DemoDatasetPanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

const DLRM_PARAMS = [
  { name: 'win_rate', label: 'Win rate', minimum: 0.1, maximum: 1.0, default: 0.9 },
  { name: 'click_rate', label: 'Click rate given a win', minimum: 0.005, maximum: 0.5, default: 0.02 },
  { name: 'conversion_rate', label: 'Conversion rate given a click', minimum: 0.01, maximum: 1.0, default: 0.05 },
];
const STATUS = {
  model_type: 'dlrm_bid_shader', prefix: 'demo-training-data/', params: DLRM_PARAMS,
  row_cap: 100000, demonstration_only: true, exists: false, metadata: null,
};
const SIZING = {
  model_type: 'dlrm_bid_shader', rows: 55556, row_cap: 100000, capped: false,
  positive_kind: 'conversions', target_positives: 50, expected_positives: 50.0,
  expected_wins: 50000.4, expected_clicks: 1000.01,
};

let calls;
let generateResponse;
let container;
let root;

beforeEach(() => {
  calls = [];
  generateResponse = jsonResp({
    ...STATUS, exists: true,
    metadata: {
      generated_at: '2026-11-15T10:00:00Z', rows_written: 55556, actual_positives: 48, seed: 20261115,
      params: { win_rate: 0.9, click_rate: 0.02, conversion_rate: 0.05 },
      sizing: SIZING,
    },
  }, { status: 201 });
  fetchHandler = (url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase();
    calls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });
    if (url.includes('/demo-dataset/estimate')) return Promise.resolve(jsonResp({ sizing: SIZING }));
    if (url.includes('/demo-dataset') && method === 'POST') return Promise.resolve(generateResponse);
    if (url.includes('/demo-dataset')) return Promise.resolve(jsonResp(STATUS));
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

async function flush(times = 6) {
  for (let i = 0; i < times; i++) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
  }
}

const q = (id) => container.querySelector(`[data-testid="${id}"]`);
const click = (el) => act(async () => { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); });

async function renderPanel(onStatusChange = () => {}) {
  await act(async () => { root.render(<DemoDatasetPanel modelType="dlrm_bid_shader" onStatusChange={onStatusChange} />); });
  await flush();
}

describe('DemoDatasetPanel', () => {
  it('shows the demonstration-only notice and "No demo dataset"', async () => {
    const onStatusChange = vi.fn();
    await renderPanel(onStatusChange);
    expect(q('demo-dataset-notice').className).toContain('cl-honest');
    expect(q('demo-dataset-notice').textContent).toContain('Demonstration only');
    expect(q('demo-dataset-status').textContent).toBe('No demo dataset');
    expect(onStatusChange).toHaveBeenCalledWith(expect.objectContaining({ exists: false }));
  });

  it('renders the notice with cl-honest inside the cl-demo-dataset panel (NFR1)', async () => {
    await renderPanel();
    const notice = q('demo-dataset-notice');
    expect(notice.classList.contains('cl-honest')).toBe(true);
    expect(notice.closest('.cl-demo-dataset')).not.toBeNull();
  });

  it('index.css darkens cl-honest inside cl-demo-dataset to #92400e (NFR1, 6.04:1 on #F1F5F9)', () => {
    const cssPath = resolve(dirname(fileURLToPath(import.meta.url)), '../index.css');
    const css = readFileSync(cssPath, 'utf8');
    const rule = css.match(/\.cl-demo-dataset\s+\.cl-honest\s*\{([^}]*)\}/);
    expect(rule).not.toBeNull();
    expect(rule[1]).toMatch(/color:\s*#92400e\s*;?/i);
  });

  it('shows the API estimate for the default rates', async () => {
    await renderPanel();
    expect(q('demo-dataset-param-win-rate').value).toBe('90');
    expect(q('demo-dataset-estimate').textContent).toContain('55,556 rows');
    expect(q('demo-dataset-estimate').textContent).toContain('50 conversions');
    const estimate = calls.find((c) => c.url.includes('/estimate'));
    expect(estimate.url).toContain('win_rate=0.9');
    expect(estimate.url).toContain('click_rate=0.02');
  });

  it('requires Confirm before posting, and posts confirmed: true', async () => {
    const onStatusChange = vi.fn();
    await renderPanel(onStatusChange);
    await click(q('demo-dataset-generate-button'));
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
    await click(q('demo-dataset-confirm-button'));
    await flush();

    const post = calls.find((c) => c.method === 'POST');
    expect(post.body).toEqual({
      model_type: 'dlrm_bid_shader',
      params: { win_rate: 0.9, click_rate: 0.02, conversion_rate: 0.05 },
      confirmed: true,
    });
    expect(q('demo-dataset-result').textContent).toContain('55,556 rows');
    expect(q('demo-dataset-result').textContent).toContain('Synthetic');
    expect(q('demo-dataset-status').textContent).toContain('2026-11-15T10:00:00Z');
    expect(onStatusChange).toHaveBeenLastCalledWith(expect.objectContaining({ exists: true }));
  });

  it('Cancel posts nothing', async () => {
    await renderPanel();
    await click(q('demo-dataset-generate-button'));
    await click(q('demo-dataset-cancel-button'));
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
    expect(q('demo-dataset-generate-button')).not.toBeNull();
  });

  it('shows the API 422 message', async () => {
    generateResponse = jsonResp(
      { error: "'label_conversion' has no positives across 100000 labelled rows", reason: 'dataset_unfit' },
      { ok: false, status: 422 },
    );
    await renderPanel();
    await click(q('demo-dataset-generate-button'));
    await click(q('demo-dataset-confirm-button'));
    await flush();
    expect(q('demo-dataset-error').textContent).toContain('has no positives');
    expect(q('demo-dataset-result')).toBeNull();
  });

  it('disables Generate for a rate outside its bound', async () => {
    await renderPanel();
    const input = q('demo-dataset-param-click-rate');
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      setter.call(input, '80');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await flush();
    expect(q('demo-dataset-generate-button').disabled).toBe(true);
    expect(q('demo-dataset-params-invalid')).not.toBeNull();
  });
});
