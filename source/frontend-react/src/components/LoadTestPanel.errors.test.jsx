/**
 * @vitest-environment jsdom
 *
 * Error reporting. orchestrator/loadtest.py counts `errors` per REQUEST (a request
 * with at least one failed container call), so it is never larger than
 * `completed`, and reports each container's failed calls in per_container[].errors,
 * also while the test runs. Before, every failed container call was added to the
 * request count: 1,890 errors for 1,818 sent, Received -72, Error Rate 103.96%.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import { LoadTestResults } from './LoadTestPanel.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const RUNNING = {
  completed: 1818,
  total: 10000,
  errors: 1702,
  rps: 9.17,
  latency_p50: 5009.1,
  latency_p95: 5342.3,
  latency_p99: 5548.5,
  per_container: [
    { name: 'dlrm-bid-shader', avg_latency_ms: 12.4, total_mutations: 0, errors: 0 },
    { name: 'ncf-deal-manager', avg_latency_ms: 4980.2, total_mutations: 310, errors: 1650 },
    { name: 'yield-optimizer-floor', avg_latency_ms: 2210.0, total_mutations: 88, errors: 512 },
  ],
};

let container;
let root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function metric(label) {
  const item = [...container.querySelectorAll('.loadtest-metric')]
    .find((el) => el.querySelector('.loadtest-metric-label')?.textContent === label);
  return item?.querySelector('.loadtest-metric-value')?.textContent;
}

function row(displayName) {
  const table = container.querySelector('table[aria-label="Per-container load test results"]');
  return [...table.querySelectorAll('tbody tr')]
    .find((tr) => tr.querySelector('.loadtest-breakdown-name')?.textContent === displayName);
}

describe('LoadTestResults error reporting', () => {
  it('shows Received and Error Rate from request-level errors', () => {
    act(() => root.render(<LoadTestResults progress={RUNNING} result={null} running={true} error={null} />));
    expect(metric('Received')).toBe('116');
    expect(metric('Errors')).toBe('1,702');
    expect(metric('Error Rate')).toBe('93.62%');
  });

  it("shows each container's failed calls while the test runs", () => {
    act(() => root.render(<LoadTestResults progress={RUNNING} result={null} running={true} error={null} />));
    const header = [...container.querySelectorAll('.loadtest-breakdown-table th')].map((th) => th.textContent);
    expect(header).toContain('Failed calls');
    const cells = [...row('Deal Scorer').querySelectorAll('td')].map((td) => td.textContent);
    expect(cells).toContain('4980.2ms');
    expect(cells).toContain('1,650');
    const pricer = [...row('Bid Pricer').querySelectorAll('td')].map((td) => td.textContent);
    expect(pricer).toContain('0');
  });
});
