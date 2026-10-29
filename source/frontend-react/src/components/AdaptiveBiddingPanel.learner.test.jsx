/**
 * @vitest-environment jsdom
 *
 * AdaptiveBiddingPanel when the Adaptive Bidding agent is not deployed: in a
 * learner-mode deployment the agent node says it is not deployed in learner
 * mode; in a standard deployment the existing message is unchanged.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

let mode = 'learner';
vi.mock('../hooks/useDeploymentMode.jsx', () => ({
  useDeploymentMode: () => ({ mode, reason: null, retry: () => {} }),
}));
const calls = [];
vi.mock('../authFetch.js', () => ({
  authFetch: async (url) => {
    calls.push(url);
    if (url.startsWith('/api/v1/closed-loop/scenarios')) {
      return jsonResp({ scenarios: [{ key: 'low_win_rate', label: 'Low win rate', model_type: 'dlrm_bid_shader' }] });
    }
    if (url.startsWith('/api/v1/closed-loop/generate')) {
      return jsonResp({ context: { before: {}, market_state: { win_rate: 0.1, roi: 1.2, total_bids: 10 } }, generated: [{ ok: true }] });
    }
    if (url.startsWith('/api/v1/closed-loop/parameters')) return jsonResp({ parameters: [] });
    if (url.startsWith('/api/v1/closed-loop/audit')) return jsonResp({ records: [] });
    return jsonResp({}, { ok: false, status: 404 });
  },
  isProxyTransport: () => false,
}));
vi.mock('../awsCredentials', () => ({
  COGNITO_REGION: 'us-east-1',
  getSigV4Credentials: async () => ({}),
  isIdentityConfigured: () => true,
}));

import AdaptiveBiddingPanel from './AdaptiveBiddingPanel.jsx';
import { agentUnavailableReason, ADAPTIVE_LEARNER_UNAVAILABLE } from '../agentCoreClient.js';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

let container;
let root;
beforeEach(() => {
  calls.length = 0;
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

async function runScenario() {
  await act(async () => { root.render(<AdaptiveBiddingPanel />); });
  await act(async () => {});
  const run = [...container.querySelectorAll('button')].find((b) => b.textContent === 'Run scenario');
  expect(run).toBeDefined();
  await act(async () => { run.click(); });
  // The flow reveals one node every 420 ms; the agent node is the second.
  await act(async () => { await new Promise((r) => setTimeout(r, 500)); });
  return container.querySelector('.cl-flow-details').textContent;
}

describe('agentUnavailableReason', () => {
  it('names learner mode for the Adaptive Bidding agent only', () => {
    expect(agentUnavailableReason('adaptive', { deploymentMode: 'learner' })).toBe(ADAPTIVE_LEARNER_UNAVAILABLE);
    expect(ADAPTIVE_LEARNER_UNAVAILABLE).toContain('not deployed in learner mode');
    expect(agentUnavailableReason('adaptive', { deploymentMode: 'standard' })).toContain('Agent not deployed');
    expect(agentUnavailableReason('adaptive')).toContain('Agent not deployed');
    expect(agentUnavailableReason('governance', { deploymentMode: 'learner' })).toContain('Agent not deployed');
  });
});

describe('AdaptiveBiddingPanel with no Adaptive Bidding agent', () => {
  it('says the agent is not deployed in learner mode', async () => {
    mode = 'learner';
    const text = await runScenario();
    expect(text).toContain(ADAPTIVE_LEARNER_UNAVAILABLE);
    expect(calls.some((u) => u.startsWith('/api/v1/closed-loop/generate'))).toBe(true);
  });

  it('keeps the existing message in a standard deployment', async () => {
    mode = 'standard';
    const text = await runScenario();
    expect(text).toContain('Agent not deployed');
    expect(text).not.toContain('learner mode');
  });
});
