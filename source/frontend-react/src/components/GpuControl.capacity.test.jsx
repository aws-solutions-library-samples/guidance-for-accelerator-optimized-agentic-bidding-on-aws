/**
 * @vitest-environment jsdom
 *
 * The orchestrator reports tritonState "no_capacity" when the GPU node group's
 * last launch failed with InsufficientInstanceCapacity. The panel says so with
 * EC2's reason, instead of "starting (Triton loading...)", and keeps Start
 * enabled so the same size can be retried.
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
import GpuControl from './GpuControl.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

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

async function mount() {
  await act(async () => { root.render(<GpuControl />); });
  await act(async () => {});
}

const DETAIL = "The GPU node group's last launch failed: Could not launch On-Demand Instances. "
  + "InsufficientInstanceCapacity - We currently do not have sufficient g6.xlarge capacity in the "
  + "Availability Zone you requested (us-west-2a). Start GPUs retries the same instance size. "
  + "Re-running deploy.sh for this prefix tries the other sizes in deployment/eks/cluster-config.yaml.";

describe('GpuControl with no GPU capacity', () => {
  it('shows no GPU capacity with the EC2 reason and leaves Start enabled', async () => {
    fetchHandler = async () => jsonResp({
      deploymentMode: 'standard', status: 'DEGRADED', desiredSize: 1, minSize: 1, maxSize: 3,
      running: false, tritonReady: false, tritonState: 'no_capacity', tritonDetail: DETAIL,
    });
    await mount();
    expect(container.querySelector('.gpu-status-badge').textContent).toBe('no GPU capacity');
    expect(container.querySelector('[data-testid="gpu-control-blocked-detail"]').textContent).toBe(DETAIL);
    const start = [...container.querySelectorAll('button')].find((b) => b.textContent === 'Start GPUs');
    expect(start.disabled).toBe(false);
  });
});
