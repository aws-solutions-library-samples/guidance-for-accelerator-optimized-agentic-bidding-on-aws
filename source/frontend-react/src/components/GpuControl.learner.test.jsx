/**
 * @vitest-environment jsdom
 *
 * GpuControl when the orchestrator reports learner mode: no GPU node group, so
 * no Start/Stop, and the state shown is Triton readiness on CPU.
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

const q = (id) => container.querySelector(`[data-testid="${id}"]`);
const buttons = () => [...container.querySelectorAll('button')].map((b) => b.textContent);

describe('GpuControl in learner mode', () => {
  it('has no Start/Stop buttons and shows Triton ready', async () => {
    fetchHandler = async () => jsonResp({ deploymentMode: 'learner', tritonReady: true, tritonState: 'ready' });
    await mount();
    expect(q('gpu-control-learner')).not.toBeNull();
    expect(buttons()).not.toContain('Start GPUs');
    expect(buttons()).not.toContain('Stop GPUs');
    expect(q('gpu-control-learner-status-badge').textContent).toBe('Triton ready');
    expect(container.textContent).toContain('Inference (CPU, learner mode)');
  });

  it('shows Triton not ready with the probe detail', async () => {
    fetchHandler = async () => jsonResp({
      deploymentMode: 'learner',
      tritonReady: false,
      tritonState: 'not_ready',
      tritonDetail: 'ConnectError connecting to triton-inference-server:8000',
    });
    await mount();
    expect(q('gpu-control-learner-status-badge').textContent).toBe('Triton not ready');
    expect(q('gpu-control-learner-detail').textContent).toContain('ConnectError');
  });
});

describe('GpuControl in standard mode', () => {
  it('keeps the Start/Stop buttons', async () => {
    fetchHandler = async () => jsonResp({
      deploymentMode: 'standard', desiredSize: 0, tritonReady: false, tritonState: 'stopped', status: 'ACTIVE',
    });
    await mount();
    expect(q('gpu-control-learner')).toBeNull();
    expect(buttons()).toEqual(['Start GPUs', 'Stop GPUs']);
  });
});
