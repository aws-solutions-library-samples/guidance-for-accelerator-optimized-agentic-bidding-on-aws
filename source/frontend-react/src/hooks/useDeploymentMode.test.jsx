/**
 * @vitest-environment jsdom
 *
 * Deployment mode as read from /api/v1/gpu/status. Only an exact "learner" or
 * "standard" is accepted; every other outcome is "unknown" with a reason.
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

import { DeploymentModeProvider, useDeploymentMode, fetchDeploymentMode } from './useDeploymentMode.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

function jsonResp(data, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => data };
}

function textResp({ ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => { throw new SyntaxError('Unexpected token <'); } };
}

let container;
let root;
let seen;

function Probe() {
  const { mode, reason } = useDeploymentMode();
  seen.push({ mode, reason });
  return <span data-testid="probe">{mode}|{reason || ''}</span>;
}

async function renderProvider() {
  await act(async () => {
    root.render(
      <DeploymentModeProvider>
        <Probe />
      </DeploymentModeProvider>,
    );
  });
  await act(async () => {});
  return container.querySelector('[data-testid="probe"]').textContent;
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  seen = [];
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe('fetchDeploymentMode', () => {
  it('reads learner', async () => {
    fetchHandler = async () => jsonResp({ deploymentMode: 'learner', tritonReady: true });
    expect(await fetchDeploymentMode()).toEqual({ mode: 'learner', reason: null });
  });

  it('reads standard', async () => {
    fetchHandler = async () => jsonResp({ deploymentMode: 'standard', desiredSize: 1 });
    expect(await fetchDeploymentMode()).toEqual({ mode: 'standard', reason: null });
  });

  it('treats a missing field as unknown, not standard', async () => {
    fetchHandler = async () => jsonResp({ desiredSize: 1, tritonReady: true });
    expect(await fetchDeploymentMode()).toEqual({
      mode: 'unknown',
      reason: 'deploymentMode missing from response',
    });
  });

  it('takes the mode from a 500 body that carries it', async () => {
    fetchHandler = async () =>
      jsonResp({ error: 'nodegroup not found', deploymentMode: 'standard' }, { ok: false, status: 500 });
    expect(await fetchDeploymentMode()).toEqual({ mode: 'standard', reason: null });
  });

  it('reports the HTTP status when the error body has no mode', async () => {
    fetchHandler = async () => jsonResp({ error: 'Authentication required' }, { ok: false, status: 401 });
    expect(await fetchDeploymentMode()).toEqual({ mode: 'unknown', reason: 'HTTP 401' });
  });

  it('treats a non-JSON body as unknown', async () => {
    fetchHandler = async () => textResp();
    expect(await fetchDeploymentMode()).toEqual({ mode: 'unknown', reason: 'response was not JSON' });
  });

  it('treats a thrown error as unknown with its message', async () => {
    fetchHandler = async () => { throw new Error('network down'); };
    expect(await fetchDeploymentMode()).toEqual({ mode: 'unknown', reason: 'network down' });
  });

  it('passes an unrecognised reported value through as the reason', async () => {
    fetchHandler = async () => jsonResp({ deploymentMode: 'unknown' });
    expect(await fetchDeploymentMode()).toEqual({
      mode: 'unknown',
      reason: 'orchestrator reported deploymentMode "unknown"',
    });
  });

  it('asks the authenticated gpu status endpoint', async () => {
    const urls = [];
    fetchHandler = async (url) => { urls.push(url); return jsonResp({ deploymentMode: 'standard' }); };
    await fetchDeploymentMode();
    expect(urls).toEqual(['/api/v1/gpu/status']);
  });
});

describe('DeploymentModeProvider', () => {
  it('starts as checking and settles on the reported mode', async () => {
    fetchHandler = async () => jsonResp({ deploymentMode: 'learner' });
    const text = await renderProvider();
    expect(seen[0].mode).toBe('checking');
    expect(text).toBe('learner|');
  });

  it('settles on unknown with the reason', async () => {
    fetchHandler = async () => { throw new Error('boom'); };
    const text = await renderProvider();
    expect(text).toBe('unknown|boom');
  });
});

describe('useDeploymentMode outside the provider', () => {
  it('is unknown, never standard', async () => {
    await act(async () => { root.render(<Probe />); });
    expect(seen[seen.length - 1].mode).toBe('unknown');
  });
});
