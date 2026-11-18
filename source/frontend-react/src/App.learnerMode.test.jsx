/**
 * @vitest-environment jsdom
 *
 * App wiring for learner mode: the notice appears on every load of a learner
 * deployment and never otherwise, the header badge outlives the notice, and the
 * Governance view cannot be reached.
 *
 * The panels App composes are stubbed; this test is about the mode wiring, and
 * the real Header and LearnerModeNotice are rendered.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

let fetchHandler;
vi.mock('./authFetch.js', () => ({
  authFetch: (...args) => fetchHandler(...args),
  isProxyTransport: () => false,
}));
vi.mock('./auth', () => ({
  isAuthConfigured: () => false,
  signOut: () => {},
  getCurrentUserEmail: async () => null,
}));
vi.mock('./components/AuctionTheater', () => ({ default: () => null }));
vi.mock('./components/Sidebar', () => ({ default: () => <div data-testid="stub-sidebar" /> }));
vi.mock('./components/RawPanel', () => ({ default: () => null }));
vi.mock('./components/ContainersPanel', () => ({ default: () => null }));
vi.mock('./components/AdaptiveBiddingPanel', () => ({ default: () => <div data-testid="stub-adaptive" /> }));
vi.mock('./components/GovernancePanel', () => ({ default: () => <div data-testid="stub-governance" /> }));
vi.mock('./components/LoadTestPanel', () => ({ LoadTestResults: () => null }));
vi.mock('./components/BidBubbleOverlay', () => ({ default: () => null }));
vi.mock('./components/DemoToggle', () => ({ default: () => null }));
vi.mock('./components/AnnotationOverlay', () => ({ default: () => null }));
vi.mock('./animation/useDemoAnimation', () => ({ default: () => ({ engine: {} }) }));
vi.mock('./animation/DemoSequenceOrchestrator', () => ({
  default: class { start() { return Promise.resolve(); } stop() { return Promise.resolve(); } },
}));
vi.mock('./components/comparison', () => ({
  ComparisonProvider: ({ children }) => <>{children}</>,
  useComparison: () => ({
    mode: 'standalone',
    standalone: { result: null, loading: false, error: null },
    fabric: { result: null, loading: false, error: null },
    submitScenario: async () => ({}),
  }),
  ModeSelector: () => null,
  ComparisonLayout: () => null,
}));

import App from './App.jsx';

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
  await act(async () => { root.render(<App />); });
  await act(async () => {});
}

const q = (id) => container.querySelector(`[data-testid="${id}"]`);

describe('App in learner mode', () => {
  beforeEach(() => {
    fetchHandler = async () => jsonResp({ deploymentMode: 'learner', tritonReady: true });
  });

  it('shows the notice on load, and the badge stays after acknowledging', async () => {
    await mount();
    expect(q('learner-mode-notice-dialog')).not.toBeNull();
    expect(q('header-learner-mode-badge')).not.toBeNull();
    act(() => q('learner-mode-notice-acknowledge-button').click());
    expect(q('learner-mode-notice-dialog')).toBeNull();
    expect(q('header-learner-mode-badge')).not.toBeNull();
  });

  it('shows the notice again on the next load', async () => {
    await mount();
    act(() => q('learner-mode-notice-acknowledge-button').click());
    act(() => root.unmount());
    root = createRoot(container);
    await mount();
    expect(q('learner-mode-notice-dialog')).not.toBeNull();
  });

  it('does not open the Governance view', async () => {
    await mount();
    act(() => q('learner-mode-notice-acknowledge-button').click());
    act(() => q('header-governance-nav-link').click());
    expect(q('stub-governance')).toBeNull();
    expect(q('stub-sidebar')).not.toBeNull();
  });
});

describe.each([
  ['standard', async () => jsonResp({ deploymentMode: 'standard' })],
  ['unknown (missing field)', async () => jsonResp({ desiredSize: 0 })],
  ['unknown (request failed)', async () => { throw new Error('offline'); }],
])('App when the mode is %s', (_label, handler) => {
  beforeEach(() => { fetchHandler = handler; });

  it('shows no notice and keeps Governance reachable', async () => {
    await mount();
    expect(q('learner-mode-notice-dialog')).toBeNull();
    expect(q('header-learner-mode-badge')).toBeNull();
    act(() => q('header-governance-nav-link').click());
    expect(q('stub-governance')).not.toBeNull();
  });
});

describe('App while the mode is still being checked', () => {
  it('shows no notice and no badge', async () => {
    fetchHandler = () => new Promise(() => {});
    await mount();
    expect(q('learner-mode-notice-dialog')).toBeNull();
    expect(q('header-learner-mode-badge')).toBeNull();
    expect(q('header-mode-unknown-badge')).toBeNull();
  });
});
