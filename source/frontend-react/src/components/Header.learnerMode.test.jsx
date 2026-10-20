/**
 * @vitest-environment jsdom
 *
 * Header in each deployment mode: the learner badge and disabled Governance
 * tab, the unknown badge with its reason, and an unchanged header otherwise.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

vi.mock('../auth', () => ({
  isAuthConfigured: () => false,
  signOut: () => {},
  getCurrentUserEmail: async () => null,
}));

import Header from './Header.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

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

function render(props) {
  const onViewChange = vi.fn();
  act(() => {
    root.render(
      <Header
        loading={false}
        error={null}
        onContainersClick={() => {}}
        view="scenarios"
        onViewChange={onViewChange}
        {...props}
      />,
    );
  });
  return onViewChange;
}

const q = (id) => container.querySelector(`[data-testid="${id}"]`);

describe('Header in learner mode', () => {
  it('disables Governance without hiding it', () => {
    const onViewChange = render({ deploymentMode: 'learner' });
    const gov = q('header-governance-nav-link');
    expect(gov).not.toBeNull();
    expect(gov.getAttribute('aria-disabled')).toBe('true');
    expect(gov.disabled).toBe(false); // focusable, so the tooltip is reachable
    expect(gov.textContent).toContain('(unavailable in learner mode)');
    expect(gov.getAttribute('title')).toMatch(/require GPUs/);
    act(() => gov.click());
    expect(onViewChange).not.toHaveBeenCalled();
  });

  it('shows the learner badge with the compliance sentence', () => {
    render({ deploymentMode: 'learner' });
    const badge = q('header-learner-mode-badge');
    expect(badge.textContent).toBe('Learner mode');
    expect(badge.getAttribute('title')).toContain('must not be used for production workloads');
    expect(badge.getAttribute('title')).toContain('CPUs instead of GPUs');
    expect(q('header-mode-unknown-badge')).toBeNull();
  });

  it('leaves the other tabs working', () => {
    const onViewChange = render({ deploymentMode: 'learner' });
    const adaptive = [...container.querySelectorAll('.header-nav-link')].find((b) => b.textContent === 'Adaptive Bidding');
    act(() => adaptive.click());
    expect(onViewChange).toHaveBeenCalledWith('adaptive');
  });
});

describe('Header in standard mode', () => {
  it('has no badge and an enabled Governance tab', () => {
    const onViewChange = render({ deploymentMode: 'standard' });
    expect(q('header-learner-mode-badge')).toBeNull();
    expect(q('header-mode-unknown-badge')).toBeNull();
    const gov = q('header-governance-nav-link');
    expect(gov.getAttribute('aria-disabled')).toBeNull();
    expect(gov.textContent).toBe('Governance');
    act(() => gov.click());
    expect(onViewChange).toHaveBeenCalledWith('governance');
  });

  it('defaults to standard when no mode is passed', () => {
    render({});
    expect(q('header-learner-mode-badge')).toBeNull();
    expect(q('header-mode-unknown-badge')).toBeNull();
  });
});

describe('Header when the mode is unknown', () => {
  it('shows the unknown badge with the reason and keeps Governance enabled', () => {
    const onViewChange = render({ deploymentMode: 'unknown', deploymentModeReason: 'HTTP 502' });
    const badge = q('header-mode-unknown-badge');
    expect(badge.textContent).toBe('Deployment mode unknown');
    expect(badge.getAttribute('title')).toBe('HTTP 502');
    expect(q('header-learner-mode-badge')).toBeNull();
    act(() => q('header-governance-nav-link').click());
    expect(onViewChange).toHaveBeenCalledWith('governance');
  });
});

describe('Header while checking', () => {
  it('shows neither badge', () => {
    render({ deploymentMode: 'checking' });
    expect(q('header-learner-mode-badge')).toBeNull();
    expect(q('header-mode-unknown-badge')).toBeNull();
  });
});
