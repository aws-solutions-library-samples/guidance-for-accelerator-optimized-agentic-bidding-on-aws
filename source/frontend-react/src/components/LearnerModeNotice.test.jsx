/**
 * @vitest-environment jsdom
 *
 * The learner-mode notice: required copy, dialog semantics, focus handling, and
 * that only the button dismisses it.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

import LearnerModeNotice from './LearnerModeNotice.jsx';

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

function render(onAcknowledge = () => {}) {
  act(() => {
    root.render(<LearnerModeNotice onAcknowledge={onAcknowledge} />);
  });
  return container.querySelector('[data-testid="learner-mode-notice-dialog"]');
}

function key(el, k, opts = {}) {
  act(() => {
    el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...opts }));
  });
}

describe('LearnerModeNotice', () => {
  it('carries the required statements', () => {
    const dialog = render();
    const text = dialog.textContent;
    expect(text).toContain('Deployed in learner mode');
    expect(text).toContain('This workload was deployed in learner mode.');
    expect(text).toContain('Inference runs on CPUs instead of GPUs');
    expect(text).toContain(
      'For this implementation to remain ARTF-compliant, learner mode must not be used for production workloads.',
    );
    expect(text).toContain('Governance tab is unavailable');
    const flat = text.replace(/\s+/g, ' ');
    expect(flat).toContain('Adaptive Bidding is also unavailable');
    expect(flat).toContain('the Adaptive Bidding agent is not deployed');
  });

  it('is a labelled modal dialog', () => {
    const dialog = render();
    expect(dialog.getAttribute('role')).toBe('dialog');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    const labelId = dialog.getAttribute('aria-labelledby');
    const descId = dialog.getAttribute('aria-describedby');
    expect(document.getElementById(labelId).textContent).toBe('Deployed in learner mode');
    expect(document.getElementById(descId).textContent).toContain('ARTF-compliant');
  });

  it('focuses the acknowledge button and keeps focus there on Tab', () => {
    const dialog = render();
    const button = container.querySelector('[data-testid="learner-mode-notice-acknowledge-button"]');
    expect(document.activeElement).toBe(button);
    button.blur();
    key(dialog, 'Tab');
    expect(document.activeElement).toBe(button);
    button.blur();
    key(dialog, 'Tab', { shiftKey: true });
    expect(document.activeElement).toBe(button);
  });

  it('is not dismissed by Escape', () => {
    const onAck = vi.fn();
    const dialog = render(onAck);
    key(dialog, 'Escape');
    expect(onAck).not.toHaveBeenCalled();
    expect(container.querySelector('[data-testid="learner-mode-notice-dialog"]')).not.toBeNull();
  });

  it('calls onAcknowledge from the button', () => {
    const onAck = vi.fn();
    render(onAck);
    const button = container.querySelector('[data-testid="learner-mode-notice-acknowledge-button"]');
    expect(button.textContent).toBe('I understand');
    act(() => button.click());
    expect(onAck).toHaveBeenCalledTimes(1);
  });

  it('does not persist the acknowledgement', () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem');
    render();
    act(() => container.querySelector('[data-testid="learner-mode-notice-acknowledge-button"]').click());
    expect(setItem).not.toHaveBeenCalled();
    setItem.mockRestore();
  });
});
