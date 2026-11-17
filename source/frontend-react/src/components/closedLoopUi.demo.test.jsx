/**
 * @vitest-environment jsdom
 *
 * FR2.3: a registry version whose training input included the synthetic demo
 * dataset carries a "Trained with demo data" label; other versions do not.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import { ModelsView } from './closedLoopUi.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const BASE = {
  approval_status: 'PendingManualApproval',
  approval_description: null,
  status: 'Completed',
  created_at: '2026-11-15T10:30:00+00:00',
  includes_demo_dataset: false,
  demo_dataset_generated_at: null,
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

describe('ModelsView demo data label', () => {
  it('labels only the version trained with demo data', () => {
    const versions = [
      { ...BASE, version: 2, includes_demo_dataset: true, demo_dataset_generated_at: '2026-11-15T10:00:00Z' },
      { ...BASE, version: 1 },
    ];
    act(() => { root.render(<ModelsView versions={versions} modelLabel="Bid Pricer (DLRM)" />); });

    const badges = container.querySelectorAll('[data-testid="model-version-demo-badge"]');
    expect(badges).toHaveLength(1);
    expect(badges[0].textContent).toBe('Trained with demo data');
    expect(badges[0].getAttribute('title')).toContain('Demonstration only');
    expect(badges[0].getAttribute('title')).toContain('2026-11-15T10:00:00Z');
    expect(badges[0].closest('tr').textContent).toContain('2');
  });

  it('shows no label when no version used demo data', () => {
    act(() => { root.render(<ModelsView versions={[{ ...BASE, version: 1 }]} modelLabel="x" />); });
    expect(container.querySelector('[data-testid="model-version-demo-badge"]')).toBeNull();
  });
});
