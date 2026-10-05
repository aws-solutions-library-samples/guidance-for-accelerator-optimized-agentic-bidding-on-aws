/**
 * @vitest-environment jsdom
 *
 * The registry table names the model it shows and lists training jobs that
 * have not (yet, or ever) produced a version.
 *
 * The table used to render a fixed "Model registry versions" title while being
 * fed by whichever model the test-versions selector happened to be on, so one
 * Approved row read as "the" model with no way to tell which. And a SageMaker
 * training job only appears in the registry once it registers a package, so the
 * table was silent for the whole training duration.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import { ModelsView } from './closedLoopUi.jsx';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const VERSION = {
  version: 1,
  approval_status: 'Approved',
  approval_description: null,
  status: 'Completed',
  created_at: '2026-10-04T20:07:09.400000+00:00',
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

function render(props) {
  act(() => { root.render(<ModelsView {...props} />); });
}

describe('ModelsView', () => {
  it('names the model in the title', () => {
    render({ versions: [VERSION], modelLabel: 'Bid Pricer (DLRM)' });
    const title = container.querySelector('[data-testid="models-view-title"]');
    expect(title.textContent).toBe('Model registry versions · Bid Pricer (DLRM) · SageMaker');
  });

  it('renders an in-progress training job above the versions with a badge', () => {
    render({
      versions: [VERSION],
      modelLabel: 'Bid Pricer (DLRM)',
      trainingJobs: [{
        job_name: 'dlrm-bid-shader-1759690000-ab12cd34',
        status: 'InProgress',
        secondary_status: 'Training',
        created_at: '2026-10-05T18:01:00+00:00',
        model_type: 'dlrm_bid_shader',
      }],
    });
    const rows = container.querySelectorAll('tbody tr');
    expect(rows).toHaveLength(2);
    expect(rows[0].getAttribute('data-testid')).toBe('training-job-row');
    expect(rows[0].textContent).toContain('dlrm-bid-shader-1759690000-ab12cd34');
    expect(rows[0].textContent).toContain('Training');
    expect(rows[0].textContent).toContain('Registers a new version when the job completes');
    const badge = rows[0].querySelector('[data-testid="training-job-badge"]');
    expect(badge.textContent).toBe('Training in progress');
    expect(badge.className).toContain('state-processing');
    expect(rows[1].textContent).toContain('Approved');
  });

  it('shows a failed job with its failure reason and no halo', () => {
    render({
      versions: [],
      modelLabel: 'Yield Optimizer — Floor (XGBoost)',
      trainingJobs: [{
        job_name: 'deal-yield-manager-floor-1-x',
        status: 'Failed',
        secondary_status: 'Failed',
        created_at: '2026-10-05T10:00:00+00:00',
        ended_at: '2026-10-05T10:20:00+00:00',
        failure_reason: 'AlgorithmError: training data was empty',
      }],
    });
    const row = container.querySelector('[data-testid="training-job-row"]');
    expect(row.textContent).toContain('AlgorithmError: training data was empty');
    const badge = row.querySelector('[data-testid="training-job-badge"]');
    expect(badge.textContent).toBe('Training failed');
    expect(badge.className).not.toContain('state-processing');
    // A job row alone is enough to render the table; the empty-state copy is wrong here.
    expect(container.textContent).not.toContain('No registered model versions yet.');
  });

  it('shows the empty state only when there are neither versions nor jobs', () => {
    render({ versions: [], trainingJobs: [], modelLabel: 'Bid Pricer (DLRM)' });
    expect(container.textContent).toContain('No registered model versions yet.');
    expect(container.querySelector('table')).toBeNull();
  });

  it('surfaces a training-jobs lookup failure without hiding the versions', () => {
    render({
      versions: [VERSION],
      modelLabel: 'Bid Pricer (DLRM)',
      trainingJobs: [],
      trainingJobsError: 'ClientError: AccessDeniedException',
    });
    const err = container.querySelector('[data-testid="models-view-jobs-error"]');
    expect(err.textContent).toContain('AccessDeniedException');
    expect(container.querySelectorAll('tbody tr')).toHaveLength(1);
  });
});
