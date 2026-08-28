/**
 * @vitest-environment jsdom
 *
 * Tests for showing WHY a model version was rejected.
 *
 * A bare "Rejected" badge hid the distinction between two very different
 * conclusions: a pipeline step failed, or the challenger was evaluated and lost.
 * The live registry's version 2 was rejected for the former — the TensorRT
 * optimization step timed out against the VPC optimizer proxy, so the A/B test
 * never ran — yet it rendered identically to a genuine model verdict.
 *
 * The reason text asserted here is the real ApprovalDescription read from the
 * deployed registry, so these tests exercise the string shape actually produced
 * rather than an invented one.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

import { ApprovalReason, isPipelineFailure, shortReason } from './closedLoopUi.jsx';

const REAL_OPTIMIZATION_FAILURE =
  'Model optimization failed: Model optimization failed for dlrm_bid_shader: ' +
  'status=502, body=proxy invoke failed: Read timeout on endpoint URL: ' +
  '"https://lambda.us-east-1.amazonaws.com/2015-03-31/functions/' +
  'arn%3Aaws%3Alambda%3Aus-east-1%3A960328030835%3Afunction%3Anv5-vpc-optimizer-proxy/invocations"';

describe('isPipelineFailure', () => {
  it('classifies the real optimization-timeout rejection as a pipeline failure', () => {
    expect(isPipelineFailure(REAL_OPTIMIZATION_FAILURE)).toBe(true);
  });

  it('classifies the other agent step failures as pipeline failures', () => {
    expect(isPipelineFailure('Canary deployment failed: CanaryLoadError')).toBe(true);
    expect(isPipelineFailure('A/B test error: evaluator raised')).toBe(true);
    expect(isPipelineFailure('Guardrail breach: p99 latency regression')).toBe(true);
  });

  it('does NOT classify a real A/B verdict as a pipeline failure', () => {
    expect(isPipelineFailure('Challenger underperformed: p=0.31, lift -2.4%')).toBe(false);
    expect(isPipelineFailure('Inconclusive: SPRT did not reach a boundary')).toBe(false);
  });

  it('treats a missing reason as not a pipeline failure rather than guessing', () => {
    expect(isPipelineFailure(null)).toBe(false);
    expect(isPipelineFailure('')).toBe(false);
    expect(isPipelineFailure(undefined)).toBe(false);
  });
});

describe('shortReason', () => {
  it('keeps the lead-in clause out of a very long reason', () => {
    expect(shortReason(REAL_OPTIMIZATION_FAILURE)).toBe('Model optimization failed');
  });

  it('caps runaway text so the table row cannot blow out', () => {
    const long = `${'x'.repeat(200)} no colon here`;
    expect(shortReason(long).length).toBeLessThanOrEqual(60);
    expect(shortReason(long).endsWith('...')).toBe(true);
  });

  it('passes a short reason through unchanged', () => {
    expect(shortReason('Promoted after A/B test')).toBe('Promoted after A/B test');
  });
});

describe('ApprovalReason rendering', () => {
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

  const render = (props) => act(() => root.render(<ApprovalReason {...props} />));

  it('puts the FULL reason on the title attribute for hover', () => {
    render({ status: 'Rejected', reason: REAL_OPTIMIZATION_FAILURE });
    const el = container.querySelector('[data-testid="approval-reason"]');
    expect(el).not.toBeNull();
    // Nothing is lost to truncation — the whole string, proxy URL included.
    expect(el.getAttribute('title')).toBe(REAL_OPTIMIZATION_FAILURE);
  });

  it('shows the short lead-in inline, not the whole error body', () => {
    render({ status: 'Rejected', reason: REAL_OPTIMIZATION_FAILURE });
    const text = container.textContent;
    expect(text).toContain('Model optimization failed');
    expect(text).not.toContain('lambda.us-east-1.amazonaws.com');
  });

  it('tags a pipeline failure so it is not read as a model verdict', () => {
    render({ status: 'Rejected', reason: REAL_OPTIMIZATION_FAILURE });
    expect(container.querySelector('.cl-approval-reason-tag')).not.toBeNull();
  });

  it('does not tag a genuine A/B rejection as a pipeline failure', () => {
    render({ status: 'Rejected', reason: 'Challenger underperformed: p=0.31' });
    expect(container.querySelector('.cl-approval-reason-tag')).toBeNull();
    expect(container.textContent).toContain('Challenger underperformed');
  });

  it('says so plainly when a decided version has no recorded reason', () => {
    render({ status: 'Rejected', reason: null });
    expect(container.textContent).toContain('No reason recorded');
  });

  it('shows a plain dash for a pending version, which has no decision yet', () => {
    render({ status: 'PendingManualApproval', reason: null });
    expect(container.textContent).toBe('—');
    expect(container.textContent).not.toContain('No reason recorded');
  });

  it('never invents a reason when none was recorded', () => {
    render({ status: 'Approved', reason: null });
    expect(container.querySelector('[data-testid="approval-reason"]')).toBeNull();
  });
});
