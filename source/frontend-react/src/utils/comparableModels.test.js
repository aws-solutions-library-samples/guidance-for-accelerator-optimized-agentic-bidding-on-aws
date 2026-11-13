import { describe, it, expect } from 'vitest';
import {
  COMPARABLE_MODEL_TYPES,
  isFilModel,
  comparableModelLabel,
} from './comparableModels.js';

describe('COMPARABLE_MODEL_TYPES', () => {
  it('lists exactly the three trainable/canary-capable models', () => {
    expect(COMPARABLE_MODEL_TYPES.map((m) => m.key)).toEqual([
      'dlrm_bid_shader',
      'deal_yield_manager_floor',
      'deal_yield_manager_margin',
    ]);
  });

  it('excludes NCF (parked for training/canary)', () => {
    const keys = COMPARABLE_MODEL_TYPES.map((m) => m.key);
    expect(keys).not.toContain('ncf_deal_manager');
    const labels = COMPARABLE_MODEL_TYPES.map((m) => m.label);
    expect(labels.some((l) => /NCF|Deal Scorer/i.test(l))).toBe(false);
  });

  it('uses friendly names with the base model in parentheses', () => {
    const byKey = Object.fromEntries(COMPARABLE_MODEL_TYPES.map((m) => [m.key, m.label]));
    expect(byKey.dlrm_bid_shader).toBe('Bid Pricer (DLRM)');
    expect(byKey.deal_yield_manager_floor).toBe('Yield Optimizer — Floor (XGBoost)');
    expect(byKey.deal_yield_manager_margin).toBe('Yield Optimizer — Margin (XGBoost)');
  });
});

describe('isFilModel', () => {
  it('is true for the two FIL/XGBoost yield sub-models', () => {
    expect(isFilModel('deal_yield_manager_floor')).toBe(true);
    expect(isFilModel('deal_yield_manager_margin')).toBe(true);
  });

  it('is false for DLRM (TensorRT, agent-staged) and everything else', () => {
    expect(isFilModel('dlrm_bid_shader')).toBe(false);
    expect(isFilModel('ncf_deal_manager')).toBe(false);
    expect(isFilModel('')).toBe(false);
    expect(isFilModel(undefined)).toBe(false);
  });
});

describe('comparableModelLabel', () => {
  it('returns the friendly label for a known key', () => {
    expect(comparableModelLabel('dlrm_bid_shader')).toBe('Bid Pricer (DLRM)');
  });

  it('falls back to the raw key for an unknown type (honest, not hidden)', () => {
    expect(comparableModelLabel('mystery_model')).toBe('mystery_model');
  });
});
