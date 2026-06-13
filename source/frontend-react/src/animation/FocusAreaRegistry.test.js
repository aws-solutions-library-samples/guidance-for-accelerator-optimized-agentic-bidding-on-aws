/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach } from 'vitest';
import FocusAreaRegistry from './FocusAreaRegistry.js';

describe('FocusAreaRegistry', () => {
  let registry;

  beforeEach(() => {
    registry = new FocusAreaRegistry();
  });

  describe('default registrations', () => {
    it('has timeline registered', () => {
      expect(registry.has('timeline')).toBe(true);
    });

    it('has request-json registered', () => {
      expect(registry.has('request-json')).toBe(true);
    });

    it('has mutations-panel registered', () => {
      expect(registry.has('mutations-panel')).toBe(true);
    });

    it('has mutation-line-{index} pattern registered', () => {
      expect(registry.has('mutation-line-3')).toBe(true);
      expect(registry.has('mutation-line-0')).toBe(true);
      expect(registry.has('mutation-line-15')).toBe(true);
    });

    it('has agent-row-{id} pattern registered', () => {
      expect(registry.has('agent-row-dlrm')).toBe(true);
      expect(registry.has('agent-row-widedeep')).toBe(true);
    });

    it('keys() returns static area IDs', () => {
      const keys = registry.keys();
      expect(keys).toContain('timeline');
      expect(keys).toContain('request-json');
      expect(keys).toContain('mutations-panel');
    });
  });

  describe('has()', () => {
    it('returns false for unregistered IDs', () => {
      expect(registry.has('nonexistent')).toBe(false);
      expect(registry.has('mutation-line-')).toBe(false);
    });

    it('returns true for static IDs', () => {
      expect(registry.has('timeline')).toBe(true);
    });

    it('returns true for dynamic pattern matches', () => {
      expect(registry.has('mutation-line-5')).toBe(true);
      expect(registry.has('agent-row-ncf')).toBe(true);
    });
  });

  describe('register()', () => {
    it('registers a new static focus area', () => {
      registry.register('custom-area', {
        selector: '.custom-element',
        containerSelector: '.custom-container',
        type: 'element',
      });
      expect(registry.has('custom-area')).toBe(true);
      expect(registry.keys()).toContain('custom-area');
    });

    it('registers a new dynamic pattern', () => {
      registry.register('tab-{name}', {
        selector: '.tab-panel[data-name="{name}"]',
        containerSelector: null,
        type: 'element',
      });
      expect(registry.has('tab-settings')).toBe(true);
      expect(registry.has('tab-overview')).toBe(true);
    });

    it('uses default values for optional params', () => {
      registry.register('simple', { selector: '.simple' });
      const def = registry.getDefinition('simple');
      expect(def.containerSelector).toBeNull();
      expect(def.type).toBe('element');
    });
  });

  describe('getDefinition()', () => {
    it('returns definition for static areas', () => {
      const def = registry.getDefinition('timeline');
      expect(def).toEqual({
        selector: '.timeline-container',
        containerSelector: '#flow-canvas',
        type: 'element',
      });
    });

    it('returns definition with substituted params for dynamic patterns', () => {
      const def = registry.getDefinition('mutation-line-3');
      expect(def).toEqual({
        selector: '.raw-json-mutation-line:nth-of-type(3)',
        containerSelector: '.main-top-right .raw-json',
        type: 'scroll-target',
      });
    });

    it('returns definition for agent-row with substituted id', () => {
      const def = registry.getDefinition('agent-row-dlrm');
      expect(def).toEqual({
        selector: '.agent-row[data-node="dlrm"]',
        containerSelector: '.timeline-container',
        type: 'element',
      });
    });

    it('returns null for unregistered IDs', () => {
      expect(registry.getDefinition('nonexistent')).toBeNull();
    });
  });

  describe('resolve()', () => {
    it('returns null when DOM element is not found', () => {
      // In test environment (jsdom), no matching elements exist
      expect(registry.resolve('timeline')).toBeNull();
    });

    it('returns null for unregistered IDs', () => {
      expect(registry.resolve('nonexistent')).toBeNull();
    });

    it('returns the DOM element when it exists', () => {
      // Create a matching element in the DOM
      const el = document.createElement('div');
      el.className = 'timeline-container';
      document.body.appendChild(el);

      expect(registry.resolve('timeline')).toBe(el);

      document.body.removeChild(el);
    });

    it('resolves dynamic patterns to DOM elements', () => {
      const el = document.createElement('div');
      el.className = 'agent-row';
      el.setAttribute('data-node', 'ncf');
      document.body.appendChild(el);

      expect(registry.resolve('agent-row-ncf')).toBe(el);

      document.body.removeChild(el);
    });
  });
});
