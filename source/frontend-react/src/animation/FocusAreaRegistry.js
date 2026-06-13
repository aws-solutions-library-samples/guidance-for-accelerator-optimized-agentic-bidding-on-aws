/**
 * FocusAreaRegistry — Maps focus area identifiers to DOM selectors.
 * Extensible: new areas can be registered without modifying the engine.
 *
 * Supports both static IDs (e.g., "timeline") and dynamic patterns
 * (e.g., "mutation-line-{index}", "agent-row-{id}") that resolve
 * parameters at lookup time.
 */
class FocusAreaRegistry {
  constructor() {
    /** @type {Map<string, { selector: string, containerSelector: string|null, type: string }>} */
    this.areas = new Map();

    /** @type {Array<{ pattern: RegExp, template: { selector: string, containerSelector: string|null, type: string }, paramNames: string[] }>} */
    this._patterns = [];

    this._registerDefaults();
  }

  /**
   * Register a focus area with its CSS selector and optional container.
   * If the id contains `{param}` placeholders, it is stored as a dynamic pattern.
   * @param {string} id - Focus area identifier (e.g., "timeline" or "mutation-line-{index}")
   * @param {{ selector: string, containerSelector?: string|null, type?: string }} definition
   */
  register(id, { selector, containerSelector = null, type = 'element' }) {
    if (id.includes('{')) {
      // Dynamic pattern — extract param names and build a regex
      const paramNames = [];
      const regexStr = id.replace(/\{(\w+)\}/g, (_, name) => {
        paramNames.push(name);
        return '([^}]+)';
      });
      this._patterns.push({
        pattern: new RegExp(`^${regexStr}$`),
        template: { selector, containerSelector, type },
        paramNames,
      });
    } else {
      this.areas.set(id, { selector, containerSelector, type });
    }
  }

  /**
   * Resolve a focus area ID to its DOM element.
   * Handles both static IDs and dynamic pattern matching.
   * @param {string} id - Focus area identifier (e.g., "timeline" or "mutation-line-3")
   * @returns {HTMLElement|null} The matching DOM element, or null if not found
   */
  resolve(id) {
    // Try static lookup first
    const definition = this._resolveDefinition(id);
    if (!definition) return null;

    const element = document.querySelector(definition.selector);
    return element || null;
  }

  /**
   * Resolve a focus area ID to its full definition (selector, container, type).
   * Useful for the AnimationEngine to access container and type info.
   * @param {string} id
   * @returns {{ selector: string, containerSelector: string|null, type: string }|null}
   */
  _resolveDefinition(id) {
    // Static lookup
    if (this.areas.has(id)) {
      return this.areas.get(id);
    }

    // Dynamic pattern matching
    for (const { pattern, template, paramNames } of this._patterns) {
      const match = id.match(pattern);
      if (match) {
        // Substitute params into the selector template
        let selector = template.selector;
        let containerSelector = template.containerSelector;
        for (let i = 0; i < paramNames.length; i++) {
          const value = match[i + 1];
          const placeholder = `{${paramNames[i]}}`;
          selector = selector.replace(placeholder, value);
          if (containerSelector) {
            containerSelector = containerSelector.replace(placeholder, value);
          }
        }
        return { selector, containerSelector, type: template.type };
      }
    }

    return null;
  }

  /**
   * Check if a focus area ID is registered (static or matches a dynamic pattern).
   * @param {string} id
   * @returns {boolean}
   */
  has(id) {
    if (this.areas.has(id)) return true;
    return this._patterns.some(({ pattern }) => pattern.test(id));
  }

  /**
   * Get all registered static area IDs.
   * Note: dynamic patterns are not enumerable by specific ID.
   * @returns {string[]}
   */
  keys() {
    return [...this.areas.keys()];
  }

  /**
   * Get the full definition for a focus area (public API for AnimationEngine).
   * @param {string} id
   * @returns {{ selector: string, containerSelector: string|null, type: string }|null}
   */
  getDefinition(id) {
    return this._resolveDefinition(id);
  }

  /** Register the default focus areas for the ARTF demo UI. */
  _registerDefaults() {
    this.register('timeline', {
      selector: '.timeline-container',
      containerSelector: '#flow-canvas',
      type: 'element',
    });

    this.register('request-json', {
      selector: '.raw-section-standalone:has(.raw-json)',
      containerSelector: '.main-top-right',
      type: 'element',
    });

    this.register('mutations-panel', {
      selector: '.raw-panel--single .raw-section-standalone',
      containerSelector: '.main-top-left',
      type: 'element',
    });

    this.register('mutation-line-{index}', {
      selector: '.raw-json-mutation-line:nth-of-type({index})',
      containerSelector: '.main-top-right .raw-json',
      type: 'scroll-target',
    });

    this.register('agent-row-{id}', {
      selector: '.agent-row[data-node="{id}"]',
      containerSelector: '.timeline-container',
      type: 'element',
    });
  }
}

export default FocusAreaRegistry;
