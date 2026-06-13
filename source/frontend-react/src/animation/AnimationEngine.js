import { gsap } from './gsapSetup.js';

/**
 * AnimationEngine — Executes structured Animation_Sequences using GSAP.
 * Decoupled from React; can be driven by any caller (demo loop, AI agent, tests).
 *
 * No demo-specific logic lives here — the engine only knows how to execute
 * generic AnimationStep commands: scroll, zoom, annotate, dwell, reverse.
 */
class AnimationEngine {
  /**
   * @param {import('./FocusAreaRegistry.js').default} registry - FocusAreaRegistry instance
   * @param {object} options - EngineOptions
   * @param {number} [options.transitionDuration=0.8] - Default transition duration in seconds
   * @param {string} [options.easing="power2.inOut"] - GSAP easing function
   * @param {number} [options.annotationFadeMs=300] - Annotation fade duration in ms
   * @param {boolean} [options.autoScroll=true] - Whether to auto-scroll containers
   * @param {(text: string, element: HTMLElement|null) => void} [options.onAnnotationShow] - Callback when annotation shows
   * @param {() => void} [options.onAnnotationHide] - Callback when annotation hides
   */
  constructor(registry, options = {}) {
    this._registry = registry;
    this._options = {
      transitionDuration: options.transitionDuration ?? 0.8,
      easing: options.easing ?? 'power2.inOut',
      annotationFadeMs: options.annotationFadeMs ?? 300,
      autoScroll: options.autoScroll ?? true,
      onAnnotationShow: options.onAnnotationShow ?? null,
      onAnnotationHide: options.onAnnotationHide ?? null,
    };

    this._running = false;
    this._stopped = false;
    this._activeTl = null;
    this._transformStack = [];
    this._dwellResolve = null;
    this._dwellTimeout = null;
    this._currentStepIndex = -1;
    this._currentSequence = null;
    this._hoveredElement = null;
  }

  /**
   * Whether the engine is currently executing a sequence.
   * @returns {boolean}
   */
  get isRunning() {
    return this._running;
  }

  /**
   * Execute a complete animation sequence.
   * Steps are sorted by `order` and executed sequentially.
   * @param {Array<{ focusArea: string, zoomLevel: number, annotation: string|null, dwellMs: number, order: number }>} sequence
   * @returns {Promise<void>} Resolves when sequence completes or is stopped
   */
  async execute(sequence) {
    if (this._running) {
      await this.stop();
    }

    this._running = true;
    this._stopped = false;
    this._currentStepIndex = -1;

    // Sort steps by order (ascending)
    const sortedSteps = [...sequence].sort((a, b) => a.order - b.order);
    this._currentSequence = sortedSteps;

    try {
      for (let i = 0; i < sortedSteps.length; i++) {
        if (this._stopped) break;

        this._currentStepIndex = i;
        const step = sortedSteps[i];

        await this._executeStep(step);

        if (this._stopped) break;
      }
    } finally {
      this._running = false;
      this._currentSequence = null;
      this._currentStepIndex = -1;
    }
  }

  /**
   * Stop the current sequence, reverse all transforms, restore UI.
   * @returns {Promise<void>}
   */
  async stop() {
    this._stopped = true;

    // Kill any active GSAP timeline
    if (this._activeTl) {
      this._activeTl.kill();
      this._activeTl = null;
    }

    // Resolve any pending dwell early
    if (this._dwellResolve) {
      clearTimeout(this._dwellTimeout);
      this._dwellResolve();
      this._dwellResolve = null;
      this._dwellTimeout = null;
    }

    // Reverse all transforms in LIFO order and restore original styles
    const reversePromises = this._transformStack.reverse().map(({ element, originalStyles, overflowAncestors }) => {
      // Restore original styles (background, position, zIndex)
      if (originalStyles) {
        element.style.backgroundColor = originalStyles.backgroundColor;
        element.style.position = originalStyles.position;
        element.style.zIndex = originalStyles.zIndex;
      }
      // Restore overflow on ancestors
      if (overflowAncestors) {
        for (const { el, originalOverflow, originalOverflowY, originalMaxHeight } of overflowAncestors) {
          if (originalOverflow !== undefined) el.style.overflow = originalOverflow;
          if (originalOverflowY !== undefined) el.style.overflowY = originalOverflowY;
          if (originalMaxHeight !== undefined) el.style.maxHeight = originalMaxHeight;
        }
      }
      return new Promise((resolve) => {
        gsap.to(element, {
          scale: 1,
          x: 0,
          y: 0,
          duration: 0.4,
          ease: 'power2.out',
          onComplete: resolve,
        });
      });
    });

    if (reversePromises.length > 0) {
      await Promise.all(reversePromises);
    }

    this._transformStack = [];

    // Clear any active annotation
    this._hideAnnotation();

    // Hook point: clear hover states (mutation hover added in Task 3.1)
    this._clearHoverState();

    this._running = false;
  }

  /**
   * Skip to the next step — resolves current dwell early, advances to next step.
   */
  skipToNext() {
    if (!this._running) return;

    // Resolve the current dwell early to advance
    if (this._dwellResolve) {
      clearTimeout(this._dwellTimeout);
      this._dwellResolve();
      this._dwellResolve = null;
      this._dwellTimeout = null;
    }
  }

  /**
   * Execute a single animation step.
   * @param {{ focusArea: string, zoomLevel: number, annotation: string|null, dwellMs: number, order: number }} step
   * @returns {Promise<void>}
   * @private
   */
  async _executeStep(step) {
    // 1. Validate focusArea exists in registry
    if (!this._registry.has(step.focusArea)) {
      console.warn(`[AnimationEngine] Focus area "${step.focusArea}" not registered — skipping step.`);
      return;
    }

    // 2. Resolve DOM element via registry
    const element = this._registry.resolve(step.focusArea);
    if (!element) {
      console.warn(`[AnimationEngine] DOM element for "${step.focusArea}" not found — skipping step.`);
      return;
    }

    const definition = this._registry.getDefinition(step.focusArea);

    // 3. Scroll into view if needed (use GSAP scrollTo for smooth scroll)
    if (this._options.autoScroll && definition.containerSelector) {
      const container = document.querySelector(definition.containerSelector);
      if (container) {
        await this._scrollToElement(container, element);
        if (this._stopped) return;
      }
    }

    // 4. For non-scroll-target types, remove overflow constraints before zoom
    //    For scroll-target types (mutation lines), keep overflow so scrollTo works
    const overflowAncestors = definition.type !== 'scroll-target'
      ? this._removeOverflowConstraints(element)
      : [];

    // 5. Apply zoom transform (also adds solid background and high z-index)
    const originalStyles = await this._applyZoom(element, step.zoomLevel);
    if (this._stopped) return;

    // Track transform for stop() reversal
    this._transformStack.push({ element, originalStyles, overflowAncestors });

    // 6. Hook point: mutation hover dispatch (added in Task 3.1)
    this._beforeDwell(step, element);

    // 7. Show annotation if present — pass the element for positioning
    if (step.annotation) {
      this._showAnnotation(step.annotation, element);
    }

    // 8. Wait for dwell duration (enforce minimum 2000ms for mutation-line focus areas)
    const dwellMs = step.focusArea.startsWith('mutation-line-')
      ? Math.max(step.dwellMs, 2000)
      : step.dwellMs;
    await this._dwell(dwellMs);
    if (this._stopped) return;

    // 9. Hide annotation
    if (step.annotation) {
      this._hideAnnotation();
    }

    // 10. Hook point: mutation hover cleanup (added in Task 3.1)
    this._afterDwell(step, element);

    // 11. Reverse zoom
    await this._reverseZoom(element, originalStyles);

    // 12. Restore overflow on ancestors
    for (const { el, originalOverflow, originalOverflowY, originalMaxHeight } of overflowAncestors) {
      if (originalOverflow !== undefined) el.style.overflow = originalOverflow;
      if (originalOverflowY !== undefined) el.style.overflowY = originalOverflowY;
      if (originalMaxHeight !== undefined) el.style.maxHeight = originalMaxHeight;
    }

    // Remove from transform stack since we reversed it ourselves
    this._transformStack = this._transformStack.filter((entry) => entry.element !== element);
  }

  /**
   * Walk up the DOM tree from an element and temporarily set overflow: visible
   * on any ancestor that has overflow: auto/scroll/hidden. Also removes max-height
   * on the element itself if set.
   * @param {HTMLElement} element
   * @returns {Array<{ el: HTMLElement, originalOverflow?: string, originalOverflowY?: string, originalMaxHeight?: string }>}
   * @private
   */
  _removeOverflowConstraints(element) {
    const ancestors = [];

    // Check the element itself for max-height
    const elStyle = getComputedStyle(element);
    if (elStyle.maxHeight && elStyle.maxHeight !== 'none') {
      ancestors.push({
        el: element,
        originalMaxHeight: element.style.maxHeight,
      });
      element.style.maxHeight = 'none';
    }

    // Walk up ancestors
    let current = element.parentElement;
    while (current && current !== document.body && current !== document.documentElement) {
      const computed = getComputedStyle(current);
      const overflow = computed.overflow;
      const overflowY = computed.overflowY;

      if (
        overflow === 'auto' || overflow === 'scroll' || overflow === 'hidden' ||
        overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'hidden'
      ) {
        ancestors.push({
          el: current,
          originalOverflow: current.style.overflow,
          originalOverflowY: current.style.overflowY,
        });
        current.style.overflow = 'visible';
        current.style.overflowY = 'visible';
      }

      current = current.parentElement;
    }

    return ancestors;
  }

  /**
   * Scroll a container to bring an element into view.
   * @param {HTMLElement} container
   * @param {HTMLElement} element
   * @returns {Promise<void>}
   * @private
   */
  _scrollToElement(container, element) {
    return new Promise((resolve) => {
      const tl = gsap.to(container, {
        scrollTo: { y: element, offsetY: 50 },
        duration: 0.6,
        ease: this._options.easing,
        onComplete: resolve,
      });
      this._activeTl = tl;
    });
  }

  /**
   * Apply zoom transform to an element. Also adds a solid background
   * to prevent content bleed-through during scale.
   * @param {HTMLElement} element
   * @param {number} zoomLevel
   * @returns {Promise<{ backgroundColor: string, position: string, zIndex: string }>} Original styles
   * @private
   */
  _applyZoom(element, zoomLevel) {
    // Store original styles before modifying
    const originalStyles = {
      backgroundColor: element.style.backgroundColor,
      position: element.style.position,
      zIndex: element.style.zIndex,
    };

    // Add solid background to prevent content bleed-through
    element.style.backgroundColor = 'var(--surface, #1a1a2e)';
    element.style.position = 'relative';
    element.style.zIndex = '5000';

    return new Promise((resolve) => {
      const tl = gsap.to(element, {
        scale: zoomLevel,
        transformOrigin: 'center center',
        duration: this._options.transitionDuration,
        ease: this._options.easing,
        onComplete: () => resolve(originalStyles),
      });
      this._activeTl = tl;
    });
  }

  /**
   * Reverse zoom on an element back to scale(1) and restore original styles.
   * @param {HTMLElement} element
   * @param {{ backgroundColor: string, position: string, zIndex: string }} [originalStyles]
   * @returns {Promise<void>}
   * @private
   */
  _reverseZoom(element, originalStyles) {
    return new Promise((resolve) => {
      const tl = gsap.to(element, {
        scale: 1,
        duration: 0.6,
        ease: this._options.easing,
        onComplete: () => {
          // Restore original styles after zoom reversal
          if (originalStyles) {
            element.style.backgroundColor = originalStyles.backgroundColor;
            element.style.position = originalStyles.position;
            element.style.zIndex = originalStyles.zIndex;
          }
          resolve();
        },
      });
      this._activeTl = tl;
    });
  }

  /**
   * Wait for the specified dwell duration.
   * Can be resolved early by skipToNext() or stop().
   * @param {number} ms
   * @returns {Promise<void>}
   * @private
   */
  _dwell(ms) {
    return new Promise((resolve) => {
      this._dwellResolve = resolve;
      this._dwellTimeout = setTimeout(() => {
        this._dwellResolve = null;
        this._dwellTimeout = null;
        resolve();
      }, ms);
    });
  }

  /**
   * Show annotation via callback, passing the focused element for positioning.
   * @param {string} text
   * @param {HTMLElement|null} element
   * @private
   */
  _showAnnotation(text, element = null) {
    if (this._options.onAnnotationShow) {
      this._options.onAnnotationShow(text, element);
    }
  }

  /**
   * Hide annotation via callback.
   * @private
   */
  _hideAnnotation() {
    if (this._options.onAnnotationHide) {
      this._options.onAnnotationHide();
    }
  }

  /**
   * Hook point: called before dwell begins.
   * Dispatches synthetic mouseenter on mutation-line focus areas to trigger tooltip.
   * @param {{ focusArea: string }} step
   * @param {HTMLElement} element
   * @private
   */
  _beforeDwell(step, element) {
    if (step.focusArea.startsWith('mutation-line-')) {
      element.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true }));
      this._hoveredElement = element;
    }
  }

  /**
   * Hook point: called after dwell ends.
   * Dispatches synthetic mouseleave on any hovered mutation-line element.
   * @param {{ focusArea: string }} step
   * @param {HTMLElement} element
   * @private
   */
  _afterDwell(step, element) {
    if (this._hoveredElement) {
      this._hoveredElement.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
      this._hoveredElement = null;
    }
  }

  /**
   * Clear any active hover state by dispatching mouseleave on the currently hovered element.
   * Called during stop() to ensure tooltips are dismissed.
   * @private
   */
  _clearHoverState() {
    if (this._hoveredElement) {
      this._hoveredElement.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
      this._hoveredElement = null;
    }
  }
}

export default AnimationEngine;
