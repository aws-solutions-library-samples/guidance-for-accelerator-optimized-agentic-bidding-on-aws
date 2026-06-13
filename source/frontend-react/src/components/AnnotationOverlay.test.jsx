/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

// Mock GSAP before importing the component
vi.mock('../animation/gsapSetup.js', () => {
  const mockGsap = {
    set: vi.fn((el, props) => {
      if (props.autoAlpha === 0) {
        el.style.visibility = 'hidden';
        el.style.opacity = '0';
      }
    }),
    to: vi.fn((el, props) => {
      if (props.autoAlpha === 1) {
        el.style.visibility = 'visible';
        el.style.opacity = '1';
      } else if (props.autoAlpha === 0) {
        el.style.visibility = 'hidden';
        el.style.opacity = '0';
      }
      if (props.onComplete) props.onComplete();
    }),
  };
  return { gsap: mockGsap };
});

import AnnotationOverlay from './AnnotationOverlay.jsx';
import { gsap } from '../animation/gsapSetup.js';

describe('AnnotationOverlay', () => {
  let container;
  let root;
  let targetEl;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);

    // Create a target element to inject annotations into
    targetEl = document.createElement('div');
    targetEl.className = 'timeline-container';
    targetEl.style.position = 'relative';
    document.body.appendChild(targetEl);

    vi.clearAllMocks();
  });

  afterEach(() => {
    act(() => { root.unmount(); });
    document.body.removeChild(container);
    if (targetEl.parentNode) document.body.removeChild(targetEl);
  });

  function render(jsx) {
    act(() => { root.render(jsx); });
  }

  it('renders nothing in the React tree (returns null)', () => {
    render(<AnnotationOverlay text="Hello" visible={true} targetElement={targetEl} />);
    expect(container.innerHTML).toBe('');
  });

  it('injects annotation into the target element when visible', () => {
    render(<AnnotationOverlay text="Test annotation" visible={true} targetElement={targetEl} />);
    const injected = targetEl.querySelector('.demo-annotation-injected');
    expect(injected).not.toBeNull();
    expect(injected.textContent).toBe('Test annotation');
  });

  it('does not inject when not visible', () => {
    render(<AnnotationOverlay text="Test" visible={false} targetElement={targetEl} />);
    const injected = targetEl.querySelector('.demo-annotation-injected');
    expect(injected).toBeNull();
  });

  it('does not inject when text is null', () => {
    render(<AnnotationOverlay text={null} visible={true} targetElement={targetEl} />);
    const injected = targetEl.querySelector('.demo-annotation-injected');
    expect(injected).toBeNull();
  });

  it('does not inject when targetElement is null', () => {
    render(<AnnotationOverlay text="Test" visible={true} targetElement={null} />);
    // Nothing should be injected anywhere
    expect(document.querySelector('.demo-annotation-injected')).toBeNull();
  });

  it('uses GSAP to fade in the annotation', () => {
    render(<AnnotationOverlay text="Fade test" visible={true} targetElement={targetEl} />);
    expect(gsap.to).toHaveBeenCalled();
    const fadeInCall = gsap.to.mock.calls.find(
      ([el, props]) => props.autoAlpha === 1
    );
    expect(fadeInCall).toBeDefined();
    expect(fadeInCall[1].duration).toBe(0.3);
  });

  it('positions annotation absolutely inside the target', () => {
    render(<AnnotationOverlay text="Position test" visible={true} targetElement={targetEl} />);
    const injected = targetEl.querySelector('.demo-annotation-injected');
    expect(injected.style.position).toBe('absolute');
  });

  it('has accessible role and aria-live', () => {
    render(<AnnotationOverlay text="A11y test" visible={true} targetElement={targetEl} />);
    const injected = targetEl.querySelector('.demo-annotation-injected');
    expect(injected.getAttribute('role')).toBe('status');
    expect(injected.getAttribute('aria-live')).toBe('polite');
  });

  it('removes annotation when visibility changes to false', () => {
    render(<AnnotationOverlay text="Remove test" visible={true} targetElement={targetEl} />);
    expect(targetEl.querySelector('.demo-annotation-injected')).not.toBeNull();

    render(<AnnotationOverlay text="Remove test" visible={false} targetElement={targetEl} />);
    // After GSAP fade-out completes (mocked to be instant), element is removed
    expect(targetEl.querySelector('.demo-annotation-injected')).toBeNull();
  });

  it('updates text when it changes', () => {
    render(<AnnotationOverlay text="First" visible={true} targetElement={targetEl} />);
    expect(targetEl.querySelector('.demo-annotation-injected').textContent).toBe('First');

    render(<AnnotationOverlay text="Second" visible={true} targetElement={targetEl} />);
    expect(targetEl.querySelector('.demo-annotation-injected').textContent).toBe('Second');
  });
});
