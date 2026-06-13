/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';

// Mock GSAP before importing the component
vi.mock('../animation/gsapSetup.js', () => {
  const calls = [];
  const mockTween = {
    kill: vi.fn(),
  };
  const mockGsap = {
    set: (el, props) => {
      calls.push({ method: 'set', el, props });
      if (props.boxShadow) {
        el.style.boxShadow = props.boxShadow;
      }
    },
    to: (el, props) => {
      calls.push({ method: 'to', el, props });
      return mockTween;
    },
    _calls: calls,
    _reset: () => { calls.length = 0; mockTween.kill.mockClear(); },
    _tween: mockTween,
  };
  return { gsap: mockGsap };
});

// Import after mock setup
import DemoToggle from './DemoToggle.jsx';
import { gsap } from '../animation/gsapSetup.js';

describe('DemoToggle', () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    gsap._reset();
  });

  afterEach(() => {
    act(() => { root.unmount(); });
    document.body.removeChild(container);
  });

  function render(jsx) {
    act(() => { root.render(jsx); });
  }

  describe('rendering', () => {
    it('renders a button element', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn).not.toBeNull();
    });

    it('renders play icon when inactive', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      const svg = btn.querySelector('svg');
      expect(svg).not.toBeNull();
      // Play icon uses a path element (triangle)
      const path = svg.querySelector('path');
      expect(path).not.toBeNull();
    });

    it('renders stop icon when active', () => {
      render(<DemoToggle isActive={true} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      const svg = btn.querySelector('svg');
      expect(svg).not.toBeNull();
      // Stop icon uses a rect element (square)
      const rect = svg.querySelector('rect');
      expect(rect).not.toBeNull();
    });
  });

  describe('positioning (Requirement 1.1)', () => {
    it('has fixed positioning', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.position).toBe('fixed');
    });

    it('is positioned bottom 24px', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.bottom).toBe('24px');
    });

    it('is positioned right 24px', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.right).toBe('24px');
    });
  });

  describe('interaction (Requirements 1.2, 1.3)', () => {
    it('calls onToggle when clicked', () => {
      const onToggle = vi.fn();
      render(<DemoToggle isActive={false} onToggle={onToggle} />);
      const btn = container.querySelector('button.demo-toggle');
      act(() => { btn.click(); });
      expect(onToggle).toHaveBeenCalledTimes(1);
    });

    it('does not call onToggle when disabled', () => {
      const onToggle = vi.fn();
      render(<DemoToggle isActive={false} onToggle={onToggle} disabled={true} />);
      const btn = container.querySelector('button.demo-toggle');
      act(() => { btn.click(); });
      expect(onToggle).not.toHaveBeenCalled();
    });

    it('sets disabled attribute when disabled prop is true', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} disabled={true} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.disabled).toBe(true);
    });
  });

  describe('visual indicator (Requirement 1.4)', () => {
    it('has green border when active', () => {
      render(<DemoToggle isActive={true} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      // jsdom normalizes hex to rgb
      expect(btn.style.border).toContain('rgb(118, 185, 0)');
    });

    it('has transparent border when inactive', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.border).toContain('transparent');
    });

    it('starts pulsing animation when active', () => {
      render(<DemoToggle isActive={true} onToggle={() => {}} />);
      const toCalls = gsap._calls.filter(c => c.method === 'to');
      const pulse = toCalls.find(c => c.props.repeat === -1 && c.props.yoyo === true);
      expect(pulse).toBeDefined();
      expect(pulse.props.boxShadow).toContain('rgba(118, 185, 0');
    });

    it('kills pulse animation when becoming inactive', () => {
      render(<DemoToggle isActive={true} onToggle={() => {}} />);
      gsap._reset();
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      expect(gsap._tween.kill).toHaveBeenCalled();
    });
  });

  describe('disabled state', () => {
    it('reduces opacity when disabled', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} disabled={true} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.opacity).toBe('0.5');
    });

    it('disables pointer events when disabled', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} disabled={true} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.pointerEvents).toBe('none');
    });
  });

  describe('accessibility', () => {
    it('has aria-label "Start demo tour" when inactive', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.getAttribute('aria-label')).toBe('Start demo tour');
    });

    it('has aria-label "Stop demo tour" when active', () => {
      render(<DemoToggle isActive={true} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.getAttribute('aria-label')).toBe('Stop demo tour');
    });

    it('SVG icons are hidden from screen readers', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const svg = container.querySelector('svg');
      expect(svg.getAttribute('aria-hidden')).toBe('true');
    });
  });

  describe('sizing and shape', () => {
    it('is 48px wide', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.width).toBe('48px');
    });

    it('is 48px tall', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.height).toBe('48px');
    });

    it('is circular (border-radius 50%)', () => {
      render(<DemoToggle isActive={false} onToggle={() => {}} />);
      const btn = container.querySelector('button.demo-toggle');
      expect(btn.style.borderRadius).toBe('50%');
    });
  });
});
