import React, { useRef, useEffect } from "react";
import { gsap } from "gsap";

/**
 * Placement-based transform origins and initial offsets.
 * The tooltip animates from a slightly offset position toward its final placement.
 */
const PLACEMENT_CONFIG = {
  top: { transformOrigin: "center bottom", y: 10, x: 0 },
  bottom: { transformOrigin: "center top", y: -10, x: 0 },
  left: { transformOrigin: "right center", x: 10, y: 0 },
  right: { transformOrigin: "left center", x: -10, y: 0 },
};

/**
 * GsapTooltip — Reusable GSAP-animated elastic tooltip.
 *
 * Appears with `elastic.out(1.2, 0.4)` at 1× speed and disappears
 * with `power3.in` reverse at 2.5× speed.
 *
 * @param {React.ReactNode} children - Content rendered inside the tooltip bubble
 * @param {boolean} visible - Controls show/hide state
 * @param {"top"|"bottom"|"left"|"right"} [placement="top"] - Tooltip position relative to anchor
 * @param {string} [className] - Additional CSS class for styling variants
 */
export default function GsapTooltip({ children, visible, placement = "top", className }) {
  const tooltipRef = useRef(null);
  const tlRef = useRef(null);

  // Initialize GSAP timeline and set initial hidden state
  useEffect(() => {
    const el = tooltipRef.current;
    if (!el) return;

    const config = PLACEMENT_CONFIG[placement] || PLACEMENT_CONFIG.top;

    // Set initial hidden state: opacity 0, scale 0, offset by placement
    gsap.set(el, {
      autoAlpha: 0,
      scale: 0,
      y: config.y,
      x: config.x,
      transformOrigin: config.transformOrigin,
    });

    // Create paused timeline for enter animation
    tlRef.current = gsap.timeline({ paused: true }).to(el, {
      autoAlpha: 1,
      scale: 1,
      y: 0,
      x: 0,
      duration: 0.8,
      ease: "elastic.out(1.2, 0.4)",
    });

    // Cleanup: kill timeline on unmount
    return () => {
      if (tlRef.current) {
        tlRef.current.kill();
        tlRef.current = null;
      }
    };
  }, [placement]);

  // React to visibility changes
  useEffect(() => {
    if (!tlRef.current) return;

    if (visible) {
      tlRef.current.timeScale(1).play();
    } else {
      tlRef.current.timeScale(2.5).reverse();
    }
  }, [visible]);

  const placementClass = `gsap-tooltip--${placement}`;

  return (
    <div
      ref={tooltipRef}
      className={`gsap-tooltip ${placementClass}${className ? ` ${className}` : ""}`}
      style={{ visibility: "hidden" }}
    >
      {children}
    </div>
  );
}
