import React, { useRef, useEffect } from "react";
import { gsap } from "../animation/gsapSetup.js";

/**
 * DemoToggle — Floating toggle button for activating/deactivating Demo Mode.
 *
 * Fixed position in the bottom-right corner. Displays a Play icon when inactive
 * and a Stop icon when active, with a pulsing border animation (GSAP) while
 * Demo Mode is running.
 *
 * @param {boolean} isActive - Whether demo mode is currently active
 * @param {function} onToggle - Callback fired when the button is clicked
 * @param {boolean} disabled - Whether the button is disabled (e.g., during loading)
 */
export default function DemoToggle({ isActive, onToggle, disabled = false }) {
  const buttonRef = useRef(null);
  const pulseRef = useRef(null); // GSAP tween reference for cleanup

  // Pulsing border animation when active
  useEffect(() => {
    const el = buttonRef.current;
    if (!el) return;

    if (isActive) {
      // Create a repeating pulse on the box-shadow to simulate pulsing border
      pulseRef.current = gsap.to(el, {
        boxShadow: "0 0 0 6px rgba(118, 185, 0, 0.4)",
        duration: 0.8,
        ease: "power2.inOut",
        repeat: -1,
        yoyo: true,
      });
    } else {
      // Kill the pulse animation and reset
      if (pulseRef.current) {
        pulseRef.current.kill();
        pulseRef.current = null;
      }
      gsap.set(el, { boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)" });
    }

    return () => {
      if (pulseRef.current) {
        pulseRef.current.kill();
        pulseRef.current = null;
      }
    };
  }, [isActive]);

  const handleClick = () => {
    if (!disabled && onToggle) {
      onToggle();
    }
  };

  return (
    <button
      ref={buttonRef}
      className="demo-toggle"
      onClick={handleClick}
      disabled={disabled}
      aria-label={isActive ? "Stop demo tour" : "Start demo tour"}
      style={{
        position: "fixed",
        bottom: "24px",
        right: "24px",
        width: "48px",
        height: "48px",
        borderRadius: "50%",
        border: isActive ? "2px solid #76b900" : "2px solid transparent",
        backgroundColor: isActive ? "#1a1a2e" : "#2d2d44",
        color: "#ffffff",
        cursor: disabled ? "not-allowed" : "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9100,
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
        opacity: disabled ? 0.5 : 1,
        pointerEvents: disabled ? "none" : "auto",
        transition: "border-color 0.2s ease",
        padding: 0,
        outline: "none",
      }}
    >
      {isActive ? (
        // Stop icon (square)
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
        >
          <rect x="2" y="2" width="12" height="12" rx="2" fill="#ffffff" />
        </svg>
      ) : (
        // Play icon (triangle)
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
        >
          <path d="M4 2L14 8L4 14V2Z" fill="#ffffff" />
        </svg>
      )}
    </button>
  );
}
