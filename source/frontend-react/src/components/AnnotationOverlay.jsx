import React, { useRef, useEffect } from "react";
import { gsap } from "../animation/gsapSetup.js";

/**
 * AnnotationOverlay — Injects an annotation label directly into the target element.
 *
 * Instead of trying to position a fixed/absolute overlay relative to a transformed
 * element (which is unreliable with CSS transforms), this component appends a child
 * div inside the target element itself. Since it's a child, it moves and scales
 * with the parent naturally.
 *
 * The annotation fades in after the zoom completes and fades out before the next step.
 *
 * @param {string|null} text - Annotation text to display
 * @param {boolean} visible - Controls show/hide state
 * @param {HTMLElement|null} targetElement - The DOM element being focused (annotation is injected into it)
 */
export default function AnnotationOverlay({ text, visible, targetElement }) {
  const injectedRef = useRef(null);

  useEffect(() => {
    // Clean up any previous injection
    if (injectedRef.current) {
      gsap.to(injectedRef.current, {
        autoAlpha: 0,
        duration: 0.2,
        onComplete: () => {
          injectedRef.current?.remove();
          injectedRef.current = null;
        },
      });
    }

    if (!visible || !text || !targetElement) return;

    // Create the annotation element and inject it into the target
    const el = document.createElement("div");
    el.className = "demo-annotation-injected";
    el.textContent = text;
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");

    // Style it as an overlay inside the target element
    Object.assign(el.style, {
      position: "absolute",
      bottom: "-36px",
      left: "50%",
      transform: "translateX(-50%)",
      backgroundColor: "rgba(0, 0, 0, 0.88)",
      color: "#ffffff",
      fontSize: "13px",
      lineHeight: "1.4",
      padding: "8px 14px",
      borderRadius: "6px",
      whiteSpace: "nowrap",
      maxWidth: "90%",
      overflow: "hidden",
      textOverflow: "ellipsis",
      pointerEvents: "none",
      zIndex: "10",
      boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
      opacity: "0",
      visibility: "hidden",
    });

    // Ensure the target has relative positioning for the absolute child
    const originalPosition = targetElement.style.position;
    if (!targetElement.style.position || targetElement.style.position === "static") {
      targetElement.style.position = "relative";
    }

    targetElement.appendChild(el);
    injectedRef.current = el;

    // Fade in
    gsap.to(el, { autoAlpha: 1, duration: 0.3, ease: "power2.out" });

    // Cleanup on unmount or when props change
    return () => {
      if (injectedRef.current) {
        injectedRef.current.remove();
        injectedRef.current = null;
      }
      // Restore position if we changed it (engine already manages this, but be safe)
      if (originalPosition !== undefined && targetElement.style.position === "relative") {
        targetElement.style.position = originalPosition;
      }
    };
  }, [visible, text, targetElement]);

  // This component doesn't render anything in the React tree —
  // it injects directly into the target DOM element
  return null;
}
