// TheaterRunSummary.jsx — the run's summary, centred over the stage.
//
// Appears once, when the run reaches its recap. Dismissed by clicking anywhere
// outside it or pressing Escape; brought back by the info control in the request
// card header.
//
// The surface never says whether the prose was generated or derived. That is the
// established rule for this feature (BR3-28) and it holds here for the same
// reason: a reader cannot act on the difference, and labelling it would make the
// derived version read as a degraded one when it states the same facts.

import { useEffect, useRef } from "react";

/**
 * @param text     the summary prose
 * @param open     whether the surface is showing
 * @param onClose  called on outside click or Escape
 * @param subtitle short provenance line, e.g. the offers notice
 */
export function TheaterRunSummary({ text, open, onClose, subtitle }) {
  const panelRef = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (!open) return undefined;

    const onKeyDown = (e) => {
      if (e.key === "Escape") closeRef.current?.();
    };
    // Pointerdown rather than click: a click that began inside the panel and
    // ended outside it (a drag while selecting the text) would otherwise dismiss
    // the thing being read.
    const onPointerDown = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) closeRef.current?.();
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown, true);
    };
  }, [open]);

  if (!open || !text) return null;

  return (
    <div className="th-summary-scrim" data-testid="theater-run-summary">
      <div
        className="th-summary"
        ref={panelRef}
        role="dialog"
        aria-modal="false"
        aria-label="What happened in this auction"
      >
        <div className="th-summary-head">
          <span className="th-summary-title">What happened</span>
          <button
            type="button"
            className="th-summary-close"
            onClick={onClose}
            aria-label="Dismiss summary"
            data-testid="theater-run-summary-close"
          >
            ×
          </button>
        </div>
        <p className="th-summary-text" data-testid="theater-run-summary-text">{text}</p>
        {subtitle ? <p className="th-summary-sub">{subtitle}</p> : null}
      </div>
    </div>
  );
}
