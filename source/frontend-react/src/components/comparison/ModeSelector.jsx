// ModeSelector.jsx — Segmented control for switching between
// Standalone, RTB Fabric, and Side-by-Side comparison modes.

import { useComparison } from "./ComparisonContext";

const MODES = [
  { id: "standalone", label: "Standalone" },
  { id: "fabric", label: "RTB Fabric" },
  { id: "side-by-side", label: "Side-by-Side" },
];

export default function ModeSelector() {
  const { mode, setMode, fabricConfig } = useComparison();

  const fabricDisabled = !fabricConfig.valid;

  const handleKeyDown = (e, modeId) => {
    const currentIndex = MODES.findIndex((m) => m.id === modeId);
    let nextIndex = currentIndex;

    if (e.key === "ArrowRight" || e.key === "ArrowDown") {
      e.preventDefault();
      nextIndex = (currentIndex + 1) % MODES.length;
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
      e.preventDefault();
      nextIndex = (currentIndex - 1 + MODES.length) % MODES.length;
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (!isDisabled(modeId)) setMode(modeId);
      return;
    } else {
      return;
    }

    // Skip disabled options
    const nextMode = MODES[nextIndex];
    if (!isDisabled(nextMode.id)) {
      setMode(nextMode.id);
    }
  };

  function isDisabled(modeId) {
    return fabricDisabled && (modeId === "fabric" || modeId === "side-by-side");
  }

  return (
    <div className="mode-selector" role="radiogroup" aria-label="View mode">
      {MODES.map((m) => {
        const disabled = isDisabled(m.id);
        const active = mode === m.id;
        return (
          <button
            key={m.id}
            role="radio"
            aria-checked={active}
            aria-disabled={disabled}
            disabled={disabled}
            className={`mode-selector-btn ${active ? "mode-selector-btn--active" : ""}`}
            onClick={() => !disabled && setMode(m.id)}
            onKeyDown={(e) => handleKeyDown(e, m.id)}
            tabIndex={active ? 0 : -1}
            title={disabled ? "RTB Fabric endpoint not configured" : undefined}
          >
            {m.label}
          </button>
        );
      })}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {`View mode: ${MODES.find((m) => m.id === mode)?.label}`}
      </div>
    </div>
  );
}
