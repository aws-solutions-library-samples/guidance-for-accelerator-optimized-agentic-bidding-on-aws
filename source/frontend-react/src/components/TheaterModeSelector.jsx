// TheaterModeSelector.jsx — sell/buy mode for the Theater.
//
// The Theater is one route carrying a mode selector (FR-23). A mode absent from
// `availableModes` renders disabled with its reason.
//
// Availability is DATA, not a hardcoded branch (BR-23). Buy side arrives later by
// being added to the list — no component edit, and no stale reason string left
// behind for someone to find.

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };

/**
 * @param mode            currently selected mode id
 * @param availableModes  [{ id, label, available, unavailableReason }]
 * @param onChange        (modeId) => void
 */
export function TheaterModeSelector({ mode, availableModes, onChange }) {
  const modes = availableModes ?? [];

  return (
    <div className="th-mode-selector" data-testid="theater-mode-selector" role="group" aria-label="Theater mode">
      {modes.map((m) => {
        const isActive = m.id === mode;
        const disabled = m.available === false;
        return (
          <button
            key={m.id}
            type="button"
            className={`th-mode-option${isActive ? " is-active" : ""}${disabled ? " is-disabled" : ""}`}
            data-testid={`theater-mode-option-${m.id}`}
            aria-pressed={isActive}
            disabled={disabled}
            title={disabled ? m.unavailableReason : undefined}
            onClick={disabled ? undefined : () => onChange?.(m.id)}
          >
            {m.label}
            {disabled ? (
              <span
                className="th-mode-reason"
                style={SUBTLE}
                data-testid={`theater-mode-reason-${m.id}`}
              >
                {m.unavailableReason}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The mode list. Buy side is unavailable with a stated reason rather than absent,
 * so a viewer can see it is planned rather than wondering if it is missing.
 */
export const THEATER_MODES = Object.freeze([
  { id: "sell", label: "Sell side", available: true },
  {
    id: "buy",
    label: "Buy side",
    available: false,
    unavailableReason: "Not built yet — this release covers the sell side",
  },
]);
