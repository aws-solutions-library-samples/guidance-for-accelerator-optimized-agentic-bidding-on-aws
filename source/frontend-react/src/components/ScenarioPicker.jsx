// ScenarioPicker.jsx — one toggle, one dropdown, one card.
//
// Replaces a flat list of every scenario card stacked in the sidebar. The toggle
// filters by ARTF surface (bid request vs bid response mutations), the dropdown
// names the scenarios on that surface with their applicable intents, and only the
// selected scenario renders its full card.
//
// The surface split is a real division in the system, not a presentational one:
// `orchestrator/app.py`'s CONTAINERS list gives exactly one container a
// response-side intent, so the two groups exercise different containers entirely.

import { useMemo, useState, useCallback } from "react";
import ScenarioCard, { SCENARIOS, SURFACES, SURFACE_REQUEST } from "./ScenarioCard.jsx";

/** Scenarios on one surface, in declaration order. */
export function scenariosOnSurface(surface) {
  return SCENARIOS.filter((s) => s.surface === surface);
}

export default function ScenarioPicker({
  activeScenarioId,
  runningScenarioId,
  disabled = false,
  onSelect,
  onSend,
  onOpenTheater,
}) {
  const [surface, setSurface] = useState(SURFACE_REQUEST);

  const options = useMemo(() => scenariosOnSurface(surface), [surface]);

  // The card shown is the active scenario when it is on the current surface, and
  // otherwise the first option. Falling back to "nothing selected" would leave an
  // empty panel after every toggle switch, which reads as a load failure.
  const shown = options.find((s) => s.id === activeScenarioId) ?? options[0] ?? null;

  const handleSurfaceChange = useCallback(
    (next) => {
      if (next === surface) return;
      setSurface(next);
      const first = scenariosOnSurface(next)[0];
      if (first) onSelect?.(first);
    },
    [surface, onSelect]
  );

  const handleDropdownChange = useCallback(
    (event) => {
      const picked = SCENARIOS.find((s) => s.id === event.target.value);
      if (picked) onSelect?.(picked);
    },
    [onSelect]
  );

  return (
    <div className="scenario-picker" data-testid="scenario-picker">
      <div
        className="scenario-surface-toggle"
        role="radiogroup"
        aria-label="Mutation surface"
        data-testid="scenario-surface-toggle"
      >
        {SURFACES.map(({ key, label }) => (
          <button
            key={key}
            type="button"
            role="radio"
            aria-checked={surface === key}
            className={`scenario-surface-option${surface === key ? " is-on" : ""}`}
            onClick={() => handleSurfaceChange(key)}
            disabled={disabled}
            data-testid={`scenario-surface-${key}`}
          >
            {label}
          </button>
        ))}
      </div>

      <label className="sidebar-label" htmlFor="scenario-select">
        Scenario
      </label>
      {/*
        A native <select>. Its options can only hold text, so the intents ride as
        text here and as coloured chips on the card and in the list below — the
        chips are the same `.tag` classes the card uses, so a colour means the same
        thing in both places.
      */}
      <select
        id="scenario-select"
        className="scenario-select"
        value={shown?.id ?? ""}
        onChange={handleDropdownChange}
        disabled={disabled || options.length === 0}
        data-testid="scenario-select"
      >
        {options.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name} — {s.tags.map((t) => t.label).join(", ")}
          </option>
        ))}
      </select>

      {shown ? (
        <div className="scenario-picker-intents" data-testid="scenario-picker-intents">
          {shown.tags.map((tag, i) => (
            <span key={i} className={`tag ${tag.cls}`}>{tag.label}</span>
          ))}
        </div>
      ) : null}

      {shown ? (
        <div className="scenarios">
          <ScenarioCard
            // Keyed on the scenario id so switching selection remounts the card.
            // Without this the tuner state would carry across scenarios, and a
            // response scenario's calibrated `defaults` would silently inherit
            // the previous scenario's slider positions — which is exactly what
            // decides its outcome class.
            key={shown.id}
            scenario={shown}
            isActive={activeScenarioId === shown.id}
            isLoading={runningScenarioId === shown.id}
            disabled={disabled}
            onSelect={onSelect}
            onSend={onSend}
            onOpenTheater={onOpenTheater}
          />
        </div>
      ) : null}
    </div>
  );
}
