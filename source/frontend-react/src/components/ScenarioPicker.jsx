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

import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import ScenarioCard, { SCENARIOS, SURFACES, SURFACE_REQUEST } from "./ScenarioCard.jsx";

/** Scenarios on one surface, in declaration order. */
export function scenariosOnSurface(surface) {
  return SCENARIOS.filter((s) => s.surface === surface);
}

/** The intent chips for a scenario, in the same colours the card uses. */
function IntentChips({ scenario }) {
  return (
    <span className="scenario-combo-chips">
      {scenario.tags.map((tag, i) => (
        <span key={i} className={`tag ${tag.cls}`}>{tag.label}</span>
      ))}
    </span>
  );
}

/**
 * Scenario chooser.
 *
 * A custom listbox rather than a native `<select>`, because a native option can
 * only hold text: the intents had to be appended to the name as a comma-separated
 * string, which both overflowed the control and threw away the colour coding that
 * tells you at a glance which container family a scenario exercises. Here each row
 * is a name with its chips underneath.
 *
 * Built to the ARIA listbox pattern, so it keeps what the native control gave away
 * for free: the popup is a real `listbox` of `option`s, the trigger reports
 * `aria-expanded`, the focused row travels via `aria-activedescendant` rather than
 * by moving focus row to row, and arrows / Home / End / Enter / Escape all work.
 */
function ScenarioSelect({ options, selected, disabled, onPick }) {
  const [open, setOpen] = useState(false);
  // Which row the keyboard is on. Separate from the SELECTION: moving through the
  // list must not submit anything until the reader commits.
  const [cursor, setCursor] = useState(0);
  const rootRef = useRef(null);
  const listRef = useRef(null);
  const triggerRef = useRef(null);

  const selectedIndex = Math.max(0, options.findIndex((s) => s.id === selected?.id));

  const close = useCallback((returnFocus = true) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  }, []);

  const openList = useCallback(() => {
    setCursor(selectedIndex);
    setOpen(true);
  }, [selectedIndex]);

  // Focus the list when it opens so the arrow keys reach it without the reader
  // having to tab into it.
  useEffect(() => {
    if (open) listRef.current?.focus();
  }, [open]);

  // A click anywhere else closes it. Focus is NOT returned to the trigger here:
  // the reader is already somewhere else, and yanking focus back would fight them.
  useEffect(() => {
    if (!open) return;
    const onDocDown = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocDown);
    return () => document.removeEventListener("mousedown", onDocDown);
  }, [open]);

  const commit = useCallback(
    (scenario) => {
      onPick(scenario);
      close();
    },
    [onPick, close]
  );

  const onTriggerKeyDown = (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openList();
    }
  };

  const onListKeyDown = (event) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setCursor((c) => Math.min(options.length - 1, c + 1));
        break;
      case "ArrowUp":
        event.preventDefault();
        setCursor((c) => Math.max(0, c - 1));
        break;
      case "Home":
        event.preventDefault();
        setCursor(0);
        break;
      case "End":
        event.preventDefault();
        setCursor(options.length - 1);
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        if (options[cursor]) commit(options[cursor]);
        break;
      case "Escape":
        event.preventDefault();
        close();
        break;
      case "Tab":
        // Let focus leave, but do not leave an orphaned popup behind it.
        setOpen(false);
        break;
      default:
        break;
    }
  };

  const optionId = (scenario) => `scenario-option-${scenario.id}`;

  return (
    <div className="scenario-combo" ref={rootRef}>
      <button
        type="button"
        ref={triggerRef}
        className={`scenario-combo-trigger${open ? " is-open" : ""}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={selected ? `Scenario: ${selected.name}` : "Choose a scenario"}
        disabled={disabled || options.length === 0}
        onClick={() => (open ? close(false) : openList())}
        onKeyDown={onTriggerKeyDown}
        data-testid="scenario-select"
      >
        <span className="scenario-combo-value">
          {selected ? (
            <>
              <span className="scenario-combo-name">{selected.name}</span>
              <IntentChips scenario={selected} />
            </>
          ) : (
            <span className="scenario-combo-name">No scenarios on this surface</span>
          )}
        </span>
        <span className="scenario-combo-caret" aria-hidden="true" />
      </button>

      {open ? (
        <ul
          className="scenario-combo-list"
          role="listbox"
          tabIndex={-1}
          ref={listRef}
          aria-label="Scenarios"
          aria-activedescendant={options[cursor] ? optionId(options[cursor]) : undefined}
          onKeyDown={onListKeyDown}
          data-testid="scenario-select-list"
        >
          {options.map((scenario, i) => (
            <li
              key={scenario.id}
              id={optionId(scenario)}
              role="option"
              aria-selected={scenario.id === selected?.id}
              className={
                "scenario-combo-option" +
                (i === cursor ? " is-cursor" : "") +
                (scenario.id === selected?.id ? " is-selected" : "")
              }
              // mousedown, not click: the document listener above closes on
              // mousedown, so a click handler would never fire.
              onMouseDown={(e) => { e.preventDefault(); commit(scenario); }}
              onMouseEnter={() => setCursor(i)}
              data-testid={`scenario-option-${scenario.id}`}
            >
              <span className="scenario-combo-name">{scenario.name}</span>
              <IntentChips scenario={scenario} />
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
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

  const handlePick = useCallback(
    (picked) => {
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

      <span className="sidebar-label">Scenario</span>
      <ScenarioSelect
        options={options}
        selected={shown}
        disabled={disabled}
        onPick={handlePick}
      />

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
