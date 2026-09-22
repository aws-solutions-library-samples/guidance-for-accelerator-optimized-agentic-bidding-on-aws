import { useState, useCallback } from "react";
import ScenarioPicker from "./ScenarioPicker";
import { loadScenarioPayload } from "../utils/scenarioPayload.js";

/**
 * The sidebar is now only the scenario picker.
 *
 * The load-test launcher used to live here; it moved to the Governance page as
 * Step 1, alongside its own results view, so the two halves of the load test are
 * on one screen instead of split across two.
 *
 * `collapsed` narrows it to a rail instead of unmounting it. It used to be
 * unmounted while the Theater was open, which made the Theater read as a separate
 * page rather than a view inside this application. The picker stays MOUNTED when
 * collapsed — hidden with CSS, not conditionally rendered — so the selected
 * scenario and its tuner values survive expanding and collapsing.
 */
export default function Sidebar({
  submit,
  onOpenTheater,
  demoActive = false,
  collapsed = false,
  onToggleCollapse,
}) {
  const [activeScenario, setActiveScenario] = useState(null);
  const [runningScenario, setRunningScenario] = useState(null);

  const handleSelect = useCallback((scenario) => {
    setActiveScenario(scenario.id);
  }, []);

  const handleSend = useCallback(
    async (scenario, params) => {
      setRunningScenario(scenario.id);
      setActiveScenario(scenario.id);

      try {
        const payload = await loadScenarioPayload(scenario, params);
        await submit(payload);
      } catch (err) {
        console.error("Scenario send failed:", err);
      } finally {
        setRunningScenario(null);
      }
    },
    [submit]
  );

  // The Theater fetches and patches the payload itself (useTheaterRun.start takes
  // the scenario plus its params), so this only has to hand the pair upward.
  const handleOpenTheater = useCallback(
    (scenario, params) => {
      setActiveScenario(scenario.id);
      onOpenTheater?.(scenario, params);
    },
    [onOpenTheater]
  );

  return (
    <aside className={`app-sidebar${collapsed ? " app-sidebar--collapsed" : ""}`}>
      {onToggleCollapse ? (
        <button
          type="button"
          className="app-sidebar-toggle"
          onClick={onToggleCollapse}
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Expand scenarios" : "Collapse scenarios"}
          title={collapsed ? "Expand scenarios" : "Collapse scenarios"}
          data-testid="sidebar-toggle"
        >
          <span className="app-sidebar-toggle-chevron" aria-hidden="true">
            {collapsed ? "›" : "‹"}
          </span>
          <span className="app-sidebar-toggle-label">Scenarios</span>
        </button>
      ) : null}

      <div className="app-sidebar-body" {...(collapsed ? { hidden: true } : {})}>
        <ScenarioPicker
          activeScenarioId={activeScenario}
          runningScenarioId={runningScenario}
          disabled={demoActive}
          onSelect={handleSelect}
          onSend={handleSend}
          onOpenTheater={handleOpenTheater}
        />
      </div>
    </aside>
  );
}
