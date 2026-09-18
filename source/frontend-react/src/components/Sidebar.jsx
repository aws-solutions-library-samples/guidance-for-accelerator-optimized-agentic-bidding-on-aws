import { useState, useCallback } from "react";
import ScenarioPicker from "./ScenarioPicker";
import { loadScenarioPayload } from "../utils/scenarioPayload.js";

/**
 * The sidebar is now only the scenario picker.
 *
 * The load-test launcher used to live here; it moved to the Governance page as
 * Step 1, alongside its own results view, so the two halves of the load test are
 * on one screen instead of split across two.
 */
export default function Sidebar({ submit, onOpenTheater, demoActive = false }) {
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
    <aside className="app-sidebar">
      <ScenarioPicker
        activeScenarioId={activeScenario}
        runningScenarioId={runningScenario}
        disabled={demoActive}
        onSelect={handleSelect}
        onSend={handleSend}
        onOpenTheater={handleOpenTheater}
      />
    </aside>
  );
}
