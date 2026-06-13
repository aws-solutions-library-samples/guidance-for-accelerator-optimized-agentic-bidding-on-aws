import { useState, useCallback } from "react";
import ScenarioCard, { SCENARIOS } from "./ScenarioCard";
import LoadTestPanel from "./LoadTestPanel";

const AGE_YOB_MAP = [
  { yobMin: 2002, yobMax: 2008 },
  { yobMin: 1992, yobMax: 2001 },
  { yobMin: 1982, yobMax: 1991 },
  { yobMin: 1972, yobMax: 1981 },
  { yobMin: 1962, yobMax: 1971 },
  { yobMin: 1940, yobMax: 1961 },
];

function applyTunerToPayload(payload, scenario, params) {
  const patched = JSON.parse(JSON.stringify(payload));
  const br = patched.bid_request || patched;
  const imp0 = br?.imp?.[0];

  if (imp0 && params.bidFloor != null) {
    imp0.bidfloor = params.bidFloor;
  }

  if (br?.user && params.ageRange != null) {
    const range = AGE_YOB_MAP[params.ageRange] || AGE_YOB_MAP[1];
    br.user.yob = Math.round((range.yobMin + range.yobMax) / 2);
  }

  if (scenario.id === "video-deals" && imp0?.pmp?.deals && params.numDeals != null) {
    const original = imp0.pmp.deals;
    if (params.numDeals <= original.length) {
      imp0.pmp.deals = original.slice(0, params.numDeals);
    }
  }

  // Model parameters
  const modelParams = {};
  if (params.shadeFactor != null && scenario.controls?.includes("shadeFactor")) {
    modelParams.shade_factor = params.shadeFactor;
  }
  if (params.convValue != null && scenario.controls?.includes("convValue")) {
    modelParams.conversion_value = params.convValue;
  }
  if (params.segThreshold != null && scenario.controls?.includes("segThreshold")) {
    modelParams.segment_threshold = params.segThreshold;
  }
  if (Object.keys(modelParams).length > 0) {
    patched.model_params = modelParams;
  }

  return patched;
}

export default function Sidebar({ onResult, submit, onLoadTestChange, demoActive = false }) {
  const [activeScenario, setActiveScenario] = useState(null);
  const [runningScenario, setRunningScenario] = useState(null);
  const [loadTestRunning, setLoadTestRunning] = useState(false);

  const handleSelect = useCallback((scenario) => {
    setActiveScenario(scenario.id);
  }, []);

  const handleSend = useCallback(
    async (scenario, params) => {
      setRunningScenario(scenario.id);
      setActiveScenario(scenario.id);

      try {
        const resp = await fetch(`/samples/${scenario.file}?t=${Date.now()}`);
        if (!resp.ok) throw new Error(`Failed to load sample: ${scenario.file}`);
        const rawPayload = await resp.json();

        const payload = applyTunerToPayload(rawPayload, scenario, params);
        console.log("[Sidebar] Sending with params:", params, "model_params:", payload.model_params);
        await submit(payload);
      } catch (err) {
        console.error("Scenario send failed:", err);
      } finally {
        setRunningScenario(null);
      }
    },
    [submit]
  );

  return (
    <aside className="app-sidebar">      
    <LoadTestPanel onRunningChange={setLoadTestRunning} onResultChange={onLoadTestChange} />
      <label className="sidebar-label">Scenarios</label>
      <div className="scenarios">
        {SCENARIOS.map((scenario) => (
          <ScenarioCard
            key={scenario.id}
            scenario={scenario}
            isActive={activeScenario === scenario.id}
            isLoading={runningScenario === scenario.id}
            disabled={loadTestRunning || demoActive}
            onSelect={handleSelect}
            onSend={handleSend}
          />
        ))}
      </div>
    </aside>
  );
}
