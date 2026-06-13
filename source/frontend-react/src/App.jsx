import { useState, useRef, useCallback } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import RawPanel from "./components/RawPanel";
import ContainersPanel from "./components/ContainersPanel";
import { LoadTestResults } from "./components/LoadTestPanel";
import BidBubbleOverlay from "./components/BidBubbleOverlay";
import DemoToggle from "./components/DemoToggle";
import AnnotationOverlay from "./components/AnnotationOverlay";
import useDemoAnimation from "./animation/useDemoAnimation";
import DemoSequenceOrchestrator from "./animation/DemoSequenceOrchestrator";
import {
  ComparisonProvider,
  useComparison,
  ModeSelector,
  ComparisonLayout,
} from "./components/comparison";

function AppContent() {
  const [showContainers, setShowContainers] = useState(false);
  const [lastPayload, setLastPayload] = useState(null);
  const [loadTestState, setLoadTestState] = useState(null);
  const [demoActive, setDemoActive] = useState(false);
  const [annotationText, setAnnotationText] = useState(null);
  const [annotationVisible, setAnnotationVisible] = useState(false);
  const [annotationTarget, setAnnotationTarget] = useState(null);

  const { mode, standalone, fabric, submitScenario } = useComparison();

  // Demo animation engine with annotation callbacks
  const { engine } = useDemoAnimation({
    onAnnotationShow: (text, element) => {
      setAnnotationText(text);
      setAnnotationTarget(element || null);
      setAnnotationVisible(true);
    },
    onAnnotationHide: () => {
      setAnnotationVisible(false);
      setAnnotationTarget(null);
    },
  });

  // DemoSequenceOrchestrator — stable ref, recreated only when engine/submitScenario change
  const orchestratorRef = useRef(null);

  // Submit function for the orchestrator: fetches sample JSON and submits via the real pipeline
  const demoSubmitFn = useCallback(async (scenario) => {
    const resp = await fetch(`/samples/${scenario.file}?t=${Date.now()}`);
    if (!resp.ok) throw new Error(`Failed to load sample: ${scenario.file}`);
    const payload = await resp.json();
    setLastPayload(payload);
    setLoadTestState(null);
    const result = await submitScenario(payload, "REST");
    return result;
  }, [submitScenario]);

  // Lazily create orchestrator (engine is stable, submitFn is stable via useCallback)
  if (!orchestratorRef.current) {
    orchestratorRef.current = new DemoSequenceOrchestrator(engine, demoSubmitFn);
  }
  // Keep submitFn up to date without recreating orchestrator
  orchestratorRef.current._submitFn = demoSubmitFn;

  const handleDemoToggle = useCallback(async () => {
    const orchestrator = orchestratorRef.current;
    if (!orchestrator) return;

    if (demoActive) {
      // Stop demo mode
      await orchestrator.stop();
      setDemoActive(false);
      setAnnotationVisible(false);
      setAnnotationText(null);
      setAnnotationTarget(null);
    } else {
      // Start demo mode
      setDemoActive(true);
      orchestrator.start().then(() => {
        // If start() resolves naturally (e.g., max failures), deactivate
        setDemoActive(false);
        setAnnotationVisible(false);
        setAnnotationText(null);
        setAnnotationTarget(null);
      });
    }
  }, [demoActive, engine]);

  const handleSubmit = async (payload) => {
    setLastPayload(payload);
    setLoadTestState(null); // Clear load test when running a scenario
    return submitScenario(payload, "REST");
  };

  // For the raw panel, show the active result based on mode
  const activeResult = mode === "fabric" ? fabric.result : standalone.result;
  const activeLoading = mode === "fabric" ? fabric.loading : standalone.loading;
  const activeError = mode === "fabric" ? fabric.error : standalone.error;

  // Is load test active (running or has results)?
  const loadTestActive = loadTestState && (loadTestState.running || loadTestState.result);

  // Show scenario view only when a scenario has been submitted (not the default)
  const showScenarioView = !!(activeResult || lastPayload);

  return (
    <div className="app">
      <Header
        loading={activeLoading}
        error={activeError}
        onContainersClick={() => setShowContainers(true)}
      />
      <div className="app-layout">
        <Sidebar
          onResult={(r) => { }}
          submit={handleSubmit}
          onLoadTestChange={setLoadTestState}
          demoActive={demoActive}
        />
        <main className="app-main">
          {/* Mode selector bar */}
          <div className="mode-selector-bar">
            <ModeSelector />
          </div>
          {/* Scenario timeline + Request JSON (shown only after a scenario is submitted) */}
          {showScenarioView && !loadTestActive && (
            <div className="main-top">
              <div className="main-top-left">
                <ComparisonLayout />
                <div className="raw-panel raw-panel--single">
                  <RawPanel
                    result={activeResult}
                    payload={lastPayload || activeResult?.submittedPayload}
                    section="mutations"
                  />
                </div>
              </div>
              <div className="main-top-right">
                <RawPanel
                  result={activeResult}
                  payload={lastPayload || activeResult?.submittedPayload}
                  section="request"
                />
              </div>
            </div>
          )}
          {/* Load test results — always visible as the default view */}
          <LoadTestResults
            progress={loadTestState?.progress}
            result={loadTestState?.result}
            running={loadTestState?.running || false}
            error={loadTestState?.error}
          />

          {/* Bid bubble overlay — scoped to main area width */}
          <BidBubbleOverlay
            running={loadTestState?.running}
            progress={loadTestState?.progress}
          />


        </main>
      </div>
      {showContainers && <ContainersPanel onClose={() => setShowContainers(false)} />}
      <DemoToggle isActive={demoActive} onToggle={handleDemoToggle} />
      <AnnotationOverlay text={annotationText} visible={annotationVisible} targetElement={annotationTarget} />

    </div>
  );
}

export default function App() {
  return (
    <ComparisonProvider>
      <AppContent />
    </ComparisonProvider>
  );
}
