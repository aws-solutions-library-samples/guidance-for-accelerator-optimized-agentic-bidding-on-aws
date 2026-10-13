import { useState, useRef, useCallback } from "react";
import Header from "./components/Header";
import AuctionTheater from "./components/AuctionTheater";
import Sidebar from "./components/Sidebar";
import RawPanel from "./components/RawPanel";
import ContainersPanel from "./components/ContainersPanel";
import AdaptiveBiddingPanel from "./components/AdaptiveBiddingPanel";
import GovernancePanel from "./components/GovernancePanel";
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

/**
 * The guided demo tour. Kept in the codebase and switched off at one place rather
 * than deleted, so turning it back on is a one-line change rather than a revert.
 */
const SHOW_DEMO_TOUR = false;

function AppContent() {
  const [showContainers, setShowContainers] = useState(false);
  const [view, setView] = useState("scenarios");
  const [lastPayload, setLastPayload] = useState(null);
  // The scenario currently open in the embedded Theater, with the tuner values it
  // was opened with. Null means the main area shows the mutation timeline instead.
  const [theaterRun, setTheaterRun] = useState(null);
  const [demoActive, setDemoActive] = useState(false);
  const [annotationText, setAnnotationText] = useState(null);
  const [annotationVisible, setAnnotationVisible] = useState(false);
  const [annotationTarget, setAnnotationTarget] = useState(null);

  // The scenario panel collapses to a rail when the Theater opens, and the reader
  // can expand it again from there.
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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
    // Sending a scenario the ordinary way returns the main area to the timeline.
    setTheaterRun(null);
    return submitScenario(payload, "REST");
  };

  // Opening the Theater COLLAPSES the sidebar to a rail; it does not unmount it.
  // The Theater is a view inside this application, so the header and the scenario
  // panel stay visible around it. Expanding the rail again pushes the Theater
  // right and the layout scrolls horizontally rather than compressing the
  // walkthrough's three columns.
  //
  // Closing restores the panel, and the timeline for whatever ran last is still
  // there — the run is not discarded.
  const handleOpenTheater = useCallback((scenario, params) => {
    setTheaterRun({ scenario, params });
    setSidebarCollapsed(true);
  }, []);
  const handleCloseTheater = useCallback(() => {
    setTheaterRun(null);
    setSidebarCollapsed(false);
  }, []);
  const handleToggleSidebar = useCallback(() => setSidebarCollapsed((c) => !c), []);

  const theaterOpen = !!theaterRun;

  // For the raw panel, show the active result based on mode
  const activeResult = mode === "fabric" ? fabric.result : standalone.result;
  const activeLoading = mode === "fabric" ? fabric.loading : standalone.loading;
  const activeError = mode === "fabric" ? fabric.error : standalone.error;

  // Show scenario view only when a scenario has been submitted (not the default)
  const showScenarioView = !!(activeResult || lastPayload);

  // The Theater no longer removes the sidebar — it collapses it. The two pages
  // that genuinely own the whole width still do.
  const showSidebar = view !== "adaptive" && view !== "governance";

  return (
    <div className="app">
      <Header
        loading={activeLoading}
        error={activeError}
        onContainersClick={() => setShowContainers(true)}
        view={view}
        onViewChange={setView}
      />
      {/* Two markers, because the Theater sizes differently against a rail than
          against an expanded panel: against the rail it must fit the remaining
          space exactly, and only an expanded panel makes it refuse to compress
          and scroll horizontally instead. */}
      <div
        className={
          `app-layout${theaterOpen ? " app-layout--theater" : ""}` +
          `${theaterOpen && !sidebarCollapsed ? " app-layout--panel-open" : ""}`
        }
      >
        {showSidebar && (
          <Sidebar
            submit={handleSubmit}
            onOpenTheater={handleOpenTheater}
            demoActive={demoActive}
            collapsed={sidebarCollapsed}
            onToggleCollapse={handleToggleSidebar}
          />
        )}
        <main className="app-main">
          {view === "adaptive" ? (
            <AdaptiveBiddingPanel />
          ) : view === "governance" ? (
            <GovernancePanel />
          ) : theaterOpen ? (
            <AuctionTheater
              scenario={theaterRun.scenario}
              params={theaterRun.params}
              onExit={handleCloseTheater}
            />
          ) : (
            <>
              {/* Mode selector bar */}
              <div className="mode-selector-bar">
                <ModeSelector />
              </div>
              {/* Scenario timeline + Request JSON (shown only after a scenario is submitted) */}
              {showScenarioView && (
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
            </>
          )}
        </main>
      </div>
      {showContainers && <ContainersPanel onClose={() => setShowContainers(false)} />}
      {/* The demo tour is hidden, not removed. Its floating button sat above every
          other surface (inline zIndex 9100) and the guided walkthrough now covers
          what it was for. Flip SHOW_DEMO_TOUR to bring it back; the orchestrator,
          the engine and the toggle handler above are all still wired. */}
      {SHOW_DEMO_TOUR && <DemoToggle isActive={demoActive} onToggle={handleDemoToggle} />}
      <AnnotationOverlay text={annotationText} visible={annotationVisible} targetElement={annotationTarget} />

    </div>
  );
}

export default function App() {
  // The Theater used to be a full-screen surface at the hash route `#/theater`,
  // rendered outside ComparisonProvider and the app layout, reached from a top-nav
  // button. It is now embedded in the main area and opened from a scenario card,
  // so the route, the button and the hash-routing hook are all gone. One entry
  // point, and the Theater always has a scenario in hand.
  return (
    <ComparisonProvider>
      <AppContent />
    </ComparisonProvider>
  );
}
