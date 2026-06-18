// ComparisonLayout.jsx — Renders the correct layout based on active mode:
// - standalone: single FlowPipeline at full width
// - fabric: single FlowPipeline at full width (fabric endpoint)
// - side-by-side: split panel with both pipelines + latency summary

import { useComparison } from "./ComparisonContext";
import FlowPipeline from "../FlowPipeline";
import FabricAnnotations from "./FabricAnnotations";
import LatencySummary from "./LatencySummary";
import PanelError from "./PanelError";
import { extractTotalLatency } from "../../utils/normalizer";

export default function ComparisonLayout() {
  const { mode, standalone, fabric, submitScenario, lastPayload } = useComparison();

  const handleRetry = (panel) => {
    if (lastPayload) {
      submitScenario(lastPayload.payload, lastPayload.transport);
    }
  };

  if (mode === "standalone") {
    return (
      <div className="comparison-layout comparison-layout--single">
        <FlowPipeline
          result={standalone.result}
          loading={standalone.loading}
          error={standalone.error}
        />
      </div>
    );
  }

  if (mode === "fabric") {
    return (
      <div className="comparison-layout comparison-layout--single">
        {fabric.error && (
          <PanelError error={fabric.error} onRetry={() => handleRetry("fabric")} />
        )}
        <FabricAnnotations result={fabric.result} />
        <FlowPipeline
          result={fabric.result}
          loading={fabric.loading}
          error={fabric.error}
        />
      </div>
    );
  }

  // Side-by-side mode
  const standaloneLatency = standalone.result
    ? extractTotalLatency(standalone.result.raw, standalone.result.browserElapsedMs)
    : null;
  const fabricLatency = fabric.result
    ? extractTotalLatency(fabric.result.raw, fabric.result.browserElapsedMs)
    : null;
  const standaloneBrowserMs = standalone.result?.browserElapsedMs || null;
  const fabricBrowserMs = fabric.result?.browserElapsedMs || null;

  return (
    <div className="comparison-layout comparison-layout--split">
      {/* Left panel: Standalone */}
      <div className="comparison-panel comparison-panel--standalone">
        <div className="comparison-panel-header">
          <span className="comparison-panel-title">Standalone</span>
          <span className="comparison-panel-subtitle">CloudFront → NLB → Orchestrator</span>
        </div>
        {standalone.error && (
          <PanelError error={standalone.error} onRetry={() => handleRetry("standalone")} />
        )}
        <FlowPipeline
          result={standalone.result}
          loading={standalone.loading}
          error={standalone.error}
        />
      </div>

      {/* Center: Latency comparison */}
      <LatencySummary
        standaloneLatencyMs={standaloneLatency}
        fabricLatencyMs={fabricLatency}
        standaloneBrowserMs={standaloneBrowserMs}
        fabricBrowserMs={fabricBrowserMs}
        standaloneLoading={standalone.loading}
        fabricLoading={fabric.loading}
        standaloneError={!!standalone.error}
        fabricError={!!fabric.error}
      />

      {/* Right panel: RTB Fabric */}
      <div className="comparison-panel comparison-panel--fabric">
        <div className="comparison-panel-header">
          <span className="comparison-panel-title">RTB Fabric</span>
          <span className="comparison-panel-subtitle">CloudFront → Fabric Gateway → Modules → Orchestrator</span>
        </div>
        {fabric.error && (
          <PanelError error={fabric.error} onRetry={() => handleRetry("fabric")} />
        )}
        <FabricAnnotations result={fabric.result} />
        <FlowPipeline
          result={fabric.result}
          loading={fabric.loading}
          error={fabric.error}
        />
      </div>
    </div>
  );
}
