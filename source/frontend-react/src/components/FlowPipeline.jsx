import { useRef, useMemo, useState } from "react";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import TotalLatency from "./TotalLatency";
import GsapTooltip from "./GsapTooltip";

// ISV logos
import nvidiaLogo from "../logos/Nvidia_logo.svg";

/**
 * AGENT_NODES — one row per ARTF agent in the timeline.
 * Uses ARTF terminology: these are "agents" not "containers".
 * id must match the stop.id from the orchestrator response.
 */
const AGENT_NODES = [
  {
    id: "ssp",
    label: "SSP / Exchange",
    intents: [],
    model: "Supply-Side Platform — originates the bid request",
    logo: null,
    color: "#64748b",
    type: "endpoint",
  },
  {
    id: "dlrm",
    label: "DLRM Bid Shader",
    intents: ["BID_SHADE"],
    model: "DLRM (Deep Learning Recommendation Model)",
    logo: nvidiaLogo,
    color: "#16a34a",
    type: "agent",
  },
  {
    id: "ncf",
    label: "NCF Deal Manager",
    intents: ["ACTIVATE_DEALS", "SUPPRESS_DEALS"],
    model: "Neural Collaborative Filtering (NeuMF)",
    logo: nvidiaLogo,
    color: "#d97706",
    type: "agent",
  },
  {
    id: "widedeep",
    label: "Wide & Deep Activator",
    intents: ["ACTIVATE_SEGMENTS"],
    model: "Wide & Deep (Cheng et al. 2016)",
    logo: nvidiaLogo,
    color: "#6366f1",
    type: "agent",
  },
  {
    id: "metrics",
    label: "Metrics Enricher",
    intents: ["ADD_METRICS"],
    model: "Rule-based (viewability + brand safety)",
    logo: null,
    color: "#0891b2",
    type: "agent",
  },
  {
    id: "dsp",
    label: "DSP / Bidder",
    intents: [],
    model: "Demand-Side Platform — receives mutated bid request",
    logo: null,
    color: "#64748b",
    type: "endpoint",
  },
];

// Map stop IDs from orchestrator response to our node IDs
const STOP_ID_MAP = {
  "dlrm-bid-shader": "dlrm",
  "widedeep-segment-activator": "widedeep",
  "ncf-deal-manager": "ncf",
  "metrics-enricher": "metrics",
};

/**
 * Compute time layout from orchestrator result.
 * Agents run in parallel; we show each agent's latency as a bar.
 */
function computeTimeLayout(result) {
  if (!result?.stops) return null;

  const stopMap = {};
  for (const s of result.stops) {
    const nodeId = STOP_ID_MAP[s.id] || s.id;
    stopMap[nodeId] = s;
  }

  // Get max latency across all agents for scaling
  const latencies = AGENT_NODES.map((n) => stopMap[n.id]?.latency?.ms ?? 0);
  const maxLatency = Math.max(...latencies, 1);

  return { stopMap, maxLatency };
}

/**
 * FlowPipeline — horizontal timeline with one row per ARTF agent.
 * Logos on the left, GSAP tooltips on hover, color-coded bars.
 */
export default function FlowPipeline({ result, loading, error }) {
  const containerRef = useRef(null);
  const prevResultRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const timeData = useMemo(() => computeTimeLayout(result), [result]);

  // Mutations grouped by source agent
  const mutationsByAgent = useMemo(() => {
    if (!result?.stops) return {};
    const map = {};
    for (const stop of result.stops) {
      const nodeId = STOP_ID_MAP[stop.id] || stop.id;
      if (stop.mutations?.length > 0) {
        map[nodeId] = stop.mutations;
      }
    }
    return map;
  }, [result]);

  // Request peek summary
  const requestPeek = useMemo(() => {
    if (!result?.packet) return "";
    const p = result.packet;
    const parts = [];
    if (result.lifecycle) {
      parts.push(result.lifecycle.replace("LIFECYCLE_", "").replace(/_/g, " "));
    }
    if (p.impressionType && p.impressionType !== "unknown") parts.push(p.impressionType.toUpperCase());
    if (p.siteDomain) parts.push(p.siteDomain);
    if (p.bidFloor != null) parts.push(`floor: $${p.bidFloor.toFixed(2)}`);
    return parts.join(" · ");
  }, [result]);

  // GSAP animation for bars appearing
  useGSAP(() => {
    if (!result || !timeData || loading) return;
    if (prevResultRef.current === result) return;
    prevResultRef.current = result;

    const container = containerRef.current;
    if (!container) return;

    const bars = container.querySelectorAll(".agent-bar-fill");
    const latencyLabels = container.querySelectorAll(".agent-latency-value");

    // Reset
    gsap.set(bars, { scaleX: 0, transformOrigin: "left center" });
    gsap.set(latencyLabels, { opacity: 0 });

    // Stagger animate bars
    const tl = gsap.timeline();
    tl.to(bars, {
      scaleX: 1,
      duration: 0.6,
      ease: "power2.out",
      stagger: 0.08,
    }, 0.2);
    tl.to(latencyLabels, {
      opacity: 1,
      duration: 0.3,
      stagger: 0.08,
    }, 0.5);

    return () => { tl.kill(); };
  }, { scope: containerRef, dependencies: [result, timeData, loading] });

  return (
    <div id="flow-canvas" ref={containerRef}>
      {error && (
        <div className="flow-error">{error.message || "An error occurred"}</div>
      )}

      {loading && !result && (
        <div className="placeholder">
          <span className="spinner" /> Submitting to orchestrator…
        </div>
      )}

      {!result && !loading && !error && (
        <div className="placeholder">
          Select a scenario from the sidebar to visualize the ARTF pipeline
        </div>
      )}

      {result && timeData && (
        <div className="timeline-container">
          {requestPeek && <div className="flow-request-peek">{requestPeek}</div>}

          {/* Agent timeline rows */}
          <div className="agent-timeline">
            {AGENT_NODES.map((node) => {
              const stop = timeData.stopMap[node.id];
              const latencyMs = stop?.latency?.ms ?? 0;
              const barWidth = timeData.maxLatency > 0
                ? (latencyMs / timeData.maxLatency) * 100
                : 0;
              const hasMutations = mutationsByAgent[node.id]?.length > 0;
              const status = stop?.status || "idle";

              return (
                <div
                  key={node.id}
                  className={`agent-row ${status === "ok" ? "agent-row--active" : ""} ${node.type === "endpoint" ? "agent-row--endpoint" : ""}`}
                  data-node={node.id}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                >
                  {/* Logo */}
                  <div className="agent-row-logo">
                    {node.logo && <img src={node.logo} alt={node.label} />}
                    {!node.logo && <span className="agent-row-logo-placeholder">{node.type === "endpoint" ? "⬡" : ""}</span>}
                  </div>

                  {/* Agent name */}
                  <div className="agent-row-name" style={{ color: node.color }}>
                    {node.label}
                    {/* Tooltip */}
                    {node.type === "agent" && (
                      <GsapTooltip
                        visible={hoveredNode === node.id}
                        placement="top"
                        className="agent-tooltip"
                      >
                        <div className="agent-tooltip-model">{node.model}</div>
                        <div className="agent-tooltip-intents">
                          {node.intents.map((intent) => (
                            <span key={intent} className="agent-tooltip-intent">{intent}</span>
                          ))}
                        </div>
                      </GsapTooltip>
                    )}
                  </div>

                  {/* Bar */}
                  <div className="agent-row-bar">
                    {node.type === "agent" && (
                      <>
                        <div
                          className="agent-bar-fill"
                          style={{
                            width: `${barWidth}%`,
                            backgroundColor: `${node.color}22`,
                            borderColor: node.color,
                          }}
                        />
                        {hasMutations && (
                          <div className="agent-bar-mutations">
                            {mutationsByAgent[node.id].map((m, i) => (
                              <span
                                key={i}
                                className="agent-mutation-dot"
                                style={{ backgroundColor: node.color }}
                                title={m.intent}
                              />
                            ))}
                          </div>
                        )}
                      </>
                    )}
                    {node.type === "endpoint" && (
                      <div className="agent-endpoint-marker" style={{ borderColor: node.color }} />
                    )}
                  </div>

                  {/* Latency */}
                  <div className="agent-latency-value">
                    {node.type === "agent" && latencyMs > 0 ? `${latencyMs.toFixed(1)}ms` : ""}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Time axis */}
          <div className="timeline-axis">
            <span>0ms</span>
            <span>{(timeData.maxLatency * 0.25).toFixed(0)}ms</span>
            <span>{(timeData.maxLatency * 0.5).toFixed(0)}ms</span>
            <span>{(timeData.maxLatency * 0.75).toFixed(0)}ms</span>
            <span>{timeData.maxLatency.toFixed(0)}ms</span>
          </div>

          <TotalLatency latencyMs={result?.totalLatencyMs} browserElapsedMs={result?.browserElapsedMs} stops={result?.stops} />
        </div>
      )}
    </div>
  );
}
