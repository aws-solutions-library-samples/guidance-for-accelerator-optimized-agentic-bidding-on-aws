import { useMemo, useRef, useState } from "react";
import GsapTooltip from "./GsapTooltip";

const INTENT_DESCRIPTIONS = {
  ACTIVATE_SEGMENTS: "Audience segments activated from user/location signals.",
  BID_SHADE: "DLRM predicted CTR and computed the optimal shaded bid price.",
  ACTIVATE_DEALS: "NCF scored user-deal relevance and activated matching deals.",
  SUPPRESS_DEALS: "NCF scored user-deal relevance and suppressed poor-fit deals.",
  ADD_METRICS: "Quality and measurement signals added to the bid request.",
  ADD_CIDS: "Identity tokens resolved from fragmented user/device signals.",
};

// Color map aligned to agent sources (matches FlowPipeline AGENT_NODES)
const AGENT_COLORS = {
  "dlrm-bid-shader": "#16a34a",
  "widedeep-segment-activator": "#6366f1",
  "ncf-deal-manager": "#d97706",
  "metrics-enricher": "#0891b2",
};

const AGENT_LABELS = {
  "dlrm-bid-shader": "DLRM",
  "widedeep-segment-activator": "Wide & Deep",
  "ncf-deal-manager": "NCF",
  "metrics-enricher": "Metrics",
};

function intentColorClass(intent) {
  if (intent.includes("SHADE")) return "shade";
  if (intent.includes("SEGMENT")) return "seg";
  if (intent.includes("DEAL")) return "deal";
  return "metric";
}

/**
 * MutationLine — A single highlighted mutation line with hover handlers.
 * Uses a short delay on mouse leave to allow adjacent mutation transitions
 * without flickering the tooltip.
 */
function MutationLine({ line, highlight, onHover, leaveTimerRef, containerRef }) {
  const handleMouseEnter = (e) => {
    // Cancel any pending leave timeout (handles adjacent mutation transitions)
    if (leaveTimerRef.current) {
      clearTimeout(leaveTimerRef.current);
      leaveTimerRef.current = null;
    }
    // Calculate position relative to the scrollable container
    const container = containerRef.current;
    const rect = e.currentTarget.getBoundingClientRect();
    const containerRect = container ? container.getBoundingClientRect() : rect;
    const top = rect.top - containerRect.top + container.scrollTop;
    onHover({
      agent: highlight.agent,
      intent: highlight.intent,
      path: highlight.path,
      color: highlight.color,
      top,
    });
  };

  const handleMouseLeave = () => {
    // Delay clearing to allow adjacent mutation enter to fire first
    leaveTimerRef.current = setTimeout(() => {
      onHover(null);
      leaveTimerRef.current = null;
    }, 30);
  };

  return (
    <span
      className="raw-json-mutation-line"
      style={{ backgroundColor: `${highlight.color}15`, borderLeftColor: highlight.color }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {line}{"\n"}
    </span>
  );
}

/**
 * RawPanel — shows the request JSON or formatted mutations.
 * When section="mutations", renders only the mutations card.
 * When section="request", renders only the request JSON card.
 * When no section is specified, renders both in a grid (legacy behavior).
 */
export default function RawPanel({ result, payload, section }) {
  const containerRef = useRef(null);
  const [hoveredMutation, setHoveredMutation] = useState(null);
  const leaveTimerRef = useRef(null);

  const requestJson = useMemo(() => {
    if (!payload) return "";
    return JSON.stringify(payload, null, 2);
  }, [payload]);

  const mutations = useMemo(() => {
    if (!result?.stops) return [];
    const all = [];
    for (const stop of result.stops) {
      if (stop.mutations?.length > 0) {
        for (const m of stop.mutations) {
          all.push({ ...m, sourceAgent: stop.id });
        }
      }
    }
    return all;
  }, [result]);

  // Build the merged "after" JSON: apply mutations into the request at their paths
  const mergedResult = useMemo(() => {
    if (!payload || mutations.length === 0) return null;
    const doc = JSON.parse(JSON.stringify(payload));

    // Apply each mutation into the document
    const insertions = []; // {path, agent, color, data}
    for (const m of mutations) {
      const agentColor = AGENT_COLORS[m.sourceAgent] || "var(--accent)";
      const agentLabel = AGENT_LABELS[m.sourceAgent] || m.sourceAgent;
      const pathParts = (m.path || "").split("/").filter(Boolean);

      // Navigate into the doc and insert the payload
      let target = doc.bid_request || doc;
      for (let i = 0; i < pathParts.length - 1; i++) {
        const key = pathParts[i];
        if (key === "bid_request" || key === "bid_response") continue;
        if (target[key] !== undefined) {
          target = target[key];
        } else if (Array.isArray(target)) {
          const found = target.find(item => item?.id === key);
          if (found) target = found;
          else break;
        } else {
          target[key] = {};
          target = target[key];
        }
      }

      // Insert at the final path key
      const lastKey = pathParts[pathParts.length - 1];
      if (lastKey && m.payload) {
        // Merge the payload into the target
        if (m.payload.metric) {
          target[lastKey] = m.payload.metric;
        } else if (m.payload.id) {
          target[lastKey] = m.payload.id;
        } else if (m.payload.price != null) {
          target[lastKey] = m.payload.price;
        } else {
          target[lastKey] = m.payload;
        }
      }

      insertions.push({ path: m.path, agent: agentLabel, color: agentColor, intent: m.intent });
    }

    return { doc, insertions };
  }, [payload, mutations]);

  if (section === "mutations") {
    return (
      <div className="raw-section raw-section-standalone" ref={containerRef}>
        <div className="raw-section-header">Mutations</div>
        {mutations.length === 0 ? (
          <div className="raw-placeholder">No mutations yet</div>
        ) : (
          <div className="raw-mutations">
            {mutations.map((m, i) => {
              const agentColor = AGENT_COLORS[m.sourceAgent] || "var(--text-muted)";
              const agentLabel = AGENT_LABELS[m.sourceAgent] || m.sourceAgent;
              return (
                <div
                  key={i}
                  className={`raw-mutation ${intentColorClass(m.intent)}`}
                  style={{ borderLeft: `3px solid ${agentColor}` }}
                >
                  <div className="raw-mutation-header">
                    <span
                      className="raw-mutation-source"
                      style={{ color: agentColor, fontWeight: 700, fontSize: "0.7rem" }}
                    >
                      {agentLabel}
                    </span>
                    <span className={`tag ${intentColorClass(m.intent)}`}>{m.intent}</span>
                    <span className="raw-mutation-op">{m.op}</span>
                    <code className="raw-mutation-path">{m.path}</code>
                  </div>
                  <p className="raw-mutation-desc">
                    {INTENT_DESCRIPTIONS[m.intent] || ""}
                  </p>
                  <pre className="raw-mutation-payload" style={{ borderColor: agentColor }}>
                    {JSON.stringify(m.payload, null, 2)}
                  </pre>
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  if (section === "request") {
    // Render the merged JSON with mutation insertions highlighted
    const mergedJson = mergedResult
      ? JSON.stringify(mergedResult.doc, null, 2)
      : requestJson;

    // Build highlight map from mergedResult
    const highlightedLines = new Map();
    if (mergedResult && mergedJson) {
      for (const ins of mergedResult.insertions) {
        const lastKey = ins.path.split("/").filter(Boolean).pop();
        if (!lastKey) continue;
        const lines = mergedJson.split("\n");
        let inBlock = false;
        let depth = 0;
        for (let i = 0; i < lines.length; i++) {
          if (!inBlock && lines[i].includes(`"${lastKey}"`)) {
            inBlock = true;
            depth = 0;
            highlightedLines.set(i, { color: ins.color, agent: ins.agent, intent: ins.intent, path: ins.path });
          } else if (inBlock) {
            highlightedLines.set(i, { color: ins.color, agent: ins.agent, intent: ins.intent, path: ins.path });
            for (const ch of lines[i]) {
              if (ch === '{' || ch === '[') depth++;
              if (ch === '}' || ch === ']') depth--;
            }
            if (depth <= 0 && (lines[i].trim().endsWith(']') || lines[i].trim().endsWith('}') || lines[i].trim().endsWith('],') || lines[i].trim().endsWith('},'))) {
              inBlock = false;
            }
          }
        }
      }
    }

    return (
      <div className="raw-section raw-section-standalone">
        <div className="raw-section-header">
          {mergedResult ? "Request + Mutations Applied" : "Request JSON"}
          {mergedResult && (
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginLeft: 8 }}>
              ({mergedResult.insertions.length} mutations merged)
            </span>
          )}
        </div>
        <div style={{ position: "relative" }} ref={containerRef}>
          <pre className="raw-json">
            {(mergedJson || "Select a scenario").split("\n").map((line, i) => {
              const highlight = highlightedLines.get(i);
              if (highlight) {
                return (
                  <MutationLine
                    key={i}
                    line={line}
                    highlight={highlight}
                    onHover={setHoveredMutation}
                    leaveTimerRef={leaveTimerRef}
                    containerRef={containerRef}
                  />
                );
              }
              return <span key={i}>{line}{"\n"}</span>;
            })}
          </pre>
          <div style={{ position: "absolute", top: hoveredMutation?.top ?? 0, left: 0, pointerEvents: "none" }}>
            <GsapTooltip
              visible={!!hoveredMutation}
              placement="top"
              className="mutation-tooltip"
            >
              {hoveredMutation?.agent && (
                <div className="mutation-tooltip-agent">{hoveredMutation.agent}</div>
              )}
              {hoveredMutation?.intent && (
                <div className="mutation-tooltip-intent">
                  {INTENT_DESCRIPTIONS[hoveredMutation.intent] || hoveredMutation.intent}
                </div>
              )}
              {hoveredMutation?.path && (
                <code className="mutation-tooltip-path">{hoveredMutation.path}</code>
              )}
            </GsapTooltip>
          </div>
        </div>
      </div>
    );
  }

  // Legacy: render both in a grid
  return (
    <div className="raw-panel" ref={containerRef}>
      <div className="raw-section">
        <div className="raw-section-header">Request JSON</div>
        <pre className="raw-json">{requestJson || "Select a scenario to see the request"}</pre>
      </div>
      <div className="raw-section">
        <div className="raw-section-header">Mutations</div>
        {mutations.length === 0 ? (
          <div className="raw-placeholder">No mutations yet</div>
        ) : (
          <div className="raw-mutations">
            {mutations.map((m, i) => (
              <div key={i} className={`raw-mutation ${intentColorClass(m.intent)}`}>
                <div className="raw-mutation-header">
                  <span className={`tag ${intentColorClass(m.intent)}`}>{m.intent}</span>
                  <span className="raw-mutation-op">{m.op}</span>
                  <code className="raw-mutation-path">{m.path}</code>
                </div>
                <p className="raw-mutation-desc">
                  {INTENT_DESCRIPTIONS[m.intent] || ""}
                </p>
                <pre className="raw-mutation-payload">
                  {JSON.stringify(m.payload, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
