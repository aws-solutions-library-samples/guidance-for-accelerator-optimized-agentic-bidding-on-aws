import { useMemo, useRef, useState } from "react";
import GsapTooltip from "./GsapTooltip";
import { applyMutationsToEnvelope, stringifyWithPointers } from "../utils/artfApplier.js";
import { DISPLAY_NAME_BY_STOP_ID } from "../utils/intentMapping.js";

const INTENT_DESCRIPTIONS = {
  ACTIVATE_SEGMENTS: "Audience segments activated from user/location signals.",
  BID_SHADE: "The bid pricer predicted CTR and computed the optimal shaded bid price.",
  ACTIVATE_DEALS: "The deal scorer scored user-deal relevance and activated matching deals.",
  SUPPRESS_DEALS: "The deal scorer scored user-deal relevance and suppressed poor-fit deals.",
  ADD_METRICS: "Quality and measurement signals added to the bid request.",
  ADD_CIDS: "Identity tokens resolved from fragmented user/device signals.",
};

// Color map aligned to agent sources (matches FlowPipeline AGENT_NODES).
// Keyed by STOP ID, which is what a mutation's `sourceAgent` carries
// (utils/normalizer.js stamps the stop, not the container name). An earlier
// version keyed these by container name, so every lookup missed and the cards
// showed the bare stop id ("ncf") in the fallback colour.
const AGENT_COLORS = {
  dlrm: "#16a34a",
  widedeep: "#6366f1",
  ncf: "#d97706",
  metrics: "#0891b2",
  "yield-floor": "#be185d",
  "yield-margin": "#be185d",
};

// Job-oriented display labels, keyed by stop id (see intentMapping.js).
const AGENT_LABELS = DISPLAY_NAME_BY_STOP_ID;

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

  // The orchestrator's returned list, in application order (stage, then registry
  // order within a stage), each stamped with its producing stop. Falls back to
  // walking the stops for a result that predates `result.mutations`; that walk is
  // in DISPLAY order, which is not the application order, so the fallback is
  // only for rendering the mutation cards, never for the merge.
  const mutations = useMemo(() => {
    if (Array.isArray(result?.mutations) && result.mutations.length > 0) return result.mutations;
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

  // The merged "after" document: the submitted envelope with every mutation
  // applied by the same rules the orchestrator (shared/artf_applier.py) and the
  // Prebid hook (ArtfMutationApplier.java) use. Each applied mutation reports the
  // JSON pointers it wrote, and the serializer reports the line range of every
  // pointer, so the highlighted lines are the nodes that actually changed.
  //
  // This replaced a generic path writer that set `imp["imp-1"] = ids` for an
  // ACTIVATE_DEALS at /imp/imp-1: a string key on an array, which JSON.stringify
  // drops, so an activated deal never appeared in this view while the offers
  // panel (reading Prebid's response) showed it. Segments failed the same way.
  const mergedResult = useMemo(() => {
    if (!payload || mutations.length === 0) return null;
    const { envelope, dispositions } = applyMutationsToEnvelope(payload, mutations);
    const { text, ranges } = stringifyWithPointers(envelope);

    const insertions = []; // { path, agent, color, intent, applied, reason, lines: [start, end][] }
    mutations.forEach((m, i) => {
      const d = dispositions[i];
      const lineRanges = (d?.written ?? [])
        .map((ptr) => ranges.get(ptr))
        .filter(Boolean)
        .map((r) => [r.start, r.end]);
      insertions.push({
        path: m.path,
        agent: AGENT_LABELS[m.sourceAgent] || m.sourceAgent,
        color: AGENT_COLORS[m.sourceAgent] || "var(--accent)",
        intent: m.intent,
        applied: d?.applied === true,
        reason: d?.reason ?? null,
        lines: lineRanges,
      });
    });

    return { doc: envelope, text, insertions, applied: insertions.filter((x) => x.applied).length };
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
    // Render the merged JSON with the written nodes highlighted. The line ranges
    // come from the serializer, so a key such as "deals" or "data" that appears at
    // several depths is never confused with the one a mutation wrote.
    const mergedJson = mergedResult ? mergedResult.text : requestJson;

    const highlightedLines = new Map();
    if (mergedResult) {
      for (const ins of mergedResult.insertions) {
        for (const [start, end] of ins.lines) {
          for (let i = start; i <= end; i++) {
            highlightedLines.set(i, { color: ins.color, agent: ins.agent, intent: ins.intent, path: ins.path });
          }
        }
      }
    }
    const notApplied = mergedResult ? mergedResult.insertions.filter((x) => !x.applied) : [];

    return (
      <div className="raw-section raw-section-standalone">
        <div className="raw-section-header">
          {mergedResult ? "Request + Mutations Applied" : "Request JSON"}
          {mergedResult && (
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginLeft: 8 }}>
              ({mergedResult.applied} of {mergedResult.insertions.length} mutations applied)
            </span>
          )}
        </div>
        {notApplied.length > 0 && (
          <ul className="raw-not-applied" data-testid="raw-panel-not-applied">
            {notApplied.map((x, i) => (
              <li key={i}>
                <span style={{ color: x.color, fontWeight: 700 }}>{x.agent}</span>{" "}
                <code>{x.path}</code>: {x.reason}
              </li>
            ))}
          </ul>
        )}
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
