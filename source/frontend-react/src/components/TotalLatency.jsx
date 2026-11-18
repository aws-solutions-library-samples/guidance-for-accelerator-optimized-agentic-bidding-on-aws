import { useMemo } from "react";

/**
 * TotalLatency — where the SERVER-SIDE time went, broken into the two parts a
 * bidder can act on:
 *   - Agent processing (the containers' own time)
 *   - Orchestrator overhead (serialization, aggregation, routing)
 *
 * The containers run in four sequential stages (enrich, deals, yield, price), the
 * containers within a stage in parallel, and the orchestrator applies each stage's
 * mutations before the next stage runs. So agent time is the SUM of the per-stage
 * ceilings, which `result.stages` carries as each stage's `latencyMs`. A result
 * without stages (the bypassed baseline pass of a Theater run, or an orchestrator
 * that predates staging) falls back to the single parallel ceiling, the max over
 * the stops, which is what one stage looks like.
 *
 * It deliberately does NOT show the browser round-trip, and does not total the
 * two figures.
 *
 * The round-trip used to be a third segment, and it dominated the bar — 27ms of
 * 55ms in one observed run. But that leg is this demo's browser reaching
 * CloudFront and an ALB; in a real bid path the caller is an exchange on a
 * private path to the orchestrator, and nothing resembling it exists. Showing it
 * made the demo's own delivery look like part of the bidding cost, and it moved
 * with the reader's distance from the region rather than with anything the system
 * does. A total that included it inherited the same problem, which is why the
 * summary line went with it: the honest headline figure here is the pair, not
 * their sum with a browser hop folded in.
 */

/** Pure: the breakdown TotalLatency renders. Exported for tests. */
export function computeLatencyBreakdown(latencyMs, stops, stages) {
  if (latencyMs == null || latencyMs === 0) return null;

  const serverMs = Math.round(latencyMs);

  const stagedStages = (stages || []).filter((s) => s && s.latencyMs != null && s.latencyMs >= 0);
  if (stagedStages.length > 0) {
    const agentMs = stagedStages.reduce((sum, s) => sum + s.latencyMs, 0);
    return {
      mode: "staged",
      agentMs: Math.round(agentMs),
      orchestratorMs: Math.round(Math.max(0, serverMs - agentMs)),
      stageCount: stagedStages.length,
      stages: stagedStages.map((s) => ({
        name: s.name,
        latencyMs: Math.round(s.latencyMs * 10) / 10,
        containers: s.containers ?? [],
      })),
      containerCount: stagedStages.reduce((n, s) => n + (s.containers?.length ?? 0), 0),
    };
  }

  // Max container latency = the parallel execution ceiling
  const containerLatencies = (stops || [])
    .filter((s) => s.latency?.ms > 0 && s.id !== "ssp" && s.id !== "dsp")
    .map((s) => s.latency.ms);

  const maxContainerMs = containerLatencies.length > 0
    ? Math.max(...containerLatencies)
    : serverMs;

  return {
    mode: "parallel",
    agentMs: Math.round(maxContainerMs),
    orchestratorMs: Math.round(Math.max(0, serverMs - maxContainerMs)),
    stageCount: 0,
    stages: [],
    containerCount: containerLatencies.length,
  };
}

export default function TotalLatency({ latencyMs, stops, stages }) {
  const breakdown = useMemo(
    () => computeLatencyBreakdown(latencyMs, stops, stages),
    [latencyMs, stops, stages],
  );

  if (!breakdown) return null;

  const { agentMs, orchestratorMs, containerCount, mode, stageCount } = breakdown;
  const barTotal = agentMs + orchestratorMs || 1;

  const agentLabel = mode === "staged"
    ? `Agents (${stageCount} sequential stage${stageCount === 1 ? "" : "s"})`
    : `Agents (parallel, max of ${containerCount})`;
  // Per-stage detail as a tooltip: "enrich 9.5ms · deals 12ms · ..."
  const agentTitle = mode === "staged"
    ? breakdown.stages.map((s) => `${s.name} ${s.latencyMs}ms (${s.containers.length})`).join(" · ")
    : `Longest of ${containerCount} containers running in parallel`;

  return (
    <div
      className="total-latency-breakdown"
      role="status"
      aria-label={`Agent processing ${agentMs}ms, orchestrator overhead ${orchestratorMs}ms`}
      data-latency-mode={mode}
    >
      {/* Stacked bar */}
      <div className="latency-bar-stack">
        {agentMs > 0 && (
          <div
            className="latency-bar-segment latency-bar-segment--agents"
            style={{ width: `${(agentMs / barTotal) * 100}%` }}
            title={`Agent processing: ${agentMs}ms`}
          />
        )}
        {orchestratorMs > 0 && (
          <div
            className="latency-bar-segment latency-bar-segment--orchestrator"
            style={{ width: `${(orchestratorMs / barTotal) * 100}%` }}
            title={`Orchestrator: ${orchestratorMs}ms`}
          />
        )}
      </div>

      {/* Legend */}
      <div className="latency-breakdown-legend">
        <div className="latency-legend-item" title={agentTitle}>
          <span className="latency-legend-dot latency-legend-dot--agents" />
          <span className="latency-legend-label" data-testid="latency-agents-label">
            {agentLabel}
          </span>
          <span className="latency-legend-value" data-testid="latency-agents-value">{agentMs}ms</span>
        </div>
        <div className="latency-legend-item">
          <span className="latency-legend-dot latency-legend-dot--orchestrator" />
          <span className="latency-legend-label">Orchestrator overhead</span>
          <span className="latency-legend-value" data-testid="latency-orchestrator-value">{orchestratorMs}ms</span>
        </div>
      </div>

      {/* Per-stage strip, only when the orchestrator reported stages */}
      {mode === "staged" && (
        <div className="latency-stage-strip" data-testid="latency-stage-strip">
          {breakdown.stages.map((s, i) => (
            <span
              key={`${s.name}-${i}`}
              className="latency-stage-chip"
              title={s.containers.join(", ") || "no containers"}
            >
              <span className="latency-stage-chip-name">{i + 1}. {s.name}</span>
              <span className="latency-stage-chip-value">{s.latencyMs}ms</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
