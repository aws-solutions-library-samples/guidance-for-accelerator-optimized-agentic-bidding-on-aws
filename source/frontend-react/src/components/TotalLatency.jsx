import { useMemo } from "react";

/**
 * TotalLatency — where the SERVER-SIDE time went, broken into the two parts a
 * bidder can act on:
 *   - Agent processing (the ceiling of the concurrent containers)
 *   - Orchestrator overhead (serialization, aggregation, routing)
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
export default function TotalLatency({ latencyMs, stops }) {
  const breakdown = useMemo(() => {
    if (latencyMs == null || latencyMs === 0) return null;

    const serverMs = Math.round(latencyMs);

    // Max container latency = the parallel execution ceiling
    const containerLatencies = (stops || [])
      .filter((s) => s.latency?.ms > 0 && s.id !== "ssp" && s.id !== "dsp")
      .map((s) => s.latency.ms);

    const maxContainerMs = containerLatencies.length > 0
      ? Math.max(...containerLatencies)
      : serverMs;

    // Orchestrator overhead = total server time minus the longest container
    const orchestratorMs = Math.max(0, serverMs - maxContainerMs);

    return {
      agentMs: Math.round(maxContainerMs),
      orchestratorMs: Math.round(orchestratorMs),
      containerCount: containerLatencies.length,
    };
  }, [latencyMs, stops]);

  if (!breakdown) return null;

  const { agentMs, orchestratorMs, containerCount } = breakdown;
  const barTotal = agentMs + orchestratorMs || 1;

  return (
    <div
      className="total-latency-breakdown"
      role="status"
      aria-label={`Agent processing ${agentMs}ms, orchestrator overhead ${orchestratorMs}ms`}
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
        <div className="latency-legend-item">
          <span className="latency-legend-dot latency-legend-dot--agents" />
          <span className="latency-legend-label">
            Agents (parallel, max of {containerCount})
          </span>
          <span className="latency-legend-value">{agentMs}ms</span>
        </div>
        <div className="latency-legend-item">
          <span className="latency-legend-dot latency-legend-dot--orchestrator" />
          <span className="latency-legend-label">Orchestrator overhead</span>
          <span className="latency-legend-value">{orchestratorMs}ms</span>
        </div>
      </div>
    </div>
  );
}
