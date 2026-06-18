import { useMemo } from "react";

/**
 * TotalLatency — bottom latency display showing end-to-end time
 * with a stacked breakdown of latency sources:
 *   - Agent processing (max of concurrent containers)
 *   - Orchestrator overhead (serialization, aggregation, routing)
 *   - Network (round-trip from browser to server and back)
 */
export default function TotalLatency({ latencyMs, browserElapsedMs, stops }) {
  const breakdown = useMemo(() => {
    if (latencyMs == null || latencyMs === 0) return null;

    const serverMs = Math.round(latencyMs);
    const browserMs = browserElapsedMs || 0;

    // Max container latency = the parallel execution ceiling
    const containerLatencies = (stops || [])
      .filter((s) => s.latency?.ms > 0 && s.id !== "ssp" && s.id !== "dsp")
      .map((s) => s.latency.ms);

    const maxContainerMs = containerLatencies.length > 0
      ? Math.max(...containerLatencies)
      : serverMs;

    // Orchestrator overhead = total server time minus the longest container
    const orchestratorMs = Math.max(0, serverMs - maxContainerMs);

    // Network = browser round-trip minus server processing
    const networkMs = browserMs > serverMs ? browserMs - serverMs : 0;

    const totalMs = browserMs > 0 ? browserMs : serverMs;

    return {
      totalMs,
      agentMs: Math.round(maxContainerMs),
      orchestratorMs: Math.round(orchestratorMs),
      networkMs: Math.round(networkMs),
      containerCount: containerLatencies.length,
    };
  }, [latencyMs, browserElapsedMs, stops]);

  if (!breakdown) return null;

  const { totalMs, agentMs, orchestratorMs, networkMs, containerCount } = breakdown;
  const barTotal = agentMs + orchestratorMs + networkMs || 1;

  return (
    <div className="total-latency-breakdown" role="status" aria-label={`Total latency: ${totalMs}ms`}>
      {/* Summary line */}
      <div className="total-latency-summary">
        Total <strong>{totalMs}ms</strong>
      </div>

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
        {networkMs > 0 && (
          <div
            className="latency-bar-segment latency-bar-segment--network"
            style={{ width: `${(networkMs / barTotal) * 100}%` }}
            title={`Network: ${networkMs}ms`}
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
        {networkMs > 0 && (
          <div className="latency-legend-item">
            <span className="latency-legend-dot latency-legend-dot--network" />
            <span className="latency-legend-label">Network (round-trip)</span>
            <span className="latency-legend-value">{networkMs}ms</span>
          </div>
        )}
      </div>
    </div>
  );
}
