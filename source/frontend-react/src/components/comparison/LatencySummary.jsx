// LatencySummary.jsx — Displays latency comparison between standalone
// and RTB Fabric paths in Side-by-Side mode.
// Shows total round-trip (network + containers) for each path.

export default function LatencySummary({
  standaloneLatencyMs,
  fabricLatencyMs,
  standaloneBrowserMs,
  fabricBrowserMs,
  standaloneLoading,
  fabricLoading,
  standaloneError,
  fabricError,
}) {
  const hasStandalone = standaloneLatencyMs != null && !standaloneError;
  const hasFabric = fabricLatencyMs != null && !fabricError;
  const hasBoth = hasStandalone && hasFabric;

  // Browser-measured total includes network; server-reported is containers only
  const standaloneTotal = standaloneBrowserMs || standaloneLatencyMs;
  const fabricTotal = fabricBrowserMs || fabricLatencyMs;

  const standaloneNetwork = hasStandalone && standaloneBrowserMs > standaloneLatencyMs
    ? Math.round(standaloneBrowserMs - standaloneLatencyMs) : 0;
  const fabricNetwork = hasFabric && fabricBrowserMs > fabricLatencyMs
    ? Math.round(fabricBrowserMs - fabricLatencyMs) : 0;

  const delta = hasBoth && standaloneTotal && fabricTotal
    ? fabricTotal - standaloneTotal : null;
  const deltaStr = delta != null
    ? `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}ms`
    : null;

  return (
    <div className="latency-summary">
      <div className="latency-summary-row">
        <span className="latency-summary-label">Standalone:</span>
        <span className="latency-summary-value">
          {standaloneLoading ? (
            <span className="latency-loading">…</span>
          ) : standaloneError ? (
            <span className="latency-error">error</span>
          ) : hasStandalone ? (
            `${Math.round(standaloneTotal)}ms`
          ) : (
            "—"
          )}
        </span>
      </div>
      {hasStandalone && standaloneNetwork > 0 && (
        <div className="latency-summary-breakdown">
          <span className="latency-breakdown-detail">
            containers: {Math.round(standaloneLatencyMs)}ms · network: {standaloneNetwork}ms
          </span>
        </div>
      )}

      <div className="latency-summary-row">
        <span className="latency-summary-label">RTB Fabric:</span>
        <span className="latency-summary-value">
          {fabricLoading ? (
            <span className="latency-loading">…</span>
          ) : fabricError ? (
            <span className="latency-error">error</span>
          ) : hasFabric ? (
            `${Math.round(fabricTotal)}ms`
          ) : (
            "—"
          )}
        </span>
      </div>
      {hasFabric && fabricNetwork > 0 && (
        <div className="latency-summary-breakdown">
          <span className="latency-breakdown-detail">
            containers: {Math.round(fabricLatencyMs)}ms · network: {fabricNetwork}ms
          </span>
        </div>
      )}

      {deltaStr && (
        <div className="latency-summary-delta">
          <span className={`latency-delta ${delta < 0 ? "latency-delta--faster" : "latency-delta--slower"}`}>
            Fabric: {deltaStr}
          </span>
        </div>
      )}
    </div>
  );
}
