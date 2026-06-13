// FabricAnnotations.jsx — Overlay for fabric-specific pipeline nodes
// (Fabric Router, Rate Limiter, OpenRTB Filter). Only renders when
// the response includes fabric metadata — never fabricates annotations.

export default function FabricAnnotations({ result }) {
  if (!result?.raw?.metadata) return null;

  const metadata = result.raw.metadata;
  const networkPath = metadata.network_path || result.networkPath;
  const fabricModules = metadata.fabric_modules;
  const fabricLinkId = metadata.rtb_fabric_link_id;

  // Only show annotations if we have real evidence this went through RTB Fabric
  if (networkPath !== "rtb-fabric" && !fabricLinkId && !fabricModules) {
    return null;
  }

  return (
    <div className="fabric-annotations">
      {/* Network path indicator */}
      <div className="fabric-annotation-badge fabric-annotation-badge--path">
        <span className="fabric-annotation-icon">◆</span>
        <span className="fabric-annotation-label">RTB Fabric Network</span>
        {fabricLinkId && (
          <span className="fabric-annotation-detail">Link: {fabricLinkId}</span>
        )}
      </div>

      {/* Modules that processed this request */}
      {Array.isArray(fabricModules) && fabricModules.length > 0 && (
        <div className="fabric-annotation-modules">
          {fabricModules.map((mod, i) => (
            <div key={i} className="fabric-annotation-badge fabric-annotation-badge--module">
              <span className="fabric-annotation-icon">◇</span>
              <span className="fabric-annotation-label">{mod.name || "Module"}</span>
              {mod.latency_ms != null && (
                <span className="fabric-annotation-detail">{mod.latency_ms.toFixed(1)}ms</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
