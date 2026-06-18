import { forwardRef } from "react";

const FlowNode = forwardRef(function FlowNode(
  { id, label, sublabel, latency, status = "idle", shape = "rect", layerIcon, active, children },
  ref
) {
  const shapeClass =
    shape === "circle"
      ? "flow-stop flow-stop-agent"
      : shape === "pill"
        ? "flow-stop flow-stop-vertical"
        : "flow-stop";

  const statusAttr = active ? "processing" : status;

  return (
    <div
      ref={ref}
      className={shapeClass}
      data-node-id={id}
      data-status={statusAttr}
      role="button"
      tabIndex={0}
      aria-label={`${label}${latency ? ` — ${latency}ms` : ""}`}
    >
      {layerIcon && <span className="flow-layer-icon">{layerIcon}</span>}
      <span className="flow-stop-name">{label}</span>
      {sublabel && <span className="flow-stop-family">{sublabel}</span>}
      {latency != null && <span className="flow-latency">{latency}ms</span>}
      {children}
    </div>
  );
});

export default FlowNode;
