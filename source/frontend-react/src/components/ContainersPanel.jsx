import { useState, useEffect, useCallback } from "react";
import GpuControl from "./GpuControl";
import { authFetch } from "../authFetch.js";

const MODELS = {
  "dlrm-bid-shader": "DLRM · BID_SHADE",
  "widedeep-segment-activator": "Wide & Deep · ACTIVATE_SEGMENTS",
  "ncf-deal-manager": "NCF · DEALS",
  "metrics-enricher": "Rules · ADD_METRICS",
};

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };
const SUBTLER = { fontSize: "10px", color: "var(--text-muted)", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" };

function formatProbe(label, probe) {
  if (!probe) return null;
  const ts = probe.checkedAt ? new Date(probe.checkedAt).toLocaleTimeString() : "—";
  const latency = probe.latencyMs != null ? `${probe.latencyMs} ms` : "—";
  const resolved = probe.resolvedAddress || "unresolved";
  const target = probe.url || probe.target || "";
  const result = probe.ok
    ? (probe.httpStatus ? `HTTP ${probe.httpStatus}` : "ok")
    : (probe.error ? `err: ${probe.error}` : (probe.httpStatus ? `HTTP ${probe.httpStatus}` : "fail"));
  return (
    <div style={{ marginTop: "4px" }}>
      <div style={SUBTLE}>{label} · {result} · {latency} · @{ts}</div>
      <div style={SUBTLER}>→ {target} → {resolved}</div>
    </div>
  );
}

/**
 * ContainersPanel — slide-out panel with GPU control + container health.
 * Matches the vanilla frontend's Container Health panel.
 */
export default function ContainersPanel({ onClose }) {
  const [containers, setContainers] = useState([]);
  const [triton, setTriton] = useState(null);
  const [urlsNote, setUrlsNote] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const resp = await authFetch("/api/v1/containers");
      if (resp.ok) {
        const data = await resp.json();
        setContainers(data.containers || []);
        setTriton(data.triton || null);
        setUrlsNote(data.urlsNote || null);
      }
    } catch (_) {
      // keep existing state
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <>
      <div className="containers-panel-backdrop" onClick={onClose} />
      <div className="containers-panel" role="dialog" aria-label="Container Health">
        <div className="containers-panel-header">
          <h2>Container Health</h2>
          <button className="containers-panel-close" onClick={onClose} aria-label="Close panel">
            ×
          </button>
        </div>

        <div className="info-bar" style={{ margin: "0 0 12px" }}>
          <div>Queried via orchestrator — each container exposes gRPC + MCP + Health</div>
          <button className="btn-secondary" onClick={refresh} style={{ padding: "5px 12px" }}>
            Refresh
          </button>
        </div>

        {urlsNote && (
          <div style={{ ...SUBTLE, margin: "0 0 12px", padding: "8px 10px", border: "1px solid var(--border)", borderRadius: "6px" }}>
            <strong>Note:</strong> {urlsNote}
          </div>
        )}

        {/* Single GPU control for the whole node group */}
        <GpuControl />

        {/* Triton evidence */}
        {triton && (
          <div className="container-item" style={{ flexDirection: "column", alignItems: "stretch" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <div>
                <strong>Triton Inference Server</strong>
                <br />
                <span style={SUBTLER}>{triton.url} <span style={SUBTLE}>(cluster DNS)</span></span>
              </div>
              <span className={`container-status ${triton.ready ? "ready" : "unreachable"}`}>
                {triton.ready ? "ready" : "offline"}
              </span>
            </div>
            {triton.evidence && formatProbe("/v2/health/ready", triton.evidence.healthProbe)}
            {triton.evidence?.modelProbes && Object.entries(triton.evidence.modelProbes).map(([name, p]) => (
              <div key={name}>{formatProbe(`/v2/models/${name}/ready`, p)}</div>
            ))}
          </div>
        )}

        {/* Container list */}
        {loading ? (
          <div className="placeholder"><span className="spinner" /> Loading…</div>
        ) : (
          <div className="containers-list">
            {containers.map((c) => {
              const statusCls =
                c.status === "ready" ? "ready" :
                c.status === "degraded" ? "degraded" : "unreachable";
              const statusLabel =
                c.status === "ready" ? "ready" :
                c.status === "degraded" && c.inferenceStatus === "gpu_offline" ? "gpu offline" :
                c.status === "degraded" ? c.inferenceStatus || "degraded" :
                "unreachable";

              return (
                <div key={c.name} className="container-item" style={{ flexDirection: "column", alignItems: "stretch" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                    <div>
                      <strong>{c.name}</strong>
                      <br />
                      <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        {MODELS[c.name] || ""}
                      </span>
                      <br />
                      <span style={SUBTLER}>
                        gRPC: {c.grpc} · MCP: {c.mcp} <span style={SUBTLE}>(cluster DNS)</span>
                      </span>
                      {c.tritonModel && (
                        <>
                          <br />
                          <span style={SUBTLE}>
                            Triton model: {c.tritonModel} ({c.inferenceStatus})
                          </span>
                        </>
                      )}
                    </div>
                    <span className={`container-status ${statusCls}`}>{statusLabel}</span>
                  </div>
                  {formatProbe("MCP /health/ready", c.evidence?.httpProbe)}
                  {formatProbe("gRPC channel_ready", c.evidence?.grpcProbe)}
                  {formatProbe("Triton model probe", c.evidence?.tritonModelProbe)}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
