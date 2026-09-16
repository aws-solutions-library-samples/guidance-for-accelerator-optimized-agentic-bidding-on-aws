import { useState, useEffect, useCallback } from "react";
import GpuControl from "./GpuControl";
import { authFetch } from "../authFetch.js";

// Job-oriented display labels, keyed by the unchanged internal container
// name — see RENAME_MAP.md. Matches FlowPipeline.jsx/RawPanel.jsx/
// LoadTestPanel.jsx's CONTAINER_LABELS/AGENT_LABELS convention.
const CONTAINER_LABELS = {
  "dlrm-bid-shader": "Bid Pricer",
  "widedeep-segment-activator": "Audience Activator",
  "ncf-deal-manager": "Deal Scorer",
  "metrics-enricher": "Signals Enricher",
  "yield-optimizer-floor": "Yield Optimizer — Floor",
  "yield-optimizer-margin": "Yield Optimizer — Margin",
};

// widedeep-segment-activator no longer scores with the Wide & Deep model —
// that model's ONNX graph couldn't compile to TensorRT (BatchNorm1d fusion
// limitation), so it was replaced with deterministic rules over bid-request
// signals (see containers/widedeep_segment_activator/app.py's docstring).
// FlowPipeline.jsx/LoadTestPanel.jsx already reflect this ("Logic"/"Rules");
// this lookup previously still named the retired model.
const MODELS = {
  "dlrm-bid-shader": "DLRM · BID_SHADE",
  "widedeep-segment-activator": "Rules · ACTIVATE_SEGMENTS",
  "ncf-deal-manager": "NCF · DEALS",
  "metrics-enricher": "Rules · ADD_METRICS",
};

const SUBTLE = { fontSize: "10px", color: "var(--text-muted)" };
const SUBTLER = { fontSize: "10px", color: "var(--text-muted)", fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" };

// Every status the orchestrator can report, mapped explicitly. Previously
// anything not "ready" or "degraded" fell through to "unreachable", which would
// have rendered a deliberately deactivated container as a failure — the one
// distinction someone bringing up their own container most needs.
// Source of truth: source/orchestrator/container_registry.py.
const STATUS_PRESENTATION = {
  ready:        { cls: "ready",       label: "ready" },
  ok:           { cls: "ready",       label: "ready" },
  no_mutations: { cls: "no-mutations", label: "no changes" },
  degraded:     { cls: "degraded",    label: "degraded" },
  disabled:     { cls: "disabled",    label: "inactive" },
  skipped:      { cls: "disabled",    label: "not applicable" },
  timeout:      { cls: "unreachable", label: "timeout" },
  error:        { cls: "unreachable", label: "error" },
  unreachable:  { cls: "unreachable", label: "unreachable" },
};

function presentStatus(c) {
  // gpu_offline is a degraded sub-state worth naming: the container is fine, the
  // GPU node is stopped. Kept ahead of the table lookup because it depends on
  // two fields, not one.
  if (c.status === "degraded" && c.inferenceStatus === "gpu_offline") {
    return { cls: "degraded", label: "gpu offline" };
  }
  const known = STATUS_PRESENTATION[c.status];
  if (known) return known;
  // An unrecognised status is shown verbatim rather than mapped onto a guess.
  return { cls: "degraded", label: c.status || "unknown" };
}

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
  const [registry, setRegistry] = useState(null);
  const [loading, setLoading] = useState(true);
  // Keyed by container name so two toggles cannot share one spinner.
  const [toggling, setToggling] = useState({});
  const [toggleError, setToggleError] = useState({});
  const [toggleNote, setToggleNote] = useState({});

  const refresh = useCallback(async () => {
    try {
      const resp = await authFetch("/api/v1/containers");
      if (resp.ok) {
        const data = await resp.json();
        setContainers(data.containers || []);
        setTriton(data.triton || null);
        setUrlsNote(data.urlsNote || null);
        // Absent on an orchestrator build predating the registry — null then
        // means "this build has no registry", which the UI states rather than
        // showing an activation control that cannot work.
        setRegistry(data.registry || null);
      }
    } catch (_) {
      // keep existing state
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Flip a store-defined container's active flag.
   *
   * Deliberately does NOT optimistically flip the control: the write can fail,
   * and a switch that moves before the server agreed would be showing a state
   * that does not exist. The real state arrives from the refresh.
   */
  const setActive = useCallback(async (name, active) => {
    setToggling((t) => ({ ...t, [name]: true }));
    setToggleError((e) => ({ ...e, [name]: null }));
    setToggleNote((n) => ({ ...n, [name]: null }));
    try {
      const resp = await authFetch(`/api/v1/containers/${encodeURIComponent(name)}/active`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active }),
      });
      // Read as text first: an orchestrator predating this endpoint answers an
      // unmatched route with the plain-text body "Not Found", and calling
      // resp.json() on that throws a SyntaxError that reads like a bug in the
      // panel rather than a missing endpoint.
      const text = await resp.text();
      let data = null;
      try {
        data = JSON.parse(text);
      } catch (_) {
        data = null;
      }
      if (!resp.ok) {
        setToggleError((e) => ({
          ...e,
          [name]: (data && data.error) ||
            (resp.status === 404 && !data
              ? "This orchestrator build does not have the container activation endpoint yet."
              : `HTTP ${resp.status}`),
        }));
        return;
      }
      if (data && data.message) {
        setToggleNote((n) => ({ ...n, [name]: data.message }));
      }
      await refresh();
    } catch (err) {
      setToggleError((e) => ({ ...e, [name]: String(err && err.message ? err.message : err) }));
    } finally {
      setToggling((t) => ({ ...t, [name]: false }));
    }
  }, [refresh]);

  // Why a container's toggle is unavailable, or null when it is available. The
  // reason is shown rather than leaving a disabled control unexplained.
  const unavailableReason = useCallback((c) => {
    if (!c.configurable) {
      return "Built into the orchestrator — cannot be renamed or deactivated.";
    }
    if (registry && registry.tableConfigured === false) {
      return "No container registry table is configured on the orchestrator (CONTAINER_REGISTRY_TABLE is unset).";
    }
    if (registry && registry.tableReachable === false) {
      return `Registry table unreachable${registry.error ? `: ${registry.error}` : "."}`;
    }
    return null;
  }, [registry]);

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

        {/* Registry state. Only shown when there is something to say: the
            feature being off, being broken, or having rejected a record. A
            working registry needs no banner. */}
        {registry && (registry.tableConfigured === false || registry.tableReachable === false ||
          (registry.warnings || []).length > 0) && (
          <div
            style={{
              ...SUBTLE, margin: "0 0 12px", padding: "8px 10px",
              border: "1px solid var(--border)", borderRadius: "6px", lineHeight: 1.5,
            }}
            data-testid="container-registry-state"
          >
            {registry.tableConfigured === false && (
              <div>
                <strong>Container registry not configured.</strong> The orchestrator has no
                CONTAINER_REGISTRY_TABLE set, so only the built-in containers are listed and none
                can be activated from here.
              </div>
            )}
            {registry.tableConfigured !== false && registry.tableReachable === false && (
              <div>
                <strong>Container registry unreachable.</strong>{" "}
                {registry.tableName ? `${registry.tableName}: ` : ""}
                {registry.error || "the last read failed"}
                {registry.consecutiveErrors > 1 ? ` (${registry.consecutiveErrors} consecutive failures)` : ""}.
                Routing falls back to the built-in containers.
              </div>
            )}
            {(registry.warnings || []).map((w, i) => (
              <div key={i} style={{ marginTop: "4px" }}>{w}</div>
            ))}
          </div>
        )}

        {/* Single GPU control for the whole node group */}
        <GpuControl />

        {/* Topology note: the model containers run on CPU by default and call
            Triton over the network, so they stay reachable even when the GPU is
            stopped — "gpu offline" on a model row means Triton (GPU) is down, not
            the container. Only Triton needs the GPU; the optimizer runs on-demand. */}
        <div style={{ ...SUBTLE, margin: "0 0 12px", padding: "8px 10px", border: "1px solid var(--border)", borderRadius: "6px" }}>
          The three model containers run on CPU and delegate inference to Triton. A
          <strong> "gpu offline"</strong> badge means Triton (the GPU node) is stopped —
          the container itself is still reachable. Only Triton holds a GPU; the Model
          Optimizer runs as an on-demand Job.
        </div>

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
              const { cls: statusCls, label: statusLabel } = presentStatus(c);
              // API-supplied name first: a store-defined container's name is not
              // known when this bundle is built, so no build-time table can
              // resolve it. CONTAINER_LABELS remains the fallback for an
              // orchestrator build that does not yet send displayName.
              const label = c.displayName || CONTAINER_LABELS[c.name] || c.name;
              const isStoreDefined = c.source === "store";
              const reason = unavailableReason(c);
              const busy = !!toggling[c.name];
              const sharedFor = (registry?.sharedIntents) || {};

              return (
                <div key={c.name} className="container-item" style={{ flexDirection: "column", alignItems: "stretch" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "8px" }}>
                    <div>
                      <strong>{label}</strong>
                      <br />
                      <span style={SUBTLER}>{c.name}</span>
                      <br />
                      <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                        {MODELS[c.name] || (isStoreDefined ? "Custom container" : "")}
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

                  {/* Configured description — store-defined containers only, since
                      it is theirs to set. */}
                  {isStoreDefined && c.description && (
                    <div style={{ ...SUBTLE, marginTop: "6px", lineHeight: 1.5 }}>{c.description}</div>
                  )}

                  {/* Intents. A shared intent is marked because both claimants
                      are called and both sets of mutations merge, so merge order
                      decides which value survives downstream. */}
                  {Array.isArray(c.intents) && c.intents.length > 0 && (
                    <div style={{ marginTop: "6px", display: "flex", flexWrap: "wrap", gap: "4px" }}>
                      {c.intents.map((intent) => {
                        const claimants = sharedFor[intent] || [];
                        const shared = claimants.length > 1;
                        return (
                          <span
                            key={intent}
                            className={`intent-chip${shared ? " intent-chip--shared" : ""}`}
                            title={shared ? `Also claimed by: ${claimants.filter((n) => n !== c.name).join(", ")}` : undefined}
                          >
                            {intent}{shared ? " ⚠" : ""}
                          </span>
                        );
                      })}
                    </div>
                  )}

                  {/* Activation control. Placed above the probe evidence so the
                      container's reachability is visible in the same glance as
                      the switch. */}
                  {isStoreDefined && (
                    <div style={{ marginTop: "8px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                        <button
                          className={c.active ? "btn-secondary" : "btn-primary"}
                          style={{ padding: "5px 12px" }}
                          disabled={busy || !!reason}
                          onClick={() => setActive(c.name, !c.active)}
                          data-testid={`container-${c.name}-toggle-active`}
                        >
                          {busy
                            ? (c.active ? "Deactivating…" : "Activating…")
                            : (c.active ? "Deactivate" : "Activate")}
                        </button>
                        {reason && <span style={SUBTLE}>{reason}</span>}
                      </div>
                      {toggleNote[c.name] && (
                        <div style={{ ...SUBTLE, marginTop: "6px" }}>{toggleNote[c.name]}</div>
                      )}
                      {toggleError[c.name] && (
                        <div style={{ fontSize: "10px", color: "var(--red)", marginTop: "6px" }}>
                          Could not change activation: {toggleError[c.name]}
                        </div>
                      )}
                    </div>
                  )}

                  {/* An inactive container is not probed, so there is no probe
                      evidence to show. Saying so beats rendering three empty
                      rows that look like failed checks. */}
                  {c.containerStatus === "not_probed" ? (
                    <div style={{ ...SUBTLE, marginTop: "6px" }}>
                      {c.evidence?.note || "Not probed."}
                    </div>
                  ) : (
                    <>
                      {formatProbe("MCP /health/ready", c.evidence?.httpProbe)}
                      {formatProbe("gRPC channel_ready", c.evidence?.grpcProbe)}
                      {(c.evidence?.tritonModelProbes || []).map((p) => (
                        <div key={p.model}>{formatProbe(`Triton model probe (${p.model})`, p)}</div>
                      ))}
                    </>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
}
