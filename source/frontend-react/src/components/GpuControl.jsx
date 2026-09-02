import { useState, useEffect, useCallback } from "react";
import { authFetch } from "../authFetch.js";

/**
 * GpuControl — Single GPU node group control (start/stop the whole cluster).
 * Calls /api/v1/gpu/status, /api/v1/gpu/start, /api/v1/gpu/stop.
 */
export default function GpuControl() {
  const [status, setStatus] = useState("checking");
  const [desiredSize, setDesiredSize] = useState(0);
  const [tritonReady, setTritonReady] = useState(false);
  const [detail, setDetail] = useState(null);
  const [busy, setBusy] = useState(false);

  const checkStatus = useCallback(async () => {
    try {
      const resp = await authFetch("/api/v1/gpu/status");
      if (!resp.ok) {
        setStatus("error");
        return;
      }
      const data = await resp.json();
      if (data.error) {
        setStatus("error");
        return;
      }
      setDesiredSize(data.desiredSize || 0);
      setTritonReady(data.tritonReady || false);
      setDetail(data.tritonDetail || null);

      // The orchestrator classifies this now: it knows how long the node group
      // has been settled, so it can tell a cold start apart from a pod that is
      // never going to become ready. Deriving it here from tritonReady alone
      // rendered a 4-day deadlock as "starting (Triton loading...)".
      const BY_STATE = {
        ready: "running",
        stopped: "stopped",
        scaling_up: "scaling-up",
        scaling_down: "scaling-down",
        starting: "starting",
        blocked: "blocked",
      };
      if (data.tritonState && BY_STATE[data.tritonState]) {
        setStatus(BY_STATE[data.tritonState]);
      } else if (data.status === "UPDATING") {
        // Orchestrator predates tritonState (rolling deploy) — fall back.
        setStatus(data.desiredSize > 0 ? "scaling-up" : "scaling-down");
      } else if (data.desiredSize === 0) {
        setStatus("stopped");
      } else {
        setStatus(data.tritonReady ? "running" : "starting");
      }
    } catch (_) {
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, [checkStatus]);

  // Re-poll faster when in a transitional state
  useEffect(() => {
    if (status === "scaling-up" || status === "scaling-down" || status === "starting" || status === "blocked") {
      const fast = setInterval(checkStatus, 10000);
      return () => clearInterval(fast);
    }
  }, [status, checkStatus]);

  const handleStart = useCallback(async () => {
    setBusy(true);
    setStatus("scaling-up");
    try {
      const resp = await authFetch("/api/v1/gpu/start", { method: "POST" });
      const data = await resp.json();
      if (data.error) {
        setStatus("error");
        alert("Failed to start GPUs: " + data.error);
      }
    } catch (_) {
      setStatus("error");
    } finally {
      setBusy(false);
    }
  }, []);

  const handleStop = useCallback(async () => {
    if (!confirm("Stop GPU nodes? Triton inference will be unavailable until you start them again.")) return;
    setBusy(true);
    setStatus("scaling-down");
    try {
      const resp = await authFetch("/api/v1/gpu/stop", { method: "POST" });
      const data = await resp.json();
      if (data.error) {
        setStatus("error");
        alert("Failed to stop GPUs: " + data.error);
      }
    } catch (_) {
      setStatus("error");
    } finally {
      setBusy(false);
    }
  }, []);

  const badgeClass =
    status === "running" ? "gpu-status-running" :
    status === "stopped" ? "gpu-status-stopped" :
    status === "blocked" ? "gpu-status-blocked" :
    status === "scaling-up" || status === "scaling-down" || status === "starting" ? "gpu-status-scaling" :
    "gpu-status-error";

  const badgeText =
    status === "running" ? `running (${desiredSize} node${desiredSize > 1 ? "s" : ""})` :
    status === "stopped" ? "stopped" :
    status === "scaling-up" ? "scaling up..." :
    status === "scaling-down" ? "scaling down..." :
    status === "starting" ? "starting (Triton loading...)" :
    status === "blocked" ? "Triton not ready — needs attention" :
    status === "checking" ? "checking..." :
    "error";

  // Blocked leaves Start enabled: scaling the node group up is a legitimate
  // remedy when the GPU is simply oversubscribed.
  const startDisabled = busy || status === "running" || status === "scaling-up" || status === "starting";
  const stopDisabled = busy || status === "stopped" || status === "scaling-down";

  return (
    <div className="gpu-control">
      <div className="gpu-control-header">
        <h3>GPU Inference (g5 family · A10G)</h3>
        <span className={`gpu-status-badge ${badgeClass}`}>{badgeText}</span>
      </div>
      <p className="gpu-control-desc">
        The GPU node runs NVIDIA Triton (the shared inference server). The model
        containers run on CPU and call Triton over the network, and the Model
        Optimizer runs on-demand, so a single GPU is enough. Scale it to control
        costs; starting takes ~3-5 min.
      </p>
      {status === "blocked" && detail && (
        <p className="gpu-control-detail" data-testid="gpu-control-blocked-detail" role="status">
          {detail}
        </p>
      )}
      <div className="gpu-control-actions">
        <button className="btn-gpu-start" onClick={handleStart} disabled={startDisabled}>
          Start GPUs
        </button>
        <button className="btn-gpu-stop" onClick={handleStop} disabled={stopDisabled}>
          Stop GPUs
        </button>
      </div>
    </div>
  );
}
