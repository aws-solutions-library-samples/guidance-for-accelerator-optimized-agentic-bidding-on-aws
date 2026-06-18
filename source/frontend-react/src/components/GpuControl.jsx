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

      if (data.status === "UPDATING") {
        setStatus(data.desiredSize > 0 ? "scaling-up" : "scaling-down");
      } else if (data.desiredSize === 0) {
        setStatus("stopped");
      } else if (data.tritonReady) {
        setStatus("running");
      } else {
        setStatus("starting");
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
    if (status === "scaling-up" || status === "scaling-down" || status === "starting") {
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
    status === "scaling-up" || status === "scaling-down" || status === "starting" ? "gpu-status-scaling" :
    "gpu-status-error";

  const badgeText =
    status === "running" ? `running (${desiredSize} node${desiredSize > 1 ? "s" : ""})` :
    status === "stopped" ? "stopped" :
    status === "scaling-up" ? "scaling up..." :
    status === "scaling-down" ? "scaling down..." :
    status === "starting" ? "starting (Triton loading...)" :
    status === "checking" ? "checking..." :
    "error";

  const startDisabled = busy || status === "running" || status === "scaling-up" || status === "starting";
  const stopDisabled = busy || status === "stopped" || status === "scaling-down";

  return (
    <div className="gpu-control">
      <div className="gpu-control-header">
        <h3>GPU Inference (g5.xlarge · A10G)</h3>
        <span className={`gpu-status-badge ${badgeClass}`}>{badgeText}</span>
      </div>
      <p className="gpu-control-desc">
        Scale the GPU node group to control costs. Starting takes ~3-5 min.
      </p>
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
