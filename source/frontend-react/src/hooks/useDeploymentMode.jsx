import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { authFetch } from "../authFetch.js";

/**
 * Deployment mode reported by the orchestrator: "learner", "standard",
 * "unknown", or "checking" until the first answer arrives.
 *
 * The orchestrator reports `deploymentMode` on the authenticated
 * /api/v1/gpu/status endpoint, from the DEPLOYMENT_MODE variable that
 * `deploy.sh --learner-mode` renders into its Deployment. Only an exact
 * "learner" or "standard" is taken at face value. Anything else (a failed call,
 * a body that is not JSON, a missing or unrecognised field) is "unknown", with
 * the reason, so the UI never presents a guess as the deployed mode.
 *
 * The body is read whatever the HTTP status: the orchestrator includes the mode
 * in its 500 body too, because the mode comes from the pod environment and is
 * known even when the GPU node group lookup fails.
 */

const DeploymentModeContext = createContext(null);

const KNOWN = new Set(["learner", "standard"]);

export async function fetchDeploymentMode() {
  let resp;
  try {
    resp = await authFetch("/api/v1/gpu/status");
  } catch (err) {
    return { mode: "unknown", reason: (err && err.message) || String(err) };
  }
  let body;
  try {
    body = await resp.json();
  } catch (_) {
    return { mode: "unknown", reason: resp.ok ? "response was not JSON" : `HTTP ${resp.status}` };
  }
  const reported = body && typeof body === "object" ? body.deploymentMode : undefined;
  if (KNOWN.has(reported)) {
    return { mode: reported, reason: null };
  }
  if (typeof reported === "string") {
    return { mode: "unknown", reason: `orchestrator reported deploymentMode "${reported}"` };
  }
  if (!resp.ok) {
    return { mode: "unknown", reason: `HTTP ${resp.status}` };
  }
  return { mode: "unknown", reason: "deploymentMode missing from response" };
}

export function DeploymentModeProvider({ children }) {
  const [state, setState] = useState({ mode: "checking", reason: null });

  const load = useCallback(async () => {
    setState({ mode: "checking", reason: null });
    setState(await fetchDeploymentMode());
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <DeploymentModeContext.Provider value={{ ...state, retry: load }}>
      {children}
    </DeploymentModeContext.Provider>
  );
}

export function useDeploymentMode() {
  const ctx = useContext(DeploymentModeContext);
  if (!ctx) {
    // Outside the provider nothing has asked the orchestrator, so the mode is
    // not known.
    return { mode: "unknown", reason: "not inside DeploymentModeProvider", retry: () => {} };
  }
  return ctx;
}
