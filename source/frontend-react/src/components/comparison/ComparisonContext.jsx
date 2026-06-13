// ComparisonContext.jsx — React context managing comparison mode state,
// dual orchestrator clients, and result synchronization.

import { createContext, useContext, useState, useCallback, useRef } from "react";
import { useOrchestratorClientWithBase } from "../../hooks/useOrchestratorClientWithBase";
import { useFabricConfig } from "../../hooks/useFabricConfig";

const ComparisonContext = createContext(null);

export function useComparison() {
  const ctx = useContext(ComparisonContext);
  if (!ctx) throw new Error("useComparison must be used within ComparisonProvider");
  return ctx;
}

export function ComparisonProvider({ children }) {
  const [mode, setModeInternal] = useState("standalone");
  const lastPayloadRef = useRef(null);

  const fabricConfig = useFabricConfig();

  // Two independent orchestrator client instances
  const standalone = useOrchestratorClientWithBase({ baseUrl: "/api", timeoutMs: 10000 });
  const fabric = useOrchestratorClientWithBase({
    baseUrl: fabricConfig.endpoint || "/fabric",
    timeoutMs: 10000,
  });

  const setMode = useCallback((newMode) => {
    // Cancel in-flight requests for deactivated views
    if (newMode === "standalone") {
      fabric.cancel();
    } else if (newMode === "fabric") {
      standalone.cancel();
    }

    setModeInternal(newMode);

    // If switching to side-by-side and we have a previous payload,
    // submit to the newly activated endpoint
    if (newMode === "side-by-side" && lastPayloadRef.current) {
      const { payload, transport } = lastPayloadRef.current;
      // Submit to whichever endpoint doesn't already have a result
      if (!fabric.result) {
        fabric.submit(payload, transport);
      }
      if (!standalone.result) {
        standalone.submit(payload, transport);
      }
    }
  }, [fabric, standalone]);

  const submitScenario = useCallback(async (payload, transport = "REST") => {
    lastPayloadRef.current = { payload, transport };

    if (mode === "standalone") {
      return standalone.submit(payload, transport);
    } else if (mode === "fabric") {
      return fabric.submit(payload, transport);
    } else {
      // side-by-side: dispatch to both simultaneously
      const [standaloneResult, fabricResult] = await Promise.allSettled([
        standalone.submit(payload, transport),
        fabric.submit(payload, transport),
      ]);
      return standaloneResult.value || fabricResult.value;
    }
  }, [mode, standalone, fabric]);

  const clearError = useCallback((panel) => {
    if (panel === "standalone") {
      standalone.setResult(null);
    } else if (panel === "fabric") {
      fabric.setResult(null);
    }
  }, [standalone, fabric]);

  const value = {
    mode,
    setMode,
    fabricConfig,
    standalone: {
      loading: standalone.loading,
      error: standalone.error,
      result: standalone.result,
    },
    fabric: {
      loading: fabric.loading,
      error: fabric.error,
      result: fabric.result,
    },
    submitScenario,
    clearError,
    lastPayload: lastPayloadRef.current,
  };

  return (
    <ComparisonContext.Provider value={value}>
      {children}
    </ComparisonContext.Provider>
  );
}
