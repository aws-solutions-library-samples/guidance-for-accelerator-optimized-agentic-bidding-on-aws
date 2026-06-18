// useFabricConfig.js — Resolves the RTB Fabric endpoint URL from environment.
// Returns { endpoint, valid, source }.

import { useState } from "react";

/**
 * Resolution order:
 * 1. Vite define: __ARTF_FABRIC_URL__ (set in vite.config.js)
 * 2. Window global: window.__ARTF_FABRIC_URL__
 * 3. Default: /fabric (relative path, works when CloudFront routes /fabric/*)
 */
function isValidUrl(value) {
  if (typeof value !== "string" || value.trim() === "") return false;
  return value.startsWith("/") || value.startsWith("http://") || value.startsWith("https://");
}

export function useFabricConfig() {
  const [config] = useState(() => {
    // Check Vite define (injected at build time)
    try {
      // eslint-disable-next-line no-undef
      const defineUrl = __ARTF_FABRIC_URL__;
      if (defineUrl && isValidUrl(defineUrl)) {
        return { endpoint: defineUrl, valid: true, source: "vite-define" };
      }
    } catch (_) { /* not defined */ }

    // Check Vite env
    const envUrl = import.meta.env?.VITE_RTB_FABRIC_URL;
    if (envUrl && isValidUrl(envUrl)) {
      return { endpoint: envUrl, valid: true, source: "env" };
    }

    // Check window global
    if (typeof window !== "undefined" && window.__ARTF_FABRIC_URL__) {
      const globalUrl = window.__ARTF_FABRIC_URL__;
      if (isValidUrl(globalUrl)) {
        return { endpoint: globalUrl, valid: true, source: "global" };
      }
    }

    // Default to /fabric relative path (CloudFront routes this)
    return { endpoint: "/fabric", valid: true, source: "default" };
  });

  return config;
}
