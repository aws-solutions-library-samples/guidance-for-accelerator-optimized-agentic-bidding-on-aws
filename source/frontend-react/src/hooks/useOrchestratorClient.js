// useOrchestratorClient.js — React hook wrapping REST + MCP submission logic

import { useState, useRef, useCallback } from "react";
import { normalize } from "../utils/normalizer.js";
import { authFetch } from "../authFetch.js";

const MCP_PROTOCOL_VERSION = "2025-03-26";
const DEFAULT_TIMEOUT_MS = 5000;

/**
 * Hook that provides a `submit` function for sending payloads to the
 * ARTF orchestrator via REST or MCP transport.
 *
 * Returns { submit, loading, error, result }
 */
export function useOrchestratorClient() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const mcpSessionRef = useRef(null);
  const rpcIdRef = useRef(2);

  const nextRpcId = () => {
    rpcIdRef.current += 1;
    return rpcIdRef.current;
  };

  const timedFetch = async (url, init, timeoutMs = DEFAULT_TIMEOUT_MS) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await authFetch(url, { ...init, signal: controller.signal });
      clearTimeout(timer);
      return resp;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        throw new Error(`Request timed out after ${timeoutMs}ms`);
      }
      throw err;
    }
  };

  const submitRest = async (payload) => {
    const resp = await timedFetch("/api/v1/mutations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`REST failed: HTTP ${resp.status}`);
    return resp.json();
  };

  const mcpInitialize = async () => {
    const initBody = {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: "artf-flow-viz-react", version: "1.0" },
      },
    };

    const initResp = await timedFetch("/api/mcp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(initBody),
    });

    if (!initResp.ok) throw new Error(`MCP initialize failed: HTTP ${initResp.status}`);

    const sessionId =
      initResp.headers?.get?.("Mcp-Session-Id") ??
      initResp.headers?.get?.("mcp-session-id") ??
      null;

    if (!sessionId) throw new Error("MCP initialize: missing Mcp-Session-Id header");
    mcpSessionRef.current = sessionId;

    try { await initResp.json(); } catch (_) { /* drain */ }

    // Send notifications/initialized
    const notifyBody = { jsonrpc: "2.0", id: 2, method: "notifications/initialized", params: {} };
    const notifyResp = await timedFetch("/api/mcp", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Mcp-Session-Id": sessionId,
      },
      body: JSON.stringify(notifyBody),
    });
    if (!notifyResp.ok) {
      mcpSessionRef.current = null;
      throw new Error(`MCP notifications/initialized failed: HTTP ${notifyResp.status}`);
    }
    try { await notifyResp.json(); } catch (_) { /* drain */ }
  };

  const submitMcp = async (payload) => {
    if (!mcpSessionRef.current) {
      await mcpInitialize();
    }

    const rpcId = nextRpcId();
    const body = {
      jsonrpc: "2.0",
      method: "tools/call",
      id: rpcId,
      params: { name: "extend_rtb", arguments: payload },
    };

    const resp = await timedFetch("/api/mcp", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Mcp-Session-Id": mcpSessionRef.current,
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) throw new Error(`MCP failed: HTTP ${resp.status}`);

    const data = await resp.json();

    if (data?.error) {
      throw new Error(data.error.message || "Unknown MCP error");
    }

    // Unwrap result.content[0].text
    const content = data?.result?.content;
    if (Array.isArray(content)) {
      const textEntry = content.find((c) => c?.type === "text" && typeof c.text === "string");
      if (textEntry) return JSON.parse(textEntry.text);
    }
    if (data?.result && "mutations" in data.result) return data.result;
    throw new Error("MCP: could not unwrap response");
  };

  const submit = useCallback(async (payload, transport = "REST") => {
    setLoading(true);
    setError(null);
    try {
      const t0 = performance.now();
      const raw = transport === "MCP" ? await submitMcp(payload) : await submitRest(payload);
      const browserElapsedMs = Math.round(performance.now() - t0);
      const normalized = normalize(raw, transport, payload);
      normalized.browserElapsedMs = browserElapsedMs;
      setResult(normalized);
      setLoading(false);
      return normalized;
    } catch (err) {
      const message = err?.message || String(err);
      setError({ message, transport });
      setLoading(false);
      return null;
    }
  }, []);

  return { submit, loading, error, result, setResult };
}
