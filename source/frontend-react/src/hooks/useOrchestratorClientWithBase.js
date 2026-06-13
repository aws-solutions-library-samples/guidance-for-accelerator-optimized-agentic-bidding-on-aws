// useOrchestratorClientWithBase.js — Extended orchestrator client hook that
// accepts a baseUrl parameter for targeting different endpoints (e.g., RTB Fabric path).

import { useState, useRef, useCallback } from "react";
import { normalize } from "../utils/normalizer.js";
import { authFetch } from "../authFetch.js";

const MCP_PROTOCOL_VERSION = "2025-03-26";
const DEFAULT_TIMEOUT_MS = 10000;

/**
 * Hook that provides a `submit` function for sending payloads to an
 * ARTF orchestrator via REST or MCP transport at a configurable base URL.
 *
 * @param {object} options
 * @param {string} options.baseUrl - Base URL prefix (default: "/api")
 * @param {number} options.timeoutMs - Request timeout (default: 10000)
 * @returns {{ submit, loading, error, result, setResult, cancel }}
 */
export function useOrchestratorClientWithBase({ baseUrl = "/api", timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const mcpSessionRef = useRef(null);
  const rpcIdRef = useRef(2);
  const abortRef = useRef(null);

  const nextRpcId = () => {
    rpcIdRef.current += 1;
    return rpcIdRef.current;
  };

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const timedFetch = async (url, init, timeout = timeoutMs) => {
    const controller = new AbortController();
    abortRef.current = controller;
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const resp = await authFetch(url, { ...init, signal: controller.signal });
      clearTimeout(timer);
      return resp;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === "AbortError") {
        throw new Error(`Request timed out after ${timeout}ms`);
      }
      throw err;
    }
  };

  const submitRest = async (payload) => {
    const url = `${baseUrl}/v1/mutations`;
    const resp = await timedFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`REST failed: HTTP ${resp.status} from ${url}`);
    return resp.json();
  };

  const mcpInitialize = async () => {
    const url = `${baseUrl}/mcp`;
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

    const initResp = await timedFetch(url, {
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

    const notifyBody = { jsonrpc: "2.0", id: 2, method: "notifications/initialized", params: {} };
    const notifyResp = await timedFetch(url, {
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
    const url = `${baseUrl}/mcp`;
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

    const resp = await timedFetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Mcp-Session-Id": mcpSessionRef.current,
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) throw new Error(`MCP failed: HTTP ${resp.status} from ${url}`);

    const data = await resp.json();
    if (data?.error) throw new Error(data.error.message || "Unknown MCP error");

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
      normalized.networkPath = raw?.metadata?.network_path || (baseUrl.includes("fabric") ? "rtb-fabric" : "direct");
      setResult(normalized);
      setLoading(false);
      return normalized;
    } catch (err) {
      if (err.name === "AbortError") {
        // User-initiated cancellation — don't show error
        setLoading(false);
        return null;
      }
      const message = err?.message || String(err);
      setError({ message, transport, endpoint: baseUrl });
      setLoading(false);
      return null;
    }
  }, [baseUrl, timeoutMs]);

  return { submit, loading, error, result, setResult, cancel };
}
