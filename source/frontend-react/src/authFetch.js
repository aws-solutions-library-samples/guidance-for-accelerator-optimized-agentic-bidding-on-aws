/**
 * Authenticated fetch wrapper and UI API transport switch.
 *
 * Every backend call in the UI goes through here. Two transports:
 *
 * 1. Lambda proxy (deployed stacks). When `VITE_UI_API_PROXY_ARN` is set and the
 *    URL is a relative `/api/...` path, the request is packaged as a JSON event,
 *    signed with the user's Identity Pool SigV4 credentials, and sent with
 *    `lambda:InvokeFunction` to the `<prefix>-ui-api-proxy` function. That function
 *    sits inside the cluster VPC and forwards to the orchestrator's ClusterIP
 *    Service, so the orchestrator has no public address. The reply is turned back
 *    into a `Response`, so callers keep using `resp.ok` / `resp.status` /
 *    `resp.json()` unchanged.
 *
 * 2. Plain fetch (local dev, `vite dev` against a port-forward, absolute URLs such
 *    as an RTB Fabric endpoint). Same as before: the Cognito access token is added
 *    as a Bearer header when auth is configured.
 *
 * In both cases the user's Cognito access token travels as `Authorization: Bearer`,
 * so the orchestrator's own JWT and scope checks are what authorise the call; the
 * proxy only moves bytes.
 */

import { LambdaClient, InvokeCommand } from "@aws-sdk/client-lambda";
import { isAuthConfigured, getAccessToken } from "./auth";
import { getSigV4Credentials } from "./awsCredentials";

export const UI_API_PROXY_ARN = import.meta.env.VITE_UI_API_PROXY_ARN || "";

/** Error raised when the proxy itself (not the orchestrator) cannot be invoked. */
export class UiApiProxyError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "UiApiProxyError";
    this.kind = kind; // "not_configured" | "not_authenticated" | "access_denied" | "not_found" | "unreachable" | "unknown"
  }
}

/** True when this build routes `/api` calls through the proxy Lambda. */
export function isProxyTransport() {
  return !!UI_API_PROXY_ARN;
}

function _regionFromArn(arn) {
  // arn:aws:lambda:REGION:ACCOUNT:function:NAME
  return arn.split(":")[3] || "us-east-1";
}

/** @type {{token: string, client: LambdaClient} | null} */
let _cached = null;

/** Exposed for tests; a token change is otherwise the only thing that resets it. */
export function _resetProxyClientCache() {
  _cached = null;
}

async function _getLambdaClient() {
  const { token, credentials, error } = await getSigV4Credentials();
  if (error) {
    throw new UiApiProxyError(error.kind, error.detail);
  }
  if (_cached && _cached.token === token) {
    return _cached.client;
  }
  const client = new LambdaClient({ region: _regionFromArn(UI_API_PROXY_ARN), credentials });
  _cached = { token, client };
  return client;
}

function _mapInvokeError(err) {
  if (err instanceof UiApiProxyError) return err;
  const name = err?.name || "";
  const msg = String(err?.message || err);
  if (/AccessDenied|not authorized|Forbidden/i.test(name + msg)) {
    return new UiApiProxyError(
      "access_denied",
      `Access denied invoking the UI API proxy. The Identity Pool authenticated role needs lambda:InvokeFunction on ${UI_API_PROXY_ARN}.`
    );
  }
  if (/ResourceNotFound|NotFound/i.test(name + msg)) {
    return new UiApiProxyError("not_found", "UI API proxy not deployed (function not found). Re-run deploy.sh.");
  }
  if (/NetworkError|Failed to fetch|CORS|TypeError|AbortError/i.test(name + msg)) {
    return new UiApiProxyError("unreachable", "Could not reach the Lambda API from the browser (network issue).");
  }
  return new UiApiProxyError("unknown", msg);
}

/**
 * Split a relative URL into `{ path, query }` for the proxy event.
 * The `/api` prefix is kept: the orchestrator serves every UI route under it.
 */
export function _splitUrl(url) {
  const qIdx = url.indexOf("?");
  const path = qIdx === -1 ? url : url.slice(0, qIdx);
  const query = {};
  if (qIdx !== -1) {
    for (const [k, v] of new URLSearchParams(url.slice(qIdx + 1))) query[k] = v;
  }
  return { path, query };
}

function _bodyToString(body) {
  if (body == null) return null;
  if (typeof body === "string") return body;
  if (body instanceof URLSearchParams) return body.toString();
  // Call sites JSON.stringify their own payloads; anything else is a bug.
  throw new UiApiProxyError("unknown", "authFetch: only string bodies can be sent through the UI API proxy.");
}

/**
 * Build the FR-2 event from fetch-style arguments.
 * @param {string} url relative `/api/...` URL
 * @param {RequestInit} init
 * @param {string} token Cognito access token for the Authorization header
 */
export function _buildProxyEvent(url, init, token) {
  const { path, query } = _splitUrl(url);
  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${token}`);
  const picked = {};
  for (const name of ["Authorization", "Content-Type", "Accept", "Mcp-Session-Id"]) {
    const v = headers.get(name);
    if (v != null) picked[name] = v;
  }
  return {
    method: (init.method || "GET").toUpperCase(),
    path,
    query,
    headers: picked,
    body: _bodyToString(init.body),
  };
}

/** Turn the FR-2 reply into a Response the call sites already know how to read. */
export function _replyToResponse(reply) {
  const status = Number(reply?.status) || 502;
  const headers = new Headers();
  for (const [k, v] of Object.entries(reply?.headers || {})) {
    if (v != null) headers.set(k, String(v));
  }
  let body = reply?.body ?? "";
  if (reply?.isBase64 && typeof body === "string") {
    const bin = atob(body);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    body = bytes;
  }
  // 204/205/304 must not carry a body.
  if (status === 204 || status === 205 || status === 304) body = null;
  return new Response(body, { status, headers });
}

async function _invokeProxy(url, init, token) {
  const event = _buildProxyEvent(url, init, token);
  let client;
  try {
    client = await _getLambdaClient();
  } catch (e) {
    throw _mapInvokeError(e);
  }
  let out;
  try {
    out = await client.send(
      new InvokeCommand({
        FunctionName: UI_API_PROXY_ARN,
        InvocationType: "RequestResponse",
        Payload: new TextEncoder().encode(JSON.stringify(event)),
      }),
      init.signal ? { abortSignal: init.signal } : undefined
    );
  } catch (e) {
    throw _mapInvokeError(e);
  }
  const text = out.Payload ? new TextDecoder().decode(out.Payload) : "";
  if (out.FunctionError) {
    // The handler itself crashed (it is written never to); surface it as a 502 so
    // the UI's existing error path shows it instead of a thrown exception.
    return new Response(JSON.stringify({ error: "ui_api_proxy_failed", detail: text }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  let reply;
  try {
    reply = JSON.parse(text);
  } catch (_) {
    throw new UiApiProxyError("unknown", "UI API proxy returned a non-JSON payload.");
  }
  return _replyToResponse(reply);
}

/** Only relative `/api` URLs are proxied; anything else is a plain fetch target. */
export function _isProxiedUrl(url) {
  return typeof url === "string" && (url === "/api" || url.startsWith("/api/") || url.startsWith("/api?"));
}

/**
 * Drop-in replacement for fetch() that adds the Authorization header and, in
 * deployed builds, routes `/api` calls through the UI API proxy Lambda.
 * Use this for all backend API calls.
 */
export async function authFetch(url, init = {}) {
  if (!isAuthConfigured()) {
    return fetch(url, init);
  }

  const token = await getAccessToken();
  if (!token) {
    // Session expired — force reload to show login
    window.location.reload();
    throw new Error("Session expired");
  }

  if (isProxyTransport() && _isProxiedUrl(url)) {
    return _invokeProxy(url, init, token);
  }

  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${token}`);

  return fetch(url, { ...init, headers });
}
