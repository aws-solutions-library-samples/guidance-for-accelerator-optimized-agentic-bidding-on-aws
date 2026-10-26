/**
 * @vitest-environment jsdom
 *
 * authFetch transport switch: the FR-2 event built from fetch-style arguments,
 * the Response rebuilt from the FR-2 reply, the routing rule (only relative
 * `/api` URLs are proxied) and the proxy error mapping. No test calls AWS; the
 * Lambda client is mocked at the module boundary.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// --- module mocks (hoisted) --------------------------------------------------

const sendMock = vi.fn();
// vitest 4 invokes mock implementations with `new`, so these must be `function`s.
vi.mock("@aws-sdk/client-lambda", () => ({
  LambdaClient: vi.fn(function () { return { send: sendMock }; }),
  InvokeCommand: vi.fn(function (input) { return { input }; }),
}));

const authState = { configured: true, accessToken: "access-tok" };
vi.mock("./auth", () => ({
  isAuthConfigured: () => authState.configured,
  getAccessToken: async () => authState.accessToken,
  getIdToken: async () => "id-tok",
}));

const credsState = { result: { token: "id-tok", credentials: () => Promise.resolve({}) } };
vi.mock("./awsCredentials", () => ({
  getSigV4Credentials: async () => credsState.result,
}));

const PROXY_ARN = "arn:aws:lambda:eu-west-1:123456789012:function:dv1-ui-api-proxy";

async function loadModule(arn) {
  vi.stubEnv("VITE_UI_API_PROXY_ARN", arn);
  vi.resetModules();
  return import("./authFetch.js");
}

function lambdaReply(obj, { functionError } = {}) {
  return {
    Payload: new TextEncoder().encode(JSON.stringify(obj)),
    FunctionError: functionError,
  };
}

let fetchMock;
beforeEach(() => {
  sendMock.mockReset();
  fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  authState.configured = true;
  authState.accessToken = "access-tok";
  credsState.result = { token: "id-tok", credentials: () => Promise.resolve({}) };
});
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

// --- pure helpers -----------------------------------------------------------

describe("_buildProxyEvent", () => {
  it("keeps the /api prefix, splits the query, and forwards only the allowlisted headers", async () => {
    const m = await loadModule(PROXY_ARN);
    const event = m._buildProxyEvent(
      "/api/v1/governance/sweep-status?model_type=dlrm%20bid&n=5",
      {
        method: "post",
        headers: { "Content-Type": "application/json", "X-Custom": "nope", "Mcp-Session-Id": "s-1" },
        body: '{"a":1}',
      },
      "tok"
    );
    expect(event).toEqual({
      method: "POST",
      path: "/api/v1/governance/sweep-status",
      query: { model_type: "dlrm bid", n: "5" },
      headers: { Authorization: "Bearer tok", "Content-Type": "application/json", "Mcp-Session-Id": "s-1" },
      body: '{"a":1}',
    });
  });

  it("defaults to GET with a null body and an empty query", async () => {
    const m = await loadModule(PROXY_ARN);
    const event = m._buildProxyEvent("/api/v1/containers", {}, "tok");
    expect(event.method).toBe("GET");
    expect(event.query).toEqual({});
    expect(event.body).toBeNull();
    expect(event.headers).toEqual({ Authorization: "Bearer tok" });
  });

  it("refuses a non-string body rather than silently sending [object Object]", async () => {
    const m = await loadModule(PROXY_ARN);
    expect(() => m._buildProxyEvent("/api/v1/x", { method: "POST", body: { a: 1 } }, "tok")).toThrow(/string bodies/);
  });
});

describe("_replyToResponse", () => {
  it("rebuilds status, headers and a JSON body callers can .json()", async () => {
    const m = await loadModule(PROXY_ARN);
    const resp = m._replyToResponse({
      status: 409,
      headers: { "content-type": "application/json", "mcp-session-id": "abc" },
      body: '{"error":"conflict"}',
      isBase64: false,
    });
    expect(resp.ok).toBe(false);
    expect(resp.status).toBe(409);
    expect(resp.headers.get("Mcp-Session-Id")).toBe("abc");
    expect(await resp.json()).toEqual({ error: "conflict" });
  });

  it("decodes base64 bodies", async () => {
    const m = await loadModule(PROXY_ARN);
    const resp = m._replyToResponse({ status: 200, headers: {}, body: "//4A", isBase64: true });
    expect(new Uint8Array(await resp.arrayBuffer())).toEqual(new Uint8Array([0xff, 0xfe, 0x00]));
  });

  it("drops the body on 204", async () => {
    const m = await loadModule(PROXY_ARN);
    const resp = m._replyToResponse({ status: 204, headers: {}, body: "" });
    expect(resp.status).toBe(204);
  });
});

describe("_isProxiedUrl", () => {
  it("proxies only relative /api URLs", async () => {
    const m = await loadModule(PROXY_ARN);
    expect(m._isProxiedUrl("/api/v1/containers")).toBe(true);
    expect(m._isProxiedUrl("/api/mcp")).toBe(true);
    expect(m._isProxiedUrl("/fabric/v1/mutations")).toBe(false);
    expect(m._isProxiedUrl("https://fabric.example.com/v1/mutations")).toBe(false);
    expect(m._isProxiedUrl("/apiary")).toBe(false);
  });
});

// --- authFetch routing --------------------------------------------------------

describe("authFetch with the proxy configured", () => {
  it("invokes the Lambda with a signed event and returns the orchestrator's reply", async () => {
    const m = await loadModule(PROXY_ARN);
    sendMock.mockResolvedValueOnce(
      lambdaReply({ status: 200, headers: { "content-type": "application/json" }, body: '{"containers":[]}' })
    );
    const resp = await m.authFetch("/api/v1/containers");
    expect(fetchMock).not.toHaveBeenCalled();
    expect(sendMock).toHaveBeenCalledTimes(1);
    const [command] = sendMock.mock.calls[0];
    expect(command.input.FunctionName).toBe(PROXY_ARN);
    expect(command.input.InvocationType).toBe("RequestResponse");
    const sent = JSON.parse(new TextDecoder().decode(command.input.Payload));
    expect(sent.path).toBe("/api/v1/containers");
    expect(sent.headers.Authorization).toBe("Bearer access-tok");
    expect(resp.ok).toBe(true);
    expect(await resp.json()).toEqual({ containers: [] });
  });

  it("passes the caller's AbortSignal to the SDK", async () => {
    const m = await loadModule(PROXY_ARN);
    sendMock.mockResolvedValueOnce(lambdaReply({ status: 200, headers: {}, body: "{}" }));
    const controller = new AbortController();
    await m.authFetch("/api/v1/containers", { signal: controller.signal });
    expect(sendMock.mock.calls[0][1]).toEqual({ abortSignal: controller.signal });
  });

  it("passes upstream 4xx/5xx through as a non-ok Response, not an exception", async () => {
    const m = await loadModule(PROXY_ARN);
    sendMock.mockResolvedValueOnce(
      lambdaReply({ status: 502, headers: { "content-type": "application/json" }, body: '{"error":"orchestrator_unreachable"}' })
    );
    const resp = await m.authFetch("/api/v1/containers");
    expect(resp.ok).toBe(false);
    expect(resp.status).toBe(502);
    expect((await resp.json()).error).toBe("orchestrator_unreachable");
  });

  it("surfaces a handler crash as a 502 Response", async () => {
    const m = await loadModule(PROXY_ARN);
    sendMock.mockResolvedValueOnce(lambdaReply({ errorMessage: "boom" }, { functionError: "Unhandled" }));
    const resp = await m.authFetch("/api/v1/containers");
    expect(resp.status).toBe(502);
    expect((await resp.json()).error).toBe("ui_api_proxy_failed");
  });

  it("maps AccessDeniedException to an access_denied error naming the function", async () => {
    const m = await loadModule(PROXY_ARN);
    const err = new Error("User is not authorized to perform: lambda:InvokeFunction");
    err.name = "AccessDeniedException";
    sendMock.mockRejectedValueOnce(err);
    await expect(m.authFetch("/api/v1/containers")).rejects.toMatchObject({
      name: "UiApiProxyError",
      kind: "access_denied",
      message: expect.stringContaining(PROXY_ARN),
    });
  });

  it("maps ResourceNotFoundException to not_found", async () => {
    const m = await loadModule(PROXY_ARN);
    const err = new Error("Function not found");
    err.name = "ResourceNotFoundException";
    sendMock.mockRejectedValueOnce(err);
    await expect(m.authFetch("/api/v1/containers")).rejects.toMatchObject({ kind: "not_found" });
  });

  it("reports a missing Identity Pool as not_configured before invoking", async () => {
    const m = await loadModule(PROXY_ARN);
    credsState.result = { error: { kind: "not_configured", detail: "no pool" } };
    await expect(m.authFetch("/api/v1/containers")).rejects.toMatchObject({ kind: "not_configured" });
    expect(sendMock).not.toHaveBeenCalled();
  });

  it("still uses plain fetch for non-/api URLs", async () => {
    const m = await loadModule(PROXY_ARN);
    await m.authFetch("https://fabric.example.com/v1/mutations", { method: "POST" });
    expect(sendMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers.get("Authorization")).toBe("Bearer access-tok");
  });

  it("builds the Lambda client for the function's region and reuses it per token", async () => {
    const { LambdaClient } = await import("@aws-sdk/client-lambda");
    const m = await loadModule(PROXY_ARN);
    LambdaClient.mockClear();
    sendMock.mockResolvedValue(lambdaReply({ status: 200, headers: {}, body: "{}" }));
    await m.authFetch("/api/v1/containers");
    await m.authFetch("/api/v1/gpu/status");
    expect(LambdaClient).toHaveBeenCalledTimes(1);
    expect(LambdaClient.mock.calls[0][0].region).toBe("eu-west-1");
  });
});

describe("authFetch without the proxy configured", () => {
  it("falls back to plain fetch with the Bearer header", async () => {
    const m = await loadModule("");
    expect(m.isProxyTransport()).toBe(false);
    await m.authFetch("/api/v1/containers");
    expect(sendMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/containers");
    expect(init.headers.get("Authorization")).toBe("Bearer access-tok");
  });

  it("passes through untouched when auth is not configured at all", async () => {
    const m = await loadModule("");
    authState.configured = false;
    await m.authFetch("/api/v1/containers", { method: "POST" });
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/containers", { method: "POST" });
  });
});
