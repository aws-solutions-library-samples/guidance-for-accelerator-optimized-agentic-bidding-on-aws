/**
 * @vitest-environment jsdom
 *
 * The auction leg of useTheaterRun.
 *
 * The first test here is the one that was missing. `/v1/auction/run` is not in the
 * orchestrator's _PUBLIC_PATHS, so a call without a bearer token is rejected with
 * 401 before it reaches the handler — and because the old code funnelled every
 * non-ok status into one silent branch, that 401 rendered as "Prebid is not
 * deployed". The feature shipped, was verified server-side, and never once read a
 * live auction in a browser.
 *
 * So these assert the AUTHORIZATION HEADER, not just the resulting state: a test
 * that only checks "bidResponse got set" passes against a mock that never cared
 * whether the request was authenticated.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

vi.mock("../auth.js", () => ({
  isAuthConfigured: () => true,
  getAccessToken: async () => "test-access-token",
}));

// The mutations call is not under test; it has to succeed for the auction leg to
// run at all, so it is stubbed at the client boundary.
const submit = vi.fn(async () => ({ mutations: [], containers: [] }));
vi.mock("./useOrchestratorClientWithBase.js", () => ({
  useOrchestratorClientWithBase: () => ({ submit }),
}));

vi.mock("../utils/theaterBeats.js", () => ({
  buildBeats: () => [{ index: 0, kind: "recap", movement: "request-to-buy", contributors: [] }],
  buildScenarioContext: () => ({}),
}));

const { useTheaterRun, AUCTION_NOT_DEPLOYED, AUCTION_FAILED, RUN_READY } =
  await import("./useTheaterRun.js");

const SCENARIO = { id: "s1", file: "s1.json", name: "S1" };
const BID_REQUEST = { id: "req-1", imp: [{ id: "imp-1" }] };
const LIVE_RESPONSE = {
  id: "req-1",
  cur: "USD",
  seatbid: [{ seat: "artfhouse", bid: [{ id: "b1", impid: "imp-1", price: 4.2 }] }],
  artf_meta: { source: "prebid", endpoint: "https://prebid-server/openrtb2/auction", hop_ms: 12 },
};

let container;
let root;
let latest;

function Probe() {
  latest = useTheaterRun({ baseUrl: "/api" });
  return null;
}

/** Routes the scenario asset to a payload and /v1/auction/run to `auction`. */
function installFetch({ auctionStatus = 200, auctionBody = LIVE_RESPONSE }) {
  const calls = [];
  global.fetch = vi.fn(async (url, init) => {
    calls.push({ url: String(url), init });
    if (String(url).includes("/samples/")) {
      return { ok: true, status: 200, json: async () => ({ bid_request: BID_REQUEST }) };
    }
    return {
      ok: auctionStatus >= 200 && auctionStatus < 300,
      status: auctionStatus,
      json: async () => auctionBody,
    };
  });
  return calls;
}

const auctionCall = (calls) => calls.find((c) => c.url.includes("/v1/auction/run"));

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  latest = null;
  submit.mockClear();
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  delete global.fetch;
});

async function run() {
  await act(async () => root.render(<Probe />));
  await act(async () => {
    await latest.start(SCENARIO);
  });
  // The auction is fired without being awaited by start(), so let it settle.
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("useTheaterRun auction leg", () => {
  it("sends a bearer token on the auction request", async () => {
    const calls = installFetch({});
    await run();

    const call = auctionCall(calls);
    expect(call).toBeDefined();
    const headers = new Headers(call.init.headers);
    expect(headers.get("Authorization")).toBe("Bearer test-access-token");
  });

  it("reads the live auction when the exchange answers", async () => {
    installFetch({});
    await run();

    expect(latest.status).toBe(RUN_READY);
    expect(latest.bidResponse).toEqual(LIVE_RESPONSE);
    expect(latest.auctionFault).toBeNull();
  });

  it("records an auth rejection as a fault, never as an undeployed exchange", async () => {
    installFetch({
      auctionStatus: 401,
      auctionBody: { error: "Authentication required" },
    });
    await run();

    expect(latest.bidResponse).toBeNull();
    expect(latest.auctionFault.kind).toBe(AUCTION_FAILED);
    expect(latest.auctionFault.status).toBe(401);
    // The precise regression: this must NOT read as "Prebid is not deployed".
    expect(latest.auctionFault.kind).not.toBe(AUCTION_NOT_DEPLOYED);
  });

  it("records the orchestrator's 501 as an undeployed exchange, with its own reason", async () => {
    installFetch({
      auctionStatus: 501,
      auctionBody: {
        error: "prebid_not_deployed",
        detail: "Prebid Server is not deployed. Deploy it with deploy.sh --with-prebid.",
      },
    });
    await run();

    expect(latest.bidResponse).toBeNull();
    expect(latest.auctionFault.kind).toBe(AUCTION_NOT_DEPLOYED);
    expect(latest.auctionFault.detail).toMatch(/--with-prebid/);
  });

  it("rejects a response that does not identify itself as coming from the exchange", async () => {
    // No artf_meta.source: shaped like an auction but not attributable to one.
    installFetch({ auctionStatus: 200, auctionBody: { id: "x", seatbid: [] } });
    await run();

    expect(latest.bidResponse).toBeNull();
    expect(latest.auctionFault.kind).toBe(AUCTION_FAILED);
  });

  it("does not fail the run when the auction fails — the mutations still happened", async () => {
    installFetch({ auctionStatus: 502, auctionBody: { error: "prebid_unreachable" } });
    await run();

    expect(latest.status).toBe(RUN_READY);
    expect(latest.beats).toHaveLength(1);
    expect(latest.auctionFault.kind).toBe(AUCTION_FAILED);
  });
});
