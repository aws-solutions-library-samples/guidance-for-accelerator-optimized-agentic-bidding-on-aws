// useTheaterRun.js — one scenario, submitted once, turned into a beat sequence.
//
// Sequences the I/O around buildBeats; contains no beat logic itself.
//
// A failed submission yields NO beats. The surface reports the real failure and
// offers a retry rather than presenting a walkthrough, because a walkthrough of
// a request that was never processed would be fabricated output (BR-33).

import { useState, useCallback, useRef } from "react";
import { useOrchestratorClientWithBase } from "./useOrchestratorClientWithBase.js";
import { authFetch } from "../authFetch.js";
import { buildBeats, buildScenarioContext } from "../utils/theaterBeats.js";
import { isLiveAuctionResponse } from "../utils/bidResponseFixture.js";
import { loadScenarioPayload } from "../utils/scenarioPayload.js";

export const RUN_IDLE = "idle";
export const RUN_SUBMITTING = "submitting";
export const RUN_READY = "ready";
export const RUN_FAILED = "failed";

/**
 * Why no live auction was read.
 *
 * NOT_DEPLOYED is the orchestrator's 501: Prebid is optional, and its absence is a
 * deployment state rather than a fault. FAILED is everything else — an auth
 * rejection, an unreachable exchange, a timeout, a response that does not identify
 * itself as coming from one.
 *
 * These are separate because collapsing them is what hid a defect for the life of
 * this feature: the browser called the endpoint without a token, got 401, and the
 * single "not ok" branch reported that as "Prebid is not deployed". A fault that
 * renders as an expected absence is a fault nobody looks for.
 */
export const AUCTION_NOT_DEPLOYED = "not_deployed";
export const AUCTION_FAILED = "failed";

export function useTheaterRun({ baseUrl = "/api" } = {}) {
  const { submit } = useOrchestratorClientWithBase({ baseUrl });
  const [status, setStatus] = useState(RUN_IDLE);
  const [scenarioId, setScenarioId] = useState(null);
  const [context, setContext] = useState(null);
  const [beats, setBeats] = useState(null);
  const [error, setError] = useState(null);
  // The live auction, when Prebid is deployed. Stays null otherwise, and the
  // offers panel then falls back to the captured fixture and says so (FR-31).
  // Never set from anything but a real auction response.
  const [bidResponse, setBidResponse] = useState(null);
  // Why no live auction was read, when none was. `{ kind, detail }` or null.
  const [auctionFault, setAuctionFault] = useState(null);
  // Guards against a slow earlier submission overwriting a later one.
  const runTokenRef = useRef(0);

  const reset = useCallback(() => {
    runTokenRef.current += 1;
    setStatus(RUN_IDLE);
    setScenarioId(null);
    setContext(null);
    setBeats(null);
    setError(null);
    setBidResponse(null);
    setAuctionFault(null);
  }, []);

  // Runs the scenario's OpenRTB request as a real auction, if one is available.
  //
  // A failure here is not the run's failure — the mutations the beats describe did
  // happen — so it never sets RUN_FAILED. But it is recorded rather than swallowed:
  // the offers panel needs to distinguish "Prebid is not deployed" from "the
  // auction could not be read", because both fall back to the same fixture and only
  // one of them is expected.
  //
  // authFetch, not fetch: `/v1/auction/run` is not in the orchestrator's
  // _PUBLIC_PATHS, so an unauthenticated call is rejected before it reaches the
  // handler. Every other backend call in this app goes through authFetch.
  const runAuction = useCallback(async (payload, token) => {
    const bidRequest = payload?.bid_request;
    if (!bidRequest?.imp?.length) {
      setAuctionFault({
        kind: AUCTION_FAILED,
        detail: "the scenario carries no OpenRTB bid request to auction",
      });
      return;
    }

    try {
      const resp = await authFetch(`${baseUrl}/v1/auction/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bidRequest),
      });
      if (token !== runTokenRef.current) return;

      // 501 is the orchestrator saying Prebid is not deployed. Its own words are
      // used where it gives them, so the reason shown is the reason it reported.
      if (resp.status === 501) {
        const body = await resp.json().catch(() => null);
        setAuctionFault({
          kind: AUCTION_NOT_DEPLOYED,
          detail: body?.detail ?? "Prebid Server is not deployed.",
        });
        return;
      }

      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        setAuctionFault({
          kind: AUCTION_FAILED,
          detail: body?.detail ?? body?.error ?? `the auction endpoint returned HTTP ${resp.status}`,
          status: resp.status,
        });
        return;
      }

      const auction = await resp.json();
      if (token !== runTokenRef.current) return;

      // Only a response that actually came from an exchange is used.
      if (!isLiveAuctionResponse(auction)) {
        setAuctionFault({
          kind: AUCTION_FAILED,
          detail: "the response did not identify itself as coming from the exchange",
        });
        return;
      }

      setBidResponse(auction);
      setAuctionFault(null);
    } catch (err) {
      if (token !== runTokenRef.current) return;
      setAuctionFault({
        kind: AUCTION_FAILED,
        detail: err instanceof Error ? err.message : String(err),
      });
    }
  }, [baseUrl]);

  /**
   * Submit one scenario and turn the result into a beat sequence.
   *
   * `params` are the scenario card's tuner values. They are applied through the
   * SAME `loadScenarioPayload` the `▶ Send` path uses, so stepping through a
   * scenario in the Theater submits the identical bytes that sending it would —
   * a second copy of the patching logic here is how the two would drift, and the
   * drift would be invisible: the Theater would narrate a run nothing else made.
   */
  const start = useCallback(async (scenario, params = {}) => {
    const token = ++runTokenRef.current;
    setScenarioId(scenario?.id ?? null);
    setStatus(RUN_SUBMITTING);
    // beats and error are mutually exclusive by construction, so a failed run
    // has nowhere to put beats.
    setBeats(null);
    setError(null);
    setContext(null);
    setBidResponse(null);
    setAuctionFault(null);

    try {
      const payload = await loadScenarioPayload(scenario, params);
      if (token !== runTokenRef.current) return null;

      setContext(buildScenarioContext(payload));

      const result = await submit(payload, "REST");
      if (token !== runTokenRef.current) return null;

      // The orchestrator answered, but the response itself may report a
      // pipeline error. That is a real failure, not a walkthrough.
      if (result?.error) {
        setError(new Error(result.error.message || "The orchestrator reported an error"));
        setStatus(RUN_FAILED);
        return null;
      }

      setBeats(buildBeats(payload, result));
      setStatus(RUN_READY);

      // The auction runs AFTER the walkthrough is ready. An unavailable auction
      // leaves bidResponse null, and the offers panel then shows the captured
      // fixture with its own notice rather than presenting a fixture as live.
      void runAuction(payload, token);
      return result;
    } catch (err) {
      if (token !== runTokenRef.current) return null;
      setError(err instanceof Error ? err : new Error(String(err)));
      setStatus(RUN_FAILED);
      return null;
    }
  }, [submit, runAuction]);

  return { status, scenarioId, context, beats, error, bidResponse, auctionFault, start, reset };
}
