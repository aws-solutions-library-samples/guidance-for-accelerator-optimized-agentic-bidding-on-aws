// useTheaterRun.js — one scenario, submitted once, turned into a beat sequence.
//
// Sequences the I/O around buildBeats; contains no beat logic itself.
//
// A failed submission yields NO beats. The surface reports the real failure and
// offers a retry rather than presenting a walkthrough, because a walkthrough of
// a request that was never processed would be fabricated output (BR-33).

import { useState, useCallback, useRef } from "react";
import { useOrchestratorClientWithBase } from "./useOrchestratorClientWithBase.js";
import { buildBeats, buildScenarioContext } from "../utils/theaterBeats.js";
import { isLiveAuctionResponse } from "../utils/bidResponseFixture.js";

export const RUN_IDLE = "idle";
export const RUN_SUBMITTING = "submitting";
export const RUN_READY = "ready";
export const RUN_FAILED = "failed";

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
  // Guards against a slow earlier submission overwriting a later one.
  const runTokenRef = useRef(0);

  const reset = useCallback(() => {
    runTokenRef.current += 1;
    setStatus(RUN_IDLE);
    setScenarioId(null);
    setContext(null);
    setBeats(null);
    setError(null);
  }, []);

  // Runs the scenario's OpenRTB request as a real auction, if one is available.
  //
  // Deliberately quiet on failure. Prebid is optional: when it is not deployed the
  // endpoint answers 501, which is not an error in this run's terms — the
  // orchestrator still applied the mutations the walkthrough describes. So a
  // failure here leaves bidResponse null and the offers panel falls back to the
  // captured fixture, which carries its own notice. Nothing synthesises an
  // auction, and no fixture is ever labelled live.
  const runAuction = useCallback(async (payload, token) => {
    const bidRequest = payload?.bid_request;
    if (!bidRequest?.imp?.length) return;

    try {
      const resp = await fetch(`${baseUrl}/v1/auction/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bidRequest),
      });
      if (!resp.ok) return;             // 501 when Prebid is absent; nothing to show
      const auction = await resp.json();
      if (token !== runTokenRef.current) return;
      // Only a response that actually came from an exchange is used.
      if (!isLiveAuctionResponse(auction)) return;
      setBidResponse(auction);
    } catch {
      // Network failure: same reasoning. The fixture notice is the honest fallback.
    }
  }, [baseUrl]);

  const start = useCallback(async (scenario) => {
    const token = ++runTokenRef.current;
    setScenarioId(scenario?.id ?? null);
    setStatus(RUN_SUBMITTING);
    // beats and error are mutually exclusive by construction, so a failed run
    // has nowhere to put beats.
    setBeats(null);
    setError(null);
    setContext(null);
    setBidResponse(null);

    try {
      const resp = await fetch(`/samples/${scenario.file}?t=${Date.now()}`);
      if (!resp.ok) throw new Error(`Could not load scenario ${scenario.file} (${resp.status})`);
      const payload = await resp.json();
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

      // The auction runs AFTER the walkthrough is ready, and its failure is not
      // the run's failure: the mutations the beats describe did happen. An
      // unavailable auction leaves bidResponse null, and the offers panel then
      // shows the captured fixture with its own notice rather than presenting a
      // fixture as live.
      void runAuction(payload, token);
      return result;
    } catch (err) {
      if (token !== runTokenRef.current) return null;
      setError(err instanceof Error ? err : new Error(String(err)));
      setStatus(RUN_FAILED);
      return null;
    }
  }, [submit, runAuction]);

  return { status, scenarioId, context, beats, error, bidResponse, start, reset };
}
