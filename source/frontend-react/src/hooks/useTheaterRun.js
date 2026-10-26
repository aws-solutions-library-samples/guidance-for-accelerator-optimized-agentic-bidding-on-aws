// useTheaterRun.js — one scenario, auctioned twice, turned into a beat sequence.
//
// Sequences the I/O around buildBeats; contains no beat logic itself.
//
// Pass 1 is `/v1/auction/run?artf=off`, pass 2 is `?artf=on`. The with-ARTF beats
// are built from that auction's own report of the mutations the Prebid hook ran
// (`artf_meta.extension_point`), so what the walkthrough narrates is what the
// auction did. No separate `/v1/mutations` call is made when Prebid is deployed.
//
// When Prebid is not deployed (501), there is no auction to read, and the beats
// come from `/v1/mutations` as before; the offers column shows the captured
// fixture with its own notice.
//
// A failed run yields NO beats. The surface reports the real failure and offers
// a retry rather than presenting a walkthrough, because a walkthrough of a
// request that was never processed would be fabricated output (BR-33). That
// includes a with-ARTF auction that failed for any reason other than Prebid not
// being deployed: substituting a separate call's mutations would narrate an
// auction that did not run.

import { useState, useCallback, useRef } from "react";
import { useOrchestratorClientWithBase } from "./useOrchestratorClientWithBase.js";
import { authFetch } from "../authFetch.js";
import { buildBeats, buildScenarioContext } from "../utils/theaterBeats.js";
import { normalizeExtensionPoint } from "../utils/normalizer.js";
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
  // The exact bytes submitted, and the orchestrator's normalized answer. Exposed
  // so the Theater's code view can render the SAME merged JSON the default
  // scenario view renders, through the same RawPanel, rather than a second
  // serializer that could disagree about what was sent.
  const [payload, setPayload] = useState(null);
  const [result, setResult] = useState(null);
  // Why no live auction was read, when none was. `{ kind, detail }` or null.
  const [auctionFault, setAuctionFault] = useState(null);
  // The BASELINE auction: the same request run through Prebid with the ARTF
  // extension point asked to propose nothing (`?artf=off`). This is pass 1 of the
  // walkthrough. It has no fixture to fall back to -- there is no captured
  // baseline -- so when it is null the pass-1 offers column states the fault, and
  // the comparison says the baseline was unavailable rather than inventing one.
  const [baselineResponse, setBaselineResponse] = useState(null);
  const [baselineFault, setBaselineFault] = useState(null);
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
    setBaselineResponse(null);
    setBaselineFault(null);
    setPayload(null);
    setResult(null);
  }, []);

  // Runs the scenario's OpenRTB request as a real auction, if one is available.
  //
  // Records the response or the fault against its pass, and also RETURNS the
  // outcome, `{ kind: "live", auction }` or the fault `{ kind, detail, status? }`,
  // so start() can decide from the with-ARTF auction how to build the beats. The
  // offers panel needs to distinguish "Prebid is not deployed" from "the auction
  // could not be read", because only one of them is expected.
  //
  // authFetch, not fetch: `/v1/auction/run` is not in the orchestrator's
  // _PUBLIC_PATHS, so an unauthenticated call is rejected before it reaches the
  // handler. Every other backend call in this app goes through authFetch.
  //
  // `artf` selects the pass. "on" is the auction the Theater has always run, with
  // the ARTF hook mutating the request inside Prebid; "off" asks the orchestrator
  // to mark the request so the hook's call proposes nothing. The two write to
  // separate state so one resolving cannot overwrite the other, and so a fault in
  // one is attributed to the pass that produced it.
  const runAuction = useCallback(async (payload, token, { artf = "on" } = {}) => {
    const setResponse = artf === "off" ? setBaselineResponse : setBidResponse;
    const setFault = artf === "off" ? setBaselineFault : setAuctionFault;

    const bidRequest = payload?.bid_request;
    if (!bidRequest?.imp?.length) {
      const fault = {
        kind: AUCTION_FAILED,
        detail: "the scenario carries no OpenRTB bid request to auction",
      };
      setFault(fault);
      return fault;
    }

    // The scenario's ARTF intents travel as a query parameter so the orchestrator
    // can state them on the request Prebid receives (top-level ext.artf). This is
    // a record, not a control: the hook asks for its configured intent set
    // regardless, so the with-ARTF auction is not narrowed by it.
    const intents = Array.isArray(payload?.applicable_intents)
      ? payload.applicable_intents.filter((i) => typeof i === "string" && i)
      : [];
    const params = new URLSearchParams({ artf });
    if (intents.length) params.set("intents", intents.join(","));

    try {
      const resp = await authFetch(`${baseUrl}/v1/auction/run?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bidRequest),
      });
      if (token !== runTokenRef.current) return null;
      // 501 is the orchestrator saying Prebid is not deployed. Its own words are
      // used where it gives them, so the reason shown is the reason it reported.
      if (resp.status === 501) {
        const body = await resp.json().catch(() => null);
        if (token !== runTokenRef.current) return null;
        const fault = {
          kind: AUCTION_NOT_DEPLOYED,
          detail: body?.detail ?? "Prebid Server is not deployed.",
        };
        setFault(fault);
        return fault;
      }
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        if (token !== runTokenRef.current) return null;
        const fault = {
          kind: AUCTION_FAILED,
          detail: body?.detail ?? body?.error ?? `the auction endpoint returned HTTP ${resp.status}`,
          status: resp.status,
        };
        setFault(fault);
        return fault;
      }
      const auction = await resp.json();
      if (token !== runTokenRef.current) return null;
      // Only a response that actually came from an exchange is used.
      if (!isLiveAuctionResponse(auction)) {
        const fault = {
          kind: AUCTION_FAILED,
          detail: "the response did not identify itself as coming from the exchange",
        };
        setFault(fault);
        return fault;
      }
      setResponse(auction);
      setFault(null);
      return { kind: "live", auction };
    } catch (err) {
      if (token !== runTokenRef.current) return null;
      const fault = {
        kind: AUCTION_FAILED,
        detail: err instanceof Error ? err.message : String(err),
      };
      setFault(fault);
      return fault;
    }
  }, [baseUrl]);

  /**
   * Run one scenario and turn the with-ARTF auction into a beat sequence.
   *
   * `params` are the scenario card's tuner values. They are applied through the
   * SAME `loadScenarioPayload` the `▶ Send` path uses, so the Theater and the Send
   * path start from identical bytes; a second copy of the patching logic here is
   * how the two would drift.
   *
   * Both auctions are fired together: they are independent, and the baseline pass
   * hits no container. The beats wait only for the with-ARTF auction, because its
   * report is what they are built from. The baseline settles on its own; when it
   * fails, the pass-1 column states the fault (it has no fixture).
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
    setBaselineResponse(null);
    setBaselineFault(null);
    setPayload(null);
    setResult(null);
    const fail = (err) => {
      setError(err instanceof Error ? err : new Error(String(err)));
      setStatus(RUN_FAILED);
      return null;
    };
    try {
      const submitted = await loadScenarioPayload(scenario, params);
      if (token !== runTokenRef.current) return null;
      setContext(buildScenarioContext(submitted));
      setPayload(submitted);

      void runAuction(submitted, token, { artf: "off" });
      const withArtf = await runAuction(submitted, token, { artf: "on" });
      if (token !== runTokenRef.current || withArtf == null) return null;

      let answered;
      if (withArtf.kind === "live") {
        // The auction's own report. An orchestrator that predates the report, or
        // a response without a usable one, yields a result with no container
        // mutations and an artfReport saying so; the recap states it in words.
        const ep = withArtf.auction?.artf_meta?.extension_point ?? {
          available: false,
          reason: "the orchestrator did not report the ARTF hook's mutations",
        };
        answered = normalizeExtensionPoint(ep, submitted);
      } else if (withArtf.kind === AUCTION_NOT_DEPLOYED) {
        // No exchange to read. The containers are asked directly, as before
        // Prebid existed, and the offers column shows the fixture with its notice.
        answered = await submit(submitted, "REST");
        if (token !== runTokenRef.current) return null;
        // The orchestrator answered, but the response itself may report a
        // pipeline error. That is a real failure, not a walkthrough.
        if (answered?.error) {
          return fail(new Error(answered.error.message || "The orchestrator reported an error"));
        }
      } else {
        // The with-ARTF auction failed. Its fault is the run's fault: narrating a
        // separate call's mutations would describe an auction that did not run.
        return fail(new Error(`The with-ARTF auction could not be read: ${withArtf.detail}`));
      }

      setResult(answered);
      setBeats(buildBeats(submitted, answered));
      setStatus(RUN_READY);
      return answered;
    } catch (err) {
      if (token !== runTokenRef.current) return null;
      return fail(err);
    }
  }, [submit, runAuction]);

  return {
    status, scenarioId, context, beats, error, bidResponse, auctionFault,
    baselineResponse, baselineFault,
    payload, result, start, reset,
  };
}
