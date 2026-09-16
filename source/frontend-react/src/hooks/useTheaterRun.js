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

  const start = useCallback(async (scenario) => {
    const token = ++runTokenRef.current;
    setScenarioId(scenario?.id ?? null);
    setStatus(RUN_SUBMITTING);
    // beats and error are mutually exclusive by construction, so a failed run
    // has nowhere to put beats.
    setBeats(null);
    setError(null);
    setContext(null);

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
      return result;
    } catch (err) {
      if (token !== runTokenRef.current) return null;
      setError(err instanceof Error ? err : new Error(String(err)));
      setStatus(RUN_FAILED);
      return null;
    }
  }, [submit]);

  return { status, scenarioId, context, beats, error, start, reset };
}
