import { describe, expect, it } from "vitest";
import { SCENARIOS } from "../components/ScenarioCard.jsx";
import { applyTunerToPayload, scenarioUsesYieldOptimizer } from "./scenarioPayload.js";

const PAYLOAD = { bid_request: { imp: [{ id: "imp-1", bidfloor: 1 }] } };

describe("yield explore opt-in", () => {
  it("is sent on every scenario that runs a yield container", () => {
    const yieldScenarios = SCENARIOS.filter(scenarioUsesYieldOptimizer);
    expect(yieldScenarios.length).toBeGreaterThan(0);
    for (const scenario of yieldScenarios) {
      expect(applyTunerToPayload(PAYLOAD, scenario).ext.model_params.explore).toBe(true);
    }
  });

  it("is never sent on a scenario without a yield container", () => {
    for (const scenario of SCENARIOS.filter((s) => !scenarioUsesYieldOptimizer(s))) {
      expect(applyTunerToPayload(PAYLOAD, scenario).ext?.model_params?.explore).toBeUndefined();
    }
  });

  it("no scenario still offers an Explore control", () => {
    for (const scenario of SCENARIOS) {
      expect(scenario.controls || []).not.toContain("explore");
    }
  });
});
