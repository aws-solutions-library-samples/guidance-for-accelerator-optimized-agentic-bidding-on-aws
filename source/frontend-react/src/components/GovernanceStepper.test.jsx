/**
 * @vitest-environment jsdom
 *
 * The stepper labels the journey; it does not gate it. Nothing on the Governance
 * page is truly sequential — a user who wants to read a past comparison should not
 * have to run a load test first — so every step stays reachable at all times.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import GovernanceStepper, { GOVERNANCE_STEPS } from "./GovernanceStepper.jsx";

let host;
let root;

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

function render(props = {}) {
  act(() => {
    root.render(
      <GovernanceStepper activeKey={GOVERNANCE_STEPS[0].key} onSelect={() => {}} {...props} /* nosemgrep: react-props-spreading test helper forwards per-test props */ />
    );
  });
}

const buttons = () => Array.from(host.querySelectorAll(".gov-step"));

describe("GovernanceStepper", () => {
  it("has three steps, numbered 1 to 3", () => {
    // Model Registry sits outside the stepper as a reference view, and the
    // outcome pipeline is part of the Train step rather than a step of its own.
    expect(GOVERNANCE_STEPS.map((s) => s.number)).toEqual([1, 2, 3]);
    expect(GOVERNANCE_STEPS.map((s) => s.key)).toEqual(["load-test", "train", "test-versions"]);
  });

  it("ends on Test Model Versions", () => {
    expect(GOVERNANCE_STEPS.at(-1).title).toBe("Test Model Versions");
  });

  it("renders one button per step, in order", () => {
    render();
    expect(buttons().map((b) => b.querySelector(".gov-step-title").textContent)).toEqual(
      GOVERNANCE_STEPS.map((s) => s.title)
    );
  });

  it("marks exactly one step current", () => {
    render({ activeKey: GOVERNANCE_STEPS[2].key });
    const current = buttons().filter((b) => b.getAttribute("aria-current") === "step");
    expect(current).toHaveLength(1);
    expect(current[0].querySelector(".gov-step-title").textContent).toBe(
      GOVERNANCE_STEPS[2].title
    );
  });

  it("leaves every step clickable regardless of which is current", () => {
    render();
    expect(buttons().filter((b) => b.disabled)).toEqual([]);
  });

  it("reports the step that was clicked", () => {
    const onSelect = vi.fn();
    render({ onSelect });
    act(() =>
      host
        .querySelector('[data-testid="governance-step-test-versions"]')
        .dispatchEvent(new MouseEvent("click", { bubbles: true }))
    );
    expect(onSelect).toHaveBeenCalledWith("test-versions");
  });

  it("allows jumping backwards as freely as forwards", () => {
    const onSelect = vi.fn();
    render({ activeKey: "test-versions", onSelect });
    act(() =>
      host
        .querySelector('[data-testid="governance-step-load-test"]')
        .dispatchEvent(new MouseEvent("click", { bubbles: true }))
    );
    expect(onSelect).toHaveBeenCalledWith("load-test");
  });
});
