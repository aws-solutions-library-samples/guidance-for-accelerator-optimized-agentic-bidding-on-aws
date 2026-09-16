/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { TheaterModeSelector, THEATER_MODES } from "./TheaterModeSelector.jsx";

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function render(node) {
  act(() => root.render(node));
}

const q = (id) => container.querySelector(`[data-testid="${id}"]`);

describe("TheaterModeSelector", () => {
  it("renders both modes", () => {
    render(<TheaterModeSelector mode="sell" availableModes={THEATER_MODES} onChange={() => {}} />);
    expect(q("theater-mode-option-sell")).not.toBeNull();
    expect(q("theater-mode-option-buy")).not.toBeNull();
  });

  it("renders the unavailable mode disabled, with its reason visible", () => {
    render(<TheaterModeSelector mode="sell" availableModes={THEATER_MODES} onChange={() => {}} />);
    const buy = q("theater-mode-option-buy");
    expect(buy.disabled).toBe(true);
    expect(q("theater-mode-reason-buy").textContent).toMatch(/not built yet/i);
  });

  it("marks the active mode", () => {
    render(<TheaterModeSelector mode="sell" availableModes={THEATER_MODES} onChange={() => {}} />);
    expect(q("theater-mode-option-sell").getAttribute("aria-pressed")).toBe("true");
    expect(q("theater-mode-option-buy").getAttribute("aria-pressed")).toBe("false");
  });

  it("reports a selection for an available mode", () => {
    const seen = [];
    const modes = [
      { id: "sell", label: "Sell side", available: true },
      { id: "buy", label: "Buy side", available: true },
    ];
    render(<TheaterModeSelector mode="sell" availableModes={modes} onChange={(m) => seen.push(m)} />);
    act(() => {
      q("theater-mode-option-buy").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(seen).toEqual(["buy"]);
  });

  it("does not report a selection for an unavailable mode", () => {
    const seen = [];
    render(<TheaterModeSelector mode="sell" availableModes={THEATER_MODES} onChange={(m) => seen.push(m)} />);
    act(() => {
      q("theater-mode-option-buy").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(seen).toEqual([]);
  });

  it("availability is data — adding a mode needs no component change", () => {
    const modes = [...THEATER_MODES, { id: "audit", label: "Audit", available: true }];
    render(<TheaterModeSelector mode="sell" availableModes={modes} onChange={() => {}} />);
    expect(q("theater-mode-option-audit")).not.toBeNull();
    expect(q("theater-mode-option-audit").disabled).toBe(false);
  });
});
