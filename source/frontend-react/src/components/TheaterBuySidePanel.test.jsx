/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { TheaterBuySidePanel } from "./TheaterPanels.jsx";
import { illustrativeOutcomeFor } from "../utils/theaterIllustrative.js";

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

describe("TheaterBuySidePanel", () => {
  it("shows the illustrative label", () => {
    render(<TheaterBuySidePanel outcome={illustrativeOutcomeFor("yield-optimizer")} revealed />);
    const label = container.querySelector('[data-testid="theater-buy-side-illustrative-label"]');
    expect(label).not.toBeNull();
    expect(label.textContent).toMatch(/illustrative/i);
  });

  it("shows the illustrative label even before the column is revealed", () => {
    render(<TheaterBuySidePanel outcome={illustrativeOutcomeFor("yield-optimizer")} revealed={false} />);
    expect(container.querySelector('[data-testid="theater-buy-side-illustrative-label"]')).not.toBeNull();
  });

  it("shows the illustrative label with no outcome at all", () => {
    render(<TheaterBuySidePanel outcome={null} revealed />);
    expect(container.querySelector('[data-testid="theater-buy-side-illustrative-label"]')).not.toBeNull();
  });

  it("states that the system does not run an auction", () => {
    render(<TheaterBuySidePanel outcome={illustrativeOutcomeFor("yield-optimizer")} revealed />);
    const text = container.querySelector('[data-testid="theater-buy-side-illustrative-label"]').textContent;
    expect(text).toMatch(/does not run\s+an auction/i);
  });

  it("renders a bidder that did not bid as 'no bid' rather than as zero", () => {
    render(<TheaterBuySidePanel outcome={illustrativeOutcomeFor("yield-optimizer")} revealed />);
    expect(container.textContent).toMatch(/no bid/);
    expect(container.textContent).not.toMatch(/\$0\.00/);
  });
});

describe("illustrativeOutcomeFor", () => {
  it("is deterministic for the same scenario", () => {
    expect(illustrativeOutcomeFor("yield-optimizer")).toEqual(illustrativeOutcomeFor("yield-optimizer"));
  });

  it("always marks itself illustrative, including for an unknown scenario", () => {
    expect(illustrativeOutcomeFor("yield-optimizer").isIllustrative).toBe(true);
    expect(illustrativeOutcomeFor("no-such-scenario").isIllustrative).toBe(true);
    expect(illustrativeOutcomeFor(undefined).isIllustrative).toBe(true);
  });
});
