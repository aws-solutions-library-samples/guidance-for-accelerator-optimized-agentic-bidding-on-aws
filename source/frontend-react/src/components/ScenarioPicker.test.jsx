/**
 * @vitest-environment jsdom
 *
 * The picker's job is to make the two ARTF surfaces navigable without showing
 * eleven cards at once. The surface split is real — `orchestrator/app.py` gives
 * exactly one container a response-side intent — so filtering by it has to be
 * driven by the declared `surface` field and nothing else.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import ScenarioPicker, { scenariosOnSurface } from "./ScenarioPicker.jsx";
import {
  SCENARIOS,
  SURFACE_REQUEST,
  SURFACE_RESPONSE,
} from "./ScenarioCard.jsx";

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
      <ScenarioPicker
        activeScenarioId={null}
        runningScenarioId={null}
        onSelect={() => {}}
        onSend={() => {}}
        onOpenTheater={() => {}}
        {...props}
      />
    );
  });
}

const click = (el) => act(() => el.dispatchEvent(new MouseEvent("click", { bubbles: true })));
const optionValues = () =>
  Array.from(host.querySelectorAll('[data-testid="scenario-select"] option')).map((o) => o.value);

describe("surface filtering", () => {
  it("lists only request-surface scenarios by default", () => {
    render();
    const expected = SCENARIOS.filter((s) => s.surface === SURFACE_REQUEST).map((s) => s.id);
    expect(optionValues()).toEqual(expected);
  });

  it("lists only response-surface scenarios after switching", () => {
    render();
    click(host.querySelector(`[data-testid="scenario-surface-${SURFACE_RESPONSE}"]`));
    const expected = SCENARIOS.filter((s) => s.surface === SURFACE_RESPONSE).map((s) => s.id);
    expect(optionValues()).toEqual(expected);
  });

  it("partitions every scenario onto exactly one surface", () => {
    // No scenario may be missing from both lists, or it would be unreachable in
    // the UI while still passing every other test.
    const request = scenariosOnSurface(SURFACE_REQUEST);
    const response = scenariosOnSurface(SURFACE_RESPONSE);
    expect(request.length + response.length).toBe(SCENARIOS.length);
    expect(request.filter((s) => response.includes(s))).toEqual([]);
  });

  it("filters on the declared surface, not on the scenario's name or file", () => {
    // A response scenario whose id does not start with "response-" must still be
    // grouped correctly. This guards against a future rename re-introducing
    // name-based filtering.
    const byField = scenariosOnSurface(SURFACE_RESPONSE).map((s) => s.id).sort();
    const byName = SCENARIOS.filter((s) => s.id.startsWith("response-")).map((s) => s.id).sort();
    expect(byField).toEqual(byName);
    expect(scenariosOnSurface(SURFACE_RESPONSE).every((s) => s.surface === SURFACE_RESPONSE)).toBe(true);
  });
});

describe("selection", () => {
  it("shows exactly one card", () => {
    render();
    expect(host.querySelectorAll(".scenario").length).toBe(1);
  });

  it("selects the first scenario of the new surface when the toggle changes", () => {
    // Leaving the previous selection would show an empty panel, which reads as a
    // load failure rather than as a switched filter.
    const onSelect = vi.fn();
    render({ onSelect });
    click(host.querySelector(`[data-testid="scenario-surface-${SURFACE_RESPONSE}"]`));
    expect(onSelect).toHaveBeenCalled();
    const picked = onSelect.mock.calls.at(-1)[0];
    expect(picked.surface).toBe(SURFACE_RESPONSE);
    expect(picked.id).toBe(scenariosOnSurface(SURFACE_RESPONSE)[0].id);
  });

  it("never leaves the card empty after a toggle switch", () => {
    render();
    click(host.querySelector(`[data-testid="scenario-surface-${SURFACE_RESPONSE}"]`));
    expect(host.querySelectorAll(".scenario").length).toBe(1);
  });

  it("reports the scenario the dropdown picked", () => {
    const onSelect = vi.fn();
    render({ onSelect });
    const select = host.querySelector('[data-testid="scenario-select"]');
    const target = scenariosOnSurface(SURFACE_REQUEST)[2];
    select.value = target.id;
    act(() => select.dispatchEvent(new Event("change", { bubbles: true })));
    expect(onSelect.mock.calls.at(-1)[0].id).toBe(target.id);
  });

  it("does not switch surface when the same toggle is clicked again", () => {
    const onSelect = vi.fn();
    render({ onSelect });
    click(host.querySelector(`[data-testid="scenario-surface-${SURFACE_REQUEST}"]`));
    expect(onSelect).not.toHaveBeenCalled();
  });
});

describe("what the option and the chips show", () => {
  it("names the scenario and its intents in each option", () => {
    render();
    const first = scenariosOnSurface(SURFACE_REQUEST)[0];
    const option = host.querySelector('[data-testid="scenario-select"] option');
    expect(option.textContent).toContain(first.name);
    for (const tag of first.tags) {
      expect(option.textContent).toContain(tag.label);
    }
  });

  it("shows the selected scenario's intents as the same chips the card uses", () => {
    render();
    const first = scenariosOnSurface(SURFACE_REQUEST)[0];
    const chips = Array.from(
      host.querySelectorAll('[data-testid="scenario-picker-intents"] .tag')
    ).map((el) => el.textContent);
    expect(chips).toEqual(first.tags.map((t) => t.label));
  });
});

describe("the two card actions", () => {
  it("offers both Send and Step through in Auction Theater", () => {
    render();
    click(host.querySelector(".scenario"));
    render({ activeScenarioId: scenariosOnSurface(SURFACE_REQUEST)[0].id });
    expect(host.querySelector('[data-testid="scenario-send"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="scenario-open-theater"]')).not.toBeNull();
  });

  it("hands the same tuner values to both actions", () => {
    // The two paths must not disagree about what was submitted. Both read from
    // the card's single getParams(), and this pins that they agree.
    const first = scenariosOnSurface(SURFACE_REQUEST)[0];
    const onSend = vi.fn();
    const onOpenTheater = vi.fn();
    render({ activeScenarioId: first.id, onSend, onOpenTheater });

    click(host.querySelector('[data-testid="scenario-send"]'));
    click(host.querySelector('[data-testid="scenario-open-theater"]'));

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onOpenTheater).toHaveBeenCalledTimes(1);
    expect(onOpenTheater.mock.calls[0][0].id).toBe(first.id);
    expect(onOpenTheater.mock.calls[0][1]).toEqual(onSend.mock.calls[0][1]);
  });
});

describe("per-scenario tuner defaults", () => {
  it("starts a response scenario at its own calibrated values", () => {
    // Each response scenario places the shader's price ceiling relative to its
    // own floor and bid, and those placements are what separate "shaded",
    // "floor-clamped" and "not shaded". A shared default would collapse them.
    const responses = scenariosOnSurface(SURFACE_RESPONSE);
    const withDefaults = responses.filter((s) => s.defaults?.convValue != null);
    expect(withDefaults.length).toBe(responses.length);

    const values = new Set(responses.map((s) => s.defaults.convValue));
    expect(values.size).toBeGreaterThan(1);
  });

  it("renders the conversion-value slider at the scenario's own default", () => {
    const target = scenariosOnSurface(SURFACE_RESPONSE).find(
      (s) => s.controls.includes("convValue")
    );
    render({ activeScenarioId: target.id });
    click(host.querySelector(`[data-testid="scenario-surface-${SURFACE_RESPONSE}"]`));
    render({ activeScenarioId: target.id });

    const rows = Array.from(host.querySelectorAll(".tuner-row"));
    const convRow = rows.find((r) => r.querySelector("label")?.textContent === "Conv Value");
    expect(convRow).toBeTruthy();
    expect(Number(convRow.querySelector("input").value)).toBe(target.defaults.convValue);
  });
});
