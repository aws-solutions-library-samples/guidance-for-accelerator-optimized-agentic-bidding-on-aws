/**
 * @vitest-environment jsdom
 *
 * The sidebar-collapse behaviour, tested where it lives. `App.jsx` owns the
 * decision, but App drags in Cognito, GSAP and the comparison provider, so this
 * exercises the same decision through a minimal host that mirrors App's structure.
 *
 * What is being pinned: opening the Theater COLLAPSES the sidebar to a rail and
 * keeps the picker mounted, the rail can be expanded again while the Theater is
 * open, closing restores the panel, and the run that produced the timeline
 * survives the round trip — the Theater reuses the run area rather than
 * discarding what was there.
 *
 * REVERSAL, recorded rather than slipped in: this file previously asserted that
 * opening the Theater REMOVED the sidebar from the DOM, and that
 * `app-layout--theater` existed only as a marker. Both were true of the old
 * behaviour and are now wrong by intent — the Theater opened as a full-viewport
 * overlay, which made it read as a separate page. The panel now stays visible
 * around it, so those two assertions are replaced by the collapse ones below.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React, { useState, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

// Resolves a minimal payload rather than hanging: the `▶ Send` path goes through
// loadScenarioPayload, so a never-resolving fetch would make Send a no-op and the
// "the run survives" assertion would pass vacuously.
const SAMPLE = {
  id: "sample-under-test",
  applicable_intents: ["ADD_METRICS"],
  bid_request: { id: "a", imp: [{ id: "imp-1", banner: { w: 300, h: 250 }, bidfloor: 1 }] },
};
vi.stubGlobal(
  "fetch",
  vi.fn(async () => ({ ok: true, status: 200, json: async () => SAMPLE }))
);

vi.mock("../hooks/useOrchestratorClientWithBase.js", () => ({
  useOrchestratorClientWithBase: () => ({
    submit: vi.fn(),
    loading: false,
    error: null,
    result: null,
    setResult: vi.fn(),
    cancel: vi.fn(),
  }),
}));

import AuctionTheater from "./AuctionTheater.jsx";
import Sidebar from "./Sidebar.jsx";
import { SCENARIOS } from "./ScenarioCard.jsx";

/** The same open/close decision App.jsx makes, with nothing else attached. */
function Host() {
  const [theaterRun, setTheaterRun] = useState(null);
  const [lastRun, setLastRun] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const open = useCallback((scenario, params) => {
    setTheaterRun({ scenario, params });
    setCollapsed(true);
  }, []);
  const close = useCallback(() => {
    setTheaterRun(null);
    setCollapsed(false);
  }, []);
  const theaterOpen = !!theaterRun;

  return (
    <div
      className={
        `app-layout${theaterOpen ? " app-layout--theater" : ""}` +
        `${theaterOpen && !collapsed ? " app-layout--panel-open" : ""}`
      }
    >
      <Sidebar
        submit={async (p) => setLastRun(p)}
        onOpenTheater={open}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((c) => !c)}
      />
      <main className="app-main">
        {theaterOpen ? (
          <AuctionTheater
            scenario={theaterRun.scenario}
            params={theaterRun.params}
            onExit={close}
          />
        ) : (
          <div data-testid="run-area">{lastRun ? "timeline" : "empty"}</div>
        )}
      </main>
    </div>
  );
}

let host;
let root;

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<Host />));
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const click = (el) => act(() => el.dispatchEvent(new MouseEvent("click", { bubbles: true })));
// No reveal click: the dropdown selects the scenario and the card shows its
// actions immediately.
const openTheater = () =>
  click(host.querySelector('[data-testid="scenario-open-theater"]'));

describe("opening the Theater from a scenario card", () => {
  it("starts with the sidebar present and no Theater", () => {
    expect(host.querySelector(".app-sidebar")).not.toBeNull();
    expect(host.querySelector(".th-stage")).toBeNull();
  });

  it("collapses the sidebar to a rail instead of removing it", () => {
    // The Theater is a view inside this application. Unmounting the panel is what
    // made it read as a separate page.
    openTheater();
    const sidebar = host.querySelector(".app-sidebar");
    expect(sidebar).not.toBeNull();
    expect(sidebar.className).toContain("app-sidebar--collapsed");
  });

  it("keeps the scenario picker mounted while collapsed", () => {
    // Hidden with CSS, not conditionally rendered, so the selected scenario and
    // its tuner values survive collapsing and expanding.
    openTheater();
    expect(host.querySelector('[data-testid="scenario-picker"]')).not.toBeNull();
  });

  it("can expand the panel again while the Theater is open", () => {
    openTheater();
    click(host.querySelector('[data-testid="sidebar-toggle"]'));
    const sidebar = host.querySelector(".app-sidebar");
    expect(sidebar.className).not.toContain("app-sidebar--collapsed");
    // The Theater stays open; expanding is not a way out of it.
    expect(host.querySelector(".th-stage")).not.toBeNull();
  });

  it("marks the layout while the Theater is open", () => {
    openTheater();
    expect(host.querySelector(".app-layout").className).toContain("app-layout--theater");
  });

  it("does not mark the panel as open while it is a rail", () => {
    // The marker is what lets the Theater refuse to compress and scroll instead.
    // Against a rail it must fit the remaining space exactly, so the marker must
    // be absent or the Theater is floored at 860px and overflows the viewport.
    openTheater();
    expect(host.querySelector(".app-layout").className).not.toContain("app-layout--panel-open");
  });

  it("marks the panel as open once it is expanded", () => {
    openTheater();
    click(host.querySelector('[data-testid="sidebar-toggle"]'));
    expect(host.querySelector(".app-layout").className).toContain("app-layout--panel-open");
  });

  it("drops the panel-open marker when the panel is collapsed again", () => {
    openTheater();
    click(host.querySelector('[data-testid="sidebar-toggle"]'));
    click(host.querySelector('[data-testid="sidebar-toggle"]'));
    expect(host.querySelector(".app-layout").className).not.toContain("app-layout--panel-open");
  });

  it("renders the Theater in the run area", () => {
    openTheater();
    expect(host.querySelector(".app-main .th-stage")).not.toBeNull();
    expect(host.querySelector('[data-testid="run-area"]')).toBeNull();
  });

  it("passes the scenario through, so the Theater names it", () => {
    openTheater();
    const shown = host.querySelector('[data-testid="theater-scenario-name"]').textContent;
    expect(SCENARIOS.map((s) => s.name)).toContain(shown);
  });
});

describe("closing the Theater", () => {
  it("expands the sidebar again and restores the run area", () => {
    openTheater();
    click(host.querySelector('[data-testid="theater-exit"]'));
    const sidebar = host.querySelector(".app-sidebar");
    expect(sidebar).not.toBeNull();
    expect(sidebar.className).not.toContain("app-sidebar--collapsed");
    expect(host.querySelector('[data-testid="run-area"]')).not.toBeNull();
    expect(host.querySelector(".th-stage")).toBeNull();
  });

  it("clears the layout marker", () => {
    openTheater();
    click(host.querySelector('[data-testid="theater-exit"]'));
    expect(host.querySelector(".app-layout").className).not.toContain("app-layout--theater");
  });

  it("does not discard a run that happened before the Theater opened", async () => {
    // Send a scenario the ordinary way first, so there is a timeline to lose.
    await act(async () => {
      host
        .querySelector('[data-testid="scenario-send"]')
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(host.querySelector('[data-testid="run-area"]').textContent).toBe("timeline");

    openTheater();
    click(host.querySelector('[data-testid="theater-exit"]'));
    expect(host.querySelector('[data-testid="run-area"]').textContent).toBe("timeline");
  });
});

describe("the sidebar after the load test moved out", () => {
  it("carries the scenario picker and nothing else", () => {
    expect(host.querySelector('[data-testid="scenario-picker"]')).not.toBeNull();
    // The load-test launcher moved to the Governance page as Step 1.
    expect(host.querySelector(".load-test-panel")).toBeNull();
    expect(host.textContent).not.toMatch(/Load Test/i);
  });
});
