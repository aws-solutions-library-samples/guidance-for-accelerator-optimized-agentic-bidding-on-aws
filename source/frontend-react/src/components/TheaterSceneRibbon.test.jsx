/**
 * @vitest-environment jsdom
 *
 * The scene ribbon reads the SUBMITTED request. A scenario is a publisher bid
 * request, described by the page it came from, the audience asserted on it and
 * the demand eligible to compete -- so the ribbon carries all three, and each
 * value has to come from the request rather than from anything a container
 * later contributed (BR-23).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

import { TheaterSceneRibbon } from "./TheaterPanels.jsx";
import { buildScenarioContext } from "../utils/theaterBeats.js";

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

function render(context) {
  act(() => {
    root.render(
      <TheaterSceneRibbon context={context} revealed stepLabel="Step 1 of 3" />
    );
  });
}

/** Facet label to rendered value. */
function facets() {
  return Object.fromEntries(
    Array.from(host.querySelectorAll(".th-scene-item")).map((item) => [
      item.querySelector(".th-scene-label").textContent,
      item.querySelector(".th-scene-value").textContent,
    ])
  );
}

const PAYLOAD = {
  id: "sample-x",
  bid_request: {
    id: "auction-x",
    imp: [
      {
        id: "imp-1",
        banner: { w: 300, h: 250 },
        bidfloor: 2.5,
        pmp: {
          deals: [
            { id: "deal-a", bidfloor: 3.0, at: 2 },
            { id: "deal-b", bidfloor: 1.0, at: 3 },
          ],
        },
      },
    ],
    site: { domain: "example.test", page: "https://example.test/a", cat: ["283"], cattax: 9 },
    user: {
      yob: 1986,
      gender: "F",
      data: [
        {
          id: "dmp-1",
          name: "Example DMP",
          segment: [
            { id: "seg-461", name: "Interior Decorating" },
            { id: "seg-142", name: "First Time Homeowner" },
          ],
        },
      ],
    },
    device: { ua: "Mozilla/5.0", geo: { country: "USA", region: "WA" } },
  },
};

describe("TheaterSceneRibbon", () => {
  it("carries all five facets of the request", () => {
    render(buildScenarioContext(PAYLOAD));
    expect(Object.keys(facets())).toEqual([
      "Publisher",
      "Page",
      "Content",
      "Audience",
      "Deals",
    ]);
  });

  it("shows the audience the request asserted, in request order", () => {
    render(buildScenarioContext(PAYLOAD));
    expect(facets().Audience).toBe("Interior Decorating, First Time Homeowner");
  });

  it("does not put raw demographic fields in the audience facet", () => {
    // yob and gender are request-borne too, but they are demographic fields the
    // request card already shows -- not audience assertions.
    render(buildScenarioContext(PAYLOAD));
    // Compared as whole entries, not as substrings: "F" is a substring of
    // "First Time Homeowner".
    const entries = facets().Audience.split(", ");
    expect(entries).toEqual(["Interior Decorating", "First Time Homeowner"]);
    expect(entries).not.toContain("1986");
    expect(entries).not.toContain("F");
  });

  it("counts the deals on every impression, not just the first", () => {
    const twoImps = JSON.parse(JSON.stringify(PAYLOAD));
    twoImps.bid_request.imp.push({
      id: "imp-2",
      banner: { w: 300, h: 600 },
      pmp: { deals: [{ id: "deal-c", bidfloor: 2.0, at: 2 }] },
    });
    render(buildScenarioContext(twoImps));
    expect(facets().Deals).toBe("3");
  });

  it("reports unknown rather than zero when the request carries no deals", () => {
    const noDeals = JSON.parse(JSON.stringify(PAYLOAD));
    delete noDeals.bid_request.imp[0].pmp;
    render(buildScenarioContext(noDeals));
    // "0" would read as a measured count of an open-market request; the ribbon's
    // absent-value treatment is the honest one.
    expect(facets().Deals).not.toBe("0");
  });

  it("reports unknown rather than an empty string when nothing was asserted", () => {
    const noData = JSON.parse(JSON.stringify(PAYLOAD));
    delete noData.bid_request.user.data;
    render(buildScenarioContext(noData));
    expect(facets().Audience.trim()).not.toBe("");
  });

  it("survives a null context without throwing", () => {
    render(null);
    expect(Object.keys(facets())).toHaveLength(5);
  });
});

/**
 * The ribbon absorbed the scenario name and the exit control when the bar above
 * it was removed to reclaim a row. Both had to survive the move: the exit is the
 * only way out of the embedded theater.
 */
describe("TheaterSceneRibbon — scenario name and exit", () => {
  const q = (id) => host.querySelector(`[data-testid="${id}"]`);

  it("shows the scenario name when given one", () => {
    act(() => root.render(
      <TheaterSceneRibbon context={null} revealed stepLabel="Step 1 of 7"
        scenarioName="Home & Lifestyle — Four-Way Deal Contest" />,
    ));
    expect(q("theater-scenario-name").textContent).toBe("Home & Lifestyle — Four-Way Deal Contest");
  });

  it("falls back to the product label when no scenario name is given", () => {
    act(() => root.render(
      <TheaterSceneRibbon context={null} revealed stepLabel="Step 1 of 7" />,
    ));
    expect(q("theater-scenario-name").textContent).toBe("ARTF Auction Theater");
  });

  it("carries the exit control, and calls onExit when it is clicked", () => {
    let exits = 0;
    act(() => root.render(
      <TheaterSceneRibbon context={null} revealed stepLabel="Step 1 of 7"
        scenarioName="S" onExit={() => { exits += 1; }} />,
    ));
    const btn = q("theater-exit");
    expect(btn).not.toBeNull();
    act(() => btn.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(exits).toBe(1);
  });

  it("renders no exit control when there is nothing to exit to", () => {
    act(() => root.render(
      <TheaterSceneRibbon context={null} revealed stepLabel="Step 1 of 7" scenarioName="S" />,
    ));
    expect(q("theater-exit")).toBeNull();
  });

  it("still carries the step label beside the new controls", () => {
    act(() => root.render(
      <TheaterSceneRibbon context={null} revealed stepLabel="Step 6 of 7"
        scenarioName="S" onExit={() => {}} />,
    ));
    expect(q("theater-progress-label").textContent).toBe("Step 6 of 7");
  });
});
