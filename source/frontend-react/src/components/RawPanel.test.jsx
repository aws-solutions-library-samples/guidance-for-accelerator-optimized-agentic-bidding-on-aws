/**
 * @vitest-environment jsdom
 */
// RawPanel.test.jsx -- the "Request + Mutations Applied" view must show the
// request as the containers left it. Regression coverage for a generic path
// writer that set `imp["imp-1"] = ids` for an ACTIVATE_DEALS at /imp/imp-1 (a
// string key on an array, dropped by JSON.stringify), so an activated deal was
// shown by the offers panel and silently absent from this view.
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";

vi.mock("./GsapTooltip", () => ({ default: ({ children }) => <div>{children}</div> }));

import RawPanel from "./RawPanel.jsx";
import { normalize } from "../utils/normalizer.js";

const payload = {
  lifecycle: "LIFECYCLE_SSP_BID_REQUEST",
  bid_request: {
    id: "req-home",
    imp: [
      {
        id: "imp-1",
        bidfloor: 2.0,
        pmp: { deals: [{ id: "deal-remnant-open", bidfloor: 1.0, at: 3 }] },
      },
    ],
    site: { domain: "hearth-and-home.example" },
    user: { id: "u-1" },
  },
};

const activate = { intent: 2, op: 1, path: "/imp/imp-1", ids: ["deal-home-premium"] };
const floor = { intent: 4, op: 3, path: "/imp/imp-1/deals/deal-home-premium", adjust_deal: { bidfloor: 6.5 } };
const missing = { intent: 4, op: 3, path: "/imp/imp-1/deals/deal-nowhere", adjust_deal: { bidfloor: 9 } };
// Path as widedeep_segment_activator/app.py emits it.
const segments = { intent: 1, op: 1, path: "/user/data/segment", ids: ["seg-home-decor"] };

function resultWith(mutations) {
  return normalize(
    {
      id: "req-home",
      mutations,
      metadata: {
        total_latency_ms: 40,
        containers: [
          { name: "ncf-deal-manager", status: "ok", latency_ms: 12, mutations: [activate] },
          { name: "widedeep-segment-activator", status: "ok", latency_ms: 8, mutations: [segments] },
          { name: "yield-optimizer-floor", status: "ok", latency_ms: 21, mutations: [floor, missing] },
        ],
      },
    },
    "grpc",
    payload,
  );
}

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => { root.unmount(); });
  document.body.removeChild(container);
});

function render(jsx) {
  act(() => { root.render(jsx); });
}

describe("RawPanel request section -- merged document", () => {
  it("shows a deal activated by ACTIVATE_DEALS inside imp.pmp.deals", () => {
    render(<RawPanel result={resultWith([activate])} payload={payload} section="request" />);
    const pre = container.querySelector("pre.raw-json");
    expect(pre.textContent).toContain('"deal-home-premium"');
    // The old writer left a stray key on the imp array; the merged document is the
    // envelope, not a mangled copy of it.
    expect(pre.textContent).not.toContain('"imp-1": [');
    expect(container.querySelector(".raw-section-header").textContent).toContain("1 of 1 mutations applied");
  });

  it("applies a later-stage floor to a deal an earlier stage activated", () => {
    render(<RawPanel result={resultWith([activate, floor])} payload={payload} section="request" />);
    const text = container.querySelector("pre.raw-json").textContent;
    const dealIdx = text.indexOf('"deal-home-premium"');
    expect(dealIdx).toBeGreaterThan(-1);
    // The floor written for the activated deal sits in that deal's object.
    expect(text.slice(dealIdx, dealIdx + 120)).toContain('"bidfloor": 6.5');
    expect(container.querySelector(".raw-section-header").textContent).toContain("2 of 2 mutations applied");
    expect(container.querySelector('[data-testid="raw-panel-not-applied"]')).toBeNull();
  });

  it("lists a mutation it could not apply instead of silently dropping it", () => {
    render(<RawPanel result={resultWith([activate, missing])} payload={payload} section="request" />);
    expect(container.querySelector(".raw-section-header").textContent).toContain("1 of 2 mutations applied");
    const list = container.querySelector('[data-testid="raw-panel-not-applied"]');
    expect(list).not.toBeNull();
    expect(list.textContent).toContain("/imp/imp-1/deals/deal-nowhere");
  });

  it("shows activated segments under user.data", () => {
    render(<RawPanel result={resultWith([segments])} payload={payload} section="request" />);
    expect(container.querySelector("pre.raw-json").textContent).toContain('"seg-home-decor"');
  });

  it("highlights only the lines the mutations wrote", () => {
    render(<RawPanel result={resultWith([activate])} payload={payload} section="request" />);
    const highlighted = container.querySelectorAll(".raw-json-mutation-line");
    expect(highlighted.length).toBeGreaterThan(0);
    const joined = Array.from(highlighted).map((el) => el.textContent).join("");
    expect(joined).toContain("deal-home-premium");
    // The untouched remnant deal is not highlighted.
    expect(joined).not.toContain("deal-remnant-open");
  });
});

describe("RawPanel mutations section -- labels", () => {
  it("labels each card with the producing container's display name, not the bare stop id", () => {
    render(<RawPanel result={resultWith([activate])} payload={payload} section="mutations" />);
    const source = container.querySelector(".raw-mutation-source");
    expect(source.textContent).not.toBe("ncf");
    expect(source.textContent.length).toBeGreaterThan(3);
  });
});
