// Which responses may be shown as a real auction.
//
// The offers column renders whatever it is handed, and the notice it shows is
// chosen from the fixture marker. So the gate on what counts as LIVE is what keeps
// a fixture, an error body, or a half-built object from being presented as an
// auction that happened.

import { describe, it, expect } from "vitest";
import {
  isLiveAuctionResponse,
  isFixtureResponse,
  capturedBidResponse,
} from "./bidResponseFixture.js";

const LIVE = {
  id: "req-1",
  cur: "USD",
  seatbid: [
    { seat: "artfhouse", bid: [{ crid: "cr-cedar-300x250", price: 6.35 }] },
    { seat: "amt", bid: [{ crid: "banner_creative_1", price: 3.25 }] },
  ],
  artf_meta: { source: "prebid", seats: ["amt", "artfhouse"], hop_ms: 239 },
};

describe("isLiveAuctionResponse", () => {
  it("accepts a response the orchestrator marked as coming from prebid", () => {
    expect(isLiveAuctionResponse(LIVE)).toBe(true);
  });

  it("rejects the captured fixture", () => {
    expect(isLiveAuctionResponse(capturedBidResponse)).toBe(false);
    // And the fixture still identifies itself, so the notice keeps working.
    expect(isFixtureResponse(capturedBidResponse)).toBe(true);
  });

  it("rejects a fixture even if something attached prebid provenance to it", () => {
    // Defends the ordering of the two checks: the fixture marker wins.
    const mislabelled = { ...capturedBidResponse, artf_meta: { source: "prebid" } };
    expect(isLiveAuctionResponse(mislabelled)).toBe(false);
  });

  it("rejects the 501 body returned when Prebid is not deployed", () => {
    const notDeployed = {
      error: "prebid_not_deployed",
      prebid: "not_configured",
      endpoint: null,
    };
    expect(isLiveAuctionResponse(notDeployed)).toBe(false);
  });

  it("rejects the failure bodies for timeout, unreachable and rejection", () => {
    for (const error of [
      "prebid_timeout",
      "prebid_unreachable",
      "prebid_returned_non_json",
      "prebid_rejected_request",
    ]) {
      expect(isLiveAuctionResponse({ error })).toBe(false);
    }
  });

  it("rejects a bid-response shape with no provenance", () => {
    // The decisive case: something that LOOKS like an auction but was not
    // reported as one. Ruling out only the fixture would have let this through.
    const { artf_meta, ...withoutProvenance } = LIVE;
    expect(artf_meta).toBeDefined();
    expect(isLiveAuctionResponse(withoutProvenance)).toBe(false);
  });

  it("rejects provenance from anywhere other than prebid", () => {
    expect(isLiveAuctionResponse({ ...LIVE, artf_meta: { source: "fixture" } })).toBe(false);
    expect(isLiveAuctionResponse({ ...LIVE, artf_meta: { source: "" } })).toBe(false);
    expect(isLiveAuctionResponse({ ...LIVE, artf_meta: {} })).toBe(false);
  });

  it("rejects non-objects rather than throwing", () => {
    for (const value of [null, undefined, "", 0, false, "prebid", []]) {
      expect(isLiveAuctionResponse(value)).toBe(false);
    }
  });

  it("accepts a live auction that produced no bids", () => {
    // An empty auction is a real outcome and must still count as live, so the
    // panel can say "no bids" instead of falling back to fixture prices.
    const empty = { id: "req-1", cur: "USD", artf_meta: { source: "prebid", seats: [] } };
    expect(isLiveAuctionResponse(empty)).toBe(true);
  });
});
