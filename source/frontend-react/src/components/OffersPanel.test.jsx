/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import React from "react";
import { createRoot } from "react-dom/client";
import { act } from "react-dom/test-utils";
import { OffersPanel } from "./OffersPanel.jsx";
import { buildOfferViewModel } from "../utils/offerPresentationService.js";
import { capturedBidResponse, FIXTURE_MARKER } from "../utils/bidResponseFixture.js";

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
const qa = (sel) => Array.from(container.querySelectorAll(sel));

describe("OffersPanel", () => {
  it("labels the column offers rather than bidders", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed />);
    const title = container.querySelector(".th-col-title");
    expect(title.textContent).toBe("Offers");
    expect(container.textContent).not.toMatch(/bidders/i);
  });

  it("renders every candidate, including those that made no offer", () => {
    // Two offers, one Prebid non-bid, two endpoint exclusions — the never-offered
    // campaigns arrive from ext.artf.excluded, which seatnonbid cannot carry.
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed />);
    expect(qa(".th-offer")).toHaveLength(5);
  });

  it("gives a non-offering candidate a visible reason rather than omitting it", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed />);
    const reason = q("offer-row-reason-camp-harbour");
    expect(reason).not.toBeNull();
    expect(reason.textContent).toMatch(/suppressed/i);
  });

  it("shows the winner as deal plus campaign, with price as supporting detail", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed />);
    const winner = q("offers-panel-winner");
    expect(winner).not.toBeNull();
    expect(winner.textContent).toMatch(/deal-home-premium/);
    expect(winner.textContent).toMatch(/Cedar & Co Furnishings/);
    expect(winner.textContent).toMatch(/cleared at \$6\.35/);
    expect(q("offers-panel-unsold")).toBeNull();
  });

  it("renders an explicit unsold state, distinct from an empty one", () => {
    const vm = buildOfferViewModel({
      seatbid: [{ seat: "s", bid: [{ id: "b", impid: "i", price: 2 }] }],
    });
    render(<OffersPanel viewModel={vm} revealed />);
    expect(q("offers-panel-unsold")).not.toBeNull();
    expect(q("offers-panel-winner")).toBeNull();
    expect(q("offers-panel-unsold").textContent).toMatch(/no winning offer/i);
  });

  it("distinguishes the three categories without relying on colour alone", () => {
    const vm = buildOfferViewModel({
      ext: {
        seatnonbid: [
          {
            seat: "artfhouse",
            nonbid: [
              { impid: "i", ext: { artf: { campaignId: "c-dec", exclusionReason: "deal_suppressed" } } },
              { impid: "i", statuscode: 101, ext: { artf: { campaignId: "c-fail" } } },
              {
                impid: "i",
                ext: { artf: { campaignId: "c-skip", exclusionReason: "SkippedInsufficientBudget" } },
              },
            ],
          },
        ],
      },
    });
    render(<OffersPanel viewModel={vm} revealed />);

    // A machine-readable category, and a text marker a viewer can read.
    expect(q("offer-row-c-dec").dataset.category).toBe("ArtfDecision");
    expect(q("offer-row-c-fail").dataset.category).toBe("InfrastructureFailure");
    expect(q("offer-row-c-skip").dataset.category).toBe("NotAttempted");

    expect(q("offer-row-outcome-c-dec").textContent).toMatch(/decision/);
    expect(q("offer-row-outcome-c-fail").textContent).toMatch(/unavailable/);
    expect(q("offer-row-outcome-c-skip").textContent).toMatch(/not attempted/);
  });

  it("shows the illustrative notice for the fixture", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed />);
    expect(q("offers-panel-notice").textContent).toMatch(/illustrative/i);
  });

  it("shows the declared-inventory notice for a real response", () => {
    const real = { ...capturedBidResponse };
    delete real[FIXTURE_MARKER];
    render(<OffersPanel viewModel={buildOfferViewModel(real)} revealed />);
    const notice = q("offers-panel-notice").textContent;
    expect(notice).toMatch(/as returned by the auction/i);
    expect(notice).not.toMatch(/illustrative/i);
  });

  it("says so when a response carries no offers at all", () => {
    render(<OffersPanel viewModel={buildOfferViewModel({})} revealed />);
    expect(container.querySelector(".th-empty").textContent).toMatch(/no offers/i);
  });

  // The auction resolves against the enriched request, so no offer, price, reason
  // or winner may be on screen while enrichment is still running.
  it("withholds every offer and the winner until revealed", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed={false} />);
    expect(qa(".th-offer")).toHaveLength(0);
    expect(q("offers-panel-winner")).toBeNull();
    expect(q("offers-panel-unsold")).toBeNull();
    expect(q("offers-panel-notice")).toBeNull();
    expect(container.textContent).not.toMatch(/6\.35/);
    expect(container.textContent).not.toMatch(/Cedar & Co/);
  });

  it("keeps the column and states why it is empty while pending", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed={false} />);
    expect(q("offers-panel")).not.toBeNull();
    expect(container.querySelector(".th-col-title").textContent).toBe("Offers");
    expect(q("offers-panel-pending").textContent).toMatch(/enriched request/i);
  });

  it("shows the offers once revealed", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(capturedBidResponse)} revealed />);
    expect(q("offers-panel-pending")).toBeNull();
    expect(qa(".th-offer")).toHaveLength(5);
  });
});

// A fixture shown because the auction FAILED must not look like a fixture shown
// because Prebid was never deployed. Conflating them is what let a 401 read as an
// expected absence for the life of the feature.
describe("OffersPanel auction fault", () => {
  const vm = () => buildOfferViewModel(capturedBidResponse);

  it("says nothing when the auction was read", () => {
    render(<OffersPanel viewModel={vm()} revealed auctionFault={null} />);
    expect(q("offers-panel-fault")).toBeNull();
  });

  it("reports a failure as a failure, and says the offers are the fixture", () => {
    render(
      <OffersPanel
        viewModel={vm()}
        revealed
        auctionFault={{ kind: "failed", status: 401, detail: "Authentication required" }}
      />,
    );
    const fault = q("offers-panel-fault");
    expect(fault).not.toBeNull();
    expect(fault.dataset.faultKind).toBe("failed");
    expect(fault.textContent).toMatch(/could not be read/i);
    expect(fault.textContent).toMatch(/Authentication required/);
    expect(fault.textContent).toMatch(/captured fixture/i);
  });

  it("reports an undeployed exchange distinctly from a failure", () => {
    render(
      <OffersPanel
        viewModel={vm()}
        revealed
        auctionFault={{ kind: "not_deployed", detail: "Prebid Server is not deployed." }}
      />,
    );
    const fault = q("offers-panel-fault");
    expect(fault.dataset.faultKind).toBe("not_deployed");
    expect(fault.textContent).toMatch(/No live auction/i);
    expect(fault.textContent).not.toMatch(/could not be read/i);
  });

  it("keeps the fault out of the pending state, which has nothing to explain yet", () => {
    render(
      <OffersPanel
        viewModel={vm()}
        revealed={false}
        auctionFault={{ kind: "failed", detail: "boom" }}
      />,
    );
    expect(q("offers-panel-fault")).toBeNull();
    expect(q("offers-panel-pending")).not.toBeNull();
  });
});


// Three rows reading "unknown / No offer — unrecognised status 0" was the whole of
// what the simulator's non-bids showed. The response named the seat and the
// orchestrator can supply the price it returned, so both belong on the row.
describe("OffersPanel seat-level non-bids", () => {
  const LIVE_ISV = {
    cur: "USD",
    seatbid: [
      {
        seat: "artfhouse",
        bid: [
          {
            id: "b1",
            impid: "imp-1",
            price: 7.2,
            dealid: "deal-premium-auto",
            adomain: ["autoline.example"],
            ext: { prebid: { targeting: { hb_bidder: "artfhouse", hb_pb: "7.20" } } },
          },
        ],
      },
    ],
    ext: {
      seatnonbid: [
        {
          seat: "amt",
          nonbid: [
            {
              impid: "imp-1",
              statuscode: 0,
              ext: {
                artf: { observed: { returnedPrice: 3.25, impFloor: 4.0, currency: "USD" } },
              },
            },
            { impid: "imp-2", statuscode: 0 },
          ],
        },
      ],
    },
  };

  it("names the seat instead of rendering unknown", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(LIVE_ISV)} revealed />);
    const row = q("offer-row-amt");
    expect(row).not.toBeNull();
    expect(row.textContent).toMatch(/amt \(seat\)/);
    expect(row.textContent).not.toMatch(/unknown/i);
  });

  it("labels it a returned no-bid in the auction category, not not-attempted", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(LIVE_ISV)} revealed />);
    expect(q("offer-row-amt").dataset.category).toBe("AuctionOutcome");
    expect(q("offer-row-outcome-amt").textContent).toMatch(/No bid returned/);
    expect(q("offer-row-outcome-amt").textContent).toMatch(/auction/);
    expect(container.textContent).not.toMatch(/unrecognised status/i);
  });

  it("shows the price the seat returned against the floor it faced", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(LIVE_ISV)} revealed />);
    const reason = q("offer-row-reason-amt").textContent;
    expect(reason).toMatch(/3\.25/);
    expect(reason).toMatch(/4\.00/);
  });

  it("does not tell the reader the floor rejected it", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(LIVE_ISV)} revealed />);
    const reason = q("offer-row-reason-amt").textContent;
    expect(reason).not.toMatch(/rejected/i);
    expect(q("offer-row-amt").dataset.category).not.toBe("ArtfDecision");
  });

  it("keeps the winner intact alongside the seat rows", () => {
    render(<OffersPanel viewModel={buildOfferViewModel(LIVE_ISV)} revealed />);
    expect(q("offers-panel-winner").textContent).toMatch(/deal-premium-auto/);
    expect(q("offers-panel-winner").textContent).toMatch(/cleared at \$7\.20/);
  });
});


// Grouped non-bids render AFTER every bid, never between them. A summary line
// interleaved with real offers breaks the scan the column exists to support.
describe("OffersPanel grouped non-bids", () => {
  const WITH_GROUPS = {
    cur: "USD",
    seatbid: [
      {
        seat: "artfhouse",
        bid: [
          {
            id: "w",
            impid: "imp-1",
            price: 4.1,
            dealid: "deal-parenting-premium",
            adomain: ["brightstart.example"],
            ext: { prebid: { targeting: { hb_bidder: "artfhouse", hb_pb: "4.10" } } },
          },
          { id: "l", impid: "imp-1", price: 3.05, adomain: ["familynetwork.example"] },
        ],
      },
    ],
    ext: {
      seatnonbid: [{ seat: "amt", nonbid: [{ impid: "imp-1", statuscode: 0 }] }],
      artf: {
        excluded: [
          { campaignId: "c1", campaignName: "Openfield", exclusionReason: "below_floor", impId: "imp-1" },
          { campaignId: "c2", campaignName: "Cedar", exclusionReason: "no_deal_on_impression", impId: "imp-1" },
          { campaignId: "c3", campaignName: "Skyline", exclusionReason: "media_type_unsupported", impId: "imp-1" },
        ],
      },
    },
  };

  const vm = () => buildOfferViewModel(WITH_GROUPS);

  it("shows the bids on their own rows and nothing else at the top level", () => {
    render(<OffersPanel viewModel={vm()} revealed />);
    const bids = q("offers-panel-bids");
    expect(bids.querySelectorAll(".th-offer")).toHaveLength(2);
    expect(bids.textContent).toMatch(/brightstart\.example/);
    expect(bids.textContent).toMatch(/familynetwork\.example/);
    expect(bids.textContent).not.toMatch(/Cedar|Skyline|Openfield/);
  });

  // The explicit instruction: not in between valid rows.
  it("places every group after all of the bids in document order", () => {
    render(<OffersPanel viewModel={vm()} revealed />);
    const bids = q("offers-panel-bids");
    for (const group of qa(".th-offers-group")) {
      // DOCUMENT_POSITION_FOLLOWING (4) means the group comes after the bids block.
      const rel = bids.compareDocumentPosition(group);
      expect(rel & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    }
  });

  it("renders a group per kind, each with its count and breakdown", () => {
    render(<OffersPanel viewModel={vm()} revealed />);
    expect(q("offers-group-decision").textContent).toMatch(
      /1 campaign stopped by a sell-side decision/,
    );
    expect(q("offers-group-ineligible").textContent).toMatch(
      /2 campaigns not eligible for this impression/,
    );
    expect(q("offers-group-ineligible").textContent).toMatch(/1 no deal on the impression/);
    expect(q("offers-group-ineligible").textContent).toMatch(/1 creative format/);
    expect(q("offers-group-no_bid").textContent).toMatch(/1 seat returned no bid/);
    expect(q("offers-group-no_bid").textContent).toMatch(/1 amt/);
  });

  it("keeps the rows and their reasons inside the group, not discarded", () => {
    render(<OffersPanel viewModel={vm()} revealed />);
    const ineligible = q("offers-group-ineligible");
    expect(ineligible.querySelectorAll(".th-offer")).toHaveLength(2);
    expect(ineligible.textContent).toMatch(/Cedar/);
    expect(ineligible.textContent).toMatch(/no slot this campaign's creative could fill/i);
  });

  it("starts collapsed, and the count is legible while collapsed", () => {
    render(<OffersPanel viewModel={vm()} revealed />);
    for (const group of qa(".th-offers-group")) {
      expect(group.open).toBe(false);
      expect(group.querySelector("summary").textContent.length).toBeGreaterThan(0);
    }
  });

  it("says so when no seat bid at all, rather than showing an empty bid block", () => {
    const noBids = {
      ext: {
        artf: {
          excluded: [{ campaignId: "c", campaignName: "C", exclusionReason: "not_targeted" }],
        },
      },
    };
    render(<OffersPanel viewModel={buildOfferViewModel(noBids)} revealed />);
    expect(q("offers-panel-no-bids")).not.toBeNull();
    expect(q("offers-panel-bids")).toBeNull();
    expect(q("offers-group-ineligible")).not.toBeNull();
  });

  it("renders no group markup when every row bid", () => {
    const allBid = {
      seatbid: [
        {
          seat: "s",
          bid: [{ id: "a", impid: "i", price: 2, ext: { prebid: { targeting: { hb_bidder: "s" } } } }],
        },
      ],
    };
    render(<OffersPanel viewModel={buildOfferViewModel(allBid)} revealed />);
    expect(qa(".th-offers-group")).toHaveLength(0);
  });
});
