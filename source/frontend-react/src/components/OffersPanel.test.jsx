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
});
