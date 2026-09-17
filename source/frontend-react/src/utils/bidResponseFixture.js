// bidResponseFixture.js — a captured bid response, for building the offers
// column before the Prebid stack is deployed.
//
// THIS IS A FIXTURE. It is labelled here, in the module, and not only in
// documentation, because the surface that renders it selects its notice from the
// shape of the data — so the marker below is what makes that selection correct
// rather than incidental (FR-31, BR-19).
//
// It is:
//   - deterministic, so the same scenario always renders the same view,
//   - inspectable, being written here rather than generated,
//   - shaped exactly like a real Prebid response, so the code path exercised is
//     the real one and not a special case.
//
// It contains no measured value. Prices are the demand catalog's declared CPMs;
// the winner marker is where Prebid would place targeting.

/** Present only on fixtures. Real responses do not carry it. */
export const FIXTURE_MARKER = "__fixture__";

export const capturedBidResponse = Object.freeze({
  id: "fixture-request-1",
  cur: "USD",
  [FIXTURE_MARKER]: true,

  seatbid: [
    {
      seat: "artfhouse",
      bid: [
        {
          id: "bid-1",
          impid: "imp-1",
          price: 6.35,
          dealid: "deal-home-premium",
          adomain: ["cedarandco.example"],
          crid: "cr-cedar-300x250",
          w: 300,
          h: 250,
          ext: {
            prebid: {
              targeting: { hb_pb: "6.30", hb_bidder: "artfhouse" },
              artf: {
                campaignId: "camp-cedar",
                campaignName: "Cedar & Co Furnishings",
                dealId: "deal-home-premium",
              },
            },
          },
        },
        {
          id: "bid-2",
          impid: "imp-1",
          price: 3.1,
          dealid: "deal-retail-run",
          adomain: ["northlakehome.example"],
          crid: "cr-northlake-300x250",
          w: 300,
          h: 250,
          ext: {
            prebid: {
              artf: {
                campaignId: "camp-northlake",
                campaignName: "Northlake Home Goods",
                dealId: "deal-retail-run",
              },
            },
          },
        },
      ],
    },
  ],

  ext: {
    // Prebid's own record: a bid it rejected below the enforced deal floor.
    seatnonbid: [
      {
        seat: "artfhouse",
        nonbid: [
          {
            impid: "imp-1",
            statuscode: 301,
            ext: {
              artf: {
                campaignId: "camp-vantage",
                campaignName: "Vantage Motorsport",
                dealId: "deal-auto-brand",
                exclusionReason: "below_floor",
              },
            },
          },
        ],
      },
    ],

    // The demand endpoint's own record: campaigns it considered and made no offer
    // for. These cannot ride in seatnonbid — Prebid records no seatnonbid entry for a
    // seat that did bid, so without this block they would be invisible.
    artf: {
      excluded: [
        {
          campaignId: "camp-harbour",
          campaignName: "Harbour Financial",
          dealId: "deal-finance-pmp",
          exclusionReason: "deal_suppressed",
        },
        {
          campaignId: "camp-openfield",
          campaignName: "Openfield Marketplace",
          dealId: null,
          exclusionReason: "not_targeted",
        },
      ],
    },
  },
});

/** True when the response is this module's fixture. */
export function isFixtureResponse(bidResponse) {
  return bidResponse?.[FIXTURE_MARKER] === true;
}

/**
 * True only for a response that came from a real exchange.
 *
 * The orchestrator writes `artf_meta.source = "prebid"` on the path that actually
 * posts to Prebid, so this is the one positive signal that a response is live.
 * Everything else — a 501 body, an error payload, a fixture, a partially built
 * object — is not an auction result and must not be shown as one.
 *
 * Checked positively rather than by ruling out the fixture: "not the fixture" is
 * true of every malformed thing as well, so it would let a non-auction through.
 */
export function isLiveAuctionResponse(bidResponse) {
  if (!bidResponse || typeof bidResponse !== "object") return false;
  if (bidResponse[FIXTURE_MARKER] === true) return false;
  return bidResponse.artf_meta?.source === "prebid";
}
