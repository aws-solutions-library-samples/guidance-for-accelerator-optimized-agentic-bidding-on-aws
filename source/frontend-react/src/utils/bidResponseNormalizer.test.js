import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { normalizeOffers, resolveWinner } from "./bidResponseNormalizer.js";
import { capturedBidResponse } from "./bidResponseFixture.js";

describe("normalizeOffers", () => {
  it("draws rows from seatbid, ext.seatnonbid and ext.artf.excluded", () => {
    const offers = normalizeOffers(capturedBidResponse);
    expect(offers.filter((o) => o.offered)).toHaveLength(2);
    // One Prebid non-bid plus two endpoint exclusions.
    expect(offers.filter((o) => !o.offered)).toHaveLength(3);
  });

  it("reads a never-offered candidate from ext.artf.excluded", () => {
    // It cannot come from seatnonbid: Prebid records no seatnonbid entry for a seat
    // that did bid, so without this source the campaign would be invisible.
    const offers = normalizeOffers(capturedBidResponse);
    const suppressed = offers.find((o) => o.campaignId === "camp-harbour");
    expect(suppressed).toBeDefined();
    expect(suppressed.exclusionReason).toBe("deal_suppressed");
    expect(suppressed.campaignName).toBe("Harbour Financial");
  });

  it("reads a Prebid-rejected bid from ext.seatnonbid", () => {
    const offers = normalizeOffers(capturedBidResponse);
    const rejected = offers.find((o) => o.campaignId === "camp-vantage");
    expect(rejected.statusCode).toBe(301);
  });

  it("keeps both sources rather than treating them as interchangeable", () => {
    const offers = normalizeOffers(capturedBidResponse);
    expect(offers.filter((o) => o.statusCode === 301)).toHaveLength(1);
    expect(offers.filter((o) => o.exclusionReason === "not_targeted")).toHaveLength(1);
  });

  it("tolerates an exclusion entry missing its identity rather than throwing", () => {
    const offers = normalizeOffers({ ext: { artf: { excluded: [{}] } } });
    expect(offers).toHaveLength(1);
    expect(offers[0].campaignId).toBeNull();
  });

  it("reports a field the response does not carry as null, not a guess", () => {
    const offers = normalizeOffers({
      seatbid: [{ seat: "s", bid: [{ impid: "i1" }] }],
    });
    expect(offers[0].price).toBeNull();
    expect(offers[0].dealId).toBeNull();
  });

  it("returns an empty array for an empty or absent response", () => {
    expect(normalizeOffers(undefined)).toEqual([]);
    expect(normalizeOffers({})).toEqual([]);
  });

  it("reads the status code from a nonbid entry", () => {
    const offers = normalizeOffers(capturedBidResponse);
    const belowFloor = offers.find((o) => o.statusCode === 301);
    expect(belowFloor).toBeDefined();
    expect(belowFloor.campaignName).toBe("Vantage Motorsport");
  });
});

describe("resolveWinner", () => {
  it("resolves the marked winner as deal plus campaign", () => {
    const winner = resolveWinner(normalizeOffers(capturedBidResponse));
    expect(winner).not.toBeNull();
    expect(winner.dealId).toBe("deal-home-premium");
    expect(winner.campaignName).toBe("Cedar & Co Furnishings");
    expect(winner.clearedPrice).toBe(6.35);
  });

  it("returns null, not an empty object, when nothing is marked as winner", () => {
    const offers = normalizeOffers({
      seatbid: [{ seat: "s", bid: [{ impid: "i1", price: 1.0 }] }],
    });
    expect(resolveWinner(offers)).toBeNull();
  });

  it("does not rank bids itself — a higher unmarked price does not win", () => {
    const offers = normalizeOffers({
      seatbid: [
        {
          seat: "s",
          bid: [
            { id: "a", impid: "i1", price: 99.0 },
            {
              id: "b",
              impid: "i1",
              price: 1.0,
              dealid: "d1",
              ext: { prebid: { targeting: { hb_pb: "1.00" } } },
            },
          ],
        },
      ],
    });
    const winner = resolveWinner(offers);
    expect(winner.clearedPrice).toBe(1.0);
    expect(winner.dealId).toBe("d1");
  });
});

/* ------------------------------------------------ property tests, NFR-5 */

const arbBid = fc.record({
  id: fc.string({ minLength: 1, maxLength: 6 }),
  impid: fc.constantFrom("imp-1", "imp-2"),
  price: fc.double({ min: 0, max: 50, noNaN: true }),
  dealid: fc.option(fc.constantFrom("d1", "d2"), { nil: undefined }),
});

const arbNonBid = fc.record({
  impid: fc.constantFrom("imp-1", "imp-2"),
  statuscode: fc.option(fc.constantFrom(101, 301, 999), { nil: undefined }),
});

const arbExcluded = fc.record({
  campaignId: fc.string({ minLength: 1, maxLength: 6 }),
  campaignName: fc.string({ maxLength: 8 }),
  dealId: fc.option(fc.constantFrom("d1", "d2"), { nil: null }),
  exclusionReason: fc.constantFrom(
    "deal_suppressed",
    "below_floor",
    "not_targeted",
    "no_deal_on_impression",
  ),
});

const arbResponse = fc.record({
  seatbid: fc.array(fc.record({ seat: fc.constant("artfhouse"), bid: fc.array(arbBid, { maxLength: 5 }) }), {
    maxLength: 3,
  }),
  ext: fc.record({
    seatnonbid: fc.array(
      fc.record({ seat: fc.constant("artfhouse"), nonbid: fc.array(arbNonBid, { maxLength: 5 }) }),
      { maxLength: 3 },
    ),
    artf: fc.record({ excluded: fc.array(arbExcluded, { maxLength: 5 }) }),
  }),
});

describe("properties", () => {
  it("row completeness: every bid, nonbid and exclusion becomes exactly one row", () => {
    fc.assert(
      fc.property(arbResponse, (resp) => {
        const bids = (resp.seatbid ?? []).reduce((n, s) => n + (s.bid ?? []).length, 0);
        const nonbids = (resp.ext?.seatnonbid ?? []).reduce((n, s) => n + (s.nonbid ?? []).length, 0);
        const excluded = (resp.ext?.artf?.excluded ?? []).length;
        return normalizeOffers(resp).length === bids + nonbids + excluded;
      }),
    );
  });

  it("winner consistency: a resolved winner is always one of the rows", () => {
    fc.assert(
      fc.property(arbResponse, (resp) => {
        const offers = normalizeOffers(resp);
        const winner = resolveWinner(offers);
        if (winner === null) return offers.every((o) => o.markedWinner === false);
        return offers.some((o) => o.key === winner.offerKey);
      }),
    );
  });

  it("idempotence: normalising the same response twice yields the same rows", () => {
    fc.assert(
      fc.property(arbResponse, (resp) => {
        return JSON.stringify(normalizeOffers(resp)) === JSON.stringify(normalizeOffers(resp));
      }),
    );
  });
});

// The winner marker is the UNPREFIXED targeting key, not "targeting exists".
//
// `includebidderkeys` defaults to true, so in a contested auction every seat's top
// bid carries a targeting object of its own hb_pb_BIDDER keys. Testing only for a
// non-empty object therefore marks every seat as the winner, and which one wins
// becomes whichever the array happens to list first. Prebid's documented signal is
// the absence of a bidder suffix: "values without prefixes on the winning bids only".
//
// Shapes below are the deployed server's actual output for a two-seat auction on
// source/frontend-react/public/samples/yield-optimizer.json.
describe("resolveWinner across two seats", () => {
  const TWO_SEATS = {
    seatbid: [
      {
        seat: "artfhouse",
        bid: [
          {
            id: "b-artf",
            impid: "imp-1",
            price: 13.4,
            dealid: "deal-guaranteed-premium",
            ext: {
              prebid: {
                targeting: {
                  hb_pb: "13.40",
                  hb_bidder: "artfhouse",
                  hb_deal: "deal-guaranteed-premium",
                  hb_pb_artfhouse: "13.40",
                  hb_bidder_artfhouse: "artfhouse",
                },
              },
            },
          },
        ],
      },
      {
        seat: "amt",
        bid: [
          {
            id: "b-amt",
            impid: "imp-1",
            price: 12.5,
            // Loser: bidder-suffixed keys only.
            ext: { prebid: { targeting: { hb_pb_amt: "12.50", hb_bidder_amt: "amt" } } },
          },
        ],
      },
    ],
  };

  it("marks only the bid carrying the unprefixed keys", () => {
    const offers = normalizeOffers(TWO_SEATS);
    expect(offers.filter((o) => o.markedWinner === true)).toHaveLength(1);
    expect(resolveWinner(offers).clearedPrice).toBe(13.4);
    expect(resolveWinner(offers).dealId).toBe("deal-guaranteed-premium");
  });

  it("does not treat a loser's own bidder keys as a win", () => {
    const offers = normalizeOffers(TWO_SEATS);
    const loser = offers.find((o) => o.price === 12.5);
    expect(loser.markedWinner).toBe(false);
  });

  it("picks the winner by the marker, not by array order or by price", () => {
    // Winner listed second AND cheaper: neither position nor price may decide it.
    const reordered = { seatbid: [...TWO_SEATS.seatbid].reverse() };
    expect(resolveWinner(normalizeOffers(reordered)).clearedPrice).toBe(13.4);
  });

  it("reports no winner when no bid carries the unprefixed keys", () => {
    // What every scenario returned before the orchestrator requested targeting:
    // real bids, and nothing saying which one won.
    const noTargeting = {
      seatbid: [
        { seat: "artfhouse", bid: [{ id: "a", impid: "imp-1", price: 13.4 }] },
        { seat: "amt", bid: [{ id: "b", impid: "imp-1", price: 12.5 }] },
      ],
    };
    const offers = normalizeOffers(noTargeting);
    expect(offers).toHaveLength(2);
    expect(resolveWinner(offers)).toBeNull();
  });

  it("ignores an empty targeting object", () => {
    const offers = normalizeOffers({
      seatbid: [{ seat: "s", bid: [{ id: "a", impid: "i", price: 1, ext: { prebid: { targeting: {} } } }] }],
    });
    expect(resolveWinner(offers)).toBeNull();
  });
});


// A seatnonbid entry is SEAT-level. OpenRTB gives it an impid and a status code and
// nothing else; only this repo's demand endpoint adds a campaign to it. Dropping the
// seat rendered those rows "unknown" while the response named them plainly.
//
// Shape below is the deployed response for isv-ecosystem, verbatim.
describe("seat-level non-bids", () => {
  const LIVE = {
    cur: "USD",
    seatbid: [],
    ext: {
      seatnonbid: [
        {
          seat: "amt",
          nonbid: [
            {
              impid: "imp-1",
              statuscode: 0,
              ext: {
                artf: {
                  observed: {
                    returnedPrice: 3.25,
                    impFloor: 4.0,
                    currency: "USD",
                    source: "bidder response, via prebid debug httpcalls",
                  },
                },
              },
            },
            { impid: "imp-2", statuscode: 0 },
          ],
        },
      ],
    },
  };

  it("carries the seat that did not bid", () => {
    const offers = normalizeOffers(LIVE);
    expect(offers).toHaveLength(2);
    expect(offers.every((o) => o.seat === "amt")).toBe(true);
  });

  it("carries the observed price and floor when the orchestrator supplied them", () => {
    const [first, second] = normalizeOffers(LIVE);
    expect(first.observed).toEqual({
      returnedPrice: 3.25,
      impFloor: 4.0,
      currency: "USD",
      source: "bidder response, via prebid debug httpcalls",
    });
    // No observed block on the second: it must be null, not a partial object.
    expect(second.observed).toBeNull();
  });

  it("still reports no price on the row itself — a non-bid has none", () => {
    expect(normalizeOffers(LIVE).every((o) => o.price === null)).toBe(true);
  });

  it("rejects a half-populated observed block rather than rendering a blank half", () => {
    const partial = {
      ext: {
        seatnonbid: [
          {
            seat: "amt",
            nonbid: [{ impid: "i", statuscode: 0, ext: { artf: { observed: { impFloor: 3.0 } } } }],
          },
        ],
      },
    };
    expect(normalizeOffers(partial)[0].observed).toBeNull();
  });

  it("gives bids their seat too, and no observed block", () => {
    const offers = normalizeOffers({
      seatbid: [{ seat: "artfhouse", bid: [{ id: "b", impid: "i", price: 7.2 }] }],
    });
    expect(offers[0].seat).toBe("artfhouse");
    expect(offers[0].observed).toBeNull();
  });

  it("gives an endpoint exclusion no seat — it happened before any seat bid", () => {
    const offers = normalizeOffers({
      ext: { artf: { excluded: [{ campaignId: "c", campaignName: "C", exclusionReason: "not_targeted" }] } },
    });
    expect(offers[0].seat).toBeNull();
  });
});


// A campaign is considered once per impression, so the same campaign and deal
// legitimately appear more than once in ext.artf.excluded. The endpoint now stamps
// impId; the key must be unique either way, because two rows sharing a React key is
// a reconciliation fault rather than a cosmetic one.
//
// Live evidence: isv-ecosystem returned 30 entries for 16 campaigns across 2
// impressions, 14 of them byte-identical, all keyed identically.
describe("excluded rows across impressions", () => {
  const SAME_CAMPAIGN_TWICE = {
    ext: {
      artf: {
        excluded: [
          {
            campaignId: "camp-cedar",
            campaignName: "Cedar & Co Furnishings",
            dealId: null,
            exclusionReason: "no_deal_on_impression",
            impId: "imp-1",
          },
          {
            campaignId: "camp-cedar",
            campaignName: "Cedar & Co Furnishings",
            dealId: null,
            exclusionReason: "media_type_unsupported",
            impId: "imp-2",
          },
        ],
      },
    },
  };

  it("keeps both rows and gives them distinct keys", () => {
    const offers = normalizeOffers(SAME_CAMPAIGN_TWICE);
    expect(offers).toHaveLength(2);
    expect(new Set(offers.map((o) => o.key)).size).toBe(2);
  });

  it("carries the impression each exclusion is about", () => {
    const offers = normalizeOffers(SAME_CAMPAIGN_TWICE);
    expect(offers.map((o) => o.impId)).toEqual(["imp-1", "imp-2"]);
  });

  it("keeps each impression's own reason rather than collapsing them", () => {
    const offers = normalizeOffers(SAME_CAMPAIGN_TWICE);
    expect(offers[0].exclusionReason).toBe("no_deal_on_impression");
    expect(offers[1].exclusionReason).toBe("media_type_unsupported");
  });

  it("keys stay unique even when the endpoint sends no impId", () => {
    // The defect as it shipped: byte-identical entries. Uniqueness must not depend
    // on the very field whose absence caused the collision.
    const noImpId = {
      ext: {
        artf: {
          excluded: [
            { campaignId: "c", campaignName: "C", dealId: null, exclusionReason: "not_targeted" },
            { campaignId: "c", campaignName: "C", dealId: null, exclusionReason: "not_targeted" },
          ],
        },
      },
    };
    const offers = normalizeOffers(noImpId);
    expect(offers).toHaveLength(2);
    expect(new Set(offers.map((o) => o.key)).size).toBe(2);
  });
});
