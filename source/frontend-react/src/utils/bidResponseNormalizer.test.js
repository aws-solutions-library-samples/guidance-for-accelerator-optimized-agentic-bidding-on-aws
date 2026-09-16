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
