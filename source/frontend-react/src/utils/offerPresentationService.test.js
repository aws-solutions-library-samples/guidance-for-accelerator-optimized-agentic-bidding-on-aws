import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  buildOfferViewModel,
  selectNotice,
  NOTICE,
} from "./offerPresentationService.js";
import { capturedBidResponse, FIXTURE_MARKER } from "./bidResponseFixture.js";
import { CATEGORY } from "./outcomeClassifier.js";

/* ------------------------------------------------------ FR-31, the notice */

describe("notice selection comes from the data, not a prop", () => {
  it("labels the fixture illustrative", () => {
    expect(selectNotice(capturedBidResponse)).toBe(NOTICE.ILLUSTRATIVE);
    expect(buildOfferViewModel(capturedBidResponse).notice).toBe(NOTICE.ILLUSTRATIVE);
  });

  it("labels a real response declared-inventory", () => {
    const real = { ...capturedBidResponse };
    delete real[FIXTURE_MARKER];
    expect(selectNotice(real)).toBe(NOTICE.DECLARED_INVENTORY);
    expect(buildOfferViewModel(real).notice).toBe(NOTICE.DECLARED_INVENTORY);
  });

  it("a fixture can never render under the real-auction notice", () => {
    // There is no argument, prop or option that flips this.
    const vm = buildOfferViewModel(capturedBidResponse);
    expect(vm.notice).not.toBe(NOTICE.DECLARED_INVENTORY);
    expect(vm.noticeText).toMatch(/illustrative/i);
  });

  it("treats an absent response as real rather than silently illustrative", () => {
    // Failing the other way would let a broken fetch present as a labelled demo.
    expect(selectNotice(undefined)).toBe(NOTICE.DECLARED_INVENTORY);
  });
});

/* ---------------------------------------------------------- the view model */

describe("buildOfferViewModel", () => {
  it("carries every candidate, offering and not", () => {
    // Two offers, one Prebid non-bid, two endpoint exclusions.
    const vm = buildOfferViewModel(capturedBidResponse);
    expect(vm.offers).toHaveLength(5);
  });

  it("gives every non-offering candidate a reason", () => {
    const vm = buildOfferViewModel(capturedBidResponse);
    for (const offer of vm.offers.filter((o) => !o.offered)) {
      expect(typeof offer.outcome.reason).toBe("string");
      expect(offer.outcome.reason.length).toBeGreaterThan(0);
    }
  });

  it("distinguishes an ARTF decision from an infrastructure failure", () => {
    const vm = buildOfferViewModel({
      ext: {
        seatnonbid: [
          {
            seat: "artfhouse",
            nonbid: [
              { impid: "i1", ext: { artf: { exclusionReason: "deal_suppressed" } } },
              { impid: "i1", statuscode: 101 },
            ],
          },
        ],
      },
    });
    const categories = vm.offers.map((o) => o.outcome.category);
    expect(categories).toContain(CATEGORY.ARTF_DECISION);
    expect(categories).toContain(CATEGORY.INFRASTRUCTURE_FAILURE);
  });

  it("resolves the winner as deal plus campaign with price as detail", () => {
    const vm = buildOfferViewModel(capturedBidResponse);
    expect(vm.winner.dealId).toBe("deal-home-premium");
    expect(vm.winner.clearedPrice).toBe(6.35);
  });

  it("returns a null winner for a response with no marked winner", () => {
    const vm = buildOfferViewModel({ seatbid: [{ seat: "s", bid: [{ impid: "i", price: 2 }] }] });
    expect(vm.winner).toBeNull();
  });
});

/* ----------------------------------------------- property tests, NFR-5 */

const arbResponse = fc.record({
  cur: fc.constant("USD"),
  seatbid: fc.array(
    fc.record({
      seat: fc.constant("artfhouse"),
      bid: fc.array(
        fc.record({
          id: fc.string({ minLength: 1, maxLength: 5 }),
          impid: fc.constant("imp-1"),
          price: fc.double({ min: 0, max: 20, noNaN: true }),
        }),
        { maxLength: 4 },
      ),
    }),
    { maxLength: 2 },
  ),
  ext: fc.record({
    seatnonbid: fc.array(
      fc.record({
        seat: fc.constant("artfhouse"),
        nonbid: fc.array(
          fc.record({
            impid: fc.constant("imp-1"),
            statuscode: fc.option(fc.constantFrom(101, 301), { nil: undefined }),
          }),
          { maxLength: 4 },
        ),
      }),
      { maxLength: 2 },
    ),
  }),
});

describe("properties", () => {
  it("render idempotence: the same response yields the same view model", () => {
    fc.assert(
      fc.property(arbResponse, (resp) => {
        return (
          JSON.stringify(buildOfferViewModel(resp)) === JSON.stringify(buildOfferViewModel(resp))
        );
      }),
    );
  });

  it("every offer carries a defined outcome category", () => {
    const known = Object.values(CATEGORY);
    fc.assert(
      fc.property(arbResponse, (resp) => {
        return buildOfferViewModel(resp).offers.every((o) => known.includes(o.outcome.category));
      }),
    );
  });

  it("a generated response is never labelled illustrative", () => {
    fc.assert(
      fc.property(arbResponse, (resp) => {
        return buildOfferViewModel(resp).notice === NOTICE.DECLARED_INVENTORY;
      }),
    );
  });
});


// Grouping: bids earn their own row, everything else is folded.
//
// Live measurement across the six scenarios: 18 real bids, 3 sell-side decisions,
// 96 campaigns that were never candidates for the impression. Four of the six had
// zero decisions, so their whole non-bid section was catalog non-applicability.
describe("buildOfferViewModel grouping", () => {
  const bid = (id, price, extra = {}) => ({
    id,
    impid: "imp-1",
    price,
    ext: { prebid: { targeting: extra.winner ? { hb_bidder: "s", hb_pb: String(price) } : {} } },
    ...extra.bid,
  });

  const RESPONSE = {
    cur: "USD",
    seatbid: [
      {
        seat: "artfhouse",
        bid: [bid("w", 4.1, { winner: true }), bid("l", 3.05)],
      },
    ],
    ext: {
      seatnonbid: [{ seat: "amt", nonbid: [{ impid: "imp-1", statuscode: 0 }] }],
      artf: {
        excluded: [
          { campaignId: "c1", campaignName: "C1", exclusionReason: "below_floor", impId: "imp-1" },
          { campaignId: "c2", campaignName: "C2", exclusionReason: "deal_suppressed", impId: "imp-1" },
          { campaignId: "c3", campaignName: "C3", exclusionReason: "no_deal_on_impression", impId: "imp-1" },
          { campaignId: "c4", campaignName: "C4", exclusionReason: "no_deal_on_impression", impId: "imp-1" },
          { campaignId: "c5", campaignName: "C5", exclusionReason: "media_type_unsupported", impId: "imp-1" },
          { campaignId: "c6", campaignName: "C6", exclusionReason: "not_targeted", impId: "imp-1" },
        ],
      },
    },
  };

  const groupById = (vm, id) => vm.groups.find((g) => g.id === id);

  it("puts only the bids on their own rows", () => {
    const vm = buildOfferViewModel(RESPONSE);
    expect(vm.bidRows).toHaveLength(2);
    expect(vm.bidRows.every((o) => o.offered === true)).toBe(true);
  });

  it("partitions offers exactly: nothing lost, nothing duplicated", () => {
    const vm = buildOfferViewModel(RESPONSE);
    const grouped = vm.groups.flatMap((g) => g.rows);
    expect(vm.bidRows.length + grouped.length).toBe(vm.offers.length);
    const keys = [...vm.bidRows, ...grouped].map((o) => o.key);
    expect(new Set(keys).size).toBe(vm.offers.length);
    expect(new Set(keys)).toEqual(new Set(vm.offers.map((o) => o.key)));
  });

  it("groups a decision separately from a campaign that was never a candidate", () => {
    const vm = buildOfferViewModel(RESPONSE);
    expect(groupById(vm, "decision").count).toBe(2);
    expect(groupById(vm, "ineligible").count).toBe(4);
  });

  it("groups a seat that returned no bid on its own", () => {
    const vm = buildOfferViewModel(RESPONSE);
    expect(groupById(vm, "no_bid").count).toBe(1);
    expect(groupById(vm, "no_bid").rows[0].seat).toBe("amt");
  });

  // A seat non-bid carries no exclusion reason, so a reason-keyed breakdown made
  // this group read "3 no reason reported" about seats whose reason is on their
  // own rows — the same misleading string this column keeps having to remove.
  it("breaks the no-bid group down by seat, not by a reason it does not have", () => {
    const twoSeats = {
      ext: {
        seatnonbid: [
          { seat: "amt", nonbid: [{ impid: "i1", statuscode: 0 }, { impid: "i2", statuscode: 0 }] },
          { seat: "artfhouse", nonbid: [{ impid: "i2", statuscode: 0 }] },
        ],
      },
    };
    const group = groupById(buildOfferViewModel(twoSeats), "no_bid");
    expect(group.breakdown).toEqual([
      { label: "amt", count: 2 },
      { label: "artfhouse", count: 1 },
    ]);
    expect(group.breakdown.map((b) => b.label)).not.toContain("no reason reported");
  });

  it("summarises each group with a count and a reason breakdown", () => {
    const vm = buildOfferViewModel(RESPONSE);
    expect(groupById(vm, "ineligible").label).toBe("4 campaigns not eligible for this impression");
    expect(groupById(vm, "ineligible").breakdown).toEqual([
      { label: "no deal on the impression", count: 2 },
      { label: "creative format", count: 1 },
      { label: "targeting did not match", count: 1 },
    ]);
  });

  it("uses the singular where there is one", () => {
    const one = {
      ext: {
        artf: { excluded: [{ campaignId: "c", campaignName: "C", exclusionReason: "below_floor" }] },
      },
    };
    expect(groupById(buildOfferViewModel(one), "decision").label).toBe(
      "1 campaign stopped by a sell-side decision",
    );
  });

  it("omits a group with nothing in it rather than rendering an empty one", () => {
    const onlyBids = {
      seatbid: [{ seat: "s", bid: [bid("a", 2.0, { winner: true })] }],
    };
    expect(buildOfferViewModel(onlyBids).groups).toEqual([]);
  });

  // A fault folded behind a summary line is a fault nobody sees, which is the
  // failure mode this column exists to prevent.
  it("never groups an infrastructure failure", () => {
    const timedOut = {
      ext: {
        seatnonbid: [{ seat: "amt", nonbid: [{ impid: "i", statuscode: 101 }] }],
      },
    };
    const vm = buildOfferViewModel(timedOut);
    expect(vm.groups).toEqual([]);
    expect(vm.bidRows).toHaveLength(1);
    expect(vm.bidRows[0].outcome.category).toBe("InfrastructureFailure");
  });

  it("keeps group order and row order stable for the same response", () => {
    const a = buildOfferViewModel(RESPONSE);
    const b = buildOfferViewModel(RESPONSE);
    expect(a.groups.map((g) => g.id)).toEqual(["decision", "ineligible", "no_bid"]);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("leaves offers as the complete set in response order", () => {
    const vm = buildOfferViewModel(RESPONSE);
    expect(vm.offers).toHaveLength(9);
  });
});
