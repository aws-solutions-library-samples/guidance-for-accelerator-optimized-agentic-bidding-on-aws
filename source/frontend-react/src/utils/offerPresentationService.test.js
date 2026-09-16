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
