import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  classify,
  isDecision,
  isInfrastructureFailure,
  isNotAttempted,
  CATEGORY,
  OUTCOME,
  STATUS,
  EXCLUSION,
  HOOK_OUTCOME,
} from "./outcomeClassifier.js";

describe("classify — ARTF decisions", () => {
  it("treats deal suppression as a decision, with a reason", () => {
    const out = classify(undefined, EXCLUSION.DEAL_SUPPRESSED);
    expect(out.category).toBe(CATEGORY.ARTF_DECISION);
    expect(out.outcome).toBe(OUTCOME.NOT_OFFERED);
    expect(out.reason).toMatch(/suppressed/i);
    expect(isDecision(out)).toBe(true);
  });

  it("treats a below-floor exclusion as a decision", () => {
    const out = classify(undefined, EXCLUSION.BELOW_FLOOR);
    expect(out.category).toBe(CATEGORY.ARTF_DECISION);
    expect(out.outcome).toBe(OUTCOME.REJECTED_BELOW_FLOOR);
  });

  it("treats status 301 with no ARTF reason as a decision", () => {
    const out = classify(STATUS.BELOW_FLOOR, undefined);
    expect(out.category).toBe(CATEGORY.ARTF_DECISION);
    expect(out.outcome).toBe(OUTCOME.REJECTED_BELOW_FLOOR);
  });
});

describe("classify — infrastructure failure", () => {
  it("treats status 101 as infrastructure failure, not a decision", () => {
    const out = classify(STATUS.TIMEOUT, undefined);
    expect(out.category).toBe(CATEGORY.INFRASTRUCTURE_FAILURE);
    expect(isDecision(out)).toBe(false);
    expect(isInfrastructureFailure(out)).toBe(true);
  });

  it("treats a hook transport failure as infrastructure failure", () => {
    const out = classify(undefined, HOOK_OUTCOME.TRANSPORT_FAILURE);
    expect(out.category).toBe(CATEGORY.INFRASTRUCTURE_FAILURE);
  });
});

describe("classify — not attempted", () => {
  it("treats a skipped call as neither a decision nor a failure", () => {
    const out = classify(undefined, HOOK_OUTCOME.SKIPPED_INSUFFICIENT_BUDGET);
    expect(out.category).toBe(CATEGORY.NOT_ATTEMPTED);
    expect(isDecision(out)).toBe(false);
    expect(isInfrastructureFailure(out)).toBe(false);
    expect(isNotAttempted(out)).toBe(true);
    expect(out.reason).toMatch(/not attempted/i);
  });

  it("gives an unrecognised status code a legible reason rather than blank", () => {
    const out = classify(999, undefined);
    expect(out.category).toBe(CATEGORY.NOT_ATTEMPTED);
    expect(out.reason).toMatch(/999/);
  });

  it("says so when no reason was reported at all", () => {
    const out = classify(undefined, undefined);
    expect(out.reason).toMatch(/no reason reported/i);
  });

  // Regression, pinned deterministically rather than left to a seed.
  //
  // `exclusionReason` arrives in the demand endpoint's response, so it is a string this
  // surface does not control. The reason lookup was an object literal, which inherits
  // from Object.prototype -- so a reason of "valueOf" resolved to a FUNCTION, passed the
  // `!= null` guard, and was reported as an ARTF decision with a function where the
  // human-readable text belongs.
  //
  // Found by the totality property below drawing "valueOf". Pinned here because a
  // property test that catches something on one seed in ten is not a regression test.
  it.each(["valueOf", "toString", "constructor", "hasOwnProperty", "__proto__", "isPrototypeOf"])(
    "treats the prototype key %s as an unknown reason, not a decision",
    (protoKey) => {
      const out = classify(undefined, protoKey);
      expect(typeof out.reason).toBe("string");
      expect(out.category).toBe(CATEGORY.NOT_ATTEMPTED);
      expect(isDecision(out)).toBe(false);
    },
  );
});

describe("classify — offered and won", () => {
  it("marks the winner as won", () => {
    const out = classify(undefined, undefined, { offered: true, markedWinner: true });
    expect(out.outcome).toBe(OUTCOME.WON);
    expect(out.category).toBe(CATEGORY.WON);
  });

  it("marks an offer that did not win as lost on price", () => {
    const out = classify(undefined, undefined, { offered: true, markedWinner: false });
    expect(out.outcome).toBe(OUTCOME.LOST_ON_PRICE);
  });

  it("does NOT call a price loss an ARTF decision", () => {
    // No container excluded it — the auction decided. If this were ArtfDecision,
    // isDecision would answer FR-29's question wrongly for every ordinary loss.
    const out = classify(undefined, undefined, { offered: true, markedWinner: false });
    expect(out.category).toBe(CATEGORY.AUCTION_OUTCOME);
    expect(isDecision(out)).toBe(false);
  });

  it("does not call a win an ARTF decision either", () => {
    const out = classify(undefined, undefined, { offered: true, markedWinner: true });
    expect(isDecision(out)).toBe(false);
  });
});

/* --------------------------------------------- property tests, FR-29, BR-9 */

const arbCode = fc.option(fc.oneof(fc.constantFrom(101, 301), fc.integer({ min: -5, max: 999 })), {
  nil: undefined,
});

const arbReason = fc.option(
  fc.oneof(
    fc.constantFrom(...Object.values(EXCLUSION)),
    fc.constantFrom(...Object.values(HOOK_OUTCOME)),
    fc.string({ maxLength: 8 }),
  ),
  { nil: undefined },
);

const arbOpts = fc.record({ offered: fc.boolean(), markedWinner: fc.boolean() });

const ALL_CATEGORIES = Object.values(CATEGORY);
const ALL_OUTCOMES = Object.values(OUTCOME);

describe("properties", () => {
  it("totality: every input yields a defined outcome in exactly one known category", () => {
    fc.assert(
      fc.property(arbCode, arbReason, arbOpts, (code, reason, opts) => {
        const out = classify(code, reason, opts);
        return (
          out != null &&
          ALL_OUTCOMES.includes(out.outcome) &&
          ALL_CATEGORIES.includes(out.category)
        );
      }),
    );
  });

  it("no input renders blank: an unwon outcome always carries a reason", () => {
    fc.assert(
      fc.property(arbCode, arbReason, (code, reason) => {
        const out = classify(code, reason, { offered: false, markedWinner: false });
        return typeof out.reason === "string" && out.reason.length > 0;
      }),
    );
  });

  it("the three category predicates are mutually exclusive", () => {
    fc.assert(
      fc.property(arbCode, arbReason, arbOpts, (code, reason, opts) => {
        const out = classify(code, reason, opts);
        const flags = [isDecision(out), isInfrastructureFailure(out), isNotAttempted(out)];
        return flags.filter(Boolean).length <= 1;
      }),
    );
  });

  it("determinism: the same input classifies the same way every time", () => {
    fc.assert(
      fc.property(arbCode, arbReason, arbOpts, (code, reason, opts) => {
        return (
          JSON.stringify(classify(code, reason, opts)) ===
          JSON.stringify(classify(code, reason, opts))
        );
      }),
    );
  });
});


// Status 0 is Prebid's NO_BID: the seat WAS called and returned nothing usable.
//
// Before this, 0 fell to the catch-all and rendered "No offer — unrecognised status
// 0" in the NOT_ATTEMPTED category. Both halves were wrong: 0 is the most ordinary
// non-bid there is, and calling an attempted seat "not attempted" states the
// opposite of what happened. Live evidence: every amt non-bid on the deployed stack
// arrives as statuscode 0.
describe("classify NO_BID (status 0)", () => {
  it("is an auction outcome, not an unattempted one", () => {
    const r = classify(0, null);
    expect(r.outcome).toBe(OUTCOME.NO_BID);
    expect(r.category).toBe(CATEGORY.AUCTION_OUTCOME);
    expect(r.category).not.toBe(CATEGORY.NOT_ATTEMPTED);
  });

  it("no longer reports it as an unrecognised status", () => {
    expect(classify(0, null).reason).not.toMatch(/unrecognised/i);
    expect(classify(0, null).reason).toMatch(/no usable bid/i);
  });

  it("states the price it returned and the floor it faced when both are observed", () => {
    const r = classify(0, null, {
      observed: { returnedPrice: 2.75, impFloor: 3.0, currency: "USD" },
    });
    expect(r.reason).toMatch(/2\.75/);
    expect(r.reason).toMatch(/3\.00/);
    expect(r.reason).toMatch(/USD/);
    expect(r.reason).toMatch(/floor/i);
  });

  // The evidence is shown; the verdict is not drawn. Prebid reported NO_BID and did
  // not say the floor was the cause, so the UI must not say so either.
  it("does not claim the floor rejected the bid", () => {
    const r = classify(0, null, {
      observed: { returnedPrice: 2.75, impFloor: 3.0, currency: "USD" },
    });
    expect(r.outcome).not.toBe(OUTCOME.REJECTED_BELOW_FLOOR);
    expect(r.category).not.toBe(CATEGORY.ARTF_DECISION);
    expect(r.reason).not.toMatch(/rejected/i);
    expect(r.reason).not.toMatch(/below the resolved floor/i);
  });

  it("falls back to the plain statement when the evidence is incomplete", () => {
    for (const observed of [
      null,
      {},
      { returnedPrice: 2.75 },
      { impFloor: 3.0 },
      { returnedPrice: "2.75", impFloor: 3.0 },
    ]) {
      expect(classify(0, null, { observed }).reason).toBe(
        "The seat was called and returned no usable bid",
      );
    }
  });

  it("still reports an above-floor price, since NO_BID had some other cause", () => {
    // 5.00 over a 3.00 floor: the floor was not the obstacle, and the two numbers
    // shown make that visible rather than implying a floor rejection.
    const r = classify(0, null, { observed: { returnedPrice: 5.0, impFloor: 3.0, currency: "USD" } });
    expect(r.reason).toMatch(/5\.00/);
    expect(r.reason).toMatch(/3\.00/);
  });

  it("an ARTF exclusion reason still outranks the status code", () => {
    const r = classify(0, "deal_suppressed", {
      observed: { returnedPrice: 2.75, impFloor: 3.0 },
    });
    expect(r.category).toBe(CATEGORY.ARTF_DECISION);
    expect(r.reason).toMatch(/suppressed/i);
  });

  it("a winner is a winner even if a status code came along", () => {
    expect(classify(0, null, { markedWinner: true }).category).toBe(CATEGORY.WON);
  });
});


// media_type_unsupported existed in the endpoint's enum and not in this module's
// map, so 12 of 30 excluded campaigns on the isv-ecosystem scenario rendered
// "No offer — no reason reported" for a reason that HAD been reported. The parity
// guard lives in source/tests/test_exclusion_reason_parity.py; these pin the
// rendering.
describe("classify media_type_unsupported", () => {
  it("is an ARTF decision with its real wording", () => {
    const r = classify(null, "media_type_unsupported");
    expect(r.category).toBe(CATEGORY.ARTF_DECISION);
    expect(r.outcome).toBe(OUTCOME.NOT_OFFERED);
    expect(r.reason).toMatch(/no slot this campaign's creative could fill/i);
  });

  it("no longer claims no reason was reported", () => {
    expect(classify(null, "media_type_unsupported").reason).not.toMatch(/no reason reported/i);
  });

  it("is a decision, not a floor rejection — nothing was priced", () => {
    const r = classify(null, "media_type_unsupported");
    expect(r.outcome).not.toBe(OUTCOME.REJECTED_BELOW_FLOOR);
  });

  it("every reason the endpoint can emit renders as itself", () => {
    // The whole closed set, so a future addition on either side shows up here too.
    for (const reason of [
      "deal_suppressed",
      "below_floor",
      "not_targeted",
      "no_deal_on_impression",
      "media_type_unsupported",
    ]) {
      const r = classify(null, reason);
      expect(r.category).toBe(CATEGORY.ARTF_DECISION);
      expect(r.reason).not.toMatch(/no reason reported/i);
      expect(typeof r.reason).toBe("string");
      expect(r.reason.length).toBeGreaterThan(0);
    }
  });
});
