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
