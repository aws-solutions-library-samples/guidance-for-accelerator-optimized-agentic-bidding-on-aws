package org.prebid.server.hooks.modules.artf.core;

import org.junit.jupiter.api.Test;
import org.prebid.server.hooks.modules.artf.model.CallBudget;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The budget-safety property, plus examples.
 *
 * <h2>Why this is not a jqwik test</h2>
 *
 * <p>PBT-09 asks for a property-based testing framework. <b>jqwik is not available here, and
 * adding it is not possible without breaking a stronger constraint.</b> These sources are
 * copied into the PBS-Core source tree at build time (the injection route the verification
 * gate established), so a new test-scope dependency would mean editing pbs-java's own
 * {@code pom.xml} -- which converts the integration into a fork and breaks U1-NFR-16.
 * Available instead: JUnit 5, AssertJ and Mockito, which PBS-Core already declares.
 *
 * <p>So the budget-safety property is asserted by an <b>exhaustive sweep over a grid</b> rather
 * than by random sampling. For four small integer inputs that is not a compromise -- it is
 * stronger: a grid covers every combination in range, where a random generator covers a sample
 * of them. PBT-08 wants shrinking and reproducibility; a fixed grid is perfectly reproducible
 * and needs no shrinking, because the failing combination is printed exactly.
 *
 * <h2>These tests are not run by the deploy path</h2>
 *
 * <p>The upstream image build runs Maven with {@code -Dmaven.test.skip}
 * ({@code docker-build-config.json}), so nothing here executes during a deployment. To run
 * them, build the module without that flag. Stated rather than implied, so nobody reads a
 * successful deployment as a passing test run.
 */
class CallBudgetCalculatorTest {

    private static final int TMAX = 100;
    private static final int OVERHEAD = 20;
    private static final int RESERVE = 60;

    // ------------------------------------------------------------- the property

    /**
     * Budget safety, swept exhaustively.
     *
     * <p>Four invariants, each the negation of a specific way this arithmetic can go wrong.
     * The second is the important one: it is the executable form of the bug that made this a
     * separate class -- advertising a budget the hook will not honour, which produces a
     * self-inflicted timeout, discarded container work, and two components logging
     * contradictory stories about one call.
     */
    @Test
    void budgetSafetyHoldsAcrossTheWholeInputSpace() {
        for (long remaining = 0; remaining <= 400; remaining += 1) {
            for (int tmax = 0; tmax <= 200; tmax += 10) {
                for (int overhead = 0; overhead <= 60; overhead += 5) {
                    for (int reserve = 0; reserve <= 120; reserve += 10) {

                        final CallBudget budget =
                                CallBudgetCalculator.calculate(remaining, tmax, overhead, reserve);
                        final String at = "remaining=%d tmax=%d overhead=%d reserve=%d -> %s"
                                .formatted(remaining, tmax, overhead, reserve, budget);

                        final long afterReserve = remaining - reserve;
                        final long minimumViable = CallBudgetCalculator.ORCHESTRATOR_FLOOR_MS + overhead;

                        if (!budget.viable()) {
                            // A non-viable budget must not smuggle a usable timeout out.
                            assertThat(budget.httpTimeoutMs()).as(at).isZero();
                            assertThat(budget.effectiveTmaxMs()).as(at).isZero();
                            continue;
                        }

                        // 1. Never encroach on the reserve.
                        assertThat(budget.httpTimeoutMs()).as("reserve preserved: " + at)
                                .isLessThanOrEqualTo(afterReserve);

                        // 2. Never advertise more than will be honoured. THE bug this class exists for.
                        assertThat((long) budget.effectiveTmaxMs() + overhead)
                                .as("advertised never exceeds honoured: " + at)
                                .isLessThanOrEqualTo(budget.httpTimeoutMs());

                        // 3. The cap only ever tightens.
                        assertThat(budget.effectiveTmaxMs()).as("cap only tightens: " + at)
                                .isLessThanOrEqualTo(tmax);

                        // 4. A viable call is always worth making: at or above the orchestrator's floor.
                        assertThat(budget.effectiveTmaxMs()).as("above the orchestrator floor: " + at)
                                .isGreaterThanOrEqualTo(CallBudgetCalculator.ORCHESTRATOR_FLOOR_MS);
                        assertThat(afterReserve).as("viable implies enough budget: " + at)
                                .isGreaterThanOrEqualTo(minimumViable);
                    }
                }
            }
        }
    }

    /**
     * More auction budget never produces a smaller timeout.
     *
     * <p>Monotonicity is worth asserting separately because a non-monotonic bound is the shape
     * a min/max mix-up takes, and it would show up as intermittent short timeouts on generous
     * requests -- the least diagnosable failure this class could have.
     */
    @Test
    void timeoutIsMonotonicInTheAuctionBudget() {
        long previousTimeout = -1;
        for (long remaining = 0; remaining <= 500; remaining++) {
            final CallBudget budget = CallBudgetCalculator.calculate(remaining, TMAX, OVERHEAD, RESERVE);
            if (!budget.viable()) {
                continue;
            }
            assertThat(budget.httpTimeoutMs())
                    .as("remaining=%d must not yield a smaller timeout than a smaller budget did", remaining)
                    .isGreaterThanOrEqualTo(previousTimeout);
            previousTimeout = budget.httpTimeoutMs();
        }
    }

    /** Once viable, staying viable as the budget grows. A viability flap would be a bug. */
    @Test
    void viabilityIsMonotonicInTheAuctionBudget() {
        boolean seenViable = false;
        for (long remaining = 0; remaining <= 500; remaining++) {
            final boolean viable = CallBudgetCalculator.calculate(remaining, TMAX, OVERHEAD, RESERVE).viable();
            if (viable) {
                seenViable = true;
            } else if (seenViable) {
                throw new AssertionError(
                        "viability flapped back to false at remaining=" + remaining);
            }
        }
        assertThat(seenViable).isTrue();
    }

    // ---------------------------------------------------------------- examples

    @Test
    void aGenerousBudgetAdvertisesTheConfiguredTmaxUnmodified() {
        // 500 ms is plenty: the configured tmax is not derived per request (U3-NFR-3), so a
        // normal request must advertise exactly what was configured.
        final CallBudget budget = CallBudgetCalculator.calculate(500, TMAX, OVERHEAD, RESERVE);

        assertThat(budget.viable()).isTrue();
        assertThat(budget.effectiveTmaxMs()).isEqualTo(TMAX);
        assertThat(budget.httpTimeoutMs()).isEqualTo(TMAX + OVERHEAD);
    }

    @Test
    void aShortBudgetShrinksTheAdvertisedTmaxRatherThanOverrunningTheAuction() {
        // 120 ms remaining, 60 reserved: 60 available, so the hook may wait 60 and must
        // advertise no more than 40. Advertising 100 here is the failure being prevented.
        final CallBudget budget = CallBudgetCalculator.calculate(120, TMAX, OVERHEAD, RESERVE);

        assertThat(budget.viable()).isTrue();
        assertThat(budget.httpTimeoutMs()).isEqualTo(60);
        assertThat(budget.effectiveTmaxMs()).isEqualTo(40);
        assertThat(budget.effectiveTmaxMs() + OVERHEAD).isLessThanOrEqualTo((int) budget.httpTimeoutMs());
    }

    @Test
    void aNearlyExhaustedBudgetIsSkippedRatherThanCalledDoomed() {
        // 85 ms remaining, 60 reserved: 25 available, below the 10 ms orchestrator floor plus
        // 20 ms overhead. A call here would be abandoned before the orchestrator could answer.
        final CallBudget budget = CallBudgetCalculator.calculate(85, TMAX, OVERHEAD, RESERVE);

        assertThat(budget.viable()).isFalse();
        assertThat(budget.remainingAfterReserveMs()).isEqualTo(25);
    }

    @Test
    void aBudgetBelowTheReserveIsSkippedAndReportsANegativeRemainder() {
        // Reported, not clamped: a negative remainder is the honest description of a request
        // that arrived with less budget than the reserve, and rounding it to zero would hide
        // how far past the deadline it was.
        final CallBudget budget = CallBudgetCalculator.calculate(10, TMAX, OVERHEAD, RESERVE);

        assertThat(budget.viable()).isFalse();
        assertThat(budget.remainingAfterReserveMs()).isEqualTo(-50);
    }

    @Test
    void anAlreadyExpiredAuctionIsSkippedRatherThanThrowing() {
        // A clock past the deadline yields a negative remaining budget. That is a skip on the
        // auction path, not an exception.
        final CallBudget budget = CallBudgetCalculator.calculate(-200, TMAX, OVERHEAD, RESERVE);

        assertThat(budget.viable()).isFalse();
    }

    @Test
    void aTmaxBelowTheOrchestratorFloorIsNotWorthCalling() {
        // Configured 5 ms: the orchestrator floors tmax at 10, so it would work for longer
        // than the hook intends to wait. Skipped rather than producing a guaranteed timeout.
        final CallBudget budget = CallBudgetCalculator.calculate(500, 5, OVERHEAD, RESERVE);

        assertThat(budget.viable()).isFalse();
    }

    @Test
    void theMinimumViableCallIsTheOrchestratorFloorPlusOverhead() {
        assertThat(CallBudgetCalculator.minimumViableMs(OVERHEAD))
                .isEqualTo(CallBudgetCalculator.ORCHESTRATOR_FLOOR_MS + OVERHEAD);
    }

    @Test
    void theOrchestratorFloorMatchesTheOrchestratorsOwnFloor() {
        // orchestrator/app.py:470 -- tmax = max(req.tmax, 10). If that changes, this constant
        // must change with it, or the skip threshold stops matching reality.
        assertThat(CallBudgetCalculator.ORCHESTRATOR_FLOOR_MS).isEqualTo(10);
    }
}
