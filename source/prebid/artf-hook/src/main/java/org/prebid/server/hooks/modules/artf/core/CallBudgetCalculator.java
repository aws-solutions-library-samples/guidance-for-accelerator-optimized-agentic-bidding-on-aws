package org.prebid.server.hooks.modules.artf.core;

import org.prebid.server.hooks.modules.artf.model.CallBudget;

/**
 * The budget arithmetic. PURE: four numbers in, a {@link CallBudget} out.
 *
 * <p>There are three nested budgets, and this class exists to keep them consistent:
 *
 * <ol>
 *   <li><b>Auction budget</b> -- Prebid's overall deadline for the request.</li>
 *   <li><b>ARTF {@code tmax}</b> -- what the orchestrator gets for its container fan-out.
 *       It uses the WHOLE value ({@code orchestrator/app.py:471}) and floors it at 10 ms
 *       ({@code app.py:470}).</li>
 *   <li><b>HTTP timeout</b> -- what this hook will actually wait.</li>
 * </ol>
 *
 * <p><b>Why this is its own class.</b> It is the part of the unit most likely to be wrong --
 * three budgets, a cap and a floor -- and the only part expressible as a pure function of
 * four numbers. Extracted, it is testable without a network, a clock or a running Prebid
 * Server. Left inside the client it could only be exercised by making calls, which is the
 * wrong way to test arithmetic.
 *
 * <p><b>The failure it eliminates.</b> Without the cap on the advertised {@code tmax}, a
 * short auction budget makes the hook promise the orchestrator 100 ms while intending to
 * wait 60. The orchestrator and its containers are still working when the hook hangs up:
 * container work is discarded, the timeout was self-inflicted, and two components log
 * contradictory stories about one call. {@link #INVARIANT_ADVERTISED_NEVER_EXCEEDS_HONOURED}
 * is that bug's negation, and the budget-safety property test is its executable form.
 *
 * <p>This class must not read configuration, read a clock, or make a call. It takes numbers
 * and returns numbers.
 */
public final class CallBudgetCalculator {

    /**
     * The orchestrator floors {@code tmax} at 10 ms: {@code tmax = max(req.tmax, 10)}
     * ({@code orchestrator/app.py:470, 934}). Below that plus transport there is nothing to
     * gain by calling, so this is the skip threshold's basis rather than a guess.
     */
    public static final int ORCHESTRATOR_FLOOR_MS = 10;

    static final String INVARIANT_ADVERTISED_NEVER_EXCEEDS_HONOURED =
            "effectiveTmax + overhead <= httpTimeout";

    private CallBudgetCalculator() {
    }

    /**
     * Compute the budget for one call.
     *
     * @param remainingAuctionBudgetMs time left before Prebid's own deadline
     * @param configuredTmaxMs         the configured ARTF {@code tmax}; 100 ms by default,
     *                                 and NOT derived per request -- the cap below only
     *                                 ever tightens it (U3-NFR-3)
     * @param overheadMs               transport allowance: connection reuse, serialisation,
     *                                 and the network hop. A configurable default chosen to
     *                                 be safe, NOT a measured figure -- to be validated
     *                                 against the orchestrator's recorded {@code latency_ms}
     * @param reserveMs                fixed milliseconds set aside for bidder fan-out and
     *                                 auction resolution. Fixed, not a percentage, so one
     *                                 log line tells you the headroom (D3)
     */
    public static CallBudget calculate(long remainingAuctionBudgetMs,
                                       int configuredTmaxMs,
                                       int overheadMs,
                                       int reserveMs) {

        // Negative inputs are clamped rather than thrown on. A clock that has already
        // passed the deadline yields a negative remaining budget, and that is a skip, not
        // an exception on the auction path.
        final long remaining = Math.max(remainingAuctionBudgetMs, 0L);
        final int configured = Math.max(configuredTmaxMs, 0);
        final int overhead = Math.max(overheadMs, 0);
        final int reserve = Math.max(reserveMs, 0);

        final long afterReserve = remaining - reserve;
        final long minimumViable = (long) ORCHESTRATOR_FLOOR_MS + overhead;

        if (afterReserve < minimumViable) {
            return CallBudget.notViable(afterReserve);
        }

        // Bounded on BOTH sides (U3-NFR-1). The first term is what we want; the second is
        // what the auction can afford.
        final long httpTimeout = Math.min((long) configured + overhead, afterReserve);

        // The cap (U3-NFR-2). Never advertise more than will be honoured.
        final long tmaxCeiling = httpTimeout - overhead;
        final int effectiveTmax = (int) Math.min(configured, Math.max(tmaxCeiling, 0L));

        // If the cap has driven the advertised budget below the orchestrator's own floor,
        // the call is not worth making. Checked after the cap, not before: it is the CAPPED
        // value that the orchestrator would receive.
        if (effectiveTmax < ORCHESTRATOR_FLOOR_MS) {
            return CallBudget.notViable(afterReserve);
        }

        return new CallBudget(httpTimeout, effectiveTmax, true, afterReserve);
    }

    /**
     * The smallest budget worth calling with, for reporting a skip with the numbers that
     * caused it.
     */
    public static long minimumViableMs(int overheadMs) {
        return (long) ORCHESTRATOR_FLOOR_MS + Math.max(overheadMs, 0);
    }
}
