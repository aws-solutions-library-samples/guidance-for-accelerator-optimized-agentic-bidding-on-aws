package org.prebid.server.hooks.modules.artf.model;

/**
 * The result of the budget arithmetic: how long to wait, what to advertise, and whether to
 * call at all.
 *
 * @param httpTimeoutMs  what this hook will actually wait for
 * @param effectiveTmaxMs what the orchestrator is told it has, never more than will be
 *                        honoured
 * @param viable          false when there is too little budget for a call worth making
 * @param remainingAfterReserveMs the budget left once the reserve is set aside, carried so
 *                        a skip can be reported with the numbers that caused it rather
 *                        than as a bare boolean
 */
public record CallBudget(
        long httpTimeoutMs,
        int effectiveTmaxMs,
        boolean viable,
        long remainingAfterReserveMs) {

    /**
     * A non-viable budget. {@code httpTimeoutMs} and {@code effectiveTmaxMs} are zero
     * because no call will be made -- not because they were unknown.
     */
    public static CallBudget notViable(long remainingAfterReserveMs) {
        return new CallBudget(0L, 0, false, remainingAfterReserveMs);
    }
}
