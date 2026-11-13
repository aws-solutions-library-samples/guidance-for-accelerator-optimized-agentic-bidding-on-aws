package org.prebid.server.hooks.modules.artf.model;

/**
 * ARTF intents, with the integer values the framework uses on the wire.
 *
 * <p>The values mirror {@code source/shared/artf_types.py} exactly. They are not
 * ordinals: the orchestrator sends integers, and relying on Java enum ordering would
 * silently remap every intent the moment a constant is inserted.
 *
 * <p>{@link #BID_SHADE} is listed because the framework defines it, and deliberately
 * cannot be applied by this hook. It targets a bid in the auction RESPONSE
 * ({@code /seatbid/{seat}/bid/{id}}), and this hook runs at
 * {@code processed-auction-request}, where no {@code seatbid} exists yet. A
 * {@code BID_SHADE} mutation arriving here is rejected with a reason rather than
 * dropped -- see {@link org.prebid.server.hooks.modules.artf.core.PathResolver}.
 */
public enum Intent {

    UNSPECIFIED(0),
    ACTIVATE_SEGMENTS(1),
    ACTIVATE_DEALS(2),
    SUPPRESS_DEALS(3),
    ADJUST_DEAL_FLOOR(4),
    ADJUST_DEAL_MARGIN(5),
    BID_SHADE(6),
    ADD_METRICS(7),
    ADD_CIDS(8);

    private final int wireValue;

    Intent(int wireValue) {
        this.wireValue = wireValue;
    }

    public int wireValue() {
        return wireValue;
    }

    /**
     * The intent for a wire value, or {@link #UNSPECIFIED} for one this build does not
     * know.
     *
     * <p>An unknown value is mapped rather than thrown on: the orchestrator may add an
     * intent before this module is rebuilt, and refusing the whole response because one
     * mutation carries an unrecognised intent would discard the mutations that ARE
     * understood. The unknown one is rejected individually by the path resolver, with a
     * reason.
     */
    public static Intent fromWireValue(int value) {
        for (Intent intent : values()) {
            if (intent.wireValue == value) {
                return intent;
            }
        }
        return UNSPECIFIED;
    }

    /**
     * The name the framework uses in {@code applicable_intents} and in configuration.
     * Matches the Python enum member name, so configuration written against the
     * framework's own vocabulary works unchanged.
     */
    public String wireName() {
        return name();
    }
}
