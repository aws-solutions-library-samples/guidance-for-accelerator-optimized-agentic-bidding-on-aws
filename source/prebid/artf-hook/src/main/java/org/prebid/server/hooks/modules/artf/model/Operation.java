package org.prebid.server.hooks.modules.artf.model;

/**
 * ARTF mutation operations, with the integer values used on the wire.
 *
 * <p>Mirrors {@code source/shared/artf_types.py}. As with {@link Intent}, the values are
 * explicit rather than ordinal.
 */
public enum Operation {

    UNSPECIFIED(0),
    ADD(1),
    REMOVE(2),
    REPLACE(3);

    private final int wireValue;

    Operation(int wireValue) {
        this.wireValue = wireValue;
    }

    public int wireValue() {
        return wireValue;
    }

    public static Operation fromWireValue(int value) {
        for (Operation operation : values()) {
            if (operation.wireValue == value) {
                return operation;
            }
        }
        return UNSPECIFIED;
    }
}
