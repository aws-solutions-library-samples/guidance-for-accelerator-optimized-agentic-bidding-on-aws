package org.prebid.server.hooks.modules.artf.model;

/**
 * What happened to one mutation.
 *
 * <p>EVERY mutation gets one (BR-15), so a caller never has to infer what happened. And a
 * rejection is a NORMAL outcome, not an error (BR-16): agents propose, the host decides,
 * and the applier declining a mutation is the framework working as designed.
 *
 * @param intent   the mutation's intent, carried so per-intent counts can be reported --
 *                 which answers the question this feature will actually be asked: WHICH
 *                 intent did nothing
 * @param path     the mutation's semantic path
 * @param applied  whether it landed
 * @param reason   why it was rejected; null when applied. Never a bare "invalid": a
 *                 rejection the operator cannot act on is barely better than a silent
 *                 no-op
 */
public record Disposition(Intent intent, String path, boolean applied, String reason) {

    public static Disposition applied(Intent intent, String path) {
        return new Disposition(intent, path, true, null);
    }

    public static Disposition rejected(Intent intent, String path, String reason) {
        return new Disposition(intent, path, false, reason);
    }
}
