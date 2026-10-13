package org.prebid.server.hooks.modules.artf.model;

import com.iab.openrtb.request.BidRequest;

import java.util.List;

/**
 * The outcome of applying a mutation set: the request the auction should proceed with, and
 * one disposition per mutation.
 *
 * <p>The two are returned together because they are the same fact. Handing back a mutated
 * request without dispositions would leave the caller unable to report what happened, and
 * dispositions without the request would leave it unable to proceed.
 *
 * @param bidRequest   the resulting request; identical to the input when nothing applied
 * @param dispositions exactly one per input mutation, in input order (BR-15)
 */
public record ApplicationResult(BidRequest bidRequest, List<Disposition> dispositions) {

    /** How many mutations landed. */
    public long appliedCount() {
        return dispositions.stream().filter(Disposition::applied).count();
    }

    /** How many were rejected. A rejection is a normal outcome (BR-16), not an error. */
    public long rejectedCount() {
        return dispositions.size() - appliedCount();
    }

    /** Whether the request differs from the one that went in. */
    public boolean anyApplied() {
        return appliedCount() > 0;
    }
}
