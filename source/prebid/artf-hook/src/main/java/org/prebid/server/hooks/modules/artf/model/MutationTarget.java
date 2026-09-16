package org.prebid.server.hooks.modules.artf.model;

/**
 * Where on the SHARED bid request a mutation lands, resolved from its ARTF path.
 *
 * <p>Only four locations are permitted (FR-4, BR-11): {@code imp.pmp.deals},
 * {@code imp.bidfloor}, {@code user.data} and {@code imp.metric}. Sealed, so the permitted
 * set is enforced by the type system rather than by a comment.
 *
 * <p>Nothing here can name a bidder. Mutations are NEVER applied to bidder-specific
 * impression data (BR-12), and that is not a stylistic choice: Prebid's module rules
 * require a module to expose its data to all bidders rather than privileging one, so
 * writing into a single bidder's namespace would block this module from ever being
 * packaged for general use.
 *
 * <p>{@link Unresolvable} is a first-class case, not an absence. A path that does not
 * resolve becomes a rejection carrying this reason (BR-14) -- silent no-ops are how a
 * feature comes to appear functional while doing nothing.
 */
public sealed interface MutationTarget {

    /** Segments on {@code user.data}. */
    record UserSegments() implements MutationTarget {
    }

    /** Deals on {@code imp[impId].pmp.deals}. */
    record ImpDeals(String impId) implements MutationTarget {
    }

    /** A specific deal's floor, under {@code imp[impId].pmp.deals[dealId]}. */
    record DealFloor(String impId, String dealId) implements MutationTarget {
    }

    /** Metrics on {@code imp[impId].metric}. */
    record ImpMetrics(String impId) implements MutationTarget {
    }

    /**
     * The path names nothing this hook can write at this stage.
     *
     * @param reason a description naming the path and why -- for
     *               {@code /seatbid/...} that it is response-side and no {@code seatbid}
     *               exists at {@code processed-auction-request}
     */
    record Unresolvable(String reason) implements MutationTarget {
    }
}
