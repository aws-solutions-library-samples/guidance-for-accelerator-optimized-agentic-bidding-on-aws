package org.prebid.server.hooks.modules.artf.core;

import org.prebid.server.hooks.modules.artf.model.MutationTarget;

/**
 * Maps an ARTF semantic path to a location on the shared bid request. PURE.
 *
 * <p>The vocabulary below was read out of the containers that emit it, not inferred from
 * the framework's documentation:
 *
 * <table>
 *   <caption>Paths the orchestrator emits</caption>
 *   <tr><th>Path</th><th>Emitted by</th><th>Target</th></tr>
 *   <tr><td>{@code /user/data/segment}</td><td>widedeep_segment_activator</td>
 *       <td>{@code user.data}</td></tr>
 *   <tr><td>{@code /imp/{impId}}</td><td>ncf_deal_manager</td>
 *       <td>{@code imp.pmp.deals}</td></tr>
 *   <tr><td>{@code /imp/{impId}/deals/{dealId}}</td><td>yield_optimizer_floor / _margin</td>
 *       <td>a deal's floor</td></tr>
 *   <tr><td>{@code /imp/{impId}/metric}</td><td>metrics_enricher</td>
 *       <td>{@code imp.metric}</td></tr>
 *   <tr><td>{@code /seatbid/{seat}/bid/{bidId}}</td><td>dlrm_bid_shader</td>
 *       <td><b>none</b></td></tr>
 * </table>
 *
 * <p><b>That last row is why this class is not a one-line map lookup.</b>
 * {@code /seatbid/...} addresses a bid in the auction RESPONSE. This hook runs at
 * {@code processed-auction-request}, before the bidder fan-out, so no {@code seatbid}
 * exists and there is nothing to write to. It resolves to
 * {@link MutationTarget.Unresolvable} with a reason that says so -- a rejection the reader
 * can act on, rather than a mutation that appears to apply and changes nothing.
 *
 * <p>Every unrecognised path is likewise a rejection with a reason (BR-14), never a silent
 * no-op.
 */
public final class PathResolver {

    private PathResolver() {
    }

    public static MutationTarget resolve(String path) {
        if (path == null || path.isBlank()) {
            return new MutationTarget.Unresolvable("mutation carries no path");
        }

        final String normalised = path.startsWith("/") ? path.substring(1) : path;
        final String[] parts = normalised.split("/");

        // /user/data/segment
        if (parts.length == 3
                && "user".equals(parts[0]) && "data".equals(parts[1]) && "segment".equals(parts[2])) {
            return new MutationTarget.UserSegments();
        }

        // /seatbid/{seat}/bid/{bidId} -- response-side, and named explicitly so the
        // rejection reason tells the reader WHY rather than just that it failed.
        if ("seatbid".equals(parts[0])) {
            return new MutationTarget.Unresolvable(
                    "path '%s' addresses the auction response, but this hook runs at "
                            .formatted(path)
                            + "processed-auction-request where no seatbid exists yet. "
                            + "A response-side intent such as BID_SHADE cannot be applied here; "
                            + "remove it from the configured intent set for this stage.");
        }

        if ("imp".equals(parts[0])) {
            // /imp/{impId}
            if (parts.length == 2 && !parts[1].isBlank()) {
                return new MutationTarget.ImpDeals(parts[1]);
            }
            // /imp/{impId}/metric
            if (parts.length == 3 && "metric".equals(parts[2]) && !parts[1].isBlank()) {
                return new MutationTarget.ImpMetrics(parts[1]);
            }
            // /imp/{impId}/deals/{dealId}
            if (parts.length == 4 && "deals".equals(parts[2])
                    && !parts[1].isBlank() && !parts[3].isBlank()) {
                return new MutationTarget.DealFloor(parts[1], parts[3]);
            }
        }

        return new MutationTarget.Unresolvable(
                "path '%s' does not name a location this hook may write. Permitted: "
                        .formatted(path)
                        + "/user/data/segment, /imp/{impId}, /imp/{impId}/metric, "
                        + "/imp/{impId}/deals/{dealId}");
    }
}
