package org.prebid.server.hooks.modules.artf.core;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.prebid.server.hooks.modules.artf.model.MutationTarget;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The path vocabulary, and the rejections.
 *
 * <p>Every accepted path here was read from the container that emits it, not from
 * documentation -- so these tests pin the resolver to what the orchestrator actually sends.
 */
class PathResolverTest {

    @Test
    void userSegmentsResolve() {
        // widedeep_segment_activator/app.py:195
        assertThat(PathResolver.resolve("/user/data/segment"))
                .isInstanceOf(MutationTarget.UserSegments.class);
    }

    @Test
    void impDealsResolve() {
        // ncf_deal_manager/app.py:181,186
        assertThat(PathResolver.resolve("/imp/imp-1"))
                .isEqualTo(new MutationTarget.ImpDeals("imp-1"));
    }

    @Test
    void impMetricsResolve() {
        // metrics_enricher/app.py:102
        assertThat(PathResolver.resolve("/imp/imp-1/metric"))
                .isEqualTo(new MutationTarget.ImpMetrics("imp-1"));
    }

    @Test
    void dealFloorsResolve() {
        // yield_optimizer_floor/app.py:136 and yield_optimizer_margin/app.py:142
        assertThat(PathResolver.resolve("/imp/imp-1/deals/deal-9"))
                .isEqualTo(new MutationTarget.DealFloor("imp-1", "deal-9"));
    }

    @Test
    void pathsWithoutALeadingSlashResolveTheSameWay() {
        // The framework's paths are absolute, but tolerating a missing leading slash costs
        // nothing and avoids a rejection whose reason would be about punctuation.
        assertThat(PathResolver.resolve("imp/imp-1/metric"))
                .isEqualTo(new MutationTarget.ImpMetrics("imp-1"));
    }

    /**
     * The finding that made this class more than a map lookup.
     *
     * <p>{@code dlrm_bid_shader} emits {@code /seatbid/{seat}/bid/{bidId}} for
     * {@code BID_SHADE}. That addresses a bid in the auction RESPONSE, and this hook runs at
     * {@code processed-auction-request} where no {@code seatbid} exists. Left unhandled it
     * would have been a silent no-op -- a mutation that appears to apply and changes nothing.
     */
    @Test
    void responseSidePathsAreRejectedWithAReasonThatExplainsWhy() {
        final MutationTarget target = PathResolver.resolve("/seatbid/artfhouse/bid/bid-1");

        assertThat(target).isInstanceOf(MutationTarget.Unresolvable.class);

        final String reason = ((MutationTarget.Unresolvable) target).reason();
        assertThat(reason)
                .contains("/seatbid/artfhouse/bid/bid-1")
                .contains("processed-auction-request")
                .contains("BID_SHADE");
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "/site/domain",
            "/imp",
            "/imp//metric",
            "/imp/imp-1/deals",
            "/imp/imp-1/deals/",
            "/user/data",
            "/user/data/segment/extra",
            "/device/ua",
    })
    void anythingElseIsRejectedWithAReasonListingWhatIsPermitted(String path) {
        final MutationTarget target = PathResolver.resolve(path);

        assertThat(target).isInstanceOf(MutationTarget.Unresolvable.class);
        // A rejection the operator cannot act on is barely better than a silent no-op, so the
        // reason names the permitted paths.
        assertThat(((MutationTarget.Unresolvable) target).reason())
                .contains("/user/data/segment")
                .contains("/imp/{impId}");
    }

    @ParameterizedTest
    @ValueSource(strings = {"", "   "})
    void blankPathsAreRejected(String path) {
        assertThat(PathResolver.resolve(path)).isInstanceOf(MutationTarget.Unresolvable.class);
    }

    @Test
    void aNullPathIsRejectedRatherThanThrowing() {
        // A malformed mutation must not take the auction down with it.
        assertThat(PathResolver.resolve(null)).isInstanceOf(MutationTarget.Unresolvable.class);
    }
}
