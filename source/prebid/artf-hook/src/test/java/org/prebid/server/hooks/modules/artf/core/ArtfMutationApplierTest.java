package org.prebid.server.hooks.modules.artf.core;

import com.iab.openrtb.request.BidRequest;
import com.iab.openrtb.request.Deal;
import com.iab.openrtb.request.Imp;
import com.iab.openrtb.request.Pmp;
import org.junit.jupiter.api.Test;
import org.prebid.server.hooks.modules.artf.model.ApplicationResult;
import org.prebid.server.hooks.modules.artf.model.ArtfMutation;
import org.prebid.server.hooks.modules.artf.model.Disposition;
import org.prebid.server.hooks.modules.artf.model.Intent;
import org.prebid.server.hooks.modules.artf.model.Operation;
import org.prebid.server.json.ObjectMapperProvider;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The applier: two properties and the examples that pin each rule.
 *
 * <p>On the absence of a PBT framework, and on these tests not running during a deployment,
 * see the header of {@link CallBudgetCalculatorTest}. The two properties here are swept with a
 * <b>fixed seed</b>, so a failure is reproducible by re-running rather than by capturing a
 * seed from a log.
 */
class ArtfMutationApplierTest {

    private static final long SEED = 20260915L;

    private final ArtfMutationApplier applier = new ArtfMutationApplier(ObjectMapperProvider.mapper());

    // -------------------------------------------------------------- properties

    /**
     * Disposition completeness -- the executable form of BR-15.
     *
     * <p>Every mutation gets a disposition, so a caller never has to infer what happened. The
     * generated list mixes valid, invalid and null mutations precisely because the count must
     * hold regardless of how many are rejected.
     */
    @Test
    void everyMutationReceivesExactlyOneDisposition() {
        final Random random = new Random(SEED);

        for (int trial = 0; trial < 500; trial++) {
            final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.valueOf(1.50));
            final List<ArtfMutation> mutations = randomMutations(random);

            final ApplicationResult result = applier.apply(request, mutations);

            assertThat(result.dispositions())
                    .as("trial %d with %d mutations", trial, mutations.size())
                    .hasSameSizeAs(mutations);
            assertThat(result.appliedCount() + result.rejectedCount())
                    .isEqualTo(mutations.size());
        }
    }

    /**
     * Application atomicity -- the executable form of BR-13.
     *
     * <p>For a single mutation the resulting request either carries the full change or is
     * equal to the input. There is no third state: a partially applied mutation would leave
     * the request in a shape neither the agent proposed nor the host validated.
     */
    @Test
    void aSingleMutationEitherAppliesFullyOrLeavesTheRequestUntouched() {
        final Random random = new Random(SEED);

        for (int trial = 0; trial < 500; trial++) {
            final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.valueOf(1.50));
            final ArtfMutation mutation = randomMutation(random);

            // Arrays.asList, not List.of: the generator can produce a null mutation and the
            // applier must give it a disposition rather than throw.
            final ApplicationResult result = applier.apply(request, Arrays.asList(mutation));
            final Disposition disposition = result.dispositions().get(0);

            if (disposition.applied()) {
                assertThat(result.bidRequest())
                        .as("trial %d: an applied mutation must change the request", trial)
                        .isNotEqualTo(request);
            } else {
                assertThat(result.bidRequest())
                        .as("trial %d: a rejected mutation must leave the request untouched", trial)
                        .isEqualTo(request);
            }
        }
    }

    /** A rejection always carries a reason. A bare rejection is not actionable. */
    @Test
    void everyRejectionCarriesANonBlankReason() {
        final Random random = new Random(SEED);
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        for (int trial = 0; trial < 500; trial++) {
            final ApplicationResult result = applier.apply(request, randomMutations(random));
            for (Disposition disposition : result.dispositions()) {
                if (!disposition.applied()) {
                    assertThat(disposition.reason()).as("trial %d", trial).isNotBlank();
                }
            }
        }
    }

    // ---------------------------------------------------------------- examples

    /**
     * The gate's finding, pinned: a floor is always written with its currency.
     *
     * <p>Prebid drops an impression from its floor-enforcement map unless
     * {@code bidfloorcur} is non-blank ({@code ExchangeService.java:766-770}). A floor without
     * a currency therefore applies and changes nothing -- the mutation is visible in the
     * request and cannot reject a single bid.
     */
    @Test
    void aDealFloorMutationWritesBothTheFloorAndTheCurrency() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.valueOf(1.00));

        final ApplicationResult result = applier.apply(request, List.of(dealFloor("imp-1", "deal-1", 2.50)));

        assertThat(result.dispositions().get(0).applied()).isTrue();

        final Imp imp = result.bidRequest().getImp().get(0);
        final Deal deal = imp.getPmp().getDeals().get(0);

        assertThat(deal.getBidfloor()).isEqualByComparingTo("2.50");
        assertThat(deal.getBidfloorcur()).isNotBlank();

        // And the IMPRESSION floor too, because that is the value Prebid's enforcer compares.
        assertThat(imp.getBidfloor()).isEqualByComparingTo("2.50");
        assertThat(imp.getBidfloorcur()).isNotBlank();
    }

    /**
     * A mutation never lowers a floor the publisher already set.
     *
     * <p>Lowering one would let the enrichment weaken the publisher's own protection, which is
     * not a decision an agent gets to make.
     */
    @Test
    void anImpressionFloorIsNeverLoweredByAMutation() {
        final BidRequest base = requestWithDeal("imp-1", "deal-1", BigDecimal.valueOf(1.00));
        final Imp withPublisherFloor = base.getImp().get(0).toBuilder()
                .bidfloor(BigDecimal.valueOf(5.00))
                .bidfloorcur("USD")
                .build();
        final BidRequest request = base.toBuilder().imp(List.of(withPublisherFloor)).build();

        final ApplicationResult result = applier.apply(request, List.of(dealFloor("imp-1", "deal-1", 2.50)));

        // The deal's own floor takes the mutation's value; the impression floor holds at the
        // publisher's higher one.
        assertThat(result.bidRequest().getImp().get(0).getPmp().getDeals().get(0).getBidfloor())
                .isEqualByComparingTo("2.50");
        assertThat(result.bidRequest().getImp().get(0).getBidfloor()).isEqualByComparingTo("5.00");
    }

    /** A floor of zero is ignored by Prebid, so it is rejected rather than written. */
    @Test
    void aNonPositiveFloorIsRejectedRatherThanWrittenAndIgnored() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.valueOf(1.00));

        final ApplicationResult result = applier.apply(request, List.of(dealFloor("imp-1", "deal-1", 0.0)));

        assertThat(result.dispositions().get(0).applied()).isFalse();
        assertThat(result.dispositions().get(0).reason()).contains("greater than zero");
        assertThat(result.bidRequest()).isEqualTo(request);
    }

    /**
     * Suppression marks the deal rather than removing it.
     *
     * <p>Removing it would leave the Theater unable to show that a deal was considered and
     * suppressed: it would simply be absent, indistinguishable from never having been offered.
     */
    @Test
    void suppressionMarksTheDealAtTheAgreedPathRatherThanRemovingIt() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of(
                new ArtfMutation(Intent.SUPPRESS_DEALS.wireValue(), Operation.REPLACE.wireValue(),
                        "/imp/imp-1", new ArtfMutation.IdsPayload(List.of("deal-1")), null, null, null)));

        assertThat(result.dispositions().get(0).applied()).isTrue();

        final List<Deal> deals = result.bidRequest().getImp().get(0).getPmp().getDeals();
        assertThat(deals).hasSize(1);
        assertThat(deals.get(0).getExt().path("artf").path("suppressed").asBoolean()).isTrue();
    }

    /**
     * A response-side intent is rejected with a reason naming the stage.
     *
     * <p>{@code BID_SHADE} targets {@code /seatbid/...}, and no {@code seatbid} exists before
     * the bidder fan-out.
     */
    @Test
    void aBidShadeMutationIsRejectedBecauseItAddressesTheResponse() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of(
                new ArtfMutation(Intent.BID_SHADE.wireValue(), Operation.REPLACE.wireValue(),
                        "/seatbid/artfhouse/bid/bid-1", null, null,
                        new ArtfMutation.AdjustBidPayload(1.25), null)));

        assertThat(result.dispositions().get(0).applied()).isFalse();
        assertThat(result.dispositions().get(0).reason()).contains("processed-auction-request");
        assertThat(result.bidRequest()).isEqualTo(request);
    }

    /** Prebid requires imp.metric[].value in [0,1]; outside that it is rejected, not clamped. */
    @Test
    void aMetricOutsideTheOpenRtbRangeIsRejectedRatherThanClamped() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of(
                new ArtfMutation(Intent.ADD_METRICS.wireValue(), Operation.ADD.wireValue(),
                        "/imp/imp-1/metric", null, null, null,
                        new ArtfMutation.AddMetricsPayload(List.of(
                                new ArtfMutation.ArtfMetric("viewability", 3.4, "artf"))))));

        assertThat(result.dispositions().get(0).applied()).isFalse();
        assertThat(result.dispositions().get(0).reason()).contains("outside");
    }

    @Test
    void aMetricInsideTheRangeIsApplied() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of(
                new ArtfMutation(Intent.ADD_METRICS.wireValue(), Operation.ADD.wireValue(),
                        "/imp/imp-1/metric", null, null, null,
                        new ArtfMutation.AddMetricsPayload(List.of(
                                new ArtfMutation.ArtfMetric("viewability", 0.75, "artf"))))));

        assertThat(result.dispositions().get(0).applied()).isTrue();
        assertThat(result.bidRequest().getImp().get(0).getMetric()).hasSize(1);
    }

    @Test
    void segmentActivationAppendsToUserDataWithoutDiscardingExistingData() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of(
                new ArtfMutation(Intent.ACTIVATE_SEGMENTS.wireValue(), Operation.ADD.wireValue(),
                        "/user/data/segment", new ArtfMutation.IdsPayload(List.of("seg-1", "seg-2")),
                        null, null, null)));

        assertThat(result.dispositions().get(0).applied()).isTrue();
        assertThat(result.bidRequest().getUser().getData()).hasSize(1);
        assertThat(result.bidRequest().getUser().getData().get(0).getSegment()).hasSize(2);
    }

    @Test
    void aMutationNamingAnUnknownImpressionIsRejected() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of(dealFloor("imp-9", "deal-1", 2.0)));

        assertThat(result.dispositions().get(0).applied()).isFalse();
        assertThat(result.dispositions().get(0).reason()).contains("imp-9");
    }

    @Test
    void anEmptyMutationListLeavesTheRequestUntouchedAndProducesNoDispositions() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        final ApplicationResult result = applier.apply(request, List.of());

        assertThat(result.bidRequest()).isEqualTo(request);
        assertThat(result.dispositions()).isEmpty();
        assertThat(result.anyApplied()).isFalse();
    }

    @Test
    void aNullMutationListIsTreatedAsEmptyRatherThanThrowing() {
        final BidRequest request = requestWithDeal("imp-1", "deal-1", BigDecimal.ONE);

        assertThat(applier.apply(request, null).bidRequest()).isEqualTo(request);
    }

    // ----------------------------------------------------------------- helpers

    private static BidRequest requestWithDeal(String impId, String dealId, BigDecimal floor) {
        return BidRequest.builder()
                .id("auction-1")
                .cur(List.of("USD"))
                .imp(List.of(Imp.builder()
                        .id(impId)
                        .pmp(Pmp.builder()
                                .deals(List.of(Deal.builder().id(dealId).bidfloor(floor).build()))
                                .build())
                        .build()))
                .build();
    }

    private static ArtfMutation dealFloor(String impId, String dealId, double floor) {
        return new ArtfMutation(
                Intent.ADJUST_DEAL_FLOOR.wireValue(),
                Operation.REPLACE.wireValue(),
                "/imp/%s/deals/%s".formatted(impId, dealId),
                null,
                new ArtfMutation.AdjustDealPayload(floor, null),
                null,
                null);
    }

    /**
     * A list that MAY CONTAIN NULLS, which is why it is not built with {@code List.of} or
     * {@code List.copyOf} -- both reject null elements. The applier is written to give a null
     * mutation a disposition rather than throw, and a generator that could not produce one
     * would never exercise that.
     */
    private static List<ArtfMutation> randomMutations(Random random) {
        final int count = random.nextInt(6);
        final List<ArtfMutation> mutations = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            mutations.add(randomMutation(random));
        }
        return Collections.unmodifiableList(mutations);
    }

    /**
     * A mutation drawn from the whole space, valid and invalid alike -- including nulls and
     * response-side paths. Completeness and atomicity must hold across all of it, not only
     * across the mutations that happen to succeed.
     */
    private static ArtfMutation randomMutation(Random random) {
        return switch (random.nextInt(8)) {
            case 0 -> dealFloor("imp-1", "deal-1", 0.5 + random.nextDouble() * 5);
            case 1 -> dealFloor("imp-1", "deal-missing", random.nextDouble() * 3);
            case 2 -> new ArtfMutation(Intent.ACTIVATE_SEGMENTS.wireValue(), Operation.ADD.wireValue(),
                    "/user/data/segment", new ArtfMutation.IdsPayload(List.of("seg-" + random.nextInt(5))),
                    null, null, null);
            case 3 -> new ArtfMutation(Intent.ACTIVATE_DEALS.wireValue(), Operation.ADD.wireValue(),
                    "/imp/imp-1", new ArtfMutation.IdsPayload(List.of("deal-" + random.nextInt(5))),
                    null, null, null);
            case 4 -> new ArtfMutation(Intent.ADD_METRICS.wireValue(), Operation.ADD.wireValue(),
                    "/imp/imp-1/metric", null, null, null,
                    new ArtfMutation.AddMetricsPayload(List.of(
                            new ArtfMutation.ArtfMetric("viewability", random.nextDouble() * 2, "artf"))));
            case 5 -> new ArtfMutation(Intent.BID_SHADE.wireValue(), Operation.REPLACE.wireValue(),
                    "/seatbid/artfhouse/bid/bid-1", null, null,
                    new ArtfMutation.AdjustBidPayload(random.nextDouble() * 4), null);
            case 6 -> new ArtfMutation(Intent.UNSPECIFIED.wireValue(), Operation.UNSPECIFIED.wireValue(),
                    "/device/ua", null, null, null, null);
            default -> null;
        };
    }
}
