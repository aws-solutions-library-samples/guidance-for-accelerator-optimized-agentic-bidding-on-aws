package org.prebid.server.hooks.modules.artf.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.iab.openrtb.request.BidRequest;
import com.iab.openrtb.request.Data;
import com.iab.openrtb.request.Deal;
import com.iab.openrtb.request.Imp;
import com.iab.openrtb.request.Metric;
import com.iab.openrtb.request.Pmp;
import com.iab.openrtb.request.Segment;
import com.iab.openrtb.request.User;
import org.prebid.server.hooks.modules.artf.model.ApplicationResult;
import org.prebid.server.hooks.modules.artf.model.ArtfMutation;
import org.prebid.server.hooks.modules.artf.model.Disposition;
import org.prebid.server.hooks.modules.artf.model.Intent;
import org.prebid.server.hooks.modules.artf.model.MutationTarget;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Applies ARTF mutations to the SHARED bid request. The only domain logic in this unit, and
 * deliberately so: keeping it in one class is what makes the unit testable in a repository
 * whose Java surface exists only for this feature.
 *
 * <p>PURE in the sense that matters: no I/O, no clock, no configuration. It takes a request
 * and a mutation list and returns a new request plus one disposition per mutation.
 *
 * <p><b>Four locations only</b> (FR-4, BR-11): {@code imp.pmp.deals}, {@code imp.bidfloor},
 * {@code user.data}, {@code imp.metric}. Nothing here can name a bidder (BR-12) -- Prebid's
 * module rules require a module to expose its data to all bidders rather than privileging
 * one, so writing into a single bidder's namespace would block this module from ever being
 * packaged for general use. That constraint is what keeps it distributable.
 *
 * <p><b>All-or-nothing per mutation</b> (BR-13). Each mutation is applied to a candidate
 * built from the current request; if any part of it fails, the candidate is discarded and
 * the previous request carries forward. A partially applied mutation would leave the request
 * in a state neither the agent proposed nor the host validated -- a third thing nobody
 * designed.
 *
 * <p><b>Every mutation gets a disposition</b> (BR-15), and a rejection is a NORMAL outcome
 * (BR-16): agents propose, the host decides.
 */
public class ArtfMutationApplier {

    /**
     * Fallback floor currency. Used only when the request declares none.
     *
     * <p>A currency is not optional here. Prebid drops an impression from its floor-
     * enforcement map unless {@code bidfloorcur} is non-blank AND the floor is {@code > 0}
     * ({@code ExchangeService.java:766-770}, {@code BidderUtil.isValidPrice}). A floor
     * written without a currency is therefore silently ignored -- it applies, changes
     * nothing, and reports no error. That is the exact failure this unit exists to avoid,
     * so the currency is always written alongside the floor.
     */
    private static final String DEFAULT_CURRENCY = "USD";

    /** Where a suppressed deal is marked. U5 reads this path; U2 emits the same shape. */
    private static final String EXT_ARTF = "artf";
    private static final String EXT_SUPPRESSED = "suppressed";

    /**
     * OpenRTB bounds for {@code imp.metric[].value}, enforced by Prebid's own
     * {@code ImpValidator} ({@code ImpValidator.java:649-652}).
     */
    private static final double METRIC_MIN = 0.0d;
    private static final double METRIC_MAX = 1.0d;

    private final ObjectMapper mapper;

    public ArtfMutationApplier(ObjectMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper);
    }

    /**
     * Apply a mutation set.
     *
     * @return the request the auction should proceed with, and one disposition per mutation
     *         in input order
     */
    public ApplicationResult apply(BidRequest bidRequest, List<ArtfMutation> mutations) {
        Objects.requireNonNull(bidRequest);
        final List<ArtfMutation> input = mutations == null ? List.of() : mutations;

        BidRequest current = bidRequest;
        final List<Disposition> dispositions = new ArrayList<>(input.size());

        for (ArtfMutation mutation : input) {
            if (mutation == null) {
                // Counted, so the disposition list still matches the input list length.
                dispositions.add(Disposition.rejected(Intent.UNSPECIFIED, null, "mutation was null"));
                continue;
            }
            final Outcome outcome = applyOne(current, mutation);
            dispositions.add(outcome.disposition());
            if (outcome.disposition().applied()) {
                current = outcome.bidRequest();
            }
        }

        return new ApplicationResult(current, List.copyOf(dispositions));
    }

    /** One mutation, all-or-nothing. */
    private Outcome applyOne(BidRequest request, ArtfMutation mutation) {
        final Intent intent = mutation.intentValue();
        final String path = mutation.path();
        final MutationTarget target = PathResolver.resolve(path);

        if (target instanceof MutationTarget.Unresolvable unresolvable) {
            return Outcome.rejected(request, intent, path, unresolvable.reason());
        }

        try {
            return switch (target) {
                case MutationTarget.UserSegments ignored -> applyUserSegments(request, mutation, intent, path);
                case MutationTarget.ImpDeals deals -> applyImpDeals(request, mutation, intent, path, deals.impId());
                case MutationTarget.DealFloor floor -> applyDealFloor(request, mutation, intent, path, floor);
                case MutationTarget.ImpMetrics metrics ->
                        applyImpMetrics(request, mutation, intent, path, metrics.impId());
                case MutationTarget.Unresolvable unresolvable ->
                        Outcome.rejected(request, intent, path, unresolvable.reason());
            };
        } catch (RuntimeException e) {
            // The original request carries forward untouched. An exception mid-apply is
            // precisely the partial-application case BR-13 forbids, so it becomes a
            // rejection rather than a corrupted request.
            return Outcome.rejected(request, intent, path,
                    "mutation could not be applied: " + e.getClass().getSimpleName()
                            + (e.getMessage() == null ? "" : (": " + e.getMessage())));
        }
    }

    // ------------------------------------------------------------- user.data

    private Outcome applyUserSegments(BidRequest request, ArtfMutation mutation, Intent intent, String path) {
        if (intent != Intent.ACTIVATE_SEGMENTS) {
            return Outcome.rejected(request, intent, path,
                    "intent %s does not write user segments".formatted(intent));
        }
        final List<String> ids = idsOf(mutation);
        if (ids.isEmpty()) {
            return Outcome.rejected(request, intent, path, "no segment ids supplied");
        }

        final List<Segment> segments = ids.stream()
                .map(id -> Segment.builder().id(id).build())
                .toList();

        final User user = request.getUser();
        final List<Data> existing = user == null || user.getData() == null ? List.of() : user.getData();

        final List<Data> updated = new ArrayList<>(existing);
        updated.add(Data.builder()
                .name(EXT_ARTF)
                .segment(segments)
                .build());

        final User updatedUser = (user == null ? User.builder() : user.toBuilder())
                .data(List.copyOf(updated))
                .build();

        return Outcome.applied(request.toBuilder().user(updatedUser).build(), intent, path);
    }

    // -------------------------------------------------------- imp.pmp.deals

    private Outcome applyImpDeals(BidRequest request, ArtfMutation mutation, Intent intent,
                                  String path, String impId) {

        final int index = impIndex(request, impId);
        if (index < 0) {
            return Outcome.rejected(request, intent, path, "no impression with id '%s'".formatted(impId));
        }
        final List<String> ids = idsOf(mutation);
        if (ids.isEmpty()) {
            return Outcome.rejected(request, intent, path, "no deal ids supplied");
        }

        final Imp imp = request.getImp().get(index);
        final Pmp pmp = imp.getPmp();
        final List<Deal> existing = pmp == null || pmp.getDeals() == null ? List.of() : pmp.getDeals();

        final List<Deal> updatedDeals = switch (intent) {
            case ACTIVATE_DEALS -> activateDeals(existing, ids);
            case SUPPRESS_DEALS -> suppressDeals(existing, ids);
            default -> null;
        };
        if (updatedDeals == null) {
            return Outcome.rejected(request, intent, path,
                    "intent %s does not write deals".formatted(intent));
        }

        final Pmp updatedPmp = (pmp == null ? Pmp.builder() : pmp.toBuilder())
                .deals(updatedDeals)
                .build();

        return Outcome.applied(withImp(request, index, imp.toBuilder().pmp(updatedPmp).build()), intent, path);
    }

    private static List<Deal> activateDeals(List<Deal> existing, List<String> ids) {
        final List<Deal> result = new ArrayList<>(existing);
        for (String id : ids) {
            final boolean alreadyPresent = existing.stream().anyMatch(d -> id.equals(d.getId()));
            if (!alreadyPresent) {
                result.add(Deal.builder().id(id).build());
            }
        }
        return List.copyOf(result);
    }

    /**
     * Suppression marks the deal rather than removing it.
     *
     * <p>Removing it would leave the Theater unable to show that a deal was considered and
     * suppressed -- it would simply be absent, indistinguishable from never having been
     * offered. The marker path {@code deal.ext.artf.suppressed} is the one U5 reads and U2
     * emits; writing anywhere else makes a suppressed deal read as live.
     */
    private List<Deal> suppressDeals(List<Deal> existing, List<String> ids) {
        final List<Deal> result = new ArrayList<>(existing.size());
        for (Deal deal : existing) {
            if (ids.contains(deal.getId())) {
                final ObjectNode ext = deal.getExt() == null
                        ? mapper.createObjectNode()
                        : deal.getExt().deepCopy();
                final ObjectNode artf = ext.get(EXT_ARTF) instanceof ObjectNode existingArtf
                        ? existingArtf
                        : ext.putObject(EXT_ARTF);
                artf.put(EXT_SUPPRESSED, true);
                result.add(deal.toBuilder().ext(ext).build());
            } else {
                result.add(deal);
            }
        }
        return List.copyOf(result);
    }

    // ---------------------------------------------- deal floor + imp.bidfloor

    private Outcome applyDealFloor(BidRequest request, ArtfMutation mutation, Intent intent,
                                   String path, MutationTarget.DealFloor target) {

        if (intent != Intent.ADJUST_DEAL_FLOOR && intent != Intent.ADJUST_DEAL_MARGIN) {
            return Outcome.rejected(request, intent, path,
                    "intent %s does not write a deal floor".formatted(intent));
        }
        final int index = impIndex(request, target.impId());
        if (index < 0) {
            return Outcome.rejected(request, intent, path,
                    "no impression with id '%s'".formatted(target.impId()));
        }
        final Imp imp = request.getImp().get(index);
        final Pmp pmp = imp.getPmp();
        if (pmp == null || pmp.getDeals() == null || pmp.getDeals().isEmpty()) {
            return Outcome.rejected(request, intent, path,
                    "impression '%s' carries no deals".formatted(target.impId()));
        }
        final int dealIndex = dealIndex(pmp.getDeals(), target.dealId());
        if (dealIndex < 0) {
            return Outcome.rejected(request, intent, path,
                    "impression '%s' has no deal '%s'".formatted(target.impId(), target.dealId()));
        }

        final Deal deal = pmp.getDeals().get(dealIndex);
        final BigDecimal newFloor = resolveFloor(mutation, deal, intent);
        if (newFloor == null) {
            return Outcome.rejected(request, intent, path,
                    "mutation supplied no usable floor or margin value");
        }
        // Prebid ignores a floor that is not strictly positive (BidderUtil.isValidPrice), so
        // a zero or negative result is a rejection rather than a write that does nothing.
        if (newFloor.compareTo(BigDecimal.ZERO) <= 0) {
            return Outcome.rejected(request, intent, path,
                    "resulting floor %s is not greater than zero, which Prebid ignores".formatted(newFloor));
        }

        final String currency = resolveCurrency(request, deal, imp);

        final List<Deal> updatedDeals = new ArrayList<>(pmp.getDeals());
        updatedDeals.set(dealIndex, deal.toBuilder()
                .bidfloor(newFloor)
                .bidfloorcur(currency)
                .build());

        final Imp.ImpBuilder impBuilder = imp.toBuilder()
                .pmp(pmp.toBuilder().deals(List.copyOf(updatedDeals)).build());

        // ------------------------------------------------------------------
        // ALSO write the IMPRESSION floor, and the currency with it.
        //
        // Prebid's floor enforcer compares a bid against the map it builds from
        // imp.bidfloor + imp.bidfloorcur (ExchangeService.java:766-770). A deal-level
        // floor is CARRIED to the bidders but is not what that enforcer compares, so a
        // deal floor written alone applies and changes nothing -- FR-19's mutation would
        // be visible in the request and unable to reject any bid.
        //
        // Raised to the maximum of the existing floor and the new one, so a mutation
        // never LOWERS a floor the publisher already set. Lowering one would let the
        // enrichment weaken the publisher's own protection, which is not a decision an
        // agent gets to make.
        // ------------------------------------------------------------------
        final BigDecimal existingImpFloor = imp.getBidfloor();
        final BigDecimal impFloor = existingImpFloor == null
                ? newFloor
                : existingImpFloor.max(newFloor);
        impBuilder.bidfloor(impFloor).bidfloorcur(currency);

        return Outcome.applied(withImp(request, index, impBuilder.build()), intent, path);
    }

    /**
     * The floor this mutation implies.
     *
     * <p>{@code ADJUST_DEAL_FLOOR} carries an absolute {@code bidfloor}.
     * {@code ADJUST_DEAL_MARGIN} carries a margin whose {@code calculation_type} is 0 for
     * CPM (absolute, added to the existing floor) or 1 for PERCENT (relative). An
     * unrecognised calculation type yields null, and therefore a rejection with a reason --
     * guessing which one was meant would silently pick a floor nobody asked for.
     */
    private static BigDecimal resolveFloor(ArtfMutation mutation, Deal deal, Intent intent) {
        final ArtfMutation.AdjustDealPayload payload = mutation.adjustDeal();
        if (payload == null) {
            return null;
        }
        if (intent == Intent.ADJUST_DEAL_FLOOR) {
            return payload.bidfloor() == null ? null : BigDecimal.valueOf(payload.bidfloor());
        }

        final ArtfMutation.Margin margin = payload.margin();
        if (margin == null || margin.value() == null) {
            // A margin intent may still carry an absolute floor; use it rather than
            // rejecting a mutation that did express a value.
            return payload.bidfloor() == null ? null : BigDecimal.valueOf(payload.bidfloor());
        }
        final BigDecimal base = deal.getBidfloor() == null ? BigDecimal.ZERO : deal.getBidfloor();
        final BigDecimal value = BigDecimal.valueOf(margin.value());

        return switch (margin.calculationType()) {
            case 0 -> base.add(value);
            case 1 -> base.add(base.multiply(value).divide(BigDecimal.valueOf(100), 4, RoundingMode.HALF_UP));
            default -> null;
        };
    }

    /**
     * The currency to write with the floor.
     *
     * <p>Resolution order: the deal's own currency, then the impression's, then the
     * request's first declared currency, then USD. The ARTF payload carries a bare number
     * with no currency, so it has to come from the request -- and Prebid compares in the
     * auction's currency, which is what {@code bidRequest.cur} declares.
     */
    private static String resolveCurrency(BidRequest request, Deal deal, Imp imp) {
        if (deal.getBidfloorcur() != null && !deal.getBidfloorcur().isBlank()) {
            return deal.getBidfloorcur();
        }
        if (imp.getBidfloorcur() != null && !imp.getBidfloorcur().isBlank()) {
            return imp.getBidfloorcur();
        }
        final List<String> cur = request.getCur();
        if (cur != null && !cur.isEmpty() && cur.get(0) != null && !cur.get(0).isBlank()) {
            return cur.get(0);
        }
        return DEFAULT_CURRENCY;
    }

    // ----------------------------------------------------------- imp.metric

    private Outcome applyImpMetrics(BidRequest request, ArtfMutation mutation, Intent intent,
                                    String path, String impId) {

        if (intent != Intent.ADD_METRICS && intent != Intent.ADD_CIDS) {
            return Outcome.rejected(request, intent, path,
                    "intent %s does not write metrics".formatted(intent));
        }
        final int index = impIndex(request, impId);
        if (index < 0) {
            return Outcome.rejected(request, intent, path, "no impression with id '%s'".formatted(impId));
        }
        final ArtfMutation.AddMetricsPayload payload = mutation.addMetrics();
        if (payload == null || payload.metric() == null || payload.metric().isEmpty()) {
            return Outcome.rejected(request, intent, path, "no metrics supplied");
        }

        final List<Metric> incoming = new ArrayList<>();
        for (ArtfMutation.ArtfMetric metric : payload.metric()) {
            if (metric == null || metric.type() == null || metric.type().isBlank() || metric.value() == null) {
                return Outcome.rejected(request, intent, path,
                        "metric requires a non-blank type and a value");
            }
            // Prebid's own ImpValidator requires imp.metric[].value in [0.0, 1.0]
            // (ImpValidator.java:649-652) -- metrics are probabilities in OpenRTB, not
            // arbitrary numbers. Rejected with a reason rather than clamped: clamping would
            // turn the agent's proposal into a different value and report it as applied,
            // which is a quieter form of the same lie.
            if (metric.value() < METRIC_MIN || metric.value() > METRIC_MAX) {
                return Outcome.rejected(request, intent, path,
                        "metric '%s' value %s is outside the [%s, %s] range OpenRTB defines for imp.metric"
                                .formatted(metric.type(), metric.value(), METRIC_MIN, METRIC_MAX));
            }
            incoming.add(Metric.builder()
                    .type(metric.type())
                    .value(metric.value().floatValue())
                    .vendor(metric.vendor())
                    .build());
        }

        final Imp imp = request.getImp().get(index);
        final List<Metric> existing = imp.getMetric() == null ? List.of() : imp.getMetric();
        final List<Metric> updated = new ArrayList<>(existing);
        updated.addAll(incoming);

        return Outcome.applied(
                withImp(request, index, imp.toBuilder().metric(List.copyOf(updated)).build()),
                intent, path);
    }

    // -------------------------------------------------------------- helpers

    private static List<String> idsOf(ArtfMutation mutation) {
        final ArtfMutation.IdsPayload ids = mutation.ids();
        if (ids == null || ids.id() == null) {
            return List.of();
        }
        return ids.id().stream().filter(Objects::nonNull).filter(s -> !s.isBlank()).toList();
    }

    private static int impIndex(BidRequest request, String impId) {
        final List<Imp> imps = request.getImp();
        if (imps == null || impId == null) {
            return -1;
        }
        for (int i = 0; i < imps.size(); i++) {
            if (impId.equals(imps.get(i).getId())) {
                return i;
            }
        }
        return -1;
    }

    private static int dealIndex(List<Deal> deals, String dealId) {
        for (int i = 0; i < deals.size(); i++) {
            if (dealId.equals(deals.get(i).getId())) {
                return i;
            }
        }
        return -1;
    }

    private static BidRequest withImp(BidRequest request, int index, Imp imp) {
        final List<Imp> updated = new ArrayList<>(request.getImp());
        updated.set(index, imp);
        return request.toBuilder().imp(List.copyOf(updated)).build();
    }

    /** A candidate request plus the disposition that produced it. */
    private record Outcome(BidRequest bidRequest, Disposition disposition) {

        static Outcome applied(BidRequest bidRequest, Intent intent, String path) {
            return new Outcome(bidRequest, Disposition.applied(intent, path));
        }

        static Outcome rejected(BidRequest unchanged, Intent intent, String path, String reason) {
            return new Outcome(unchanged, Disposition.rejected(intent, path, reason));
        }
    }
}
