package org.prebid.server.hooks.modules.artf.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.iab.openrtb.request.BidRequest;
import io.vertx.core.Future;
import org.prebid.server.hooks.modules.artf.client.ExtensionPointClient;
import org.prebid.server.hooks.modules.artf.model.ApplicationResult;
import org.prebid.server.hooks.modules.artf.model.ArtfModuleProperties;
import org.prebid.server.hooks.modules.artf.model.CallBudget;
import org.prebid.server.hooks.modules.artf.model.ExtensionPointOutcome;
import org.prebid.server.hooks.modules.artf.model.Intent;
import org.prebid.server.hooks.modules.artf.model.RtbRequest;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Coordinates one pass of the extension point: budget, token, call, apply.
 *
 * <p>This class exists because step three is identical for three of the five outcomes. Put
 * anywhere else, the hook grows parallel paths that drift apart.
 *
 * <p><b>The skip decision lives here</b>, before the client is invoked. Not in the client: a
 * client that sometimes declined to call would have two responsibilities and a contract that
 * reads "makes a call, unless it doesn't".
 */
public class MutationApplicationService {

    private final ExtensionPointClient client;
    private final ArtfMutationApplier applier;
    private final ArtfModuleProperties properties;
    private final ObjectMapper mapper;

    /**
     * Consecutive failures, reported but never acted on (U3-NFR-15, D4).
     *
     * <p>This is the diagnostic value of a circuit breaker without the behavioural coupling.
     * A breaker would make the Theater's rendering depend on hidden state, so two identical
     * scenarios could render differently because of an earlier, unrelated failure.
     */
    private final AtomicLong consecutiveFailures = new AtomicLong(0);

    public MutationApplicationService(ExtensionPointClient client,
                                     ArtfMutationApplier applier,
                                     ArtfModuleProperties properties,
                                     ObjectMapper mapper) {
        this.client = Objects.requireNonNull(client);
        this.applier = Objects.requireNonNull(applier);
        this.properties = Objects.requireNonNull(properties);
        this.mapper = Objects.requireNonNull(mapper);
    }

    /**
     * Run one pass.
     *
     * @param bidRequest        the processed auction request
     * @param remainingBudgetMs time left before Prebid's own deadline, from the invocation
     *                          context's {@code Timeout.remaining()}
     * @param groupTimeoutMs    the execution group's own timeout from the hooks plan, or a
     *                          non-positive value when unknown. Included because the group
     *                          timeout bounds the hook independently of the auction deadline,
     *                          and ignoring it would let the hook plan for a budget the host
     *                          will cut short
     * @return a Future that always SUCCEEDS. The hook never fails the auction (BR-8).
     */
    public Future<Pass> runOnce(BidRequest bidRequest, long remainingBudgetMs, long groupTimeoutMs) {
        final long effectiveRemaining = groupTimeoutMs > 0
                ? Math.min(remainingBudgetMs, groupTimeoutMs)
                : remainingBudgetMs;

        final CallBudget budget = CallBudgetCalculator.calculate(
                effectiveRemaining,
                properties.getTmaxMs(),
                properties.getOverheadMs(),
                properties.getReserveMs());

        if (!budget.viable()) {
            // A skip is NOT a failure (BR-7a, U3-NFR-9). Nothing was asked, so nothing failed
            // and the counter is untouched.
            final ExtensionPointOutcome outcome = new ExtensionPointOutcome.SkippedInsufficientBudget(
                    budget.remainingAfterReserveMs(),
                    CallBudgetCalculator.minimumViableMs(properties.getOverheadMs()));
            return Future.succeededFuture(Pass.unchanged(bidRequest, outcome, consecutiveFailures.get()));
        }

        final RtbRequest envelope = buildEnvelope(bidRequest, budget);

        return client.fetchMutations(envelope, budget)
                .map(outcome -> {
                    if (outcome.failure()) {
                        consecutiveFailures.incrementAndGet();
                    } else {
                        consecutiveFailures.set(0);
                    }
                    if (outcome instanceof ExtensionPointOutcome.MutationsReturned returned) {
                        final ApplicationResult applied = applier.apply(bidRequest, returned.mutations());
                        return new Pass(applied.bidRequest(), outcome, applied, consecutiveFailures.get());
                    }
                    return Pass.unchanged(bidRequest, outcome, consecutiveFailures.get());
                });
    }

    /**
     * Build the ARTF envelope.
     *
     * <p>The correlation id is {@code source.tid} (BR-19). One key present on the request the
     * containers saw and on the response the frontend renders is what lets a rendered auction
     * be tied back to the mutations that shaped it; without it the Theater would assert a
     * correspondence it could not demonstrate.
     */
    private RtbRequest buildEnvelope(BidRequest bidRequest, CallBudget budget) {
        final String correlationId = correlationId(bidRequest);

        @SuppressWarnings("unchecked")
        final Map<String, Object> bidRequestMap = mapper.convertValue(bidRequest, Map.class);

        return new RtbRequest(
                correlationId,
                RtbRequest.LIFECYCLE_PUBLISHER_BID_REQUEST,
                budget.effectiveTmaxMs(),
                bidRequestMap,
                new RtbRequest.Originator("TYPE_EXCHANGE", "prebid-server"),
                effectiveIntents(bidRequest),
                null);
    }

    /**
     * The ARTF request id, taken from {@code source.tid}.
     *
     * <p>Falls back to the bid request id when no {@code tid} is present. Reported as such
     * rather than generating one: a fabricated id would correlate to nothing, and a
     * correlation key that leads nowhere is worse than an obviously reused one.
     */
    public static String correlationId(BidRequest bidRequest) {
        if (bidRequest.getSource() != null
                && bidRequest.getSource().getTid() != null
                && !bidRequest.getSource().getTid().isBlank()) {
            return bidRequest.getSource().getTid();
        }
        return bidRequest.getId();
    }

    /**
     * The configured intent set narrowed by the request's {@code applicable_intents} (BR-17).
     *
     * <p>Prebid's {@code BidRequest} has no {@code applicable_intents} field -- it is an ARTF
     * concept -- so it travels in {@code ext.prebid.artf.applicable_intents} when a scenario
     * sets one. Absent, the configured set is used unnarrowed, which is the correct reading of
     * "narrowed by": nothing to narrow with means no narrowing.
     */
    private List<String> effectiveIntents(BidRequest bidRequest) {
        final Set<String> configured = new LinkedHashSet<>(properties.getIntents());
        final Set<String> applicable = requestApplicableIntents(bidRequest);

        if (applicable.isEmpty()) {
            return List.copyOf(configured);
        }
        configured.retainAll(applicable);
        return List.copyOf(configured);
    }

    private Set<String> requestApplicableIntents(BidRequest bidRequest) {
        try {
            if (bidRequest.getExt() == null) {
                return Set.of();
            }
            final var node = mapper.valueToTree(bidRequest.getExt())
                    .path("prebid").path("artf").path("applicable_intents");
            if (!node.isArray()) {
                return Set.of();
            }
            final Set<String> result = new LinkedHashSet<>();
            node.forEach(element -> {
                if (element.isTextual()) {
                    result.add(element.textValue());
                } else if (element.isInt()) {
                    result.add(Intent.fromWireValue(element.intValue()).wireName());
                }
            });
            return result;
        } catch (RuntimeException e) {
            // A malformed ext must not decide the intent set by accident. Treated as absent,
            // which means no narrowing rather than an empty set -- an empty set would ask for
            // nothing and look identical to a healthy call that proposed nothing.
            return Set.of();
        }
    }

    /**
     * One pass: the request to proceed with, what happened, and the dispositions if any
     * mutations were applied.
     *
     * @param applicationResult null for the four outcomes that leave the request unchanged
     */
    public record Pass(BidRequest bidRequest,
                       ExtensionPointOutcome outcome,
                       ApplicationResult applicationResult,
                       long consecutiveFailures) {

        static Pass unchanged(BidRequest bidRequest, ExtensionPointOutcome outcome, long consecutiveFailures) {
            return new Pass(bidRequest, outcome, null, consecutiveFailures);
        }

        /** Whether the request actually changed, and so whether the hook should report an update. */
        public boolean mutated() {
            return applicationResult != null && applicationResult.anyApplied();
        }
    }
}
