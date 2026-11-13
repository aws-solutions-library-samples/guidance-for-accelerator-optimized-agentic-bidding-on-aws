package org.prebid.server.hooks.modules.artf.v1;

import io.vertx.core.Future;
import org.prebid.server.hooks.execution.v1.InvocationResultImpl;
import org.prebid.server.hooks.execution.v1.auction.AuctionRequestPayloadImpl;
import org.prebid.server.hooks.modules.artf.core.MutationApplicationService;
import org.prebid.server.hooks.modules.artf.report.ArtfAnalyticsTagWriter;
import org.prebid.server.hooks.v1.InvocationAction;
import org.prebid.server.hooks.v1.InvocationResult;
import org.prebid.server.hooks.v1.InvocationStatus;
import org.prebid.server.hooks.v1.auction.AuctionInvocationContext;
import org.prebid.server.hooks.v1.auction.AuctionRequestPayload;
import org.prebid.server.hooks.v1.auction.ProcessedAuctionRequestHook;

import java.util.Objects;

/**
 * The ARTF hook, at {@code processed-auction-request}.
 *
 * <p><b>The placement is load-bearing, not conventional</b> (BR-1, FR-2). Mutations must land
 * on the request the bidders will see, and after the bidder fan-out there is no request left
 * to enrich.
 *
 * <p><b>The hook never fails the auction</b> (BR-8, FR-7, NFR-1). On timeout, transport
 * failure, an empty mutation set or a skip, the result carries the unmodified payload and the
 * auction proceeds.
 *
 * <p>The subtlety is what "fail closed" means here. The failure mode to avoid is <b>not</b> a
 * failed auction -- an auction that ran without ARTF enrichment is a perfectly valid auction.
 * It is an <b>unenriched auction presented as enriched</b>. So this hook degrades the
 * enrichment while refusing to degrade the report (BR-9, SECURITY-15): every outcome, including
 * every failure, is reported distinctly through the analytics tags.
 *
 * <p>{@code InvocationStatus.failure} is therefore reserved for a genuine module fault. A
 * failed <i>call</i> is a successful <i>invocation</i> that reports a failed call -- conflating
 * the two would make the module look broken every time the orchestrator was slow.
 */
public class ArtfProcessedAuctionRequestHook implements ProcessedAuctionRequestHook {

    /**
     * The hook implementation code. This exact string must appear as
     * {@code hook-impl-code} in the hooks execution plan.
     *
     * <p>A plan naming a hook that is not registered throws at startup
     * ({@code HookStageExecutor.java:169-177}), so a mismatch here fails loudly rather than
     * silently never invoking the module. That is the safer direction, and it is why naming
     * the hook in the plan is preferable to omitting it.
     */
    public static final String CODE = "artf-orchestrator-processed-auction-request";

    private final MutationApplicationService service;
    private final ArtfAnalyticsTagWriter tagWriter;

    public ArtfProcessedAuctionRequestHook(MutationApplicationService service,
                                          ArtfAnalyticsTagWriter tagWriter) {
        this.service = Objects.requireNonNull(service);
        this.tagWriter = Objects.requireNonNull(tagWriter);
    }

    @Override
    public Future<InvocationResult<AuctionRequestPayload>> call(AuctionRequestPayload payload,
                                                               AuctionInvocationContext context) {

        final var bidRequest = payload.bidRequest();
        final String correlationId = MutationApplicationService.correlationId(bidRequest);

        // Timeout.remaining() is the auction's own remaining budget. The group timeout from
        // the execution plan is not exposed on the invocation context, so it is passed as
        // unknown (0) and the service uses the auction budget alone. Recorded rather than
        // silently assumed equal: if a group timeout tighter than the auction budget is
        // configured, the host will cut the hook short and the budget arithmetic will not
        // have known about it.
        final long remaining = context.timeout() == null ? 0L : context.timeout().remaining();

        // The explicit type witness on map() is load-bearing. Java generics are invariant, so
        // Future<InvocationResultImpl<T>> is NOT a Future<InvocationResult<T>>, and without it
        // inference takes the builder's concrete type from the lambda and the method does not
        // compile. Caught by the first real compile of this class -- javac reported
        // "incompatible types: Future<InvocationResultImpl<AuctionRequestPayload>> cannot be
        // converted to Future<InvocationResult<AuctionRequestPayload>>".
        return service.runOnce(bidRequest, remaining, 0L)
                .<InvocationResult<AuctionRequestPayload>>map(pass -> {
                    final var tags = tagWriter.write(pass, correlationId);

                    if (!pass.mutated()) {
                        // Unchanged request, but the outcome is still reported. This is the
                        // branch that keeps a broken orchestrator distinguishable from a
                        // healthy one with nothing to propose.
                        return InvocationResultImpl.<AuctionRequestPayload>builder()
                                .status(InvocationStatus.success)
                                .action(InvocationAction.no_action)
                                .analyticsTags(tags)
                                .build();
                    }

                    final var mutated = pass.bidRequest();
                    return InvocationResultImpl.<AuctionRequestPayload>builder()
                            .status(InvocationStatus.success)
                            .action(InvocationAction.update)
                            .payloadUpdate(initial -> AuctionRequestPayloadImpl.of(mutated))
                            .analyticsTags(tags)
                            .build();
                })
                .otherwise(error -> InvocationResultImpl.<AuctionRequestPayload>builder()
                        // A fault in this module, not in the call. The auction still proceeds
                        // with the unmodified request -- no_action, never reject.
                        .status(InvocationStatus.failure)
                        .message("ARTF hook failed: " + error.getClass().getSimpleName())
                        .action(InvocationAction.no_action)
                        .build());
    }

    @Override
    public String code() {
        return CODE;
    }
}
