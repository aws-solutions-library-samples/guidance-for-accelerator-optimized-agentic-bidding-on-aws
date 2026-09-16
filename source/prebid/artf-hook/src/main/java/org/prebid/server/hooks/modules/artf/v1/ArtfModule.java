package org.prebid.server.hooks.modules.artf.v1;

import org.prebid.server.hooks.v1.Hook;
import org.prebid.server.hooks.v1.InvocationContext;
import org.prebid.server.hooks.v1.Module;

import java.util.Collection;
import java.util.List;
import java.util.Objects;

/**
 * The ARTF host module.
 *
 * <p>Registered as a Spring bean; {@code HookCatalog hookCatalog(Collection<Module> modules)}
 * collects every {@link Module} bean, so no separate registry needs editing. Follows the shape
 * of pbs-java's own {@code Ortb2BlockingModule}.
 *
 * <p>This module makes Prebid Server an ARTF <b>host</b>: it introduces no new ARTF intent, and
 * changes nothing about the containers, the models, the training paths, or the orchestrator's
 * public contract (BR-3, NFR-3). It also does not resolve the auction -- no winner, no clearing
 * price; Prebid does that (BR-4, FR-17).
 */
public class ArtfModule implements Module {

    /**
     * The module code. This exact string must appear as {@code module-code} in the hooks
     * execution plan, and as the {@code hooks.<code>.enabled} property prefix.
     */
    public static final String CODE = "artf-orchestrator";

    private final List<? extends Hook<?, ? extends InvocationContext>> hooks;

    public ArtfModule(ArtfProcessedAuctionRequestHook processedAuctionRequestHook) {
        Objects.requireNonNull(processedAuctionRequestHook);
        this.hooks = List.of(processedAuctionRequestHook);
    }

    @Override
    public String code() {
        return CODE;
    }

    @Override
    public Collection<? extends Hook<?, ? extends InvocationContext>> hooks() {
        return hooks;
    }
}
