package org.prebid.server.hooks.modules.artf.client;

import io.vertx.core.Future;
import org.prebid.server.hooks.modules.artf.model.CallBudget;
import org.prebid.server.hooks.modules.artf.model.ExtensionPointOutcome;
import org.prebid.server.hooks.modules.artf.model.RtbRequest;

/**
 * One call to the orchestrator's ARTF extension point, whatever the transport.
 *
 * <p>Two implementations: {@link ArtfExtensionPointClient} (HTTP/1.1 {@code POST /v1/mutations})
 * and {@link ArtfExtensionPointGrpcClient} ({@code RTBExtensionPoint/GetMutations} over gRPC).
 * {@code hooks.artf-orchestrator.transport} selects one at startup; the rest of the module sees
 * only this contract, so the two differ in nothing but the wire.
 */
public interface ExtensionPointClient {

    /**
     * Make one call.
     *
     * @return a Future that always SUCCEEDS, carrying one of the five outcomes
     */
    Future<ExtensionPointOutcome> fetchMutations(RtbRequest request, CallBudget budget);
}
