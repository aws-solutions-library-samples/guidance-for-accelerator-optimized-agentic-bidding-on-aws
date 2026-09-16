package org.prebid.server.hooks.modules.artf.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.vertx.core.Future;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpHeaders;
import org.prebid.server.hooks.modules.artf.model.CallBudget;
import org.prebid.server.hooks.modules.artf.model.ExtensionPointOutcome;
import org.prebid.server.hooks.modules.artf.model.RtbRequest;
import org.prebid.server.hooks.modules.artf.model.RtbResponse;
import org.prebid.server.vertx.httpclient.HttpClient;
import org.prebid.server.vertx.httpclient.model.HttpClientResponse;

import java.time.Clock;
import java.util.Objects;

/**
 * Calls the orchestrator's {@code POST /v1/mutations}. Transport only.
 *
 * <p><b>Uses Prebid Server's own async client</b> ({@link HttpClient}), for two reasons and
 * in this order of seriousness:
 *
 * <ol>
 *   <li><b>Threading.</b> Prebid Server Java is Vert.x-based, and the hook's {@code call}
 *       returns {@link Future}, so it runs on an event loop. A blocking client here stalls
 *       UNRELATED auctions, and the symptom -- general latency with no obvious link to this
 *       module -- points nowhere near the cause (U3-NFR-13).</li>
 *   <li><b>Packageability.</b> Every dependency this module adds is one a host platform must
 *       accept in order to run it.</li>
 * </ol>
 *
 * <p><b>It never throws into the hook.</b> Every failure becomes one of the five outcomes, so
 * the auction always has something to proceed with (BR-8). It also never calls ARTF containers
 * directly (BR-2): the orchestrator owns fan-out and merge, and duplicating that here would
 * break the framework's ownership model.
 *
 * <p><b>No retry</b> (U3-NFR-6) and <b>no circuit breaker</b> (U3-NFR-11). A retry on a budget
 * measured in tens of milliseconds doubles the worst case and fires exactly when the budget is
 * already spent. A breaker would make the Theater's rendering depend on hidden state, so two
 * identical scenarios could differ because of an earlier unrelated failure.
 */
public class ArtfExtensionPointClient {

    private static final long MAX_RESPONSE_BYTES = 1_048_576L;

    private final HttpClient httpClient;
    private final ObjectMapper mapper;
    private final TokenCache tokenCache;
    private final Clock clock;
    private final String endpointUrl;

    public ArtfExtensionPointClient(HttpClient httpClient,
                                    ObjectMapper mapper,
                                    TokenCache tokenCache,
                                    Clock clock,
                                    String endpointUrl) {
        this.httpClient = Objects.requireNonNull(httpClient);
        this.mapper = Objects.requireNonNull(mapper);
        this.tokenCache = Objects.requireNonNull(tokenCache);
        this.clock = Objects.requireNonNull(clock);
        this.endpointUrl = Objects.requireNonNull(endpointUrl);
    }

    /**
     * Make one call.
     *
     * <p>The budget is decided by the caller; this method does not decide whether to call.
     * A client that sometimes declined would have two responsibilities and a contract
     * reading "makes a call, unless it doesn't" -- the skip decision lives in
     * {@code MutationApplicationService}.
     *
     * @return a Future that always SUCCEEDS, carrying one of the five outcomes
     */
    public Future<ExtensionPointOutcome> fetchMutations(RtbRequest request, CallBudget budget) {
        final long startedAt = clock.millis();

        final String token = tokenCache.current().orElse(null);
        if (token == null) {
            // Token acquisition failure is a TRANSPORT failure (U3-NFR-10): the call could
            // not be made. The reason names the cache's state, never the token.
            final String detail = tokenCache.lastFailure()
                    .map(f -> "no valid credential (%s); %s".formatted(f, tokenCache.describe()))
                    .orElseGet(() -> "no valid credential; " + tokenCache.describe());
            return Future.succeededFuture(new ExtensionPointOutcome.TransportFailure(detail, 0L));
        }

        final String body;
        try {
            body = mapper.writeValueAsString(request);
        } catch (Exception e) {
            return Future.succeededFuture(new ExtensionPointOutcome.TransportFailure(
                    "could not serialise the ARTF envelope: " + e.getClass().getSimpleName(), 0L));
        }

        final MultiMap headers = MultiMap.caseInsensitiveMultiMap()
                .add(HttpHeaders.CONTENT_TYPE, "application/json")
                .add(HttpHeaders.ACCEPT, "application/json")
                // The endpoint is Cognito-protected and is NOT exempted from authentication
                // (BR-27, FR-39). The scope the token must carry is
                // artf-orchestrator/mutations:write.
                .add(HttpHeaders.AUTHORIZATION, "Bearer " + token);

        return httpClient
                .post(endpointUrl, headers, body, budget.httpTimeoutMs())
                .map(response -> interpret(response, startedAt))
                .otherwise(error -> failureFrom(error, startedAt, budget));
    }

    private ExtensionPointOutcome interpret(HttpClientResponse response, long startedAt) {
        final long latency = clock.millis() - startedAt;

        if (response.getStatusCode() != 200) {
            // A 401 or 403 here means the credential was rejected -- most likely the scope
            // is missing. Named in the reason, because "HTTP 403" alone sends the reader to
            // the network rather than to the token.
            final String hint = switch (response.getStatusCode()) {
                case 401 -> " (credential rejected; check the token and its issuer)";
                case 403 -> " (credential accepted but not authorised; check the "
                        + "artf-orchestrator/mutations:write scope)";
                default -> "";
            };
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator returned HTTP %d%s".formatted(response.getStatusCode(), hint), latency);
        }

        final RtbResponse decoded;
        try {
            decoded = mapper.readValue(response.getBody(), RtbResponse.class);
        } catch (Exception e) {
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator response could not be parsed: " + e.getClass().getSimpleName(), latency);
        }
        if (decoded == null) {
            return new ExtensionPointOutcome.TransportFailure("orchestrator returned an empty body", latency);
        }

        // An answer with nothing in it is NOT a failure. The orchestrator was asked and
        // proposed nothing, which is a different fact from being unreachable, and the report
        // has to say so (BR-6).
        return decoded.mutationsOrEmpty().isEmpty()
                ? new ExtensionPointOutcome.NoMutations(latency, decoded.metadata())
                : new ExtensionPointOutcome.MutationsReturned(
                        decoded.mutationsOrEmpty(), latency, decoded.metadata());
    }

    /**
     * Distinguish a timeout from any other transport problem.
     *
     * <p>Worth the effort: a timeout says the orchestrator was reachable but slow, and any
     * other failure says it was not reachable at all. Collapsing them would send the reader
     * to the wrong place.
     */
    private ExtensionPointOutcome failureFrom(Throwable error, long startedAt, CallBudget budget) {
        final long latency = clock.millis() - startedAt;

        if (isTimeout(error)) {
            return new ExtensionPointOutcome.Timeout(latency, budget.httpTimeoutMs());
        }
        final String message = error.getMessage() == null
                ? error.getClass().getSimpleName()
                : "%s: %s".formatted(error.getClass().getSimpleName(), error.getMessage());
        return new ExtensionPointOutcome.TransportFailure(message, latency);
    }

    private static boolean isTimeout(Throwable error) {
        for (Throwable t = error; t != null; t = t.getCause()) {
            if (t instanceof java.util.concurrent.TimeoutException) {
                return true;
            }
            final String message = t.getMessage();
            if (message != null && message.toLowerCase().contains("timeout")) {
                return true;
            }
            if (t.getCause() == t) {
                break;
            }
        }
        return false;
    }
}
