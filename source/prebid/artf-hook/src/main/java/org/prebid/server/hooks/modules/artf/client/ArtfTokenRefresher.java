package org.prebid.server.hooks.modules.artf.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.vertx.core.MultiMap;
import io.vertx.core.Promise;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpHeaders;
import org.prebid.server.log.Logger;
import org.prebid.server.log.LoggerFactory;
import org.prebid.server.vertx.Initializable;
import org.prebid.server.vertx.httpclient.HttpClient;
import org.prebid.server.vertx.httpclient.model.HttpClientResponse;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Objects;

/**
 * Fills {@link TokenCache}. Without this, nothing ever does.
 *
 * <p>The cache was built with a deliberately synchronous, non-blocking read path so no auction
 * waits on token acquisition, and its javadoc says renewal happens "elsewhere, on the
 * framework's periodic timer". This is that elsewhere. Until it existed the cache was
 * permanently empty, every call reported {@code transport_failure / "no valid credential; no
 * token held"}, and the ARTF mutation path could not work at all -- honestly reported, and
 * completely inert.
 *
 * <p><b>Why the credential arrives as an id and secret rather than a Secrets Manager ARN.</b>
 * The module cannot read Secrets Manager from inside this JVM. The Prebid Server image is built
 * from the pinned upstream release whose pom declares only the AWS SDK's {@code s3} module: the
 * shipped jar contains {@code s3}, {@code auth}, {@code regions} and {@code apache-client}, and
 * <b>neither {@code secretsmanager} nor {@code sts}</b>. Without {@code secretsmanager} there is
 * no client to call; without {@code sts} the SDK cannot even use the pod's IRSA web identity,
 * which the container states plainly at startup:
 *
 * <pre>To use web identity tokens, the 'sts' service module must be on the class path.</pre>
 *
 * Adding either dependency means editing the upstream pom, which is a fork and is exactly what
 * U1-NFR-16 forbids. So the deploy script reads the secret with credentials it does have and
 * hands the pod a Kubernetes Secret, which arrives here as configuration. The trust boundary is
 * unchanged -- Secrets Manager is still the source of record -- and no AWS SDK is involved.
 *
 * <p><b>Threading.</b> {@link #initialize(Promise)} runs on a Vert.x event-loop thread, so the
 * first fetch is fired and not awaited, and renewal runs on {@code vertx.setPeriodic}. Nothing
 * here blocks; a blocking token fetch would stall unrelated auctions (U3-NFR-13), which is the
 * whole reason the cache splits read from write.
 *
 * <p><b>The pod serves auctions before the first token arrives.</b> That window is real and is
 * reported rather than hidden: the hook returns a transport failure naming the cache's state, and
 * the auction proceeds unmutated. Blocking startup on an external token endpoint would trade a
 * few unenriched auctions for a pod that cannot start when Cognito is slow.
 *
 * <p><b>No credential or token value is logged</b> (U3-NFR-18, BR-30), including on failure.
 * Failure reasons name the status code or the exception type, never the body -- an OAuth2 error
 * response can echo request parameters back.
 */
public class ArtfTokenRefresher implements Initializable {

    private static final Logger logger = LoggerFactory.getLogger(ArtfTokenRefresher.class);

    private final HttpClient httpClient;
    private final TokenCache tokenCache;
    private final ObjectMapper mapper;
    private final Vertx vertx;
    private final String tokenEndpoint;
    private final String scope;
    private final String basicAuthHeader;
    private final long refreshPeriodMs;
    private final long timeoutMs;

    public ArtfTokenRefresher(HttpClient httpClient,
                              TokenCache tokenCache,
                              ObjectMapper mapper,
                              Vertx vertx,
                              String tokenEndpoint,
                              String scope,
                              String clientId,
                              String clientSecret,
                              long refreshPeriodMs,
                              long timeoutMs) {
        this.httpClient = Objects.requireNonNull(httpClient);
        this.tokenCache = Objects.requireNonNull(tokenCache);
        this.mapper = Objects.requireNonNull(mapper);
        this.vertx = Objects.requireNonNull(vertx);
        this.tokenEndpoint = Objects.requireNonNull(tokenEndpoint);
        this.scope = Objects.requireNonNull(scope);
        this.refreshPeriodMs = refreshPeriodMs;
        this.timeoutMs = timeoutMs;

        // Built once. RFC 6749 section 2.3.1 puts the client id and secret in an HTTP Basic
        // header for the client_credentials grant, which is what Cognito's hosted token
        // endpoint expects. Kept in a field so the secret is encoded once rather than on
        // every renewal.
        this.basicAuthHeader = "Basic " + Base64.getEncoder().encodeToString(
                "%s:%s".formatted(
                        Objects.requireNonNull(clientId, "clientId"),
                        Objects.requireNonNull(clientSecret, "clientSecret"))
                        .getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public void initialize(Promise<Void> initializePromise) {
        // Fired, not awaited -- matching HttpPeriodicRefreshService. Startup does not depend
        // on an external endpoint being reachable.
        refresh();

        if (refreshPeriodMs > 0) {
            vertx.setPeriodic(refreshPeriodMs, id -> refreshIfDue());
        } else {
            logger.warn("ARTF token refresh period is {} ms; the token will never be renewed"
                    + " and ARTF calls will stop working when it expires", refreshPeriodMs);
        }
        initializePromise.tryComplete();
    }

    /**
     * Renew only when the cache says it is time.
     *
     * <p>The tick is deliberately shorter than a token's life so a FAILED renewal is retried
     * soon rather than once per token lifetime. {@link TokenCache#dueForRenewal()} is what
     * stops that short tick from fetching a new token every time: it is true only inside the
     * cache's renew-ahead window, or when nothing is held.
     */
    private void refreshIfDue() {
        if (tokenCache.dueForRenewal()) {
            refresh();
        }
    }

    /**
     * One {@code client_credentials} exchange.
     *
     * <p>Never throws and never returns a failed future to the caller: a renewal problem must
     * not become an unhandled error on the event loop. Failures are recorded on the cache,
     * which keeps serving an existing token while it remains valid.
     */
    void refresh() {
        // The scope contains '/' and ':' (artf-orchestrator/mutations:write), both of which
        // must be percent-encoded in an application/x-www-form-urlencoded body. Sent raw, the
        // endpoint reads a different scope than intended and returns invalid_scope.
        final String body = "grant_type=client_credentials&scope="
                + URLEncoder.encode(scope, StandardCharsets.UTF_8);

        final MultiMap headers = MultiMap.caseInsensitiveMultiMap()
                .add(HttpHeaders.CONTENT_TYPE, "application/x-www-form-urlencoded")
                .add(HttpHeaders.ACCEPT, "application/json")
                .add(HttpHeaders.AUTHORIZATION, basicAuthHeader);

        httpClient.post(tokenEndpoint, headers, body, timeoutMs)
                .map(this::store)
                .otherwise(this::recordTransportFailure);
    }

    private Void store(HttpClientResponse response) {
        if (response.getStatusCode() != 200) {
            // Status only. An OAuth2 error body can echo request parameters, and this reason
            // reaches analytics tags.
            final String hint = switch (response.getStatusCode()) {
                case 400 -> " (check the requested scope exists on the resource server)";
                case 401 -> " (client id or secret rejected)";
                default -> "";
            };
            fail("token endpoint returned HTTP %d%s".formatted(response.getStatusCode(), hint));
            return null;
        }

        final JsonNode parsed;
        try {
            parsed = mapper.readTree(response.getBody());
        } catch (Exception e) {
            fail("token response could not be parsed: " + e.getClass().getSimpleName());
            return null;
        }

        final JsonNode token = parsed == null ? null : parsed.get("access_token");
        if (token == null || !token.isTextual() || token.asText().isEmpty()) {
            fail("token response carried no access_token");
            return null;
        }

        // OAuth2 reports expires_in in SECONDS; the cache stores milliseconds. Getting this
        // wrong by 1000x either retires a valid token immediately or serves an expired one.
        final JsonNode expiresIn = parsed.get("expires_in");
        final long lifetimeMs = expiresIn != null && expiresIn.canConvertToLong()
                ? expiresIn.asLong() * 1000L
                // Absent expires_in is not fatal, but it must not be treated as unlimited. A
                // conservative floor means the token is renewed sooner than needed rather
                // than used after it has expired.
                : 300_000L;

        tokenCache.store(token.asText(), lifetimeMs);
        logger.info("ARTF credential renewed for scope {}; {}", scope, tokenCache.describe());
        return null;
    }

    private Void recordTransportFailure(Throwable error) {
        fail(error.getMessage() == null
                ? error.getClass().getSimpleName()
                : "%s: %s".formatted(error.getClass().getSimpleName(), error.getMessage()));
        return null;
    }

    private void fail(String reason) {
        tokenCache.recordFailure(reason);
        // WARN, not ERROR: an existing token keeps being served while valid, so a single
        // failed renewal is not yet an outage. The hook's analytics tags carry the same
        // reason, so the auction-side report and the log agree.
        logger.warn("ARTF credential renewal failed: {}; {}", reason, tokenCache.describe());
    }
}
