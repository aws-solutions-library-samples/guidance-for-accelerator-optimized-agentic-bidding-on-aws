package org.prebid.server.hooks.modules.artf.client;

import io.vertx.core.Future;
import io.vertx.core.MultiMap;
import io.vertx.core.Vertx;
import org.junit.jupiter.api.Test;
import org.prebid.server.json.ObjectMapperProvider;
import org.prebid.server.vertx.httpclient.HttpClient;
import org.prebid.server.vertx.httpclient.model.HttpClientResponse;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * The contract of the component that fills {@link TokenCache}.
 *
 * <p>These tests do NOT run here or in the image build: this machine has no JDK or Maven, and
 * the upstream build passes {@code -Dmaven.test.skip}. They are written because they state the
 * contract precisely -- particularly the two unit conversions that are silently wrong if
 * mishandled -- and because the behaviour they describe was verified live against the deployed
 * pod instead.
 */
class ArtfTokenRefresherTest {

    private static final Clock CLOCK = Clock.fixed(Instant.parse("2026-01-01T00:00:00Z"), ZoneOffset.UTC);
    private static final String ENDPOINT = "https://example.auth.us-east-1.amazoncognito.com/oauth2/token";
    private static final String SCOPE = "artf-orchestrator/mutations:write";

    private final List<String> bodies = new ArrayList<>();
    private final List<MultiMap> headers = new ArrayList<>();
    private final TokenCache cache = new TokenCache(CLOCK);

    private ArtfTokenRefresher refresherReturning(int status, String body) {
        final HttpClient httpClient = mock(HttpClient.class);
        final HttpClientResponse response = HttpClientResponse.of(status, MultiMap.caseInsensitiveMultiMap(), body);
        when(httpClient.post(anyString(), any(MultiMap.class), anyString(), anyLong()))
                .thenAnswer(invocation -> {
                    headers.add(invocation.getArgument(1));
                    bodies.add(invocation.getArgument(2));
                    return Future.succeededFuture(response);
                });
        return new ArtfTokenRefresher(httpClient, cache, ObjectMapperProvider.mapper(),
                mock(Vertx.class), ENDPOINT, SCOPE, "the-client", "the-secret", 60_000L, 5_000L);
    }

    @Test
    void storesTheTokenFromASuccessfulExchange() {
        refresherReturning(200, "{\"access_token\":\"abc\",\"expires_in\":3600}").refresh();

        assertThat(cache.current()).contains("abc");
    }

    @Test
    void treatsExpiresInAsSecondsNotMilliseconds() {
        // The one conversion that is silently wrong by 1000x. expires_in 3600 is an hour, so
        // the cache must not report roughly three seconds of life.
        refresherReturning(200, "{\"access_token\":\"abc\",\"expires_in\":3600}").refresh();

        assertThat(cache.describe()).isEqualTo("token held, 3600000 ms remaining");
    }

    @Test
    void percentEncodesTheScopeInTheFormBody() {
        // The scope contains '/' and ':'. Sent raw, Cognito reads a different scope and
        // answers invalid_scope -- a failure that looks like a misconfigured resource server.
        refresherReturning(200, "{\"access_token\":\"abc\",\"expires_in\":3600}").refresh();

        assertThat(bodies).hasSize(1);
        assertThat(bodies.get(0)).isEqualTo(
                "grant_type=client_credentials&scope=artf-orchestrator%2Fmutations%3Awrite");
    }

    @Test
    void sendsTheCredentialAsHttpBasicAndNotInTheBody() {
        refresherReturning(200, "{\"access_token\":\"abc\",\"expires_in\":3600}").refresh();

        // base64("the-client:the-secret")
        assertThat(headers.get(0).get("Authorization")).isEqualTo("Basic dGhlLWNsaWVudDp0aGUtc2VjcmV0");
        assertThat(bodies.get(0)).doesNotContain("the-secret");
    }

    @Test
    void recordsAFailureWithoutTheResponseBodyOnNon200() {
        // An OAuth2 error body can echo request parameters back, and this reason travels into
        // analytics tags on the auction response.
        refresherReturning(400, "{\"error\":\"invalid_scope\",\"hint\":\"the-secret\"}").refresh();

        assertThat(cache.current()).isEmpty();
        assertThat(cache.lastFailure()).isPresent();
        assertThat(cache.lastFailure().get()).contains("HTTP 400").doesNotContain("the-secret");
    }

    @Test
    void recordsAFailureWhenNoAccessTokenIsPresent() {
        refresherReturning(200, "{\"token_type\":\"Bearer\"}").refresh();

        assertThat(cache.current()).isEmpty();
        assertThat(cache.lastFailure().orElseThrow()).contains("no access_token");
    }

    @Test
    void recordsAFailureOnAnUnparsableBody() {
        refresherReturning(200, "not json").refresh();

        assertThat(cache.lastFailure()).isPresent();
        assertThat(cache.current()).isEmpty();
    }

    @Test
    void fallsBackToAConservativeLifetimeWhenExpiresInIsAbsent() {
        // Absent expires_in must not be read as unlimited. Renewing sooner than needed is
        // recoverable; serving an expired token is not.
        refresherReturning(200, "{\"access_token\":\"abc\"}").refresh();

        assertThat(cache.current()).contains("abc");
        assertThat(cache.describe()).isEqualTo("token held, 300000 ms remaining");
    }

    @Test
    void keepsServingAValidTokenAfterAFailedRenewal() {
        // A renewal problem must not become an outage while a usable token is still held.
        cache.store("still-good", 3_600_000L);
        refresherReturning(500, "boom").refresh();

        assertThat(cache.current()).contains("still-good");
        assertThat(cache.lastFailure()).isPresent();
    }
}
