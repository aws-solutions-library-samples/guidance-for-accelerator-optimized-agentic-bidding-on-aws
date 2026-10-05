package org.prebid.server.hooks.modules.artf.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.vertx.core.Future;
import io.vertx.core.Vertx;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.http.HttpClient;
import io.vertx.core.http.HttpClientOptions;
import io.vertx.core.http.HttpClientResponse;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.http.HttpVersion;
import io.vertx.core.http.RequestOptions;
import org.prebid.server.hooks.modules.artf.model.CallBudget;
import org.prebid.server.hooks.modules.artf.model.ExtensionPointOutcome;
import org.prebid.server.hooks.modules.artf.model.RtbRequest;
import org.prebid.server.hooks.modules.artf.model.RtbResponse;

import java.nio.charset.StandardCharsets;
import java.time.Clock;
import java.util.Objects;

/**
 * Calls the orchestrator's {@code RTBExtensionPoint/GetMutations} over gRPC. Transport only.
 *
 * <p><b>gRPC on Vert.x core, with no gRPC library.</b> The pinned upstream Prebid Server
 * build (3.43.0) has protobuf-java on its classpath but neither grpc-java nor
 * vertx-grpc-client, and this module may add sources but never edit the upstream pom
 * (U1-NFR-16; see deploy_prebid.sh on why the pod cannot read Secrets Manager for the same
 * reason). A unary gRPC call is small enough to carry on the HTTP/2 client Vert.x core
 * already provides:
 *
 * <ul>
 *   <li>HTTP/2 cleartext with prior knowledge ({@code h2c}, no HTTP/1.1 upgrade), {@code POST}
 *       to {@code /com.iabtechlab.bidstream.mutation.services.v1.RTBExtensionPoint/GetMutations};</li>
 *   <li>headers {@code content-type: application/grpc}, {@code te: trailers},
 *       {@code grpc-timeout}, and {@code authorization: Bearer ...} as call metadata;</li>
 *   <li>body: one length-prefixed message -- 1 byte compressed flag (0), 4 byte big-endian
 *       length, then the message -- and the reply framed the same way;</li>
 *   <li>the call's result in the {@code grpc-status} trailer (or header, for a trailers-only
 *       reply).</li>
 * </ul>
 *
 * <p>The message is the same JSON the HTTP client posts (JSON-over-gRPC, matching what the
 * orchestrator's and the containers' generic servicers accept), so the ONLY variable between
 * this client and {@link ArtfExtensionPointClient} is the transport -- which is what a
 * latency comparison between them needs.
 *
 * <p>Runs on the event loop like the HTTP client: a blocking gRPC stub here would stall
 * unrelated auctions (U3-NFR-13). Same contract as the HTTP client otherwise: it never throws
 * into the hook, every failure is one of the five outcomes, no retry, no breaker.
 */
public class ArtfExtensionPointGrpcClient implements ExtensionPointClient {

    static final String METHOD_PATH =
            "/com.iabtechlab.bidstream.mutation.services.v1.RTBExtensionPoint/GetMutations";

    private static final long MAX_RESPONSE_BYTES = 1_048_576L;
    private static final int FRAME_HEADER_BYTES = 5;

    // gRPC status codes this client distinguishes (grpc/codes). Everything else is a
    // transport failure named by number.
    static final int STATUS_OK = 0;
    static final int STATUS_DEADLINE_EXCEEDED = 4;
    static final int STATUS_PERMISSION_DENIED = 7;
    static final int STATUS_UNIMPLEMENTED = 12;
    static final int STATUS_UNAVAILABLE = 14;
    static final int STATUS_UNAUTHENTICATED = 16;

    private final HttpClient http2;
    private final ObjectMapper mapper;
    private final TokenCache tokenCache;
    private final Clock clock;
    private final String host;
    private final int port;

    public ArtfExtensionPointGrpcClient(Vertx vertx,
                                        ObjectMapper mapper,
                                        TokenCache tokenCache,
                                        Clock clock,
                                        String target) {
        this(vertx.createHttpClient(defaultOptions()), mapper, tokenCache, clock, target);
    }

    ArtfExtensionPointGrpcClient(HttpClient http2,
                                 ObjectMapper mapper,
                                 TokenCache tokenCache,
                                 Clock clock,
                                 String target) {
        this.http2 = Objects.requireNonNull(http2);
        this.mapper = Objects.requireNonNull(mapper);
        this.tokenCache = Objects.requireNonNull(tokenCache);
        this.clock = Objects.requireNonNull(clock);
        final String[] hostPort = parseTarget(Objects.requireNonNull(target, "grpc target"));
        this.host = hostPort[0];
        this.port = Integer.parseInt(hostPort[1]);
    }

    /**
     * HTTP/2 cleartext, prior knowledge. A few multiplexed connections rather than one: the
     * orchestrator's gRPC Service is headless, so each connection may land on a different
     * replica, and a stream limit per connection keeps a burst from queueing on one.
     */
    static HttpClientOptions defaultOptions() {
        return new HttpClientOptions()
                .setProtocolVersion(HttpVersion.HTTP_2)
                .setHttp2ClearTextUpgrade(false)
                .setSsl(false)
                .setKeepAlive(true)
                .setHttp2MaxPoolSize(4)
                .setHttp2MultiplexingLimit(16)
                .setHttp2KeepAliveTimeout(60);
    }

    static String[] parseTarget(String target) {
        String t = target.trim();
        if (t.startsWith("dns:///")) {
            t = t.substring("dns:///".length());
        }
        final int colon = t.lastIndexOf(':');
        if (colon <= 0 || colon == t.length() - 1) {
            throw new IllegalArgumentException(
                    "grpc-target must be host:port, got '" + target + "'");
        }
        return new String[] {t.substring(0, colon), t.substring(colon + 1)};
    }

    @Override
    public Future<ExtensionPointOutcome> fetchMutations(RtbRequest request, CallBudget budget) {
        final long startedAt = clock.millis();

        final String token = tokenCache.current().orElse(null);
        if (token == null) {
            final String detail = tokenCache.lastFailure()
                    .map(f -> "no valid credential (%s); %s".formatted(f, tokenCache.describe()))
                    .orElseGet(() -> "no valid credential; " + tokenCache.describe());
            return Future.succeededFuture(new ExtensionPointOutcome.TransportFailure(detail, 0L));
        }

        final byte[] message;
        try {
            message = mapper.writeValueAsBytes(request);
        } catch (Exception e) {
            return Future.succeededFuture(new ExtensionPointOutcome.TransportFailure(
                    "could not serialise the ARTF envelope: " + e.getClass().getSimpleName(), 0L));
        }

        final RequestOptions options = new RequestOptions()
                .setMethod(HttpMethod.POST)
                .setHost(host)
                .setPort(port)
                .setURI(METHOD_PATH)
                .setSsl(false)
                // The whole call, like the HTTP client's timeout argument.
                .setTimeout(budget.httpTimeoutMs());

        return http2.request(options)
                .compose(req -> {
                    req.putHeader("content-type", "application/grpc")
                            .putHeader("te", "trailers")
                            // Deadline the server may enforce; "m" = milliseconds.
                            .putHeader("grpc-timeout", budget.httpTimeoutMs() + "m")
                            // Call metadata. The scope the token must carry is
                            // artf-orchestrator/mutations:write (BR-27, FR-39).
                            .putHeader("authorization", "Bearer " + token);
                    return req.send(frame(message));
                })
                .compose(response -> response.body().map(body -> interpret(response, body, startedAt, budget)))
                .otherwise(error -> failureFrom(error, startedAt, budget));
    }

    /** Length-Prefixed-Message: compressed flag, big-endian length, payload. */
    static Buffer frame(byte[] message) {
        return Buffer.buffer(FRAME_HEADER_BYTES + message.length)
                .appendByte((byte) 0)
                .appendInt(message.length)
                .appendBytes(message);
    }

    /** The first message in a framed body, or null if the body carries none. */
    static byte[] unframe(Buffer body) {
        if (body == null || body.length() < FRAME_HEADER_BYTES) {
            return null;
        }
        final int length = body.getInt(1);
        if (length < 0 || FRAME_HEADER_BYTES + length > body.length()) {
            throw new IllegalStateException("gRPC frame length %d exceeds body of %d bytes"
                    .formatted(length, body.length()));
        }
        return body.getBytes(FRAME_HEADER_BYTES, FRAME_HEADER_BYTES + length);
    }

    private ExtensionPointOutcome interpret(HttpClientResponse response, Buffer body, long startedAt,
                                            CallBudget budget) {
        final long latency = clock.millis() - startedAt;

        if (response.statusCode() != 200) {
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator gRPC endpoint returned HTTP %d".formatted(response.statusCode()), latency);
        }
        if (body != null && body.length() > MAX_RESPONSE_BYTES) {
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator reply exceeded %d bytes".formatted(MAX_RESPONSE_BYTES), latency);
        }

        // Trailers carry the result; a trailers-only reply puts them in the headers.
        String status = response.trailers().get("grpc-status");
        String grpcMessage = response.trailers().get("grpc-message");
        if (status == null) {
            status = response.getHeader("grpc-status");
            grpcMessage = response.getHeader("grpc-message");
        }
        if (status == null) {
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator reply carried no grpc-status", latency);
        }
        final int code;
        try {
            code = Integer.parseInt(status.trim());
        } catch (NumberFormatException e) {
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator reply carried an unreadable grpc-status '%s'".formatted(status), latency);
        }
        if (code != STATUS_OK) {
            return statusFailure(code, grpcMessage, latency, budget);
        }

        final byte[] message;
        try {
            message = unframe(body);
        } catch (IllegalStateException e) {
            return new ExtensionPointOutcome.TransportFailure(e.getMessage(), latency);
        }
        if (message == null) {
            return new ExtensionPointOutcome.TransportFailure("orchestrator returned an empty body", latency);
        }

        final RtbResponse decoded;
        try {
            decoded = mapper.readValue(message, RtbResponse.class);
        } catch (Exception e) {
            return new ExtensionPointOutcome.TransportFailure(
                    "orchestrator response could not be parsed: " + e.getClass().getSimpleName(), latency);
        }
        if (decoded == null) {
            return new ExtensionPointOutcome.TransportFailure("orchestrator returned an empty body", latency);
        }

        // An answer with nothing in it is NOT a failure (BR-6), same as the HTTP client.
        return decoded.mutationsOrEmpty().isEmpty()
                ? new ExtensionPointOutcome.NoMutations(latency, decoded.metadata())
                : new ExtensionPointOutcome.MutationsReturned(
                        decoded.mutationsOrEmpty(), latency, decoded.metadata());
    }

    /**
     * A non-OK gRPC status, named so the reader is sent to the right place -- the credential
     * for UNAUTHENTICATED / PERMISSION_DENIED (the HTTP client's 401 / 403), the budget for
     * DEADLINE_EXCEEDED, the deployment for UNAVAILABLE / UNIMPLEMENTED.
     */
    private static ExtensionPointOutcome statusFailure(int code, String grpcMessage, long latency,
                                                       CallBudget budget) {
        final String detail = grpcMessage == null || grpcMessage.isBlank()
                ? ""
                : ": " + percentDecode(grpcMessage);
        return switch (code) {
            case STATUS_DEADLINE_EXCEEDED -> new ExtensionPointOutcome.Timeout(latency, budget.httpTimeoutMs());
            case STATUS_UNAUTHENTICATED -> new ExtensionPointOutcome.TransportFailure(
                    "orchestrator returned grpc UNAUTHENTICATED (credential rejected; check the token "
                            + "and its issuer)" + detail, latency);
            case STATUS_PERMISSION_DENIED -> new ExtensionPointOutcome.TransportFailure(
                    "orchestrator returned grpc PERMISSION_DENIED (credential accepted but not authorised; "
                            + "check the artf-orchestrator/mutations:write scope)" + detail, latency);
            case STATUS_UNAVAILABLE -> new ExtensionPointOutcome.TransportFailure(
                    "orchestrator returned grpc UNAVAILABLE" + detail, latency);
            case STATUS_UNIMPLEMENTED -> new ExtensionPointOutcome.TransportFailure(
                    "orchestrator returned grpc UNIMPLEMENTED (no RTBExtensionPoint/GetMutations on "
                            + "this target; is the gRPC server enabled?)" + detail, latency);
            default -> new ExtensionPointOutcome.TransportFailure(
                    "orchestrator returned grpc status %d".formatted(code) + detail, latency);
        };
    }

    /** grpc-message is percent-encoded; decode what we can and keep the rest readable. */
    static String percentDecode(String s) {
        try {
            return java.net.URLDecoder.decode(s.replace("+", "%2B"), StandardCharsets.UTF_8);
        } catch (IllegalArgumentException e) {
            return s;
        }
    }

    private ExtensionPointOutcome failureFrom(Throwable error, long startedAt, CallBudget budget) {
        final long latency = clock.millis() - startedAt;
        if (ArtfExtensionPointClient.isTimeout(error)) {
            return new ExtensionPointOutcome.Timeout(latency, budget.httpTimeoutMs());
        }
        final String message = error.getMessage() == null
                ? error.getClass().getSimpleName()
                : "%s: %s".formatted(error.getClass().getSimpleName(), error.getMessage());
        return new ExtensionPointOutcome.TransportFailure(message, latency);
    }
}
