package org.prebid.server.hooks.modules.artf.client;

import java.time.Clock;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Holds a valid credential token so no auction ever waits on token acquisition.
 *
 * <p><b>Why this is a correctness requirement, not an optimisation</b> (U3-NFR-17, BR-29):
 * a token fetch per auction puts an external HTTPS round trip on a path budgeted in tens of
 * milliseconds. That breaches NFR-1 outright -- the auction would be spending its budget on
 * authentication rather than on enrichment.
 *
 * <p>The read path is synchronous and does no I/O: {@link #current()} returns whatever is in
 * memory. Renewal happens elsewhere, on the framework's periodic timer, via
 * {@link #store(String, long)}. That split is deliberate -- a cache that fetched on read
 * would block a Vert.x event-loop thread (U3-NFR-13) and stall unrelated auctions.
 *
 * <p>On renewal failure the existing token keeps being served while it remains valid. When no
 * valid token exists the caller reports a <b>transport failure</b> (U3-NFR-10) rather than
 * waiting: the call could not be made, which is exactly what a transport failure is.
 *
 * <p>Scope is in-memory, per pod. Two pods holding separate tokens is fine -- they are
 * independent clients of the same machine identity, and Cognito issues per-request tokens
 * rather than a single shared session.
 *
 * <p>No method here returns or logs a token value in any diagnostic form (U3-NFR-18, BR-30).
 * {@link #describe()} exists so state can be reported without the secret.
 */
public class TokenCache {

    /**
     * How long before expiry a token is treated as due for renewal. A token that expires
     * mid-flight would fail a call that had already committed its budget, so it is retired
     * early rather than used to the last second.
     */
    private static final long RENEW_AHEAD_MS = 300_000L;

    private final Clock clock;
    private final AtomicReference<Entry> entry = new AtomicReference<>(null);
    private final AtomicReference<String> lastFailure = new AtomicReference<>(null);

    public TokenCache(Clock clock) {
        this.clock = Objects.requireNonNull(clock);
    }

    /**
     * The current token, if one is held and still valid.
     *
     * <p>Synchronous, non-blocking, no I/O, no lock held across anything. Empty means the
     * caller must report a transport failure -- never wait.
     */
    public Optional<String> current() {
        final Entry held = entry.get();
        if (held == null) {
            return Optional.empty();
        }
        if (clock.millis() >= held.expiresAtMs()) {
            return Optional.empty();
        }
        return Optional.of(held.token());
    }

    /**
     * Whether the held token is close enough to expiry that renewal should be attempted.
     * True when nothing is held.
     */
    public boolean dueForRenewal() {
        final Entry held = entry.get();
        if (held == null) {
            return true;
        }
        return clock.millis() >= held.expiresAtMs() - RENEW_AHEAD_MS;
    }

    /**
     * Replace the held token.
     *
     * @param token       the token value; never logged
     * @param expiresInMs lifetime from now, as the token endpoint reported it
     */
    public void store(String token, long expiresInMs) {
        Objects.requireNonNull(token, "token");
        entry.set(new Entry(token, clock.millis() + Math.max(expiresInMs, 0L)));
        lastFailure.set(null);
    }

    /**
     * Record a renewal failure. The existing token is deliberately NOT cleared: while it
     * remains valid it is still usable, and discarding it would turn a recoverable renewal
     * problem into an outage.
     *
     * @param reason a short description. Must not contain a credential value.
     */
    public void recordFailure(String reason) {
        lastFailure.set(reason);
    }

    /** The last renewal failure, if the most recent renewal did not succeed. */
    public Optional<String> lastFailure() {
        return Optional.ofNullable(lastFailure.get());
    }

    /**
     * State for logs and reports, with no token value in it: whether a token is held, and
     * how many milliseconds of life it has left.
     */
    public String describe() {
        final Entry held = entry.get();
        if (held == null) {
            return "no token held";
        }
        final long remaining = held.expiresAtMs() - clock.millis();
        return remaining > 0
                ? "token held, %d ms remaining".formatted(remaining)
                : "token held but expired";
    }

    private record Entry(String token, long expiresAtMs) {
    }
}
