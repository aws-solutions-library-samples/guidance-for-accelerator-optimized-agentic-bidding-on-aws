package org.prebid.server.hooks.modules.artf.model;

import java.util.List;

/**
 * The module's entire configuration surface.
 *
 * <p><b>No model, framework or container name appears here</b> (BR-18, FR-5). That is what
 * makes this a general ARTF host module rather than a bespoke client for one repository's
 * containers -- the difference between something another platform could adopt and something
 * only this demonstration can use. A field named after a model would quietly end that.
 *
 * <p>Mutable with getters and setters because Spring Boot's relaxed binding populates it;
 * {@code hooks.artf-orchestrator.extension-point-url} binds to {@code extensionPointUrl}.
 */
public class ArtfModuleProperties {

    /** The orchestrator's mutations endpoint. In-cluster service DNS in this deployment. */
    private String extensionPointUrl;

    /**
     * Which transport reaches the extension point: {@code http} ({@code POST} to
     * {@code extensionPointUrl}) or {@code grpc} ({@code RTBExtensionPoint/GetMutations} at
     * {@code grpcTarget}). The default is {@code http} until a measured run shows gRPC
     * beats it on this hop; the deployment sets it, so a measurement run flips it without
     * rebuilding the image.
     */
    private String transport = "http";

    /** {@code host:port} of the orchestrator's gRPC listener. Required when transport is grpc. */
    private String grpcTarget;

    /** Cognito hosted token endpoint for the {@code client_credentials} grant. */
    private String tokenEndpoint;

    /**
     * The scope the machine credential requests. The orchestrator must REQUIRE it
     * (U3-NFR-19); requesting it without enforcement makes least privilege decorative.
     */
    private String scope;

    /**
     * ARN of the Secrets Manager secret that is the source of record for the credential.
     *
     * <p><b>Provenance only -- this module never reads it.</b> It cannot: the shipped jar has no
     * {@code secretsmanager} AWS SDK module and no {@code sts}, so there is neither a client to
     * call with nor a way to use the pod's IRSA identity, and adding either dependency would
     * mean editing the upstream pom (a fork, U1-NFR-16). The deploy script resolves this ARN and
     * delivers the value as {@link #clientId} and {@link #clientSecret}.
     *
     * <p>Kept because it records WHERE the credential came from, which a reader of the running
     * configuration would otherwise have to guess. Never the value.
     */
    private String credentialSecretArn;

    /**
     * Cognito app client id for the {@code client_credentials} grant.
     *
     * <p>Delivered from the Secrets Manager secret named by {@link #credentialSecretArn} via a
     * Kubernetes Secret, for the reason recorded there.
     */
    private String clientId;

    /**
     * Cognito app client secret. Never logged, never reported, never echoed in a failure
     * reason (U3-NFR-18, BR-30).
     */
    private String clientSecret;

    /**
     * How often to check whether the credential is due for renewal, in milliseconds.
     *
     * <p>Deliberately much shorter than a token's lifetime. This is a CHECK interval, not a
     * fetch interval -- {@link org.prebid.server.hooks.modules.artf.client.TokenCache#dueForRenewal()}
     * gates the actual exchange to the renew-ahead window. A short tick costs nothing and means
     * a FAILED renewal is retried within a minute instead of once per token lifetime.
     */
    private long tokenRefreshPeriodMs = 60_000L;

    /**
     * Timeout for the token exchange, in milliseconds.
     *
     * <p>Unrelated to the auction budget: this call happens on a timer, never on an auction's
     * critical path, so it can afford a normal HTTPS timeout rather than the tens of
     * milliseconds an auction allows.
     */
    private long tokenTimeoutMs = 5_000L;

    /**
     * The ARTF {@code tmax} advertised to the orchestrator, in milliseconds.
     *
     * <p>Default 100 ms, matching the framework's own default
     * ({@code source/shared/artf_types.py:120}). NOT derived per request: deriving it would
     * make container behaviour vary request to request, so a slow container becomes
     * intermittent rather than diagnosable. The budget calculator's cap only ever tightens
     * it (U3-NFR-3).
     */
    private int tmaxMs = 100;

    /**
     * Transport allowance in milliseconds: connection reuse, serialisation and the network
     * hop.
     *
     * <p>A configurable default chosen to be safe, <b>not a measured figure</b>. The
     * orchestrator records {@code latency_ms} in every branch, so this is to be validated
     * against observed latency and adjusted. Presenting it as measured would be a claim this
     * module cannot support.
     */
    private int overheadMs = 20;

    /**
     * Milliseconds reserved for bidder fan-out and auction resolution.
     *
     * <p>Fixed milliseconds, not a percentage (U3-NFR-4, D3): a fixed reserve is inspectable
     * and comparable -- one log line tells you the headroom. A percentage yields different
     * absolute headroom on every request, which makes two deployments, or two moments in one
     * deployment, hard to compare when diagnosing.
     */
    private int reserveMs = 60;

    /**
     * The intents this host asks for, by framework name.
     *
     * <p>Narrowed per request by the request's {@code applicable_intents} (BR-17): the hook
     * asks only for what the scenario calls for.
     *
     * <p><b>{@code BID_SHADE} does not belong here.</b> It addresses a bid in the auction
     * RESPONSE, and this hook runs before the bidder fan-out where no {@code seatbid} exists.
     * Including it produces mutations that can only be rejected. The default set below is
     * the request-side intents.
     */
    private List<String> intents = List.of(
            Intent.ACTIVATE_SEGMENTS.wireName(),
            Intent.ACTIVATE_DEALS.wireName(),
            Intent.SUPPRESS_DEALS.wireName(),
            Intent.ADJUST_DEAL_FLOOR.wireName(),
            Intent.ADJUST_DEAL_MARGIN.wireName(),
            Intent.ADD_METRICS.wireName(),
            Intent.ADD_CIDS.wireName());

    public String getExtensionPointUrl() {
        return extensionPointUrl;
    }

    public String getTransport() {
        return transport;
    }

    public void setTransport(String transport) {
        this.transport = transport;
    }

    public String getGrpcTarget() {
        return grpcTarget;
    }

    public void setGrpcTarget(String grpcTarget) {
        this.grpcTarget = grpcTarget;
    }

    public void setExtensionPointUrl(String extensionPointUrl) {
        this.extensionPointUrl = extensionPointUrl;
    }

    public String getTokenEndpoint() {
        return tokenEndpoint;
    }

    public void setTokenEndpoint(String tokenEndpoint) {
        this.tokenEndpoint = tokenEndpoint;
    }

    public String getScope() {
        return scope;
    }

    public void setScope(String scope) {
        this.scope = scope;
    }

    public String getCredentialSecretArn() {
        return credentialSecretArn;
    }

    public void setCredentialSecretArn(String credentialSecretArn) {
        this.credentialSecretArn = credentialSecretArn;
    }

    public String getClientId() {
        return clientId;
    }

    public void setClientId(String clientId) {
        this.clientId = clientId;
    }

    public String getClientSecret() {
        return clientSecret;
    }

    public void setClientSecret(String clientSecret) {
        this.clientSecret = clientSecret;
    }

    public long getTokenRefreshPeriodMs() {
        return tokenRefreshPeriodMs;
    }

    public void setTokenRefreshPeriodMs(long tokenRefreshPeriodMs) {
        this.tokenRefreshPeriodMs = tokenRefreshPeriodMs;
    }

    public long getTokenTimeoutMs() {
        return tokenTimeoutMs;
    }

    public void setTokenTimeoutMs(long tokenTimeoutMs) {
        this.tokenTimeoutMs = tokenTimeoutMs;
    }

    public int getTmaxMs() {
        return tmaxMs;
    }

    public void setTmaxMs(int tmaxMs) {
        this.tmaxMs = tmaxMs;
    }

    public int getOverheadMs() {
        return overheadMs;
    }

    public void setOverheadMs(int overheadMs) {
        this.overheadMs = overheadMs;
    }

    public int getReserveMs() {
        return reserveMs;
    }

    public void setReserveMs(int reserveMs) {
        this.reserveMs = reserveMs;
    }

    public List<String> getIntents() {
        return intents;
    }

    public void setIntents(List<String> intents) {
        this.intents = intents == null ? List.of() : List.copyOf(intents);
    }
}
