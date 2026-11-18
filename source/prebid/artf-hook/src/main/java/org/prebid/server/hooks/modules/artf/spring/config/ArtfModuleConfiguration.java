package org.prebid.server.hooks.modules.artf.spring.config;

import io.vertx.core.Vertx;
import org.prebid.server.hooks.modules.artf.client.ArtfExtensionPointClient;
import org.prebid.server.hooks.modules.artf.client.ArtfExtensionPointGrpcClient;
import org.prebid.server.hooks.modules.artf.client.ExtensionPointClient;
import org.prebid.server.hooks.modules.artf.client.ArtfTokenRefresher;
import org.prebid.server.hooks.modules.artf.client.TokenCache;
import org.prebid.server.hooks.modules.artf.core.ArtfMutationApplier;
import org.prebid.server.hooks.modules.artf.core.MutationApplicationService;
import org.prebid.server.hooks.modules.artf.model.ArtfModuleProperties;
import org.prebid.server.hooks.modules.artf.report.ArtfAnalyticsTagWriter;
import org.prebid.server.hooks.modules.artf.v1.ArtfModule;
import org.prebid.server.hooks.modules.artf.v1.ArtfProcessedAuctionRequestHook;
import org.prebid.server.json.ObjectMapperProvider;
import org.prebid.server.vertx.httpclient.HttpClient;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Clock;

/**
 * Spring wiring for the ARTF host module.
 *
 * <p>Gated the way pbs-java's own modules are gated -- see
 * {@code Ortb2BlockingModuleConfiguration}. With
 * {@code hooks.artf-orchestrator.enabled} unset or false, none of these beans exist, so the
 * module is absent rather than present-and-idle.
 *
 * <p>This class lives under {@code org.prebid.server.hooks.modules.artf}, which is inside the
 * {@code @SpringBootApplication} scan base {@code org.prebid.server}
 * ({@code Application.java}). So component scan finds it wherever under that base it sits, and
 * no upstream file in {@code org/prebid/server/spring/config/} needs touching -- which is what
 * keeps the integration additive rather than a fork.
 *
 * <p><b>Enabling the module is not sufficient to invoke it.</b> Prebid does not infer
 * invocation from a module being on the classpath: the hooks execution plan must also name it.
 * That plan lives in {@code prebid-config.yaml}, and omitting it is the one genuinely silent
 * failure in this feature -- the module loads, the server serves auctions, and the hook is
 * never called.
 */
@ConditionalOnProperty(prefix = "hooks." + ArtfModule.CODE, name = "enabled", havingValue = "true")
@Configuration
public class ArtfModuleConfiguration {

    @Bean
    @ConfigurationProperties(prefix = "hooks." + ArtfModule.CODE)
    ArtfModuleProperties artfModuleProperties() {
        return new ArtfModuleProperties();
    }

    @Bean
    TokenCache artfTokenCache(Clock clock) {
        return new TokenCache(clock);
    }

    /**
     * The component that FILLS the cache.
     *
     * <p>Registered as an {@link org.prebid.server.vertx.Initializable}, which
     * {@code InitializationConfiguration} collects into {@code DaemonVerticle} and invokes on an
     * event-loop thread. That is how pbs-java starts its own periodic services
     * ({@code HttpPeriodicRefreshService}, {@code S3PeriodicRefreshService}), so the module needs
     * no scheduler of its own and nothing upstream changes.
     *
     * <p>Without this bean the cache is never written and every ARTF call reports
     * {@code transport_failure / "no valid credential; no token held"} -- correct, honest, and
     * permanently inert. Its absence was invisible to compilation and to the config, and only a
     * live auction surfaced it.
     */
    @Bean
    ArtfTokenRefresher artfTokenRefresher(HttpClient httpClient,
                                          TokenCache artfTokenCache,
                                          Vertx vertx,
                                          ArtfModuleProperties artfModuleProperties) {
        return new ArtfTokenRefresher(
                httpClient,
                artfTokenCache,
                ObjectMapperProvider.mapper(),
                vertx,
                artfModuleProperties.getTokenEndpoint(),
                artfModuleProperties.getScope(),
                artfModuleProperties.getClientId(),
                artfModuleProperties.getClientSecret(),
                artfModuleProperties.getTokenRefreshPeriodMs(),
                artfModuleProperties.getTokenTimeoutMs());
    }

    @Bean
    ArtfMutationApplier artfMutationApplier() {
        return new ArtfMutationApplier(ObjectMapperProvider.mapper());
    }

    /**
     * The extension-point client, chosen by {@code hooks.artf-orchestrator.transport}.
     *
     * <p>{@code http}: Prebid Server's own async HTTP client. {@code grpc}: an HTTP/2 client from
     * the Vert.x core PBS already ships, speaking the gRPC wire format directly (see
     * {@link ArtfExtensionPointGrpcClient} for why no gRPC library). Either way no new
     * dependency enters the build and the module stays on the Vert.x event loop, where a
     * blocking client would stall unrelated auctions (U3-NFR-13, D1).
     *
     * <p>An unknown value fails startup rather than silently picking one: a transport setting
     * that reads "grpc" and runs HTTP would make every measurement under it a lie.
     */
    @Bean
    ExtensionPointClient artfExtensionPointClient(HttpClient httpClient,
                                                  Vertx vertx,
                                                  TokenCache artfTokenCache,
                                                  Clock clock,
                                                  ArtfModuleProperties artfModuleProperties) {
        final String transport = artfModuleProperties.getTransport() == null
                ? "http"
                : artfModuleProperties.getTransport().trim().toLowerCase();
        return switch (transport) {
            case "http" -> new ArtfExtensionPointClient(
                    httpClient,
                    ObjectMapperProvider.mapper(),
                    artfTokenCache,
                    clock,
                    artfModuleProperties.getExtensionPointUrl());
            case "grpc" -> {
                if (artfModuleProperties.getGrpcTarget() == null
                        || artfModuleProperties.getGrpcTarget().isBlank()) {
                    throw new IllegalStateException(
                            "hooks.artf-orchestrator.transport is grpc but grpc-target is not set");
                }
                yield new ArtfExtensionPointGrpcClient(
                        vertx,
                        ObjectMapperProvider.mapper(),
                        artfTokenCache,
                        clock,
                        artfModuleProperties.getGrpcTarget());
            }
            default -> throw new IllegalStateException(
                    "hooks.artf-orchestrator.transport must be http or grpc, got '" + transport + "'");
        };
    }

    @Bean
    MutationApplicationService artfMutationApplicationService(ExtensionPointClient artfExtensionPointClient,
                                                             ArtfMutationApplier artfMutationApplier,
                                                             ArtfModuleProperties artfModuleProperties) {
        return new MutationApplicationService(
                artfExtensionPointClient,
                artfMutationApplier,
                artfModuleProperties,
                ObjectMapperProvider.mapper());
    }

    @Bean
    ArtfAnalyticsTagWriter artfAnalyticsTagWriter() {
        return new ArtfAnalyticsTagWriter(ObjectMapperProvider.mapper());
    }

    @Bean
    ArtfProcessedAuctionRequestHook artfProcessedAuctionRequestHook(
            MutationApplicationService artfMutationApplicationService,
            ArtfAnalyticsTagWriter artfAnalyticsTagWriter) {
        return new ArtfProcessedAuctionRequestHook(artfMutationApplicationService, artfAnalyticsTagWriter);
    }

    @Bean
    ArtfModule artfModule(ArtfProcessedAuctionRequestHook artfProcessedAuctionRequestHook) {
        return new ArtfModule(artfProcessedAuctionRequestHook);
    }
}
