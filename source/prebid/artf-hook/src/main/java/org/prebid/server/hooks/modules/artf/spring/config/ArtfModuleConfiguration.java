package org.prebid.server.hooks.modules.artf.spring.config;

import io.vertx.core.Vertx;
import org.prebid.server.hooks.modules.artf.client.ArtfExtensionPointClient;
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
     * The extension-point client, on Prebid Server's own async HTTP client.
     *
     * <p>No new HTTP dependency: the framework's client keeps the module on the intended
     * Vert.x threading model, and a blocking client here would stall unrelated auctions
     * (U3-NFR-13, D1).
     */
    @Bean
    ArtfExtensionPointClient artfExtensionPointClient(HttpClient httpClient,
                                                     TokenCache artfTokenCache,
                                                     Clock clock,
                                                     ArtfModuleProperties artfModuleProperties) {
        return new ArtfExtensionPointClient(
                httpClient,
                ObjectMapperProvider.mapper(),
                artfTokenCache,
                clock,
                artfModuleProperties.getExtensionPointUrl());
    }

    @Bean
    MutationApplicationService artfMutationApplicationService(ArtfExtensionPointClient artfExtensionPointClient,
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
