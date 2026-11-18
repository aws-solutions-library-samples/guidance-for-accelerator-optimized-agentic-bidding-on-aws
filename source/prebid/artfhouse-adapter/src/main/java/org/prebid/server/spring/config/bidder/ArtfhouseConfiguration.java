package org.prebid.server.spring.config.bidder;

import org.prebid.server.bidder.BidderDeps;
import org.prebid.server.bidder.artfhouse.ArtfhouseBidder;
import org.prebid.server.hooks.modules.artf.client.TokenCache;
import org.prebid.server.json.JacksonMapper;
import org.prebid.server.spring.config.bidder.model.BidderConfigurationProperties;
import org.prebid.server.spring.config.bidder.util.BidderDepsAssembler;
import org.prebid.server.spring.env.YamlPropertySourceFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

/**
 * Registers the {@code artfhouse} bidder.
 *
 * <p>Follows the shape every other adapter uses -- see {@code AdfConfiguration}. The name
 * {@code artfhouse} has to agree across FOUR places or the server does not start:
 *
 * <ol>
 *   <li>{@link #BIDDER_NAME} here</li>
 *   <li>the {@code adapters.artfhouse} key in {@code bidder-config/artfhouse.yaml}</li>
 *   <li>the {@code @ConfigurationProperties} prefix below</li>
 *   <li>the filename {@code static/bidder-params/artfhouse.json}</li>
 * </ol>
 *
 * <p>{@code source/prebid/verify_symbols.py} checks all four agree, because a mismatch is a
 * startup failure rather than a runtime one and is therefore worth catching without a
 * deployment.
 *
 * <p><b>The params schema is mandatory.</b> {@code BidderParamValidator.create(bidderCatalog,
 * "static/bidder-params", mapper)} loads one per registered bidder, so a bidder without a
 * schema fails at startup even though it declares no parameters.
 *
 * <p><b>No {@code usersyncerCreator}.</b> Most adapters wire one; this endpoint performs no
 * user syncing -- it matches campaigns on deals and content categories carried on the
 * request and holds no user identity -- so a syncer here would add a cookie surface that
 * syncs nothing.
 */
@Configuration
@PropertySource(
        value = "classpath:/bidder-config/artfhouse.yaml",
        factory = YamlPropertySourceFactory.class)
public class ArtfhouseConfiguration {

    private static final String BIDDER_NAME = "artfhouse";

    @Bean("artfhouseConfigurationProperties")
    @ConfigurationProperties("adapters.artfhouse")
    BidderConfigurationProperties configurationProperties() {
        return new BidderConfigurationProperties();
    }

    /**
     * @param artfTokenCache the credential cache, shared with the ARTF hook module.
     *                       <p>One cache serving two callers is correct here, not a
     *                       compromise. The M2M client in {@code prebid_cfn.yaml} is
     *                       granted BOTH scopes -- {@code artf-orchestrator/mutations:write}
     *                       and {@code artf-demand/bid:read} -- so a single
     *                       {@code client_credentials} token carries both, and the hook and
     *                       this adapter each present the one the endpoint they call
     *                       requires.
     *                       <p>The constraint that follows: the token request must ask for
     *                       both scopes, or omit {@code scope} to receive everything the
     *                       client is granted. A token fetched for one scope alone would
     *                       authenticate at both endpoints and be refused authorization at
     *                       the other -- a 403 that looks like a network problem.
     */
    @Bean
    BidderDeps artfhouseBidderDeps(BidderConfigurationProperties artfhouseConfigurationProperties,
                                   JacksonMapper mapper,
                                   TokenCache artfTokenCache) {
        return BidderDepsAssembler.<BidderConfigurationProperties>forBidder(BIDDER_NAME)
                .withConfig(artfhouseConfigurationProperties)
                .bidderCreator(config -> new ArtfhouseBidder(config.getEndpoint(), mapper, artfTokenCache))
                .assemble();
    }
}
