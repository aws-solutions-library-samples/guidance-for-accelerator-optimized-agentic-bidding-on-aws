package org.prebid.server.bidder.artfhouse;

import com.iab.openrtb.request.BidRequest;
import com.iab.openrtb.request.Imp;
import com.iab.openrtb.response.Bid;
import com.iab.openrtb.response.BidResponse;
import com.iab.openrtb.response.SeatBid;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpHeaders;
import io.vertx.core.http.HttpMethod;
import org.prebid.server.bidder.Bidder;
import org.prebid.server.bidder.model.BidderBid;
import org.prebid.server.bidder.model.BidderCall;
import org.prebid.server.bidder.model.BidderError;
import org.prebid.server.bidder.model.HttpRequest;
import org.prebid.server.bidder.model.Result;
import org.prebid.server.hooks.modules.artf.client.TokenCache;
import org.prebid.server.json.JacksonMapper;
import org.prebid.server.proto.openrtb.ext.response.BidType;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The {@code artfhouse} bid adapter. TRANSLATION ONLY (FR-14).
 *
 * <p>Campaign selection, deal matching and pricing all live in the demand endpoint. This
 * class decides nothing about demand: it forwards the enriched bid request, and maps the
 * response back into {@link BidderBid}s. An adapter that decided anything would duplicate
 * the endpoint and make the party separation the demonstration exists to show untrue.
 *
 * <h2>What the endpoint expects and returns</h2>
 *
 * It speaks OpenRTB in both directions, which is why this adapter is thin. It reads
 * {@code imp.id}, {@code imp.bidfloor}, {@code imp.pmp.deals[]} (including
 * {@code deal.ext.artf.suppressed}) and {@code imp.ext.artf.categories[]}, and returns a
 * complete bid response with a single {@code seatbid} whose seat is {@code artfhouse} --
 * one bid per eligible campaign, no aliases (FR-11).
 *
 * <h2>Prices are NOT mirrored onto the request</h2>
 *
 * FR-15 reads as though each campaign's declared CPM should travel on the request in an
 * {@code imp.ext} field. It does not, and that was settled deliberately (OPEN-1): the
 * catalog is the single source of truth for price, and mirroring prices would create a
 * second source of truth for the same fact that would eventually disagree. FR-15's actual
 * purpose -- that {@code bidfloor} keeps meaning a floor and a price is never smuggled
 * into it -- holds precisely because the price never travels on the request at all.
 *
 * <h2>One bid per campaign only survives if the request asks for it</h2>
 *
 * Prebid keeps <b>one</b> bid per impression per seat unless the request carries
 * {@code ext.prebid.multibid} naming this bidder with a {@code maxbids}
 * ({@code BidResponseCreator.limitMultiBid}, default limit 1). This adapter cannot set
 * that -- it is request-side. Without it the response is well formed and shows a single
 * campaign, which looks correct, so the requirement is recorded in
 * {@code prebid_config_template.yaml} rather than left to be discovered.
 *
 * <h2>The endpoint's exclusions cannot cross this contract</h2>
 *
 * {@code ext.artf.excluded[]} -- campaigns considered and never offered -- has no route out
 * of this class. {@code makeBids} returns bids, and {@code CompositeBidderResponse} carries
 * bids, errors, FLEDGE configs and IGI with <b>no channel for a response-level
 * {@code ext}</b>. An excluded campaign has no bid to attach it to.
 *
 * <p>It is not destroyed. Prebid copies each bidder's raw response into
 * {@code ext.debug.httpcalls.artfhouse[].responsebody}, so the block survives verbatim --
 * but only when debug is enabled, and debug <b>is</b> account-gated
 * ({@code BidResponseCreator.java:1047}, {@code DebugResolver.isDebugAllowedByAccount}).
 * That is a stricter gate than the one on hook analytics tags, which need only
 * {@code ext.prebid.trace}.
 *
 * <p>FR-30 depends on those exclusions being renderable, so the consequence is worth
 * stating: read them from the debug block or from the endpoint directly, never from
 * {@code seatbid}, where they will never appear.
 */
public class ArtfhouseBidder implements Bidder<BidRequest> {

    /**
     * The scope the demand endpoint's Cognito authorizer requires. Distinct from the
     * orchestrator's scope: these are two different endpoints with two different
     * authorities, and one credential that opened both would defeat the point of scoping
     * either.
     */
    public static final String DEMAND_SCOPE = "artf-demand/bid:read";

    private final String endpointUrl;
    private final JacksonMapper mapper;
    private final TokenCache tokenCache;

    public ArtfhouseBidder(String endpointUrl, JacksonMapper mapper, TokenCache tokenCache) {
        this.endpointUrl = Objects.requireNonNull(endpointUrl);
        this.mapper = Objects.requireNonNull(mapper);
        this.tokenCache = Objects.requireNonNull(tokenCache);
    }

    @Override
    public Result<List<HttpRequest<BidRequest>>> makeHttpRequests(BidRequest request) {
        // Read synchronously from memory. makeHttpRequests is a synchronous method on a
        // Vert.x-based server, so acquiring a token here would block an event-loop
        // thread -- which is why TokenCache serves from memory and renews on a timer.
        final String token = tokenCache.current().orElse(null);
        if (token == null) {
            // NEVER an unauthenticated call. The endpoint is not exempted from
            // authentication (FR-39, SECURITY-08), and a request without the bearer
            // token would be refused by the authorizer anyway -- reporting the missing
            // credential is more useful than reporting the 401 it would cause.
            // The reason names the cache's state, never the token (SECURITY-12).
            return Result.withError(BidderError.failedToRequestBids(
                    "artfhouse: no credential available for scope %s (%s)"
                            .formatted(DEMAND_SCOPE, tokenCache.describe())));
        }

        final MultiMap headers = MultiMap.caseInsensitiveMultiMap()
                .add(HttpHeaders.CONTENT_TYPE, "application/json")
                .add(HttpHeaders.ACCEPT, "application/json")
                .add(HttpHeaders.AUTHORIZATION, "Bearer " + token);

        // The request is forwarded UNCHANGED. Everything the endpoint needs -- the deals
        // the ARTF hook wrote, the floors it adjusted, the categories -- is already on it,
        // and rewriting any of it here would put the contract in two places.
        return Result.withValue(HttpRequest.<BidRequest>builder()
                .method(HttpMethod.POST)
                .uri(endpointUrl)
                .headers(headers)
                .impIds(impIds(request))
                .body(mapper.encodeToBytes(request))
                .payload(request)
                .build());
    }

    @Override
    public Result<List<BidderBid>> makeBids(BidderCall<BidRequest> httpCall, BidRequest bidRequest) {
        final int status = httpCall.getResponse().getStatusCode();

        // 204, or a 200 with nothing in it, means NO CAMPAIGN WISHED TO OFFER. That is a
        // demand decision, not a fault, and reporting it as an error would mark the seat
        // as failing every time the floors did their job.
        if (status == 204) {
            return Result.empty();
        }
        if (status != 200) {
            return Result.withError(BidderError.badServerResponse(
                    "artfhouse: demand endpoint returned HTTP %d".formatted(status)));
        }

        final String body = httpCall.getResponse().getBody();
        if (body == null || body.isBlank()) {
            return Result.empty();
        }

        final BidResponse bidResponse;
        try {
            bidResponse = mapper.decodeValue(body, BidResponse.class);
        } catch (Exception e) {
            return Result.withError(BidderError.badServerResponse(
                    "artfhouse: demand response could not be parsed: " + e.getClass().getSimpleName()));
        }
        if (bidResponse == null || bidResponse.getSeatbid() == null || bidResponse.getSeatbid().isEmpty()) {
            // The endpoint OMITS seatbid rather than sending it empty, precisely so this
            // reads as "no offers" and not as a seat that offered nothing.
            return Result.empty();
        }

        return extractBids(bidResponse);
    }

    /**
     * Map the response's single seatbid to {@link BidderBid}s.
     *
     * <p>The currency comes from the response, not from an assumption. The endpoint states
     * what it prices in and rejects a request asking for anything else, so silently
     * reinterpreting it here could only ever misreport a price.
     */
    private Result<List<BidderBid>> extractBids(BidResponse bidResponse) {
        final List<BidderBid> bids = new ArrayList<>();
        final List<BidderError> errors = new ArrayList<>();
        final String currency = bidResponse.getCur();

        for (SeatBid seatBid : bidResponse.getSeatbid()) {
            if (seatBid == null || seatBid.getBid() == null) {
                continue;
            }
            for (Bid bid : seatBid.getBid()) {
                if (bid == null) {
                    continue;
                }
                // bid.ext.prebid.artf carries the campaign identity, and it is passed
                // through UNTOUCHED. The Theater reads it to label a row, and OpenRTB's
                // bid object has no campaign field, so rewriting or normalising it here
                // would put that contract in two places.
                bids.add(BidderBid.builder()
                        .bid(bid)
                        .seat(seatBid.getSeat())
                        .type(BidType.banner)
                        .bidCurrency(currency)
                        .build());
            }
        }

        // A well-formed response whose seatbid contained no usable bid is worth saying
        // out loud: it is not "no offers", it is a response this adapter could not read.
        if (bids.isEmpty()) {
            errors.add(BidderError.badServerResponse(
                    "artfhouse: response carried a seatbid with no usable bids"));
        }

        return Result.of(bids, errors);
    }

    /**
     * Impression ids this request covers, so Prebid can attribute a timeout to the right
     * impressions rather than to the whole auction.
     */
    private static Set<String> impIds(BidRequest request) {
        final Set<String> ids = new LinkedHashSet<>();
        if (request.getImp() != null) {
            for (Imp imp : request.getImp()) {
                if (imp != null && imp.getId() != null) {
                    ids.add(imp.getId());
                }
            }
        }
        return ids;
    }

}
