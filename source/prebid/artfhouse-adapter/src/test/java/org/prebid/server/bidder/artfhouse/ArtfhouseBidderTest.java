package org.prebid.server.bidder.artfhouse;

import com.iab.openrtb.request.BidRequest;
import com.iab.openrtb.request.Imp;
import io.vertx.core.http.HttpMethod;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.prebid.server.bidder.model.BidderBid;
import org.prebid.server.bidder.model.BidderCall;
import org.prebid.server.bidder.model.HttpRequest;
import org.prebid.server.bidder.model.HttpResponse;
import org.prebid.server.bidder.model.Result;
import org.prebid.server.hooks.modules.artf.client.TokenCache;
import org.prebid.server.json.JacksonMapper;
import org.prebid.server.json.ObjectMapperProvider;
import org.prebid.server.proto.openrtb.ext.response.BidType;

import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The adapter: request construction, response mapping, and the failure modes that must
 * NOT be reported as demand decisions.
 *
 * <p>On the absence of a property-testing framework and on these tests not running during
 * a deployment, see the header of
 * {@code org.prebid.server.hooks.modules.artf.core.CallBudgetCalculatorTest}. The same two
 * constraints apply: no JDK on the machine this was written on, and the upstream image
 * build runs Maven with {@code -Dmaven.test.skip}.
 */
class ArtfhouseBidderTest {

    private static final String ENDPOINT = "https://demand.example.com/v1/openrtb2/bid";

    private final JacksonMapper mapper = new JacksonMapper(ObjectMapperProvider.mapper());
    private final Clock clock = Clock.fixed(Instant.parse("2026-09-15T12:00:00Z"), ZoneOffset.UTC);

    private TokenCache tokenCache;
    private ArtfhouseBidder bidder;

    @BeforeEach
    void setUp() {
        tokenCache = new TokenCache(clock);
        tokenCache.store("a-token", 3_600_000L);
        bidder = new ArtfhouseBidder(ENDPOINT, mapper, tokenCache);
    }

    // ------------------------------------------------------- makeHttpRequests

    @Test
    void oneRequestIsMadeToTheConfiguredEndpoint() {
        final Result<List<HttpRequest<BidRequest>>> result = bidder.makeHttpRequests(request("imp-1"));

        assertThat(result.getErrors()).isEmpty();
        assertThat(result.getValue()).hasSize(1);
        assertThat(result.getValue().get(0).getUri()).isEqualTo(ENDPOINT);
        assertThat(result.getValue().get(0).getMethod()).isEqualTo(HttpMethod.POST);
    }

    @Test
    void theBearerTokenIsAttached() {
        final HttpRequest<BidRequest> httpRequest = bidder.makeHttpRequests(request("imp-1")).getValue().get(0);

        assertThat(httpRequest.getHeaders().get("Authorization")).isEqualTo("Bearer a-token");
    }

    @Test
    void withoutACredentialTheCallIsNotMadeAtAll() {
        // NEVER an unauthenticated call: the endpoint is not exempted from authentication,
        // and the authorizer would refuse it anyway. Reporting the missing credential is
        // more useful than reporting the 401 it would cause.
        final ArtfhouseBidder noToken = new ArtfhouseBidder(ENDPOINT, mapper, new TokenCache(clock));

        final Result<List<HttpRequest<BidRequest>>> result = noToken.makeHttpRequests(request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).hasSize(1);
        assertThat(result.getErrors().get(0).getMessage())
                .contains(ArtfhouseBidder.DEMAND_SCOPE)
                .contains("no token held");
    }

    @Test
    void theErrorNeverContainsTheTokenItself() {
        final TokenCache expired = new TokenCache(clock);
        expired.store("super-secret-token", 0L);
        final ArtfhouseBidder withExpired = new ArtfhouseBidder(ENDPOINT, mapper, expired);

        final Result<List<HttpRequest<BidRequest>>> result = withExpired.makeHttpRequests(request("imp-1"));

        assertThat(result.getErrors().get(0).getMessage()).doesNotContain("super-secret-token");
    }

    @Test
    void impressionIdsAreCarriedSoATimeoutIsAttributable() {
        final HttpRequest<BidRequest> httpRequest =
                bidder.makeHttpRequests(request("imp-1", "imp-2")).getValue().get(0);

        assertThat(httpRequest.getImpIds()).containsExactly("imp-1", "imp-2");
    }

    @Test
    void theRequestIsForwardedUnchanged() {
        // Translation only (FR-14). Everything the endpoint needs -- the deals the hook
        // wrote, the floors it adjusted -- is already on the request.
        final BidRequest original = request("imp-1");

        final HttpRequest<BidRequest> httpRequest = bidder.makeHttpRequests(original).getValue().get(0);

        assertThat(httpRequest.getPayload()).isEqualTo(original);
    }

    @Test
    void noCampaignPriceIsWrittenOntoTheRequest() {
        // OPEN-1, settled: the catalog is the single source of truth for price, and
        // mirroring prices onto the request would create a second that eventually
        // disagrees. Asserted so a later "helpful" addition has to argue with a test.
        final HttpRequest<BidRequest> httpRequest = bidder.makeHttpRequests(request("imp-1")).getValue().get(0);
        final String body = new String(httpRequest.getBody());

        assertThat(body).doesNotContain("declaredCpm").doesNotContain("declared_cpm");
    }

    // -------------------------------------------------------------- makeBids

    @Test
    void bidsAreMappedFromTheSingleSeatbid() {
        final Result<List<BidderBid>> result = bidder.makeBids(call(twoBidResponse()), request("imp-1"));

        assertThat(result.getErrors()).isEmpty();
        assertThat(result.getValue()).hasSize(2);
        assertThat(result.getValue()).allSatisfy(bid -> {
            assertThat(bid.getSeat()).isEqualTo("artfhouse");
            assertThat(bid.getType()).isEqualTo(BidType.banner);
            assertThat(bid.getBidCurrency()).isEqualTo("USD");
        });
    }

    @Test
    void theCurrencyComesFromTheResponseRatherThanAnAssumption() {
        final String body = """
                {"id":"a1","cur":"EUR","seatbid":[{"seat":"artfhouse","bid":[
                  {"id":"b1","impid":"imp-1","price":2.5,"crid":"c1","adomain":["x.example"],"w":300,"h":250}]}]}
                """;

        final Result<List<BidderBid>> result = bidder.makeBids(call(body), request("imp-1"));

        assertThat(result.getValue().get(0).getBidCurrency()).isEqualTo("EUR");
    }

    @Test
    void campaignIdentityIsPassedThroughUntouched() {
        // The Theater reads bid.ext.prebid.artf to label a row. OpenRTB's bid object has
        // no campaign field, so rewriting this here would put the contract in two places.
        final Result<List<BidderBid>> result = bidder.makeBids(call(twoBidResponse()), request("imp-1"));

        final var ext = result.getValue().get(0).getBid().getExt();
        assertThat(ext.path("prebid").path("artf").path("campaignId").asText()).isEqualTo("camp-1");
        assertThat(ext.path("prebid").path("artf").path("floorBoundBy").asText()).isEqualTo("deal");
    }

    @Test
    void aDealIdSurvives() {
        final Result<List<BidderBid>> result = bidder.makeBids(call(twoBidResponse()), request("imp-1"));

        assertThat(result.getValue().get(0).getBid().getDealid()).isEqualTo("deal-1");
    }

    @Test
    void noBidsIsNotAnError() {
        // The endpoint OMITS seatbid rather than sending it empty, precisely so this reads
        // as "no campaign wished to offer". That is a demand decision, and reporting it as
        // an error would mark the seat as failing every time the floors did their job.
        final Result<List<BidderBid>> result =
                bidder.makeBids(call("{\"id\":\"a1\",\"cur\":\"USD\"}"), request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).isEmpty();
    }

    @Test
    void a204IsNotAnError() {
        final Result<List<BidderBid>> result = bidder.makeBids(call(204, ""), request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).isEmpty();
    }

    @Test
    void anEmptyBodyIsNotAnError() {
        final Result<List<BidderBid>> result = bidder.makeBids(call(200, ""), request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).isEmpty();
    }

    @Test
    void aNon200IsAnErrorNamingTheStatus() {
        final Result<List<BidderBid>> result = bidder.makeBids(call(403, "denied"), request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).hasSize(1);
        assertThat(result.getErrors().get(0).getMessage()).contains("403");
    }

    @Test
    void aMalformedBodyIsAnErrorRatherThanSilentlyNoBids() {
        // A body this adapter cannot read is a fault, and reporting it as "no offers"
        // would assert something substantive about demand that is not known to be true.
        final Result<List<BidderBid>> result = bidder.makeBids(call("not json at all"), request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).hasSize(1);
        assertThat(result.getErrors().get(0).getMessage()).contains("could not be parsed");
    }

    @Test
    void aSeatbidCarryingNoUsableBidIsReportedRatherThanTreatedAsNoOffers() {
        final Result<List<BidderBid>> result = bidder.makeBids(
                call("{\"id\":\"a1\",\"cur\":\"USD\",\"seatbid\":[{\"seat\":\"artfhouse\",\"bid\":[]}]}"),
                request("imp-1"));

        assertThat(result.getValue()).isEmpty();
        assertThat(result.getErrors()).hasSize(1);
        assertThat(result.getErrors().get(0).getMessage()).contains("no usable bids");
    }

    // --------------------------------------------------------------- helpers

    private static BidRequest request(String... impIds) {
        return BidRequest.builder()
                .id("auction-1")
                .cur(List.of("USD"))
                .imp(java.util.Arrays.stream(impIds)
                        .map(id -> Imp.builder().id(id).build())
                        .toList())
                .build();
    }

    /** A response in exactly the shape U2's {@code to_seatbid} produces. */
    private static String twoBidResponse() {
        return """
                {"id":"auction-1","cur":"USD","seatbid":[{"seat":"artfhouse","bid":[
                  {"id":"bid-camp-1","impid":"imp-1","price":6.35,"dealid":"deal-1",
                   "adomain":["one.example"],"crid":"cr-1","w":300,"h":250,
                   "ext":{"prebid":{"artf":{"campaignId":"camp-1","campaignName":"One",
                          "dealId":"deal-1","bindingFloor":2.0,"floorBoundBy":"deal"}}}},
                  {"id":"bid-camp-2","impid":"imp-1","price":3.10,"dealid":"deal-2",
                   "adomain":["two.example"],"crid":"cr-2","w":300,"h":250,
                   "ext":{"prebid":{"artf":{"campaignId":"camp-2","campaignName":"Two",
                          "dealId":"deal-2","bindingFloor":1.0,"floorBoundBy":"impression"}}}}
                ]}]}
                """;
    }

    private BidderCall<BidRequest> call(String body) {
        return call(200, body);
    }

    private BidderCall<BidRequest> call(int status, String body) {
        // HttpResponse.of takes (statusCode, headers, body) -- the field declaration order,
        // which Lombok's staticConstructor follows.
        return BidderCall.succeededHttp(
                HttpRequest.<BidRequest>builder().uri(ENDPOINT).build(),
                HttpResponse.of(status, null, body),
                null);
    }
}
