package org.prebid.server.hooks.modules.artf.report;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.iab.openrtb.request.BidRequest;
import org.junit.jupiter.api.Test;
import org.prebid.server.hooks.modules.artf.core.MutationApplicationService;
import org.prebid.server.hooks.modules.artf.model.ApplicationResult;
import org.prebid.server.hooks.modules.artf.model.Disposition;
import org.prebid.server.hooks.modules.artf.model.ExtensionPointOutcome;
import org.prebid.server.hooks.modules.artf.model.Intent;
import org.prebid.server.hooks.v1.analytics.Activity;
import org.prebid.server.hooks.v1.analytics.Tags;
import org.prebid.server.json.ObjectMapperProvider;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The two reporting additions the Auction Theater reads through {@code artf_meta.extension_point}:
 * the orchestrator's answer forwarded verbatim, and one disposition per mutation in order.
 *
 * <p>These tests are not run by the deploy path. The upstream image build runs Maven with
 * {@code -Dmaven.test.skip}; see the header of
 * {@link org.prebid.server.hooks.modules.artf.core.CallBudgetCalculatorTest}.
 */
class ArtfAnalyticsTagWriterTest {

    private final ObjectMapper mapper = ObjectMapperProvider.mapper();
    private final ArtfAnalyticsTagWriter writer = new ArtfAnalyticsTagWriter(mapper);

    private static final String BODY = """
            {"id":"tid-1",
             "mutations":[{"intent":3,"op":1,"path":"imp.0.pmp.deals","ids":{"id":["deal-a"]}},
                          {"intent":5,"op":2,"path":"imp.0.pmp.deals.deal-b","adjust_deal":{"bidfloor":4.5}}],
             "metadata":{"model_version":"orch-1",
                         "containers":[{"name":"ncf-deal-manager","status":"ok","latency_ms":3.0,
                                        "model_version":"ncf:7","mutations":[]}],
                         "stages":[{"name":"deals"}],
                         "conflicts":[]}}
            """;

    @Test
    void forwardsTheOrchestratorAnswerVerbatim() throws Exception {
        final JsonNode response = mapper.readTree(BODY);
        final ExtensionPointOutcome outcome = new ExtensionPointOutcome.NoMutations(12L, null, response);
        final Tags tags = writer.write(pass(outcome, null), "tid-1");

        final ObjectNode values = valuesOf(tags, ArtfAnalyticsTagWriter.ACTIVITY_EXTENSION_POINT);
        assertThat(values.get("response")).isEqualTo(response);
        assertThat(values.get("response").at("/metadata/stages/0/name").asText()).isEqualTo("deals");
    }

    @Test
    void writesOneDispositionPerMutationInOrder() throws Exception {
        final JsonNode response = mapper.readTree(BODY);
        final ApplicationResult applied = new ApplicationResult(BidRequest.builder().build(), List.of(
                Disposition.applied(Intent.fromWireValue(3), "imp.0.pmp.deals"),
                Disposition.rejected(Intent.fromWireValue(5), "imp.0.pmp.deals.deal-b", "deal not found")));
        final ExtensionPointOutcome outcome = new ExtensionPointOutcome.MutationsReturned(
                List.of(), 12L, null, response);
        final Tags tags = writer.write(pass(outcome, applied), "tid-1");

        final ObjectNode values = valuesOf(tags, ArtfAnalyticsTagWriter.ACTIVITY_MUTATIONS);
        final JsonNode dispositions = values.get("dispositions");
        assertThat(dispositions).hasSize(2);
        assertThat(dispositions.get(0).get("index").asInt()).isZero();
        assertThat(dispositions.get(0).get("applied").asBoolean()).isTrue();
        assertThat(dispositions.get(0).has("reason")).isFalse();
        assertThat(dispositions.get(1).get("index").asInt()).isEqualTo(1);
        assertThat(dispositions.get(1).get("applied").asBoolean()).isFalse();
        assertThat(dispositions.get(1).get("reason").asText()).isEqualTo("deal not found");
        // The existing rejections list is kept for verify_auction.py.
        assertThat(values.get("rejections")).hasSize(1);
    }

    private static MutationApplicationService.Pass pass(ExtensionPointOutcome outcome, ApplicationResult applied) {
        return new MutationApplicationService.Pass(BidRequest.builder().build(), outcome, applied, 0L);
    }

    private static ObjectNode valuesOf(Tags tags, String activityName) {
        final Activity activity = tags.activities().stream()
                .filter(candidate -> activityName.equals(candidate.name()))
                .findFirst()
                .orElseThrow();
        return activity.results().get(0).values();
    }
}
