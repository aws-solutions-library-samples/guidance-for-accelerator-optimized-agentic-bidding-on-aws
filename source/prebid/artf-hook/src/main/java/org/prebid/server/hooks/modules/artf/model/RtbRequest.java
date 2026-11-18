package org.prebid.server.hooks.modules.artf.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * The ARTF envelope sent to {@code POST /v1/mutations}.
 *
 * <p>Mirrors {@code RTBRequest} in {@code source/shared/artf_types.py}. Field names are
 * the wire names, which are snake_case -- the orchestrator is Python and its model is
 * authoritative here.
 *
 * <p>{@code tmax} is the budget the orchestrator gets for its container fan-out, and it
 * uses the WHOLE value ({@code orchestrator/app.py:471}). It is capped by
 * {@link CallBudget#effectiveTmax()} so it never exceeds what this hook will actually
 * wait for -- advertising more than will be honoured is the failure class
 * {@link org.prebid.server.hooks.modules.artf.core.CallBudgetCalculator} exists to
 * eliminate.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record RtbRequest(
        @JsonProperty("id") String id,
        @JsonProperty("lifecycle") String lifecycle,
        @JsonProperty("tmax") int tmax,
        @JsonProperty("bid_request") Map<String, Object> bidRequest,
        @JsonProperty("originator") Originator originator,
        @JsonProperty("applicable_intents") List<String> applicableIntents,
        @JsonProperty("ext") Map<String, Object> ext) {

    /**
     * The lifecycle this hook operates in. The hook runs at
     * {@code processed-auction-request}, before the bidder fan-out, so what it holds is
     * the publisher's bid request.
     */
    public static final String LIFECYCLE_PUBLISHER_BID_REQUEST = "LIFECYCLE_PUBLISHER_BID_REQUEST";

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record Originator(
            @JsonProperty("type") String type,
            @JsonProperty("id") String id) {
    }
}
