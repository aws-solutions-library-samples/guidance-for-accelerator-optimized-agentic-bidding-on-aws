package org.prebid.server.hooks.modules.artf.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * One mutation, as the orchestrator sends it.
 *
 * <p>Mirrors {@code Mutation} in {@code source/shared/artf_types.py}: an intent, an
 * operation, a semantic path, and exactly one of four payloads. The payload that is set
 * is determined by the intent; the others are null.
 *
 * <p>{@code @JsonIgnoreProperties(ignoreUnknown = true)} throughout these DTOs is
 * deliberate. The orchestrator's response model grows -- {@code metadata.containers}
 * gained {@code display_name} recently -- and a strict reader would reject an entire
 * response because of a field it did not need. Ignoring unknown fields is what lets an
 * older module keep working against a newer orchestrator.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record ArtfMutation(
        @JsonProperty("intent") int intent,
        @JsonProperty("op") int op,
        @JsonProperty("path") String path,
        @JsonProperty("ids") IdsPayload ids,
        @JsonProperty("adjust_deal") AdjustDealPayload adjustDeal,
        @JsonProperty("adjust_bid") AdjustBidPayload adjustBid,
        @JsonProperty("add_metrics") AddMetricsPayload addMetrics) {

    public Intent intentValue() {
        return Intent.fromWireValue(intent);
    }

    public Operation operationValue() {
        return Operation.fromWireValue(op);
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record IdsPayload(@JsonProperty("id") List<String> id) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Margin(
            @JsonProperty("value") Double value,
            /**
             * 0 = CPM (absolute), 1 = PERCENT (relative). Mirrors
             * {@code MarginCalculationType}. Kept as an int rather than an enum because
             * an unrecognised value must be rejectable with a reason rather than
             * unparseable.
             */
            @JsonProperty("calculation_type") int calculationType) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AdjustDealPayload(
            @JsonProperty("bidfloor") Double bidfloor,
            @JsonProperty("margin") Margin margin) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AdjustBidPayload(@JsonProperty("price") Double price) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ArtfMetric(
            @JsonProperty("type") String type,
            @JsonProperty("value") Double value,
            @JsonProperty("vendor") String vendor) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record AddMetricsPayload(@JsonProperty("metric") List<ArtfMetric> metric) {
    }
}
