package org.prebid.server.hooks.modules.artf.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * The orchestrator's response from {@code POST /v1/mutations}.
 *
 * <p>Mirrors {@code RTBResponse} in {@code source/shared/artf_types.py}.
 *
 * <p>{@code metadata.containers} is read but not acted on. It carries the orchestrator's
 * own per-container status vocabulary -- {@code ok, no_mutations, unreachable, error,
 * timeout, disabled, skipped} -- which this hook forwards for reporting rather than
 * interpreting. Interpreting it here would put the orchestrator's fan-out semantics in
 * two places.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record RtbResponse(
        @JsonProperty("id") String id,
        @JsonProperty("mutations") List<ArtfMutation> mutations,
        @JsonProperty("metadata") Metadata metadata) {

    /**
     * Mutations, never null. A response with no {@code mutations} key and one with an
     * empty list mean the same thing -- the orchestrator proposed nothing -- and callers
     * should not have to distinguish them.
     */
    public List<ArtfMutation> mutationsOrEmpty() {
        return mutations == null ? List.of() : mutations;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Metadata(
            @JsonProperty("api_version") String apiVersion,
            @JsonProperty("model_version") String modelVersion,
            @JsonProperty("containers") List<ContainerInvocation> containers) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record ContainerInvocation(
            @JsonProperty("name") String name,
            @JsonProperty("status") String status,
            @JsonProperty("latency_ms") Double latencyMs,
            @JsonProperty("model_version") String modelVersion,
            @JsonProperty("display_name") String displayName) {
    }
}
