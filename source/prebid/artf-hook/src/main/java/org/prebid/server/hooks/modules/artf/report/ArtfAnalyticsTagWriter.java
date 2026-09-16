package org.prebid.server.hooks.modules.artf.report;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.prebid.server.hooks.execution.v1.analytics.ActivityImpl;
import org.prebid.server.hooks.execution.v1.analytics.ResultImpl;
import org.prebid.server.hooks.execution.v1.analytics.TagsImpl;
import org.prebid.server.hooks.modules.artf.core.MutationApplicationService;
import org.prebid.server.hooks.modules.artf.model.ApplicationResult;
import org.prebid.server.hooks.modules.artf.model.Disposition;
import org.prebid.server.hooks.modules.artf.model.ExtensionPointOutcome;
import org.prebid.server.hooks.modules.artf.model.Intent;
import org.prebid.server.hooks.v1.analytics.Activity;
import org.prebid.server.hooks.v1.analytics.Result;
import org.prebid.server.hooks.v1.analytics.Tags;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Turns one pass into analytics tags.
 *
 * <p>Five signals per auction (U3-NFR-15, D7): call latency, outcome state, per-mutation
 * disposition counts, <b>per-intent</b> counts, and the consecutive-failure counter -- each
 * carrying the {@code source.tid} correlation id (U3-NFR-16).
 *
 * <p>Per-intent counts are here because they answer the question this feature will actually
 * be asked: <i>which</i> ARTF intent did nothing. An aggregate outcome cannot.
 *
 * <p><b>How these reach the client, verified against pbs-java 3.43.0.</b> Prebid copies hook
 * analytics tags into {@code ext.prebid.modules.trace.stages[].outcomes[].groups[]
 * .invocationResults[].analyticsTags}, and <b>only at verbose trace level</b>
 * ({@code HookDebugInfoEnricher.java:191-192}). The trace level is read straight from
 * {@code ext.prebid.trace} on the request with no account gate
 * ({@code DebugResolver.java:80-82}), so a request carrying {@code "trace": "verbose"} gets
 * them and one without gets none at all.
 *
 * <p>That asymmetry matters for the consumer: hook <i>errors and warnings</i> in
 * {@code ext.prebid.modules} additionally require account-gated debug, whereas tags do not.
 * A reader seeing no tags should check the request's trace level before concluding the module
 * is silent.
 *
 * <p><b>No credential value ever appears in a tag</b> (U3-NFR-18, BR-30). The outcome reasons
 * this class copies are written by the client, which names the cache's state rather than the
 * token.
 */
public class ArtfAnalyticsTagWriter {

    /** Activity name. Stable, because a consumer keys on it. */
    public static final String ACTIVITY_EXTENSION_POINT = "artf-extension-point";
    public static final String ACTIVITY_MUTATIONS = "artf-mutations";

    private static final String STATUS_SUCCESS = "success";
    private static final String STATUS_ERROR = "error";

    private final ObjectMapper mapper;

    public ArtfAnalyticsTagWriter(ObjectMapper mapper) {
        this.mapper = Objects.requireNonNull(mapper);
    }

    public Tags write(MutationApplicationService.Pass pass, String correlationId) {
        final List<Activity> activities = new ArrayList<>(2);
        activities.add(callActivity(pass, correlationId));

        final ApplicationResult applied = pass.applicationResult();
        if (applied != null) {
            activities.add(mutationActivity(applied, correlationId));
        }
        return TagsImpl.of(List.copyOf(activities));
    }

    /**
     * The call itself: outcome state, latency, and the failure counter.
     *
     * <p>The activity status is {@code error} only for genuine failures. A skip and a
     * no-mutations answer are reported as {@code success} because neither is a failure
     * (BR-7a) -- marking them otherwise would make a healthy system look degraded and inflate
     * every dashboard built on this signal.
     */
    private Activity callActivity(MutationApplicationService.Pass pass, String correlationId) {
        final ExtensionPointOutcome outcome = pass.outcome();

        final ObjectNode values = mapper.createObjectNode();
        values.put("correlation_id", correlationId);
        values.put("outcome", outcome.state());
        values.put("consecutive_failures", pass.consecutiveFailures());
        values.put("request_mutated", pass.mutated());

        if (outcome.latencyMs() != null) {
            values.put("latency_ms", outcome.latencyMs());
        }

        // Each state contributes the facts only it has. Written as an exhaustive switch over
        // the sealed type, so a sixth state cannot be added without this failing to compile.
        switch (outcome) {
            case ExtensionPointOutcome.MutationsReturned returned -> {
                values.put("mutations_returned", returned.mutations().size());
                putMetadata(values, returned.metadata());
            }
            case ExtensionPointOutcome.NoMutations noMutations -> {
                values.put("mutations_returned", 0);
                putMetadata(values, noMutations.metadata());
            }
            case ExtensionPointOutcome.Timeout timeout -> values.put("budget_ms", timeout.budgetMs());
            case ExtensionPointOutcome.TransportFailure failure -> values.put("reason", failure.reason());
            case ExtensionPointOutcome.SkippedInsufficientBudget skipped -> {
                values.put("remaining_budget_ms", skipped.remainingBudgetMs());
                values.put("minimum_viable_ms", skipped.minimumViableMs());
            }
        }

        final Result result = ResultImpl.of(
                outcome.failure() ? STATUS_ERROR : STATUS_SUCCESS, values, null);

        return ActivityImpl.of(
                ACTIVITY_EXTENSION_POINT,
                outcome.failure() ? STATUS_ERROR : STATUS_SUCCESS,
                List.of(result));
    }

    /**
     * Per-mutation dispositions and per-intent counts.
     *
     * <p>Rejections are listed with their reasons, and the activity status stays
     * {@code success}: a rejection is a normal outcome, not an error (BR-16). Agents propose,
     * the host decides, and reporting a decision as an error would misdescribe the framework's
     * own model.
     */
    private Activity mutationActivity(ApplicationResult applied, String correlationId) {
        final ObjectNode values = mapper.createObjectNode();
        values.put("correlation_id", correlationId);
        values.put("applied", applied.appliedCount());
        values.put("rejected", applied.rejectedCount());

        final ObjectNode byIntentApplied = values.putObject("applied_by_intent");
        final ObjectNode byIntentRejected = values.putObject("rejected_by_intent");
        for (Map.Entry<Intent, long[]> entry : countByIntent(applied.dispositions()).entrySet()) {
            final String name = entry.getKey().wireName();
            if (entry.getValue()[0] > 0) {
                byIntentApplied.put(name, entry.getValue()[0]);
            }
            if (entry.getValue()[1] > 0) {
                byIntentRejected.put(name, entry.getValue()[1]);
            }
        }

        final ArrayNode rejections = values.putArray("rejections");
        for (Disposition disposition : applied.dispositions()) {
            if (!disposition.applied()) {
                final ObjectNode entry = rejections.addObject();
                entry.put("intent", disposition.intent().wireName());
                entry.put("path", disposition.path());
                entry.put("reason", disposition.reason());
            }
        }

        return ActivityImpl.of(ACTIVITY_MUTATIONS, STATUS_SUCCESS,
                List.of(ResultImpl.of(STATUS_SUCCESS, values, null)));
    }

    /** applied and rejected counts per intent, in first-seen order. */
    private static Map<Intent, long[]> countByIntent(List<Disposition> dispositions) {
        final Map<Intent, long[]> counts = new LinkedHashMap<>();
        for (Disposition disposition : dispositions) {
            final long[] pair = counts.computeIfAbsent(disposition.intent(), ignored -> new long[2]);
            if (disposition.applied()) {
                pair[0]++;
            } else {
                pair[1]++;
            }
        }
        return counts;
    }

    /**
     * The orchestrator's own metadata, forwarded rather than interpreted.
     *
     * <p>Per-container status is the orchestrator's vocabulary. Reinterpreting it here would
     * put its fan-out semantics in two places, and the two would drift.
     */
    private void putMetadata(ObjectNode values, org.prebid.server.hooks.modules.artf.model.RtbResponse.Metadata metadata) {
        if (metadata == null) {
            return;
        }
        if (metadata.modelVersion() != null && !metadata.modelVersion().isBlank()) {
            values.put("model_version", metadata.modelVersion());
        }
        if (metadata.containers() == null || metadata.containers().isEmpty()) {
            return;
        }
        final ArrayNode containers = values.putArray("containers");
        metadata.containers().forEach(container -> {
            final ObjectNode node = containers.addObject();
            node.put("name", container.name());
            node.put("status", container.status());
            if (container.latencyMs() != null) {
                node.put("latency_ms", container.latencyMs());
            }
            if (container.displayName() != null && !container.displayName().isBlank()) {
                node.put("display_name", container.displayName());
            }
        });
    }
}
