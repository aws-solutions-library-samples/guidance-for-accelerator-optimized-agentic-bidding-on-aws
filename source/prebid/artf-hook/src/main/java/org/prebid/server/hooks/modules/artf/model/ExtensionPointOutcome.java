package org.prebid.server.hooks.modules.artf.model;

import java.util.List;

/**
 * What one pass of the extension point produced. Closed over FIVE states.
 *
 * <p>Sealed, not a nullable list. Four of the five leave the bid request unchanged, and
 * they must produce four DIFFERENT reports: collapsing them into "no change" would make a
 * broken orchestrator indistinguishable from a healthy one with nothing to propose, and
 * the demonstration could then not tell working from silently failing (BR-5, BR-6, FR-7).
 *
 * <p>Sealing it means a new state cannot be added without every {@code switch} over it
 * failing to compile -- which is the point. The fifth state was itself added late, during
 * NFR Requirements, and an exhaustive switch is what stops the sixth being handled in some
 * places and not others.
 *
 * <table>
 *   <caption>Outcomes</caption>
 *   <tr><th>Outcome</th><th>Request</th><th>Meaning</th></tr>
 *   <tr><td>{@link MutationsReturned}</td><td>Mutated</td>
 *       <td>The orchestrator had something to say and said it</td></tr>
 *   <tr><td>{@link NoMutations}</td><td>Unchanged</td>
 *       <td>The orchestrator answered and proposed nothing</td></tr>
 *   <tr><td>{@link Timeout}</td><td>Unchanged</td>
 *       <td>The orchestrator did not answer within the budget</td></tr>
 *   <tr><td>{@link TransportFailure}</td><td>Unchanged</td>
 *       <td>The orchestrator could not be reached</td></tr>
 *   <tr><td>{@link SkippedInsufficientBudget}</td><td>Unchanged</td>
 *       <td>Too little time remained to make a call worth making -- never asked</td></tr>
 * </table>
 */
public sealed interface ExtensionPointOutcome {

    /** Stable identifier for reporting. One value per state, never derived from a class name. */
    String state();

    /**
     * Whether this outcome represents a failure.
     *
     * <p>A SKIP IS NOT A FAILURE (BR-7a, U3-NFR-9). Nothing was asked, so nothing failed.
     * Neither is {@link NoMutations}: the orchestrator answered, it simply had nothing to
     * propose. Getting this wrong would inflate the consecutive-failure counter with
     * non-events and make a healthy system look degraded.
     */
    boolean failure();

    /** Observed call latency in milliseconds, or null when no call was made. */
    Long latencyMs();

    record MutationsReturned(List<ArtfMutation> mutations, Long latencyMs, RtbResponse.Metadata metadata)
            implements ExtensionPointOutcome {

        @Override
        public String state() {
            return "mutations_returned";
        }

        @Override
        public boolean failure() {
            return false;
        }
    }

    record NoMutations(Long latencyMs, RtbResponse.Metadata metadata) implements ExtensionPointOutcome {

        @Override
        public String state() {
            return "no_mutations";
        }

        @Override
        public boolean failure() {
            return false;
        }
    }

    record Timeout(Long latencyMs, long budgetMs) implements ExtensionPointOutcome {

        @Override
        public String state() {
            return "timeout";
        }

        @Override
        public boolean failure() {
            return true;
        }
    }

    /**
     * The call could not be made or completed. Includes token acquisition failure
     * (U3-NFR-10): the call could not be made, which is exactly what a transport failure
     * is.
     *
     * <p>{@code reason} is a short description for the report. It must never contain a
     * credential value (U3-NFR-18, BR-30).
     */
    record TransportFailure(String reason, Long latencyMs) implements ExtensionPointOutcome {

        @Override
        public String state() {
            return "transport_failure";
        }

        @Override
        public boolean failure() {
            return true;
        }
    }

    /**
     * The remaining auction budget minus the reserve fell below a call worth making, so
     * no call was made.
     *
     * <p>Distinct from {@link Timeout} and {@link NoMutations} on purpose. Reporting it as
     * a timeout would describe an event that did not occur; reporting it as no-mutations
     * would claim the orchestrator answered when it was never asked. Either would put a
     * false statement into the channel FR-7 exists to keep honest.
     */
    record SkippedInsufficientBudget(long remainingBudgetMs, long minimumViableMs)
            implements ExtensionPointOutcome {

        @Override
        public String state() {
            return "skipped_insufficient_budget";
        }

        @Override
        public boolean failure() {
            return false;
        }

        @Override
        public Long latencyMs() {
            return null;
        }
    }
}
