"""Real per-sample statistical comparison (ComparisonService).

Loads two selected load-test runs' real per-request outcome sample arrays
(source: LoadTestStatus.outcome_samples, captured by
loadtest_instrumentation.py) and calls the real ABEvaluator.evaluate() with
them as control/treatment — never a summary-stat-to-synthetic-sample
expansion (the ABSamples.materialize() demo-scenario pattern), per NFR-2.

Maps to: FR-8 (Story 5, governance-comparison-promotion unit).
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig, ABTestResult


class InsufficientSamplesError(Exception):
    """Raised when a selected run has no real per-sample data to compare."""

    def __init__(self, run_id: str):
        super().__init__(
            f"Load-test run '{run_id}' has no captured outcome samples to compare."
        )
        self.run_id = run_id


@dataclass(frozen=True)
class ComparisonRequest:
    """Identifies the two runs to compare and the model type context."""

    model_type: str
    current_run_id: str
    current_samples: list[float]
    challenger_run_id: str
    challenger_samples: list[float]


def compare(request: ComparisonRequest) -> ABTestResult:
    """Run the real ABEvaluator against two load-test runs' real samples.

    control_data = current_run's outcome_samples, treatment_data =
    challenger_run's outcome_samples — real per-request values, not summary
    statistics. Raises InsufficientSamplesError before ever calling
    ABEvaluator if either run's sample array is empty (never silently
    compares against zero data).
    """
    if not request.current_samples:
        raise InsufficientSamplesError(request.current_run_id)
    if not request.challenger_samples:
        raise InsufficientSamplesError(request.challenger_run_id)

    config = ABTestConfig(
        model_type=request.model_type,
        control_version=request.current_run_id,
        treatment_version=request.challenger_run_id,
        traffic_percentage=100.0,  # N/A for a load-test-derived comparison; both groups are 100% of their own run
        min_samples=1,
        max_duration_hours=0.0,  # N/A — both runs are already complete
        significance_level=0.05,
        primary_metric="revenue_per_bid",
        guardrail_metrics=[],
    )
    evaluator = ABEvaluator(config=config)
    return evaluator.evaluate(
        control_data=request.current_samples,
        treatment_data=request.challenger_samples,
    )
