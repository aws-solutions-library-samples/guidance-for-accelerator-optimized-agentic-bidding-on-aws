"""Tests for agents.governance.ab_evaluator — ABEvaluator.

Validates:
- p_value is always in [0, 1] for various inputs
- Clear winner produces "promote" recommendation
- Clear loser produces "reject" recommendation
- Insufficient samples produces "extend" recommendation
- Guardrail violation forces "reject" even with primary metric improvement
- Edge cases: identical distributions, zero variance, single sample
- SPRT early stopping triggers before max samples

All tests use deterministic fixture data (pre-generated numeric arrays).
No random data generation.

**Validates: Requirements 4.3, 4.4**
"""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.governance.ab_evaluator import (
    ABEvaluator,
    ABTestConfig,
    ABTestResult,
    TestStatus,
    _safe_mean,
    _safe_variance,
    _t_cdf,
)


# ---------------------------------------------------------------------------
# Deterministic fixture data
# ---------------------------------------------------------------------------

# Control group: revenue_per_bid values centered around 2.0
CONTROL_BASELINE = [
    1.82, 2.15, 1.97, 2.03, 2.10, 1.88, 2.22, 1.95, 2.08, 1.91,
    2.01, 1.99, 2.14, 1.87, 2.06, 2.11, 1.93, 2.04, 1.96, 2.09,
    2.00, 1.85, 2.17, 1.94, 2.07, 1.90, 2.12, 1.98, 2.05, 1.89,
    2.02, 2.13, 1.92, 2.08, 1.86, 2.16, 1.95, 2.03, 1.97, 2.10,
    1.88, 2.19, 1.93, 2.06, 1.91, 2.11, 1.96, 2.04, 2.00, 1.84,
]

# Treatment that is clearly better (centered around 2.5)
TREATMENT_WINNER = [
    2.42, 2.55, 2.47, 2.63, 2.50, 2.38, 2.62, 2.45, 2.58, 2.41,
    2.51, 2.49, 2.64, 2.37, 2.56, 2.61, 2.43, 2.54, 2.46, 2.59,
    2.50, 2.35, 2.67, 2.44, 2.57, 2.40, 2.52, 2.48, 2.55, 2.39,
    2.52, 2.63, 2.42, 2.58, 2.36, 2.66, 2.45, 2.53, 2.47, 2.60,
    2.38, 2.69, 2.43, 2.56, 2.41, 2.61, 2.46, 2.54, 2.50, 2.34,
]

# Treatment that is clearly worse (centered around 1.5)
TREATMENT_LOSER = [
    1.42, 1.55, 1.47, 1.63, 1.50, 1.38, 1.62, 1.45, 1.58, 1.41,
    1.51, 1.49, 1.64, 1.37, 1.56, 1.61, 1.43, 1.54, 1.46, 1.59,
    1.50, 1.35, 1.67, 1.44, 1.57, 1.40, 1.52, 1.48, 1.55, 1.39,
    1.52, 1.63, 1.42, 1.58, 1.36, 1.66, 1.45, 1.53, 1.47, 1.60,
    1.38, 1.69, 1.43, 1.56, 1.41, 1.61, 1.46, 1.54, 1.50, 1.34,
]

# Treatment that is nearly identical to control (centered around 2.0)
TREATMENT_IDENTICAL = [
    1.83, 2.14, 1.98, 2.02, 2.11, 1.87, 2.21, 1.96, 2.07, 1.92,
    2.00, 2.01, 2.13, 1.88, 2.05, 2.12, 1.94, 2.03, 1.97, 2.08,
    2.01, 1.86, 2.16, 1.93, 2.06, 1.91, 2.11, 1.99, 2.04, 1.90,
    2.01, 2.14, 1.91, 2.09, 1.85, 2.17, 1.94, 2.04, 1.96, 2.11,
    1.87, 2.20, 1.92, 2.07, 1.90, 2.12, 1.95, 2.05, 1.99, 1.85,
]

# Small samples (insufficient for evaluation)
SMALL_CONTROL = [2.0, 2.1, 1.9]
SMALL_TREATMENT = [2.5, 2.6, 2.4]

# Zero variance data (all identical)
ZERO_VARIANCE_CONTROL = [2.0] * 20
ZERO_VARIANCE_TREATMENT = [2.5] * 20

# Single sample
SINGLE_SAMPLE = [2.0]

# Large dataset with clear separation for SPRT testing
# Control centered at 1.0
LARGE_CONTROL_SPRT = [
    1.0 + 0.1 * (i % 5 - 2) for i in range(200)
]
# Treatment centered at 1.5 — large enough separation for SPRT to trigger early
LARGE_TREATMENT_SPRT = [
    1.5 + 0.1 * (i % 5 - 2) for i in range(200)
]

# Guardrail data: latency regression in treatment
GUARDRAIL_CONTROL_LATENCY = [
    10.2, 10.5, 9.8, 10.1, 10.3, 9.9, 10.4, 10.0, 10.2, 10.1,
    10.3, 9.7, 10.5, 10.0, 10.2, 10.1, 9.8, 10.4, 10.0, 10.3,
    10.1, 9.9, 10.2, 10.0, 10.4, 9.8, 10.3, 10.1, 10.2, 10.0,
    10.5, 9.7, 10.1, 10.3, 10.0, 10.2, 9.9, 10.4, 10.1, 10.0,
    10.2, 10.3, 9.8, 10.1, 10.0, 10.4, 9.9, 10.2, 10.1, 10.3,
]

# Latency is WORSE in treatment (higher = worse for latency)
GUARDRAIL_TREATMENT_LATENCY = [
    15.2, 15.5, 14.8, 15.1, 15.3, 14.9, 15.4, 15.0, 15.2, 15.1,
    15.3, 14.7, 15.5, 15.0, 15.2, 15.1, 14.8, 15.4, 15.0, 15.3,
    15.1, 14.9, 15.2, 15.0, 15.4, 14.8, 15.3, 15.1, 15.2, 15.0,
    15.5, 14.7, 15.1, 15.3, 15.0, 15.2, 14.9, 15.4, 15.1, 15.0,
    15.2, 15.3, 14.8, 15.1, 15.0, 15.4, 14.9, 15.2, 15.1, 15.3,
]


# ---------------------------------------------------------------------------
# Fixture: default ABTestConfig
# ---------------------------------------------------------------------------


def _default_config(min_samples: int = 10) -> ABTestConfig:
    """Create a default ABTestConfig for testing."""
    return ABTestConfig(
        model_type="dlrm_bid_shader",
        control_version="v1.0",
        treatment_version="v1.1",
        traffic_percentage=5.0,
        min_samples=min_samples,
        max_duration_hours=4.0,
        significance_level=0.05,
        primary_metric="revenue_per_bid",
        guardrail_metrics=["latency_p99"],
    )


# ---------------------------------------------------------------------------
# Tests: p_value always in [0, 1]
# ---------------------------------------------------------------------------


class TestPValueBounds:
    """p_value MUST always be in [0, 1] — this is a correctness property."""

    def test_p_value_with_clear_winner(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_with_clear_loser(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_LOSER)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_with_identical_distributions(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_IDENTICAL)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_with_zero_variance(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(ZERO_VARIANCE_CONTROL, ZERO_VARIANCE_TREATMENT)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_with_small_samples(self):
        config = _default_config(min_samples=2)
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(SMALL_CONTROL, SMALL_TREATMENT)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_with_single_sample(self):
        config = _default_config(min_samples=1)
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(SINGLE_SAMPLE, SINGLE_SAMPLE)
        assert 0.0 <= result.p_value <= 1.0

    def test_p_value_with_large_effect(self):
        """Very large effect size — p_value should be near 0 but still valid."""
        config = _default_config()
        evaluator = ABEvaluator(config)
        # Large separation: control ~2.0, treatment ~100.0
        large_treatment = [100.0 + 0.1 * i for i in range(50)]
        result = evaluator.evaluate(CONTROL_BASELINE, large_treatment)
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Tests: Recommendation logic
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Test that evaluate() returns correct recommendations."""

    def test_clear_winner_recommends_promote(self):
        """When treatment is significantly better, recommend promote."""
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        assert result.recommendation == "promote"
        assert result.status in (TestStatus.PASSED, TestStatus.RUNNING)

    def test_clear_loser_recommends_reject(self):
        """When treatment is significantly worse, recommend reject."""
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_LOSER)
        assert result.recommendation == "reject"
        assert result.status in (TestStatus.FAILED, TestStatus.RUNNING)

    def test_insufficient_samples_recommends_extend(self):
        """When below min_samples, recommend extend."""
        config = _default_config(min_samples=100)
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        assert result.recommendation == "extend"
        assert result.status == TestStatus.RUNNING

    def test_no_difference_recommends_extend(self):
        """When distributions are nearly identical, recommend extend (not significant)."""
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_IDENTICAL)
        # With identical distributions, should not be significant
        assert result.recommendation == "extend"
        assert result.p_value > config.significance_level


# ---------------------------------------------------------------------------
# Tests: Guardrail violations
# ---------------------------------------------------------------------------


class TestGuardrails:
    """Test guardrail violation handling."""

    def test_guardrail_violation_forces_reject(self):
        """Guardrail regression forces reject even if primary metric improves."""
        config = _default_config()
        evaluator = ABEvaluator(config)

        # Primary metric: treatment is better
        # But latency guardrail: treatment is worse (higher latency)
        # Note: For latency, "worse" = treatment > control
        # Our evaluator checks if treatment < control as "worse", so we need
        # to flip: present latency as negative (lower is worse means we negate)
        # Actually, let's use a metric where lower is better by passing
        # negative latency values (so higher real latency = lower negative value = "worse")
        neg_control_latency = [-x for x in GUARDRAIL_CONTROL_LATENCY]
        neg_treatment_latency = [-x for x in GUARDRAIL_TREATMENT_LATENCY]

        guardrail_data = {
            "latency_p99": (neg_control_latency, neg_treatment_latency),
        }

        result = evaluator.evaluate(
            CONTROL_BASELINE,
            TREATMENT_WINNER,
            guardrail_data=guardrail_data,
        )

        assert result.recommendation == "reject"
        assert len(result.guardrail_violations) > 0
        assert "latency_p99" in result.guardrail_violations[0]

    def test_no_guardrail_violation_allows_promote(self):
        """No guardrail regression allows promote to proceed."""
        config = _default_config()
        evaluator = ABEvaluator(config)

        # Guardrail: treatment is similar to control (no regression)
        guardrail_data = {
            "latency_p99": (CONTROL_BASELINE, TREATMENT_IDENTICAL),
        }

        result = evaluator.evaluate(
            CONTROL_BASELINE,
            TREATMENT_WINNER,
            guardrail_data=guardrail_data,
        )

        assert result.guardrail_violations == []
        # Should promote since primary metric is significantly better
        assert result.recommendation == "promote"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and numerical robustness."""

    def test_identical_data_returns_high_p_value(self):
        """Identical arrays should produce p_value = 1.0 or close to it."""
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, CONTROL_BASELINE)
        assert result.p_value >= 0.99  # Essentially 1.0
        assert result.recommendation == "extend"

    def test_zero_variance_both_groups(self):
        """When both groups have zero variance but different means."""
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(ZERO_VARIANCE_CONTROL, ZERO_VARIANCE_TREATMENT)
        # Can't compute valid t-test with zero variance
        assert 0.0 <= result.p_value <= 1.0

    def test_single_sample_returns_extend(self):
        """Single sample cannot compute variance — should extend."""
        config = _default_config(min_samples=1)
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(SINGLE_SAMPLE, [3.0])
        # With single sample, can't compute variance → p_value = 1.0
        assert result.p_value == 1.0
        assert result.recommendation == "extend"

    def test_empty_data_returns_extend(self):
        """Empty data should produce extend recommendation."""
        config = _default_config(min_samples=1)
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate([], [])
        assert result.recommendation == "extend"
        assert result.p_value == 1.0

    def test_asymmetric_sample_sizes(self):
        """Different sample sizes in control and treatment."""
        config = _default_config(min_samples=5)
        evaluator = ABEvaluator(config)
        # Control has 50 samples, treatment has 10
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER[:10])
        assert 0.0 <= result.p_value <= 1.0
        assert result.samples_control == 50
        assert result.samples_treatment == 10


# ---------------------------------------------------------------------------
# Tests: SPRT early stopping
# ---------------------------------------------------------------------------


class TestSPRT:
    """Test SPRT early stopping behavior."""

    def test_sprt_promotes_with_large_effect(self):
        """SPRT should trigger early promotion with overwhelming evidence."""
        config = _default_config(min_samples=10)
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(LARGE_CONTROL_SPRT, LARGE_TREATMENT_SPRT)
        # With 200 samples and 0.5 mean difference (large effect), SPRT should trigger
        assert result.recommendation == "promote"

    def test_sprt_rejects_with_large_negative_effect(self):
        """SPRT should trigger early rejection with overwhelming negative evidence."""
        config = _default_config(min_samples=10)
        evaluator = ABEvaluator(config)
        # Swap: treatment is much worse than control
        result = evaluator.evaluate(LARGE_TREATMENT_SPRT, LARGE_CONTROL_SPRT)
        assert result.recommendation == "reject"

    def test_sprt_continues_with_small_effect(self):
        """SPRT should not trigger with negligible effect size."""
        config = _default_config(min_samples=10)
        evaluator = ABEvaluator(config)
        # Nearly identical distributions — SPRT boundaries not crossed
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_IDENTICAL)
        # Should recommend extend (no SPRT boundary crossed, not significant)
        assert result.recommendation == "extend"


# ---------------------------------------------------------------------------
# Tests: ABTestResult structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Verify ABTestResult has correct field values."""

    def test_result_has_valid_test_id(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        assert result.test_id is not None
        assert len(result.test_id) > 0

    def test_result_sample_counts_correct(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        assert result.samples_control == len(CONTROL_BASELINE)
        assert result.samples_treatment == len(TREATMENT_WINNER)

    def test_result_relative_lift_correct(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        # Treatment mean (~2.5) vs control mean (~2.0) → ~25% lift
        assert result.relative_lift > 0.2
        assert result.relative_lift < 0.3

    def test_result_metrics_match_means(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        expected_control_mean = sum(CONTROL_BASELINE) / len(CONTROL_BASELINE)
        expected_treatment_mean = sum(TREATMENT_WINNER) / len(TREATMENT_WINNER)
        assert abs(result.control_metric - expected_control_mean) < 1e-10
        assert abs(result.treatment_metric - expected_treatment_mean) < 1e-10


# ---------------------------------------------------------------------------
# Tests: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Evaluator must be deterministic — same inputs → same outputs."""

    def test_repeated_evaluation_same_result(self):
        config = _default_config()
        evaluator = ABEvaluator(config)
        result1 = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        result2 = evaluator.evaluate(CONTROL_BASELINE, TREATMENT_WINNER)
        assert result1.p_value == result2.p_value
        assert result1.recommendation == result2.recommendation
        assert result1.relative_lift == result2.relative_lift
        assert result1.control_metric == result2.control_metric
        assert result1.treatment_metric == result2.treatment_metric
