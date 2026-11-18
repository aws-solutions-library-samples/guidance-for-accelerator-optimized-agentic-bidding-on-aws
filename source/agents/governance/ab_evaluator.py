"""Sequential A/B Test Evaluator with Welch's t-test and SPRT boundaries.

Provides statistical evaluation of A/B tests for model governance decisions.
Uses Welch's t-test (unequal variance) for hypothesis testing and the
Sequential Probability Ratio Test (SPRT) for early stopping while bounding
false-positive rate.

The evaluator is deterministic — same inputs always produce the same outputs.

Requirements: 4.3, 4.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TestStatus(str, Enum):
    """Status of an A/B test."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ABTestConfig:
    """Configuration for an A/B test evaluation."""

    model_type: str
    control_version: str
    treatment_version: str
    traffic_percentage: float
    min_samples: int  # per group before evaluation
    max_duration_hours: float
    significance_level: float  # default 0.05
    primary_metric: str  # "revenue_per_bid" | "ctr" | "win_rate"
    guardrail_metrics: list[str] = field(default_factory=list)


@dataclass
class ABTestResult:
    """Result of an A/B test evaluation."""

    test_id: str
    status: TestStatus
    control_metric: float
    treatment_metric: float
    relative_lift: float
    p_value: float  # Must be in [0, 1]
    samples_control: int
    samples_treatment: int
    guardrail_violations: list[str] = field(default_factory=list)
    recommendation: str = "extend"  # "promote" | "reject" | "extend"


# ---------------------------------------------------------------------------
# Mathematical helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _gamma_lanczos(z: float) -> float:
    """Compute the Gamma function using the Lanczos approximation.

    Accurate to ~15 significant digits for z > 0.5.
    """
    # Lanczos coefficients (g=7, n=9)
    g = 7
    coefficients = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if z < 0.5:
        # Reflection formula: Gamma(z) * Gamma(1-z) = pi / sin(pi*z)
        return math.pi / (math.sin(math.pi * z) * _gamma_lanczos(1 - z))

    z -= 1
    x = coefficients[0]
    for i in range(1, len(coefficients)):
        x += coefficients[i] / (z + i)

    t = z + g + 0.5
    return math.sqrt(2 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x


def _log_gamma(z: float) -> float:
    """Compute log(Gamma(z)) for z > 0, directly in log-space to avoid overflow."""
    if z <= 0:
        return float("inf")

    # Lanczos approximation computed entirely in log-space
    g = 7
    coefficients = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if z < 0.5:
        # Reflection formula in log-space:
        # log(Gamma(z)) = log(pi) - log(sin(pi*z)) - log(Gamma(1-z))
        return math.log(math.pi) - math.log(abs(math.sin(math.pi * z))) - _log_gamma(1 - z)

    z -= 1
    x = coefficients[0]
    for i in range(1, len(coefficients)):
        x += coefficients[i] / (z + i)

    t = z + g + 0.5
    # log(Gamma(z+1)) = 0.5*log(2*pi) + (z+0.5)*log(t) - t + log(x)
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def _beta_function(a: float, b: float) -> float:
    """Compute the Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    return math.exp(_log_gamma(a) + _log_gamma(b) - _log_gamma(a + b))


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Compute the regularized incomplete beta function I_x(a, b).

    Uses the continued fraction expansion for numerical stability.
    This is the CDF of the Beta distribution evaluated at x.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation for numerical stability when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)

    # Continued fraction (Lentz's method)
    front = math.exp(
        a * math.log(x) + b * math.log(1.0 - x) - math.log(a) - _log_gamma(a)
        - _log_gamma(b) + _log_gamma(a + b)
    )

    # Evaluate continued fraction
    max_iterations = 200
    epsilon = 1e-14
    tiny = 1e-30

    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iterations + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return front * f


def _t_cdf(t_stat: float, df: float) -> float:
    """Compute the CDF of the Student's t-distribution at t_stat with df degrees of freedom.

    Uses the relationship between the t-distribution CDF and the
    regularized incomplete beta function.
    """
    if df <= 0:
        return 0.5

    x = df / (df + t_stat * t_stat)
    beta_val = _regularized_incomplete_beta(x, df / 2.0, 0.5)

    if t_stat >= 0:
        return 1.0 - 0.5 * beta_val
    else:
        return 0.5 * beta_val


# ---------------------------------------------------------------------------
# ABEvaluator
# ---------------------------------------------------------------------------


class ABEvaluator:
    """Sequential A/B test evaluator using Welch's t-test with SPRT boundaries.

    Supports early stopping via the Sequential Probability Ratio Test while
    bounding false-positive rate at the configured significance level.

    The evaluator is deterministic: same inputs → same outputs.

    Parameters:
        config: ABTestConfig specifying test parameters.
    """

    def __init__(self, config: ABTestConfig) -> None:
        self._config = config

    @property
    def config(self) -> ABTestConfig:
        """Return the test configuration."""
        return self._config

    def evaluate(
        self,
        control_data: list[float],
        treatment_data: list[float],
        guardrail_data: dict[str, tuple[list[float], list[float]]] | None = None,
    ) -> ABTestResult:
        """Evaluate A/B test data and return a recommendation.

        Args:
            control_data: Metric observations for the control group.
            treatment_data: Metric observations for the treatment group.
            guardrail_data: Optional dict mapping metric name to
                (control_values, treatment_values) for guardrail checks.

        Returns:
            ABTestResult with computed statistics and recommendation.
        """
        n_control = len(control_data)
        n_treatment = len(treatment_data)
        min_samples = self._config.min_samples

        # Compute basic statistics
        control_mean = _safe_mean(control_data)
        treatment_mean = _safe_mean(treatment_data)

        # Relative lift
        if control_mean != 0.0:
            relative_lift = (treatment_mean - control_mean) / abs(control_mean)
        else:
            relative_lift = 0.0 if treatment_mean == 0.0 else float("inf")
            # Clamp infinite lift to a large value for practical purposes
            if relative_lift == float("inf"):
                relative_lift = 10.0
            elif relative_lift == float("-inf"):
                relative_lift = -10.0

        # Check min samples before statistical evaluation
        if n_control < min_samples or n_treatment < min_samples:
            return ABTestResult(
                test_id=self._generate_test_id(),
                status=TestStatus.RUNNING,
                control_metric=control_mean,
                treatment_metric=treatment_mean,
                relative_lift=relative_lift,
                p_value=1.0,  # No evidence yet — conservative default
                samples_control=n_control,
                samples_treatment=n_treatment,
                guardrail_violations=[],
                recommendation="extend",
            )

        # Compute Welch's t-test
        t_stat, p_value = self._welch_t_test(control_data, treatment_data)

        # Check guardrail metrics
        guardrail_violations: list[str] = []
        if guardrail_data:
            guardrail_violations = self._check_guardrails(guardrail_data)

        # If any guardrail is violated, force reject regardless of primary metric
        if guardrail_violations:
            return ABTestResult(
                test_id=self._generate_test_id(),
                status=TestStatus.FAILED,
                control_metric=control_mean,
                treatment_metric=treatment_mean,
                relative_lift=relative_lift,
                p_value=p_value,
                samples_control=n_control,
                samples_treatment=n_treatment,
                guardrail_violations=guardrail_violations,
                recommendation="reject",
            )

        # Apply SPRT boundaries for early stopping
        sprt_decision = self._sprt_decision(control_data, treatment_data)

        if sprt_decision is not None:
            # SPRT reached a boundary — early stop
            status = TestStatus.PASSED if sprt_decision == "promote" else TestStatus.FAILED
            return ABTestResult(
                test_id=self._generate_test_id(),
                status=status,
                control_metric=control_mean,
                treatment_metric=treatment_mean,
                relative_lift=relative_lift,
                p_value=p_value,
                samples_control=n_control,
                samples_treatment=n_treatment,
                guardrail_violations=[],
                recommendation=sprt_decision,
            )

        # No SPRT boundary crossed — use standard significance test
        significance = self._config.significance_level
        if p_value < significance and treatment_mean > control_mean:
            recommendation = "promote"
            status = TestStatus.PASSED
        elif p_value < significance and treatment_mean <= control_mean:
            recommendation = "reject"
            status = TestStatus.FAILED
        else:
            recommendation = "extend"
            status = TestStatus.RUNNING

        return ABTestResult(
            test_id=self._generate_test_id(),
            status=status,
            control_metric=control_mean,
            treatment_metric=treatment_mean,
            relative_lift=relative_lift,
            p_value=p_value,
            samples_control=n_control,
            samples_treatment=n_treatment,
            guardrail_violations=[],
            recommendation=recommendation,
        )

    def _welch_t_test(
        self, control: list[float], treatment: list[float]
    ) -> tuple[float, float]:
        """Compute Welch's t-test (unequal variance assumption).

        Returns:
            Tuple of (t_statistic, p_value). p_value is always in [0, 1].
        """
        n_c = len(control)
        n_t = len(treatment)

        if n_c < 2 or n_t < 2:
            # Cannot compute variance with fewer than 2 samples
            return (0.0, 1.0)

        mean_c = _safe_mean(control)
        mean_t = _safe_mean(treatment)
        var_c = _safe_variance(control, mean_c)
        var_t = _safe_variance(treatment, mean_t)

        # Welch's t-statistic: t = (mean_t - mean_c) / sqrt(var_t/n_t + var_c/n_c)
        se_sq = var_t / n_t + var_c / n_c

        if se_sq <= 0.0:
            # Zero variance in both groups — identical distributions
            return (0.0, 1.0)

        se = math.sqrt(se_sq)
        t_stat = (mean_t - mean_c) / se

        # Welch-Satterthwaite degrees of freedom
        numerator = se_sq ** 2
        denom = (
            (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
        )

        if denom <= 0.0:
            df = max(n_c, n_t) - 1.0
        else:
            df = numerator / denom

        # Ensure df is at least 1
        df = max(1.0, df)

        # Two-sided p-value from t-distribution CDF
        cdf_val = _t_cdf(abs(t_stat), df)
        p_value = 2.0 * (1.0 - cdf_val)

        # Clamp p-value to [0, 1] to handle numerical issues
        p_value = max(0.0, min(1.0, p_value))

        return (t_stat, p_value)

    def _sprt_decision(
        self, control: list[float], treatment: list[float]
    ) -> str | None:
        """Apply Sequential Probability Ratio Test for early stopping.

        Computes a log-likelihood ratio based on the observed effect size and
        compares against SPRT boundaries.

        SPRT boundaries:
            Upper: log((1 - beta) / alpha) → evidence for treatment being better
            Lower: log(beta / (1 - alpha)) → evidence against treatment

        Returns:
            "promote" if upper boundary crossed (overwhelming evidence for treatment).
            "reject" if lower boundary crossed (overwhelming evidence against treatment).
            None if neither boundary crossed (continue testing).
        """
        alpha = self._config.significance_level  # Type I error rate
        beta = 0.1  # Type II error rate

        n_c = len(control)
        n_t = len(treatment)

        if n_c < 2 or n_t < 2:
            return None

        mean_c = _safe_mean(control)
        mean_t = _safe_mean(treatment)
        var_c = _safe_variance(control, mean_c)
        var_t = _safe_variance(treatment, mean_t)

        # Pooled variance estimate for the effect size calculation
        pooled_var = (var_c + var_t) / 2.0

        if pooled_var <= 0.0:
            # Zero variance — no information for SPRT
            return None

        # Observed effect size (Cohen's d)
        effect_size = (mean_t - mean_c) / math.sqrt(pooled_var)

        # Total information (number of observations contributing to the test)
        # Using harmonic mean of sample sizes as the effective sample size
        n_eff = 2.0 * n_c * n_t / (n_c + n_t)

        # Log-likelihood ratio approximation for SPRT
        # Under H1 (treatment better by delta), the LLR accumulates as:
        # LLR ≈ effect_size * sqrt(n_eff / 2) * effect_size / 2
        # Simplified: LLR = n_eff * effect_size^2 / 4
        # This is proportional to the non-centrality parameter
        llr = n_eff * effect_size * abs(effect_size) / 4.0

        # SPRT boundaries (Wald's sequential test)
        upper_boundary = math.log((1.0 - beta) / alpha)
        lower_boundary = math.log(beta / (1.0 - alpha))

        if llr >= upper_boundary:
            # Overwhelming evidence that treatment is better
            return "promote"
        elif llr <= lower_boundary:
            # Overwhelming evidence that treatment is worse
            return "reject"

        return None

    def _check_guardrails(
        self, guardrail_data: dict[str, tuple[list[float], list[float]]]
    ) -> list[str]:
        """Check guardrail metrics for significant regressions.

        For each guardrail metric, compute whether the treatment is
        significantly worse than control. "Worse" means the treatment
        metric is lower (for metrics where higher is better).

        Args:
            guardrail_data: Dict mapping metric name to
                (control_values, treatment_values).

        Returns:
            List of violation descriptions for metrics that regressed.
        """
        violations: list[str] = []
        significance = self._config.significance_level

        for metric_name, (control_vals, treatment_vals) in guardrail_data.items():
            if len(control_vals) < 2 or len(treatment_vals) < 2:
                continue

            t_stat, p_value = self._welch_t_test(control_vals, treatment_vals)
            treatment_mean = _safe_mean(treatment_vals)
            control_mean = _safe_mean(control_vals)

            # Violation: treatment is significantly worse (lower) than control
            if p_value < significance and treatment_mean < control_mean:
                violations.append(
                    f"{metric_name}: treatment ({treatment_mean:.4f}) "
                    f"significantly worse than control ({control_mean:.4f}), "
                    f"p={p_value:.4f}"
                )

        return violations

    def _generate_test_id(self) -> str:
        """Generate a deterministic test ID based on config."""
        # Use a stable ID derived from the config for reproducibility
        return (
            f"ab_{self._config.model_type}_"
            f"{self._config.control_version}_vs_"
            f"{self._config.treatment_version}"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _safe_mean(data: list[float]) -> float:
    """Compute the arithmetic mean, returning 0.0 for empty input."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def _safe_variance(data: list[float], mean: float) -> float:
    """Compute sample variance (unbiased, Bessel's correction).

    Returns 0.0 if fewer than 2 samples.
    """
    n = len(data)
    if n < 2:
        return 0.0
    return sum((x - mean) ** 2 for x in data) / (n - 1)
