"""Scenario presets for the closed-loop demo.

Each scenario encodes a deterministic set of **input** market metrics and the
**decision it is expected to drive** through the real Part 2 decision code.

The numeric targets here were chosen against the *actual* thresholds used by
``agents.bid_shading.agent.BidShadingStrategyAgent`` and
``agents.governance.ab_evaluator.ABEvaluator`` so that a generated scenario
produces a specific, verifiable decision:

Bid-shading agent defaults (see ``DEFAULT_CONFIG`` in the agent):
    target_win_rate = 0.35   win_rate_tolerance = 0.05
    max_adjustment  = 0.05   learning_rate      = 0.10
    min_samples     = 1000

shade_factor policy:
    error = win_rate - target
    if |error| <= tolerance: no change
    raw   = -error * learning_rate * (0.5 + 0.5 * clamp(roi, 0, 1))
    adj   = clamp(raw, -max_adjustment, +max_adjustment)   → clamp value to [0.30, 0.95]

conversion_value policy:
    roi < 0 and win_rate > target  → decrease
    roi > 0 and win_rate < target  → increase
    else                           → no change

These presets are pure data; nothing here performs I/O or randomness that a
user sees as a "result". The A/B sample lists are materialized deterministically
from a fixed seed purely so the real statistical evaluator has realistic input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Input metric shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BidOutcomeMetrics:
    """Target values for the ``ARTF/BidOutcome`` CloudWatch metrics.

    These are the six metrics the Bid Shading Strategy Agent reads to compute
    its MarketState (see ``BidShadingStrategyAgent.get_market_state``).
    """

    total_bids: int
    wins: int
    avg_price_paid: float
    avg_shaded_price: float
    total_revenue: float
    total_cost: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_bids if self.total_bids > 0 else 0.0

    @property
    def roi(self) -> float:
        return (
            (self.total_revenue - self.total_cost) / self.total_cost
            if self.total_cost > 0
            else 0.0
        )

    def as_metric_map(self) -> dict[str, float]:
        """Return the CloudWatch metric-name → value map for this scenario.

        Keys match the metric names the agent queries in namespace
        ``ARTF/BidOutcome``.
        """
        return {
            "TotalBids": float(self.total_bids),
            "Wins": float(self.wins),
            "AvgPricePaid": float(self.avg_price_paid),
            "AvgShadedPrice": float(self.avg_shaded_price),
            "TotalRevenue": float(self.total_revenue),
            "TotalCost": float(self.total_cost),
        }


@dataclass(frozen=True)
class ABSamples:
    """Deterministic A/B sample generators for the real ABEvaluator.

    ``control_mean``/``treatment_mean`` set the effect direction; the evaluator
    computes a real Welch's t-test + SPRT decision from the materialized lists.
    """

    control_mean: float
    control_std: float
    treatment_mean: float
    treatment_std: float
    n_per_group: int
    primary_metric: str = "revenue_per_bid"
    seed: int = 1729

    def materialize(self) -> tuple[list[float], list[float]]:
        """Produce (control_values, treatment_values) deterministically.

        Uses a fixed-seed Gaussian purely to give the *real* statistical
        evaluator realistic input. Same seed ⇒ same lists ⇒ reproducible,
        inspectable decision (no hidden randomness in the result).
        """
        rng = random.Random(self.seed)
        control = [rng.gauss(self.control_mean, self.control_std) for _ in range(self.n_per_group)]
        # Advance deterministically for the treatment group
        rng2 = random.Random(self.seed + 1)
        treatment = [rng2.gauss(self.treatment_mean, self.treatment_std) for _ in range(self.n_per_group)]
        return control, treatment


@dataclass(frozen=True)
class InferenceMetrics:
    """Target values for the ``ARTF/Inference`` CloudWatch metrics.

    Used to populate the metrics view and to exercise the real guardrail
    detection logic (canary p99 latency vs stable, canary error rate).
    """

    stable_latency_p99_ms: float
    canary_latency_p99_ms: float
    canary_error_rate: float


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

LOOP_AGENTIC = "agentic"
LOOP_GOVERNANCE = "governance"


@dataclass(frozen=True)
class Scenario:
    """A single controllable demo scenario."""

    key: str
    label: str
    loop: str  # LOOP_AGENTIC | LOOP_GOVERNANCE
    description: str
    expected_decision: str
    model_type: str = "dlrm_bid_shader"
    bid_metrics: Optional[BidOutcomeMetrics] = None
    ab_samples: Optional[ABSamples] = None
    inference_metrics: Optional[InferenceMetrics] = None

    def summary(self) -> dict:
        """A JSON-serializable summary for the API / UI (no materialized lists)."""
        out: dict = {
            "key": self.key,
            "label": self.label,
            "loop": self.loop,
            "description": self.description,
            "expected_decision": self.expected_decision,
            "model_type": self.model_type,
        }
        if self.bid_metrics is not None:
            out["bid_metrics"] = {
                **self.bid_metrics.as_metric_map(),
                "win_rate": round(self.bid_metrics.win_rate, 4),
                "roi": round(self.bid_metrics.roi, 4),
            }
        if self.ab_samples is not None:
            out["ab_samples"] = {
                "control_mean": self.ab_samples.control_mean,
                "treatment_mean": self.ab_samples.treatment_mean,
                "n_per_group": self.ab_samples.n_per_group,
                "primary_metric": self.ab_samples.primary_metric,
            }
        if self.inference_metrics is not None:
            out["inference_metrics"] = {
                "stable_latency_p99_ms": self.inference_metrics.stable_latency_p99_ms,
                "canary_latency_p99_ms": self.inference_metrics.canary_latency_p99_ms,
                "canary_error_rate": self.inference_metrics.canary_error_rate,
            }
        return out


# ---------------------------------------------------------------------------
# Agentic parameter-loop scenarios
# ---------------------------------------------------------------------------
# Each drives a specific, verified decision by the real bid-shading agent.

_UNDERBIDDING = Scenario(
    key="underbidding",
    label="Underbidding — raise shade_factor",
    loop=LOOP_AGENTIC,
    description=(
        "Win rate 20% (well below the 35% target) with healthy ROI. The agent "
        "should bid more aggressively: raise shade_factor and raise "
        "conversion_value."
    ),
    expected_decision="shade_factor ↑ and conversion_value ↑",
    bid_metrics=BidOutcomeMetrics(
        total_bids=2000,
        wins=400,            # win_rate = 0.20
        avg_price_paid=2.00,
        avg_shaded_price=1.30,
        total_revenue=1200.0,
        total_cost=800.0,    # roi = +0.50
    ),
)

_OVERPAYING = Scenario(
    key="overpaying",
    label="Overpaying — lower shade_factor",
    loop=LOOP_AGENTIC,
    description=(
        "Win rate 55% (well above target) but negative ROI — we are winning too "
        "much and overpaying. The agent should pull back: lower shade_factor and "
        "lower conversion_value."
    ),
    expected_decision="shade_factor ↓ and conversion_value ↓",
    bid_metrics=BidOutcomeMetrics(
        total_bids=2000,
        wins=1100,           # win_rate = 0.55
        avg_price_paid=2.00,
        avg_shaded_price=1.70,
        total_revenue=1050.0,
        total_cost=1500.0,   # roi = -0.30
    ),
)

_HEALTHY = Scenario(
    key="healthy",
    label="Healthy — no change (stability)",
    loop=LOOP_AGENTIC,
    description=(
        "Win rate at the 35% target with slightly positive ROI. Within tolerance, "
        "so the agent should make no adjustment — demonstrating convergence and "
        "that the loop does not oscillate."
    ),
    expected_decision="no change (within tolerance)",
    bid_metrics=BidOutcomeMetrics(
        total_bids=2000,
        wins=700,            # win_rate = 0.35 (== target)
        avg_price_paid=2.00,
        avg_shaded_price=1.30,
        total_revenue=1050.0,
        total_cost=1000.0,   # roi = +0.05
    ),
)

_INSUFFICIENT = Scenario(
    key="insufficient_data",
    label="Insufficient data — agent skips",
    loop=LOOP_AGENTIC,
    description=(
        "Only 500 bids in the window — below the 1000-sample minimum. The agent "
        "should skip this cycle and make no change, demonstrating the min-samples "
        "safety guard."
    ),
    expected_decision="skipped (below min_samples)",
    bid_metrics=BidOutcomeMetrics(
        total_bids=500,
        wins=100,            # win_rate = 0.20 but below min_samples
        avg_price_paid=2.00,
        avg_shaded_price=1.30,
        total_revenue=300.0,
        total_cost=200.0,
    ),
)


# ---------------------------------------------------------------------------
# Governance / A/B scenarios
# ---------------------------------------------------------------------------
# Drive the real ABEvaluator (Welch's t-test + SPRT) to a specific decision.

_CHALLENGER_WINS = Scenario(
    key="challenger_wins",
    label="Challenger wins — promote",
    loop=LOOP_GOVERNANCE,
    description=(
        "A retrained challenger delivers ~25% higher revenue-per-bid than the "
        "incumbent with tight variance. The real A/B evaluator should find this "
        "statistically significant and recommend promote."
    ),
    expected_decision="promote",
    ab_samples=ABSamples(
        control_mean=1.00,
        control_std=0.30,
        treatment_mean=1.25,
        treatment_std=0.30,
        n_per_group=300,
    ),
)

_CHALLENGER_LOSES = Scenario(
    key="challenger_loses",
    label="Challenger loses — reject",
    loop=LOOP_GOVERNANCE,
    description=(
        "The challenger performs ~20% worse than the incumbent. The real A/B "
        "evaluator should recommend reject and the incumbent stays in production."
    ),
    expected_decision="reject",
    ab_samples=ABSamples(
        control_mean=1.25,
        control_std=0.30,
        treatment_mean=1.00,
        treatment_std=0.30,
        n_per_group=300,
    ),
)

_INCONCLUSIVE = Scenario(
    key="inconclusive",
    label="Too close to call — extend",
    loop=LOOP_GOVERNANCE,
    description=(
        "Challenger and incumbent perform almost identically. With no significant "
        "difference, the real evaluator should recommend extend (keep testing) "
        "rather than promote or reject."
    ),
    expected_decision="extend",
    ab_samples=ABSamples(
        control_mean=1.00,
        control_std=0.30,
        treatment_mean=1.01,
        treatment_std=0.30,
        n_per_group=120,
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_SCENARIOS: tuple[Scenario, ...] = (
    _UNDERBIDDING,
    _OVERPAYING,
    _HEALTHY,
    _INSUFFICIENT,
    _CHALLENGER_WINS,
    _CHALLENGER_LOSES,
    _INCONCLUSIVE,
)

SCENARIOS: dict[str, Scenario] = {s.key: s for s in _ALL_SCENARIOS}


def list_scenarios(loop: Optional[str] = None) -> list[Scenario]:
    """Return all scenarios, optionally filtered by loop."""
    if loop is None:
        return list(_ALL_SCENARIOS)
    return [s for s in _ALL_SCENARIOS if s.loop == loop]


def get_scenario(key: str) -> Scenario:
    """Look up a scenario by key. Raises KeyError if unknown."""
    try:
        return SCENARIOS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown scenario '{key}'. Valid keys: {sorted(SCENARIOS)}"
        ) from exc
