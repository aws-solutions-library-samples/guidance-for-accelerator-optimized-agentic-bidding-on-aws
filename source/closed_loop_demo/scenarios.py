"""Scenario presets for the closed-loop demo.

Each scenario encodes a deterministic set of **input** market metrics (and, for
governance scenarios, deterministic A/B samples). These are the *inputs* fed to
the real Part 2 decision code — not a pre-computed result.

Two different decision surfaces consume these inputs:

* Adaptive Bidding (``agents.adaptive_bidding.agent.AdaptiveBiddingStrategyAgent``)
  is a **reasoning agent** (Strands + Amazon Bedrock). It has **no** fixed target
  win rate, learning rate, tolerance, or gradient formula — the model weighs the
  real metrics (win rate, ROI, prices, sample counts) and explains its own choice.
  Hard safety limits (``shade_factor ∈ [0.3, 0.95]``, ``conversion_value ∈
  [1.0, 50.0]``, a per-update max delta, and optimistic version checks) are
  enforced by the DynamoDB Parameter Store write layer, not by this scenario data.
  Because the decision is the agent's reasoning, each agentic scenario's
  ``expected_decision`` is guidance — "**what to watch for**" — not a guaranteed
  outcome. The authoritative result is whatever the agent actually reasons and
  writes at run time.

* Governance (``agents.governance.ab_evaluator.ABEvaluator``) is a **real
  statistical** gate (Welch's t-test + SPRT). Here the A/B sample means/variances
  deterministically drive a specific, verifiable recommendation (promote / reject
  / extend), so the governance ``expected_decision`` values are reliable.

These presets are pure data; nothing here performs I/O or randomness that a user
sees as a "result". The A/B sample lists are materialized deterministically from a
fixed seed purely so the real statistical evaluator has realistic input.
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

    These are the six metrics the Adaptive Bidding Strategy Agent reads to compute
    its MarketState (see ``AdaptiveBiddingStrategyAgent.compute_market_state``).
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

    def sample_outcomes(self, n: int = 10, seed: int = 4242) -> list[dict]:
        """Deterministically generate ``n`` illustrative individual bid-outcome records.

        These are NOT a reconstruction of specific historical bids — there is no
        per-bid data behind this aggregate scenario preset. Instead this produces a
        small, fixed-seed sample of individual-record *shapes* (``won`` /
        ``price_paid`` / ``impression`` / ``click`` / ``conversion``, matching the
        real ``BidShadingOutcomeEvent`` schema in ``shared.feedback_models``) that are
        statistically consistent with this scenario's aggregate metrics (win_rate,
        avg prices). Same seed ⇒ same records on every call — reproducible and
        inspectable, not randomized per-request to "look real".

        Callers MUST label this data as an illustrative synthetic sample (see
        ``Scenario.sample_outcomes``), never as an observed/real outcome feed.
        """
        if self.total_bids <= 0 or n <= 0:
            return []
        n = min(n, self.total_bids)
        rng = random.Random(seed)

        # Deterministically distribute the expected number of wins across the n
        # sampled slots (rather than flipping a per-record biased coin, which
        # would make small samples drift noticeably from the target win_rate).
        target_wins = round(n * self.win_rate)
        won_flags = [False] * n
        if target_wins > 0:
            step = n / target_wins
            for i in range(target_wins):
                idx = min(n - 1, int(i * step))
                won_flags[idx] = True

        price_std = max(self.avg_shaded_price * 0.08, 0.01)
        paid_std = max(self.avg_price_paid * 0.08, 0.01)

        records = []
        for i in range(n):
            won = won_flags[i]
            shaded_price = round(max(0.0, rng.gauss(self.avg_shaded_price, price_std)), 4)
            price_paid = (
                round(max(0.0, rng.gauss(self.avg_price_paid, paid_std)), 4) if won else None
            )
            # Downstream signals are monotonic per the real BidShadingOutcomeEvent contract
            # (impression ⇒ won; click ⇒ impression; conversion ⇒ click). Click/
            # conversion rates below are illustrative assumptions, not scenario
            # inputs — they exist only to give the sample table realistic shape.
            impression = won
            click = bool(impression and rng.random() < 0.05)
            conversion = bool(click and rng.random() < 0.10)
            conversion_value = round(rng.uniform(5.0, 25.0), 2) if conversion else None
            records.append(
                {
                    "index": i,
                    "won": won,
                    "shaded_price": shaded_price,
                    "price_paid": price_paid,
                    "impression": impression,
                    "click": click,
                    "conversion": conversion,
                    "conversion_value": conversion_value,
                }
            )
        return records


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

    def sample_outcomes(self, n: int = 10) -> dict:
        """Return the first ``n`` control/treatment values actually fed to the
        real ``ABEvaluator`` (the same lists ``materialize()`` produces — this is
        not a separate random draw, just a truncated view of the real input).
        """
        control, treatment = self.materialize()
        n = max(0, min(n, len(control), len(treatment)))
        return {
            "primary_metric": self.primary_metric,
            "n_shown": n,
            "n_per_group": self.n_per_group,
            "control": [round(v, 4) for v in control[:n]],
            "treatment": [round(v, 4) for v in treatment[:n]],
        }


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

    def sample_outcomes(self, n: int = 10) -> dict:
        """A JSON-serializable subset of individual synthetic sample records.

        For agentic scenarios: ``n`` illustrative individual bid-outcome records
        (won/price_paid/impression/click/conversion) consistent with this
        scenario's aggregate ``bid_metrics``.

        For governance scenarios: the first ``n`` control/treatment values from
        the same deterministic A/B sample lists fed to the real ``ABEvaluator``.

        Always deterministic (fixed seed) and explicitly labelled as synthetic
        sample input — never a fabricated "result".
        """
        if self.bid_metrics is not None:
            return {
                "loop": self.loop,
                "kind": "bid_outcome_records",
                "records": self.bid_metrics.sample_outcomes(n=n),
                "note": (
                    "Illustrative synthetic bid-outcome records consistent with this "
                    "scenario's aggregate metrics (win_rate, avg prices). Not a "
                    "reconstruction of real historical bids."
                ),
            }
        if self.ab_samples is not None:
            return {
                "loop": self.loop,
                "kind": "ab_samples",
                **self.ab_samples.sample_outcomes(n=n),
                "note": (
                    "First N values of the deterministic control/treatment lists "
                    "actually fed to the real ABEvaluator (Welch's t-test + SPRT) "
                    "for this scenario."
                ),
            }
        return {"loop": self.loop, "kind": "none", "records": [], "note": "No sample data for this scenario."}


# ---------------------------------------------------------------------------
# Agentic parameter-loop scenarios
# ---------------------------------------------------------------------------
# Each sets a distinct real market condition. The Adaptive Bidding reasoning agent
# decides what (if anything) to change; the ``expected_decision`` text below is
# "what to watch for" guidance, not a guaranteed formula output.

_UNDERBIDDING = Scenario(
    key="underbidding",
    label="Underbidding — likely raise shade_factor",
    loop=LOOP_AGENTIC,
    description=(
        "Low win rate (~20%) with healthy positive ROI — the platform is leaving "
        "winnable, profitable impressions on the table. Watch whether the agent "
        "reasons toward bidding more aggressively (raising shade_factor and/or "
        "conversion_value)."
    ),
    expected_decision="likely bids up — watch for shade_factor ↑ / conversion_value ↑",
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
    label="Overpaying — likely lower shade_factor",
    loop=LOOP_AGENTIC,
    description=(
        "High win rate (~55%) but negative ROI — the platform is winning too much "
        "and overpaying. Watch whether the agent reasons toward pulling back "
        "(lowering shade_factor and/or conversion_value)."
    ),
    expected_decision="likely pulls back — watch for shade_factor ↓ / conversion_value ↓",
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
    label="Healthy — likely no change (stability)",
    loop=LOOP_AGENTIC,
    description=(
        "A moderate win rate (~35%) with slightly positive ROI — parameters look "
        "well-placed. Watch whether the agent reasons that no adjustment is "
        "warranted, demonstrating that the loop holds steady rather than "
        "oscillating."
    ),
    expected_decision="likely holds steady — watch for no change",
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
    label="Thin data — agent likely declines to act",
    loop=LOOP_AGENTIC,
    description=(
        "Only 500 bids in the window — a thin sample. The agent is told the sample "
        "counts and is instructed to decline to act on data too sparse to draw a "
        "conclusion. Watch whether it reasons to make no change on low confidence."
    ),
    expected_decision="likely declines — watch for no change on thin data",
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
        control_std=0.05,
        treatment_mean=0.99,
        treatment_std=0.05,
        n_per_group=120,
    ),
)

# NCF Deal Manager variants of the same three governance outcomes. Each set of
# ab_samples below was verified against the real ABEvaluator (not just picked
# to match DESIGN_BRIEF.md's illustrative figures) — see
# aidlc-docs/construction/model-governance-panel/ncf-governance-scenario-verification.md
# for the exact evaluate() output each produces.

_CHALLENGER_WINS_NCF = Scenario(
    key="challenger_wins_ncf",
    label="Challenger wins — promote",
    loop=LOOP_GOVERNANCE,
    description=(
        "A retrained NCF challenger delivers a higher deal_hit_rate than the "
        "incumbent with tight variance. The real A/B evaluator should find this "
        "statistically significant and recommend promote."
    ),
    expected_decision="promote",
    model_type="ncf_deal_manager",
    ab_samples=ABSamples(
        control_mean=0.567,
        control_std=0.05,
        treatment_mean=0.612,
        treatment_std=0.05,
        n_per_group=300,
        primary_metric="deal_hit_rate",
    ),
)

_CHALLENGER_LOSES_NCF = Scenario(
    key="challenger_loses_ncf",
    label="Challenger loses — reject",
    loop=LOOP_GOVERNANCE,
    description=(
        "The NCF challenger's deal_hit_rate is meaningfully worse than the "
        "incumbent's. The real A/B evaluator should recommend reject and the "
        "incumbent stays in production."
    ),
    expected_decision="reject",
    model_type="ncf_deal_manager",
    ab_samples=ABSamples(
        control_mean=0.567,
        control_std=0.05,
        treatment_mean=0.498,
        treatment_std=0.05,
        n_per_group=300,
        primary_metric="deal_hit_rate",
    ),
)

_INCONCLUSIVE_NCF = Scenario(
    key="inconclusive_ncf",
    label="Too close to call — extend",
    loop=LOOP_GOVERNANCE,
    description=(
        "The NCF challenger and incumbent perform almost identically on "
        "deal_hit_rate. With no significant difference, the real evaluator "
        "should recommend extend (keep testing) rather than promote or reject."
    ),
    expected_decision="extend",
    model_type="ncf_deal_manager",
    ab_samples=ABSamples(
        control_mean=0.567,
        control_std=0.05,
        treatment_mean=0.552,
        treatment_std=0.05,
        n_per_group=120,
        primary_metric="deal_hit_rate",
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
    _CHALLENGER_WINS_NCF,
    _CHALLENGER_LOSES_NCF,
    _INCONCLUSIVE_NCF,
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
