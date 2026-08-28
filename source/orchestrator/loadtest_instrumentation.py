"""Load test outcome-capture instrumentation (LoadTestInstrumentation).

Extends a load-test run's per-request container calls so that, for the
request's ``target_model_type`` container specifically, the synthetic
win/loss/price/CTR outcome is constructed using the SAME scenario-generation
approach ``source/closed_loop_demo`` already uses (BR-2 — no new ad hoc
generator), tagged ``source="load_test"``, and emitted via the real
``emit_load_test_bid_outcome()`` path (BR-7 — exactly once per targeted
request; see orchestrator/feedback_integration.py).

Maps to: FR-1, FR-2 (Story 1, load-test-outcome-capture unit).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from statistics import NormalDist

from closed_loop_demo.scenarios import BidOutcomeMetrics
from orchestrator.feedback_integration import emit_load_test_bid_outcome

# A single, fixed illustrative scenario shared by all load-test outcome
# generation — deliberately reusing closed_loop_demo's own statistical
# approach (BidOutcomeMetrics.sample_outcomes) rather than inventing new
# random-value logic (BR-2). Values chosen to be a plausible mid-range
# scenario, not tuned to produce any particular governance outcome.
_LOAD_TEST_SCENARIO = BidOutcomeMetrics(
    total_bids=1000,
    wins=400,
    avg_price_paid=3.20,
    avg_shaded_price=3.50,
    total_revenue=1800.0,
    total_cost=1280.0,
)

_SHADE_FACTOR_USED = 0.65  # matches dlrm_bid_shader's own SHADE_FACTOR default
_CONVERSION_VALUE_ESTIMATE_USED = 12.0  # matches dlrm_bid_shader's EST_CONVERSION_VALUE


@dataclass(frozen=True)
class LoadTestOutcomeSample:
    """One synthetic outcome sample, in the shape emit_load_test_bid_outcome() needs."""

    won: bool
    shaded_price: float
    price_paid: float | None
    impression: bool
    click: bool
    conversion: bool
    conversion_value: float | None


# Synthetic market model (see generate_outcome_sample).
#
# Spread of the competing/clearing price around its mean. Wide enough that a
# model's price choice materially changes its win rate, narrow enough that
# outcomes stay in the scenario's plausible range.
_MARKET_PRICE_STD = 0.80

# Win rate the scenario preset describes, used to calibrate the market mean so
# that a model bidding at the scenario's own avg_shaded_price wins about as
# often as the scenario says. Derived from the preset rather than hardcoded, so
# retuning _LOAD_TEST_SCENARIO keeps the market consistent with it.
_TARGET_WIN_RATE = _LOAD_TEST_SCENARIO.wins / _LOAD_TEST_SCENARIO.total_bids

# For market ~ Normal(mu, sigma), P(market <= p) = target  =>  p = mu + z*sigma
# with z = inv_cdf(target). Solving for mu at p = avg_shaded_price.
_MARKET_PRICE_MEAN = (
    _LOAD_TEST_SCENARIO.avg_shaded_price
    - NormalDist().inv_cdf(_TARGET_WIN_RATE) * _MARKET_PRICE_STD
)


# What winning a given impression is worth to the advertiser. Deliberately a
# property of the REQUEST (drawn from the seed), never of the model — otherwise a
# model could inflate its own scoring by bidding higher. Centred above the market
# mean so that bidding is profitable on average, which is what makes the surplus
# metric below have an interior optimum rather than rewarding "always bid more".
_IMPRESSION_VALUE_MEAN = 4.20
_IMPRESSION_VALUE_STD = 0.90


def impression_value(seed: int, request_index: int) -> float:
    """What winning this impression is worth to the advertiser.

    Fixed by (seed, request_index), so two runs replayed at the same seed value
    each impression identically and the only variable is what the model bid.
    """
    rng = random.Random(f"value:{seed}:{request_index}")
    return max(0.01, round(rng.gauss(_IMPRESSION_VALUE_MEAN, _IMPRESSION_VALUE_STD), 4))


def market_clearing_price(seed: int, request_index: int) -> float:
    """The synthetic competing price this request must beat to win.

    Deterministic in (seed, request_index) ONLY — deliberately not in run_id.
    A load test replayed at the same preset and seed therefore faces an
    identical sequence of market conditions, so the only thing that can differ
    between two runs is the model's own pricing. That is what makes a
    stable-vs-canary comparison attributable to the model.

    Seeded with a string: random.Random hashes str input with sha512, which is
    stable across processes. The builtin hash() is NOT (it is salted per process
    unless PYTHONHASHSEED is fixed), so hash()-derived draws differ between
    orchestrator pods for identical inputs.
    """
    rng = random.Random(f"market:{seed}:{request_index}")
    return max(0.01, round(rng.gauss(_MARKET_PRICE_MEAN, _MARKET_PRICE_STD), 4))


def generate_outcome_sample(
    *,
    seed: int,
    request_index: int,
    shaded_price: float,
) -> LoadTestOutcomeSample:
    """Resolve one synthetic auction outcome for a price the model actually chose.

    ``shaded_price`` is the real price the target container returned for this
    request (its adjust_bid mutation), so the outcome responds to the model:
    bid above the request's market clearing price and you win, below and you
    lose. Second-price settlement — a win pays the clearing price, never more
    than it bid.

    Previously this drew ``won``/``price_paid`` from a fixed 100-record pool
    indexed by ``hash((run_id, request_index))``, ignoring the container's
    output entirely. Two consequences that made a model comparison meaningless:
    the model could not influence its own outcome at all, and because run_id is
    a fresh uuid4 per run, two runs differed even at an identical seed. An A/B
    test over those samples measured run_id, not the model.

    Downstream engagement (click/conversion) still comes from the
    closed_loop_demo scenario preset (BR-2 — no new ad hoc generator for
    behavior the model does not determine), drawn deterministically from the
    same (seed, request_index) and gated on winning, since an unserved
    impression cannot be clicked (validate_bid_outcome rule 6, monotonic
    conversion -> click -> impression -> won).
    """
    clearing_price = market_clearing_price(seed, request_index)
    won = shaded_price >= clearing_price

    # Second price: pay the clearing price, capped at what was bid.
    price_paid = round(min(clearing_price, shaded_price), 4) if won else None

    # Engagement flags reused from the scenario preset, indexed deterministically.
    pool_size = 100
    records = _LOAD_TEST_SCENARIO.sample_outcomes(n=pool_size, seed=4242)
    engagement_rng = random.Random(f"engagement:{seed}:{request_index}")
    record = records[engagement_rng.randrange(pool_size)]

    click = bool(won and record["click"])
    conversion = bool(click and record["conversion"])
    return LoadTestOutcomeSample(
        won=won,
        shaded_price=shaded_price,
        price_paid=price_paid,
        impression=won,
        click=click,
        conversion=conversion,
        conversion_value=record["conversion_value"] if conversion else None,
    )


def aggregate_run_model_version(per_request_versions: list[str]) -> str:
    """Resolve a run's LoadTestStatus.model_version from per-request versions.

    Returns the majority (most common) version observed. Ties broken by
    first-occurrence order (deterministic). Empty input returns "" (a real
    "unknown" — e.g. no requests reached the target container, or none
    returned a resolved version — never fabricated).

    Pure function; property-tested (see test_loadtest_instrumentation.py):
    the result is always either "" (empty input) or one of the input
    values (never a version that wasn't actually observed).
    """
    if not per_request_versions:
        return ""
    counts: dict[str, int] = {}
    order: list[str] = []
    for v in per_request_versions:
        if v not in counts:
            order.append(v)
        counts[v] = counts.get(v, 0) + 1
    best = order[0]
    for v in order:
        if counts[v] > counts[best]:
            best = v
    return best


def emit_load_test_outcome(
    run_id: str,
    request_index: int,
    model_version: str,
    model_type: str,
    *,
    seed: int,
    shaded_price: float | None,
) -> float | None:
    """Construct and emit exactly one load-test-origin BidShadingOutcomeEvent.

    Uses the real emit_load_test_bid_outcome()/FeedbackCollector path with
    source="load_test" (never "live") and the resolved model_version for
    the request that was actually sent to the target_model_type container.
    Fire-and-forget, non-blocking — matches the existing emit_bid_outcome()
    contract (never raises, never impacts the load-test loop's timing).

    ``shaded_price`` is the price the target container actually returned for
    this request, and ``seed`` is the run's PRNG seed. Together they make the
    outcome a function of the model's decision under market conditions fixed by
    the seed (see generate_outcome_sample).

    Returns the per-request **advertiser surplus**: the impression's value to the
    advertiser minus what was actually paid for it on a win, and 0.0 on a loss
    (nothing served, nothing spent). This is the primary metric
    ComparisonService's ABEvaluator consumes as real per-sample data (FR-8).

    Why surplus and not price_paid: this used to return `price_paid if won else
    0.0`. Once outcomes became a function of the model's price, that metric
    rewarded the wrong behavior — a model that shaded less won more and paid
    more, and so scored HIGHER, even though the entire point of bid shading is to
    win at the lowest price that still wins. Surplus is maximized by winning
    often AND cheaply, so "higher is better" now matches the objective. It also
    has an interior optimum: bidding too low forfeits winnable impressions,
    bidding too high overpays for them, and both reduce the score.

    Returns None when the container produced no price for this request. There is
    then no model decision to score, and inventing one would put a fabricated
    sample into the comparison population. Callers must not count a None as an
    emitted sample.
    """
    if shaded_price is None or shaded_price <= 0:
        return None

    sample = generate_outcome_sample(
        seed=seed, request_index=request_index, shaded_price=shaded_price
    )
    # request_id stays run-scoped: it is the Glue ETL's de-duplication key, so
    # two runs replaying the same seed must still produce distinct identities or
    # the second run's records would collapse into the first's.
    request_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}-{request_index}"))

    # Price ordering (original_price >= shaded_price >= bid_floor) must hold
    # per BidShadingOutcomeEvent's validation rules. bid_floor is derived from the
    # sample's shaded_price (a plausible floor below it), and original_price
    # is derived from shaded_price/shade_factor — same relationship
    # dlrm_bid_shader/app.py's real mutate() uses.
    bid_floor = round(sample.shaded_price * 0.7, 4)
    original_price = round(sample.shaded_price / _SHADE_FACTOR_USED, 4)
    original_price = max(original_price, sample.shaded_price)

    emit_load_test_bid_outcome(
        request_id=request_id,
        model_version=model_version,
        model_type=model_type,
        won=sample.won,
        shaded_price=sample.shaded_price,
        original_price=original_price,
        bid_floor=bid_floor,
        price_paid=sample.price_paid,
        impression=sample.impression,
        click=sample.click,
        conversion=sample.conversion,
        conversion_value=sample.conversion_value,
        shade_factor_used=_SHADE_FACTOR_USED,
        conversion_value_estimate_used=_CONVERSION_VALUE_ESTIMATE_USED,
    )
    if not sample.won or sample.price_paid is None:
        # Lost the auction: nothing served and nothing spent, so zero surplus.
        return 0.0
    return round(impression_value(seed, request_index) - sample.price_paid, 4)
