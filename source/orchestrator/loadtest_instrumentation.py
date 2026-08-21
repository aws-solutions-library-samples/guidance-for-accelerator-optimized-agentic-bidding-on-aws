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

import uuid
from dataclasses import dataclass

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


def generate_outcome_sample(run_id: str, request_index: int) -> LoadTestOutcomeSample:
    """Generate one synthetic outcome using closed_loop_demo's own generator.

    Deterministic per (run_id, request_index) — the same run always
    produces the same samples, matching closed_loop_demo's existing
    fixed-seed reproducibility convention. Reuses
    BidOutcomeMetrics.sample_outcomes() (already-reviewed logic) rather than
    a new ad hoc generator (BR-2).
    """
    # BidOutcomeMetrics.sample_outcomes() distributes exactly
    # round(n * win_rate) wins deterministically across n slots by
    # position, not by seed — so varying only the seed with n=1 would
    # always draw slot 0 and always return the same won flag. To get real
    # per-request variation while staying deterministic, draw a
    # request-specific slot from a larger, fixed sample pool.
    pool_size = 100
    records = _LOAD_TEST_SCENARIO.sample_outcomes(n=pool_size, seed=4242)
    index = (hash((run_id, request_index)) & 0xFFFFFFFF) % pool_size
    record = records[index]
    return LoadTestOutcomeSample(
        won=record["won"],
        shaded_price=record["shaded_price"],
        price_paid=record["price_paid"],
        impression=record["impression"],
        click=record["click"],
        conversion=record["conversion"],
        conversion_value=record["conversion_value"],
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
) -> float:
    """Construct and emit exactly one load-test-origin BidShadingOutcomeEvent.

    Uses the real emit_load_test_bid_outcome()/FeedbackCollector path with
    source="load_test" (never "live") and the resolved model_version for
    the request that was actually sent to the target_model_type container.
    Fire-and-forget, non-blocking — matches the existing emit_bid_outcome()
    contract (never raises, never impacts the load-test loop's timing).

    Returns the sample's real per-request revenue value (price_paid if won,
    else 0.0) — the primary metric ComparisonService's ABEvaluator call
    needs as real per-sample data (FR-8), not a fabricated or re-derived
    number; this is exactly the value used to construct the emitted event.
    """
    sample = generate_outcome_sample(run_id, request_index)
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
    return sample.price_paid if sample.won and sample.price_paid is not None else 0.0
