"""Deal Yield feedback integration -- emits DealYieldOutcomeEvents from the
orchestrator's bid path.

Mirrors source/orchestrator/feedback_integration.py's emit_bid_outcome()
contract exactly (fire-and-forget, never raises, < 1ms overhead), but for
deal floor/margin outcomes rather than bid-shading outcomes. Deliberately a
SEPARATE Kinesis stream from BidOutcomeStream -- see
aidlc-docs/construction/deal-yield-outcome-capture/functional-design/
business-logic-model.md for why sharing the stream would silently drop this
event's fields (Firehose's Glue-schema-based Parquet conversion maps by
field name).

Env vars:
- DEAL_YIELD_FEEDBACK_STREAM_NAME -- Kinesis stream name. If unset, emission
  is disabled and emit_deal_yield_outcome() is a no-op.
- DEAL_YIELD_FEEDBACK_STREAM_REGION -- falls back to
  FEEDBACK_STREAM_REGION -> AWS_REGION -> AWS_DEFAULT_REGION -> us-east-1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional

from closed_loop_demo.scenarios import BidOutcomeMetrics
from shared.artf_types import RTBRequest, RTBResponse, Mutation
from shared.feedback_collector import FeedbackCollector, fire_and_forget_emit
from shared.feedback_models import DealYieldOutcomeEvent

logger = logging.getLogger(__name__)

_DEAL_PATH_RE = re.compile(r"^/imp/([^/]+)/deals/([^/]+)$")

_DEAL_YIELD_STREAM_NAME: Optional[str] = os.environ.get("DEAL_YIELD_FEEDBACK_STREAM_NAME")
_DEAL_YIELD_REGION: str = os.environ.get(
    "DEAL_YIELD_FEEDBACK_STREAM_REGION",
    os.environ.get(
        "FEEDBACK_STREAM_REGION",
        os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
    ),
)

_deal_yield_collector: Optional[FeedbackCollector] = None

if _DEAL_YIELD_STREAM_NAME:
    _deal_yield_collector = FeedbackCollector(
        stream_name=_DEAL_YIELD_STREAM_NAME,
        region=_DEAL_YIELD_REGION,
    )


def _adjust_deal_mutations(mutations: list[Mutation]) -> list[Mutation]:
    """Filters to mutations that carry an adjust_deal payload (BR-1)."""
    return [m for m in mutations if m.adjust_deal is not None]


def _parse_deal_path(path: str) -> tuple[str, str] | None:
    """Parses an ARTF-proto deal path ("/imp/{imp_id}/deals/{deal_id}").

    Returns None (never raises) if the path doesn't match -- BR-8: skip
    silently rather than emit with fabricated imp_id/deal_id values.
    """
    match = _DEAL_PATH_RE.match(path or "")
    if not match:
        return None
    return match.group(1), match.group(2)


def _find_deal(bid_request: dict, imp_id: str, deal_id: str) -> dict | None:
    """Locates the deal dict in the original bid request, for context
    features and the original bidfloor. Returns None if not found."""
    for imp in bid_request.get("imp", []):
        if imp.get("id") != imp_id:
            continue
        pmp = imp.get("pmp") or {}
        for deal in pmp.get("deals", []):
            if deal.get("id") == deal_id:
                return deal
    return None


def _classify_category_tier(bid_request: dict) -> float:
    """Classifies the same real site/app.cat fields the yield models see.

    Calls shared.yield_features.classify_content_tier directly rather than
    reimplementing it: the outcome event this feeds must describe the request
    the way the model that acted on it did, so a second independent
    implementation could drift and silently mislabel training data. (An earlier
    version of this docstring claimed it was NOT a call into that module while
    the code below imported it anyway -- the function now lives in shared/,
    where both the containers and this reader legitimately depend on it.)
    """
    from shared.yield_features import classify_content_tier

    site = bid_request.get("site") or bid_request.get("app") or {}
    return classify_content_tier(site.get("cat", []))


def _build_deal_yield_event(
    req: RTBRequest,
    mutation: Mutation,
    *,
    source: str,
    model_version: str,
) -> DealYieldOutcomeEvent | None:
    """Builds one DealYieldOutcomeEvent from a single adjust_deal mutation.

    Returns None (skip, never raises) if the mutation's path is malformed
    or the referenced deal can't be found in the original request -- a
    real "can't build this event", never a fabricated one.
    """
    parsed = _parse_deal_path(mutation.path)
    if parsed is None:
        return None
    imp_id, deal_id = parsed

    bid_request = req.bid_request or {}
    deal = _find_deal(bid_request, imp_id, deal_id)
    if deal is None:
        return None

    original_bidfloor = float(deal.get("bidfloor", 0.0) or 0.0)
    adjust_deal = mutation.adjust_deal

    intent = "ADJUST_DEAL_FLOOR" if adjust_deal.bidfloor is not None else "ADJUST_DEAL_MARGIN"

    now = datetime.now(timezone.utc)

    try:
        category_tier = _classify_category_tier(bid_request)
    except Exception:
        category_tier = 0.0

    return DealYieldOutcomeEvent(
        request_id=req.id,
        timestamp=time.time(),
        model_version=model_version,
        source=source,
        imp_id=imp_id,
        deal_id=deal_id,
        intent=intent,
        original_bidfloor=original_bidfloor,
        adjusted_bidfloor=adjust_deal.bidfloor,
        margin_value=adjust_deal.margin.value if adjust_deal.margin else None,
        margin_calculation_type=adjust_deal.margin.calculation_type if adjust_deal.margin else None,
        won=False,
        price_paid=None,
        auction_type=deal.get("at"),
        category_tier=category_tier,
        hour_of_day=now.hour,
        day_of_week=now.weekday(),
    )


# ---------------------------------------------------------------------------
# Load-test-origin outcome synthesis
#
# Per deal-yield-outcome-capture's functional design (Logic Flow 4): "for
# load-test-origin events (Unit 3's genesis-bootstrap use case), the load
# test framework's own known synthetic outcome is used directly, mirroring
# emit_load_test_bid_outcome()'s precedent." This is that precedent applied
# to deal-yield: won/price_paid are real "unknown" (False/None) on live
# traffic (BR-7 -- no downstream-signal path exists for deal-level outcomes
# yet), but a load test's own synthetic scenario has a known outcome, so it
# is used directly rather than left unknown -- deliberate bootstrap data,
# not a live-traffic value.
#
# Reuses the same fixed illustrative scenario / deterministic sampling
# approach loadtest_instrumentation.py already established for bid-shading
# load tests (BR-2 precedent: no new ad hoc generator), rather than inventing
# a second one for deal-yield.
# ---------------------------------------------------------------------------

_LOAD_TEST_SCENARIO = BidOutcomeMetrics(
    total_bids=1000,
    wins=400,
    avg_price_paid=3.20,
    avg_shaded_price=3.50,
    total_revenue=1800.0,
    total_cost=1280.0,
)
_LOAD_TEST_SAMPLE_POOL_SIZE = 100


def _synthesize_load_test_outcome(run_id: str, request_index: int, deal_id: str) -> tuple[bool, float | None]:
    """Deterministically derives (won, price_paid) for one load-test deal
    outcome from the same fixed scenario pool loadtest_instrumentation.py
    uses for bid-shading. Keyed by (run_id, request_index, deal_id) so a
    request with two adjust_deal mutations (floor + margin, same deal) gets
    the SAME synthetic outcome for both -- they describe the same deal
    clearing, not two independent events.
    """
    records = _LOAD_TEST_SCENARIO.sample_outcomes(n=_LOAD_TEST_SAMPLE_POOL_SIZE, seed=4242)
    # Indexed via random.Random over a string key rather than builtin hash().
    # hash() is salted per process for str input (unless PYTHONHASHSEED is
    # pinned), so the same (run_id, request_index, deal_id) produced different
    # outcomes in different orchestrator pods -- the docstring's "deterministic"
    # claim only held within a single process. random.Random seeds from a str via
    # sha512, which is stable everywhere.
    #
    # KNOWN LIMITATION (tracked in
    # aidlc-docs/construction/model-comparison-fix/tasks.md, task 2.6): this
    # outcome still does not depend on the yield model's own decision -- the
    # adjusted bidfloor / margin value never enters. The bid-shading path was
    # fixed for this (loadtest_instrumentation.generate_outcome_sample now scores
    # the container's real price against a seeded market), but the equivalent
    # market model for floors/margins is an open design question. Until then, a
    # canary-vs-stable comparison for the yield models would measure run_id, not
    # the model, so do not present one as a model verdict.
    rng = random.Random(f"deal-yield:{run_id}:{request_index}:{deal_id}")
    record = records[rng.randrange(_LOAD_TEST_SAMPLE_POOL_SIZE)]
    return record["won"], record["price_paid"]


def emit_load_test_deal_yield_outcome(
    req: RTBRequest,
    resp: RTBResponse,
    *,
    run_id: str,
    request_index: int,
) -> list[float]:
    """Fire-and-forget: build and emit a DealYieldOutcomeEvent per
    adjust_deal-bearing mutation, with a real, disclosed synthetic
    won/price_paid outcome (source="load_test") instead of the live-path's
    unknown defaults.

    Mirrors emit_deal_yield_outcome()'s exact contract (never raises,
    fire-and-forget via asyncio.create_task, <1ms overhead) -- this is a
    thin wrapper that overrides won/price_paid on the events
    emit_deal_yield_outcome()'s own event-construction logic would
    otherwise build, since _build_deal_yield_event() has no visibility into
    a load test's synthetic scenario.

    Returns the real per-event revenue-like sample value (price_paid if
    won, else 0.0) for each event actually emitted -- 0, 1, or 2 values per
    call, since floor and margin mutations on the same deal are independent
    (BR-5). Callers aggregate this into LoadTestStatus.outcome_samples the
    same way loadtest.py's bid-shading path already does (real per-sample
    data for ComparisonService's ABEvaluator, not a summary statistic).
    """
    if _deal_yield_collector is None:
        return []

    mutations = _adjust_deal_mutations(resp.mutations)
    if not mutations:
        return []

    model_version = resp.metadata.model_version if resp.metadata else ""
    samples: list[float] = []

    for mutation in mutations:
        try:
            event = _build_deal_yield_event(
                req, mutation, source="load_test", model_version=model_version
            )
            if event is None:
                continue
            won, price_paid = _synthesize_load_test_outcome(run_id, request_index, event.deal_id)
            # DealYieldOutcomeEvent is frozen (model_config=ConfigDict(frozen=True)) --
            # model_copy(update=...) is pydantic's supported way to derive a
            # modified copy of an immutable model.
            event = event.model_copy(update={"won": won, "price_paid": price_paid})
            fire_and_forget_emit(
                _deal_yield_collector.emit(event),
                description=f"load-test deal yield outcome emit (deal={event.deal_id})",
            )
            samples.append(price_paid if won and price_paid is not None else 0.0)
        except Exception:
            logger.warning("Failed to emit load-test deal yield outcome event", exc_info=True)

    return samples


def emit_deal_yield_outcome(
    req: RTBRequest,
    resp: RTBResponse,
    *,
    source: str = "live",
) -> None:
    """Fire-and-forget: build and emit a DealYieldOutcomeEvent for each
    adjust_deal-bearing mutation in resp.mutations.

    Mirrors emit_bid_outcome()'s exact contract: never raises, fire-and-
    forget via asyncio.create_task, < 1ms overhead. Called from
    orchestrator/app.py immediately after the existing emit_bid_outcome()
    call -- both run independently.
    """
    if _deal_yield_collector is None:
        return

    mutations = _adjust_deal_mutations(resp.mutations)
    if not mutations:
        return

    model_version = resp.metadata.model_version if resp.metadata else ""

    for mutation in mutations:
        try:
            event = _build_deal_yield_event(
                req, mutation, source=source, model_version=model_version
            )
            if event is not None:
                fire_and_forget_emit(
                    _deal_yield_collector.emit(event),
                    description=f"deal yield outcome emit (deal={event.deal_id})",
                )
        except Exception:
            logger.warning("Failed to emit deal yield outcome event", exc_info=True)
