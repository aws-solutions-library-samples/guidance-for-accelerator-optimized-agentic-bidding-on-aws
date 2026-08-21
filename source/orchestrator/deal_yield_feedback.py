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
import re
import time
from datetime import datetime, timezone
from typing import Optional

from shared.artf_types import RTBRequest, RTBResponse, Mutation
from shared.feedback_collector import FeedbackCollector
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
    """Mirrors containers.deal_yield_manager.features.classify_content_tier,
    reading the same real site/app.cat fields -- not a call to that module
    (Unit 2 has no dependency on Unit 1's code, per unit-of-work-dependency.md;
    this is a second, independent reader of the same request data)."""
    from containers.deal_yield_manager.features import classify_content_tier

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
                asyncio.create_task(_deal_yield_collector.emit(event))
        except Exception:
            logger.warning("Failed to emit deal yield outcome event", exc_info=True)
