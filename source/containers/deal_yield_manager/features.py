"""Pure feature-engineering for the Yield Optimizer (deal floor/margin) model.

No I/O, no Triton calls, no randomness -- every value is derived from a real
field on the bid request or the real wall-clock time. Testable in isolation
(see source/tests/test_deal_yield_features.py).

Feature vector shape (fixed, 7 elements, never varies with input
completeness -- see business-rules.md BR-1):
    [0] is_first_price   (0.0/1.0, from deal.at == 1)
    [1] is_second_price  (0.0/1.0, from deal.at == 2)
    [2] bidfloor         (raw float, deal.bidfloor)
    [3] bidfloor_tier    (0.0=remnant, 1.0=mid, 2.0=premium)
    [4] category_tier    (0.0=none, 1.0=standard, 2.0=premium)
    [5] hour_norm        (real UTC hour / 24, [0,1])
    [6] weekday_norm     (real UTC weekday / 7, [0,1])
"""

from __future__ import annotations

from datetime import datetime, timezone

FEATURE_VECTOR_LENGTH = 7

_REMNANT_BIDFLOOR_MAX = 1.0
_PREMIUM_BIDFLOOR_MIN = 5.0

# IAB tier-1 codes classified as premium content for floor/margin purposes.
# Deliberately independent of widedeep_segment_activator's own
# _IAB_SEGMENT_MAP -- see business-rules.md BR-2 (container atomicity;
# no cross-container/shared-module dependency even where category codes
# overlap).
_PREMIUM_IAB_CATEGORIES: frozenset[str] = frozenset({
    "IAB17",  # Sports
    "IAB1",   # Arts & Entertainment
    "IAB9",   # Hobbies & Interests
    "IAB19",  # Technology & Computing
})
_STANDARD_IAB_CATEGORIES: frozenset[str] = frozenset({
    "IAB2", "IAB8", "IAB13", "IAB18", "IAB20",
})


def classify_content_tier(iab_categories: list[str]) -> float:
    """Maps IAB tier-1 codes to an ordinal content-value tier.

    Returns 2.0 (premium) if any category is in the premium set, else 1.0
    (standard) if any category is in the standard set, else 0.0 (none /
    unclassified). Pure, deterministic, never raises on unexpected input.
    """
    categories = set(iab_categories or [])
    if categories & _PREMIUM_IAB_CATEGORIES:
        return 2.0
    if categories & _STANDARD_IAB_CATEGORIES:
        return 1.0
    return 0.0


def _bidfloor_tier(bidfloor: float) -> float:
    """Ordinal tier for a deal's existing bidfloor.

    Thresholds bracket this project's own sample data range
    ($0.50-$10.00, see source/samples/video-deals.json): <= $1.00 is
    remnant, >= $5.00 is premium, otherwise mid-tier.
    """
    if bidfloor <= _REMNANT_BIDFLOOR_MAX:
        return 0.0
    if bidfloor >= _PREMIUM_BIDFLOOR_MIN:
        return 2.0
    return 1.0


def _auction_type_onehot(at: int | None) -> tuple[float, float]:
    """Returns (is_first_price, is_second_price). Unrecognized/missing
    `at` values encode to (0.0, 0.0) -- a real "neither", never fabricated.
    """
    if at == 1:
        return 1.0, 0.0
    if at == 2:
        return 0.0, 1.0
    return 0.0, 0.0


def build_feature_vector(
    bid_request: dict, deal: dict, request_time: datetime | None = None
) -> list[float]:
    """Builds the fixed-length feature vector for one deal.

    Args:
        bid_request: The full OpenRTB bid request (used for site/app
            category context).
        deal: One entry from imp[].pmp.deals[] (has at least `bidfloor`,
            `at`; `id` is read by the caller, not needed here).
        request_time: Real wall-clock time at request-processing time.
            Defaults to datetime.now(timezone.utc) if not supplied (tests
            pass a fixed value for determinism).

    Returns:
        A list of exactly FEATURE_VECTOR_LENGTH floats. Never raises --
        missing/malformed fields encode to a defined neutral value.
    """
    if request_time is None:
        request_time = datetime.now(timezone.utc)

    at = deal.get("at")
    is_first_price, is_second_price = _auction_type_onehot(at)

    bidfloor = float(deal.get("bidfloor", 0.0) or 0.0)
    bidfloor_tier = _bidfloor_tier(bidfloor)

    site = bid_request.get("site") or bid_request.get("app") or {}
    category_tier = classify_content_tier(site.get("cat", []))

    hour_norm = request_time.hour / 24.0
    weekday_norm = request_time.weekday() / 7.0

    return [
        is_first_price,
        is_second_price,
        bidfloor,
        bidfloor_tier,
        category_tier,
        hour_norm,
        weekday_norm,
    ]
