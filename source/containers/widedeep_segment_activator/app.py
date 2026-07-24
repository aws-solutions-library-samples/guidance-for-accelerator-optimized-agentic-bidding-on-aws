"""Segment Activator — ARTF container for ACTIVATE_SEGMENTS.

Deterministic, rule-based audience-segment activation. This container
previously scored segments with a Wide & Deep neural network served on
NVIDIA Triton. That model is being replaced by a partner ISV implementation
(see GUIDANCE.md — "[Future] Location Activator (ISV)" and related entries).
Until that partner model is wired in, this container activates segments from
transparent, inspectable rules over real bid-request signals:

- IAB content-category → interest-segment mapping (site/app ``cat``)
- Age bucketing from the user's year of birth (``user.yob``)
- Keyword matching against any first/third-party DMP segments already present
  on the request (``user.data[].segment[].name``)
- Contextual signals: bid floor tier, mobile user-agent, video inventory

No model inference, no randomness, no fabricated scores — every activation is
traceable to a specific rule and a specific field in the bid request.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    IDsPayload, Intent, Metadata, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)

MODEL_VERSION = "segment-rules-v1"
ACTIVATION_THRESHOLD = 0.55

# IAB content-category (tier-1) -> interest segment. Real IAB2 taxonomy codes.
_IAB_SEGMENT_MAP: dict[str, str] = {
    "IAB2": "int-auto",
    "IAB8": "int-food",
    "IAB9": "int-gaming",
    "IAB13": "int-finance",
    "IAB17": "int-sports",
    "IAB18": "int-fashion",
    "IAB19": "int-tech",
    "IAB20": "int-travel",
}

# Keyword -> interest segment, matched against existing DMP segment names
# already present on the request (real first/third-party signal, not inferred).
_KEYWORD_SEGMENT_MAP: tuple[tuple[str, str], ...] = (
    ("sport", "int-sports"),
    ("auto", "int-auto"),
    ("travel", "int-travel"),
    ("fashion", "int-fashion"),
    ("style", "int-fashion"),
    ("finance", "int-finance"),
    ("food", "int-food"),
    ("gam", "int-gaming"),
    ("tech", "int-tech"),
)

_AGE_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (18, 24, "demo-18-24"),
    (25, 34, "demo-25-34"),
    (35, 44, "demo-35-44"),
    (45, 54, "demo-45-54"),
)

_MOBILE_UA_MARKERS = ("Mobile", "iPhone", "Android")


def _score_segments(bid_request: dict, threshold: float = ACTIVATION_THRESHOLD) -> list[str]:
    """Score candidate segments from real bid-request signals via fixed rules.

    Returns the segment IDs whose accumulated score exceeds ``threshold``.
    Scores are deterministic point values from the rules below, clamped to
    [0, 1] — not a model's probability output.
    """
    scores: dict[str, float] = {}

    def _bump(segment: str, value: float) -> None:
        scores[segment] = min(1.0, max(scores.get(segment, 0.0), value))

    site = bid_request.get("site") or bid_request.get("app") or {}
    user = bid_request.get("user") or {}
    device = bid_request.get("device") or {}
    imps = bid_request.get("imp") or [{}]

    # Rule 1: IAB content category -> interest segment
    for cat in site.get("cat", []):
        segment = _IAB_SEGMENT_MAP.get(cat)
        if segment:
            _bump(segment, 0.85)

    # Rule 2: age bucket from year of birth
    yob = user.get("yob")
    if isinstance(yob, int) and 1900 < yob <= datetime.now(timezone.utc).year:
        age = datetime.now(timezone.utc).year - yob
        for lo, hi, segment in _AGE_BUCKETS:
            if lo <= age <= hi:
                _bump(segment, 0.9)
                break

    # Rule 3: keyword match against existing DMP-provided segments (real
    # first/third-party signal already on the request — we only reclassify it
    # into our own segment taxonomy, we don't invent it).
    for provider in user.get("data", []):
        for seg in provider.get("segment", []):
            name = (seg.get("name") or seg.get("id") or "").lower()
            for keyword, segment in _KEYWORD_SEGMENT_MAP:
                if keyword in name:
                    _bump(segment, 0.75)

    # Rule 4: contextual signals from the impression / device
    for imp in imps:
        bidfloor = float(imp.get("bidfloor", 0.0) or 0.0)
        if bidfloor >= 4.0:
            _bump("ctx-premium", 0.8)
        elif bidfloor >= 2.5:
            _bump("ctx-premium", 0.6)
        if imp.get("video"):
            _bump("ctx-video", 0.9)

    ua = device.get("ua", "")
    if any(marker in ua for marker in _MOBILE_UA_MARKERS):
        _bump("ctx-mobile", 0.85)

    return sorted(seg for seg, score in scores.items() if score > threshold)


def mutate(req: RTBRequest) -> RTBResponse:
    if not intent_applicable(Intent.ACTIVATE_SEGMENTS, req.applicable_intents):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    # Read parameter overrides from the request (frontend sliders)
    params = req.model_params or {}
    threshold = params.get("segment_threshold", ACTIVATION_THRESHOLD)

    activated = _score_segments(req.bid_request, threshold=threshold)

    mutations = []
    if activated:
        mutations.append(Mutation(intent=Intent.ACTIVATE_SEGMENTS, op=Operation.ADD,
                                  path="/user/data/segment", ids=IDsPayload(id=activated)))
    return RTBResponse(id=req.id, mutations=mutations,
                       metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION))


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="segment-activator", grpc_port=50051, mcp_port=8081, health_port=8080)
