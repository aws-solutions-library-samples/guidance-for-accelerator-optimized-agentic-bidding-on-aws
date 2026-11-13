"""Segment Activator — ARTF container for ACTIVATE_SEGMENTS.

Deterministic, rule-based audience-segment activation. This container
previously scored segments with a Wide & Deep neural network served on
NVIDIA Triton. That model is being replaced by a partner ISV implementation
(see GUIDANCE.md — "[Future] Location Activator (ISV)" and related entries).
Until that partner model is wired in, this container activates segments from
transparent, inspectable rules over real bid-request signals:

- IAB content-category → interest-segment mapping (site/app ``cat``)
- Age bucketing from the user's year of birth (``user.yob``)

Taxonomy support (see shared/iab_taxonomy):

- The request's ``cattax`` selects which taxonomy its category codes belong to.
  Absent or 1 keeps the legacy Content Taxonomy 1.0 behaviour unchanged; 7 and 9
  resolve against Content Taxonomy 3.0/3.1 and emit real IAB Audience Taxonomy
  1.1 segment identifiers. Content Taxonomy 2.x is unrecognised, because its
  identifiers are not interchangeable with 3.x.
- A content category carrying IAB's Special Category Data flag derives no
  segment and contributes no score. That flag is a privacy control: it marks
  categories that could be used to build sensitive profiles about a person, so
  inferring a user interest segment from one is the pattern it warns against.
- Age buckets are IAB Audience Taxonomy identifiers using the taxonomy's own
  five-year ranges, replacing the ten-year ranges this container invented. This
  applies on every path, including legacy requests, because the bucket format is
  a property of the output space rather than of the incoming content taxonomy.

Contextual segments (``ctx-``) and keyword-reclassified segments (``int-``) have
no IAB equivalent and keep their vendor prefix. Standard and vendor identifiers
share one mutation because ARTF's ``IDsPayload`` has no field for per-identifier
provenance; the prefix is what distinguishes them.
- Keyword matching against any first/third-party DMP segments already present
  on the request (``user.data[].segment[].name``)
- Contextual signals: bid floor tier, mobile user-agent, video inventory

No model inference, no randomness, no fabricated scores — every activation is
traceable to a specific rule and a specific field in the bid request.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    IDsPayload, Intent, Metadata, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)
from shared import iab_taxonomy

MODEL_VERSION = "segment-rules-v2-iab-taxonomy"
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

_MOBILE_UA_MARKERS = ("Mobile", "iPhone", "Android")

# `_AGE_BUCKETS` is gone. Age buckets now come from IAB Audience Taxonomy 1.1's own
# Age Range rows (see shared/iab_taxonomy), so they cannot drift from the standard,
# and they are five-year ranges rather than the ten-year ranges invented here.


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

    # Rule 1: content category -> interest segment.
    #
    # Which taxonomy the codes belong to comes from the request's own `cattax`.
    # The legacy path is preserved byte-for-byte so requests that do not declare a
    # modern taxonomy behave exactly as before.
    categories = [c for c in site.get("cat", []) if isinstance(c, str)]
    taxonomy = iab_taxonomy.resolve_taxonomy(site.get("cattax"))

    if taxonomy == iab_taxonomy.TAXONOMY_CONTENT_1_0:
        for cat in categories:
            segment = _IAB_SEGMENT_MAP.get(cat)
            if segment:
                _bump(segment, 0.85)
    elif taxonomy == iab_taxonomy.TAXONOMY_CONTENT_3_X:
        # A withheld category contributes no score at all. It must not push a
        # segment over the threshold that it is barred from deriving.
        for outcome in iab_taxonomy.segments_for_categories(categories):
            if outcome.disposition == iab_taxonomy.MAPPED and outcome.segment_id:
                _bump(outcome.segment_id, 0.85)
    # An unrecognised taxonomy derives nothing from categories. Every other rule
    # below still runs, because they read fields unrelated to the content taxonomy.

    # Rule 2: age bucket from year of birth.
    #
    # Emits an IAB Audience Taxonomy identifier using the taxonomy's own five-year
    # ranges. This applies on every path, legacy included: the bucket format belongs
    # to the output space, and `cattax` describes incoming *content* categories, so
    # gating the demographic format on it would conflate two unrelated things.
    age_segment = iab_taxonomy.age_bucket_id(user.get("yob"))
    if age_segment:
        _bump(age_segment, 0.9)

    # Rule 3: reclassify audience data the request already carries.
    #
    # These are segments a first or third party has already asserted about the
    # person. Translating someone else's assertion into the standard taxonomy is not
    # an inference, which is why this rule can reach segments that Rule 1 must not:
    # life-stage segments such as Audience Taxonomy 98 "Parents with Children" are
    # reachable ONLY from asserted data. Deriving one from page content would be
    # inferring a personal circumstance from what someone happened to read, which is
    # the inference IAB's Special Category Data flag exists to discourage.
    #
    # A name that matches a taxonomy node exactly (after normalising case and
    # separators) yields that real identifier. Anything else falls through to the
    # keyword map, so names this taxonomy does not describe still activate a vendor
    # segment as they did before.
    for provider in user.get("data", []):
        for seg in provider.get("segment", []):
            raw_name = seg.get("name") or seg.get("id") or ""
            asserted = iab_taxonomy.segment_for_asserted_name(raw_name)
            if asserted:
                _bump(asserted, 0.75)
                continue
            name = raw_name.lower()
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
