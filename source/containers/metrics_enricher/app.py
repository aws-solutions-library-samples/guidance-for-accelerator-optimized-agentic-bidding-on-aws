"""Metrics Enricher — ARTF container for ADD_METRICS (rule-based)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    Intent, Metadata, Metric, AddMetricsPayload, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)
from shared import iab_taxonomy

MODEL_VERSION = "metrics-rules-v2-iab-taxonomy"
# Content Taxonomy 1.0 codes treated as brand-safe. This list is not a considered
# suitability judgement and never was -- it excludes Science, Pets, Style & Fashion,
# Real Estate and Shopping, which nobody would argue are unsafe. It is preserved
# byte-for-byte anyway, because changing it would change the output of every existing
# request that declares no `cattax`.
_SAFE_CATS = {"IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7",
              "IAB8", "IAB9", "IAB10", "IAB12", "IAB13", "IAB17", "IAB19", "IAB20"}

# Content Taxonomy 3.x tier-1 categories treated as UNSUITABLE. Defined by exclusion
# because the safe set is the large one, and stated as this project's judgement --
# IAB publishes no brand-safety classification in the Content Taxonomy, so this
# cannot be presented as a standard (the same caveat that applies to the
# content-to-audience mapping).
#
# These seven are the tier-1 categories that name their own unsuitability. Note what
# is deliberately NOT used here: the Special Category Data flag. SCD is a privacy
# control about what may be inferred from what someone read, and says nothing about
# whether an advertiser wants to appear beside it. Tier-1 186 "Family and
# Relationships" is SCD-flagged and is plainly brand-safe; category 380 "Crime" is
# not SCD-flagged and plainly is not.
_UNSUITABLE_TIER1_3X = {
    "Crime",
    "Disasters",
    "Law",
    "Politics",
    "Religion & Spirituality",
    "Sensitive Topics",
    "War and Conflicts",
}


def _viewability(imp: dict) -> float:
    pos = imp.get("pos", 0)
    base = 0.70 if pos == 1 else 0.50 if pos in (0, 3) else 0.35
    if imp.get("banner"):
        area = imp["banner"].get("w", 300) * imp["banner"].get("h", 250)
        base += 0.10 if area >= 250000 else 0.05 if area >= 90000 else 0.0
    if imp.get("video"):
        base += 0.12
    return min(1.0, base)


def _brand_safety(site: dict) -> float:
    """Fraction of the site's categories judged brand-safe, scaled into [0.60, 1.00].

    Which taxonomy the codes belong to comes from the request's own ``cattax``, the
    same way the Audience Activator reads it. Before this, only Content Taxonomy 1.0
    codes were recognised, so a request declaring 3.x scored 0.60 -- the floor for
    "has categories, none recognised" -- and therefore scored WORSE than a request
    carrying no categories at all, which scores 0.80. Declaring a modern taxonomy was
    penalised.
    """
    # `or []` as well as the default: a request may carry `"cat": null`, and the
    # pre-change implementation raised TypeError on it.
    cats = [c for c in (site.get("cat") or []) if isinstance(c, str)]
    if not cats:
        return 0.80

    taxonomy = iab_taxonomy.resolve_taxonomy(site.get("cattax"))

    if taxonomy == iab_taxonomy.TAXONOMY_CONTENT_3_X:
        safe = 0
        for cat in cats:
            tier1 = iab_taxonomy.content_tier1(cat)
            # An unknown code is not assumed safe. It contributes nothing, exactly as
            # an unrecognised 1.0 code does.
            if tier1 is not None and tier1 not in _UNSUITABLE_TIER1_3X:
                safe += 1
        return min(1.0, 0.60 + 0.40 * (safe / len(cats)))

    # Content Taxonomy 1.0, absent, or unrecognised: unchanged.
    return min(1.0, 0.60 + 0.40 * (len(set(cats) & _SAFE_CATS) / len(cats)))


def mutate(req: RTBRequest) -> RTBResponse:
    if not intent_applicable(Intent.ADD_METRICS, req.applicable_intents):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    site = req.bid_request.get("site", req.bid_request.get("app", {}))
    bs = _brand_safety(site)
    mutations = []
    for imp in req.bid_request.get("imp", []):
        v = _viewability(imp)
        mutations.append(Mutation(
            intent=Intent.ADD_METRICS, op=Operation.ADD,
            path=f"/imp/{imp.get('id', '')}/metric",
            add_metrics=AddMetricsPayload(metric=[
                Metric(type="viewability", value=round(v, 4), vendor="nvidia-artf"),
                Metric(type="brand_safety", value=round(bs, 4), vendor="nvidia-artf"),
            ]),
        ))
    return RTBResponse(id=req.id, mutations=mutations, metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION))


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="metrics-enricher", grpc_port=50051, mcp_port=8081, health_port=8080)
