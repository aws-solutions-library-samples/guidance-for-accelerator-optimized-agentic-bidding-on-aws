"""Metrics Enricher — ARTF container for ADD_METRICS (rule-based)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    Intent, Metadata, Metric, AddMetricsPayload, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)

MODEL_VERSION = "metrics-rules-v1"
_SAFE_CATS = {"IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7",
              "IAB8", "IAB9", "IAB10", "IAB12", "IAB13", "IAB17", "IAB19", "IAB20"}


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
    cats = set(site.get("cat", []))
    if not cats:
        return 0.80
    return min(1.0, 0.60 + 0.40 * (len(cats & _SAFE_CATS) / max(len(cats), 1)))


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
