"""Wide & Deep Segment Activator — ARTF container for ACTIVATE_SEGMENTS.

Implements the Wide & Deep architecture (Cheng et al. 2016) as used in
NVIDIA Merlin Models and NVIDIA DeepLearningExamples.

- NVIDIA Merlin: https://nvidia-merlin.github.io/models/
- Paper: https://arxiv.org/abs/1606.07792

When USE_TRITON=true, inference is delegated to NVIDIA Triton Inference
Server via tritonclient.http.  Otherwise, PyTorch runs inline (CPU).
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    IDsPayload, Intent, Metadata, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)

USE_TRITON = os.environ.get("USE_TRITON", "").lower() in ("1", "true", "yes")

CANDIDATE_SEGMENTS = [
    "demo-18-24", "demo-25-34", "demo-35-44", "demo-45-54",
    "int-sports", "int-tech", "int-fashion", "int-auto",
    "int-finance", "int-travel", "int-gaming", "int-food",
    "ctx-premium", "ctx-mobile", "ctx-video",
]
NUM_SEGMENTS = len(CANDIDATE_SEGMENTS)
ACTIVATION_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Inference backend — Triton or PyTorch
# ---------------------------------------------------------------------------

if USE_TRITON:
    import numpy as np
    from container.triton_inference import predict_segments as _triton_predict_segments

    MODEL_VERSION = "widedeep-nvidia-triton-v1"

    def _score_segments(bid_request: dict, threshold: float = ACTIVATION_THRESHOLD) -> list[str]:
        wide, deep = _extract_np(bid_request)
        scores = _triton_predict_segments(wide, deep)
        return [CANDIDATE_SEGMENTS[i] for i in range(min(len(scores), NUM_SEGMENTS))
                if scores[i] > threshold]

    def _extract_np(br: dict):
        user, site, device = br.get("user", {}), br.get("site", {}), br.get("device", {})
        imp = (br.get("imp") or [{}])[0]
        age = max(0.0, min(1.0, (2025 - user.get("yob", 1990)) / 80.0))
        g = 0.5 if user.get("gender", "") not in ("M", "F") else (0.0 if user.get("gender") == "M" else 1.0)
        cats = site.get("cat", [])
        ch, dh = _h(cats[0] if cats else "x"), _h(site.get("domain", "x"))
        uh, gh = _h(device.get("ua", "")[:30]), _h((device.get("geo") or {}).get("country", "US"))
        bf = float(imp.get("bidfloor", 1.0)) / 20.0
        vid = 1.0 if imp.get("video") else 0.0
        wide = np.array([[age*ch, g*dh, gh*uh, bf*vid, age*g, ch*gh, dh*bf, uh*vid]], dtype=np.float32)
        deep = np.array([[age, g, ch, dh, bf, vid]], dtype=np.float32)
        return wide, deep

else:
    import torch
    import torch.nn as nn

    class WideAndDeepModel(nn.Module):
        """Wide & Deep model following NVIDIA Merlin's WideAndDeepModel architecture."""

        def __init__(self, num_wide: int = 8, num_deep: int = 6, num_outputs: int = 15):
            super().__init__()
            self.wide = nn.Linear(num_wide, num_outputs)
            self.deep = nn.Sequential(
                nn.Linear(num_deep, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Linear(32, num_outputs),
            )

        def forward(self, wide_features, deep_features):
            return torch.sigmoid(self.wide(wide_features) + self.deep(deep_features))

    _model = WideAndDeepModel()
    _model.eval()
    MODEL_VERSION = "widedeep-nvidia-merlin-arch-v1"

    def _score_segments(bid_request: dict, threshold: float = ACTIVATION_THRESHOLD) -> list[str]:
        wide, deep = _extract_torch(bid_request)
        with torch.no_grad():
            scores = _model(wide, deep).squeeze(0)
        return [CANDIDATE_SEGMENTS[i] for i in range(min(len(scores), NUM_SEGMENTS))
                if scores[i].item() > threshold]

    def _extract_torch(br: dict):
        user, site, device = br.get("user", {}), br.get("site", {}), br.get("device", {})
        imp = (br.get("imp") or [{}])[0]
        age = max(0.0, min(1.0, (2025 - user.get("yob", 1990)) / 80.0))
        g = 0.5 if user.get("gender", "") not in ("M", "F") else (0.0 if user.get("gender") == "M" else 1.0)
        cats = site.get("cat", [])
        ch, dh = _h(cats[0] if cats else "x"), _h(site.get("domain", "x"))
        uh, gh = _h(device.get("ua", "")[:30]), _h((device.get("geo") or {}).get("country", "US"))
        bf = float(imp.get("bidfloor", 1.0)) / 20.0
        vid = 1.0 if imp.get("video") else 0.0
        wide = torch.tensor([[age*ch, g*dh, gh*uh, bf*vid, age*g, ch*gh, dh*bf, uh*vid]], dtype=torch.float32)
        deep = torch.tensor([[age, g, ch, dh, bf, vid]], dtype=torch.float32)
        return wide, deep


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _h(val: str, dim: int = 1000) -> float:
    return (int(hashlib.md5(val.encode(), usedforsecurity=False).hexdigest(), 16) % dim) / dim  # nosec B324


# ---------------------------------------------------------------------------
# ARTF mutate
# ---------------------------------------------------------------------------

def mutate(req: RTBRequest) -> RTBResponse:
    if not intent_applicable(Intent.ACTIVATE_SEGMENTS, req.applicable_intents):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    # Read model parameter overrides from the request (frontend sliders)
    params = req.model_params or {}
    threshold = params.get('segment_threshold', ACTIVATION_THRESHOLD)

    activated = _score_segments(req.bid_request, threshold=threshold)

    mutations = []
    if activated:
        mutations.append(Mutation(intent=Intent.ACTIVATE_SEGMENTS, op=Operation.ADD,
                                  path="/user/data/segment", ids=IDsPayload(id=activated)))
    return RTBResponse(id=req.id, mutations=mutations,
                       metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION))


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="widedeep-segment-activator", grpc_port=50051, mcp_port=8081, health_port=8080)
