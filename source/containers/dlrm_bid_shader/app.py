"""DLRM Bid Shader — ARTF container using NVIDIA DLRM for BID_SHADE.

Implements the Deep Learning Recommendation Model (DLRM) architecture
as defined in NVIDIA's DeepLearningExamples and optimized by Meta's
torchrec library.

- NVIDIA DLRM: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM
- torchrec DLRM: https://github.com/pytorch/torchrec (torchrec.models.dlrm.DLRM)
- Paper: https://arxiv.org/abs/1906.00091

When USE_TRITON=true, inference is delegated to NVIDIA Triton Inference
Server via tritonclient.http.  Otherwise, PyTorch runs inline (CPU).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    AdjustBidPayload, Intent, Metadata, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)

USE_TRITON = os.environ.get("USE_TRITON", "").lower() in ("1", "true", "yes")

EMBEDDING_DIM = 16
NUM_DENSE = 4
NUM_SPARSE = 3
VOCAB_SIZE = 1000
SHADE_FACTOR = 0.65
EST_CONVERSION_VALUE = 12.0


# ---------------------------------------------------------------------------
# Inference backend — Triton or PyTorch
# ---------------------------------------------------------------------------

if USE_TRITON:
    import numpy as np
    from container.triton_inference import predict_ctr as _triton_predict_ctr

    MODEL_VERSION = "dlrm-nvidia-triton-v1"

    def _predict_ctr_from_request(bid_request: dict) -> float:
        dense, s_user, s_domain, s_device = _extract_features_np(bid_request)
        return _triton_predict_ctr(
            dense_features=dense,
            sparse_user=s_user,
            sparse_domain=s_domain,
            sparse_device=s_device,
        )

    def _extract_features_np(bid_request: dict):
        imps = bid_request.get("imp", [{}])
        imp = imps[0] if imps else {}
        user = bid_request.get("user", {})
        site = bid_request.get("site", {})
        device = bid_request.get("device", {})

        bidfloor = float(imp.get("bidfloor", 1.0))
        hour = time.localtime().tm_hour / 24.0
        age_norm = max(0.0, min(1.0, (2025 - user.get("yob", 1990)) / 80.0))
        has_video = 1.0 if imp.get("video") else 0.0

        dense = np.array([[bidfloor, hour, age_norm, has_video]], dtype=np.float32)
        s_user = np.array([[_hash_to_idx(user.get("id", "unknown"))]], dtype=np.int64)
        s_domain = np.array([[_hash_to_idx(site.get("domain", "unknown"))]], dtype=np.int64)
        s_device = np.array([[_hash_to_idx(device.get("ua", "unknown")[:20])]], dtype=np.int64)
        return dense, s_user, s_domain, s_device

else:
    import torch
    import torch.nn as nn

    class DLRMModel(nn.Module):
        """DLRM following NVIDIA's DeepLearningExamples architecture."""

        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.EmbeddingBag(VOCAB_SIZE, EMBEDDING_DIM, mode="sum")
                for _ in range(NUM_SPARSE)
            ])
            self.bottom_mlp = nn.Sequential(
                nn.Linear(NUM_DENSE, 32), nn.ReLU(),
                nn.Linear(32, EMBEDDING_DIM), nn.ReLU(),
            )
            interaction_size = EMBEDDING_DIM + (NUM_SPARSE + 1) * NUM_SPARSE // 2
            self.top_mlp = nn.Sequential(
                nn.Linear(interaction_size, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )

        def _interaction(self, dense_out, sparse_outs):
            all_embeds = torch.stack([dense_out] + sparse_outs, dim=1)
            interactions = torch.bmm(all_embeds, all_embeds.transpose(1, 2))
            triu_indices = torch.triu_indices(NUM_SPARSE + 1, NUM_SPARSE + 1, offset=1)
            flat = interactions[:, triu_indices[0], triu_indices[1]]
            return torch.cat([dense_out, flat], dim=1)

        def forward(self, dense, sparse_indices):
            dense_out = self.bottom_mlp(dense)
            sparse_outs = [
                emb(idx.unsqueeze(0) if idx.dim() == 0 else idx.unsqueeze(1))
                for emb, idx in zip(self.embeddings, sparse_indices)
            ]
            interaction_out = self._interaction(dense_out, sparse_outs)
            return torch.sigmoid(self.top_mlp(interaction_out)).squeeze(-1)

    _model = DLRMModel()
    torch.manual_seed(42)
    for m in _model.modules():
        if isinstance(m, (nn.Linear, nn.EmbeddingBag)):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight) if m.weight.dim() > 1 else nn.init.normal_(m.weight, std=0.01)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
    _model.eval()
    MODEL_VERSION = "dlrm-nvidia-arch-v1"

    def _predict_ctr_from_request(bid_request: dict) -> float:
        dense, sparse = _extract_features_torch(bid_request)
        with torch.no_grad():
            return _model(dense, sparse).item()

    def _extract_features_torch(bid_request: dict):
        imps = bid_request.get("imp", [{}])
        imp = imps[0] if imps else {}
        user = bid_request.get("user", {})
        site = bid_request.get("site", {})
        device = bid_request.get("device", {})

        bidfloor = float(imp.get("bidfloor", 1.0))
        hour = time.localtime().tm_hour / 24.0
        age_norm = max(0.0, min(1.0, (2025 - user.get("yob", 1990)) / 80.0))
        has_video = 1.0 if imp.get("video") else 0.0

        dense = torch.tensor([[bidfloor, hour, age_norm, has_video]], dtype=torch.float32)
        sparse = [
            torch.tensor([_hash_to_idx(user.get("id", "unknown"))]),
            torch.tensor([_hash_to_idx(site.get("domain", "unknown"))]),
            torch.tensor([_hash_to_idx(device.get("ua", "unknown")[:20])]),
        ]
        return dense, sparse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hash_to_idx(value: str, vocab: int = VOCAB_SIZE) -> int:
    return int(hashlib.md5(value.encode(), usedforsecurity=False).hexdigest(), 16) % vocab  # nosec B324


# ---------------------------------------------------------------------------
# ARTF mutate
# ---------------------------------------------------------------------------

def mutate(req: RTBRequest) -> RTBResponse:
    """ARTF GetMutations — BID_SHADE via DLRM CTR prediction."""
    if not intent_applicable(Intent.BID_SHADE, req.applicable_intents):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    bid_response = req.bid_response
    if not bid_response:
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    # Read model parameter overrides from the request (frontend sliders)
    params = req.model_params or {}
    shade_factor = params.get('shade_factor', SHADE_FACTOR)
    conversion_value = params.get('conversion_value', EST_CONVERSION_VALUE)

    predicted_ctr = _predict_ctr_from_request(req.bid_request)

    mutations: list[Mutation] = []
    for seatbid in bid_response.get("seatbid", []):
        seat = seatbid.get("seat", "unknown")
        for bid in seatbid.get("bid", []):
            original_price = bid.get("price", 0.0)
            if original_price <= 0:
                continue
            ev = predicted_ctr * conversion_value
            shaded = min(original_price, ev * shade_factor)
            imp_id = bid.get("impid", "")
            floor = 0.0
            for imp in req.bid_request.get("imp", []):
                if imp.get("id") == imp_id:
                    floor = imp.get("bidfloor", 0.0)
                    break
            shaded = max(shaded, floor)
            if abs(shaded - original_price) > 0.01:
                mutations.append(Mutation(
                    intent=Intent.BID_SHADE, op=Operation.REPLACE,
                    path=f"/seatbid/{seat}/bid/{bid.get('id', '')}",
                    adjust_bid=AdjustBidPayload(price=round(shaded, 4)),
                ))

    return RTBResponse(id=req.id, mutations=mutations, metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION))


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="dlrm-bid-shader", grpc_port=50051, mcp_port=8081, health_port=8080)
