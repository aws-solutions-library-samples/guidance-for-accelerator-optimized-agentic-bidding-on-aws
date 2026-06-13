"""NCF Deal Manager — ARTF container for ACTIVATE_DEALS / SUPPRESS_DEALS.

Implements Neural Collaborative Filtering (He et al. 2017) following
NVIDIA's DeepLearningExamples NCF implementation.

- NVIDIA NCF: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
- Paper: https://arxiv.org/abs/1708.05031

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

ACTIVATE_THRESHOLD = 0.499
SUPPRESS_THRESHOLD = 0.497


# ---------------------------------------------------------------------------
# Inference backend — Triton or PyTorch
# ---------------------------------------------------------------------------

if USE_TRITON:
    import numpy as np
    from container.triton_inference import predict_relevance as _triton_predict_relevance

    MODEL_VERSION = "ncf-neumf-triton-v1"

    def _score_deals(uid_hash: int, deal_ids: list[str],
                     activate_threshold: float = ACTIVATE_THRESHOLD,
                     suppress_threshold: float = SUPPRESS_THRESHOLD) -> tuple[list[str], list[str]]:
        user_arr = np.array([uid_hash] * len(deal_ids), dtype=np.int64)
        deal_arr = np.array([_h(d) for d in deal_ids], dtype=np.int64)
        scores = _triton_predict_relevance(user_arr, deal_arr)
        to_act = [deal_ids[j] for j in range(len(deal_ids)) if scores[j] >= activate_threshold]
        to_sup = [deal_ids[j] for j in range(len(deal_ids)) if scores[j] < suppress_threshold]
        return to_act, to_sup

else:
    import torch
    import torch.nn as nn

    class NeuMF(nn.Module):
        """NeuMF following NVIDIA's DeepLearningExamples NCF implementation."""

        def __init__(self, nb_users=2000, nb_items=2000, mf_dim=64, mlp_layer_sizes=None):
            super().__init__()
            if mlp_layer_sizes is None:
                mlp_layer_sizes = [256, 128, 64]
            self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
            self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
            mlp_embed_dim = mlp_layer_sizes[0] // 2
            self.mlp_user_embed = nn.Embedding(nb_users, mlp_embed_dim)
            self.mlp_item_embed = nn.Embedding(nb_items, mlp_embed_dim)
            mlp_layers = []
            input_size = mlp_layer_sizes[0]
            for output_size in mlp_layer_sizes[1:]:
                mlp_layers.append(nn.Linear(input_size, output_size))
                mlp_layers.append(nn.ReLU())
                input_size = output_size
            self.mlp = nn.Sequential(*mlp_layers)
            self.final = nn.Sequential(nn.Linear(mf_dim + mlp_layer_sizes[-1], 1), nn.Sigmoid())
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, user_ids, item_ids):
            mf_user = self.mf_user_embed(user_ids)
            mf_item = self.mf_item_embed(item_ids)
            gmf_out = mf_user * mf_item
            mlp_user = self.mlp_user_embed(user_ids)
            mlp_item = self.mlp_item_embed(item_ids)
            mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)
            mlp_out = self.mlp(mlp_in)
            concat = torch.cat([gmf_out, mlp_out], dim=-1)
            return self.final(concat).squeeze(-1)

    _model = NeuMF()
    _model.eval()
    MODEL_VERSION = "ncf-neumf-v1"

    def _score_deals(uid_hash: int, deal_ids: list[str],
                     activate_threshold: float = ACTIVATE_THRESHOLD,
                     suppress_threshold: float = SUPPRESS_THRESHOLD) -> tuple[list[str], list[str]]:
        deal_hashes = torch.tensor([_h(d) for d in deal_ids])
        user_tensor = torch.tensor([uid_hash] * len(deal_ids))
        with torch.no_grad():
            scores = _model(user_tensor, deal_hashes)
        to_act = [deal_ids[j] for j in range(len(deal_ids)) if scores[j].item() >= activate_threshold]
        to_sup = [deal_ids[j] for j in range(len(deal_ids)) if scores[j].item() < suppress_threshold]
        return to_act, to_sup


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _h(v: str, n: int = 2000) -> int:
    return int(hashlib.md5(v.encode(), usedforsecurity=False).hexdigest(), 16) % n  # nosec B324


# ---------------------------------------------------------------------------
# ARTF mutate
# ---------------------------------------------------------------------------

def mutate(req: RTBRequest) -> RTBResponse:
    act_ok = intent_applicable(Intent.ACTIVATE_DEALS, req.applicable_intents)
    sup_ok = intent_applicable(Intent.SUPPRESS_DEALS, req.applicable_intents)
    if not act_ok and not sup_ok:
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    # Read model parameter overrides from the request (frontend sliders)
    params = req.model_params or {}
    activate_threshold = params.get('activate_threshold', ACTIVATE_THRESHOLD)
    suppress_threshold = params.get('suppress_threshold', SUPPRESS_THRESHOLD)

    uid = _h(req.bid_request.get("user", {}).get("id", "unknown"))
    mutations: list[Mutation] = []

    for imp in req.bid_request.get("imp", []):
        imp_id = imp.get("id", "")
        deals = (imp.get("pmp") or {}).get("deals", [])
        if not deals:
            continue

        deal_ids = [d.get("id", f"deal-{i}") for i, d in enumerate(deals)]
        to_act, to_sup = _score_deals(uid, deal_ids,
                                      activate_threshold=activate_threshold,
                                      suppress_threshold=suppress_threshold)

        if to_act and act_ok:
            mutations.append(Mutation(
                intent=Intent.ACTIVATE_DEALS, op=Operation.ADD,
                path=f"/imp/{imp_id}", ids=IDsPayload(id=to_act),
            ))
        if to_sup and sup_ok:
            mutations.append(Mutation(
                intent=Intent.SUPPRESS_DEALS, op=Operation.REMOVE,
                path=f"/imp/{imp_id}", ids=IDsPayload(id=to_sup),
            ))

    return RTBResponse(id=req.id, mutations=mutations,
                       metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION))


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="ncf-deal-manager", grpc_port=50051, mcp_port=8081, health_port=8080)
