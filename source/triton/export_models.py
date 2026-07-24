#!/usr/bin/env python3
"""Export ARTF PyTorch models to ONNX format for NVIDIA Triton Inference Server.

Creates the Triton model repository structure:
    triton/model_repository/
        dlrm_bid_shader/
            config.pbtxt
            1/model.onnx
        ncf_deal_manager/
            config.pbtxt
            1/model.onnx

NOTE: Wide & Deep (segment activation) is NOT exported here. It was replaced
by deterministic rule-based logic in
source/containers/widedeep_segment_activator/app.py — its ONNX graph could
not be compiled to a TensorRT engine (BatchNorm1d fusion is unsupported for
this shape by TensorRT 10.3.0) and it is slated to be swapped for a partner
ISV implementation. See GUIDANCE.md.

Usage:
    python triton/export_models.py                    # export all
    python triton/export_models.py --model dlrm       # export one
    python triton/export_models.py --output-dir /tmp   # custom output
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# DLRM Model (matches containers/dlrm_bid_shader/app.py)
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 16
NUM_DENSE = 4
NUM_SPARSE = 3
VOCAB_SIZE = 1000


class DLRMModel(nn.Module):
    """DLRM following NVIDIA DeepLearningExamples architecture."""

    def __init__(self):
        super().__init__()
        # NOTE: plain nn.Embedding, not nn.EmbeddingBag. Every call site feeds
        # a bag of exactly one index (sparse_X.unsqueeze(1) -> mode="sum"), and
        # summing a single-element bag is mathematically identical to a direct
        # embedding lookup. EmbeddingBag's ONNX export lowers to a Loop node
        # that TensorRT's Myelin builder cannot compile here ("Device to shape
        # host node should not be folded into myelin" — Error Code 2). A plain
        # Embedding lookup exports as a single Gather with no control flow, so
        # trtexec can build the engine, with no change to the model's outputs.
        self.embeddings = nn.ModuleList([
            nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
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

    def forward(self, dense, sparse_0, sparse_1, sparse_2):
        """Forward with explicit sparse inputs for ONNX export (no list args)."""
        dense_out = self.bottom_mlp(dense)
        # sparse_X is a 1-D tensor of one index per batch row; nn.Embedding
        # looks each up directly (no bagging/summing needed for a bag of 1).
        s0 = self.embeddings[0](sparse_0)
        s1 = self.embeddings[1](sparse_1)
        s2 = self.embeddings[2](sparse_2)
        interaction_out = self._interaction(dense_out, [s0, s1, s2])
        out = self.top_mlp(interaction_out)
        # NOTE: reshape(-1) instead of squeeze(-1). top_mlp's last Linear has a
        # statically-known output width of 1, but torch.onnx's exporter cannot
        # prove that at trace time for squeeze() on a non-batch dim, so it
        # lowers squeeze(-1) into an ONNX If/Loop conditional (branches on
        # whether the dim size == 1). TensorRT's ONNX parser cannot parse the
        # resulting IIfConditionalOutputLayer here (mismatched branch shapes
        # [-1] vs [-1,1]), so trtexec fails with "Invalid Node - /If" and the
        # base engine build never completes. reshape(-1) is a static op with
        # no conditional, and is exactly equivalent here since dim -1 is
        # always size 1.
        return torch.sigmoid(out).reshape(-1)


def export_dlrm(output_dir: str) -> str:
    """Export DLRM to ONNX with seeded weights matching the container."""
    model = DLRMModel()
    torch.manual_seed(42)
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            if hasattr(m, "weight"):
                if m.weight.dim() > 1 and not isinstance(m, nn.Embedding):
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.normal_(m.weight, std=0.01)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
    model.eval()

    dense = torch.randn(1, NUM_DENSE)
    s0 = torch.tensor([42])
    s1 = torch.tensor([7])
    s2 = torch.tensor([99])

    model_dir = os.path.join(output_dir, "dlrm_bid_shader", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")

    torch.onnx.export(
        model, (dense, s0, s1, s2), onnx_path,
        input_names=["dense_features", "sparse_user", "sparse_domain", "sparse_device"],
        output_names=["ctr_prediction"],
        dynamic_axes={
            "dense_features": {0: "batch"},
            "sparse_user": {0: "batch"},
            "sparse_domain": {0: "batch"},
            "sparse_device": {0: "batch"},
            "ctr_prediction": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  Exported DLRM → {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# NCF / NeuMF Model (matches containers/ncf_deal_manager/app.py)
# ---------------------------------------------------------------------------

class NeuMF(nn.Module):
    """NeuMF following NVIDIA DeepLearningExamples NCF architecture."""

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

        self.final = nn.Sequential(
            nn.Linear(mf_dim + mlp_layer_sizes[-1], 1),
            nn.Sigmoid(),
        )
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


def export_ncf(output_dir: str) -> str:
    """Export NCF/NeuMF to ONNX."""
    model = NeuMF()
    model.eval()

    users = torch.tensor([42])
    items = torch.tensor([7])

    model_dir = os.path.join(output_dir, "ncf_deal_manager", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")

    torch.onnx.export(
        model, (users, items), onnx_path,
        input_names=["user_ids", "item_ids"],
        output_names=["relevance_scores"],
        dynamic_axes={
            "user_ids": {0: "batch"},
            "item_ids": {0: "batch"},
            "relevance_scores": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  Exported NCF → {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export ARTF models to ONNX for Triton")
    parser.add_argument("--model", choices=["dlrm", "ncf", "all"], default="all")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "triton", "model_repository"))
    args = parser.parse_args()

    print(f"Exporting models to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model in ("dlrm", "all"):
        export_dlrm(args.output_dir)
    if args.model in ("ncf", "all"):
        export_ncf(args.output_dir)

    print("Done. Model repository ready for Triton Inference Server.")


if __name__ == "__main__":
    main()
