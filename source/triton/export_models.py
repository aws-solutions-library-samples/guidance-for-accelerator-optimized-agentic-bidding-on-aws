#!/usr/bin/env python3
"""Export ARTF PyTorch models to ONNX format for NVIDIA Triton Inference Server.

Creates the Triton model repository structure:
    triton/model_repository/
        dlrm_bid_shader/
            config.pbtxt
            1/model.onnx
        ncf_deal_manager/
            config.pbtxt
            1/model.onnx, manifest.json, interactions.csv

The DLRM export is seeded initialisation. The NCF (Deal Scorer) export is trained
on sample interactions and must pass the scenario acceptance gate
(training/ncf_sample/build.py); a gate failure exits 1 and writes no NCF model.

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
    python triton/export_models.py --model ncf --ncf-cohorts /tmp/cohorts.json --output-dir /tmp/x
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# DLRM Model (matches containers/dlrm_bid_shader/app.py)
# ---------------------------------------------------------------------------

from shared import dlrm_features  # noqa: E402  (needs REPO_ROOT on sys.path)
from shared import onnx_compat  # noqa: E402

EMBEDDING_DIM = 16

# Widths, vocabulary sizes and served input names all come from the feature
# spec. They used to be local constants, and the local NUM_SPARSE=3 with a
# single VOCAB_SIZE=1000 sized every embedding table identically — so an index
# from a larger vocabulary was an out-of-range lookup, and the served input
# names described a feature set the container does not send.
NUM_DENSE = dlrm_features.DENSE_WIDTH
NUM_SPARSE = dlrm_features.CATEGORICAL_WIDTH

# The export signature below takes its categorical inputs as three positional
# arguments, because ONNX export traces a fixed arity. Adding a categorical
# feature to the spec therefore requires editing this function, and this
# assertion is what makes that a loud failure at export rather than a silent
# mismatch in a served engine.
assert NUM_SPARSE == 3, (
    f"export_dlrm exports {NUM_SPARSE} categorical inputs as fixed positional "
    "arguments; update export_dlrm and the Triton config.pbtxt files to match "
    "the feature spec before changing CATEGORICAL_COLUMNS."
)


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
        #
        # One table per categorical feature, each sized to that feature's own
        # vocabulary, in the spec's column order.
        self.embeddings = nn.ModuleList([
            nn.Embedding(dlrm_features.VOCAB_SIZES[column], EMBEDDING_DIM)
            for column in dlrm_features.CATEGORICAL_COLUMNS
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
    # One trace index per categorical, each inside its own vocabulary. Tracing
    # an index past a table's end would fail the export itself.
    s0, s1, s2 = (
        torch.tensor([min(42, dlrm_features.VOCAB_SIZES[column] - 1)])
        for column in dlrm_features.CATEGORICAL_COLUMNS
    )

    model_dir = os.path.join(output_dir, "dlrm_bid_shader", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")

    # Input names come from the spec, which the container also reads, so the
    # served signature and the tensors the container sends cannot disagree.
    input_names = [
        dlrm_features.TRITON_DENSE_INPUT,
        *dlrm_features.TRITON_CATEGORICAL_INPUTS,
    ]
    torch.onnx.export(
        model, (dense, s0, s1, s2), onnx_path,
        input_names=input_names,
        output_names=["ctr_prediction"],
        dynamic_axes={
            **{name: {0: "batch"} for name in input_names},
            "ctr_prediction": {0: "batch"},
        },
        opset_version=17,
        **onnx_compat.onnx_export_kwargs(dynamo=False),
    )
    # Provenance beside the artifact, in the same shape the trainer writes. Both
    # producers emit one so the promotion path has a single rule -- a model with no
    # manifest is refused -- rather than an exemption for genesis artifacts that
    # would also silently cover a trainer that stopped writing it.
    #
    # These weights are seeded initialisation, not a trained checkpoint. Saying so
    # here means a reader of the manifest does not have to infer it from the path.
    manifest_path = os.path.join(model_dir, dlrm_features.MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            dlrm_features.manifest(
                "dlrm_bid_shader",
                producer="triton/export_models.py",
                weights="seeded_initialisation",
                torch_manual_seed=42,
            ),
            f,
            indent=2,
        )

    print(f"  Exported DLRM → {onnx_path}")
    print(f"  Wrote manifest → {manifest_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# NCF / NeuMF (Deal Scorer): trained on sample interactions and gated
# ---------------------------------------------------------------------------

from training.ncf_sample import build as ncf_build  # noqa: E402


def export_ncf(output_dir: str, cohorts_path: str = ncf_build.DEFAULT_COHORTS,
               cache_dir: str = ncf_build.DEFAULT_CACHE) -> str:
    """Train, export and gate the Deal Scorer (training/ncf_sample/build.py).

    Writes model.onnx, manifest.json and interactions.csv to
    <output_dir>/ncf_deal_manager/1/. Raises ncf_build.GateFailed when the model
    does not reproduce every scenario target; nothing is written then.
    """
    model_dir = os.path.join(output_dir, "ncf_deal_manager", "1")
    result = ncf_build.build(model_dir, cohorts_path=cohorts_path, cache_dir=cache_dir)
    onnx_path = os.path.join(model_dir, ncf_build.MODEL_FILE)
    print(f"  Exported NCF ({result.model_version}) to {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export ARTF models to ONNX for Triton")
    parser.add_argument("--model", choices=["dlrm", "ncf", "all"], default="all")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "triton", "model_repository"))
    parser.add_argument("--ncf-cohorts", default=ncf_build.DEFAULT_COHORTS,
                        help="cohort spec the Deal Scorer is trained and gated on")
    parser.add_argument("--ncf-cache-dir", default=ncf_build.DEFAULT_CACHE,
                        help="Deal Scorer build cache (default deployment/.ncf-cache)")
    args = parser.parse_args(argv)

    print(f"Exporting models to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model in ("dlrm", "all"):
        export_dlrm(args.output_dir)
    if args.model in ("ncf", "all"):
        try:
            export_ncf(args.output_dir, cohorts_path=args.ncf_cohorts, cache_dir=args.ncf_cache_dir)
        except ncf_build.GateFailed as exc:
            print(f"NCF export failed: {exc}; no NCF model written", file=sys.stderr)
            return 1
        except ncf_build.gen.SpecError as exc:
            print(f"NCF export failed: cohort spec error: {exc}", file=sys.stderr)
            return 1

    print("Done. Model repository ready for Triton Inference Server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
