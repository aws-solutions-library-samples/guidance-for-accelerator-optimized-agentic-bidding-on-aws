"""DLRM model architecture for NeMo-RL training.

Based on NVIDIA DeepLearningExamples DLRM:
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM
"""

import torch
import torch.nn as nn

# In the training image this resolves to /opt/ml/code/shared/, staged from
# source/shared/ by source/training/stage_shared.sh. Same file the serving
# container imports.
from shared import dlrm_features

EMBEDDING_DIM = 16

# Widths and vocabulary sizes come from the feature spec. They were local
# constants, and the single VOCAB_SIZE=1000 sized every embedding table
# identically -- so an index drawn from the 10,000-entry site_domain vocabulary
# was an out-of-range lookup that raised IndexError partway into a training run.
NUM_DENSE = dlrm_features.DENSE_WIDTH
NUM_SPARSE = dlrm_features.CATEGORICAL_WIDTH


class DLRMModel(nn.Module):
    """DLRM with dot-product feature interaction (NVIDIA architecture).

    Reconciled to the SERVING architecture in source/triton/export_models.py so a
    retrained model is servable (and thus comparable) as a Triton canary:

    - Plain ``nn.Embedding`` instead of ``nn.EmbeddingBag``. Every call site feeds
      a bag of exactly one index, and summing a single-element bag is identical to
      a direct lookup — so this is weight-compatible with the old EmbeddingBag —
      while exporting as a single Gather (no ONNX Loop), which TensorRT can compile.
    - The training-time ``forward(x)`` keeps its single width-(NUM_DENSE+NUM_SPARSE)
      tensor interface and returns RAW logits (the trainer uses BCEWithLogitsLoss).
      The sigmoid activation and the four-named-input serving I/O signature are
      applied at export time by ``DLRMExportModel`` — see ``export_to_onnx``.
    """

    def __init__(self):
        super().__init__()
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

    def logits(self, dense, sparse_indices):
        """Raw (pre-sigmoid) score from dense features and a list of NUM_SPARSE
        1-D int64 index tensors. Shared by the training forward and the serving
        export wrapper so the two never diverge."""
        dense_out = self.bottom_mlp(dense)
        sparse_outs = [self.embeddings[i](sparse_indices[i]) for i in range(NUM_SPARSE)]
        return self.top_mlp(self._interaction(dense_out, sparse_outs))

    def forward(self, x):
        # Training interface: x is [batch, NUM_DENSE+NUM_SPARSE]; the last
        # NUM_SPARSE columns are embedding indices (stored as floats, cast here).
        dense = x[:, :NUM_DENSE]
        sparse_indices = [x[:, NUM_DENSE + i].long() for i in range(NUM_SPARSE)]
        # reshape(-1) (not squeeze(-1)): a static op with no If/Loop, matching the
        # serving export so training and serving share one graph shape.
        return self.logits(dense, sparse_indices).reshape(-1)


class DLRMExportModel(nn.Module):
    """Wraps a trained :class:`DLRMModel` to expose the SERVED input/output
    signature for ONNX export: one dense input plus one input per categorical
    feature (``dlrm_features.TRITON_DENSE_INPUT`` and
    ``TRITON_CATEGORICAL_INPUTS``), and a sigmoid'd ``ctr_prediction`` output —
    the exact contract in source/triton/export_models.py and the
    ``dlrm_bid_shader_stable`` Triton config. Reuses the trained submodules (no
    weight copy), so the exported engine serves exactly what was trained.

    The categorical inputs are three fixed positional arguments because ONNX
    export traces a fixed arity; the assertion below is what turns a spec change
    into a loud failure here rather than a silent mismatch in a served engine.
    """

    def __init__(self, model: "DLRMModel"):
        super().__init__()
        assert NUM_SPARSE == 3, (
            f"DLRMExportModel exports {NUM_SPARSE} categorical inputs as fixed "
            "positional arguments; update this wrapper, export_to_onnx, "
            "source/triton/export_models.py and the Triton config.pbtxt files "
            "before changing CATEGORICAL_COLUMNS."
        )
        self.model = model

    def forward(self, dense_features, sparse_site_domain, sparse_device_type, sparse_geo_country):
        logits = self.model.logits(
            dense_features,
            [sparse_site_domain, sparse_device_type, sparse_geo_country],
        )
        return torch.sigmoid(logits).reshape(-1)
