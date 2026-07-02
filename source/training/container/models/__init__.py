"""DLRM model architecture for NeMo-RL training.

Based on NVIDIA DeepLearningExamples DLRM:
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM
"""

import torch
import torch.nn as nn

EMBEDDING_DIM = 16
NUM_DENSE = 4
NUM_SPARSE = 3
VOCAB_SIZE = 1000


class DLRMModel(nn.Module):
    """DLRM with dot-product feature interaction (NVIDIA architecture)."""

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

    def forward(self, x):
        # For training: x is [batch, total_features] — split into dense + sparse indices
        dense = x[:, :NUM_DENSE]
        sparse_indices = x[:, NUM_DENSE:NUM_DENSE + NUM_SPARSE].long()

        dense_out = self.bottom_mlp(dense)
        sparse_outs = [
            self.embeddings[i](sparse_indices[:, i].unsqueeze(1))
            for i in range(NUM_SPARSE)
        ]
        all_embeds = torch.stack([dense_out] + sparse_outs, dim=1)
        interactions = torch.bmm(all_embeds, all_embeds.transpose(1, 2))
        triu_indices = torch.triu_indices(NUM_SPARSE + 1, NUM_SPARSE + 1, offset=1)
        flat = interactions[:, triu_indices[0], triu_indices[1]]
        out = torch.cat([dense_out, flat], dim=1)
        return self.top_mlp(out).squeeze(-1)
