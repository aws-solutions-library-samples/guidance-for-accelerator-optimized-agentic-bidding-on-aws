"""NCF model architecture for training — NVIDIA DeepLearningExamples NeuMF."""

import torch
import torch.nn as nn

EMBEDDING_DIM = 32
NUM_USERS = 1000
NUM_ITEMS = 500


class NCFModel(nn.Module):
    """Neural Collaborative Filtering (NeuMF) — NVIDIA architecture."""

    def __init__(self):
        super().__init__()
        # GMF path
        self.user_embed_gmf = nn.Embedding(NUM_USERS, EMBEDDING_DIM)
        self.item_embed_gmf = nn.Embedding(NUM_ITEMS, EMBEDDING_DIM)
        # MLP path
        self.user_embed_mlp = nn.Embedding(NUM_USERS, EMBEDDING_DIM)
        self.item_embed_mlp = nn.Embedding(NUM_ITEMS, EMBEDDING_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, EMBEDDING_DIM), nn.ReLU(),
        )
        self.output = nn.Linear(EMBEDDING_DIM * 2, 1)

    def forward(self, x):
        user_ids = x[:, 0].long() % NUM_USERS
        item_ids = x[:, 1].long() % NUM_ITEMS

        # GMF
        gmf = self.user_embed_gmf(user_ids) * self.item_embed_gmf(item_ids)
        # MLP
        mlp_input = torch.cat([self.user_embed_mlp(user_ids), self.item_embed_mlp(item_ids)], dim=1)
        mlp_out = self.mlp(mlp_input)
        # Combine
        combined = torch.cat([gmf, mlp_out], dim=1)
        return self.output(combined).squeeze(-1)


__all__ = ["NCFModel"]
