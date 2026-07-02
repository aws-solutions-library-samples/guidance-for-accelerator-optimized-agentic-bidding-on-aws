"""Wide & Deep model architecture for training — NVIDIA DeepLearningExamples."""

import torch
import torch.nn as nn


class WideDeepModel(nn.Module):
    """Wide & Deep Learning (Google 2016) — NVIDIA variant.

    Wide path: linear model on crossed features (8 dims)
    Deep path: MLP on dense features (6 dims)
    Output: 5 segment scores
    """

    def __init__(self, wide_dim=8, deep_dim=6, output_dim=5):
        super().__init__()
        self.wide_dim = wide_dim
        self.deep_dim = deep_dim
        self.wide = nn.Linear(wide_dim, output_dim)
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        wide_input = x[:, :self.wide_dim]
        deep_input = x[:, self.wide_dim:self.wide_dim + self.deep_dim]
        return (self.wide(wide_input) + self.deep(deep_input)).squeeze(-1)


__all__ = ["WideDeepModel"]
