"""The Deal Scorer's NeuMF model and its served ONNX signature.

NeuMF (He et al. 2017) as in NVIDIA DeepLearningExamples NCF:
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF

One definition used by the serving container's inline path
(containers/ncf_deal_manager/app.py), the deploy-time sample trainer
(training/ncf_sample/build.py) and the SageMaker trainer
(training/container/train.py). The ONNX signature written by ``export_onnx`` is the
one ncf_deal_manager_stable/config.pbtxt declares and the Model Optimizer compiles:
inputs ``user_ids`` and ``item_ids`` (INT64, shape [batch]), output
``relevance_scores`` (FP32, shape [batch]), dynamic batch, opset 17, tracing
exporter.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from shared import ncf_ids, onnx_compat

INPUT_NAMES = ("user_ids", "item_ids")
OUTPUT_NAME = "relevance_scores"
OPSET = 17


class NeuMF(nn.Module):
    """NeuMF following NVIDIA's DeepLearningExamples NCF implementation.

    ``forward`` returns the sigmoid score in [0, 1], shape [batch].
    """

    def __init__(self, nb_users=ncf_ids.NB_USERS, nb_items=ncf_ids.NB_ITEMS, mf_dim=64, mlp_layer_sizes=None):
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
        # squeeze(-1) exports as a single Squeeze node here (no If), which the
        # TensorRT build of ncf_deal_manager_stable already compiles.
        return self.final(concat).squeeze(-1)


class ExportWrapper(nn.Module):
    """The served signature around a NeuMF: (user_ids, item_ids) -> relevance_scores.

    Reuses the wrapped model's submodules, so the export carries the trained weights.
    """

    def __init__(self, model: NeuMF):
        super().__init__()
        self.model = model

    def forward(self, user_ids, item_ids):
        return self.model(user_ids, item_ids)


def export_onnx(model: NeuMF, path: str) -> str:
    """Write ``model`` to ``path`` as ONNX with the Triton input and output names."""
    wrapper = ExportWrapper(model).eval()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    users = torch.tensor([42], dtype=torch.int64)
    items = torch.tensor([7], dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (users, items), path,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAMES[0]: {0: "batch"},
                INPUT_NAMES[1]: {0: "batch"},
                OUTPUT_NAME: {0: "batch"},
            },
            opset_version=OPSET,
            **onnx_compat.onnx_export_kwargs(dynamo=False),
        )
    return path
