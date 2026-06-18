"""DLRM inference via NVIDIA Triton Inference Server.

Replaces inline PyTorch inference with tritonclient HTTP calls to a
Triton sidecar running the ONNX-exported DLRM model on GPU.

The Triton model repository is loaded from S3 at pod startup.
Dynamic batching is handled server-side by Triton.

References:
- NVIDIA Triton Client: https://github.com/triton-inference-server/client
- NVIDIA DLRM: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM
"""

from __future__ import annotations

import os

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
MODEL_NAME = "dlrm_bid_shader"

_client: httpclient.InferenceServerClient | None = None


def _get_client() -> httpclient.InferenceServerClient:
    global _client
    if _client is None:
        _client = httpclient.InferenceServerClient(
            url=TRITON_URL,
            verbose=False,
            concurrency=4,
            connection_timeout=5.0,
            network_timeout=10.0,
        )
    return _client


def predict_ctr(
    dense_features: np.ndarray,
    sparse_user: np.ndarray,
    sparse_domain: np.ndarray,
    sparse_device: np.ndarray,
) -> float:
    """Call Triton to predict CTR using the DLRM model.

    Args:
        dense_features: shape [1, 4] float32 — bidfloor, hour, age, video
        sparse_user:    shape [1, 1] int64   — hashed user ID
        sparse_domain:  shape [1, 1] int64   — hashed domain
        sparse_device:  shape [1, 1] int64   — hashed device UA

    Returns:
        Predicted click-through rate (0.0 to 1.0)
    """
    client = _get_client()

    inputs = [
        httpclient.InferInput("dense_features", list(dense_features.shape), "FP32"),
        httpclient.InferInput("sparse_user", list(sparse_user.shape), "INT64"),
        httpclient.InferInput("sparse_domain", list(sparse_domain.shape), "INT64"),
        httpclient.InferInput("sparse_device", list(sparse_device.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(dense_features.astype(np.float32))
    inputs[1].set_data_from_numpy(sparse_user.astype(np.int64))
    inputs[2].set_data_from_numpy(sparse_domain.astype(np.int64))
    inputs[3].set_data_from_numpy(sparse_device.astype(np.int64))

    outputs = [httpclient.InferRequestedOutput("ctr_prediction")]

    try:
        result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        return float(result.as_numpy("ctr_prediction").flat[0])
    except InferenceServerException as e:
        print(f"[triton] DLRM inference failed: {e.message()}")
        return 0.5  # fallback CTR


def is_triton_ready() -> bool:
    """Check if Triton server and DLRM model are ready."""
    try:
        client = _get_client()
        return client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        return False
