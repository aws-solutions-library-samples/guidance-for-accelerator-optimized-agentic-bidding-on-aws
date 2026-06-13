"""Wide & Deep inference via NVIDIA Triton Inference Server.

Replaces inline PyTorch inference with tritonclient HTTP calls to a
Triton sidecar running the ONNX-exported Wide & Deep model on GPU.

References:
- NVIDIA Merlin: https://nvidia-merlin.github.io/models/
- NVIDIA Triton Client: https://github.com/triton-inference-server/client
"""

from __future__ import annotations

import os

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
MODEL_NAME = "widedeep_segment_activator"

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


def predict_segments(
    wide_features: np.ndarray,
    deep_features: np.ndarray,
) -> np.ndarray:
    """Call Triton to score candidate segments using Wide & Deep.

    Args:
        wide_features: shape [1, 8] float32 — crossed feature interactions
        deep_features: shape [1, 6] float32 — dense features

    Returns:
        np.ndarray of shape [15] — activation score per candidate segment
    """
    client = _get_client()

    inputs = [
        httpclient.InferInput("wide_features", list(wide_features.shape), "FP32"),
        httpclient.InferInput("deep_features", list(deep_features.shape), "FP32"),
    ]
    inputs[0].set_data_from_numpy(wide_features.astype(np.float32))
    inputs[1].set_data_from_numpy(deep_features.astype(np.float32))

    outputs = [httpclient.InferRequestedOutput("segment_scores")]

    try:
        result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        return result.as_numpy("segment_scores").flatten()
    except InferenceServerException as e:
        print(f"[triton] Wide & Deep inference failed: {e.message()}")
        return np.zeros(15, dtype=np.float32)


def is_triton_ready() -> bool:
    """Check if Triton server and Wide & Deep model are ready."""
    try:
        client = _get_client()
        return client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        return False
