"""NCF / NeuMF inference via NVIDIA Triton Inference Server.

Replaces inline PyTorch inference with tritonclient HTTP calls to a
Triton sidecar running the ONNX-exported NeuMF model on GPU.

References:
- NVIDIA NCF: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
- NVIDIA Triton Client: https://github.com/triton-inference-server/client
"""

from __future__ import annotations

import os

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
MODEL_NAME = "ncf_deal_manager"

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


def predict_relevance(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    """Call Triton to predict user-deal relevance using NeuMF.

    Args:
        user_ids: shape [N] int64 — hashed user IDs (repeated for each deal)
        item_ids: shape [N] int64 — hashed deal IDs

    Returns:
        np.ndarray of shape [N] — relevance scores (0.0 to 1.0)
    """
    client = _get_client()

    # Triton expects [batch, 1] for the reshape config
    u = user_ids.reshape(-1, 1).astype(np.int64)
    d = item_ids.reshape(-1, 1).astype(np.int64)

    inputs = [
        httpclient.InferInput("user_ids", list(u.shape), "INT64"),
        httpclient.InferInput("item_ids", list(d.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(u)
    inputs[1].set_data_from_numpy(d)

    outputs = [httpclient.InferRequestedOutput("relevance_scores")]

    try:
        result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        return result.as_numpy("relevance_scores").flatten()
    except InferenceServerException as e:
        print(f"[triton] NCF inference failed: {e.message()}")
        return np.full(len(user_ids), 0.5, dtype=np.float32)


def is_triton_ready() -> bool:
    """Check if Triton server and NCF model are ready."""
    try:
        client = _get_client()
        return client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        return False
