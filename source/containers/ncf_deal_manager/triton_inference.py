"""NCF / NeuMF inference via NVIDIA Triton Inference Server.

Replaces inline PyTorch inference with tritonclient HTTP calls to a
Triton sidecar running the ONNX-exported NeuMF model on GPU.

References:
- NVIDIA NCF: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
- NVIDIA Triton Client: https://github.com/triton-inference-server/client
"""

from __future__ import annotations

import os
import threading

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from shared import hop_timing, triton_client

TRITON_URL = triton_client.TRITON_URL  # kept for log/health callers; see shared/triton_client.py
MODEL_NAME = "ncf_deal_manager"

# One client per thread. `mutate()` runs on a ThreadPoolExecutor of
# ARTF_MUTATE_WORKERS threads (shared/server.py), and tritonclient.http's
# connection pool is greenlet-bound to the thread that opened it, so a client
# built on one worker thread raises "Cannot switch to a different thread" when
# a later request lands on another.
_local = threading.local()


def _get_client() -> httpclient.InferenceServerClient:
    client = getattr(_local, "client", None)
    if client is None:
        # Protocol (HTTP :8000 or gRPC :8001) and endpoint come from
        # shared/triton_client.py; this module only keeps the per-thread cache.
        client = triton_client.new_client()
        _local.client = client
    return client


def predict_relevance(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    target_variant: str | None = None,
) -> tuple[np.ndarray, str, str]:
    """Call Triton to predict user-deal relevance using NeuMF.

    Args:
        user_ids: shape [N] int64 — hashed user IDs (repeated for each deal)
        item_ids: shape [N] int64 — hashed deal IDs
        target_variant: "stable" | "canary" | None. When set, forces the
            router to use that specific variant for THIS request only. Only
            ever passed by the orchestrator's load-test invocation path —
            never by live bid-serving, which always passes None.

    Returns:
        (scores, served_variant, served_model_version). ``scores`` is an
        np.ndarray of shape [N] (0.0 to 1.0). ``served_variant`` /
        ``served_model_version`` are the router's real per-request
        resolution, or "" if unavailable (a real "unknown", never
        fabricated). On any Triton error, falls back to a safe default
        score array with empty variant/version — never raises.
    """
    client = _get_client()

    # Triton expects [batch, 1] for the reshape config
    u = user_ids.reshape(-1, 1).astype(np.int64)
    d = item_ids.reshape(-1, 1).astype(np.int64)

    inputs = [
        triton_client.api().InferInput("user_ids", list(u.shape), "INT64"),
        triton_client.api().InferInput("item_ids", list(d.shape), "INT64"),
    ]
    inputs[0].set_data_from_numpy(u)
    inputs[1].set_data_from_numpy(d)

    if target_variant in ("stable", "canary"):
        variant_input = triton_client.api().InferInput("target_variant", [1], "BYTES")
        variant_input.set_data_from_numpy(
            np.array([target_variant.encode("utf-8")], dtype=object)
        )
        inputs.append(variant_input)

    outputs = [
        triton_client.api().InferRequestedOutput("relevance_scores"),
        triton_client.api().InferRequestedOutput("served_variant"),
        triton_client.api().InferRequestedOutput("served_model_version"),
    ]

    try:
        with hop_timing.triton_call():
            result = triton_client.infer(client, model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        scores = result.as_numpy("relevance_scores").flatten()
        served_variant = _decode_str_output(result, "served_variant")
        served_model_version = _decode_str_output(result, "served_model_version")
        return scores, served_variant, served_model_version
    except InferenceServerException as e:
        print(f"[triton] NCF inference failed: {e.message()}")
        return np.full(len(user_ids), 0.5, dtype=np.float32), "", ""


def _decode_str_output(result, name: str) -> str:
    """Best-effort decode of an optional STRING/BYTES Triton output.

    Returns "" (a real "unknown", not a fabricated value) if the output
    isn't present on this router's config.
    """
    try:
        arr = result.as_numpy(name)
        if arr is None or arr.size == 0:
            return ""
        value = arr.flat[0]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    except Exception:
        return ""


def is_triton_ready() -> bool:
    """Check if Triton server and NCF model are ready."""
    try:
        client = _get_client()
        return client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        return False
