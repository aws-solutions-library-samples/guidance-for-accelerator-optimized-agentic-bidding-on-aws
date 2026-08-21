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
    target_variant: str | None = None,
) -> tuple[float, str, str]:
    """Call Triton to predict CTR using the DLRM model.

    Args:
        dense_features: shape [1, 4] float32 — bidfloor, hour, age, video
        sparse_user:    shape [1, 1] int64   — hashed user ID
        sparse_domain:  shape [1, 1] int64   — hashed domain
        sparse_device:  shape [1, 1] int64   — hashed device UA
        target_variant: "stable" | "canary" | None. When set, forces the
            router to use that specific variant for THIS request only,
            bypassing its normal random split. Only ever passed by the
            orchestrator's load-test invocation path (see
            source/orchestrator/loadtest_targeting.py) — never by live
            bid-serving, which always passes None.

    Returns:
        (ctr, served_variant, served_model_version). ``served_variant`` is
        "stable"/"canary" if the router declares that output, else "".
        ``served_model_version`` is the router's resolved version-ARN for
        the variant that served this request, or "" if the router doesn't
        declare that output or hasn't been given a version-ARN yet (a real
        "unknown" state, never a fabricated placeholder). On any Triton
        error, falls back to a safe default CTR with empty variant/version
        — this function never raises, matching the existing fallback
        contract this call site already relies on.
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

    if target_variant in ("stable", "canary"):
        variant_input = httpclient.InferInput("target_variant", [1], "BYTES")
        variant_input.set_data_from_numpy(
            np.array([target_variant.encode("utf-8")], dtype=object)
        )
        inputs.append(variant_input)

    outputs = [
        httpclient.InferRequestedOutput("ctr_prediction"),
        httpclient.InferRequestedOutput("served_variant"),
        httpclient.InferRequestedOutput("served_model_version"),
    ]

    try:
        result = client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        ctr = float(result.as_numpy("ctr_prediction").flat[0])
        served_variant = _decode_str_output(result, "served_variant")
        served_model_version = _decode_str_output(result, "served_model_version")
        return ctr, served_variant, served_model_version
    except InferenceServerException as e:
        print(f"[triton] DLRM inference failed: {e.message()}")
        return 0.5, "", ""  # fallback CTR; real "unknown" for variant/version


def _decode_str_output(result, name: str) -> str:
    """Best-effort decode of an optional STRING/BYTES Triton output.

    Returns "" (a real "unknown", not a fabricated value) if the output
    isn't present on this router's config — e.g. an older-generated
    router config that doesn't declare served_variant/served_model_version.
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
    """Check if Triton server and DLRM model are ready."""
    try:
        client = _get_client()
        return client.is_server_ready() and client.is_model_ready(MODEL_NAME)
    except Exception:
        return False
