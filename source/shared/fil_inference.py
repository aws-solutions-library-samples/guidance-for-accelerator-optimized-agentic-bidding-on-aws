"""Shared NVIDIA Triton FIL-backend client for the two Yield Optimizer containers.

Both yield containers serve an XGBoost model through Triton's FIL (Forest
Inference Library) backend, which is bundled in the same
nvcr.io/nvidia/tritonserver:24.08-py3 image this project already uses for its
ONNX/TensorRT-backed containers -- no additional Triton image is required.

FIL's tensor contract is fixed and identical for every model it serves: exactly
one input named "input__0" and one output named "output__0" (confirmed against
NVIDIA's FIL backend docs). That contract, and the client setup around it, is
therefore genuinely common code rather than per-model logic, so it lives here
instead of being duplicated in each container -- two copies of the same client
boilerplate would let a fix to one silently miss the other.

What stays per-container (in each container's own triton_inference.py) is
everything model-specific: the Triton model name, the "no change" fallback
value, and the interpretation of the scalar that comes back. That split keeps
the two containers independent per business-rules.md BR-2/BR-5 while sharing
only the mechanical transport.

WHY FIL AND NOT ONNX: FIL's single-output limitation is the reason there are two
yield models at all. A tree model with two outputs (floor_multiplier and
margin_value) cannot load in FIL, so the yield model was split into two
single-target XGBoost models -- which is also the more honest representation,
since ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN are independent atomic intents
(BR-5) that were never meant to be gated on each other.

References:
- Triton FIL backend: https://github.com/triton-inference-server/fil_backend
- Triton client: https://github.com/triton-inference-server/client
"""

from __future__ import annotations

import os

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")

_client: httpclient.InferenceServerClient | None = None


def get_client() -> httpclient.InferenceServerClient:
    """Process-wide lazily-created Triton HTTP client.

    One instance per container process (each yield container is its own
    process, so there is no sharing of connection state between the floor and
    margin models).
    """
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


def infer_single_output(model_name: str, feature_vector: list[float]) -> float | None:
    """Call one single-input/single-output FIL model.

    Returns None if the call fails -- never fabricates a value. The caller
    decides its own model-appropriate fallback, because "no change" is a
    different number for a floor multiplier (1.0) than for a margin value
    (0.0).
    """
    client = get_client()

    features_np = np.array([feature_vector], dtype=np.float32)
    inputs = [httpclient.InferInput("input__0", list(features_np.shape), "FP32")]
    inputs[0].set_data_from_numpy(features_np)
    outputs = [httpclient.InferRequestedOutput("output__0")]

    try:
        result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        return float(result.as_numpy("output__0").flat[0])
    except InferenceServerException as e:
        print(f"[triton] {model_name} inference failed: {e.message()}")
        return None


def is_model_ready(model_name: str) -> bool:
    """True only if the Triton server AND the named model report ready."""
    try:
        client = get_client()
        return client.is_server_ready() and client.is_model_ready(model_name)
    except Exception:
        return False
