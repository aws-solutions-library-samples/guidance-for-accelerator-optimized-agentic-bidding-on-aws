"""Deal Yield Manager inference via NVIDIA Triton Inference Server (FIL backend).

Unlike dlrm_bid_shader/ncf_deal_manager, which serve ONNX/TensorRT deep
learning graphs, this container's model is served by Triton's FIL (Forest
Inference Library) backend -- purpose-built for tree models (XGBoost). FIL
is bundled in the same nvcr.io/nvidia/tritonserver:24.08-py3 image already
used by this project; no new Triton image is required.

CORRECTION (found during Unit 3 implementation): Triton's FIL backend does
NOT support multi-output regression models, and requires exactly one input
tensor named "input__0" and one output tensor named "output__0" (confirmed
against NVIDIA's official FIL backend docs). The originally-committed
single `deal_yield_manager` model (one input "features", two outputs
"floor_multiplier"/"margin_value") would fail to load in Triton. This
module now calls TWO independent single-target FIL models --
`deal_yield_manager_floor` and `deal_yield_manager_margin` -- since
ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN are genuinely distinct intents
(business-rules.md BR-5: independent, atomic, never gated on each other).

References:
- NVIDIA Triton Client: https://github.com/triton-inference-server/client
- Triton FIL backend: https://github.com/triton-inference-server/fil_backend
"""

from __future__ import annotations

import os

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
FLOOR_MODEL_NAME = "deal_yield_manager_floor"
MARGIN_MODEL_NAME = "deal_yield_manager_margin"

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


def _infer_single_output(model_name: str, feature_vector: list[float]) -> float | None:
    """Call one single-input/single-output FIL model. Returns None (never
    fabricates a value) if the call fails -- caller decides the fallback.
    """
    client = _get_client()

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


def predict_yield_adjustment(
    feature_vector: list[float], target_variant: str | None = None
) -> tuple[float, float, str, str]:
    """Call Triton to predict a floor multiplier and a margin value, via two
    independent single-target FIL models (see module docstring correction
    note).

    Args:
        feature_vector: The 7-element feature vector from
            features.build_feature_vector().
        target_variant: "stable" | "canary" | None. Load-test-only override
            (see source/orchestrator/loadtest_targeting.py); always None on
            live bid-serving traffic. Unit 1's models have no canary router
            yet (see infrastructure-design.md) -- this parameter is
            accepted now for interface parity with dlrm_bid_shader/
            ncf_deal_manager and to avoid a signature change once Unit 3
            adds the router, but has no effect until then.

    Returns:
        (floor_multiplier, margin_value, served_variant, served_model_version).
        floor_multiplier of 1.0 means "no floor change recommended";
        margin_value of 0.0 means "no margin adjustment recommended". Each
        sub-model is called independently -- a failure in one does not
        block the other's real prediction (matches BR-5: the two intents
        are independent and atomic). served_variant/served_model_version
        are "" until Unit 3's canary router exists (a real "unknown",
        never fabricated). On inference error for a given sub-model, that
        sub-model falls back to its own "no change" value -- never raises,
        matching the existing fallback contract every Triton-backed
        container already implements (see business-rules.md BR-9).
    """
    floor_multiplier = _infer_single_output(FLOOR_MODEL_NAME, feature_vector)
    margin_value = _infer_single_output(MARGIN_MODEL_NAME, feature_vector)

    return (
        floor_multiplier if floor_multiplier is not None else 1.0,
        margin_value if margin_value is not None else 0.0,
        "",
        "",
    )


def is_triton_ready() -> bool:
    """Check if Triton server and BOTH deal_yield_manager sub-models are ready."""
    try:
        client = _get_client()
        return (
            client.is_server_ready()
            and client.is_model_ready(FLOOR_MODEL_NAME)
            and client.is_model_ready(MARGIN_MODEL_NAME)
        )
    except Exception:
        return False
