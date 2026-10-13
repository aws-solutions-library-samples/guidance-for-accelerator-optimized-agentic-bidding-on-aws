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
import threading

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from shared import dlrm_features, hop_timing, triton_client

TRITON_URL = triton_client.TRITON_URL  # kept for log/health callers; see shared/triton_client.py
MODEL_NAME = "dlrm_bid_shader"

# One client per thread. `mutate()` runs on a ThreadPoolExecutor of
# ARTF_MUTATE_WORKERS threads (shared/server.py), and tritonclient.http's
# connection pool is greenlet-bound to the thread that opened it, so a client
# built on one worker thread raises "Cannot switch to a different thread" when
# a later request lands on another. Measured 2 failures in 6 sequential
# requests on a live pod before this change.
_local = threading.local()


class InferenceUnavailable(RuntimeError):
    """Triton did not produce a prediction, so there is no prediction to use."""


def _get_client() -> httpclient.InferenceServerClient:
    client = getattr(_local, "client", None)
    if client is None:
        # Protocol (HTTP :8000 or gRPC :8001) and endpoint come from
        # shared/triton_client.py; this module only keeps the per-thread cache.
        client = triton_client.new_client()
        _local.client = client
    return client


def predict_ctr(
    dense: np.ndarray,
    categorical: list[np.ndarray],
    target_variant: str | None = None,
) -> tuple[float, str, str]:
    """Call Triton to predict CTR using the DLRM model.

    Args:
        dense: shape [1, DENSE_WIDTH] float32, in DENSE_COLUMNS order.
        categorical: one [1, 1] int64 array per CATEGORICAL_COLUMNS entry, in
            that order. Input names come from
            `dlrm_features.TRITON_CATEGORICAL_INPUTS`, so the caller does not
            name them and cannot mis-order them relative to the spec.
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
        "unknown" state, never a fabricated placeholder).

    Raises:
        InferenceUnavailable: no usable prediction came back — whether Triton
            reported an error, could not be reached at all, or returned a
            result this could not read. The caller emits no mutation; see the
            note below.

    This function used to return a CTR of 0.5 on any Triton error. That is a
    fabricated prediction, and it is indistinguishable downstream from a real one:
    0.5 x the conversion-value estimate x the shade factor is a plausible price, so
    a shaded bid was published on the strength of a number no model produced.

    It also hides a deployment whose every inference fails: if the container sends
    input names the served engine does not declare, the only symptom is a slightly
    different shaded price, with Triton's own counter at `success=0` while the
    response says `status: ok`.

    Raising instead lets the caller abstain. No mutation is a legitimate ARTF
    response — the auction proceeds on the original price — whereas a mutation
    derived from a placeholder is a wrong answer presented as a right one.
    """
    client = _get_client()

    if len(categorical) != len(dlrm_features.TRITON_CATEGORICAL_INPUTS):
        # A count mismatch means the caller and the spec disagree about how
        # many categoricals exist. Raising is right: the alternative is sending
        # a tensor under the wrong feature's input name.
        raise ValueError(
            f"expected {len(dlrm_features.TRITON_CATEGORICAL_INPUTS)} categorical "
            f"arrays for {dlrm_features.CATEGORICAL_COLUMNS}, got {len(categorical)}"
        )

    dense_input = triton_client.api().InferInput(
        dlrm_features.TRITON_DENSE_INPUT, list(dense.shape), "FP32"
    )
    dense_input.set_data_from_numpy(dense.astype(np.float32))
    inputs = [dense_input]

    for name, array in zip(dlrm_features.TRITON_CATEGORICAL_INPUTS, categorical):
        tensor = triton_client.api().InferInput(name, list(array.shape), "INT64")
        tensor.set_data_from_numpy(array.astype(np.int64))
        inputs.append(tensor)

    if target_variant in ("stable", "canary"):
        variant_input = triton_client.api().InferInput("target_variant", [1], "BYTES")
        variant_input.set_data_from_numpy(
            np.array([target_variant.encode("utf-8")], dtype=object)
        )
        inputs.append(variant_input)

    outputs = [
        triton_client.api().InferRequestedOutput("ctr_prediction"),
        triton_client.api().InferRequestedOutput("served_variant"),
        triton_client.api().InferRequestedOutput("served_model_version"),
    ]

    try:
        with hop_timing.triton_call():
            result = triton_client.infer(client, model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        ctr = float(result.as_numpy("ctr_prediction").flat[0])
        served_variant = _decode_str_output(result, "served_variant")
        served_model_version = _decode_str_output(result, "served_model_version")
        return ctr, served_variant, served_model_version
    except InferenceServerException as exc:
        raise InferenceUnavailable(exc.message()) from exc
    except Exception as exc:
        # InferenceServerException covers errors the SERVER reported. It does not
        # cover not reaching the server at all: with Triton scaled to zero the client
        # raises a connection error, which escaped this function entirely, propagated
        # out of `mutate`, and arrived at the orchestrator as `status: error` with the
        # abstention reason lost (`mutations: 0, status=error, abstained_reason=None`).
        #
        # Both cases are the same fact for a caller: there is no prediction. The
        # exception type is kept in the message so the two remain distinguishable in
        # a log.
        raise InferenceUnavailable(f"{type(exc).__name__}: {exc}") from exc


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
