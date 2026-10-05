"""Triton client selection: HTTP (port 8000) or gRPC (port 8001).

The three Triton-backed inference modules (dlrm_bid_shader, ncf_deal_manager, shared/fil_inference) build
their per-thread client and their InferInput/InferRequestedOutput objects through
this module so one env var moves all of them between protocols:

    TRITON_PROTOCOL   http (default) | grpc
    TRITON_URL        host:port of the HTTP endpoint       (default localhost:8000)
    TRITON_GRPC_URL   host:port of the gRPC endpoint        (default: TRITON_URL's host, port 8001)
    TRITON_GRPC_CLIENT_TIMEOUT_S  per-call deadline on gRPC (default 10, matching the
                                  HTTP client's network_timeout)

``tritonclient.grpc`` and ``tritonclient.http`` expose the same InferInput /
InferRequestedOutput / infer() / as_numpy() surface, so the callers do not branch;
they ask ``api()`` for the right module and ``infer()`` for the right call.

Both clients are still built per thread (see the callers' ``_local``): the HTTP
client's pool is greenlet-bound to its thread, and keeping the gRPC client on the
same per-thread shape means the switch changes the wire and nothing else.
"""

from __future__ import annotations

import os

import tritonclient.http as httpclient

try:  # tritonclient[grpc]; the container image installs both extras
    import tritonclient.grpc as grpcclient
except ImportError:  # pragma: no cover - exercised only in images without the extra
    grpcclient = None

TRITON_PROTOCOL = os.environ.get("TRITON_PROTOCOL", "http").strip().lower()
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
TRITON_GRPC_CLIENT_TIMEOUT_S = float(os.environ.get("TRITON_GRPC_CLIENT_TIMEOUT_S", "10"))


def _default_grpc_url(http_url: str) -> str:
    host = http_url.split("://", 1)[-1].split("/", 1)[0]
    host = host.rsplit(":", 1)[0] if ":" in host else host
    return f"{host}:8001"


TRITON_GRPC_URL = os.environ.get("TRITON_GRPC_URL") or _default_grpc_url(TRITON_URL)


def use_grpc() -> bool:
    return TRITON_PROTOCOL == "grpc"


def api():
    """The tritonclient module whose InferInput/InferRequestedOutput to build."""
    if use_grpc():
        if grpcclient is None:
            raise RuntimeError(
                "TRITON_PROTOCOL=grpc but tritonclient[grpc] is not installed in this image"
            )
        return grpcclient
    return httpclient


def new_client():
    """A fresh InferenceServerClient for the selected protocol (one per thread)."""
    if use_grpc():
        return api().InferenceServerClient(url=TRITON_GRPC_URL, verbose=False)
    return httpclient.InferenceServerClient(
        url=TRITON_URL,
        verbose=False,
        concurrency=4,
        connection_timeout=5.0,
        network_timeout=10.0,
    )


def infer(client, *, model_name: str, inputs, outputs):
    """``client.infer`` with the per-call deadline the protocol supports.

    The HTTP client takes its timeouts at construction (network_timeout); the gRPC
    client takes them per call (client_timeout, seconds).
    """
    if use_grpc():
        return client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs,
            client_timeout=TRITON_GRPC_CLIENT_TIMEOUT_S,
        )
    return client.infer(model_name=model_name, inputs=inputs, outputs=outputs)


def endpoint_description() -> str:
    return f"grpc {TRITON_GRPC_URL}" if use_grpc() else f"http {TRITON_URL}"
