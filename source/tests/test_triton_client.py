"""shared.triton_client: protocol switch, gRPC target derivation, per-call deadline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload(monkeypatch, **env):
    for key in ("TRITON_PROTOCOL", "TRITON_URL", "TRITON_GRPC_URL", "TRITON_GRPC_CLIENT_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import shared.triton_client as module
    return importlib.reload(module)


class _FakeClient:
    def __init__(self):
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return "result"


def test_defaults_to_http(monkeypatch):
    module = _reload(monkeypatch)
    assert module.use_grpc() is False
    assert module.api() is module.httpclient
    assert module.TRITON_URL == "localhost:8000"
    assert module.endpoint_description() == "http localhost:8000"


def test_grpc_url_derived_from_http_host_on_port_8001(monkeypatch):
    module = _reload(monkeypatch, TRITON_URL="triton-inference-server:8000")
    assert module.TRITON_GRPC_URL == "triton-inference-server:8001"


@pytest.mark.parametrize(
    "http_url,expected",
    [
        ("http://triton:8000", "triton:8001"),
        ("triton", "triton:8001"),
        ("http://triton:8000/v2", "triton:8001"),
    ],
)
def test_default_grpc_url_strips_scheme_port_and_path(http_url, expected):
    import shared.triton_client as module
    assert module._default_grpc_url(http_url) == expected


def test_explicit_grpc_url_wins(monkeypatch):
    module = _reload(
        monkeypatch,
        TRITON_URL="triton-inference-server:8000",
        TRITON_GRPC_URL="triton-grpc.other.svc:9001",
    )
    assert module.TRITON_GRPC_URL == "triton-grpc.other.svc:9001"


def test_grpc_protocol_selects_grpc_module(monkeypatch):
    module = _reload(monkeypatch, TRITON_PROTOCOL="grpc")
    if module.grpcclient is None:
        pytest.skip("tritonclient[grpc] not installed in this interpreter")
    assert module.use_grpc() is True
    assert module.api() is module.grpcclient
    assert module.endpoint_description().startswith("grpc ")


def test_grpc_protocol_without_extra_raises(monkeypatch):
    module = _reload(monkeypatch, TRITON_PROTOCOL="grpc")
    monkeypatch.setattr(module, "grpcclient", None)
    with pytest.raises(RuntimeError, match="tritonclient\\[grpc\\]"):
        module.api()


def test_infer_http_passes_no_client_timeout(monkeypatch):
    module = _reload(monkeypatch)
    client = _FakeClient()
    assert module.infer(client, model_name="m", inputs=[1], outputs=[2]) == "result"
    assert client.calls == [{"model_name": "m", "inputs": [1], "outputs": [2]}]


def test_infer_grpc_passes_per_call_deadline(monkeypatch):
    module = _reload(monkeypatch, TRITON_PROTOCOL="grpc", TRITON_GRPC_CLIENT_TIMEOUT_S="2.5")
    client = _FakeClient()
    module.infer(client, model_name="m", inputs=[1], outputs=[2])
    assert client.calls == [
        {"model_name": "m", "inputs": [1], "outputs": [2], "client_timeout": 2.5}
    ]


def test_new_client_http_targets_triton_url(monkeypatch):
    module = _reload(monkeypatch, TRITON_URL="triton-inference-server:8000")
    captured = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module.httpclient, "InferenceServerClient", _Stub)
    module.new_client()
    assert captured["url"] == "triton-inference-server:8000"
    assert captured["network_timeout"] == 10.0


def test_new_client_grpc_targets_grpc_url(monkeypatch):
    module = _reload(monkeypatch, TRITON_PROTOCOL="grpc", TRITON_URL="triton-inference-server:8000")
    if module.grpcclient is None:
        pytest.skip("tritonclient[grpc] not installed in this interpreter")
    captured = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module.grpcclient, "InferenceServerClient", _Stub)
    module.new_client()
    assert captured == {"url": "triton-inference-server:8001", "verbose": False}
