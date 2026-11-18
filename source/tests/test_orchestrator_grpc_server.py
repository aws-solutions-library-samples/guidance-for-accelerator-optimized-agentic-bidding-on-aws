"""The orchestrator's gRPC RTBExtensionPoint/GetMutations (plan P3.2).

A real grpc.aio server built by app._build_grpc_server on an ephemeral port, called
with a plain grpc.aio channel the way the Prebid hook's client calls it
(JSON-over-gRPC, bearer token as `authorization` metadata). Pinned:

- The reply is the same response dict POST /v1/mutations returns, with
  metadata.timing carrying `auth` from the gRPC-side verification.
- Fail closed: no pool configured -> UNAVAILABLE; no/invalid token ->
  UNAUTHENTICATED; valid token without the route's scope -> PERMISSION_DENIED.
- AUTH_DISABLED=true bypasses, as on HTTP.
- A non-JSON message is INVALID_ARGUMENT, not INTERNAL.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import grpc
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import app as oapp  # noqa: E402
from orchestrator import auth as oauth  # noqa: E402
from orchestrator.container_registry import STATUS_OK, merge_registry  # noqa: E402
from shared.artf_types import ContainerInvocationModel  # noqa: E402

ENVELOPE = {"id": "r1", "tmax": 100, "bid_request": {"id": "br", "imp": [{"id": "imp-1"}]}}


def _install(monkeypatch):
    entries, _ = merge_registry([{"name": "a", "intents": {"ADD_METRICS"}, "mcp": "http://a:8081", "display_name": "A"}], [])

    async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
        return ContainerInvocationModel(name="a", status=STATUS_OK, latency_ms=1.0, mutations=[], display_name="A", model_version="v")

    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
    monkeypatch.setattr(oapp, "_call_container_timed", fake_call)
    monkeypatch.setattr(oapp, "emit_bid_outcome", lambda *a, **k: None)
    monkeypatch.setattr(oapp, "emit_deal_yield_outcome", lambda *a, **k: None)


def _call(payload: bytes, metadata=None):
    async def run():
        server, port = oapp._build_grpc_server(0)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as ch:
                method = ch.unary_unary(oapp._GRPC_METHOD, request_serializer=lambda x: x, response_deserializer=lambda x: x)
                try:
                    return "ok", await method(payload, timeout=5.0, metadata=metadata)
                except grpc.aio.AioRpcError as exc:
                    return exc.code(), exc.details()
        finally:
            await server.stop(0)

    return asyncio.run(run())


class TestAuthDisabled:
    def test_reply_matches_http_shape_with_auth_timing(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.setenv("AUTH_DISABLED", "true")
        status, body = _call(json.dumps(ENVELOPE).encode())
        assert status == "ok"
        resp = json.loads(body)
        assert resp["id"] == "r1"
        md = resp["metadata"]
        assert md["containers"][0]["name"] == "a"
        assert "auth" in md["timing"] and "fan_out" in md["timing"]
        assert md["network_path"] == "direct"

    def test_fabric_header_as_metadata(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.setenv("AUTH_DISABLED", "true")
        status, body = _call(json.dumps(ENVELOPE).encode(), metadata=(("x-rtb-fabric-link-id", "link-7"),))
        assert status == "ok"
        md = json.loads(body)["metadata"]
        assert md["network_path"] == "rtb-fabric" and md["rtb_fabric_link_id"] == "link-7"

    def test_non_json_is_invalid_argument(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.setenv("AUTH_DISABLED", "true")
        status, detail = _call(b"\x00not json")
        assert status == grpc.StatusCode.INVALID_ARGUMENT


class TestFailClosed:
    def test_no_pool_is_unavailable(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.delenv("COGNITO_USER_POOL_ID", raising=False)
        status, detail = _call(json.dumps(ENVELOPE).encode(), metadata=(("authorization", "Bearer x"),))
        assert status == grpc.StatusCode.UNAVAILABLE
        assert "not configured" in detail

    def test_missing_token_is_unauthenticated(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_test")
        monkeypatch.setattr(oauth, "_SIGNATURE_VERIFIER_AVAILABLE", True)
        status, _ = _call(json.dumps(ENVELOPE).encode())
        assert status == grpc.StatusCode.UNAUTHENTICATED

    def test_invalid_token_is_unauthenticated(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_test")
        monkeypatch.setattr(oauth, "_SIGNATURE_VERIFIER_AVAILABLE", True)
        monkeypatch.setattr(oauth, "_verify_token", lambda token, region, pool: None)
        status, _ = _call(json.dumps(ENVELOPE).encode(), metadata=(("authorization", "Bearer bad"),))
        assert status == grpc.StatusCode.UNAUTHENTICATED

    def test_valid_token_without_scope_is_permission_denied(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_test")
        monkeypatch.setattr(oauth, "_SIGNATURE_VERIFIER_AVAILABLE", True)
        monkeypatch.setattr(oauth, "_verify_token", lambda token, region, pool: {"client_id": "c", "scope": "other/scope"})
        monkeypatch.setattr(oauth, "authorize", lambda path, claims: (False, "scope artf-orchestrator/mutations:write required"))
        status, detail = _call(json.dumps(ENVELOPE).encode(), metadata=(("authorization", "Bearer ok"),))
        assert status == grpc.StatusCode.PERMISSION_DENIED
        assert "mutations:write" in detail

    def test_valid_token_with_scope_is_served(self, monkeypatch):
        _install(monkeypatch)
        monkeypatch.delenv("AUTH_DISABLED", raising=False)
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_test")
        monkeypatch.setattr(oauth, "_SIGNATURE_VERIFIER_AVAILABLE", True)
        monkeypatch.setattr(oauth, "_verify_token", lambda token, region, pool: {"client_id": "c", "scope": "artf-orchestrator/mutations:write"})
        monkeypatch.setattr(oauth, "authorize", lambda path, claims: (True, "machine_scope"))
        status, body = _call(json.dumps(ENVELOPE).encode(), metadata=(("authorization", "Bearer ok"),))
        assert status == "ok"
        assert json.loads(body)["metadata"]["timing"]["auth"] >= 0
