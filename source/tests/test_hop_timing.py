"""Per-hop timing reported on RTBResponse.metadata.timing (shared/hop_timing.py).

Before any transport changes, every container answer and every orchestrator answer states where its
time went, so a later gRPC number can be attributed to a segment rather than
compared as one opaque total.

Covered:
- Container REST /mutate: all CONTAINER_SEGMENTS present; `triton` equals the
  time spent inside hop_timing.triton_call() blocks executed on the worker
  thread (the accumulator crosses the copy_context boundary); `mutate` >= `triton`.
- Container gRPC GetMutations: same segments on the JSON-over-gRPC reply.
- No accumulator outside a timed request: triton_call() is a no-op and never
  raises.
- Orchestrator /v1/mutations: ORCHESTRATOR_SEGMENTS present, `auth` carried from
  request.state when the middleware set it and absent otherwise, and the
  segments are consistent with the fan-out it ran.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import hop_timing  # noqa: E402
from shared.artf_types import ContainerInvocationModel, Metadata, RTBRequest, RTBResponse  # noqa: E402
from shared.server import _RTBExtensionPointServicer, _build_mcp_app  # noqa: E402
from orchestrator.container_registry import STATUS_OK, merge_registry  # noqa: E402

_REQ = {"id": "req-1", "bid_request": {"id": "br-1", "imp": [{"id": "imp-1"}]}}


def _mutate_with_two_triton_calls(req: RTBRequest) -> RTBResponse:
    with hop_timing.triton_call():
        time.sleep(0.02)
    with hop_timing.triton_call():
        time.sleep(0.01)
    time.sleep(0.005)
    return RTBResponse(id=req.id, metadata=Metadata(model_version="m-1"))


class TestContainerRest:
    def test_all_segments_present_and_triton_attributed(self):
        client = TestClient(_build_mcp_app(_mutate_with_two_triton_calls, "t"))
        r = client.post("/mutate", json=_REQ)
        assert r.status_code == 200
        timing = r.json()["metadata"]["timing"]
        assert set(timing) == set(hop_timing.CONTAINER_SEGMENTS)
        # Two sleeps inside triton_call(), 30 ms together, run on the executor
        # thread; the handler still sees them through the shared list.
        assert 25 <= timing["triton"] <= 200
        assert timing["mutate"] >= timing["triton"]
        assert timing["total"] >= timing["mutate"] + timing["parse"]
        assert all(v >= 0 for v in timing.values())
        # The private handler-start marker never reaches the wire.
        assert "_handler_start" not in timing
        # Existing metadata is preserved alongside the new key.
        assert r.json()["metadata"]["model_version"] == "m-1"

    def test_triton_segment_is_zero_when_mutate_makes_no_triton_call(self):
        client = TestClient(_build_mcp_app(lambda req: RTBResponse(id=req.id), "t"))
        timing = client.post("/mutate", json=_REQ).json()["metadata"]["timing"]
        assert timing["triton"] == 0.0

    def test_two_concurrent_requests_do_not_share_an_accumulator(self):
        """Each request's triton figure is its own: 20 ms vs 0 ms, not pooled."""

        def mutate(req: RTBRequest) -> RTBResponse:
            if req.id == "slow":
                with hop_timing.triton_call():
                    time.sleep(0.02)
            return RTBResponse(id=req.id)

        async def run():
            transport = httpx.ASGITransport(app=_build_mcp_app(mutate, "t"))
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await asyncio.gather(
                    c.post("/mutate", json={"id": "slow"}),
                    c.post("/mutate", json={"id": "fast"}),
                )

        slow, fast = asyncio.run(run())
        assert slow.json()["metadata"]["timing"]["triton"] >= 15
        assert fast.json()["metadata"]["timing"]["triton"] == 0.0


class TestContainerGrpc:
    def test_servicer_reply_carries_timing(self):
        class Ctx:
            def set_code(self, *_): ...
            def set_details(self, *_): ...
            def invocation_metadata(self): return ()

        servicer = _RTBExtensionPointServicer(_mutate_with_two_triton_calls)
        out = json.loads(servicer.GetMutations(json.dumps(_REQ).encode(), Ctx()))
        timing = out["metadata"]["timing"]
        assert set(timing) == set(hop_timing.CONTAINER_SEGMENTS)
        assert 25 <= timing["triton"] <= 200
        assert timing["mutate"] >= timing["triton"]
        assert out["metadata"]["model_version"] == "m-1"


class TestAccumulatorOutsideRequest:
    def test_triton_call_is_a_noop_without_accumulator(self):
        hop_timing.clear_triton_accumulator()
        with hop_timing.triton_call():
            pass  # must not raise
        assert hop_timing.triton_ms(None) == 0.0


# --------------------------------------------------------------- orchestrator

_CONTAINERS = [
    {"name": "a", "intents": {"ADD_METRICS"}, "mcp": "http://a:8081", "display_name": "A"},
    {"name": "b", "intents": {"ACTIVATE_SEGMENTS"}, "mcp": "http://b:8081", "display_name": "B"},
]


def _install(monkeypatch):
    from orchestrator import app as oapp

    entries, _ = merge_registry(_CONTAINERS, [])

    async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
        await asyncio.sleep(0.01 if container["name"] == "a" else 0.0)
        return ContainerInvocationModel(
            name=container["name"], status=STATUS_OK, latency_ms=10.0 if container["name"] == "a" else 1.0,
            mutations=[], display_name=container["display_name"], model_version="v",
        )

    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
    monkeypatch.setattr(oapp, "_call_container_timed", fake_call)
    monkeypatch.setattr(oapp, "emit_bid_outcome", lambda *a, **k: None)
    monkeypatch.setattr(oapp, "emit_deal_yield_outcome", lambda *a, **k: None)
    return oapp


def _post(oapp, body, *, auth_ms=None):
    from starlette.applications import Starlette
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.routing import Route

    class StampAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if auth_ms is not None:
                request.state.auth_ms = auth_ms
            return await call_next(request)

    async def run():
        app = Starlette(routes=[Route("/v1/mutations", oapp.get_mutations, methods=["POST"])])
        app.add_middleware(StampAuth)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            return await c.post("/v1/mutations", json=body)

    return asyncio.run(run())


class TestOrchestrator:
    def test_segments_present_with_auth_from_middleware(self, monkeypatch):
        oapp = _install(monkeypatch)
        r = _post(oapp, {"id": "r1", "tmax": 100, "bid_request": {"id": "br"}}, auth_ms=1.25)
        assert r.status_code == 200
        md = r.json()["metadata"]
        timing = md["timing"]
        assert set(timing) == set(hop_timing.ORCHESTRATOR_SEGMENTS)
        assert timing["auth"] == 1.25
        # The fan-out waited on the 10 ms container.
        assert timing["fan_out"] >= 9
        assert timing["total"] >= timing["parse"] + timing["fan_out"] + timing["merge"] + timing["emit"]
        # total_latency_ms (fan-out wall clock) and the segment agree in kind.
        assert abs(md["total_latency_ms"] - timing["fan_out"]) < 5

    def test_auth_absent_when_middleware_did_not_run(self, monkeypatch):
        oapp = _install(monkeypatch)
        timing = _post(oapp, {"id": "r1", "bid_request": {"id": "br"}}).json()["metadata"]["timing"]
        assert "auth" not in timing
        assert set(timing) == set(hop_timing.ORCHESTRATOR_SEGMENTS) - {"auth"}

    def test_bypassed_request_still_reports_timing(self, monkeypatch):
        oapp = _install(monkeypatch)
        body = {"id": "r1", "bid_request": {"id": "br", "ext": {"artf": {"bypass": True}}}}
        md = _post(oapp, body).json()["metadata"]
        assert md["bypassed"] is True
        assert md["timing"]["fan_out"] < 5
        assert "total" in md["timing"]
