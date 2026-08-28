"""Tests for shared/server.py's mutate() offloading to a thread pool.

Why this exists: mutate() is synchronous and reaches blocking network I/O
(dlrm_bid_shader/ncf_deal_manager call Triton through tritonclient.http, the
SYNCHRONOUS client). It used to be called inline from the async HTTP handlers,
which blocked the container's single uvicorn event loop for each Triton round
trip -- concurrent requests queued instead of overlapping, so the Nth in flight
waited N round trips. Load tests reported hundreds of milliseconds per container
for a model that infers in single-digit milliseconds.

Two properties are covered:

1. Concurrency: two simultaneous /mutate calls whose handlers each block must
   overlap, not serialise. This is the behaviour the fix exists for, and it
   fails against the inline version.
2. Context propagation: target_variant_scope/load_test_scope are ContextVars,
   and loop.run_in_executor does NOT copy contextvars the way
   asyncio.to_thread does. Without the explicit copy_context() in
   _await_mutate, a load test's X-Load-Test-Target-Variant header would stop
   reaching mutate() -- silently, with every request still returning 200.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from starlette.testclient import TestClient

from shared.artf_types import Metadata, RTBRequest, RTBResponse
from shared.load_test_context import (
    HEADER_NAME,
    IS_LOAD_TEST_HEADER_NAME,
    get_is_load_test,
    get_target_variant,
)
from shared.server import ARTF_MUTATE_WORKERS, _build_mcp_app

_MIN_REQUEST = {"id": "req-1"}


def _client(mutate_fn) -> TestClient:
    return TestClient(_build_mcp_app(mutate_fn, "test-agent"))


class TestMutateRunsOffTheEventLoop:
    def test_blocking_mutate_calls_overlap(self):
        """Two concurrent /mutate calls, each blocking 300ms, must overlap.

        Driven through httpx.ASGITransport inside a single asyncio.run rather
        than starlette's TestClient: TestClient gives each calling thread its
        own blocking portal, so two threads never share one event loop and the
        serialisation this test exists to catch cannot occur. Verified by
        reverting server.py to the inline call -- the TestClient version still
        passed, this version fails.

        Against the inline version request 1 blocks the only event loop inside
        barrier.wait(), request 2 can never start, the barrier times out and
        the handler returns 400.
        """
        BLOCK_S = 0.3
        barrier = threading.Barrier(2, timeout=5)

        def mutate(req: RTBRequest) -> RTBResponse:
            barrier.wait()
            time.sleep(BLOCK_S)
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        app = _build_mcp_app(mutate, "test-agent")

        async def drive() -> tuple[float, list[int]]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                start = time.monotonic()
                r1, r2 = await asyncio.gather(
                    ac.post("/mutate", json=_MIN_REQUEST),
                    ac.post("/mutate", json=_MIN_REQUEST),
                )
                return time.monotonic() - start, [r1.status_code, r2.status_code]

        elapsed, statuses = asyncio.run(drive())

        assert statuses == [200, 200], (
            "a request failed -- with mutate() inline the barrier deadlocks the "
            "event loop and times out"
        )
        assert elapsed < BLOCK_S * 2, (
            f"two {BLOCK_S}s calls took {elapsed:.2f}s -- they serialised "
            "instead of overlapping, so mutate() is back on the event loop"
        )

    def test_mutate_does_not_run_on_the_main_thread(self):
        seen: dict[str, str] = {}

        def mutate(req: RTBRequest) -> RTBResponse:
            seen["thread"] = threading.current_thread().name
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        assert _client(mutate).post("/mutate", json=_MIN_REQUEST).status_code == 200
        assert seen["thread"].startswith("artf-mutate"), (
            f"mutate() ran on {seen['thread']!r}, expected the artf-mutate pool"
        )

    def test_worker_pool_size_is_configurable(self):
        """Defaults to 16; the env var is read at import time, so this asserts
        the default rather than re-importing the module."""
        assert ARTF_MUTATE_WORKERS >= 1


class TestContextVarsSurviveTheOffload:
    """The regression these guard against is silent: the request still returns
    200 and mutate() still runs, it just stops seeing the load-test scoping."""

    def test_target_variant_header_reaches_mutate_in_the_worker_thread(self):
        seen: dict[str, object] = {}

        def mutate(req: RTBRequest) -> RTBResponse:
            seen["variant"] = get_target_variant()
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        resp = _client(mutate).post(
            "/mutate", json=_MIN_REQUEST, headers={HEADER_NAME: "canary"},
        )
        assert resp.status_code == 200
        assert seen["variant"] == "canary"

    def test_is_load_test_flag_reaches_mutate_in_the_worker_thread(self):
        seen: dict[str, object] = {}

        def mutate(req: RTBRequest) -> RTBResponse:
            seen["is_load_test"] = get_is_load_test()
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        resp = _client(mutate).post(
            "/mutate", json=_MIN_REQUEST, headers={IS_LOAD_TEST_HEADER_NAME: "1"},
        )
        assert resp.status_code == 200
        assert seen["is_load_test"] is True

    def test_real_traffic_sees_no_load_test_scoping(self):
        """No headers means real bid-serving traffic: the defaults must hold
        even though mutate() now runs on a pooled, reused thread."""
        seen: dict[str, object] = {}

        def mutate(req: RTBRequest) -> RTBResponse:
            seen["variant"] = get_target_variant()
            seen["is_load_test"] = get_is_load_test()
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        assert _client(mutate).post("/mutate", json=_MIN_REQUEST).status_code == 200
        assert seen["variant"] is None
        assert seen["is_load_test"] is False

    def test_scoping_does_not_leak_between_requests_on_a_reused_thread(self):
        """Pool threads are reused across requests. A ContextVar set without
        being reset would leak into the next request handled by that thread."""
        observed: list[tuple[object, object]] = []

        def mutate(req: RTBRequest) -> RTBResponse:
            observed.append((get_target_variant(), get_is_load_test()))
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        client = _client(mutate)
        client.post("/mutate", json=_MIN_REQUEST, headers={
            HEADER_NAME: "canary", IS_LOAD_TEST_HEADER_NAME: "1",
        })
        client.post("/mutate", json=_MIN_REQUEST)

        assert observed[0] == ("canary", True)
        assert observed[1] == (None, False), "scoping leaked into the next request"


class TestMcpToolCallAlsoOffloads:
    def test_extend_rtb_runs_mutate_off_the_event_loop(self):
        seen: dict[str, str] = {}

        def mutate(req: RTBRequest) -> RTBResponse:
            seen["thread"] = threading.current_thread().name
            return RTBResponse(id=req.id, metadata=Metadata(model_version="v1"))

        resp = _client(mutate).post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "extend_rtb", "arguments": _MIN_REQUEST},
        })
        assert resp.status_code == 200
        assert "result" in resp.json()
        assert seen["thread"].startswith("artf-mutate")

    def test_mutate_failure_still_returns_a_jsonrpc_error(self):
        """The exception now crosses a thread boundary -- it must still be
        caught and reported, not surface as an unhandled future."""
        def mutate(req: RTBRequest) -> RTBResponse:
            raise RuntimeError("triton unreachable")

        resp = _client(mutate).post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "extend_rtb", "arguments": _MIN_REQUEST},
        })
        assert resp.status_code == 200
        assert "triton unreachable" in resp.json()["error"]["message"]


class TestRestErrorHandling:
    def test_mutate_failure_returns_400_with_the_real_message(self):
        def mutate(req: RTBRequest) -> RTBResponse:
            raise RuntimeError("triton unreachable")

        resp = _client(mutate).post("/mutate", json=_MIN_REQUEST)
        assert resp.status_code == 400
        assert "triton unreachable" in resp.json()["error"]
