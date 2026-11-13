"""Orchestrator -> container over gRPC (Phase 2 of the gRPC transport plan).

A real gRPC server (shared/server.py's generic RTBExtensionPoint handler) on an
ephemeral port, called by the orchestrator's `_call_grpc` over a cached grpc.aio
channel. What is pinned:

- Outcome parity with the REST branch: a parsed reply is `reached`, carries
  mutations / model_version / abstained_reason / the container's `timing`; a
  servicer INTERNAL is `reached` + `error`; nothing listening is not reached.
- Load-test headers travel as gRPC metadata and reach the container's
  ContextVars (`target_variant_scope` / `load_test_scope`) inside mutate().
- Transport order in `_call_container`: with ARTF_CONTAINER_TRANSPORT=grpc the gRPC
  reply is definitive; a failed gRPC call falls through to REST /mutate and the
  gRPC error is the one reported if every transport fails; with `http` the gRPC
  port is never touched.
- The channel is cached per target (one object across calls) and `dns:///` is
  prepended so round_robin sees every address of a headless Service.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent import futures

import grpc
import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import app as oapp  # noqa: E402
from orchestrator.container_registry import derive_status  # noqa: E402
from shared import server as cserver  # noqa: E402
from shared.artf_types import Metadata, Mutation, RTBRequest, RTBResponse  # noqa: E402
from shared.load_test_context import HEADER_NAME, IS_LOAD_TEST_HEADER_NAME, get_is_load_test, get_target_variant  # noqa: E402

PAYLOAD = {"id": "r1", "tmax": 100, "bid_request": {"id": "br", "imp": [{"id": "imp-1"}]}}
PAYLOAD_BYTES = json.dumps(PAYLOAD).encode()


def _serve(mutate_fn) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    handler = grpc.method_handlers_generic_handler(cserver._SERVICE, {"GetMutations": cserver._grpc_handler(mutate_fn)})
    server.add_generic_rpc_handlers([handler])
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


@pytest.fixture(autouse=True)
def _fresh_channels():
    yield
    asyncio.run(oapp._close_grpc_channels())


def _run(coro):
    return asyncio.run(coro)


class TestCallGrpc:
    def test_parsed_reply_is_definitive_and_carries_metadata(self):
        def mutate(req: RTBRequest) -> RTBResponse:
            return RTBResponse(
                id=req.id,
                mutations=[Mutation(intent=4, op=2, path="/imp/imp-1/bidfloor")],
                metadata=Metadata(model_version="dlrm-v7", abstained_reason=None),
            )

        server, port = _serve(mutate)
        try:
            out = _run(oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0))
        finally:
            server.stop(0)
        assert out.reached and out.error is None
        assert len(out.mutations) == 1 and out.mutations[0].path == "/imp/imp-1/bidfloor"
        assert out.model_version == "dlrm-v7"
        assert out.timing and set(out.timing) == set(cserver.hop_timing.CONTAINER_SEGMENTS)
        assert derive_status(out) == "ok"

    def test_empty_reply_with_reason_is_no_mutations_not_error(self):
        def mutate(req):
            return RTBResponse(id=req.id, metadata=Metadata(abstained_reason="no prediction"))

        server, port = _serve(mutate)
        try:
            out = _run(oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0))
        finally:
            server.stop(0)
        assert out.reached and out.error is None and out.mutations == []
        assert out.abstained_reason == "no prediction"
        assert derive_status(out) == "no_mutations"

    def test_servicer_internal_is_reached_with_error(self):
        def mutate(req):
            raise RuntimeError("triton down")

        server, port = _serve(mutate)
        try:
            out = _run(oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0))
        finally:
            server.stop(0)
        assert out.reached is True
        assert out.error == "grpc INTERNAL"
        assert derive_status(out) == "error"

    def test_nothing_listening_is_unreached(self):
        server, port = _serve(lambda req: RTBResponse(id=req.id))
        server.stop(0)  # port now closed
        out = _run(oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 1.0))
        assert out.reached is False
        assert out.error == "grpc UNAVAILABLE"
        assert derive_status(out) == "unreachable"

    def test_headers_arrive_as_metadata_and_set_contextvars(self):
        seen = {}

        def mutate(req):
            seen["variant"] = get_target_variant()
            seen["load_test"] = get_is_load_test()
            return RTBResponse(id=req.id)

        server, port = _serve(mutate)

        # Both calls inside ONE event loop: a grpc.aio channel belongs to the loop
        # that created it, which is also why the orchestrator caches channels for
        # the life of its single uvicorn loop and never across loops.
        async def both():
            await oapp._call_grpc(
                f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0,
                headers={HEADER_NAME: "canary", IS_LOAD_TEST_HEADER_NAME: "1"},
            )
            first = dict(seen)
            seen.clear()
            await oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0)
            return first, dict(seen)

        try:
            first, second = _run(both())
        finally:
            server.stop(0)
        assert first == {"variant": "canary", "load_test": True}
        assert second == {"variant": None, "load_test": False}

    def test_channel_is_cached_per_target(self):
        server, port = _serve(lambda req: RTBResponse(id=req.id))
        try:
            async def two_calls():
                await oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0)
                first = oapp._GRPC_CHANNELS[f"127.0.0.1:{port}"]
                await oapp._call_grpc(f"127.0.0.1:{port}", PAYLOAD_BYTES, 2.0)
                return first is oapp._GRPC_CHANNELS[f"127.0.0.1:{port}"], len(oapp._GRPC_CHANNELS)
            same, count = _run(two_calls())
        finally:
            server.stop(0)
        assert same and count == 1

    def test_dns_scheme_for_round_robin(self):
        assert oapp._grpc_target("bid-pricer-grpc:50051") == "dns:///bid-pricer-grpc:50051"
        assert oapp._grpc_target("dns:///x:1") == "dns:///x:1"
        assert oapp._grpc_target("unix:///tmp/s") == "unix:///tmp/s"
        assert ("grpc.lb_policy_name", "round_robin") in oapp._GRPC_CHANNEL_OPTIONS


# ---------------------------------------------------------------- transport order

def _rest_app(calls: list, status=200, mcp_status=200):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def mutate(request):
        calls.append("rest")
        return JSONResponse({"id": "r1", "mutations": [], "metadata": {"model_version": "rest-v"}}, status_code=status)

    async def mcp(request):
        calls.append("mcp")
        return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": json.dumps({"id": "r1", "mutations": [], "metadata": {"model_version": "mcp-v"}})}]}}, status_code=mcp_status)

    return Starlette(routes=[Route("/mutate", mutate, methods=["POST"]), Route("/mcp", mcp, methods=["POST"])])


def _call(container, transport, monkeypatch, rest_calls, rest_status=200, mcp_status=200):
    monkeypatch.setattr(oapp, "ARTF_CONTAINER_TRANSPORT", transport)

    async def run():
        transport_ = httpx.ASGITransport(app=_rest_app(rest_calls, rest_status, mcp_status))
        async with httpx.AsyncClient(transport=transport_, base_url="http://c") as client:
            return await oapp._call_container(client, container, PAYLOAD, PAYLOAD_BYTES, 1.0)

    return _run(run())


class TestTransportOrder:
    def test_grpc_reply_is_definitive_rest_untouched(self, monkeypatch):
        server, port = _serve(lambda req: RTBResponse(id=req.id, metadata=Metadata(model_version="grpc-v")))
        rest_calls: list = []
        try:
            out = _call({"name": "a", "grpc": f"127.0.0.1:{port}", "mcp": "http://c"}, "grpc", monkeypatch, rest_calls)
        finally:
            server.stop(0)
        assert out.model_version == "grpc-v" and rest_calls == []

    def test_grpc_unreachable_falls_through_to_rest(self, monkeypatch):
        server, port = _serve(lambda req: RTBResponse(id=req.id))
        server.stop(0)
        rest_calls: list = []
        out = _call({"name": "a", "grpc": f"127.0.0.1:{port}", "mcp": "http://c"}, "grpc", monkeypatch, rest_calls)
        assert out.reached and out.error is None and out.model_version == "rest-v"
        assert rest_calls == ["rest"]

    def test_all_transports_fail_reports_grpc_error_first(self, monkeypatch):
        server, port = _serve(lambda req: RTBResponse(id=req.id))
        server.stop(0)
        rest_calls: list = []
        out = _call({"name": "a", "grpc": f"127.0.0.1:{port}", "mcp": "http://c"}, "grpc", monkeypatch, rest_calls, rest_status=500, mcp_status=500)
        # REST and MCP both answered badly (reached); the first transport's error leads.
        assert out.reached is True
        assert out.error == "grpc UNAVAILABLE"
        assert rest_calls == ["rest", "mcp"]

    def test_grpc_fails_rest_fails_mcp_answers(self, monkeypatch):
        server, port = _serve(lambda req: RTBResponse(id=req.id))
        server.stop(0)
        rest_calls: list = []
        out = _call({"name": "a", "grpc": f"127.0.0.1:{port}", "mcp": "http://c"}, "grpc", monkeypatch, rest_calls, rest_status=500)
        assert out.error is None and out.model_version == "mcp-v"
        assert rest_calls == ["rest", "mcp"]

    def test_http_transport_never_opens_grpc(self, monkeypatch):
        rest_calls: list = []
        out = _call({"name": "a", "grpc": "127.0.0.1:1", "mcp": "http://c"}, "http", monkeypatch, rest_calls)
        assert out.model_version == "rest-v" and rest_calls == ["rest"]
        assert oapp._GRPC_CHANNELS == {}

    def test_grpc_skipped_when_container_has_no_grpc_target(self, monkeypatch):
        rest_calls: list = []
        out = _call({"name": "a", "mcp": "http://c"}, "grpc", monkeypatch, rest_calls)
        assert out.model_version == "rest-v" and oapp._GRPC_CHANNELS == {}


class TestServicerMetadataParsing:
    def test_load_test_signals(self):
        f = cserver._load_test_signals
        assert f(None) == (None, False)
        assert f([(HEADER_NAME.lower(), "stable"), (IS_LOAD_TEST_HEADER_NAME.lower(), "1")]) == ("stable", True)
        assert f([(HEADER_NAME.lower(), "bogus"), (IS_LOAD_TEST_HEADER_NAME.lower(), "yes")]) == (None, False)
