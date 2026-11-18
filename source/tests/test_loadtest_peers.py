"""Cross-replica load test lookup (orchestrator/loadtest_peers.py).

The orchestrator runs with two or more replicas and keeps a running load test in
the memory of the replica that accepted the POST. Behind the internal NLB the
UI's poll and stop requests land on either replica; on the wrong one the test
did not exist and the UI reset to idle one second after Run. A replica that
misses a test now
asks its siblings, found through the headless orchestrator-grpc Service, and the
POST guard asks them whether anything is running before starting.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
from starlette.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import loadtest, loadtest_peers  # noqa: E402
from orchestrator.loadtest_peers import NO_FORWARD_HEADER  # noqa: E402

AUTH = "Bearer test-token"


_RealAsyncClient = httpx.AsyncClient


def _mock_client_factory(handler):
    """Make loadtest_peers' httpx.AsyncClient talk to `handler` instead of the network.

    `loadtest_peers.httpx` is the shared httpx module, so the patch replaces
    `httpx.AsyncClient` globally for the duration; the factory must build the real
    class captured at import time or it calls itself."""
    def factory(**kwargs):
        kwargs.pop("transport", None)
        return _RealAsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    return factory


def _app():
    return Starlette(routes=[
        Route("/v1/loadtest", loadtest.start_loadtest, methods=["POST"]),
        Route("/v1/loadtest/running", loadtest.get_running_loadtest, methods=["GET"]),
        Route("/v1/loadtest/{id}", loadtest.get_loadtest, methods=["GET"]),
        Route("/v1/loadtest/{id}", loadtest.cancel_loadtest, methods=["DELETE"]),
    ])


@pytest.fixture(autouse=True)
def _clean_state():
    loadtest._active_tests.clear()
    loadtest._expiry_times.clear()
    loadtest._active_task = None
    yield
    loadtest._active_tests.clear()
    loadtest._expiry_times.clear()
    loadtest._active_task = None


def _seed_running(test_id, completed=7):
    loadtest._active_tests[test_id] = loadtest.LoadTestStatus(
        id=test_id, state="running", preset="100", total_requests=100,
        completed=0, errors=0, elapsed_ms=0.0, rps=0.0,
        latency_p50=0.0, latency_p95=0.0, latency_p99=0.0,
        latency_min=0.0, latency_avg=0.0, latency_max=0.0,
        histogram={"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0},
        per_container=[],
    )
    loadtest._progress_completed[test_id] = completed
    loadtest._progress_latencies[test_id] = [5.0] * completed
    loadtest._progress_errors[test_id] = 0
    loadtest._progress_start_time[test_id] = __import__("time").monotonic() - 1.0


class TestDiscoverPeers:
    def _infos(self, *ips):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]

    def test_resolves_headless_service_and_excludes_self(self):
        async def fake_getaddrinfo(*a, **k):
            return self._infos("10.0.0.5", "10.0.0.6", "10.0.0.6", "10.0.0.7")

        with patch.dict(os.environ, {"ORCHESTRATOR_PEER_DNS": "orchestrator-grpc", "POD_IP": "10.0.0.6"}), \
             patch("asyncio.base_events.BaseEventLoop.getaddrinfo", fake_getaddrinfo):
            peers = asyncio.run(loadtest_peers.discover_peers())
        assert peers == ["10.0.0.5", "10.0.0.7"]

    def test_empty_dns_name_disables_forwarding(self):
        with patch.dict(os.environ, {"ORCHESTRATOR_PEER_DNS": ""}):
            assert asyncio.run(loadtest_peers.discover_peers()) == []

    def test_unresolvable_name_is_no_peers(self):
        async def failing(*a, **k):
            raise socket.gaierror("nope")

        with patch.dict(os.environ, {"ORCHESTRATOR_PEER_DNS": "orchestrator-grpc"}), \
             patch("asyncio.base_events.BaseEventLoop.getaddrinfo", failing):
            assert asyncio.run(loadtest_peers.discover_peers()) == []


class TestForwardToOwner:
    def _get(self, client, test_id, headers=None):
        h = {"authorization": AUTH}
        h.update(headers or {})
        return client.get(f"/v1/loadtest/{test_id}", headers=h)

    def test_unknown_test_is_served_from_the_owning_sibling(self):
        seen = []

        def handler(req: httpx.Request):
            seen.append(req)
            if req.url.host == "10.0.0.5":
                return httpx.Response(404, json={"error": "Load test not found"})
            return httpx.Response(200, json={"id": "lt-owner", "state": "running", "completed": 42})

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5", "10.0.0.7"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)):
            r = self._get(TestClient(_app()), "lt-owner")

        assert r.status_code == 200
        assert r.json()["completed"] == 42
        # Both siblings were asked, on the HTTP port, same path, with the caller's
        # token and the no-forward marker.
        assert sorted(req.url.host for req in seen) == ["10.0.0.5", "10.0.0.7"]
        for req in seen:
            assert req.url.port == 8000
            assert req.url.path == "/v1/loadtest/lt-owner"
            assert req.headers["authorization"] == AUTH
            assert req.headers[NO_FORWARD_HEADER] == "1"

    def test_all_siblings_404_gives_404(self):
        def handler(req):
            return httpx.Response(404, json={"error": "Load test not found"})

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)):
            r = self._get(TestClient(_app()), "lt-nowhere")
        assert r.status_code == 404
        assert r.json()["error"] == "Load test not found"

    def test_sibling_timeout_is_tolerated(self):
        def handler(req):
            if req.url.host == "10.0.0.5":
                raise httpx.ConnectTimeout("slow", request=req)
            return httpx.Response(200, json={"id": "lt-x", "state": "running", "completed": 3})

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5", "10.0.0.7"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)):
            r = self._get(TestClient(_app()), "lt-x")
        assert r.status_code == 200
        assert r.json()["completed"] == 3

    def test_forwarded_request_never_forwards_again(self):
        asked = []

        def handler(req):
            asked.append(req)
            return httpx.Response(200, json={"id": "lt-loop"})

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)):
            r = self._get(TestClient(_app()), "lt-loop", headers={NO_FORWARD_HEADER: "1"})
        assert r.status_code == 404
        assert asked == []

    def test_no_siblings_means_plain_404(self):
        with patch.object(loadtest_peers, "discover_peers", return_value=[]):
            r = self._get(TestClient(_app()), "lt-solo")
        assert r.status_code == 404

    def test_local_test_is_answered_locally_without_asking(self):
        _seed_running("lt-mine", completed=7)
        with patch.object(loadtest_peers, "discover_peers", side_effect=AssertionError("must not be called")):
            r = self._get(TestClient(_app()), "lt-mine")
        assert r.status_code == 200
        assert r.json()["completed"] == 7

    def test_stop_is_forwarded_to_the_owner(self):
        seen = []

        def handler(req):
            seen.append(req)
            return httpx.Response(200, json={"ok": True, "message": "Load test cancelled and task killed"})

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)):
            r = TestClient(_app()).delete("/v1/loadtest/lt-remote", headers={"authorization": AUTH})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert seen[0].method == "DELETE"
        assert seen[0].url.path == "/v1/loadtest/lt-remote"


class TestRunningEndpointAndPostGuard:
    def test_running_reports_this_replicas_test(self):
        _seed_running("lt-here")
        task = MagicMock()
        task.done.return_value = False
        loadtest._active_task = task
        r = TestClient(_app()).get("/v1/loadtest/running", headers={"authorization": AUTH})
        assert r.json() == {"id": "lt-here"}

    def test_running_is_null_when_idle(self):
        r = TestClient(_app()).get("/v1/loadtest/running", headers={"authorization": AUTH})
        assert r.json() == {"id": None}

    def test_post_refuses_when_a_sibling_is_running(self):
        seen = []

        def handler(req):
            seen.append(req)
            return httpx.Response(200, json={"id": "lt-elsewhere"})

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)):
            r = TestClient(_app()).post(
                "/v1/loadtest", json={"preset": "100", "seed": 42},
                headers={"authorization": AUTH},
            )
        assert r.status_code == 409
        assert r.json()["running_id"] == "lt-elsewhere"
        assert seen[0].url.path == "/v1/loadtest/running"
        assert seen[0].headers["authorization"] == AUTH
        assert seen[0].headers[NO_FORWARD_HEADER] == "1"
        assert loadtest._active_task is None

    def test_post_proceeds_when_siblings_are_idle_or_silent(self):
        def handler(req):
            if req.url.host == "10.0.0.5":
                raise httpx.ConnectError("down", request=req)
            return httpx.Response(200, json={"id": None})

        started = {}

        async def fake_run(test_id, *a, **k):
            started["id"] = test_id

        with patch.object(loadtest_peers, "discover_peers", return_value=["10.0.0.5", "10.0.0.7"]), \
             patch.object(loadtest_peers.httpx, "AsyncClient", _mock_client_factory(handler)), \
             patch.object(loadtest, "_run_load_test", fake_run):
            r = TestClient(_app()).post(
                "/v1/loadtest", json={"preset": "100", "seed": 42},
                headers={"authorization": AUTH},
            )
        assert r.status_code == 202
        assert started["id"] == r.json()["id"]

    def test_any_peer_running_skips_when_request_is_itself_forwarded(self):
        scope = {
            "type": "http", "method": "POST", "path": "/v1/loadtest", "query_string": b"",
            "headers": [(NO_FORWARD_HEADER.encode(), b"1")],
        }
        with patch.object(loadtest_peers, "discover_peers", side_effect=AssertionError("must not be called")):
            assert asyncio.run(loadtest_peers.any_peer_running(Request(scope))) is None
