"""Aggregation of container-reported timing (Phase 0 of the gRPC transport plan).

- loadtest.py: each invocation's ``timing`` is recorded per segment, a derived
  ``transport`` segment is latency_ms - timing.total, and per_container entries
  carry p50/p95 of latency and of each segment. A container that reports no
  timing contributes nothing and the entry shows {} / 0.0 (never a fabricated
  figure).
- measure_hops.py: the summary attributes the client round trip into hop A
  transport (client - orchestrator auth - total) and, per container, hop B
  transport (latency_ms - container total); unreported segments render n/a.
- _call_container passes a container's ``metadata.timing`` through unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import loadtest  # noqa: E402
from orchestrator import measure_hops  # noqa: E402
from shared.artf_types import ContainerInvocationModel  # noqa: E402


def _inv(latency, timing):
    return ContainerInvocationModel(name="a", status="ok", latency_ms=latency, timing=timing)


class TestLoadTestRecording:
    def test_segments_and_transport_recorded(self):
        store: dict = {}
        loadtest._record_container_timing(store, _inv(12.0, {"parse": 1.0, "mutate": 8.0, "triton": 6.0, "total": 10.0}))
        loadtest._record_container_timing(store, _inv(14.0, {"parse": 1.5, "mutate": 9.0, "triton": 7.0, "total": 11.0}))
        assert store["triton"] == [6.0, 7.0]
        assert store["transport"] == [2.0, 3.0]

    def test_no_timing_records_nothing(self):
        store: dict = {}
        loadtest._record_container_timing(store, _inv(12.0, None))
        assert store == {}
        assert loadtest.summarize_container_timing(store) == {
            "timing_p50": {}, "timing_p95": {}, "transport_p50_ms": 0.0, "transport_p95_ms": 0.0,
        }

    def test_transport_never_negative(self):
        store: dict = {}
        # A container total larger than the orchestrator's own clock (clock skew
        # between processes) is clamped rather than reported as negative time.
        loadtest._record_container_timing(store, _inv(5.0, {"total": 9.0}))
        assert store["transport"] == [0.0]

    def test_summary_has_p50_per_segment(self):
        store = {"triton": [1.0, 2.0, 3.0, 4.0], "total": [5.0, 6.0, 7.0, 8.0], "transport": [1.0, 1.0, 2.0, 9.0]}
        out = loadtest.summarize_container_timing(store)
        assert out["timing_p50"]["triton"] == 3.0
        assert out["timing_p50"]["total"] == 7.0
        assert "transport" not in out["timing_p50"]
        assert out["transport_p50_ms"] == 2.0
        assert out["transport_p95_ms"] == 9.0

    def test_missing_store_is_ignored(self):
        loadtest._record_container_timing(None, _inv(1.0, {"total": 1.0}))  # no raise


class TestCallContainerPassThrough:
    def test_timing_from_mutate_response_reaches_outcome(self):
        from orchestrator import app as oapp

        async def handler(request):
            from starlette.responses import JSONResponse
            return JSONResponse({"id": "r", "mutations": [], "metadata": {"model_version": "v", "timing": {"total": 3.5, "triton": 2.0}}})

        from starlette.applications import Starlette
        from starlette.routing import Route

        app = Starlette(routes=[Route("/mutate", handler, methods=["POST"])])

        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://c") as client:
                return await oapp._call_container(
                    client, {"name": "a", "mcp": "http://c", "grpc": "c:50051", "intents": set()},
                    {"id": "r"}, b"{}", 1.0,
                )

        outcome = asyncio.run(run())
        assert outcome.reached and outcome.error is None
        assert outcome.timing == {"total": 3.5, "triton": 2.0}


def _record(client_ms, timing, containers, status=200):
    return {
        "i": 0, "client_ms": client_ms, "status": status,
        "body": {"id": "r", "mutations": [], "metadata": {"timing": timing, "total_latency_ms": timing.get("fan_out"), "containers": containers}},
    }


class TestMeasureHopsSummary:
    def test_attribution(self):
        recs = [
            _record(
                40.0,
                {"auth": 1.0, "parse": 2.0, "fan_out": 20.0, "merge": 1.0, "emit": 0.5, "total": 24.0},
                [
                    {"name": "a", "status": "ok", "latency_ms": 18.0, "timing": {"total": 15.0, "triton": 5.0}},
                    {"name": "b", "status": "skipped", "latency_ms": 0},
                ],
            )
        ] * 3
        s = measure_hops.summarize(recs)
        assert s["requests"] == 3 and s["http_200"] == 3
        assert s["client_ms"]["p50"] == 40.0
        # 40 - (24 + 1) = 15 ms outside the orchestrator's own accounting.
        assert s["orchestrator"]["hop_a_transport_ms"]["p50"] == 15.0
        assert s["orchestrator"]["segments"]["fan_out"]["p50"] == 20.0
        assert s["containers"]["a"]["hop_b_transport_ms"]["p50"] == 3.0
        assert s["containers"]["a"]["segments"]["triton"]["p50"] == 5.0
        # Skipped containers count in statuses but not in latency.
        assert s["containers"]["b"]["statuses"] == {"skipped": 3}
        assert s["containers"]["b"]["latency_ms"]["p50"] is None
        md = measure_hops.render_markdown(s, label="t")
        assert "| orchestrator.fan_out | 20.00 |" in md
        assert "n/a" in md  # b has no latency

    def test_non_200_excluded_from_server_segments_but_counted(self):
        recs = [_record(5.0, {}, [], status=401)]
        recs[0]["body"] = {"error": "x"}
        s = measure_hops.summarize(recs)
        assert s["http_200"] == 0 and s["statuses"] == {"401": 1}
        assert s["orchestrator"]["segments"] == {}
        assert s["client_ms"]["p50"] == 5.0

    def test_payload_wrapping(self, tmp_path):
        bare = tmp_path / "br.json"
        bare.write_text(json.dumps({"id": "br-1", "imp": [{"id": "1"}]}))
        env = measure_hops._read_payload(str(bare))
        assert env["bid_request"]["id"] == "br-1" and env["tmax"] == 100
        already = tmp_path / "env.json"
        already.write_text(json.dumps({"id": "e", "bid_request": {"id": "x"}}))
        assert measure_hops._read_payload(str(already)) == {"id": "e", "bid_request": {"id": "x"}}
