"""Unit tests for orchestrator.loadtest -- focused on the yield load-test
targeting wiring added to close the Yield Optimizer's cold-start gap (a genesis
model never emits a mutation on its own, so without this wiring neither live nor
load-test traffic could ever produce real outcome data to train on).

Covers:
- deal_yield_manager_floor and deal_yield_manager_margin are both selectable
  target_model_types, and each maps to its OWN CONTAINERS registry entry.
- The pre-split container-level "deal_yield_manager" target is rejected rather
  than aliased, so a stale caller fails loudly instead of recording runs that
  no training target can match.
- ALL_INTENTS includes ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN so BOTH yield
  containers are included in the fan-out (a container whose intents don't
  overlap applicable_intents is filtered out by _filter_containers -- see
  orchestrator/app.py). Checked per container: each serves exactly one intent
  now, so dropping either intent would starve one model while the other kept
  receiving traffic.
- _run_load_test's yield branch calls emit_load_test_deal_yield_outcome() (not
  the bid-shading emit_load_test_outcome()) and aggregates its returned samples
  into outcome_sample_count/outcome_samples correctly, including the
  zero-samples and multiple-samples-per-request cases.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import loadtest
from orchestrator.loadtest import ALL_INTENTS, LoadTestRequest, _MODEL_TYPE_TO_CONTAINER_NAME, _TARGET_MODEL_TYPES
from shared.artf_types import ContainerInvocationModel


class TestTargetModelTypeRegistry:
    @pytest.mark.parametrize(
        "model_type", ["deal_yield_manager_floor", "deal_yield_manager_margin"]
    )
    def test_both_yield_model_types_are_selectable(self, model_type):
        assert model_type in _TARGET_MODEL_TYPES

    @pytest.mark.parametrize(
        "model_type,container",
        [
            ("deal_yield_manager_floor", "yield-optimizer-floor"),
            ("deal_yield_manager_margin", "yield-optimizer-margin"),
        ],
    )
    def test_yield_model_types_map_to_their_own_container(self, model_type, container):
        assert _MODEL_TYPE_TO_CONTAINER_NAME[model_type] == container

    @pytest.mark.parametrize(
        "model_type", ["deal_yield_manager_floor", "deal_yield_manager_margin"]
    )
    def test_load_test_request_accepts_both_yield_model_types(self, model_type):
        req = LoadTestRequest(preset="100", target_model_type=model_type)
        assert req.target_model_type == model_type

    def test_the_pre_split_combined_model_type_is_no_longer_accepted(self):
        """The container-level "deal_yield_manager" target matched neither real
        training model type. It is gone rather than aliased, so a stale caller
        fails loudly instead of silently recording runs nothing can train on."""
        with pytest.raises(Exception):
            LoadTestRequest(preset="100", target_model_type="deal_yield_manager")
        assert "deal_yield_manager" not in _TARGET_MODEL_TYPES
        assert "deal_yield_manager" not in _MODEL_TYPE_TO_CONTAINER_NAME

    def test_load_test_request_rejects_unknown_model_type(self):
        with pytest.raises(Exception):
            LoadTestRequest(preset="100", target_model_type="not_a_real_model")

    def test_every_selectable_type_maps_to_a_container(self):
        """A type selectable in the API but absent from the container map would
        target nothing and silently capture zero outcomes."""
        for t in _TARGET_MODEL_TYPES:
            assert t in _MODEL_TYPE_TO_CONTAINER_NAME, t


class TestAllIntentsIncludesDealYield:
    def test_adjust_deal_floor_included(self):
        assert "ADJUST_DEAL_FLOOR" in ALL_INTENTS

    def test_adjust_deal_margin_included(self):
        assert "ADJUST_DEAL_MARGIN" in ALL_INTENTS

    def test_both_yield_containers_would_be_included_in_fanout(self):
        """Regression guard read off the REAL CONTAINERS registry, not a
        hardcoded intent set: each yield container's own intents must overlap
        ALL_INTENTS, or _filter_containers(ALL_INTENTS) silently drops it from
        every load test's fan-out regardless of target_model_type targeting.

        Checked per container because each now serves exactly one intent --
        dropping ADJUST_DEAL_MARGIN from ALL_INTENTS would starve the margin
        model while the floor model kept receiving traffic, which the old
        combined check could not have caught."""
        # orchestrator.app imports grpc (a real runtime dependency, present in
        # the container image and CI). Skip rather than assert a weaker
        # hardcoded restatement of the registry when it is unavailable.
        pytest.importorskip("grpc")
        from orchestrator.app import CONTAINERS

        yield_containers = [
            c for c in CONTAINERS if c["name"].startswith("yield-optimizer-")
        ]
        assert len(yield_containers) == 2, [c["name"] for c in yield_containers]
        for c in yield_containers:
            assert c["intents"] & set(ALL_INTENTS), c["name"]


def _make_invocation(name: str, mutations=None, model_version="v1", status="ok") -> ContainerInvocationModel:
    return ContainerInvocationModel(
        name=name, status=status, latency_ms=1.0, mutations=mutations or [], model_version=model_version,
    )


class TestRunLoadTestDealYieldWiring:
    """Exercises _run_load_test's yield branch directly (not through the HTTP
    route), mocking only the external-service boundary
    (_call_container_timed / FeedbackCollector), per this project's testing
    convention.

    Targets the floor container; the margin container travels the identical
    code path (same branch, same emitter) with its own model type."""

    def _payload(self):
        return {
            "id": "br-0",
            "tmax": 100,
            "applicable_intents": ALL_INTENTS,
            "bid_request": {
                "id": "br-0",
                "imp": [
                    {
                        "id": "imp-0",
                        "bidfloor": 3.0,
                        "pmp": {"deals": [{"id": "deal-a", "bidfloor": 3.0, "at": 1}]},
                    }
                ],
            },
        }

    def test_deal_yield_target_uses_deal_yield_emitter_not_bid_shading_emitter(self, monkeypatch):
        """The yield branch must call emit_load_test_deal_yield_outcome(),
        never the bid-shading emit_load_test_outcome() (which would construct
        the wrong event type)."""
        from shared.artf_types import AdjustDealPayload, Intent, Operation, Mutation

        floor_mutation = Mutation(
            intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
            path="/imp/imp-0/deals/deal-a",
            adjust_deal=AdjustDealPayload(bidfloor=2.5),
        )

        fake_containers = [{"name": "yield-optimizer-floor", "intents": {"ADJUST_DEAL_FLOOR"}}]

        async def fake_filter_containers(intents):
            return fake_containers

        # _filter_containers is a plain (sync) function in app.py; match that.
        def fake_filter_containers_sync(intents):
            return fake_containers

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("yield-optimizer-floor", mutations=[floor_mutation], model_version="deal-yield-floor-xgboost-v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, fake_filter_containers_sync),
        )

        mock_deal_yield_emit = MagicMock(return_value=[1.5])
        mock_bid_shade_emit = MagicMock(return_value=0.0)
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", mock_deal_yield_emit)
        monkeypatch.setattr(loadtest, "emit_load_test_outcome", mock_bid_shade_emit)

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-1", "100", seed=1, duration_s=5,
                target_model_type="deal_yield_manager_floor", target_variant="current",
            )
        )

        mock_deal_yield_emit.assert_called()
        mock_bid_shade_emit.assert_not_called()

        status = loadtest._active_tests["test-dy-1"]
        assert status.target_model_type == "deal_yield_manager_floor"
        assert status.outcome_sample_count > 0
        assert status.outcome_samples == [1.5] * status.outcome_sample_count

    def test_zero_mutations_produces_zero_samples(self, monkeypatch):
        """When a yield container returns no mutations for a request (the
        genesis model's normal constant-output behavior), the run must record
        zero samples for that request rather than fabricating one."""
        fake_containers = [{"name": "yield-optimizer-floor", "intents": {"ADJUST_DEAL_FLOOR"}}]

        def fake_filter_containers_sync(intents):
            return fake_containers

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("yield-optimizer-floor", mutations=[], model_version="deal-yield-floor-xgboost-v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, fake_filter_containers_sync),
        )

        mock_deal_yield_emit = MagicMock(return_value=[])
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", mock_deal_yield_emit)

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-2", "100", seed=1, duration_s=5,
                target_model_type="deal_yield_manager_floor", target_variant="current",
            )
        )

        status = loadtest._active_tests["test-dy-2"]
        assert status.outcome_sample_count == 0
        assert status.outcome_samples == []

    def test_seed_and_duration_are_recorded_in_status_and_history(self, monkeypatch):
        """The Governance compare flow replays a baseline run's seed and duration
        cap for the challenger run, so both must reach the persisted record."""
        fake_containers = [{"name": "yield-optimizer-floor", "intents": {"ADJUST_DEAL_FLOOR"}}]

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("yield-optimizer-floor", mutations=[], model_version="deal-yield-floor-xgboost-v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, lambda intents: fake_containers),
        )
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", MagicMock(return_value=[]))
        monkeypatch.setattr(loadtest, "_LOADTEST_TABLE", "")

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-seed", "100", seed=7, duration_s=5,
                target_model_type="deal_yield_manager_floor", target_variant="current",
            )
        )

        status = loadtest._active_tests["test-dy-seed"]
        assert status.seed == 7
        assert status.duration_s == 5
        saved = next(r for r in loadtest._history_cache if r["id"] == "test-dy-seed")
        assert saved["seed"] == 7
        assert saved["duration_s"] == 5

    def test_records_without_seed_or_duration_still_load(self):
        """History written before these fields existed has neither key."""
        status = loadtest.LoadTestStatus(
            id="lt-old", state="complete", preset="1k", total_requests=1, completed=1,
            errors=0, elapsed_ms=1.0, rps=1.0, latency_p50=0, latency_p95=0, latency_p99=0,
            latency_min=0, latency_avg=0, latency_max=0, histogram={}, per_container=[],
        )
        assert status.seed is None
        assert status.duration_s is None

    def test_multiple_mutations_per_request_are_all_counted(self, monkeypatch):
        """One request can still carry several mutations from a single yield
        container -- one per PMP deal it adjusts -- and every one must be
        counted, not deduplicated to one.

        Pre-split this exercised floor+margin on a single deal from one
        container. Each container serves one intent now, so the multi-mutation
        case is multi-deal instead; the counting path under test is the same."""
        fake_containers = [{"name": "yield-optimizer-floor", "intents": {"ADJUST_DEAL_FLOOR"}}]

        def fake_filter_containers_sync(intents):
            return fake_containers

        async def fake_call_container_timed(client, container, payload, payload_bytes, timeout_s, headers=None):
            return _make_invocation("yield-optimizer-floor", mutations=[], model_version="v1")

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: (fake_containers, fake_call_container_timed, fake_filter_containers_sync),
        )

        # Every targeted request "emits" exactly 2 samples (two adjusted deals).
        mock_deal_yield_emit = MagicMock(return_value=[1.0, 2.0])
        monkeypatch.setattr(loadtest, "emit_load_test_deal_yield_outcome", mock_deal_yield_emit)

        asyncio.run(
            loadtest._run_load_test(
                "test-dy-3", "100", seed=1, duration_s=5,
                target_model_type="deal_yield_manager_floor", target_variant="current",
            )
        )

        status = loadtest._active_tests["test-dy-3"]
        # preset "100" == 100 requests, each contributing 2 samples.
        assert status.outcome_sample_count == 200



class TestGetLoadtestLiveProgress:
    """GET /v1/loadtest/{id} is the UI's transport behind the UI API proxy Lambda,
    which cannot carry the SSE stream. While a test runs, the stored
    LoadTestStatus is the start-time snapshot (all zeros), so the endpoint must
    merge the live counters from the _progress_* dicts; otherwise the panel
    shows zeros for the whole run and the bid-bubble animation (driven by
    `completed` increasing) never fires."""

    def _seed_running(self, test_id, total=1000):
        loadtest._active_tests[test_id] = loadtest.LoadTestStatus(
            id=test_id, state="running", preset="1k", total_requests=total,
            completed=0, errors=0, elapsed_ms=0.0, rps=0.0,
            latency_p50=0.0, latency_p95=0.0, latency_p99=0.0,
            latency_min=0.0, latency_avg=0.0, latency_max=0.0,
            histogram={"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0},
            per_container=[],
        )
        loadtest._progress_latencies[test_id] = [4.0, 12.0, 35.0, 80.0]
        loadtest._progress_completed[test_id] = 4
        loadtest._progress_errors[test_id] = 1
        # Started two seconds ago, so rps is a real positive number.
        loadtest._progress_start_time[test_id] = __import__("time").monotonic() - 2.0
        loadtest._expiry_times.pop(test_id, None)

    def _request(self, test_id):
        return MagicMock(path_params={"id": test_id})

    def _body(self, response):
        import json
        return json.loads(response.body)

    def test_running_test_returns_live_counters(self):
        tid = "lt-live-1"
        self._seed_running(tid)
        body = self._body(asyncio.run(loadtest.get_loadtest(self._request(tid))))
        assert body["state"] == "running"
        assert body["total_requests"] == 1000
        assert body["completed"] == 4
        assert body["errors"] == 1
        assert body["elapsed_ms"] >= 2000.0
        assert body["rps"] > 0
        assert body["latency_min"] == 4.0
        assert body["latency_max"] == 80.0
        assert body["latency_p50"] == 35.0
        assert body["histogram"] == {"lt_10ms": 1, "10_30ms": 1, "30_50ms": 1, "gt_50ms": 1}

    def test_running_test_keeps_stored_identity_fields(self):
        tid = "lt-live-2"
        self._seed_running(tid)
        loadtest._active_tests[tid].target_model_type = "deal_yield_manager_margin"
        loadtest._active_tests[tid].scenario = "late_night"
        body = self._body(asyncio.run(loadtest.get_loadtest(self._request(tid))))
        assert body["id"] == tid
        assert body["preset"] == "1k"
        assert body["target_model_type"] == "deal_yield_manager_margin"
        assert body["scenario"] == "late_night"

    def test_completed_test_returns_stored_status_untouched(self):
        tid = "lt-done-1"
        self._seed_running(tid)
        final = loadtest._active_tests[tid].model_copy(update={
            "state": "complete", "completed": 1000, "errors": 3, "elapsed_ms": 20893.97,
            "rps": 47.86, "latency_p50": 9.1, "latency_p95": 21.4, "latency_p99": 38.0,
            "latency_min": 2.0, "latency_avg": 10.2, "latency_max": 90.0,
            "histogram": {"lt_10ms": 600, "10_30ms": 350, "30_50ms": 40, "gt_50ms": 10},
            "total_mutations": 1953,
        })
        loadtest._active_tests[tid] = final
        # Stale live dicts must not leak into a finished test's body.
        loadtest._progress_completed[tid] = 999
        body = self._body(asyncio.run(loadtest.get_loadtest(self._request(tid))))
        assert body == final.model_dump()

    def test_poll_body_matches_sse_progress_event(self):
        """Both transports read _live_progress, so the numbers cannot drift."""
        import json
        tid = "lt-sse-1"
        self._seed_running(tid)

        async def first_event():
            gen = loadtest._sse_event_generator(tid)
            try:
                return await gen.__anext__()
            finally:
                await gen.aclose()

        event = asyncio.run(first_event())
        assert event.startswith("event: progress\n")
        sse = json.loads(event.split("data: ", 1)[1].strip())
        poll = self._body(asyncio.run(loadtest.get_loadtest(self._request(tid))))
        for key in ("completed", "errors", "latency_p50", "latency_p95", "latency_p99", "histogram"):
            assert sse[key] == poll[key], key
        assert sse["total"] == poll["total_requests"]
        # elapsed_ms/rps are clock-derived; the two reads are milliseconds apart.
        assert abs(sse["elapsed_ms"] - poll["elapsed_ms"]) < 500

    def test_unknown_test_is_404(self):
        response = asyncio.run(loadtest.get_loadtest(self._request("lt-missing")))
        assert response.status_code == 404


class TestPacedPreset:
    """FR3.1 to FR3.5, NFR4: the 60k-paced preset starts requests at a fixed
    rate for a fixed window; the back-to-back presets are unchanged."""

    _CONTAINER = {"name": "dlrm-bid-shader", "intents": {"BID_SHADE"}}

    def _stub(self, monkeypatch, latency_s: float, starts: list, payloads: list | None = None):
        import time as _time

        async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
            starts.append(_time.monotonic())
            if payloads is not None:
                payloads.append(payload)
            await asyncio.sleep(latency_s)
            return _make_invocation(self._CONTAINER["name"])

        monkeypatch.setattr(
            loadtest, "_get_app_deps",
            lambda: ([self._CONTAINER], fake_call, lambda intents: [self._CONTAINER]),
        )
        monkeypatch.setattr(loadtest, "_LOADTEST_TABLE", "")

    def test_existing_presets_are_unchanged(self):
        assert loadtest.PRESET_CONFIG["100"] == {"total": 100, "concurrency": 10}
        assert loadtest.PRESET_CONFIG["1k"] == {"total": 1_000, "concurrency": 25}
        assert loadtest.PRESET_CONFIG["10k"] == {"total": 10_000, "concurrency": 50}
        assert loadtest.PRESET_CONFIG["100k"] == {"total": 100_000, "concurrency": 100}
        assert loadtest.PRESET_CONFIG["60k-paced"] == {
            "total": 60_000, "concurrency": 100, "rate_per_s": 100, "duration_s": 600,
        }

    def test_paced_preset_is_accepted_and_duration_forced_to_600(self, monkeypatch):
        monkeypatch.setattr(loadtest, "any_peer_running", AsyncMock(return_value=None))
        monkeypatch.setattr(loadtest, "_run_load_test", AsyncMock(return_value=None))
        monkeypatch.setattr(loadtest, "_active_task", None)
        request = MagicMock()
        request.json = AsyncMock(return_value={"preset": "60k-paced", "seed": 3, "duration_s": 30})

        response = asyncio.run(loadtest.start_loadtest(request))

        assert response.status_code == 202
        test_id = __import__("json").loads(response.body)["id"]
        status = loadtest._active_tests[test_id]
        assert status.duration_s == 600
        assert status.total_requests == 60_000
        assert status.target_rps == 100.0
        assert loadtest._run_load_test.call_args.args[3] == 600
        monkeypatch.setattr(loadtest, "_active_task", None)

    def test_paced_run_spreads_requests_across_the_window(self, monkeypatch):
        monkeypatch.setitem(
            loadtest.PRESET_CONFIG, "60k-paced",
            {"total": 50, "concurrency": 100, "rate_per_s": 50, "duration_s": 1},
        )
        starts: list = []
        payloads: list = []
        self._stub(monkeypatch, latency_s=0.01, starts=starts, payloads=payloads)

        asyncio.run(loadtest._run_load_test("lt-paced-ok", "60k-paced", seed=5, duration_s=1))

        status = loadtest._active_tests["lt-paced-ok"]
        assert status.completed == 50 == len(starts)
        assert status.errors == 0
        assert status.target_rps == 50.0
        # 50 starts at 50/s span about 0.98 s rather than arriving in one batch.
        assert 0.85 <= max(starts) - min(starts) <= 1.1
        assert 45 <= status.rps <= 55  # NFR4: within 10% of the target rate

        # Lazy generation gives the same request stream as pre-generating.
        import random as _random

        rng = _random.Random(5)
        profile = loadtest.get_traffic_profile(loadtest.DEFAULT_SCENARIO)
        expected = [loadtest.generate_payload(rng, i, profile) for i in range(50)]
        assert payloads == expected

    def test_slow_containers_stop_at_the_deadline_without_counting_unsent(self, monkeypatch):
        monkeypatch.setitem(
            loadtest.PRESET_CONFIG, "60k-paced",
            {"total": 50, "concurrency": 2, "rate_per_s": 50, "duration_s": 1},
        )
        starts: list = []
        self._stub(monkeypatch, latency_s=0.3, starts=starts)

        asyncio.run(loadtest._run_load_test("lt-paced-slow", "60k-paced", seed=5, duration_s=1))

        status = loadtest._active_tests["lt-paced-slow"]
        assert status.total_requests == 50
        assert 2 <= status.completed < 50
        assert status.completed == len(starts)
        assert status.rps < 50

    def test_batch_presets_report_no_target_rate(self, monkeypatch):
        starts: list = []
        self._stub(monkeypatch, latency_s=0.0, starts=starts)
        asyncio.run(loadtest._run_load_test("lt-batch", "100", seed=1, duration_s=5))
        status = loadtest._active_tests["lt-batch"]
        assert status.completed == 100
        assert status.target_rps is None

    def test_request_model_accepts_paced_preset(self):
        assert LoadTestRequest(preset="60k-paced").preset == "60k-paced"
        with pytest.raises(Exception):
            LoadTestRequest(preset="60k")


class TestErrorCounting:
    """errors counts REQUESTS with a failed container call; per_container[].errors
    counts each container's failed calls. Adding every failed call to the request
    count let errors exceed completed (UI: 1,890 errors for 1,818 sent, received
    -72, error rate 103.96%)."""

    def _run(self, monkeypatch, test_id, statuses):
        containers = [{"name": n, "intents": {"BID_SHADE"}} for n in statuses]

        async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
            inv = _make_invocation(container["name"], mutations=[], model_version="v1")
            return inv.model_copy(update={"status": statuses[container["name"]]})

        monkeypatch.setattr(loadtest, "_get_app_deps", lambda: (containers, fake_call, lambda intents: containers))
        monkeypatch.setattr(loadtest, "emit_load_test_outcome", MagicMock(return_value=None))
        monkeypatch.setattr(loadtest, "_LOADTEST_TABLE", "")
        asyncio.run(loadtest._run_load_test(test_id, "100", seed=1, duration_s=5))
        return loadtest._active_tests[test_id]

    def test_two_failing_containers_count_each_request_once(self, monkeypatch):
        status = self._run(monkeypatch, "lt-err-2of3", {"a": "unreachable", "b": "timeout", "c": "ok"})
        assert status.completed == 100
        assert status.errors == 100
        assert status.errors <= status.completed
        by_name = {c["name"]: c for c in status.per_container}
        assert by_name["a"]["errors"] == 100
        assert by_name["b"]["errors"] == 100
        assert by_name["c"]["errors"] == 0

    def test_no_failures_counts_no_errors(self, monkeypatch):
        status = self._run(monkeypatch, "lt-err-none", {"a": "ok", "b": "no_mutations"})
        assert status.errors == 0
        assert all(c["errors"] == 0 for c in status.per_container)

    def test_live_progress_carries_the_per_container_breakdown(self):
        tid = "lt-live-pc"
        loadtest._progress_latencies[tid] = [5.0]
        loadtest._progress_completed[tid] = 1
        loadtest._progress_errors[tid] = 1
        loadtest._progress_start_time[tid] = __import__("time").monotonic() - 1.0
        loadtest._progress_per_container_latencies[tid] = {"a": [3.0], "b": [5.0]}
        loadtest._progress_per_container_mutations[tid] = {"a": 1, "b": 0}
        loadtest._progress_per_container_errors[tid] = {"a": 0, "b": 1}
        loadtest._progress_per_container_timing[tid] = {"a": {}, "b": {}}
        live = loadtest._live_progress(tid)
        by_name = {c["name"]: c for c in live["per_container"]}
        assert by_name["a"]["total_mutations"] == 1
        assert by_name["b"]["errors"] == 1
        assert by_name["b"]["avg_latency_ms"] == 5.0
