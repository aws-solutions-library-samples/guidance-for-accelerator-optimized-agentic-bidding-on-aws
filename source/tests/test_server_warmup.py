"""Readiness warm-up in shared/server.py and its use by the Triton-backed containers.

run_artf_server(..., warmup=fn) keeps /health/ready at 503 until two consecutive
warm-up mutate() calls complete under ARTF_WARMUP_BUDGET_MS (Q18 A). Covered here:

- the gate: 503 before, 200 after two in-budget calls in a row; a failed or
  over-budget call resets the run and is retried with doubling backoff; at the
  attempt cap the state is failed, an error is logged and the pod stays 503
  while retrying; a Hypothesis property checks the end state against a
  reference model;
- no warmup kwarg: readiness is 200 from the start with the old body, on both
  the health port app and the MCP app;
- run_artf_server wires the gate to both HTTP apps and starts it;
- warmup_from_request: validates the canned request, requires a Triton call
  when asked, applies the container's check;
- the four Triton-backed containers: each canned request validates as an
  RTBRequest, carries the artf-warmup- id prefix, reaches the code path that
  calls Triton, and is passed to run_artf_server;
- Triton model_warmup blocks: each parses as a ModelConfig and its inputs match
  the declared inputs by name, data type and dims.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.testclient import TestClient

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from shared import hop_timing, server  # noqa: E402
from shared.artf_types import Metadata, RTBRequest, RTBResponse  # noqa: E402
from shared.server import (  # noqa: E402
    WARMUP_ID_PREFIX,
    WarmupGate,
    WarmupNotServed,
    _build_health_app,
    _build_mcp_app,
    warmup_from_request,
)

import containers.dlrm_bid_shader.app as dlrm_app  # noqa: E402
import containers.ncf_deal_manager.app as ncf_app  # noqa: E402
import containers.yield_optimizer_floor.app as floor_app  # noqa: E402
import containers.yield_optimizer_margin.app as margin_app  # noqa: E402

CONTAINERS = {
    "dlrm_bid_shader": dlrm_app,
    "ncf_deal_manager": ncf_app,
    "yield_optimizer_floor": floor_app,
    "yield_optimizer_margin": margin_app,
}


class _FakeClock:
    """A perf_counter whose readings advance by a scripted amount per warm-up."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _Exhausted(BaseException):
    """The scripted warm-up has no steps left (not an Exception, so the gate does not retry it)."""


def _scripted_warmup(clock: _FakeClock, steps):
    """A warm-up that, per call, takes the next step: a duration in ms or an exception."""
    remaining = list(steps)
    calls = []

    def _warmup() -> None:
        if not remaining:
            raise _Exhausted()
        step = remaining.pop(0)
        calls.append(step)
        if isinstance(step, Exception):
            raise step
        clock.now += step / 1000.0

    return _warmup, calls


def _gate(steps, *, budget_ms=20.0, max_attempts=50):
    clock = _FakeClock()
    warmup, calls = _scripted_warmup(clock, steps)
    sleeps = []
    gate = WarmupGate(
        warmup, budget_ms=budget_ms, max_attempts=max_attempts,
        backoff_s=(0.1, 0.4), clock=clock, sleep=sleeps.append,
    )
    return gate, calls, sleeps


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class TestReadinessGate:
    def test_defaults_are_q18_a(self):
        gate = WarmupGate(lambda: None)
        assert (gate.budget_ms, gate.consecutive, gate.max_attempts, gate.backoff_s) == (20.0, 2, 50, (0.1, 5.0))

    def test_ready_is_503_before_and_200_after_two_in_budget_calls(self):
        gate, calls, sleeps = _gate([12.0, 12.0])
        client = TestClient(_build_health_app(gate))

        before = client.get("/health/ready")
        assert before.status_code == 503
        assert before.json()["status"] == "warming"
        assert client.get("/health/live").status_code == 200

        gate.run()

        after = client.get("/health/ready")
        assert after.status_code == 200
        assert after.json()["status"] == "ok"
        assert after.json()["warmup"]["state"] == "ready"
        assert after.json()["warmup"]["last_ms"] == pytest.approx(12.0, abs=0.01)
        assert calls == [12.0, 12.0]
        assert sleeps == []

    def test_one_in_budget_call_is_not_enough(self):
        gate, calls, _ = _gate([5.0])
        with pytest.raises(_Exhausted):
            gate.run()
        assert gate.state == "warming" and gate.in_budget_run == 1
        assert TestClient(_build_health_app(gate)).get("/health/ready").status_code == 503

    def test_an_over_budget_call_resets_the_run_and_backs_off(self, caplog):
        gate, calls, sleeps = _gate([90.0, 10.0, 70.0, 10.0, 15.0])
        with caplog.at_level("INFO", logger="artf.server"):
            gate.run()
        assert gate.state == "ready"
        assert gate.attempts == 5
        assert sleeps == [0.1, 0.2]
        attempt_lines = [r.getMessage() for r in caplog.records if "warm-up attempt" in r.getMessage()]
        assert len(attempt_lines) == 5

    def test_the_cap_logs_an_error_and_keeps_the_pod_unready(self, caplog):
        gate, calls, _ = _gate([90.0, 90.0, 90.0], max_attempts=3)
        client = TestClient(_build_health_app(gate))
        with caplog.at_level("INFO", logger="artf.server"), pytest.raises(_Exhausted):
            gate.run()
        assert gate.state == "failed"
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1 and "not ready after 3 attempts" in errors[0].getMessage()
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "failed"

    def test_after_the_cap_only_an_in_budget_pair_makes_it_ready(self):
        boom = RuntimeError("Triton not ready")
        gate, calls, sleeps = _gate([boom, boom, boom, boom, 10.0, 10.0], max_attempts=1)
        client = TestClient(_build_health_app(gate))
        gate.run()
        assert gate.state == "ready"
        assert gate.attempts == 6
        assert sleeps == [0.1, 0.2, 0.4, 0.4]
        assert client.get("/health/ready").status_code == 200

    def test_a_failure_leaves_ready_503_with_the_error(self):
        class _Stop(Exception):
            pass

        def _stop_after_first_attempt(_seconds):
            raise _Stop()

        def _failing_warmup():
            raise RuntimeError("Triton not ready")

        gate = WarmupGate(_failing_warmup, sleep=_stop_after_first_attempt)
        with pytest.raises(_Stop):
            gate.run()
        resp = TestClient(_build_health_app(gate)).get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["warmup"]["attempts"] == 1
        assert resp.json()["warmup"]["last_error"] == "RuntimeError: Triton not ready"

    @settings(max_examples=200, deadline=None)
    @given(
        steps=st.lists(
            st.one_of(st.integers(min_value=0, max_value=60).map(float), st.just("fail")),
            max_size=12,
        ),
        cap=st.integers(min_value=1, max_value=6),
    )
    def test_gate_state_matches_a_reference_model(self, steps, cap):
        """Property (PBT-03, oracle): ready only after two in-budget completions in a
        row; a failure or an over-budget completion resets the run; after ``cap``
        attempts without readiness the state is failed, never ready."""
        budget = 20.0
        expected_state, expected_attempts, run = "warming", len(steps) + 1, 0
        for i, step in enumerate(steps):
            if step != "fail" and step <= budget:
                run += 1
                if run >= 2:
                    expected_state, expected_attempts = "ready", i + 1
                    break
                continue
            run = 0
            if i + 1 >= cap and expected_state == "warming":
                expected_state = "failed"

        script = [RuntimeError("down") if s == "fail" else s for s in steps]
        gate, _, _ = _gate(script, budget_ms=budget, max_attempts=cap)
        try:
            gate.run()
            exhausted = False
        except _Exhausted:
            exhausted = True
        assert exhausted == (expected_state != "ready")
        assert gate.state == expected_state
        assert gate.attempts == expected_attempts
        assert gate.ready == (expected_state == "ready")

    def test_mcp_app_ready_route_reports_the_same_gate(self):
        gate, _, _ = _gate([5.0, 5.0])
        client = TestClient(_build_mcp_app(lambda r: RTBResponse(id=r.id), "t", gate=gate))
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health/live").status_code == 200
        gate.run()
        assert client.get("/health/ready").status_code == 200


class TestWithoutWarmup:
    def test_health_app_ready_is_200_with_the_old_body(self):
        client = TestClient(_build_health_app(None))
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert client.get("/health/live").json() == {"status": "ok"}

    def test_mcp_app_ready_is_200_with_the_old_body(self):
        client = TestClient(_build_mcp_app(lambda r: RTBResponse(id=r.id), "t"))
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_run_artf_server_without_warmup_starts_no_gate(self, monkeypatch):
        served = _capture_run(monkeypatch, warmup=None)
        assert served["health"].get("/health/ready").json() == {"status": "ok"}
        assert served["mcp"].get("/health/ready").json() == {"status": "ok"}
        assert served["grpc_started"] is True


def _capture_run(monkeypatch, **kwargs):
    """Call run_artf_server with gRPC and uvicorn replaced; return test clients for both apps."""
    import uvicorn

    apps = {}
    state = {"grpc_started": False}

    class _FakeGrpc:
        def start(self):
            state["grpc_started"] = True

    def _fake_run(app, host, port, log_level):
        apps[port] = app

    monkeypatch.setattr(server, "_build_grpc_server", lambda fn, port: _FakeGrpc())
    monkeypatch.setattr(uvicorn, "run", _fake_run)
    server.run_artf_server(
        lambda r: RTBResponse(id=r.id), agent_name="t",
        grpc_port=1, mcp_port=8081, health_port=8080, **kwargs,
    )
    deadline = time.monotonic() + 5
    while 8080 not in apps and time.monotonic() < deadline:
        time.sleep(0.01)  # the health app runs on its own thread
    return {
        "health": TestClient(apps[8080]),
        "mcp": TestClient(apps[8081]),
        "grpc_started": state["grpc_started"],
    }


class TestRunArtfServerWiring:
    def test_warmup_gates_both_ports_then_reports_ready(self, monkeypatch):
        release = []

        def _warmup():
            deadline = time.monotonic() + 5
            while not release and time.monotonic() < deadline:
                time.sleep(0.005)

        # A generous budget, so the only thing holding readiness back is the
        # warm-up that has not returned yet.
        original = server.WarmupGate
        monkeypatch.setattr(
            server, "WarmupGate",
            lambda fn: original(fn, budget_ms=10_000.0, backoff_s=(0.0, 0.0)),
        )
        served = _capture_run(monkeypatch, warmup=_warmup)
        assert served["grpc_started"] is True
        assert served["health"].get("/health/ready").status_code == 503
        assert served["mcp"].get("/health/ready").status_code == 503
        assert served["health"].get("/health/live").status_code == 200

        release.append(True)
        deadline = time.monotonic() + 5
        while served["health"].get("/health/ready").status_code != 200 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert served["health"].get("/health/ready").status_code == 200
        assert served["mcp"].get("/health/ready").status_code == 200


# ---------------------------------------------------------------------------
# warmup_from_request
# ---------------------------------------------------------------------------

_REQ = {"id": "artf-warmup-test", "bid_request": {}}


class TestWarmupFromRequest:
    def test_rejects_an_id_without_the_prefix(self):
        with pytest.raises(ValueError, match="artf-warmup-"):
            warmup_from_request(lambda r: RTBResponse(id=r.id), {"id": "req-1"}, require_triton=False)

    def test_rejects_a_request_that_does_not_validate(self):
        with pytest.raises(Exception):
            warmup_from_request(lambda r: RTBResponse(id=r.id), {"bid_request": {}}, require_triton=False)

    def test_require_triton_fails_when_mutate_makes_no_triton_call(self):
        warm = warmup_from_request(lambda r: RTBResponse(id=r.id), _REQ, require_triton=True)
        with pytest.raises(WarmupNotServed):
            warm()

    def test_require_triton_passes_when_mutate_calls_triton(self):
        def _mutate(req):
            with hop_timing.triton_call():
                pass
            return RTBResponse(id=req.id)

        warmup_from_request(_mutate, _REQ, require_triton=True)()

    def test_check_rejects_the_response(self):
        def _check(resp):
            raise RuntimeError("abstained")

        warm = warmup_from_request(lambda r: RTBResponse(id=r.id), _REQ, require_triton=False, check=_check)
        with pytest.raises(RuntimeError, match="abstained"):
            warm()

    def test_mutate_receives_the_validated_request(self):
        seen = []
        warmup_from_request(lambda r: seen.append(r) or RTBResponse(id=r.id), _REQ, require_triton=False)()
        assert isinstance(seen[0], RTBRequest) and seen[0].id == "artf-warmup-test"


# ---------------------------------------------------------------------------
# The four Triton-backed containers
# ---------------------------------------------------------------------------

def _load_deal_book(monkeypatch):
    """Load the book from DEALS_SEED_FILE (conftest.py), as ncf __main__ primes it.

    The snapshot is a module global; monkeypatch puts the previous value back.
    """
    monkeypatch.setattr(ncf_app.deal_library, "_book", ncf_app.deal_library._book)
    monkeypatch.setattr(ncf_app.deal_library, "_loaded_at", ncf_app.deal_library._loaded_at)
    ncf_app.deal_library.load()


@pytest.mark.parametrize("name", sorted(CONTAINERS))
class TestContainerWarmupRequests:
    def test_canned_request_validates_and_is_prefixed(self, name):
        req = RTBRequest(**CONTAINERS[name].WARMUP_REQUEST)
        assert req.id.startswith(WARMUP_ID_PREFIX)
        assert req.bid_request["user"]["id"].startswith(WARMUP_ID_PREFIX)

    def test_main_passes_warmup_to_run_artf_server(self, name):
        tree = ast.parse((SOURCE_DIR / "containers" / name / "app.py").read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_artf_server"
        ]
        assert len(calls) == 1
        kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
        assert "warmup" in kwargs
        warm = kwargs["warmup"]
        assert isinstance(warm, ast.Call) and getattr(warm.func, "id", "") == "warmup_from_request"
        assert [a.id for a in warm.args] == ["mutate", "WARMUP_REQUEST"]
        warm_kwargs = {kw.arg: kw.value for kw in warm.keywords}
        assert getattr(warm_kwargs["require_triton"], "id", "") == "USE_TRITON"
        assert getattr(warm_kwargs["check"], "id", "") == "warmup_check"

    def test_warmup_runs_on_the_inline_path(self, name, monkeypatch):
        # USE_TRITON is unset in tests, so mutate() runs in-process and no
        # Triton call is required.
        app = CONTAINERS[name]
        if name == "ncf_deal_manager":
            _load_deal_book(monkeypatch)
        warmup_from_request(app.mutate, app.WARMUP_REQUEST, require_triton=False, check=app.warmup_check)()


class TestCannedRequestsReachInference:
    """Each canned request reaches the call that goes to Triton on the served path."""

    def test_dlrm_predicts_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            dlrm_app, "_predict_ctr_from_request",
            lambda br: calls.append(br) or (0.01, "stable", ""),
        )
        resp = dlrm_app.mutate(RTBRequest(**dlrm_app.WARMUP_REQUEST))
        assert len(calls) == 1
        assert len(resp.mutations) == 1  # the 25.00 bid is shaded toward the floor

    def test_dlrm_check_rejects_an_abstention(self):
        abstained = RTBResponse(id="x", metadata=Metadata(abstained_reason="inference_unavailable: down"))
        with pytest.raises(RuntimeError, match="abstained"):
            dlrm_app.warmup_check(abstained)
        dlrm_app.warmup_check(RTBResponse(id="x", metadata=Metadata()))

    def test_ncf_scores_the_offered_deal(self, monkeypatch):
        scored = []

        def _score(uid, deal_ids, **kwargs):
            scored.append(list(deal_ids))
            return [], [], "", ""

        monkeypatch.setattr(ncf_app, "_score_deals", _score)
        _load_deal_book(monkeypatch)
        ncf_app.mutate(RTBRequest(**ncf_app.WARMUP_REQUEST))
        # One impression, one Triton batch: the offered deal first, then any
        # deal-book candidates for this request.
        assert len(scored) == 1
        assert scored[0][0] == "deal-remnant-open-hearth"

    @pytest.mark.parametrize("app,predict", [(floor_app, "_predict_floor"), (margin_app, "_predict_margin")])
    def test_yield_predicts_once_per_deal(self, monkeypatch, app, predict):
        vectors = []
        no_change = 1.0 if app is floor_app else 0.0
        monkeypatch.setattr(app, predict, lambda fv, target_variant=None: vectors.append(fv) or (no_change, "", ""))
        resp = app.mutate(RTBRequest(**app.WARMUP_REQUEST))
        assert len(vectors) == 2
        assert all(len(v) == 7 for v in vectors)
        assert resp.mutations == []

    @pytest.mark.parametrize("app", [ncf_app, floor_app, margin_app])
    def test_triton_readiness_check_is_skipped_inline(self, app):
        assert app.USE_TRITON is False
        app.warmup_check(RTBResponse(id="x"))


# ---------------------------------------------------------------------------
# Triton model_warmup blocks
# ---------------------------------------------------------------------------

REPO = SOURCE_DIR / "triton" / "model_repository"
WARMED_MODELS = [
    "dlrm_bid_shader_stable",
    "dlrm_bid_shader_canary",
    "ncf_deal_manager_stable",
    "deal_yield_manager_floor",
    "deal_yield_manager_margin",
]


def _model_config(name):
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2

    return text_format.Parse((REPO / name / "config.pbtxt").read_text(), model_config_pb2.ModelConfig())


@pytest.mark.parametrize("name", WARMED_MODELS)
def test_model_warmup_inputs_match_declared_inputs(name):
    cfg = _model_config(name)
    assert cfg.name == name
    assert len(cfg.model_warmup) >= 1
    declared = {i.name: i for i in cfg.input}
    for sample in cfg.model_warmup:
        assert 1 <= sample.batch_size <= cfg.max_batch_size
        assert set(sample.inputs) == set(declared), sample.name
        for key, value in sample.inputs.items():
            assert value.data_type == declared[key].data_type, key
            assert list(value.dims) == list(declared[key].dims), key
            assert value.WhichOneof("input_data_type") == "zero_data", key


def _bootstrap_opt_batches() -> dict:
    """Engine name -> the opt batch of its input profiles in deployment/optimizer-bootstrap.json."""
    text = (SOURCE_DIR.parent / "deployment" / "optimizer-bootstrap.json").read_text()
    spec = json.loads(text.replace("__FEATURE_SPEC_VERSION__", "0"))
    batches = {}
    for model in spec["models"]:
        engine = model["output_uri"].split("/triton-models/", 1)[1].split("/", 1)[0]
        opts = {profile["opt"][0] for profile in model["input_profiles"].values()}
        assert len(opts) == 1, engine
        batches[engine] = opts.pop()
    return batches


@pytest.mark.parametrize("name", WARMED_MODELS)
def test_model_warmup_covers_batch_1_and_the_bootstrap_opt_batch(name):
    """FR-16: TensorRT engines warm batch 1 and their profile's opt batch; FIL has no profile."""
    opt = _bootstrap_opt_batches()
    expected = [1, opt[name]] if name in opt else [1]
    assert [s.batch_size for s in _model_config(name).model_warmup] == expected
    if not name.startswith("deal_yield_manager"):
        assert name in opt


def test_routers_have_no_model_warmup():
    # The Python-backend routers forward to the engines; only the engines warm.
    for router in ("dlrm_bid_shader", "ncf_deal_manager"):
        assert len(_model_config(router).model_warmup) == 0
