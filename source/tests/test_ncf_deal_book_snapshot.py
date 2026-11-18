"""The Deal Scorer's deal book is a snapshot: primed at startup, refreshed off the request path.

ARTF hot-path rule: an ARTF container makes no external call while handling a
request. These tests pin that for the Deal Scorer:

  * a failed prime raises, and app.py's __main__ exits non-zero before the server
    starts, so the pod never becomes ready;
  * a failed refresh keeps the last good snapshot and logs the failure;
  * a successful refresh swaps in the new records, and the background thread does
    it on its own schedule;
  * with every DynamoDB entry point patched to raise, mutate() still answers from
    the snapshot and none of the patched entry points is called.
"""
import importlib.util
import json
import logging
import os
import runpy
import subprocess
import sys
import time

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)

from shared.artf_types import Intent, RTBRequest  # noqa: E402

_CONTAINER = os.path.join(_ROOT, "containers", "ncf_deal_manager")
_APP = os.path.join(_CONTAINER, "app.py")
_SAMPLES = os.path.join(_ROOT, "frontend-react", "public", "samples")

_counter = 0


def _fresh(path, prefix):
    global _counter
    _counter += 1
    spec = importlib.util.spec_from_file_location(f"{prefix}_{_counter}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def library():
    module = _fresh(os.path.join(_ROOT, "shared", "deal_library.py"), "deal_library_snapshot")
    yield module
    module.stop()


@pytest.fixture
def seed():
    with open(os.environ["DEALS_SEED_FILE"], encoding="utf-8") as handle:
        return json.load(handle)


def _write_seed(path, deals):
    path.write_text(json.dumps({"format": "artf-deals-seed/1", "dealsapi": "1.0", "deals": deals, "campaigns": []}))


def _parenting():
    with open(os.path.join(_SAMPLES, "parenting-narrative.json"), encoding="utf-8") as handle:
        return json.load(handle)["bid_request"]


def _ridesharer_airport():
    return {
        "id": "r-airport",
        "app": {"bundle": "com.ridesharer.app", "domain": "ridesharer.example", "cat": ["653"], "cattax": 9},
        "user": {"id": "u-1"},
        "imp": [{
            "id": "imp-1", "banner": {"w": 320, "h": 480}, "bidfloor": 5.0,
            "pmp": {"private_auction": 0, "deals": [
                {"id": "deal-ridesharer-airport-open-travel", "bidfloor": 12.0, "bidfloorcur": "USD", "at": 1, "guar": 0},
                {"id": "deal-ridesharer-airport-open-luggage", "bidfloor": 12.0, "bidfloorcur": "USD", "at": 1, "guar": 0},
            ]},
        }],
    }


# ------------------------------------------------------------------ priming


def test_nothing_is_served_before_the_book_is_loaded(library):
    with pytest.raises(RuntimeError, match="deal book not loaded"):
        library.deals_for(_parenting())
    with pytest.raises(RuntimeError, match="deal book not loaded"):
        library.all_deal_ids()


def test_start_raises_when_the_source_cannot_be_read(library, monkeypatch):
    monkeypatch.setenv("DEALS_TABLE", "stack-deals")
    monkeypatch.delenv("DEALS_SEED_FILE", raising=False)

    def unreachable(name):
        raise ConnectionError("table unreachable")

    monkeypatch.setattr(library, "_table", unreachable)
    with pytest.raises(ConnectionError, match="table unreachable"):
        library.start()
    with pytest.raises(RuntimeError, match="deal book not loaded"):
        library.deals_for(_parenting())


def test_start_raises_when_no_source_is_configured(library, monkeypatch):
    monkeypatch.delenv("DEALS_TABLE", raising=False)
    monkeypatch.delenv("DEALS_SEED_FILE", raising=False)
    with pytest.raises(ValueError, match="DEALS_TABLE"):
        library.start()


def test_an_empty_source_is_a_failed_prime(library, monkeypatch, tmp_path):
    empty = tmp_path / "empty.json"
    _write_seed(empty, [])
    monkeypatch.setenv("DEALS_SEED_FILE", str(empty))
    with pytest.raises(ValueError, match="no deal records"):
        library.load()


# ---------------------------------------------------------------- refreshing


def test_a_failed_refresh_keeps_the_last_good_snapshot_and_logs(library, monkeypatch, caplog):
    library.load()
    before = library.deals_for(_parenting())
    monkeypatch.setattr(library, "read_deal_records", lambda: (_ for _ in ()).throw(OSError("throttled")))
    with caplog.at_level(logging.ERROR, logger="ncf.deal_library"):
        assert library.refresh_once() is False
    assert library.deals_for(_parenting()) == before
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "deal book refresh from DEALS_SEED_FILE=" in message
    assert "throttled" in message
    assert "keeping snapshot loaded at" in message


def test_a_successful_refresh_swaps_in_the_new_records(library, monkeypatch, tmp_path, seed):
    path = tmp_path / "seed.json"
    _write_seed(path, seed["deals"])
    monkeypatch.setenv("DEALS_SEED_FILE", str(path))
    library.load()
    assert "deal-parenting-premium" in library.all_deal_ids()

    _write_seed(path, [d for d in seed["deals"] if d["id"] != "deal-parenting-premium"])
    assert library.refresh_once() is True
    assert "deal-parenting-premium" not in library.all_deal_ids()


def test_the_background_thread_refreshes_on_its_own(library, monkeypatch, tmp_path, seed):
    path = tmp_path / "seed.json"
    _write_seed(path, seed["deals"])
    monkeypatch.setenv("DEALS_SEED_FILE", str(path))
    monkeypatch.setattr(library, "REFRESH_SECONDS", 0.05)
    library.start()
    _write_seed(path, seed["deals"][:3])
    deadline = time.monotonic() + 5
    while len(library.all_deal_ids()) != 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(library.all_deal_ids()) == 3


def test_the_refresh_interval_is_thirty_seconds(library):
    assert library.REFRESH_SECONDS == 30


# ------------------------------------------------------------- the hot path


@pytest.fixture
def app_with_dynamodb_unreachable(monkeypatch):
    """The Deal Scorer app, primed from the seed, with every DynamoDB entry point raising."""
    import boto3

    app = _fresh(_APP, "ncf_app_hot_path")
    app.deal_library.load()
    calls = []

    def refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("DynamoDB was called on the request path")

    monkeypatch.setattr(boto3, "resource", refuse)
    monkeypatch.setattr(boto3, "client", refuse)
    monkeypatch.setattr(app.deal_library, "_table", refuse)
    yield app, calls
    app.deal_library.stop()


def _scripted(app, monkeypatch, high):
    def _score(uid_hash, deal_ids, activate_threshold=0.0, suppress_threshold=0.0):
        return [d for d in deal_ids if d in high], [], "", ""
    monkeypatch.setattr(app, "_score_deals", _score)


def _request(bid_request):
    return RTBRequest(
        id="t", bid_request=bid_request,
        applicable_intents=["ACTIVATE_DEALS", "SUPPRESS_DEALS", "ADJUST_DEAL_FLOOR"],
    )


def test_a_request_is_served_from_the_snapshot_with_dynamodb_unreachable(app_with_dynamodb_unreachable, monkeypatch):
    app, calls = app_with_dynamodb_unreachable
    _scripted(app, monkeypatch, high={"deal-parenting-premium"})
    resp = app.mutate(_request(_parenting()))
    assert [(m.intent, m.path) for m in resp.mutations] == [
        (Intent.ACTIVATE_DEALS, "/imp/imp-1"),
        (Intent.ADJUST_DEAL_FLOOR, "/imp/imp-1/deals/deal-parenting-premium"),
    ]
    assert resp.mutations[1].adjust_deal.bidfloor == 3.40
    assert calls == []


def test_an_app_request_is_served_from_the_snapshot_with_dynamodb_unreachable(app_with_dynamodb_unreachable, monkeypatch):
    app, calls = app_with_dynamodb_unreachable
    _scripted(app, monkeypatch, high={"deal-ridesharer-eats-open-kitchen"})
    resp = app.mutate(_request(_ridesharer_airport()))
    assert [(m.intent, m.path) for m in resp.mutations] == [
        (Intent.ACTIVATE_DEALS, "/imp/imp-1"),
        (Intent.ADJUST_DEAL_FLOOR, "/imp/imp-1/deals/deal-ridesharer-eats-open-kitchen"),
    ]
    assert resp.mutations[0].ids.id == ["deal-ridesharer-eats-open-kitchen"]
    assert resp.mutations[1].adjust_deal.bidfloor == 12.00
    assert calls == []


def test_the_unscripted_model_path_also_answers_with_dynamodb_unreachable(app_with_dynamodb_unreachable):
    app, calls = app_with_dynamodb_unreachable
    resp = app.mutate(_request(_ridesharer_airport()))
    assert resp.id == "t"
    assert calls == []


# ------------------------------------------------------------ the entrypoint


def _env_without_a_source():
    env = dict(os.environ)
    env.pop("DEALS_TABLE", None)
    env.pop("DEALS_SEED_FILE", None)
    return env


def test_the_process_exits_non_zero_before_serving_when_the_prime_fails():
    result = subprocess.run(
        [sys.executable, _APP], env=_env_without_a_source(),
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode != 0
    assert "deal book prime failed from" in result.stderr


def test_the_server_is_never_started_when_the_prime_fails(monkeypatch):
    import shared.server

    started = []
    monkeypatch.setattr(shared.server, "run_artf_server", lambda *a, **k: started.append((a, k)))
    monkeypatch.delenv("DEALS_TABLE", raising=False)
    monkeypatch.delenv("DEALS_SEED_FILE", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(_APP, run_name="__main__")
    assert exit_info.value.code == 1
    assert started == []


def test_the_server_starts_after_a_successful_prime(monkeypatch):
    import shared.server

    started = []
    monkeypatch.setattr(shared.server, "run_artf_server", lambda *a, **k: started.append(k))
    namespace = runpy.run_path(_APP, run_name="__main__")
    try:
        assert started and started[0]["agent_name"] == "ncf-deal-manager"
        assert namespace["deal_library"].all_deal_ids()
    finally:
        namespace["deal_library"].stop()
