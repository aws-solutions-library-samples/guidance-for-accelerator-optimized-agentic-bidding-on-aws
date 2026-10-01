"""Every Triton HTTP client must be per-thread, not per-process.

`mutate()` runs on a ThreadPoolExecutor of ARTF_MUTATE_WORKERS threads
(shared/server.py:71), and tritonclient.http's connection pool is greenlet-bound
to the thread that opened it. A process-wide singleton therefore serves exactly
one worker thread; every request landing on another raises

    Cannot switch to a different thread
      Current:  <greenlet.greenlet object at ... current active started main>
      Expected: <greenlet.greenlet object at ... suspended active started main>

which the container surfaces as `inference_unavailable` and no mutation. Observed
2 of 6 identical sequential requests to the bid
shader abstained, while Triton's own counter recorded all of them as successes --
so server-side metrics could not see it.

These tests assert the fix's invariant (distinct thread -> distinct client) rather
than the absence of the error string, because the error only reproduces against a
real greenlet-backed socket.
"""

from __future__ import annotations

import importlib
import sys
import threading
from concurrent import futures
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


# (module path, accessor name) for every module that builds a Triton HTTP client.
# A new Triton-backed container must be added here.
CLIENT_MODULES = [
    ("containers.dlrm_bid_shader.triton_inference", "_get_client"),
    ("containers.ncf_deal_manager.triton_inference", "_get_client"),
    ("shared.fil_inference", "get_client"),
]


class _FakeClient:
    """Stands in for tritonclient's client so no socket is opened."""

    _counter = 0

    def __init__(self, *args, **kwargs):
        type(self)._counter += 1
        self.serial = type(self)._counter


@pytest.fixture(params=CLIENT_MODULES, ids=[m for m, _ in CLIENT_MODULES])
def accessor(request, monkeypatch):
    """Yield a freshly-imported client accessor with the socket stubbed out."""
    module_path, accessor_name = request.param
    module = importlib.import_module(module_path)
    monkeypatch.setattr(module.httpclient, "InferenceServerClient", _FakeClient)
    # Clear any client cached by an earlier test in this process.
    monkeypatch.setattr(module, "_local", threading.local())
    return getattr(module, accessor_name)


def test_two_threads_get_two_different_clients(accessor):
    """The invariant. A shared client is the defect."""
    seen: dict[int, object] = {}

    def grab():
        seen[threading.get_ident()] = accessor()

    threads = [threading.Thread(target=grab) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == 2, "expected two distinct threads to report"
    clients = list(seen.values())
    assert clients[0] is not clients[1], (
        "both threads received the same client object -- tritonclient's pool is "
        "bound to the thread that opened it, so the second thread's inference "
        "will fail with 'Cannot switch to a different thread'"
    )


def test_the_same_thread_reuses_its_client(accessor):
    """Per-thread must not become per-call: a new connection pool on every
    inference would add a TCP handshake to the bid path."""
    first = accessor()
    second = accessor()
    assert first is second


def test_every_worker_in_a_pool_gets_its_own_client(accessor):
    """The live shape: ARTF_MUTATE_WORKERS threads each serving requests."""
    workers = 4
    barrier = threading.Barrier(workers)

    def grab():
        # Force all workers to be live at once so the executor cannot satisfy
        # every task with a single reused thread.
        barrier.wait(timeout=5)
        return id(accessor())

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        client_ids = set(pool.map(lambda _: grab(), range(workers)))

    assert len(client_ids) == workers, (
        f"expected {workers} distinct clients across {workers} worker threads, "
        f"got {len(client_ids)}"
    )


def test_no_module_keeps_a_process_wide_client_singleton():
    """Guards the pattern itself, so a future edit cannot reintroduce a
    module-level `_client` without this failing."""
    offenders = []
    for module_path, _ in CLIENT_MODULES:
        module = importlib.import_module(module_path)
        if hasattr(module, "_client"):
            offenders.append(module_path)
    assert not offenders, (
        f"module-level Triton client singleton found in {offenders} -- use a "
        "threading.local() so each mutate() worker thread owns its own pool"
    )
