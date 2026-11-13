"""Per-hop timing for the ARTF request path.

Both the orchestrator and the containers report where a request's time went as
``metadata.timing`` on their RTBResponse (see ``Metadata.timing`` in
``artf_types.py``). The segment names are fixed here so the two sides and the
load test agree on them.

Container segments (shared/server.py)::

    parse    body received -> RTBRequest validated
    queue    handler -> mutate() starts on the worker thread (executor wait)
    mutate   mutate() wall clock, including triton
    triton   time inside tritonclient infer() calls, summed over the request
    build    RTBResponse -> JSON bytes
    total    handler entry -> response ready

Orchestrator segments (orchestrator/app.py)::

    auth     Cognito middleware: token verification + route authorization
    parse    body received -> RTBRequest validated + re-serialised
    fan_out  container calls (asyncio.gather) wall clock
    merge    conflict resolution + response build
    emit     emit_bid_outcome / emit_deal_yield_outcome, as called before return
    total    handler entry -> response ready (excludes auth, which runs before)

The Triton figure is collected through a ContextVar holding a MUTABLE
accumulator. ``shared/server.py`` runs ``mutate()`` on a worker thread under
``contextvars.copy_context().run``; a value SET inside that copy would not be
visible to the handler afterwards, but a list appended to in place is.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Iterator

CONTAINER_SEGMENTS = ("parse", "queue", "mutate", "triton", "build", "total")
ORCHESTRATOR_SEGMENTS = ("auth", "parse", "fan_out", "merge", "emit", "total")

# Holds a list of per-call Triton durations (seconds) for the current request,
# or None outside a timed request.
_triton_calls: contextvars.ContextVar[list[float] | None] = contextvars.ContextVar(
    "artf_triton_calls", default=None
)


def start_triton_accumulator() -> list[float]:
    """Begin collecting Triton call durations for the current context.

    Returns the accumulator so the caller can read it after the offloaded
    mutate() returns; the list object is shared with the copied context.
    """
    calls: list[float] = []
    _triton_calls.set(calls)
    return calls


def clear_triton_accumulator() -> None:
    _triton_calls.set(None)


@contextmanager
def triton_call() -> Iterator[None]:
    """Wrap one ``client.infer(...)`` so its wall clock is attributed to ``triton``.

    A no-op outside a timed request (accumulator unset), so inference code paths
    exercised from tests or scripts pay nothing.
    """
    calls = _triton_calls.get()
    if calls is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        calls.append(time.perf_counter() - start)


def triton_ms(calls: list[float] | None) -> float:
    return round(sum(calls or ()) * 1000.0, 3)


def ms(start: float, end: float | None = None) -> float:
    """Milliseconds between two ``time.perf_counter()`` readings, 3 dp."""
    end = time.perf_counter() if end is None else end
    return round((end - start) * 1000.0, 3)
