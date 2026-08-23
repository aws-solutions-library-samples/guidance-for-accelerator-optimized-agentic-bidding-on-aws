"""Per-request load-test targeting context.

Carries two out-of-band signals from a container's request-handling
entrypoint (server.py's REST/MCP routes) down to the container's own
mutate()/Triton-client logic, WITHOUT adding a parameter to the fixed
``mutate(req: RTBRequest) -> RTBResponse`` contract every container
implements identically:

- ``X-Load-Test-Target-Variant`` (get_target_variant()) — forces a specific
  Triton router variant ("stable"/"canary") for THIS run's calls to its
  one targeted container. Only set when a load test explicitly targets a
  challenger variant (see loadtest_targeting.py).
- ``X-Load-Test`` (get_is_load_test()) — a plain "this call originated from
  a load test" signal, sent on EVERY container call a load test makes
  (target or not, and regardless of target_variant). This is the signal
  the Yield Optimizer's bounded exploration (containers/deal_yield_manager/
  exploration.py) gates on: exploration perturbs a real prediction before
  it's used, so it must never fire on live auction traffic, only on
  load-test-originated calls used to bootstrap training data (see
  CLOSED_LOOP.md's "Yield Optimizer" section). get_target_variant() alone
  cannot serve this purpose — it is also None for a load test targeting
  "current" (no override header at all per build_override_headers()), so
  it can't distinguish "live traffic" from "load test not targeting a
  challenger".

Only the orchestrator's load-test invocation path
(source/orchestrator/loadtest.py / loadtest_targeting.py) ever sets either
header. Real bid-serving traffic never sends them, so both getters are
always their "off" default (None / False) on that path — this is what
makes both overrides structurally load-test-exclusive (see
business-rules.md BR-4 for the Train-from-Load-Test feature).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Literal, Optional

_target_variant: ContextVar[Optional[str]] = ContextVar("target_variant", default=None)
_is_load_test: ContextVar[bool] = ContextVar("is_load_test", default=False)

TargetVariant = Literal["stable", "canary"]

HEADER_NAME = "X-Load-Test-Target-Variant"
IS_LOAD_TEST_HEADER_NAME = "X-Load-Test"


def get_target_variant() -> Optional[str]:
    """Return the current request's target-variant override, or None.

    None means "no override" — the caller should use its normal (random
    split) routing behavior. Only "stable" or "canary" are ever considered
    valid; any other value observed at the header-reading layer is treated
    as absent (see server.py's header parsing).
    """
    return _target_variant.get()


def get_is_load_test() -> bool:
    """Return True iff this request originated from the orchestrator's
    load-test invocation path (see module docstring). False (the default)
    on every real bid-serving call — real auction traffic never sends the
    X-Load-Test header, so this can never be True on that path."""
    return _is_load_test.get()


@contextmanager
def target_variant_scope(value: Optional[str]) -> Iterator[None]:
    """Set the target-variant override for the duration of a single request.

    Used by server.py's request handlers to scope the override to exactly
    one ``mutate()`` call, then reset it — so it can never leak into a
    later, unrelated request handled by the same worker thread/task.
    """
    token = _target_variant.set(value)
    try:
        yield
    finally:
        _target_variant.reset(token)


@contextmanager
def load_test_scope(value: bool) -> Iterator[None]:
    """Set the is-load-test flag for the duration of a single request.

    Same per-request scoping contract as target_variant_scope() — set by
    server.py's request handlers around exactly one mutate() call, then
    reset, so it can never leak into a later request handled by the same
    worker thread/task.
    """
    token = _is_load_test.set(value)
    try:
        yield
    finally:
        _is_load_test.reset(token)
