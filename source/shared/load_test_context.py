"""Per-request load-test targeting context.

Carries the out-of-band ``X-Load-Test-Target-Variant`` signal from a
container's request-handling entrypoint (server.py's REST/MCP routes) down
to the container's Triton client call, WITHOUT adding a parameter to the
fixed ``mutate(req: RTBRequest) -> RTBResponse`` contract every container
implements identically.

Only the orchestrator's load-test invocation path
(source/orchestrator/loadtest_targeting.py) ever sets the
``X-Load-Test-Target-Variant`` header this module reads. Real bid-serving
traffic never sends it, so ``get_target_variant()`` is always None on that
path — this is what makes the override structurally load-test-exclusive
(see business-rules.md BR-4 for the Train-from-Load-Test feature).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Literal, Optional

_target_variant: ContextVar[Optional[str]] = ContextVar("target_variant", default=None)

TargetVariant = Literal["stable", "canary"]

HEADER_NAME = "X-Load-Test-Target-Variant"


def get_target_variant() -> Optional[str]:
    """Return the current request's target-variant override, or None.

    None means "no override" — the caller should use its normal (random
    split) routing behavior. Only "stable" or "canary" are ever considered
    valid; any other value observed at the header-reading layer is treated
    as absent (see server.py's header parsing).
    """
    return _target_variant.get()


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
