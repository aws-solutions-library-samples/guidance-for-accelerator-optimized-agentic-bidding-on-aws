"""Deployment mode reported by the orchestrator (standard or learner).

deploy.sh renders DEPLOYMENT_MODE=learner into the orchestrator Deployment when
it is run with --learner-mode, and leaves the variable out otherwise. The value
is surfaced on the authenticated /v1/gpu/status endpoint so the UI can show the
learner-mode notice and disable the Governance tab, and it gates the endpoints
that need GPUs.

Learner mode serves every Triton model on CPU nodes. It is a training and
workshop configuration: to remain ARTF-compliant (IAB Tech Lab Agentic Real
Time Framework), production workloads must not use it.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

MODE_ENV = "DEPLOYMENT_MODE"

_warned: set[str] = set()

STANDARD = "standard"
LEARNER = "learner"
UNKNOWN = "unknown"


def current_mode() -> str:
    """Return "learner", "standard", or "unknown".

    Unset or empty means standard: that is what deploy.sh renders for a normal
    deploy. Any other value is reported as unknown rather than guessed, so a
    typo in the manifest cannot read as either mode.
    """
    raw = os.environ.get(MODE_ENV, "")
    value = raw.strip()
    if value == "":
        return STANDARD
    if value == LEARNER:
        return LEARNER
    if raw not in _warned:
        _warned.add(raw)
        logger.warning("%s=%r is not a known deployment mode; reporting unknown", MODE_ENV, raw)
    return UNKNOWN


def is_learner() -> bool:
    return current_mode() == LEARNER


def learner_unavailable(feature: str) -> JSONResponse:
    """409 for an endpoint that needs GPUs, in learner mode."""
    return JSONResponse(
        {
            "error": f"{feature} is unavailable in learner mode: it requires GPUs.",
            "deploymentMode": LEARNER,
        },
        status_code=409,
    )


Handler = Callable[[Request], Awaitable[JSONResponse]]


def require_standard_mode(feature: str) -> Callable[[Handler], Handler]:
    """Wrap a route handler so it returns 409 in learner mode.

    Only an exact learner value blocks. An unknown value does not, because the
    endpoint then fails or succeeds on its own merits instead of on a guess.
    """

    def decorate(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(request: Request) -> JSONResponse:
            if is_learner():
                return learner_unavailable(feature)
            return await handler(request)

        return wrapped

    return decorate
