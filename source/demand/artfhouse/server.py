"""The artfhouse demand service: an in-cluster HTTP server next to Prebid Server.

Routes:
  POST /openrtb2/auction   the bid endpoint Prebid's artfhouse adapter calls
  GET  /health/live        the process is up
  GET  /health/ready       200 only once the catalog snapshot and the signing keys
                           are loaded; 503 before that

The bid path reads memory only: the catalog snapshot (snapshot.py) and the JWKS
(bearer.py) are loaded at startup and refreshed by background threads. No DynamoDB
or network call happens while a bid is handled.

Run with:  uvicorn demand.artfhouse.server:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import snapshot
from .bearer import TokenVerifier
from .handler import handle

logger = logging.getLogger("artfhouse.demand.server")

_verifier: Optional[TokenVerifier] = None


def set_verifier(verifier: Optional[TokenVerifier]) -> None:
    """Install the token verifier (the lifespan, and tests)."""
    global _verifier
    _verifier = verifier


def _ready() -> tuple[bool, dict]:
    snap = snapshot.current()
    keys = _verifier is not None and _verifier.ready()
    detail = {
        "catalog": (
            {"campaigns": snap.campaigns, "deals": snap.deals,
             "loaded_at": snap.loaded_at.isoformat()}
            if snap else None
        ),
        "signing_keys": keys,
    }
    return snap is not None and keys, detail


async def live(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "live"})


async def ready(_request: Request) -> JSONResponse:
    ok, detail = _ready()
    return JSONResponse({"status": "ready" if ok else "not_ready", **detail},
                        status_code=200 if ok else 503)


async def auction(request: Request) -> JSONResponse:
    if _verifier is None or not _verifier.ready() or not snapshot.ready():
        return JSONResponse({"error": "demand service not ready"}, status_code=503)
    ok, reason = _verifier.verify(request.headers.get("authorization"))
    if not ok:
        logger.info(json.dumps({"event": "unauthorized", "reason": reason}))
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    status, body = handle(snapshot.service(), await request.body())
    return JSONResponse(body, status_code=status)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    if _verifier is None:
        verifier = TokenVerifier.from_env()
        verifier.start()
        set_verifier(verifier)
    snapshot.start()
    yield
    snapshot.stop()
    if _verifier is not None:
        _verifier.stop()


routes = [
    Route("/openrtb2/auction", auction, methods=["POST"]),
    Route("/health/live", live, methods=["GET"]),
    Route("/health/ready", ready, methods=["GET"]),
]

app = Starlette(routes=routes, lifespan=_lifespan)
