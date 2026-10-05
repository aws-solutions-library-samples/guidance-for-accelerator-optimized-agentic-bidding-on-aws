"""Cross-replica lookup for load tests.

A running load test lives in the memory of the orchestrator replica that accepted
the POST (loadtest.py's ``_active_tests`` and ``_progress_*`` dicts). The
orchestrator runs with two or more replicas behind the internal NLB that the UI
API proxy Lambda calls, and the Lambda opens a connection per invoke, so the
poll and the stop request for a test land on either replica. On the wrong one
the test does not exist.

Rather than copy the live counters out of the owner, a replica that does not
know a test asks its siblings. The headless Service ``orchestrator-grpc``
(deployment/eks/orchestrator-deployment.yaml) publishes one A record per ready
replica, which is exactly the address list needed. The sibling receives the
caller's own ``Authorization`` header, so the forwarded request is authorized
the same way the original was, and a marker header stops a forwarded request
from being forwarded again.

Environment:
- ``ORCHESTRATOR_PEER_DNS``  name to resolve for sibling addresses
  (default ``orchestrator-grpc``; empty disables forwarding, e.g. local dev).
- ``ORCHESTRATOR_PEER_PORT`` HTTP port the siblings serve on (default 8000).
- ``POD_IP``                 this replica's address (Downward API), excluded
  from the sibling list.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

NO_FORWARD_HEADER = "x-artf-loadtest-no-forward"

# One in-cluster hop. Siblings answer the status endpoint from memory in
# milliseconds; anything slower is a sibling that is not going to answer.
PEER_TIMEOUT_SECONDS = 2.0


def _peer_dns() -> str:
    return os.environ.get("ORCHESTRATOR_PEER_DNS", "orchestrator-grpc").strip()


def _peer_port() -> int:
    try:
        return int(os.environ.get("ORCHESTRATOR_PEER_PORT", "8000"))
    except ValueError:
        return 8000


def _self_ip() -> str:
    return os.environ.get("POD_IP", "").strip()


async def discover_peers() -> list[str]:
    """Addresses of the other replicas, from the headless Service's A records.

    Returns [] when forwarding is disabled, the name does not resolve, or this
    replica is the only one.
    """
    name = _peer_dns()
    if not name:
        return []
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(name, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except (socket.gaierror, OSError) as exc:
        logger.warning("loadtest peer discovery: %s did not resolve: %s", name, exc)
        return []
    self_ip = _self_ip()
    peers: list[str] = []
    for info in infos:
        ip = info[4][0]
        if ip and ip != self_ip and ip not in peers:
            peers.append(ip)
    return peers


def is_forwarded(request: Request) -> bool:
    """True when this request already came from a sibling; it must not fan out again."""
    return bool(request.headers.get(NO_FORWARD_HEADER))


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {NO_FORWARD_HEADER: "1"}
    auth = request.headers.get("authorization")
    if auth:
        headers["authorization"] = auth
    return headers


async def forward_to_owner(request: Request) -> Response | None:
    """Replay ``request`` (method and path) against every sibling; return the first
    answer that is not a 404.

    Returns None when there is nothing to forward to (no siblings, forwarding
    disabled, or the request is itself a forward), or when every sibling says
    404 or fails to answer. The caller then returns its own 404.
    """
    if is_forwarded(request):
        return None
    peers = await discover_peers()
    if not peers:
        return None

    port = _peer_port()
    path = request.url.path
    headers = _forward_headers(request)

    async with httpx.AsyncClient(timeout=httpx.Timeout(PEER_TIMEOUT_SECONDS)) as client:
        async def ask(ip: str) -> httpx.Response | None:
            try:
                return await client.request(request.method, f"http://{ip}:{port}{path}", headers=headers)
            except httpx.HTTPError as exc:
                logger.warning("loadtest peer %s did not answer %s %s: %s", ip, request.method, path, exc)
                return None

        replies = await asyncio.gather(*(ask(ip) for ip in peers))

    for reply in replies:
        if reply is not None and reply.status_code != 404:
            return Response(
                content=reply.content,
                status_code=reply.status_code,
                media_type=reply.headers.get("content-type", "application/json"),
            )
    return None


async def any_peer_running(request: Request) -> str | None:
    """Return the id of a test running on a sibling, or None.

    Used by the POST guard so two replicas cannot each run a load test at the
    same time. Asks ``GET /v1/loadtest/running`` on every sibling with the
    caller's Authorization header (the siblings enforce the same Cognito auth);
    a sibling that does not answer is treated as not running. The guard protects
    the numbers; it must not block starts because one replica is restarting.
    """
    if is_forwarded(request):
        return None
    peers = await discover_peers()
    if not peers:
        return None
    port = _peer_port()
    headers = _forward_headers(request)
    async with httpx.AsyncClient(timeout=httpx.Timeout(PEER_TIMEOUT_SECONDS)) as client:
        async def ask(ip: str) -> str | None:
            try:
                r = await client.get(f"http://{ip}:{port}/v1/loadtest/running", headers=headers)
            except httpx.HTTPError as exc:
                logger.warning("loadtest peer %s did not answer running check: %s", ip, exc)
                return None
            if r.status_code != 200:
                return None
            try:
                return r.json().get("id") or None
            except ValueError:
                return None

        for found in await asyncio.gather(*(ask(ip) for ip in peers)):
            if found:
                return found
    return None


def json_404() -> JSONResponse:
    return JSONResponse({"error": "Load test not found"}, status_code=404)
