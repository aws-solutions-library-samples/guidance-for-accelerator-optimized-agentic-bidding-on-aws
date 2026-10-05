"""Cognito JWT verification middleware for the orchestrator.

Validates the Authorization: Bearer <token> header against the configured
Cognito User Pool.  Health check endpoints are exempt.

Signature verification is mandatory and there is no degraded mode. If the verifier
cannot be imported, every authenticated request is refused with 503 rather than
accepted on the strength of the token's own unverified claims.

Environment variables:
    COGNITO_USER_POOL_ID  — e.g. us-east-1_AbCdEfGhI
    COGNITO_REGION        — e.g. us-east-1
    AUTH_DISABLED         — set to "true" to bypass auth (local dev only)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib.request import urlopen

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("orchestrator.auth")

# Paths that do NOT require authentication
_PUBLIC_PATHS = frozenset({
    "/health/live",
    "/health/ready",
    "/api/health/ready",
    "/fabric/health/ready",
})

# ---------------------------------------------------------------------------
# Authorization: several mechanisms, any one of which may grant a request
#
# Authentication proves the caller holds a token from this pool. That is not
# authorization, and until this section existed it was the whole check: any valid
# pool token reached every route, so a machine client's narrow scope constrained
# nothing and SECURITY-06's least privilege held on paper only.
#
# A PROTECTED ROUTE HAS MORE THAN ONE LEGITIMATE WAY IN, so this is modelled as a
# LIST of independent mechanisms rather than a single rule. A request is authorized
# when ANY applicable mechanism grants it. Two exist today:
#
#   - user_session  a Cognito USER token. The frontend calls /api/v1/mutations with
#                   one, and a user token can never carry a resource-server scope,
#                   so any rule keyed only on scope would lock the UI out of itself.
#   - machine_scope a client_credentials token carrying one of the route's accepted
#                   scopes. This is the Prebid ARTF hook's route in.
#
# The earlier draft of this collapsed the two into one branch -- "machine tokens
# need the scope, everything else passes" -- which made the user path an implicit
# fallthrough rather than a stated decision, and left no room for a third mechanism
# without rewriting the condition. Adding one here is adding one entry to
# _MECHANISMS.
#
# A route accepts a SET of scopes, not one, for the same reason: a later
# administrative or read-only scope should be expressible without changing this
# code's shape.
#
# Order matters only for the diagnostics. Every declining mechanism records why, and
# the refusal lists all of them -- so a 403 says which doors were tried and what each
# wanted, instead of naming one requirement and leaving the reader to guess whether
# others existed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OFF BY DEFAULT.
#
# The only machine caller of the mutations route is the OPTIONAL Prebid ARTF host.
# A deployment without it has no client that could hold a resource-server scope, so
# it must not acquire an authorization rule either -- with
# ARTF_MUTATIONS_REQUIRED_SCOPE empty or unset, _ROUTE_SCOPES is empty, no path is
# protected, and authorize() returns True on its first line. The behaviour is what it
# was before any of this existed.
#
# deploy.sh sets the variable only with --with-prebid; deploy_prebid.sh sets it when
# run standalone and clears it on --destroy.
#
# Read at import, deliberately: an authorization rule that could change under a
# running process would make two identical requests behave differently for reasons
# absent from the request.
# ---------------------------------------------------------------------------

#: Comma-separated; any ONE of them satisfies the machine_scope mechanism. A set
#: rather than a single value so a later administrative or read-only scope needs no
#: change to this code's shape.
_MUTATIONS_SCOPES = frozenset(
    s.strip() for s in os.environ.get("ARTF_MUTATIONS_REQUIRED_SCOPE", "").split(",") if s.strip()
)

#: Route -> the scopes that satisfy the machine_scope mechanism. ANY one suffices.
#:
#: Mutation aliases are listed individually rather than pattern-matched, so adding a
#: route neither silently inherits enforcement nor silently loses it. app.py
#: registers the same handler three times, and enforcing on one would leave two
#: unguarded doors into it.
_ROUTE_SCOPES: dict[str, frozenset[str]] = (
    {
        "/v1/mutations": _MUTATIONS_SCOPES,
        "/api/v1/mutations": _MUTATIONS_SCOPES,
        "/fabric/v1/mutations": _MUTATIONS_SCOPES,
    }
    if _MUTATIONS_SCOPES
    else {}
)

if _MUTATIONS_SCOPES:
    logger.info(
        "Scope authorization ACTIVE on the mutations routes (required: %s)",
        ", ".join(sorted(_MUTATIONS_SCOPES)),
    )
else:
    logger.info(
        "Scope authorization inactive: ARTF_MUTATIONS_REQUIRED_SCOPE is unset, so the "
        "mutations routes authenticate exactly as before"
    )


def _is_machine_token(claims: dict) -> bool:
    """Whether these claims came from a client_credentials grant.

    A Cognito user access token carries ``username``; a client_credentials token
    does not, carrying ``client_id`` alone. An ``id`` token is always a user token.

    Fails safe: anything not positively identifiable as a user token is treated as a
    machine token, so an unfamiliar shape is held to the stricter rule rather than
    waved through.
    """
    if claims.get("token_use") == "id":
        return False
    return not claims.get("username")


def _granted_scopes(claims: dict) -> set[str]:
    """Scopes on the token.

    Cognito sends ``scope`` as a space-delimited string. A list is also accepted
    because other issuers use one, and silently reading nothing off a list would deny
    a correctly-scoped caller.
    """
    raw = claims.get("scope") or claims.get("scp") or ""
    if isinstance(raw, str):
        return {s for s in raw.split() if s}
    if isinstance(raw, (list, tuple)):
        return {str(s) for s in raw if s}
    return set()


def _mechanism_user_session(path: str, claims: dict) -> tuple[bool, str]:
    """A Cognito user token, on a route the UI uses.

    Stated as its own mechanism rather than left as a fallthrough, so that "the
    frontend's user token is authorized here" is a decision visible in one place and
    reviewable on its own.
    """
    if _is_machine_token(claims):
        return False, "not a user session (no username claim; this is a machine credential)"
    return True, ""


def _mechanism_machine_scope(path: str, claims: dict) -> tuple[bool, str]:
    """A machine credential carrying one of the route's accepted scopes."""
    if not _is_machine_token(claims):
        return False, "not a machine credential"

    accepted = _ROUTE_SCOPES.get(path, frozenset())
    if not accepted:
        return False, "no machine scope is declared for this route"

    if _granted_scopes(claims) & accepted:
        return True, ""

    # Names what was REQUIRED, never what the caller holds: echoing granted scopes
    # back tells an unauthorized caller what it was issued.
    wanted = " or ".join(sorted(accepted))
    return False, f"machine credential lacks the required scope ({wanted})"


#: The mechanisms, in the order they are tried. Any one granting is sufficient.
_MECHANISMS: tuple[tuple[str, "object"], ...] = (
    ("user_session", _mechanism_user_session),
    ("machine_scope", _mechanism_machine_scope),
)


def authorize(path: str, claims: dict) -> tuple[bool, str]:
    """Whether these claims may reach this path.

    Pure, so it is testable without a request, a pool or a network.

    Routes with no declared requirement are unprotected by this layer and return
    ``True`` immediately -- enforcement is per-route by design, and a blanket rule
    would have changed every endpoint's behaviour at once.

    :returns: ``(allowed, reason)``. ``reason`` is empty when allowed, and otherwise
        names every mechanism that was tried and why each declined.
    """
    if path not in _ROUTE_SCOPES:
        return True, ""

    declined: list[str] = []
    for name, mechanism in _MECHANISMS:
        allowed, why = mechanism(path, claims)
        if allowed:
            return True, ""
        declined.append(f"{name}: {why}")

    return False, "no authorization mechanism granted this request -- " + "; ".join(declined)


def granting_mechanism(path: str, claims: dict) -> str | None:
    """Which mechanism authorized the request, or None if none did.

    Recorded on the request so a handler and a log line can say HOW a caller was
    authorized, not merely that it was. Two callers with entirely different
    privileges otherwise look identical downstream.
    """
    if path not in _ROUTE_SCOPES:
        return "unprotected_route"
    for name, mechanism in _MECHANISMS:
        if mechanism(path, claims)[0]:
            return name
    return None

# ---------------------------------------------------------------------------
# The signature verifier, probed ONCE at import.
#
# This used to be imported inside _verify_token, inside a try whose `except
# ImportError` returned the UNVERIFIED payload -- so a missing library turned every
# self-signed JWT with a correct `iss` and a future `exp` into a valid credential,
# carrying whatever username and scope its author chose.
#
# Probing here, separately from the call, does three things in order of weight:
#
#   1. It makes availability a STARTUP fact -- logged once below, and assertable in a
#      test -- rather than a per-request surprise.
#   2. It stops an ImportError raised from DEEP INSIDE jwt.decode() from being read as
#      "library absent". python-jose installed with no usable crypto backend is a real
#      state, and the old clause wrapped the decode call as well as the import, so that
#      state took the same accept-anything path. Now it reaches the broad `except
#      Exception` around the verification and is refused.
#   3. It stops re-running an import on every request.
#
# Both images pin python-jose[cryptography]==3.3.0 (source/Dockerfile,
# source/Dockerfile.orchestrator), so this is True in every deployed pod and the
# refusal path below is unreachable there. tests/test_orchestrator_auth_verifier.py
# asserts that pin, because dropping the dependency would now mean 503 rather than a
# quiet downgrade -- fail closed is only safe if something notices the dependency
# leaving.
# ---------------------------------------------------------------------------
try:
    from jose import jwt as _jose_jwt

    _SIGNATURE_VERIFIER_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - depends on the environment
    _jose_jwt = None
    _SIGNATURE_VERIFIER_AVAILABLE = False
    logger.error(
        "JWT signature verification is UNAVAILABLE (%s). python-jose is not importable, "
        "so no token can be verified and every authenticated request will be refused "
        "with 503. Install python-jose[cryptography]; both Dockerfiles pin it, so this "
        "should only ever happen outside the container images.",
        _exc,
    )

# JWKS cache
_jwks_cache: dict[str, Any] | None = None
_jwks_fetched_at: float = 0
_JWKS_TTL = 3600  # re-fetch keys every hour


def _get_jwks(region: str, pool_id: str) -> dict[str, Any]:
    """Fetch and cache the JWKS from Cognito."""
    global _jwks_cache, _jwks_fetched_at
    if _jwks_cache and (time.time() - _jwks_fetched_at) < _JWKS_TTL:
        return _jwks_cache

    url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json"
    with urlopen(url) as resp:
        _jwks_cache = json.loads(resp.read())
    _jwks_fetched_at = time.time()
    return _jwks_cache


def _base64url_decode(s: str) -> bytes:
    """Decode base64url without padding."""
    s += "=" * (4 - len(s) % 4)
    import base64
    return base64.urlsafe_b64decode(s)


def _decode_jwt_unverified(token: str) -> tuple[dict, dict]:
    """Decode JWT header and payload without signature verification (for kid lookup)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    header = json.loads(_base64url_decode(parts[0]))
    payload = json.loads(_base64url_decode(parts[1]))
    return header, payload


def _verify_token(token: str, region: str, pool_id: str) -> dict | None:
    """Verify a Cognito JWT token. Returns claims dict or None if invalid.

    Signature verification is REQUIRED. There is no degraded mode: the issuer, expiry
    and ``token_use`` checks below all read fields the token's author wrote, so on their
    own they establish nothing. When the verifier is unavailable this returns ``None``
    and never the decoded payload -- the middleware refuses such requests earlier with
    a 503, and this branch remains so that a direct caller cannot obtain unverified
    claims from this function either.
    """
    if not _SIGNATURE_VERIFIER_AVAILABLE:
        logger.error("Refusing token: JWT signature verification is unavailable")
        return None

    try:
        _header, payload = _decode_jwt_unverified(token)
    except Exception as e:
        logger.warning("JWT decode failed: %s", e)
        return None

    # Check issuer
    expected_issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"
    if payload.get("iss") != expected_issuer:
        logger.warning("JWT issuer mismatch: %s", payload.get("iss"))
        return None

    # Check expiry
    exp = payload.get("exp", 0)
    if time.time() > exp:
        logger.warning("JWT expired")
        return None

    # Check token_use (accept both access and id tokens)
    token_use = payload.get("token_use", "")
    if token_use not in ("access", "id"):
        logger.warning("JWT token_use invalid: %s", token_use)
        return None

    # Full signature verification. Everything above this line is a cheap pre-filter on
    # self-asserted fields; THIS is the check that decides whether the token is real.
    #
    # NOTE: `aud` is deliberately not verified, because Cognito omits it on access
    # tokens. The consequence is recorded rather than hidden: token_use accepts "id" as
    # well as "access", and Cognito DOES set `aud` to the app client id on id tokens, so
    # an id token minted for a different app client in the SAME pool verifies here.
    # Closing that needs the app client id passed to this process, which is a
    # deployment change, not a change to this function.
    try:
        jwks = _get_jwks(region, pool_id)
        claims = _jose_jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=None,  # Cognito access tokens don't have aud
            issuer=expected_issuer,
            options={"verify_aud": False},
        )
        return claims
    except Exception as e:
        # Deliberately broad, and deliberately WITHOUT an ImportError special case: a
        # missing crypto backend surfacing from inside decode() is a verification
        # failure like any other, and must refuse rather than downgrade.
        logger.warning("JWT signature verification failed: %s", e)
        return None


def grpc_authenticate(authorization: str | None, path: str):
    """Authenticate a gRPC call's ``authorization`` metadata the way the middleware
    authenticates an HTTP request.

    Returns ``(claims, None, "")`` on success, or ``(None, grpc.StatusCode, reason)``
    naming the refusal. Same fail-closed rules as ``CognitoAuthMiddleware.dispatch``:
    no pool or no verifier refuses every call (UNAVAILABLE, the 503 analogue), a
    missing or invalid token is UNAUTHENTICATED (401), a valid token without the
    route's scope is PERMISSION_DENIED (403). AUTH_DISABLED=true bypasses, as on HTTP.
    Imports grpc lazily so this module stays importable where grpc is absent.
    """
    import grpc

    if os.environ.get("AUTH_DISABLED", "").lower() == "true":
        logger.warning("AUTH_DISABLED=true — gRPC authentication bypassed (local dev only)")
        return {}, None, ""
    pool_id = os.environ.get("COGNITO_USER_POOL_ID", "")
    region = os.environ.get("COGNITO_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    if not pool_id:
        logger.error("COGNITO_USER_POOL_ID not set — refusing gRPC call (fail closed)")
        return None, grpc.StatusCode.UNAVAILABLE, "Authentication is not configured"
    if not _SIGNATURE_VERIFIER_AVAILABLE:
        logger.error("JWT signature verification unavailable — refusing gRPC call (fail closed)")
        return None, grpc.StatusCode.UNAVAILABLE, "Authentication is not configured"
    if not authorization or not authorization.startswith("Bearer "):
        return None, grpc.StatusCode.UNAUTHENTICATED, "Authentication required: authorization metadata 'Bearer <token>'"
    claims = _verify_token(authorization[7:], region, pool_id)
    if claims is None:
        return None, grpc.StatusCode.UNAUTHENTICATED, "Invalid or expired token"
    allowed, reason = authorize(path, claims)
    if not allowed:
        logger.warning("Authorization refused for grpc %s: %s (client_id=%s)", path, reason, claims.get("client_id", "unknown"))
        return None, grpc.StatusCode.PERMISSION_DENIED, reason
    return claims, None, ""


class CognitoAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces Cognito JWT auth on all non-health endpoints."""

    async def dispatch(self, request: Request, call_next):
        # Health checks must stay reachable for load balancer probes. They
        # expose no data (just {"status": "ok"}), so they are exempt.
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # CORS preflight carries no credentials and exposes no data.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Explicit local-dev bypass — must be set intentionally, never in prod.
        if os.environ.get("AUTH_DISABLED", "").lower() == "true":
            logger.warning("AUTH_DISABLED=true — authentication bypassed (local dev only)")
            return await call_next(request)

        pool_id = os.environ.get("COGNITO_USER_POOL_ID", "")
        region = os.environ.get("COGNITO_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

        # FAIL CLOSED: without a configured pool we cannot authenticate, so we
        # refuse every non-health request rather than passing it through
        # unauthenticated. Guarantees no unauthenticated access to the
        # orchestrator even if Cognito provisioning failed at deploy time.
        if not pool_id:
            logger.error("COGNITO_USER_POOL_ID not set — refusing request (fail closed)")
            return JSONResponse(
                {"error": "Authentication is not configured"},
                status_code=503,
            )

        # FAIL CLOSED, for the same reason and with the same answer: without the
        # signature verifier no token can be checked, so no request may pass.
        #
        # 503 and not 401, matching the guard above. A 401 would report "Invalid or
        # expired token" for every caller and send them off to re-authenticate against a
        # fault no credential can fix -- the same reasoning that makes the authorization
        # refusal below a 403. This is a server misconfiguration, and it says so.
        #
        # The body is the existing message, unchanged, and does NOT name the missing
        # library: an unauthenticated caller has no business learning the server's
        # dependency state. The operator gets that from the ERROR logged at import.
        if not _SIGNATURE_VERIFIER_AVAILABLE:
            logger.error(
                "JWT signature verification unavailable — refusing request (fail closed)"
            )
            return JSONResponse(
                {"error": "Authentication is not configured"},
                status_code=503,
            )

        # Extract Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Authentication required", "message": "Provide Authorization: Bearer <token>"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]
        # Timed so the request handler can report what authentication cost
        # (metadata.timing.auth); the JWKS fetch on a cold cache lands here too.
        auth_start = time.perf_counter()
        claims = _verify_token(token, region, pool_id)
        if claims is None:
            return JSONResponse(
                {"error": "Invalid or expired token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # A valid token is not the same as an authorized one. 403, not 401: the
        # credential was accepted and simply does not permit this route, and a 401
        # would send the caller off to re-authenticate against a problem that
        # re-authenticating cannot fix.
        allowed, reason = authorize(request.url.path, claims)
        if not allowed:
            logger.warning(
                "Authorization refused for %s: %s (client_id=%s)",
                request.url.path,
                reason,
                claims.get("client_id", "unknown"),
            )
            return JSONResponse(
                {"error": "Not authorized for this route", "message": reason},
                status_code=403,
            )

        # Attach claims to request state for downstream use
        request.state.user = claims
        # And HOW the caller was authorized. Without this, a machine credential and a
        # signed-in user are indistinguishable downstream, so a handler cannot make a
        # different decision for one than for the other even when it should.
        request.state.auth_mechanism = granting_mechanism(request.url.path, claims)
        request.state.auth_ms = round((time.perf_counter() - auth_start) * 1000.0, 3)
        return await call_next(request)
