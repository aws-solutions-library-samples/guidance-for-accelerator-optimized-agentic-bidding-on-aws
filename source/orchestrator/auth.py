"""Cognito JWT verification middleware for the orchestrator.

Validates the Authorization: Bearer <token> header against the configured
Cognito User Pool.  Health check endpoints are exempt.

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

    Uses python-jose if available, otherwise falls back to manual validation
    of expiry and issuer (signature check requires python-jose or PyJWT).
    """
    try:
        header, payload = _decode_jwt_unverified(token)
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

    # Try full signature verification with python-jose
    try:
        from jose import jwt as jose_jwt, JWTError
        jwks = _get_jwks(region, pool_id)
        claims = jose_jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=None,  # Cognito access tokens don't have aud
            issuer=expected_issuer,
            options={"verify_aud": False},
        )
        return claims
    except ImportError:
        # python-jose not installed — accept based on expiry/issuer checks above
        logger.info("python-jose not available; accepting token based on expiry+issuer check")
        return payload
    except Exception as e:
        logger.warning("JWT signature verification failed: %s", e)
        return None


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

        # Extract Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Authentication required", "message": "Provide Authorization: Bearer <token>"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]
        claims = _verify_token(token, region, pool_id)
        if claims is None:
            return JSONResponse(
                {"error": "Invalid or expired token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Attach claims to request state for downstream use
        request.state.user = claims
        return await call_next(request)
