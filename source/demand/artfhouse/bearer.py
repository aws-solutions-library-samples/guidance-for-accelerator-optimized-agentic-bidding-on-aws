"""Bearer-token check for the demand service.

The service checks the Cognito access token itself; nothing in front of it does.
The checks match what the former API Gateway JWT authorizer enforced:

  * RS256 signature against the user pool's JWKS
  * ``iss`` is the pool, ``exp`` is in the future
  * ``token_use`` is ``access`` (an id token is refused)
  * ``client_id`` is the Prebid host's app client (the authorizer's audience)
  * ``scope`` includes ``artf-demand/bid:read``

The JWKS is fetched at startup and then hourly by a background thread, so a bid
makes no network call: verification is CPU only. Until the first fetch succeeds,
``ready()`` is False and every request is refused with 503.

Configured by environment:
  COGNITO_REGION, COGNITO_USER_POOL_ID   the pool that issues the tokens
  DEMAND_ALLOWED_CLIENT_ID               the Prebid host's app client id
  DEMAND_REQUIRED_SCOPE                  defaults to artf-demand/bid:read
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Callable, Optional
from urllib.request import urlopen

logger = logging.getLogger("artfhouse.demand.bearer")

DEFAULT_SCOPE = "artf-demand/bid:read"
JWKS_REFRESH_SECONDS = 3600
_JWKS_TIMEOUT_S = 5


def _fetch_jwks(url: str) -> dict:
    with urlopen(url, timeout=_JWKS_TIMEOUT_S) as resp:  # noqa: S310  # nosec B310  # nosemgrep: dynamic-urllib-use-detected  fixed https JWKS URL from deploy-time config
        return json.loads(resp.read())


class TokenVerifier:
    def __init__(
        self,
        region: str,
        pool_id: str,
        client_id: str,
        scope: str = DEFAULT_SCOPE,
        fetch: Callable[[str], dict] = _fetch_jwks,
    ) -> None:
        if not (region and pool_id and client_id):
            raise ValueError(
                "COGNITO_REGION, COGNITO_USER_POOL_ID and DEMAND_ALLOWED_CLIENT_ID are required"
            )
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"
        self.jwks_url = f"{self.issuer}/.well-known/jwks.json"
        self.client_id = client_id
        self.scope = scope
        self._fetch = fetch
        self._jwks: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_env(cls) -> "TokenVerifier":
        return cls(
            os.environ.get("COGNITO_REGION", "").strip(),
            os.environ.get("COGNITO_USER_POOL_ID", "").strip(),
            os.environ.get("DEMAND_ALLOWED_CLIENT_ID", "").strip(),
            os.environ.get("DEMAND_REQUIRED_SCOPE", "").strip() or DEFAULT_SCOPE,
        )

    def ready(self) -> bool:
        return self._jwks is not None

    def refresh_keys(self) -> bool:
        try:
            self._jwks = self._fetch(self.jwks_url)
            return True
        except Exception as exc:  # noqa: BLE001 - keep the last keys
            logger.warning(json.dumps({
                "event": "jwks_refresh_failed", "error": f"{type(exc).__name__}: {exc}",
                "have_keys": self._jwks is not None,
            }))
            return False

    def start(self) -> None:
        self.refresh_keys()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()

            def loop() -> None:
                # Retry quickly until the first fetch succeeds, then hourly.
                while not self._stop.wait(JWKS_REFRESH_SECONDS if self.ready() else 5):
                    self.refresh_keys()

            self._thread = threading.Thread(target=loop, name="jwks-refresh", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def verify(self, authorization: Optional[str]) -> tuple[bool, str]:
        """(True, "") for an acceptable token, else (False, reason). Never raises."""
        jwks = self._jwks
        if jwks is None:
            return False, "signing keys not loaded"
        if not authorization or not authorization.startswith("Bearer "):
            return False, "missing bearer token"
        token = authorization[len("Bearer "):].strip()
        try:
            from jose import jwt as jose_jwt

            claims: dict[str, Any] = jose_jwt.decode(
                token, jwks, algorithms=["RS256"], issuer=self.issuer,
                options={"verify_aud": False},
            )
        except Exception as exc:  # noqa: BLE001 - any decode failure is a refusal
            return False, f"token rejected: {type(exc).__name__}"
        if claims.get("token_use") != "access":
            return False, "not an access token"
        if claims.get("client_id") != self.client_id:
            return False, "token issued to another client"
        if self.scope not in str(claims.get("scope", "")).split():
            return False, f"missing scope {self.scope}"
        return True, ""
