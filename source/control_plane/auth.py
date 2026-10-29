"""Bearer token verification for the control plane. Fails CLOSED.

Ported from the embedded DSP guidance's token verifier so the two control planes
refuse the same things for the same reasons. Built on ``cryptography``'s RSA
primitive rather than a JWT library: the one code path where being wrong means an
unauthenticated caller can write to the deals table should not rest on a dependency
that is not otherwise in this image, and every refusal below is exercised by a test
that signs real tokens with a generated key.

THE KNOWN PITFALLS, EACH REFUSED EXPLICITLY

  - ``alg: none``                 a token with no signature must never verify
  - algorithm confusion           an HS256 token must not be verified with an RSA key as
                                  though the public key were an HMAC secret
  - no ``kid``, or an unknown one there is then no key to verify against, and picking any
                                  key from the set would accept a token signed with the
                                  wrong one
  - valid signature, wrong issuer another pool's token is still a real signature
  - valid token, missing scope    authentication is not authorisation

Anything not explicitly accepted is refused. There is no branch that returns success on
an error.

AgentCore's inbound JWT authorizer verifies the token before this container is reached.
This module runs anyway, on every call: a component whose authorisation depends entirely
on its deployment configuration is one misconfiguration away from being open.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

#: The only algorithm accepted. Cognito signs access tokens with RS256, and accepting a
#: second algorithm is how confusion attacks become possible.
ALLOWED_ALGORITHM = "RS256"

#: Small tolerance for clock skew between the token issuer and this runtime.
LEEWAY_SECONDS = 60


class AuthOutcome(str, Enum):
    """Why a token was accepted or refused. Closed, and each is logged distinctly."""

    VALID = "valid"
    MISSING = "missing"
    MALFORMED = "malformed"
    UNSUPPORTED_ALGORITHM = "unsupported_algorithm"
    UNKNOWN_KEY = "unknown_key"
    BAD_SIGNATURE = "bad_signature"
    EXPIRED = "expired"
    WRONG_ISSUER = "wrong_issuer"
    MISSING_SCOPE = "missing_scope"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"


@dataclass(frozen=True)
class AuthResult:
    outcome: AuthOutcome
    claims: Optional[dict] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.outcome is AuthOutcome.VALID


# The crypto backend is resolved ONCE, at import, and its absence is a startup fact
# rather than a per-request surprise. An ImportError surfacing from inside a verify call
# would otherwise be indistinguishable from a bad signature.
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    SIGNATURE_VERIFIER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by deployment, not by tests
    SIGNATURE_VERIFIER_AVAILABLE = False


def _b64url(segment: str) -> bytes:
    padding_needed = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding_needed)


def _int_from_b64url(segment: str) -> int:
    return int.from_bytes(_b64url(segment), "big")


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    """The token from an Authorization header, or None.

    The scheme is compared case-insensitively because HTTP schemes are, but nothing else
    about the header is normalised; a header this does not recognise yields None and the
    request is refused.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def verify(
    token: Optional[str],
    *,
    jwks: Mapping[str, Any],
    issuer: str,
    required_scopes: Sequence[str] = (),
    now: Optional[float] = None,
) -> AuthResult:
    """Verify a Cognito access token.

    ``jwks`` is passed in rather than fetched here, so this function performs no I/O and a
    test can supply a key set it controls. ``required_scopes``: when non-empty, the token
    must carry at least ONE of them.
    """
    if not SIGNATURE_VERIFIER_AVAILABLE:
        # Refuse rather than downgrade. A deployment without a crypto backend must not
        # accept writes from unverified callers.
        return AuthResult(
            AuthOutcome.VERIFIER_UNAVAILABLE,
            detail="no RSA verifier is available, so no token can be verified",
        )

    if not token:
        return AuthResult(AuthOutcome.MISSING, detail="no bearer token")

    parts = token.split(".")
    if len(parts) != 3:
        return AuthResult(AuthOutcome.MALFORMED, detail="not three segments")

    try:
        header = json.loads(_b64url(parts[0]))
        claims = json.loads(_b64url(parts[1]))
        signature = _b64url(parts[2])
    except (ValueError, TypeError, json.JSONDecodeError):
        return AuthResult(AuthOutcome.MALFORMED, detail="segment is not valid base64 JSON")

    if not isinstance(header, dict) or not isinstance(claims, dict):
        return AuthResult(AuthOutcome.MALFORMED, detail="header or claims is not an object")

    # Checked BEFORE any key lookup. "none" and HS256 both die here, which is what makes
    # algorithm confusion impossible rather than merely unlikely.
    if header.get("alg") != ALLOWED_ALGORITHM:
        return AuthResult(
            AuthOutcome.UNSUPPORTED_ALGORITHM,
            detail=f"alg {header.get('alg')!r} is not {ALLOWED_ALGORITHM}",
        )

    kid = header.get("kid")
    if not kid:
        return AuthResult(AuthOutcome.UNKNOWN_KEY, detail="header carries no kid")

    key = None
    for candidate in jwks.get("keys") or []:
        if isinstance(candidate, dict) and candidate.get("kid") == kid:
            key = candidate
            break

    if key is None:
        # NOT "try every key". Accepting a token that verifies against any key in the set
        # would accept one signed by a key the issuer has since rotated out.
        return AuthResult(AuthOutcome.UNKNOWN_KEY, detail=f"no key for kid {kid!r}")

    try:
        public_numbers = rsa.RSAPublicNumbers(
            e=_int_from_b64url(key["e"]), n=_int_from_b64url(key["n"])
        )
        public_key = public_numbers.public_key()
        public_key.verify(
            signature,
            f"{parts[0]}.{parts[1]}".encode("ascii"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except KeyError:
        return AuthResult(AuthOutcome.UNKNOWN_KEY, detail="key is missing n or e")
    except Exception as exc:  # noqa: BLE001 - any verification failure is a refusal
        return AuthResult(AuthOutcome.BAD_SIGNATURE, detail=str(exc) or "signature invalid")

    # Only now are the claims trusted enough to read.
    moment = time.time() if now is None else now

    expiry = claims.get("exp")
    if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
        return AuthResult(AuthOutcome.EXPIRED, detail="no usable exp claim")
    if moment > float(expiry) + LEEWAY_SECONDS:
        return AuthResult(AuthOutcome.EXPIRED, detail="token expired")

    if claims.get("iss") != issuer:
        return AuthResult(
            AuthOutcome.WRONG_ISSUER,
            detail=f"iss {claims.get('iss')!r} is not {issuer!r}",
        )

    if required_scopes and not any(has_scope(claims, s) for s in required_scopes):
        # Authentication is not authorisation. A valid token for a different capability
        # must not open this one.
        return AuthResult(
            AuthOutcome.MISSING_SCOPE,
            detail=f"token carries none of the scopes {list(required_scopes)!r}",
        )

    return AuthResult(AuthOutcome.VALID, claims=claims)


def granted_scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    """The scopes a token carries, as whole tokens.

    Cognito puts scopes in a SPACE-DELIMITED string. Split on whitespace and compare whole
    tokens; a substring test would let ``deals:readonly`` satisfy ``deals:read``.
    """
    raw = claims.get("scope")
    if isinstance(raw, str):
        return frozenset(raw.split())
    if isinstance(raw, (list, tuple)):
        return frozenset(str(s) for s in raw)
    return frozenset()


def has_scope(claims: Mapping[str, Any], required: str) -> bool:
    """Whether the token carries exactly this scope."""
    return required in granted_scopes(claims)


class JwksCache:
    """Fetches and caches a key set.

    Separate from ``verify`` so verification stays pure. The fetcher is injected, so a
    test drives this without network access.
    """

    def __init__(
        self,
        fetcher: Callable[[], Mapping[str, Any]],
        *,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._fetcher = fetcher
        self._ttl = ttl_seconds
        self._cached: Optional[Mapping[str, Any]] = None
        self._fetched_at = 0.0

    def get(self, *, now: Optional[float] = None) -> Mapping[str, Any]:
        moment = time.time() if now is None else now
        if self._cached is not None and (moment - self._fetched_at) < self._ttl:
            return self._cached
        fetched = self._fetcher()
        # Only replace a good cache with a good result. A fetch that returns nothing
        # usable must not blank a working key set and start refusing every request.
        if fetched and fetched.get("keys"):
            self._cached = fetched
            self._fetched_at = moment
        elif self._cached is None:
            self._cached = {"keys": []}
            self._fetched_at = moment
        return self._cached


def cognito_jwks_url(region: str, user_pool_id: str) -> str:
    return f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"


def cognito_issuer(region: str, user_pool_id: str) -> str:
    return f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
