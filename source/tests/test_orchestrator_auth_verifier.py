"""The orchestrator refuses tokens it cannot verify.

`auth._verify_token` checks `iss`, `exp` and `token_use` against a JWT decoded WITHOUT
its signature, then verifies the signature with python-jose. It used to carry an
``except ImportError`` clause that returned the **unverified payload** -- so with the
library absent, a self-signed JWT authenticated, and it carried whatever ``username``,
``client_id`` and ``scope`` its author wrote. Those forged claims then flowed into the
authorization mechanisms, which read claims and trust that something else verified them.

The first test here is the regression: it **fails against the code as it was**. Verified
by running it against HEAD before the fix, where the forged token below was accepted
with ``username="attacker"`` and the mutations scope, and ``authorize()`` granted it.

Two design choices are also pinned, because both are easy to undo by accident:

- **503, not 401.** A missing verifier is a server misconfiguration, so it gets the same
  answer as a missing user pool id. A 401 would tell every caller its token was invalid
  and send it to re-authenticate against a fault no credential can fix.
- **Health probes and AUTH_DISABLED are evaluated first**, so a verifier outage does not
  take the readiness probe down with it.

The last test asserts the dependency pin. That is not padding: fail-closed converts a
silent security downgrade into a total outage, which is the right trade only if
something notices the dependency leaving before production does.
"""

import base64
import json
import re
import sys
import time
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from orchestrator import auth  # noqa: E402

REGION = "us-east-1"
POOL_ID = "us-east-1_TESTPOOL"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL_ID}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(payload: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def forged_token(**extra_claims) -> str:
    """A JWT that passes every unverified check and has a garbage signature.

    Correct issuer, an hour of validity, ``token_use: access`` -- all three are fields
    the author simply writes. That is the whole point: they cost an attacker nothing, so
    a check that consults only them is not a check.
    """
    claims = {
        "iss": ISSUER,
        "exp": int(time.time()) + 3600,
        "token_use": "access",
        "username": "attacker",
        "scope": "artf-orchestrator/mutations:write",
        "client_id": "forged-client",
    }
    claims.update(extra_claims)
    return ".".join([
        _b64({"alg": "RS256", "kid": "no-such-key"}),
        _b64(claims),
        base64.urlsafe_b64encode(b"not-a-signature").decode().rstrip("="),
    ])


@pytest.fixture
def verifier_absent(monkeypatch):
    """Force the unavailable state, whether or not python-jose is installed here."""
    monkeypatch.setattr(auth, "_SIGNATURE_VERIFIER_AVAILABLE", False)
    monkeypatch.setattr(auth, "_jose_jwt", None)


class _StubJoseJwt:
    """Stands in for ``jose.jwt``, so the available path is testable without the library.

    Records the arguments it was called with, because "verification happened" and
    "verification happened against the right issuer" are different claims.
    """

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls: list[dict] = []

    def decode(self, token, key, **kwargs):
        self.calls.append({"token": token, "key": key, **kwargs})
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture
def verifier_present(monkeypatch):
    """Install a stub verifier and a stub JWKS fetch; return a factory."""
    def _install(result=None, raises=None) -> _StubJoseJwt:
        stub = _StubJoseJwt(result=result, raises=raises)
        monkeypatch.setattr(auth, "_SIGNATURE_VERIFIER_AVAILABLE", True)
        monkeypatch.setattr(auth, "_jose_jwt", stub)
        monkeypatch.setattr(auth, "_get_jwks", lambda region, pool: {"keys": []})
        return stub

    return _install


def build_client(monkeypatch, **env) -> TestClient:
    """An app with only the auth middleware, and one route that echoes the claims."""
    monkeypatch.setenv("COGNITO_USER_POOL_ID", env.pop("pool_id", POOL_ID))
    monkeypatch.setenv("COGNITO_REGION", REGION)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    async def echo(request):
        return JSONResponse({
            "username": getattr(request.state, "user", {}).get("username"),
            "mechanism": getattr(request.state, "auth_mechanism", None),
        })

    async def ready(request):
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[
        Route("/v1/mutations", echo, methods=["POST", "GET"]),
        Route("/health/ready", ready),
    ])
    app.add_middleware(auth.CognitoAuthMiddleware)
    return TestClient(app)


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_forged_token_is_refused_when_verifier_unavailable(verifier_absent):
    """THE regression test. Fails against the code before this unit.

    Before: returned the decoded payload, so this token authenticated as "attacker"
    holding the mutations scope.
    """
    assert auth._verify_token(forged_token(), REGION, POOL_ID) is None


def test_forged_claims_would_satisfy_authorization_if_they_got_through(verifier_absent):
    """Why the refusal matters, shown rather than asserted in prose.

    Authorization reads claims and trusts that verification already happened. So the
    forged claim set is not merely accepted -- it is *authorized*: the user_session
    mechanism grants it on the strength of a ``username`` its author typed. The only
    thing standing between the forged token and the route is `_verify_token` refusing to
    produce these claims at all, which is the previous test.
    """
    forged_claims = {
        "token_use": "access",
        "username": "attacker",
        "scope": "artf-orchestrator/mutations:write",
    }
    granted, _ = auth._mechanism_user_session("/v1/mutations", forged_claims)
    assert granted, "authorization trusts these claims -- verification is the only gate"
    assert auth._granted_scopes(forged_claims) == {"artf-orchestrator/mutations:write"}

    # And the gate holds.
    assert auth._verify_token(forged_token(**forged_claims), REGION, POOL_ID) is None


def test_unverified_payload_is_never_returned(verifier_absent):
    """Not merely "falsy" -- specifically not the payload.

    A future refactor returning ``payload`` for "diagnostics" would reintroduce the
    exact defect, so the shape of the return is pinned, not just its truthiness.
    """
    result = auth._verify_token(forged_token(username="someone"), REGION, POOL_ID)
    assert result is None
    assert not isinstance(result, dict)


def test_wellformed_but_unsigned_token_refused_even_with_valid_pool(verifier_absent):
    """No combination of correct metadata rescues an unverifiable token."""
    token = forged_token(exp=int(time.time()) + 86400, token_use="id")
    assert auth._verify_token(token, REGION, POOL_ID) is None


# ---------------------------------------------------------------------------
# The status code, and where the guard sits
# ---------------------------------------------------------------------------

def test_middleware_returns_503_not_401(monkeypatch, verifier_absent):
    """503: the server cannot authenticate anyone. 401 would blame the caller."""
    client = build_client(monkeypatch)
    response = client.post("/v1/mutations", headers={"Authorization": f"Bearer {forged_token()}"})
    assert response.status_code == 503
    assert response.status_code != 401


def test_503_body_matches_the_existing_misconfiguration_message(monkeypatch, verifier_absent):
    """Same class of fault as a missing pool id, so the same message. No new convention."""
    client = build_client(monkeypatch)
    response = client.post("/v1/mutations", headers={"Authorization": f"Bearer {forged_token()}"})
    assert response.json() == {"error": "Authentication is not configured"}


def test_503_body_does_not_disclose_the_missing_dependency(monkeypatch, verifier_absent):
    """An unauthenticated caller learns nothing about the server's dependency state."""
    client = build_client(monkeypatch)
    response = client.post("/v1/mutations", headers={"Authorization": f"Bearer {forged_token()}"})
    body = response.text.lower()
    for leak in ("jose", "python-jose", "import", "signature", "cryptography"):
        assert leak not in body, f"response body discloses {leak!r}"


def test_missing_pool_id_still_wins_and_still_returns_503(monkeypatch, verifier_absent):
    """The pre-existing guard is unchanged and still first of the two."""
    client = build_client(monkeypatch, pool_id="")
    response = client.post("/v1/mutations", headers={"Authorization": f"Bearer {forged_token()}"})
    assert response.status_code == 503


def test_no_token_still_returns_401_when_verifier_available(monkeypatch, verifier_present):
    """A caller with no credential is still told so -- 503 has not swallowed the 401."""
    verifier_present(result={"token_use": "access", "username": "u"})
    client = build_client(monkeypatch)
    assert client.post("/v1/mutations").status_code == 401


def test_health_probe_survives_a_verifier_outage(monkeypatch, verifier_absent):
    """The guard sits AFTER the public-path check, so readiness stays green.

    Position matters more than the guard: placed a few lines earlier it would fail the
    load balancer's probe and take the pod out of service for a fault that does not stop
    it serving health.
    """
    client = build_client(monkeypatch)
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_auth_disabled_still_bypasses_when_verifier_unavailable(monkeypatch, verifier_absent):
    """The sanctioned local-dev escape hatch still works.

    This is what makes the fix acceptable off-container: developers without python-jose
    set AUTH_DISABLED=true deliberately, rather than being silently handed an
    accept-anything verifier.
    """
    client = build_client(monkeypatch, AUTH_DISABLED="true")
    response = client.post("/v1/mutations", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 200


def test_options_preflight_survives_a_verifier_outage(monkeypatch, verifier_absent):
    """CORS preflight carries no credential and exposes nothing; unchanged."""
    client = build_client(monkeypatch)
    response = client.options("/v1/mutations")
    assert response.status_code != 503


# ---------------------------------------------------------------------------
# Unchanged when the verifier IS available -- which is every deployed pod
# ---------------------------------------------------------------------------

def test_valid_token_accepted_unchanged(monkeypatch, verifier_present):
    """The deployed path. Nothing about it changed, and that is the claim under test."""
    verifier_present(result={"token_use": "access", "username": "real-user", "sub": "abc"})
    client = build_client(monkeypatch)
    response = client.post("/v1/mutations", headers={"Authorization": f"Bearer {forged_token()}"})
    assert response.status_code == 200
    assert response.json()["username"] == "real-user"


def test_verifier_is_called_with_the_expected_issuer(verifier_present):
    """"It verified" is weaker than "it verified against this pool"."""
    stub = verifier_present(result={"token_use": "access", "username": "u"})
    auth._verify_token(forged_token(), REGION, POOL_ID)
    assert len(stub.calls) == 1
    assert stub.calls[0]["issuer"] == ISSUER
    assert stub.calls[0]["algorithms"] == ["RS256"]


def test_claims_come_from_the_verifier_not_the_unverified_payload(verifier_present):
    """The returned claims are the VERIFIER's, so a forged claim cannot survive.

    The token says ``username: attacker``; the verifier says ``verified-user``. If the
    function ever returned the payload again, this test names the difference.
    """
    verifier_present(result={"token_use": "access", "username": "verified-user"})
    claims = auth._verify_token(forged_token(username="attacker"), REGION, POOL_ID)
    assert claims["username"] == "verified-user"


# ---------------------------------------------------------------------------
# The second way in: ImportError from inside decode()
# ---------------------------------------------------------------------------

def test_importerror_from_inside_decode_is_refused(verifier_present):
    """python-jose present, crypto backend missing -- a real installation state.

    The old clause wrapped the decode call as well as the import, so this raised
    ``ImportError``, was read as "library absent", and took the accept-anything path.
    Probing the import separately is what makes this a refusal, and this test is why
    the probe is at module scope rather than inline.
    """
    verifier_present(raises=ImportError("No module named 'cryptography.hazmat'"))
    assert auth._verify_token(forged_token(), REGION, POOL_ID) is None


def test_signature_failure_is_refused(verifier_present):
    """The ordinary case, kept so the broad except is not vacuously passing."""
    verifier_present(raises=ValueError("Signature verification failed."))
    assert auth._verify_token(forged_token(), REGION, POOL_ID) is None


def test_jwks_fetch_failure_is_refused(monkeypatch, verifier_present):
    """A JWKS outage refuses rather than falling back. Fail closed both ways."""
    verifier_present(result={"token_use": "access", "username": "u"})
    monkeypatch.setattr(
        auth, "_get_jwks", lambda region, pool: (_ for _ in ()).throw(OSError("unreachable"))
    )
    assert auth._verify_token(forged_token(), REGION, POOL_ID) is None


# ---------------------------------------------------------------------------
# The pin that keeps fail-closed from becoming an outage
# ---------------------------------------------------------------------------

DOCKERFILES = [
    _SOURCE / "Dockerfile",
    _SOURCE / "Dockerfile.orchestrator",
]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_dockerfile_pins_the_signature_verifier(dockerfile):
    """Both images must install python-jose with a crypto backend.

    This is the counterweight to failing closed. Before this unit, dropping the
    dependency degraded security quietly; now it would refuse every request in
    production. This test is what makes that discoverable in CI instead.

    The ``[cryptography]`` extra is part of the assertion: python-jose without a backend
    raises ``ImportError`` from inside ``decode()``, which is refused -- correct, but a
    total outage all the same.
    """
    assert dockerfile.is_file(), f"{dockerfile} is missing"
    content = dockerfile.read_text()
    assert re.search(r"python-jose\[cryptography\]==3\.3\.0", content), (
        f"{dockerfile.name} no longer pins python-jose[cryptography]==3.3.0. "
        "auth.py refuses every request without it (503), so this is an outage, not a "
        "downgrade. Restore the pin or change auth.py deliberately."
    )


def test_verifier_available_in_a_correctly_provisioned_environment():
    """Documents which state this machine is in, without asserting either.

    python-jose is absent here, which is why the regression above is a real regression
    rather than only a mocked one. Asserting availability would fail locally and pass in
    CI for reasons unrelated to the code, so this records the fact instead.
    """
    assert isinstance(auth._SIGNATURE_VERIFIER_AVAILABLE, bool)
    if not auth._SIGNATURE_VERIFIER_AVAILABLE:
        assert auth._jose_jwt is None
