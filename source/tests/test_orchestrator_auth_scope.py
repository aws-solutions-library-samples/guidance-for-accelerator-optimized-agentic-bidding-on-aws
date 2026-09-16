"""Authorization on the orchestrator's mutations routes.

Two things are under test, and the first matters more than the second:

1. **A deployment without the Prebid ARTF host is UNAFFECTED.** The orchestrator is an
   existing component. Its only machine caller is the optional Prebid host, so with
   ``ARTF_MUTATIONS_REQUIRED_SCOPE`` unset no route is protected and authorization
   behaves exactly as it did before this feature existed. That is asserted directly,
   because "the new rule is inert by default" is a claim that has to be checked rather
   than described.

2. **When enabled, more than one mechanism can grant a request.** A protected route has
   several legitimate ways in, so the tests that matter pull in opposite directions:

   - a machine token WITHOUT the scope must be refused, or the change does nothing
   - a USER token without the scope must still be allowed, or the frontend is locked
     out of its own UI

   Satisfying either alone is easy and wrong.

The module reads its configuration at import, so these tests reload it under a patched
environment rather than mutating module state -- which also proves the import-time read
does what it claims.
"""

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

MUTATIONS_SCOPE = "artf-orchestrator/mutations:write"
MUTATION_PATHS = ["/v1/mutations", "/api/v1/mutations", "/fabric/v1/mutations"]


@contextmanager
def auth_module(required_scope: str | None):
    """Import orchestrator.auth with ARTF_MUTATIONS_REQUIRED_SCOPE set or unset."""
    previous = os.environ.get("ARTF_MUTATIONS_REQUIRED_SCOPE")
    if required_scope is None:
        os.environ.pop("ARTF_MUTATIONS_REQUIRED_SCOPE", None)
    else:
        os.environ["ARTF_MUTATIONS_REQUIRED_SCOPE"] = required_scope
    try:
        import orchestrator.auth as module

        yield importlib.reload(module)
    finally:
        if previous is None:
            os.environ.pop("ARTF_MUTATIONS_REQUIRED_SCOPE", None)
        else:
            os.environ["ARTF_MUTATIONS_REQUIRED_SCOPE"] = previous
        import orchestrator.auth as module

        importlib.reload(module)


def machine_token(scope: str | None = MUTATIONS_SCOPE) -> dict:
    """A Cognito client_credentials access token: client_id, no username."""
    claims = {"token_use": "access", "client_id": "abc123machine"}
    if scope is not None:
        claims["scope"] = scope
    return claims


def user_token(scope: str = "openid email profile") -> dict:
    """A Cognito user access token from the implicit flow.

    This is the shape the frontend actually sends: authFetch.js attaches
    getAccessToken(), and a user pool access token carries ``username``. The live pool
    for this stack issues exactly these scopes.
    """
    return {
        "token_use": "access",
        "client_id": "frontendclient",
        "username": "demo@example.com",
        "scope": scope,
    }


# =====================================================================
# 1. WITHOUT PREBID: the orchestrator is unchanged
# =====================================================================


@pytest.mark.parametrize("path", MUTATION_PATHS)
@pytest.mark.parametrize(
    "claims",
    [
        machine_token(),
        machine_token(scope="something/else"),
        machine_token(scope=None),
        user_token(),
        {},
    ],
    ids=["machine-with-scope", "machine-wrong-scope", "machine-no-scope", "user", "empty-claims"],
)
def test_with_the_scope_unset_every_caller_is_authorized_exactly_as_before(path, claims):
    # The decisive test for an existing component: no configuration, no new refusals.
    # Any token that authenticated before this feature still reaches the route.
    with auth_module(None) as auth:
        assert auth.authorize(path, claims) == (True, "")


@pytest.mark.parametrize("unset", [None, "", "   ", ",", " , "])
def test_an_absent_or_blank_setting_registers_no_protected_routes(unset):
    # Blank and comma-only values are treated as unset rather than as a scope named ""
    # -- an empty required scope that nothing can satisfy would deny every machine
    # caller while looking configured.
    with auth_module(unset) as auth:
        assert auth._ROUTE_SCOPES == {}
        for path in MUTATION_PATHS:
            assert auth.authorize(path, machine_token(scope=None))[0] is True


def test_with_the_scope_unset_the_granting_mechanism_reports_an_unprotected_route():
    with auth_module(None) as auth:
        assert auth.granting_mechanism("/v1/mutations", machine_token()) == "unprotected_route"


# =====================================================================
# 2. WITH PREBID: several mechanisms, any one may grant
# =====================================================================


@pytest.mark.parametrize("path", MUTATION_PATHS)
def test_a_machine_token_without_the_scope_is_refused(path):
    with auth_module(MUTATIONS_SCOPE) as auth:
        allowed, reason = auth.authorize(path, machine_token(scope="some/other:scope"))

        assert allowed is False
        assert MUTATIONS_SCOPE in reason


@pytest.mark.parametrize("path", MUTATION_PATHS)
def test_a_user_token_without_the_scope_is_still_allowed(path):
    # The frontend calls /api/v1/mutations with exactly this shape. A user token can
    # never carry a resource-server scope, so a rule keyed only on scope would break
    # the UI while appearing to tighten security.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth.authorize(path, user_token()) == (True, "")


@pytest.mark.parametrize("path", MUTATION_PATHS)
def test_a_machine_token_with_the_scope_is_allowed(path):
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth.authorize(path, machine_token()) == (True, "")


def test_the_two_mechanisms_are_reported_separately():
    # Which mechanism granted is recorded, so a machine credential and a signed-in user
    # are distinguishable downstream. Without it a handler cannot treat one differently
    # from the other even when it should.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth.granting_mechanism("/v1/mutations", user_token()) == "user_session"
        assert auth.granting_mechanism("/v1/mutations", machine_token()) == "machine_scope"
        assert auth.granting_mechanism("/v1/mutations", machine_token(scope="no")) is None


def test_a_refusal_names_every_mechanism_that_was_tried():
    # A 403 saying only "scope missing" leaves the reader guessing whether other routes
    # in existed. Naming each mechanism and why it declined is the diagnostic value of
    # modelling them as a list.
    with auth_module(MUTATIONS_SCOPE) as auth:
        _, reason = auth.authorize("/v1/mutations", machine_token(scope="no"))

        assert "user_session" in reason
        assert "machine_scope" in reason


def test_more_than_one_scope_can_satisfy_a_route():
    # A route accepts a SET. A later administrative or read-only scope must be
    # expressible without changing the shape of this code.
    with auth_module(f"{MUTATIONS_SCOPE},artf-orchestrator/admin") as auth:
        assert auth.authorize("/v1/mutations", machine_token())[0] is True
        assert auth.authorize(
            "/v1/mutations", machine_token(scope="artf-orchestrator/admin")
        )[0] is True
        assert auth.authorize("/v1/mutations", machine_token(scope="neither/one"))[0] is False


def test_the_scope_is_matched_exactly_not_by_prefix():
    # Substring matching is the classic way a scope check becomes decorative.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth.authorize("/v1/mutations", machine_token(scope=MUTATIONS_SCOPE + "extra"))[0] is False


def test_one_matching_scope_among_several_on_the_token_is_sufficient():
    with auth_module(MUTATIONS_SCOPE) as auth:
        claims = machine_token(scope=f"other/read {MUTATIONS_SCOPE} third/write")
        assert auth.authorize("/v1/mutations", claims)[0] is True


@pytest.mark.parametrize("scope", [None, "", "   "])
def test_a_machine_token_carrying_no_scope_is_refused_when_enforcement_is_on(scope):
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth.authorize("/v1/mutations", machine_token(scope=scope))[0] is False


def test_the_refusal_reason_does_not_echo_the_granted_scopes():
    # Telling an unauthorized caller what it WAS issued is a disclosure it has no need
    # for.
    with auth_module(MUTATIONS_SCOPE) as auth:
        _, reason = auth.authorize("/v1/mutations", machine_token(scope="secret/internal:admin"))

        assert "secret/internal:admin" not in reason


def test_the_refusal_reason_contains_no_credential_material():
    with auth_module(MUTATIONS_SCOPE) as auth:
        claims = machine_token(scope="none")
        claims["client_secret"] = "should-never-appear"
        _, reason = auth.authorize("/v1/mutations", claims)

        assert "should-never-appear" not in reason


# =====================================================================
# Other routes are untouched either way
# =====================================================================


@pytest.mark.parametrize(
    "path",
    ["/v1/containers", "/api/v1/containers", "/v1/loadtest/start", "/governance/status", "/"],
)
@pytest.mark.parametrize("required", [None, MUTATIONS_SCOPE], ids=["without-prebid", "with-prebid"])
def test_routes_without_a_declared_requirement_are_unaffected(path, required):
    # Enforcement is per-route by design. A blanket rule would have changed every
    # endpoint's behaviour at once, which is not a change this feature is scoped to
    # make.
    with auth_module(required) as auth:
        assert auth.authorize(path, machine_token(scope="nothing/useful"))[0] is True
        assert auth.authorize(path, user_token())[0] is True


def test_all_three_mutation_aliases_are_covered_when_enabled():
    # app.py registers the mutations handler three times. Enforcing on one would leave
    # two unguarded doors into the same handler.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert set(auth._ROUTE_SCOPES) == set(MUTATION_PATHS)
        for path in MUTATION_PATHS:
            assert auth._ROUTE_SCOPES[path] == frozenset({MUTATIONS_SCOPE})


# =====================================================================
# Principal recognition and scope parsing
# =====================================================================


def test_an_id_token_is_a_user_token():
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._is_machine_token({"token_use": "id", "cognito:username": "someone"}) is False


def test_an_access_token_with_a_username_is_a_user_token():
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._is_machine_token({"token_use": "access", "username": "someone"}) is False


def test_an_access_token_without_a_username_is_a_machine_token():
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._is_machine_token({"token_use": "access", "client_id": "abc"}) is True


def test_an_unrecognisable_token_shape_is_treated_as_a_machine_token():
    # Fails SAFE when enforcement is on: an unfamiliar shape is held to the stricter
    # rule rather than waved through, so a future token type cannot quietly bypass it.
    # Note this is only consequential with enforcement enabled -- the first test group
    # covers the unset case.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._is_machine_token({}) is True


def test_scopes_parse_from_a_space_delimited_string():
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._granted_scopes({"scope": "a/b c/d"}) == {"a/b", "c/d"}


def test_scopes_parse_from_a_list_for_non_cognito_issuers():
    # Reading nothing off a list would deny a correctly-scoped caller.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._granted_scopes({"scope": ["a/b", "c/d"]}) == {"a/b", "c/d"}


def test_the_scp_claim_is_also_read():
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._granted_scopes({"scp": "a/b"}) == {"a/b"}


@pytest.mark.parametrize("claims", [{}, {"scope": None}, {"scope": ""}, {"scope": 42}])
def test_absent_or_unusable_scope_claims_yield_no_scopes_rather_than_throwing(claims):
    # A malformed claim must not raise inside auth middleware: an exception there would
    # surface as a 500 on every request rather than as a clean refusal.
    with auth_module(MUTATIONS_SCOPE) as auth:
        assert auth._granted_scopes(claims) == set()
