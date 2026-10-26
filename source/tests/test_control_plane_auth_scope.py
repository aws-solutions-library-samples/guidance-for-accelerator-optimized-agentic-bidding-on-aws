"""Token verification and entitlement for the control plane.

Every refusal in control_plane/auth.py is exercised with a REAL signature from a
generated RSA key, so the tests prove what the verifier refuses, not what it is
documented to refuse. Entitlement parsing is checked by example and by property
(Hypothesis): a malformed grant never widens to a wildcard, and format -> parse is
a round trip.
"""
from __future__ import annotations

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from hypothesis import given, settings
from hypothesis import strategies as st

from control_plane import auth
from control_plane.scope import (
    WILDCARD,
    Entitlement,
    NotEntitled,
    entitlement_from_claims,
    format_client_entitlements,
    parse_client_entitlements,
    visible,
)

ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL"
KID = "k1"
SCOPE_W = "artf-control/campaigns:write"
SCOPE_R = "artf-control/campaigns:read"
NOW = 1_700_000_000.0


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="module")
def keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(keypair):
    numbers = keypair.public_key().public_numbers()
    return {"keys": [{
        "kid": KID, "kty": "RSA", "alg": "RS256",
        "n": b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }]}


def sign(keypair, header: dict, claims: dict) -> str:
    h = b64url(json.dumps(header).encode())
    c = b64url(json.dumps(claims).encode())
    signature = keypair.sign(f"{h}.{c}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return f"{h}.{c}.{b64url(signature)}"


def token(keypair, *, scope=SCOPE_W, exp=NOW + 600, iss=ISSUER, alg="RS256", kid=KID, client_id="client-a", extra=None):
    header = {"alg": alg, "kid": kid} if kid is not None else {"alg": alg}
    claims = {"iss": iss, "exp": exp, "client_id": client_id, "token_use": "access", "scope": scope}
    claims.update(extra or {})
    return sign(keypair, header, claims)


# ---------------------------------------------------------------------- auth


def test_a_good_token_verifies_and_returns_its_claims(keypair, jwks):
    result = auth.verify(token(keypair), jwks=jwks, issuer=ISSUER, required_scopes=(SCOPE_W,), now=NOW)
    assert result.ok
    assert result.claims["client_id"] == "client-a"


def test_bearer_token_extraction():
    assert auth.bearer_token("Bearer abc") == "abc"
    assert auth.bearer_token("bearer abc") == "abc"
    assert auth.bearer_token("Basic abc") is None
    assert auth.bearer_token("Bearer ") is None
    assert auth.bearer_token(None) is None


@pytest.mark.parametrize("mutate,outcome", [
    (lambda kp: None, auth.AuthOutcome.MISSING),
    (lambda kp: "not.a.jwt.at.all", auth.AuthOutcome.MALFORMED),
    (lambda kp: "a.b", auth.AuthOutcome.MALFORMED),
    (lambda kp: token(kp, alg="none"), auth.AuthOutcome.UNSUPPORTED_ALGORITHM),
    (lambda kp: token(kp, alg="HS256"), auth.AuthOutcome.UNSUPPORTED_ALGORITHM),
    (lambda kp: token(kp, kid=None), auth.AuthOutcome.UNKNOWN_KEY),
    (lambda kp: token(kp, kid="rotated-out"), auth.AuthOutcome.UNKNOWN_KEY),
    (lambda kp: token(kp, exp=NOW - 3600), auth.AuthOutcome.EXPIRED),
    (lambda kp: token(kp, iss="https://cognito-idp.us-east-1.amazonaws.com/us-east-1_OTHER"), auth.AuthOutcome.WRONG_ISSUER),
    (lambda kp: token(kp, scope="artf-control/deals:read"), auth.AuthOutcome.MISSING_SCOPE),
    (lambda kp: token(kp, scope="artf-control/campaigns:writeonly"), auth.AuthOutcome.MISSING_SCOPE),
])
def test_each_refusal_is_distinct(keypair, jwks, mutate, outcome):
    result = auth.verify(mutate(keypair), jwks=jwks, issuer=ISSUER, required_scopes=(SCOPE_W,), now=NOW)
    assert result.outcome is outcome
    assert not result.ok
    assert result.claims is None


def test_a_token_signed_by_another_key_under_the_same_kid_is_refused(other_keypair, jwks):
    result = auth.verify(token(other_keypair), jwks=jwks, issuer=ISSUER, now=NOW)
    assert result.outcome is auth.AuthOutcome.BAD_SIGNATURE


def test_a_tampered_claim_breaks_the_signature(keypair, jwks):
    good = token(keypair, scope=SCOPE_R)
    h, c, s = good.split(".")
    claims = json.loads(base64.urlsafe_b64decode(c + "=" * (-len(c) % 4)))
    claims["scope"] = SCOPE_W
    forged = f"{h}.{b64url(json.dumps(claims).encode())}.{s}"
    result = auth.verify(forged, jwks=jwks, issuer=ISSUER, required_scopes=(SCOPE_W,), now=NOW)
    assert result.outcome is auth.AuthOutcome.BAD_SIGNATURE


def test_any_one_of_several_required_scopes_suffices(keypair, jwks):
    result = auth.verify(token(keypair, scope=SCOPE_R), jwks=jwks, issuer=ISSUER, required_scopes=(SCOPE_R, SCOPE_W), now=NOW)
    assert result.ok


def test_leeway_tolerates_small_skew_only(keypair, jwks):
    assert auth.verify(token(keypair, exp=NOW - 30), jwks=jwks, issuer=ISSUER, now=NOW).ok
    assert not auth.verify(token(keypair, exp=NOW - 120), jwks=jwks, issuer=ISSUER, now=NOW).ok


def test_jwks_cache_keeps_a_good_set_when_a_refresh_returns_nothing():
    calls = []
    sets = [{"keys": [{"kid": "a"}]}, {}, {"keys": []}]

    def fetch():
        calls.append(1)
        return sets[min(len(calls) - 1, len(sets) - 1)]

    cache = auth.JwksCache(fetch, ttl_seconds=10)
    first = cache.get(now=0)
    assert first["keys"][0]["kid"] == "a"
    assert cache.get(now=5) is first, "within ttl: no fetch"
    assert cache.get(now=20)["keys"][0]["kid"] == "a", "an empty refresh does not blank the cache"
    assert len(calls) == 2


def test_cognito_urls():
    assert auth.cognito_issuer("us-east-1", "us-east-1_X") == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_X"
    assert auth.cognito_jwks_url("us-east-1", "us-east-1_X").endswith("/us-east-1_X/.well-known/jwks.json")


# --------------------------------------------------------------------- scope


def test_no_claims_means_no_entitlement():
    e = entitlement_from_claims(None)
    assert not e.unrestricted and not e.principals
    with pytest.raises(NotEntitled):
        e.require("acme.example", kind="advertiser")


def test_a_machine_token_resolves_through_the_client_map():
    e = entitlement_from_claims({"client_id": "client-a"}, client_entitlements={"client-a": ["Acme.example"]})
    assert e.permits("acme.example")
    assert e.permits("ACME.EXAMPLE"), "domains compare case-insensitively"
    assert not e.permits("globex.example")
    assert e.client_id == "client-a"


def test_an_unmapped_client_is_entitled_to_nothing():
    e = entitlement_from_claims({"client_id": "client-z"}, client_entitlements={"client-a": ["acme.example"]})
    assert not e.permits("acme.example")


def test_the_tokens_own_claim_wins_over_the_map():
    e = entitlement_from_claims(
        {"client_id": "client-a", "custom:advertisers": "globex.example"},
        client_entitlements={"client-a": [WILDCARD]},
    )
    assert not e.unrestricted
    assert e.permits("globex.example") and not e.permits("acme.example")


def test_wildcard_is_unrestricted():
    e = entitlement_from_claims({"client_id": "c"}, client_entitlements={"c": ["*"]})
    assert e.unrestricted and e.permits("anything.example")


def test_require_names_the_kind_in_its_error():
    e = Entitlement(frozenset({"acme.example"}), unrestricted=False)
    with pytest.raises(NotEntitled, match="seller 'globex.example'"):
        e.require("globex.example", kind="seller")
    with pytest.raises(NotEntitled, match="no advertiser"):
        e.require("", kind="advertiser")


def test_visible_filters_by_the_named_principal_field():
    e = Entitlement(frozenset({"acme.example"}), unrestricted=False)
    items = [{"adomain": "acme.example", "x": 1}, {"adomain": "globex.example", "x": 2}, {"x": 3}]
    assert [i["x"] for i in visible(e, items, key="adomain")] == [1]
    assert visible(Entitlement(frozenset(), unrestricted=True), items, key="adomain") == items


@pytest.mark.parametrize("raw,expected", [
    ("client-a=acme.example,globex.example;client-b=*", {"client-a": ["acme.example", "globex.example"], "client-b": ["*"]}),
    ("", {}),
    (None, {}),
    ("garbage", {}),
    ("=acme.example", {}),
    ("client-a=", {}),
    ("client-a=;client-b=acme.example", {"client-b": ["acme.example"]}),
    (" client-a = Acme.Example , ", {"client-a": ["acme.example"]}),
])
def test_parse_client_entitlements_examples(raw, expected):
    assert parse_client_entitlements(raw) == expected


_client = st.from_regex(r"[a-z0-9]{1,20}", fullmatch=True)
_domain = st.from_regex(r"[a-z0-9]{1,12}\.example", fullmatch=True)
_grant = st.one_of(st.just(["*"]), st.lists(_domain, min_size=1, max_size=4, unique=True))


@settings(max_examples=200, deadline=None)
@given(st.dictionaries(_client, _grant, min_size=0, max_size=5))
def test_format_then_parse_is_a_round_trip(mapping):
    assert parse_client_entitlements(format_client_entitlements(mapping)) == mapping


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=st.characters(blacklist_characters="*"), max_size=200))
def test_text_without_a_star_never_parses_to_a_wildcard(raw):
    # The worst failure of this parser is a malformed grant becoming "everything".
    parsed = parse_client_entitlements(raw)
    for grants in parsed.values():
        assert WILDCARD not in grants
        e = entitlement_from_claims({"client_id": next(iter(parsed))}, client_entitlements=parsed)
        assert not e.unrestricted


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=200))
def test_parsing_never_raises_and_never_yields_an_empty_grant(raw):
    parsed = parse_client_entitlements(raw)
    for client, grants in parsed.items():
        assert client and grants
        assert all(g for g in grants)
