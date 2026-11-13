"""The in-cluster demand service: readiness, the bearer check, the catalog snapshot,
and no DynamoDB call on the bid path."""
import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt as jose_jwt
from starlette.testclient import TestClient

from demand.artfhouse import deals_source, server, snapshot
from demand.artfhouse.bearer import TokenVerifier

REGION = "us-east-1"
POOL = "us-east-1_TESTPOOL"
CLIENT = "prebid-host-client"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}"


def _b64(n: int) -> str:
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    pub = key.public_key().public_numbers()
    jwks = {"keys": [{"kty": "RSA", "kid": "k1", "alg": "RS256", "use": "sig",
                      "n": _b64(pub.n), "e": _b64(pub.e)}]}
    return pem, jwks


def token(pem, **overrides):
    claims = {"iss": ISSUER, "exp": int(time.time()) + 600, "token_use": "access",
              "client_id": CLIENT, "scope": "artf-demand/bid:read"}
    claims.update(overrides)
    return jose_jwt.encode(claims, pem, algorithm="RS256", headers={"kid": "k1"})


def verifier(jwks, fetch=None):
    v = TokenVerifier(REGION, POOL, CLIENT, fetch=fetch or (lambda _url: jwks))
    v.refresh_keys()
    return v


@pytest.fixture
def loaded(keypair):
    """A loaded snapshot (from DEALS_SEED_FILE) and verifier, without the lifespan threads."""
    _pem, jwks = keypair
    snapshot.load()
    server.set_verifier(verifier(jwks))
    yield
    server.set_verifier(None)
    snapshot._snapshot = None


def client():
    # No `with`: the lifespan (and its refresh threads) is not run in these tests.
    return TestClient(server.app)


BID = {"id": "req-1", "imp": [{"id": "imp-1", "bidfloor": 1.0,
                               "pmp": {"deals": [{"id": "deal-home-premium"}]}}]}


def test_readiness_fails_until_the_catalog_is_loaded(keypair):
    _pem, jwks = keypair
    snapshot._snapshot = None
    server.set_verifier(verifier(jwks))
    try:
        assert client().get("/health/ready").status_code == 503
        assert client().post("/openrtb2/auction", json=BID).status_code == 503
        snapshot.load()
        resp = client().get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["catalog"]["campaigns"] > 0
    finally:
        server.set_verifier(None)
        snapshot._snapshot = None


def test_readiness_fails_until_the_signing_keys_are_loaded(keypair):
    snapshot.load()
    failing = TokenVerifier(REGION, POOL, CLIENT, fetch=lambda _u: (_ for _ in ()).throw(OSError("down")))
    failing.refresh_keys()
    server.set_verifier(failing)
    try:
        assert client().get("/health/ready").status_code == 503
    finally:
        server.set_verifier(None)
        snapshot._snapshot = None


def test_a_valid_token_gets_a_bid_response(loaded, keypair):
    pem, _ = keypair
    resp = client().post("/openrtb2/auction", json=BID,
                         headers={"Authorization": f"Bearer {token(pem)}"})
    assert resp.status_code == 200
    assert resp.json()["seatbid"][0]["seat"] == "artfhouse"


@pytest.mark.parametrize("overrides", [
    {"scope": "artf-orchestrator/mutations:write"},
    {"client_id": "someone-else"},
    {"token_use": "id"},
    {"exp": int(time.time()) - 10},
    {"iss": "https://cognito-idp.us-east-1.amazonaws.com/other"},
])
def test_a_token_the_authorizer_would_refuse_is_401(loaded, keypair, overrides):
    pem, _ = keypair
    resp = client().post("/openrtb2/auction", json=BID,
                         headers={"Authorization": f"Bearer {token(pem, **overrides)}"})
    assert resp.status_code == 401
    assert "seatbid" not in resp.json()


def test_no_token_is_401(loaded):
    assert client().post("/openrtb2/auction", json=BID).status_code == 401


def test_a_token_signed_by_another_key_is_401(loaded):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    resp = client().post("/openrtb2/auction", json=BID,
                         headers={"Authorization": f"Bearer {token(other)}"})
    assert resp.status_code == 401


def test_a_bid_makes_no_dynamodb_call(loaded, keypair, monkeypatch):
    """The bid path reads the snapshot only. Every route to the table is stubbed to
    record a call; handling bids must record none."""
    pem, _ = keypair
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("DynamoDB reached on the bid path")

    monkeypatch.setattr(deals_source, "_table", record)
    monkeypatch.setattr(deals_source, "campaign_records", record)
    monkeypatch.setattr(deals_source, "deal_records", record)
    import boto3
    monkeypatch.setattr(boto3, "resource", record)
    monkeypatch.setattr(boto3, "client", record)
    c = client()
    for _ in range(5):
        resp = c.post("/openrtb2/auction", json=BID,
                      headers={"Authorization": f"Bearer {token(pem)}"})
        assert resp.status_code == 200
    assert calls == []


def test_a_failed_refresh_keeps_the_last_snapshot(loaded, monkeypatch):
    before = snapshot.current()
    monkeypatch.setattr(deals_source, "campaign_records",
                        lambda: (_ for _ in ()).throw(OSError("throttled")))
    assert snapshot.refresh_once() is False
    assert snapshot.current() is before
    assert snapshot.ready()


def test_a_refresh_swaps_in_a_new_catalog(loaded, monkeypatch):
    before = snapshot.current()
    records = deals_source.campaign_records()
    monkeypatch.setattr(deals_source, "campaign_records", lambda: records[:1])
    assert snapshot.refresh_once() is True
    assert snapshot.current() is not before
    assert snapshot.current().campaigns == 1


def test_an_empty_source_is_a_fault_not_an_empty_catalog(monkeypatch):
    snapshot._snapshot = None
    monkeypatch.setattr(deals_source, "campaign_records", lambda: [])
    assert snapshot.refresh_once() is False
    assert not snapshot.ready()


def test_the_refresh_period_matches_the_deal_library():
    from shared import deal_library
    assert snapshot.REFRESH_SECONDS == deal_library.REFRESH_SECONDS == 30
