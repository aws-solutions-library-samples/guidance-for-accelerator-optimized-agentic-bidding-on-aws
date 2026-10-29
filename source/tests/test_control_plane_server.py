"""The registration layer, driven through a stand-in for the MCP SDK.

The SDK is not installed in this environment (and the embedded DSP guidance keeps it
out of its development environment too), so ``create_server`` is exercised against a
fake ``mcp.server.fastmcp`` that records what is registered and lets the tests call
the registered functions directly. What this proves:

  - exactly the ten tools are registered, each taking ``ctx`` first
  - every tool resolves the caller from the ``Authorization`` header, verifies it,
    requires the right scope, and maps refusals to stable error shapes
  - a read scope opens reads and not writes; a campaign scope does not open deals
  - an unexpected exception becomes a generic error, never a traceback to the caller

When the real SDK is present the same tests run against it unchanged, because the
fake mirrors the two SDK surfaces the server touches: ``FastMCP(...)`` and
``@server.tool()``.
"""
from __future__ import annotations

import base64
import importlib
import json
import sys
import time
import types
import typing

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from control_plane import server as server_module
from control_plane.operations import ControlPlaneOperations, MemoryStore
from control_plane.server import (
    Unauthorized,
    create_server,
    entitlement_for,
    guarded,
    headers_from_context,
    scope_names,
    scopes_for,
)

ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TESTPOOL"
KID = "k1"
PREFIX = "artf-control"
S = scope_names(PREFIX)
ORIGIN = "tst-prebid.example"
TOOLS = (
    "create_campaign", "update_campaign", "activate_campaign", "pause_campaign",
    "get_campaign", "list_campaigns", "create_deal", "update_deal", "get_deal", "list_deals",
)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="module")
def keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(keypair):
    numbers = keypair.public_key().public_numbers()
    return {"keys": [{
        "kid": KID, "kty": "RSA", "alg": "RS256",
        "n": b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }]}


def token_for(keypair, *, scope: str, client_id="client-a", exp=None) -> str:
    """Expiry in REAL time: entitlement_for reads the wall clock deliberately."""
    header = b64url(json.dumps({"alg": "RS256", "kid": KID}).encode())
    claims = b64url(json.dumps({
        "iss": ISSUER, "exp": exp if exp is not None else time.time() + 300,
        "client_id": client_id, "token_use": "access", "scope": scope,
    }).encode())
    sig = keypair.sign(f"{header}.{claims}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{claims}.{b64url(sig)}"


class Ctx:
    """What FastMCP hands a tool: a request context carrying the Starlette request."""

    def __init__(self, headers=None):
        request = types.SimpleNamespace(headers=headers or {})
        self.request_context = types.SimpleNamespace(request=request)


# ------------------------------------------------------- the SDK stand-in


class FakeFastMCP:
    instances: list["FakeFastMCP"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tools: dict[str, callable] = {}
        FakeFastMCP.instances.append(self)

    def tool(self, **_kwargs):
        def register(fn):
            # The real SDK evaluates the annotations to find the Context parameter;
            # an annotation that does not resolve fails the server at startup.
            hints = typing.get_type_hints(fn)
            assert hints.get("ctx") is not None, f"{fn.__name__}: ctx has no resolvable annotation"
            self.tools[fn.__name__] = fn
            return fn
        return register

    def run(self, **_kwargs):  # pragma: no cover - never run in tests
        raise AssertionError("run() must not be called by tests")


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install the stand-in only when the real SDK is absent."""
    try:
        importlib.import_module("mcp.server.fastmcp")
        yield "real"
        return
    except ImportError:
        pass
    mcp = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = FakeFastMCP
    fastmcp.Context = Ctx
    mcp.server = mcp_server
    mcp_server.fastmcp = fastmcp
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp)
    FakeFastMCP.instances.clear()
    yield "fake"


def _tools_of(server):
    if isinstance(server, FakeFastMCP):
        return server.tools
    # Real SDK: the tool manager holds Tool objects whose .fn is the function.
    return {name: tool.fn for name, tool in server._tool_manager._tools.items()}


@pytest.fixture
def store():
    deal = {
        "record_type": "deal", "id": "deal-a", "seq": 0, "origin": ORIGIN, "seller": "pub.example",
        "terms": {"dealfloor": 1.5, "cur": "USD", "pricetype": 1}, "adtypes": [1], "pubcount": 1,
        "dinventory": 1, "inventory": {"sitedomains": ["pub.example"]},
    }
    campaign = {
        "record_type": "campaign", "id": "camp-a", "seq": 0, "campaign_id": "camp-a",
        "campaign_name": "A", "adomain": "adv.example", "creative_id": "cr", "width": 300, "height": 250,
        "declared_cpm": 3.0, "deal_ids": ["deal-a"], "open_market": False, "target_categories": [],
        "media_type": "banner", "status": "paused",
    }
    return MemoryStore([deal, campaign])


@pytest.fixture
def served(fake_sdk, store, jwks):
    operations = ControlPlaneOperations(store, deal_origin=ORIGIN, price_ceiling=50.0)
    server = create_server(
        operations, jwks_provider=lambda: jwks, issuer=ISSUER, scope_prefix=PREFIX,
        client_entitlements={"client-a": ["*"], "client-narrow": ["other.example"]},
    )
    return _tools_of(server)


def call(tools, name, token, **kwargs):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return tools[name](Ctx(headers), **kwargs)


# ------------------------------------------------------------------- tests


def test_exactly_the_ten_tools_are_registered(served):
    assert set(served) == set(TOOLS)


def test_every_tool_takes_the_context_first(served):
    import inspect
    for name, fn in served.items():
        params = list(inspect.signature(fn).parameters)
        assert params and params[0] == "ctx", name


def test_the_server_is_stateless_http_on_the_agentcore_bind(fake_sdk, store, jwks):
    if fake_sdk != "fake":
        pytest.skip("constructor kwargs are only observable on the stand-in")
    create_server(ControlPlaneOperations(store, deal_origin=ORIGIN), jwks_provider=lambda: jwks, issuer=ISSUER)
    kwargs = FakeFastMCP.instances[-1].kwargs
    assert kwargs == {"host": "0.0.0.0", "port": 8000, "stateless_http": True}


def test_no_token_is_unauthorized(served):
    out = call(served, "list_campaigns", None)
    assert out["error"] == "unauthorized"
    assert "missing" in out["message"]


def test_a_read_scope_opens_reads_and_not_writes(served, keypair):
    reader = token_for(keypair, scope=S["campaigns:read"])
    assert call(served, "list_campaigns", reader)["count"] == 1
    assert call(served, "get_campaign", reader, campaign_id="camp-a")["campaign"]["campaign_id"] == "camp-a"
    refused = call(served, "activate_campaign", reader, campaign_id="camp-a")
    assert refused["error"] == "unauthorized" and "missing_scope" in refused["message"]


def test_a_write_scope_also_opens_reads(served, keypair):
    writer = token_for(keypair, scope=S["campaigns:write"])
    assert call(served, "list_campaigns", writer)["count"] == 1


def test_a_campaign_scope_does_not_open_deals(served, keypair):
    writer = token_for(keypair, scope=S["campaigns:write"])
    assert call(served, "list_deals", writer)["error"] == "unauthorized"
    both = token_for(keypair, scope=f"{S['campaigns:write']} {S['deals:read']}")
    assert call(served, "list_deals", both)["count"] == 1


def test_the_full_write_path_through_the_tool_surface(served, keypair, store):
    writer = token_for(keypair, scope=f"{S['campaigns:write']} {S['deals:write']}")
    created = call(served, "create_deal", writer, deal={
        "id": "deal-b", "seller": "pub.example", "terms": {"dealfloor": 2.0}, "inventory": {"sitedomains": ["pub.example"]},
    })
    assert created["deal"]["origin"] == ORIGIN
    camp = call(served, "create_campaign", writer, campaign_id="camp-b", campaign_name="B", adomain="adv.example",
                creative_id="cr-b", width=300, height=250, declared_cpm=2.5, deal_ids=["deal-b"])
    assert camp["campaign"]["status"] == "paused"
    activated = call(served, "activate_campaign", writer, campaign_id="camp-b")
    assert activated["campaign"]["status"] == "active"
    assert store.get("campaign", "camp-b")["managed_by"] == "control-plane"


def test_refusals_are_structured_and_name_the_rule(served, keypair):
    writer = token_for(keypair, scope=S["campaigns:write"])
    out = call(served, "create_campaign", writer, campaign_id="Bad_Id", campaign_name="B", adomain="adv.example",
               creative_id="cr", width=300, height=250, declared_cpm=2.5, open_market=True)
    assert out["error"] == "invalid_request" and "lowercase" in out["message"]
    out = call(served, "get_campaign", writer, campaign_id="camp-missing")
    assert out["error"] == "not_found"


def test_an_unentitled_client_is_forbidden_on_create_and_not_found_on_read(served, keypair):
    narrow = token_for(keypair, scope=f"{S['campaigns:write']} {S['campaigns:read']}", client_id="client-narrow")
    out = call(served, "create_campaign", narrow, campaign_id="camp-c", campaign_name="C", adomain="adv.example",
               creative_id="cr", width=300, height=250, declared_cpm=2.5, open_market=True)
    assert out["error"] == "forbidden"
    out = call(served, "get_campaign", narrow, campaign_id="camp-a")
    assert out["error"] == "not_found", "not yours reads as not found"


def test_an_unmapped_client_is_denied_everything(served, keypair):
    stranger = token_for(keypair, scope=S["campaigns:read"], client_id="client-unknown")
    assert call(served, "list_campaigns", stranger)["count"] == 0
    assert call(served, "get_campaign", stranger, campaign_id="camp-a")["error"] == "not_found"


def test_an_expired_token_is_refused_by_the_container_itself(served, keypair):
    stale = token_for(keypair, scope=S["campaigns:read"], exp=time.time() - 3600)
    out = call(served, "list_campaigns", stale)
    assert out["error"] == "unauthorized" and "expired" in out["message"]


def test_a_header_with_the_wrong_scheme_is_no_token(served, keypair):
    good = token_for(keypair, scope=S["campaigns:read"])
    out = served["list_campaigns"](Ctx({"Authorization": f"Basic {good}"}))
    assert out["error"] == "unauthorized"


# ----------------------------------------------------------- pure helpers


def test_scopes_for_covers_every_tool_with_the_right_prefix():
    scopes = scopes_for("my-prefix")
    assert set(scopes) == set(TOOLS)
    for tool, accepted in scopes.items():
        assert all(s.startswith("my-prefix/") for s in accepted), tool
    assert scopes["create_deal"] == ("my-prefix/deals:write",)
    assert set(scopes["get_deal"]) == {"my-prefix/deals:read", "my-prefix/deals:write"}


def test_headers_from_context_lowercases_and_tolerates_absence():
    assert headers_from_context(Ctx({"Authorization": "Bearer x", "X-Thing": "1"})) == {"authorization": "Bearer x", "x-thing": "1"}
    assert headers_from_context(object()) == {}
    assert headers_from_context(None) == {}


def test_entitlement_for_raises_rather_than_returning_empty(jwks):
    with pytest.raises(Unauthorized):
        entitlement_for(None, jwks_provider=lambda: jwks, issuer=ISSUER)


def test_guarded_turns_an_unexpected_exception_into_a_generic_error(caplog):
    def boom(entitlement, **_kwargs):
        raise RuntimeError("table on fire: arn:aws:dynamodb:...")

    out = guarded(boom, resolve=lambda: object(), tool="list_deals")
    assert out["error"] == "internal_error"
    assert "arn:aws" not in out["message"], "internal detail stays in the log"
    assert any("control_plane_tool_failed" in r.getMessage() for r in caplog.records)


def test_guarded_logs_the_outcome_without_the_token(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="artf.control_plane")

    def ok(entitlement, **_kwargs):
        return {"fine": True}

    assert guarded(ok, resolve=lambda: object(), tool="get_deal") == {"fine": True}
    logged = [json.loads(r.getMessage()) for r in caplog.records if "control_plane_tool" in r.getMessage()]
    assert logged and logged[-1]["tool"] == "get_deal" and logged[-1]["outcome"] == "ok"
    assert "authorization" not in json.dumps(logged).lower()


def test_main_module_exposes_the_env_contract():
    # The deploy script sets these; a rename here without a rename there is a runtime
    # that starts and then refuses everything. Pin the names.
    src = open(server_module.__file__, encoding="utf-8").read()
    for name in ("ARTF_CONTROL_TABLE", "ARTF_CONTROL_TOKEN_ISSUER", "ARTF_CONTROL_DEAL_ORIGIN",
                 "ARTF_CONTROL_SCOPE_PREFIX", "ARTF_CONTROL_CLIENT_ENTITLEMENTS", "ARTF_CONTROL_PRICE_CEILING"):
        assert name in src, name
