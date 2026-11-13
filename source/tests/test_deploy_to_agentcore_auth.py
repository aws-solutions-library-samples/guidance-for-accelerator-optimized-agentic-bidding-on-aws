"""deployment/scripts/deploy_to_agentcore.py: the inbound-auth parameters.

Three runtimes go through this script. Only the campaign/deals control plane passes
an authorizer and a header allowlist; the ARTF MCP runtime and the closed-loop agents
stay on IAM. So the property that matters most is that ABSENT MEANS ABSENT: a caller
that passes neither must produce create/update calls with neither key, exactly as
before. Driven with a fake control-plane client; nothing reaches AWS.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
_SCRIPT = os.path.join(_REPO, "deployment", "scripts", "deploy_to_agentcore.py")
AUTH = {"customJWTAuthorizer": {"discoveryUrl": "https://example/.well-known/openid-configuration",
                                "allowedClients": ["client-a"], "allowedScopes": ["artf-control/deals:read"]}}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("deploy_to_agentcore_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, existing=None):
        self.existing = existing
        self.calls = []

    def create_agent_runtime(self, **kwargs):
        self.calls.append(("create", kwargs))
        return {"agentRuntimeId": "rt-1"}

    def update_agent_runtime(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {}


@pytest.fixture
def patched(mod, monkeypatch):
    def install(existing=None):
        client = FakeClient(existing)
        monkeypatch.setattr(mod, "_find_runtime", lambda c, name: existing)
        monkeypatch.setattr(mod, "_wait_ready", lambda c, rid, timeout=600: f"arn:aws:bedrock-agentcore:us-east-1:1:runtime/{rid}")
        return client
    return install


@pytest.mark.parametrize("existing", [None, {"agentRuntimeId": "rt-9"}])
def test_absent_means_absent_on_create_and_update(mod, patched, existing):
    client = patched(existing)
    mod.deploy(client, name="x_mcp", role_arn="arn:role", container_uri="img:tag")
    (_kind, kwargs), = client.calls
    assert "authorizerConfiguration" not in kwargs
    assert "requestHeaderConfiguration" not in kwargs


@pytest.mark.parametrize("existing,kind", [(None, "create"), ({"agentRuntimeId": "rt-9"}, "update")])
def test_both_parameters_reach_create_and_update(mod, patched, existing, kind):
    client = patched(existing)
    mod.deploy(client, name="x_control", role_arn="arn:role", container_uri="img:tag",
               authorizer=AUTH, request_header_allowlist=["Authorization"])
    (got_kind, kwargs), = client.calls
    assert got_kind == kind
    assert kwargs["authorizerConfiguration"] == AUTH
    assert kwargs["requestHeaderConfiguration"] == {"requestHeaderAllowlist": ["Authorization"]}


def test_an_authorizer_without_custom_jwt_is_refused(mod, patched):
    client = patched(None)
    with pytest.raises(ValueError):
        mod.deploy(client, name="x", role_arn="arn:role", container_uri="img", authorizer={"other": {}})
    assert client.calls == []


def test_main_parses_the_authorizer_json_and_allowlist(mod, monkeypatch):
    seen = {}

    def fake_deploy(client, **kwargs):
        seen.update(kwargs)
        return "arn"

    monkeypatch.setattr(mod, "deploy", fake_deploy)
    monkeypatch.setattr(mod, "_get_client", lambda region: object())
    code = mod.main(["--action", "deploy", "--runtime-name", "x", "--role-arn", "r", "--container-uri", "c",
                     "--region", "us-east-1", "--authorizer-json", json.dumps(AUTH),
                     "--request-header-allowlist", "Authorization"])
    assert code == 0
    assert seen["authorizer"] == AUTH
    assert seen["request_header_allowlist"] == ["Authorization"]


@pytest.mark.parametrize("bad", ["{not json", json.dumps({"x": 1}), json.dumps([1])])
def test_main_rejects_a_malformed_authorizer_before_calling_aws(mod, monkeypatch, bad):
    monkeypatch.setattr(mod, "deploy", lambda *a, **k: pytest.fail("deploy must not be called"))
    monkeypatch.setattr(mod, "_get_client", lambda region: object())
    code = mod.main(["--action", "deploy", "--runtime-name", "x", "--role-arn", "r", "--container-uri", "c",
                     "--region", "us-east-1", "--authorizer-json", bad])
    assert code == 1
