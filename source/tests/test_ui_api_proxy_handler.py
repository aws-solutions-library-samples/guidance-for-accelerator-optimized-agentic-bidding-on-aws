"""The UI API proxy Lambda (deployment/ui_api_proxy_cfn.yaml) forwards browser calls
to the orchestrator ClusterIP Service and nothing else.

The handler is inline in the CloudFormation template, so these tests extract the
ZipFile text and exec it with urllib stubbed, then check the four behaviours the
Express relies on: path and method allowlists, header allowlist, upstream 4xx/5xx
passthrough, and unreachable -> 502 / timeout -> 504 without raising.
"""

from __future__ import annotations

import io
import json
import re
import socket
import urllib.error
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "deployment" / "ui_api_proxy_cfn.yaml"


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that keeps CloudFormation short-form intrinsics (!If, !Sub, ...) as
    plain {tag: value} dicts so the template can be inspected."""


def _cfn_tag(loader, tag_suffix, node):
    name = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"
    if isinstance(node, yaml.ScalarNode):
        return {name: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {name: loader.construct_sequence(node)}
    return {name: loader.construct_mapping(node)}


_CfnLoader.add_multi_constructor("!", _cfn_tag)


def _load_template():
    return yaml.load(TEMPLATE.read_text(), Loader=_CfnLoader)


def _load_handler_module(monkeypatch, env=None):
    doc = _load_template()
    code = doc["Resources"]["ProxyFunction"]["Properties"]["Code"]["ZipFile"]
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ORCHESTRATOR_BASE_URL", "http://orchestrator.default.svc.cluster.local")
    ns: dict = {}
    exec(compile(code, "ui_api_proxy_index.py", "exec"), ns)
    return ns


class _FakeResponse(io.BytesIO):
    def __init__(self, status, body: bytes, content_type="application/json", extra_headers=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Type": content_type, **(extra_headers or {})}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _capture_urlopen(ns, monkeypatch, *, status=200, body=b'{"ok":true}', content_type="application/json",
                     raise_exc=None, extra_headers=None):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["data"] = req.data
        seen["timeout"] = timeout
        if raise_exc is not None:
            raise raise_exc
        return _FakeResponse(status, body, content_type, extra_headers)

    monkeypatch.setattr(ns["urllib"].request, "urlopen", fake_urlopen)
    return seen


# --- template shape -----------------------------------------------------------

def test_template_properties_match_express():
    doc = _load_template()
    fn = doc["Resources"]["ProxyFunction"]["Properties"]
    assert fn["Timeout"] == 60
    assert fn["Runtime"] == "python3.12"
    assert "VpcConfig" in fn
    role = doc["Resources"]["ProxyFunctionRole"]["Properties"]
    assert set(role["ManagedPolicyArns"]) == {
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole",
    }
    assert "Policies" not in role, "the proxy role needs nothing beyond logs + VPC ENIs"
    trust = role["AssumeRolePolicyDocument"]["Statement"][0]
    assert trust["Condition"]["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}
    assert doc["Resources"]["ProxyLogGroup"]["Properties"]["RetentionInDays"] >= 90
    assert set(doc["Outputs"]) >= {"ProxyFunctionArn", "ProxyFunctionName"}


def test_template_is_ascii():
    raw = TEMPLATE.read_bytes()
    assert all(b < 128 for b in raw), "non-ASCII byte in CFN template"


# --- allowlists ---------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/mcp", "/v1/containers", "/health/ready", "/fabric/v1/mutations",
                                  "/api/../etc", "/api//x", "api/v1/containers", "/api"])
def test_rejects_paths_outside_allowlist(monkeypatch, path):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    out = ns["handler"]({"method": "GET", "path": path}, None)
    assert out["status"] == 400
    assert json.loads(out["body"])["error"] == "path_not_allowed"
    assert "url" not in seen, "nothing may be forwarded for a rejected path"


@pytest.mark.parametrize("path", ["/api/v1/containers", "/api/v1/loadtest/abc", "/api/health/ready", "/api/mcp"])
def test_allows_api_prefix_unchanged(monkeypatch, path):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    out = ns["handler"]({"method": "GET", "path": path}, None)
    assert out["status"] == 200
    assert seen["url"] == "http://orchestrator.default.svc.cluster.local" + path


def test_rejects_disallowed_method(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    out = ns["handler"]({"method": "TRACE", "path": "/api/v1/containers"}, None)
    assert out["status"] == 405
    assert "url" not in seen


def test_rejects_non_object_event(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    assert ns["handler"]("nope", None)["status"] == 400
    assert ns["handler"]({"method": "GET", "path": "/api/v1/x", "query": "a=b"}, None)["status"] == 400


def test_forwards_only_allowlisted_headers(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    ns["handler"]({
        "method": "GET", "path": "/api/v1/containers",
        "headers": {"Authorization": "Bearer tok", "Accept": "application/json", "Mcp-Session-Id": "s-1",
                    "Cookie": "session=1", "X-Forwarded-For": "1.2.3.4", "Host": "evil"},
    }, None)
    assert seen["headers"]["authorization"] == "Bearer tok"
    assert seen["headers"]["accept"] == "application/json"
    assert seen["headers"]["mcp-session-id"] == "s-1"
    assert "cookie" not in seen["headers"]
    assert "x-forwarded-for" not in seen["headers"]


def test_mcp_session_id_response_header_is_returned(monkeypatch):
    """The UI's MCP client reads Mcp-Session-Id off the initialize response."""
    ns = _load_handler_module(monkeypatch)
    _capture_urlopen(ns, monkeypatch, body=b'{"jsonrpc":"2.0","id":1,"result":{}}',
                     extra_headers={"Mcp-Session-Id": "abc123", "Server": "uvicorn"})
    out = ns["handler"]({"method": "POST", "path": "/api/mcp", "body": "{}"}, None)
    assert out["status"] == 200
    assert out["headers"]["mcp-session-id"] == "abc123"
    assert out["headers"]["content-type"] == "application/json"
    assert "server" not in out["headers"]


def test_sse_requests_are_refused_without_calling_upstream(monkeypatch):
    """A synchronous Invoke cannot stream; refuse fast so LoadTestPanel polls instead."""
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    out = ns["handler"]({"method": "GET", "path": "/api/v1/loadtest/abc/stream",
                         "headers": {"Accept": "text/event-stream"}}, None)
    assert out["status"] == 406
    assert json.loads(out["body"])["error"] == "streaming_not_supported"
    assert "url" not in seen


def test_query_is_url_encoded(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    ns["handler"]({"method": "GET", "path": "/api/v1/governance/sweep-status",
                   "query": {"model_type": "dlrm bid", "n": 5}}, None)
    assert seen["url"].endswith("?model_type=dlrm+bid&n=5")


# --- body handling ------------------------------------------------------------

def test_post_body_string_forwarded_with_json_content_type(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch, status=202, body=b'{"started":true}')
    out = ns["handler"]({"method": "POST", "path": "/api/v1/loadtest",
                         "body": json.dumps({"duration_s": 10})}, None)
    assert seen["method"] == "POST"
    assert seen["data"] == b'{"duration_s": 10}'
    assert seen["headers"]["content-type"] == "application/json"
    assert out["status"] == 202 and out["body"] == '{"started":true}'
    assert out["isBase64"] is False


def test_get_never_sends_a_body(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    seen = _capture_urlopen(ns, monkeypatch)
    ns["handler"]({"method": "GET", "path": "/api/v1/containers", "body": "ignored"}, None)
    assert seen["data"] is None


def test_binary_body_is_base64(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    _capture_urlopen(ns, monkeypatch, body=b"\xff\xfe\x00", content_type="application/octet-stream")
    out = ns["handler"]({"method": "GET", "path": "/api/v1/x"}, None)
    assert out["isBase64"] is True
    assert out["body"] == "//4A"


# --- upstream errors ------------------------------------------------------------

def test_upstream_4xx_passes_through(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    err = urllib.error.HTTPError("http://x", 409, "Conflict", {"Content-Type": "application/json"},
                                 io.BytesIO(b'{"error":"registry conflict"}'))
    _capture_urlopen(ns, monkeypatch, raise_exc=err)
    out = ns["handler"]({"method": "POST", "path": "/api/v1/containers/x/active", "body": "{}"}, None)
    assert out["status"] == 409
    assert json.loads(out["body"])["error"] == "registry conflict"


def test_unreachable_is_502(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    _capture_urlopen(ns, monkeypatch, raise_exc=urllib.error.URLError(ConnectionRefusedError(111, "refused")))
    out = ns["handler"]({"method": "GET", "path": "/api/v1/containers"}, None)
    assert out["status"] == 502
    assert json.loads(out["body"])["error"] == "orchestrator_unreachable"


def test_timeout_is_504(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    _capture_urlopen(ns, monkeypatch, raise_exc=urllib.error.URLError(socket.timeout("timed out")))
    out = ns["handler"]({"method": "GET", "path": "/api/v1/auction/run"}, None)
    assert out["status"] == 504
    assert json.loads(out["body"])["error"] == "orchestrator_timeout"


def test_upstream_timeout_is_below_lambda_timeout(monkeypatch):
    ns = _load_handler_module(monkeypatch)
    doc = _load_template()
    assert ns["UPSTREAM_TIMEOUT_S"] < doc["Resources"]["ProxyFunction"]["Properties"]["Timeout"]
