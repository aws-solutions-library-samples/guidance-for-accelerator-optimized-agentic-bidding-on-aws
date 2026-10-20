"""Deployment mode signal on the orchestrator (standard, learner, unknown).

deploy.sh --learner-mode renders DEPLOYMENT_MODE=learner into the orchestrator
Deployment. These tests pin how that value reaches the UI and what it gates:

- /v1/gpu/status reports ``deploymentMode`` and, in learner mode, makes no EKS
  call (there is no GPU node group to describe).
- An unrecognised value reads as ``unknown``, never as either mode.
- GPU start/stop and every governance route answer 409 in learner mode.
- The signal is only on the authenticated endpoint: /health/ready is unchanged
  and /v1/gpu/status without a token is refused.

Requests go through ``httpx.ASGITransport`` inside ``asyncio.run`` or call the
handler directly, never starlette's threaded TestClient (see
test_container_registry.py for why).
"""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from orchestrator import app as oapp  # noqa: E402
from orchestrator import deployment_mode  # noqa: E402
from orchestrator.container_registry import RegistryEntry  # noqa: E402

# A local port nothing listens on, so a probe fails fast with a refused connection.
_DEAD = "127.0.0.1:9"


def _request(path="/v1/gpu/status", method="GET"):
    from starlette.requests import Request

    return Request({"type": "http", "method": method, "path": path, "headers": [], "query_string": b""})


def _json(response):
    import json

    return json.loads(response.body)


class _ForbiddenBoto:
    """boto3 stand-in that fails the test if the handler reaches EKS."""

    calls = 0

    def client(self, *args, **kwargs):
        _ForbiddenBoto.calls += 1
        raise AssertionError("learner mode must not call EKS")


class _FakeEks:
    def __init__(self, desired=0, raises=None):
        self.desired = desired
        self.raises = raises
        self.updates = []

    def describe_nodegroup(self, **kwargs):
        if self.raises:
            raise self.raises
        return {
            "nodegroup": {
                "scalingConfig": {"desiredSize": self.desired, "minSize": 0, "maxSize": 5},
                "status": "ACTIVE",
            }
        }

    def update_nodegroup_config(self, **kwargs):
        self.updates.append(kwargs)


def _patch_boto(monkeypatch, eks):
    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: eks)


@pytest.fixture
def no_mode(monkeypatch):
    monkeypatch.delenv(deployment_mode.MODE_ENV, raising=False)


# ---------------------------------------------------------------------------
# current_mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "standard"),
        ("", "standard"),
        ("   ", "standard"),
        ("learner", "learner"),
        (" learner ", "learner"),
        ("Learner", "unknown"),
        ("standard", "unknown"),
        ("gpu", "unknown"),
    ],
)
def test_current_mode(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(deployment_mode.MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(deployment_mode.MODE_ENV, raw)
    assert deployment_mode.current_mode() == expected


def test_standard_is_not_a_recognised_explicit_value(monkeypatch):
    # deploy.sh never renders DEPLOYMENT_MODE=standard; it leaves the variable
    # out. An explicit value nobody renders is a mistake, reported as unknown.
    monkeypatch.setenv(deployment_mode.MODE_ENV, "standard")
    assert not deployment_mode.is_learner()
    assert deployment_mode.current_mode() == "unknown"


# ---------------------------------------------------------------------------
# /v1/gpu/status
# ---------------------------------------------------------------------------

def test_gpu_status_standard_reports_mode(monkeypatch, no_mode):
    _patch_boto(monkeypatch, _FakeEks(desired=0))
    body = _json(asyncio.run(oapp.gpu_status(_request())))
    assert body["deploymentMode"] == "standard"
    assert body["desiredSize"] == 0
    assert body["tritonState"] == "stopped"


def test_gpu_status_unknown_mode_still_describes_nodegroup(monkeypatch):
    monkeypatch.setenv(deployment_mode.MODE_ENV, "Learner")
    eks = _FakeEks(desired=0)
    _patch_boto(monkeypatch, eks)
    body = _json(asyncio.run(oapp.gpu_status(_request())))
    assert body["deploymentMode"] == "unknown"
    assert body["status"] == "ACTIVE"


def test_gpu_status_error_still_carries_mode(monkeypatch, no_mode):
    _patch_boto(monkeypatch, _FakeEks(raises=RuntimeError("no such nodegroup")))
    resp = asyncio.run(oapp.gpu_status(_request()))
    assert resp.status_code == 500
    body = _json(resp)
    assert body["deploymentMode"] == "standard"
    assert "no such nodegroup" in body["error"]


def test_gpu_status_learner_skips_eks_and_reports_triton_down(monkeypatch):
    import boto3

    monkeypatch.setenv(deployment_mode.MODE_ENV, "learner")
    monkeypatch.setenv("TRITON_URL", _DEAD)
    _ForbiddenBoto.calls = 0
    monkeypatch.setattr(boto3, "client", _ForbiddenBoto().client)
    resp = asyncio.run(oapp.gpu_status(_request()))
    assert resp.status_code == 200
    body = _json(resp)
    assert _ForbiddenBoto.calls == 0
    assert body["deploymentMode"] == "learner"
    assert body["inferenceDevice"] == "cpu"
    assert body["tritonReady"] is False
    assert body["running"] is False
    assert body["tritonState"] == "not_ready"
    assert _DEAD in body["tritonDetail"]
    # No GPU node group: the size fields are null, not zero.
    assert body["desiredSize"] is None
    assert body["status"] is None


def test_gpu_status_learner_reports_triton_ready(monkeypatch):
    monkeypatch.setenv(deployment_mode.MODE_ENV, "learner")
    seen = []

    class _OkClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            seen.append(url)
            return httpx.Response(200)

    monkeypatch.setattr(oapp.httpx, "AsyncClient", _OkClient)
    body = _json(asyncio.run(oapp.gpu_status(_request())))
    assert body["tritonReady"] is True
    assert body["running"] is True
    assert body["tritonState"] == "ready"
    assert body["tritonDetail"] is None
    assert seen and seen[0].endswith("/v2/health/ready")


# ---------------------------------------------------------------------------
# GPU start/stop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("handler", ["gpu_start", "gpu_stop"])
def test_gpu_scaling_refused_in_learner_mode(monkeypatch, handler):
    import boto3

    monkeypatch.setenv(deployment_mode.MODE_ENV, "learner")
    _ForbiddenBoto.calls = 0
    monkeypatch.setattr(boto3, "client", _ForbiddenBoto().client)
    resp = asyncio.run(getattr(oapp, handler)(_request(method="POST")))
    assert resp.status_code == 409
    body = _json(resp)
    assert body["deploymentMode"] == "learner"
    assert "requires GPUs" in body["error"]
    assert _ForbiddenBoto.calls == 0


@pytest.mark.parametrize("handler,desired", [("gpu_start", 3), ("gpu_stop", 0)])
def test_gpu_scaling_unchanged_in_standard_mode(monkeypatch, no_mode, handler, desired):
    eks = _FakeEks()
    _patch_boto(monkeypatch, eks)
    resp = asyncio.run(getattr(oapp, handler)(_request(method="POST")))
    assert resp.status_code == 200
    assert _json(resp)["ok"] is True
    assert eks.updates[0]["scalingConfig"]["desiredSize"] == desired


# ---------------------------------------------------------------------------
# Governance routes
# ---------------------------------------------------------------------------

def _governance_paths():
    return [
        r.path for r in oapp.app.routes
        if getattr(r, "path", "").startswith(("/v1/governance/", "/api/v1/governance/"))
    ]


def _call(path, method="GET"):
    async def run():
        transport = httpx.ASGITransport(app=oapp.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://orch") as c:
            return await c.request(method, path, json={} if method == "POST" else None)

    return asyncio.run(run())


@pytest.mark.skipif(not oapp._GOVERNANCE_API_AVAILABLE, reason="governance API not importable here")
def test_every_governance_route_refused_in_learner_mode(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED", "true")
    monkeypatch.setenv(deployment_mode.MODE_ENV, "learner")
    paths = _governance_paths()
    # Nine routes under each of the two prefixes the frontend and direct
    # callers use.
    assert len(paths) == 18, paths
    for route in oapp.app.routes:
        if getattr(route, "path", "") not in paths:
            continue
        method = sorted(route.methods - {"HEAD"})[0]
        resp = _call(route.path, method)
        assert resp.status_code == 409, (route.path, resp.status_code, resp.text)
        assert resp.json()["deploymentMode"] == "learner"


@pytest.mark.skipif(not oapp._GOVERNANCE_API_AVAILABLE, reason="governance API not importable here")
def test_governance_route_not_gated_in_standard_mode(monkeypatch, no_mode):
    monkeypatch.setenv("AUTH_DISABLED", "true")
    resp = _call("/api/v1/governance/train", "POST")
    # Whatever the handler decides about an empty body, it is not the
    # learner-mode refusal.
    assert resp.status_code != 409 or "deploymentMode" not in resp.json()


# ---------------------------------------------------------------------------
# /v1/containers
# ---------------------------------------------------------------------------

def _dead_registry():
    entry = RegistryEntry(
        name="ncf-deal-manager",
        display_name="NCF Deal Manager",
        description="test",
        intents=frozenset({"ACTIVATE_DEALS"}),
        endpoint=f"http://{_DEAD}/mcp",
        grpc=_DEAD,
        active=True,
        source="code",
        configurable=False,
    )
    return [entry], []


@pytest.mark.parametrize("mode,expected", [("learner", "triton_offline"), (None, "gpu_offline")])
def test_containers_name_the_right_offline_cause(monkeypatch, mode, expected):
    if mode is None:
        monkeypatch.delenv(deployment_mode.MODE_ENV, raising=False)
    else:
        monkeypatch.setenv(deployment_mode.MODE_ENV, mode)
    monkeypatch.setenv("TRITON_URL", _DEAD)
    monkeypatch.setattr(oapp, "_effective_registry", _dead_registry)
    resp = asyncio.run(oapp.list_containers(_request("/v1/containers")))
    entries = _json(resp)["containers"]
    ncf = next(e for e in entries if e["name"] == "ncf-deal-manager")
    assert ncf["inferenceStatus"] == expected


# ---------------------------------------------------------------------------
# The signal stays behind authentication
# ---------------------------------------------------------------------------

def test_health_ready_is_unchanged_in_learner_mode(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv(deployment_mode.MODE_ENV, "learner")
    for path in ("/health/ready", "/api/health/ready"):
        resp = _call(path)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_gpu_status_requires_a_token(monkeypatch):
    from orchestrator import auth

    if not auth._SIGNATURE_VERIFIER_AVAILABLE:
        pytest.skip("JWT signature verifier not installed; the middleware answers 503")
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_TestPool")
    monkeypatch.setenv(deployment_mode.MODE_ENV, "learner")
    for path in ("/v1/gpu/status", "/api/v1/gpu/status"):
        resp = _call(path)
        assert resp.status_code == 401
        assert "deploymentMode" not in resp.json()
