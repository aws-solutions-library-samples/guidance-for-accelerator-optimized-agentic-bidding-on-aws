"""The per-request ARTF bypass on ``/v1/mutations``.

The Theater's baseline pass runs a real auction in which the ARTF extension point
proposes nothing. The auction endpoint stamps ``ext.artf.bypass: true`` on the bid
request it forwards to Prebid (top-level ext -- Prebid drops unknown keys under
``ext.prebid``, which is where a first version put it and lost it); Prebid's hook
forwards the parsed request back here inside its envelope; and this endpoint must
then answer with an empty
mutation set WITHOUT consulting any container and WITHOUT emitting a feedback
event.

Two things these tests protect. First, the bypass is exact: only the boolean
``True`` at exactly that path is a bypass, so no stray value can run an auction
without ARTF by accident. Second, the ordinary path is unchanged: a request without
the marker fans out to every container it did before.

Same transport discipline as ``test_precedence_wiring.py``: ``httpx.ASGITransport``
inside one ``asyncio.run``, never the threaded ``TestClient``.
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.container_registry import STATUS_OK, merge_registry  # noqa: E402
from shared.artf_types import (  # noqa: E402
    ContainerInvocationModel,
    Mutation,
    is_artf_bypass,
)

FLOOR_INTENT = 4
REPLACE_OP = 2

CODE_CONTAINERS = [
    {
        "name": "segment-activator",
        "intents": {"ACTIVATE_SEGMENTS"},
        "mcp": "http://segment:8081",
        "display_name": "Audience Activator",
    },
    {
        "name": "yield-optimizer-floor",
        "intents": {"ADJUST_DEAL_FLOOR"},
        "mcp": "http://floor:8081",
        "display_name": "Yield Optimizer",
    },
]


class Recorder:
    """Counts container calls and feedback emissions, so absence is provable."""

    def __init__(self):
        self.container_calls: list[str] = []
        self.bid_outcomes = 0
        self.deal_outcomes = 0


def _install(monkeypatch) -> tuple:
    from orchestrator import app as oapp

    recorder = Recorder()
    entries, _ = merge_registry(CODE_CONTAINERS, [])

    async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
        name = container["name"]
        recorder.container_calls.append(name)
        return ContainerInvocationModel(
            name=name,
            status=STATUS_OK,
            latency_ms=1.0,
            mutations=[Mutation(intent=FLOOR_INTENT, op=REPLACE_OP, path="/imp/imp-1/deals/d")]
            if name == "yield-optimizer-floor"
            else [],
            display_name=container.get("display_name", ""),
            model_version=f"{name}-v1",
        )

    def fake_bid_outcome(*args, **kwargs):
        recorder.bid_outcomes += 1

    def fake_deal_outcome(*args, **kwargs):
        recorder.deal_outcomes += 1

    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
    monkeypatch.setattr(oapp, "_call_container_timed", fake_call)
    monkeypatch.setattr(oapp, "emit_bid_outcome", fake_bid_outcome)
    monkeypatch.setattr(oapp, "emit_deal_yield_outcome", fake_deal_outcome)
    return oapp, recorder


def _post_mutations(oapp, body):
    from starlette.applications import Starlette
    from starlette.routing import Route

    async def run():
        route_app = Starlette(routes=[Route("/v1/mutations", oapp.get_mutations, methods=["POST"])])
        transport = httpx.ASGITransport(app=route_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post("/v1/mutations", json=body)

    return asyncio.run(run())


def _envelope(bid_request: dict) -> dict:
    """What the Prebid hook sends: the parsed bid request inside the ARTF envelope."""
    return {
        "id": "req-1",
        "tmax": 100,
        "lifecycle": "LIFECYCLE_PUBLISHER_BID_REQUEST",
        "originator": {"type": "TYPE_EXCHANGE", "name": "prebid-server"},
        "bid_request": bid_request,
    }


BYPASSED = {"id": "br-1", "imp": [{"id": "imp-1"}], "ext": {"artf": {"bypass": True}}}
ORDINARY = {"id": "br-1", "imp": [{"id": "imp-1"}], "ext": {"artf": {}}}


# ------------------------------------------------------------- the predicate

@pytest.mark.parametrize(
    "bid_request",
    [
        ORDINARY,
        {"id": "br-1"},
        {"id": "br-1", "ext": None},
        {"id": "br-1", "ext": {"artf": None}},
        # The value has to be the boolean. Each of these is "something that looks
        # like a yes" and none of them is the marker.
        {"id": "br-1", "ext": {"artf": {"bypass": "true"}}},
        {"id": "br-1", "ext": {"artf": {"bypass": 1}}},
        {"id": "br-1", "ext": {"artf": {"bypass": "off"}}},
        {"id": "br-1", "ext": {"artf": {"bypass": False}}},
        # Right key, wrong place. `ext.prebid.artf` is where the first version put
        # it; Prebid drops that key, so a request carrying it there must NOT be
        # treated as a bypass -- that would be the orchestrator agreeing with a
        # marker Prebid itself never forwards.
        {"id": "br-1", "ext": {"prebid": {"artf": {"bypass": True}}}},
        {"id": "br-1", "ext": {"bypass": True}},
        "not a dict",
        None,
    ],
)
def test_only_the_exact_marker_is_a_bypass(bid_request):
    assert is_artf_bypass(bid_request) is False


def test_the_exact_marker_is_a_bypass():
    assert is_artf_bypass(BYPASSED) is True


# ------------------------------------------------------------- the endpoint

def test_a_bypassed_request_consults_no_container_and_returns_no_mutation(monkeypatch):
    pytest.importorskip("grpc")
    oapp, recorder = _install(monkeypatch)

    response = _post_mutations(oapp, _envelope(BYPASSED))

    assert response.status_code == 200
    payload = response.json()
    assert payload["mutations"] == []
    # Not "skipped", not "disabled": never asked. The containers list is empty
    # because no invocation happened, which is a different fact from two
    # invocations that each produced nothing.
    assert payload["metadata"]["containers"] == []
    assert payload["metadata"]["bypassed"] is True
    assert "bypassed" in payload["metadata"]["model_version"]
    assert recorder.container_calls == []


def test_a_bypassed_request_emits_no_feedback_event(monkeypatch):
    pytest.importorskip("grpc")
    oapp, recorder = _install(monkeypatch)

    _post_mutations(oapp, _envelope(BYPASSED))

    # The feedback feed trains the shader on decisions it made. A pass in which it
    # was deliberately not asked carries none.
    assert recorder.bid_outcomes == 0
    assert recorder.deal_outcomes == 0


def test_an_ordinary_request_still_fans_out_and_emits(monkeypatch):
    pytest.importorskip("grpc")
    oapp, recorder = _install(monkeypatch)

    response = _post_mutations(oapp, _envelope(ORDINARY))

    payload = response.json()
    assert sorted(recorder.container_calls) == ["segment-activator", "yield-optimizer-floor"]
    assert len(payload["mutations"]) == 1
    assert "bypassed" not in payload["metadata"]
    assert "bypassed" not in payload["metadata"]["model_version"]
    assert recorder.bid_outcomes == 1
    assert recorder.deal_outcomes == 1


def test_the_marker_is_read_from_the_bid_request_not_the_envelope(monkeypatch):
    # A `bypass` key on the ARTF envelope itself is not the marker: the hook builds
    # the envelope, and the only field the auction endpoint can influence is the
    # bid request Prebid carries through.
    pytest.importorskip("grpc")
    oapp, recorder = _install(monkeypatch)

    body = _envelope(ORDINARY)
    body["bypass"] = True
    body["ext"] = {"bypass": True}

    response = _post_mutations(oapp, body)

    assert len(recorder.container_calls) == 2
    assert "bypassed" not in response.json()["metadata"]
