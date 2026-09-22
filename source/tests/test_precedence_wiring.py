"""Precedence wiring in the orchestrator — both response paths and the endpoints.

The resolver itself is tested in ``test_mutation_precedence.py``. This file tests
that it is actually connected, on **both** response paths, and that connecting it
did not change the response for a deployment with no priorities set.

``test_no_priorities_response_is_unchanged`` is the one that matters most: it is
the wiring-level form of the guarantee that attaching a container changes nothing
until an operator sets a priority.

Transport tests use ``httpx.ASGITransport`` inside a single ``asyncio.run``, never
starlette's threaded ``TestClient`` — a threaded client gives each thread its own
portal and will pass against code that is broken under real concurrency.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.container_registry import (  # noqa: E402
    STATUS_OK,
    merge_registry,
)
from shared.artf_types import ContainerInvocationModel, Mutation  # noqa: E402

FLOOR_INTENT = 4
METRICS_INTENT = 7
REPLACE_OP = 2
DEAL_PATH = "/imp/imp-1/deals/deal-a"

#: Two code-defined containers, the second of which claims the floor intent —
#: mirroring the real `yield-optimizer-floor`.
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


def _mutation(path: str = DEAL_PATH, intent: int = FLOOR_INTENT) -> Mutation:
    return Mutation(intent=intent, op=REPLACE_OP, path=path)


def _record(name, *, intents=("ADJUST_DEAL_FLOOR",), active=True, priority=None):
    record = {
        "registry": "artf-containers",
        "name": name,
        "endpoint": f"http://{name}:8081",
        "intents": list(intents),
        "active": active,
    }
    if priority is not None:
        record["priority"] = priority
    return record


def _install(monkeypatch, entries, *, mutations_by_container=None):
    """Point the orchestrator at a fixed registry and a fake container call."""
    from orchestrator import app as oapp

    muts = mutations_by_container or {}

    async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
        name = container["name"]
        return ContainerInvocationModel(
            name=name,
            status=STATUS_OK,
            latency_ms=1.0,
            mutations=list(muts.get(name, [])),
            display_name=container.get("display_name", ""),
            model_version=f"{name}-v1",
        )

    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
    monkeypatch.setattr(oapp, "_call_container_timed", fake_call)
    return oapp


def _post_mutations(oapp, body):
    from starlette.applications import Starlette
    from starlette.routing import Route

    async def run():
        route_app = Starlette(routes=[Route("/v1/mutations", oapp.get_mutations, methods=["POST"])])
        transport = httpx.ASGITransport(app=route_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post("/v1/mutations", json=body)

    return asyncio.run(run())


REQUEST = {"id": "req-1", "tmax": 100, "bid_request": {"id": "br-1"}}


# ---------------------------------------------------------------------------
# The regression guarantee (plan step 6.5)
# ---------------------------------------------------------------------------

class TestNoPriorityMeansNoChange:
    def test_no_priorities_response_is_unchanged(self, monkeypatch):
        """Two containers contest the floor and neither has a priority.

        Before precedence existed, both mutations were returned and a consumer
        applied them in order, so the LAST one won and the first silently did
        nothing. Now only the last is returned. The floor a consumer ends up with
        is the same — which is the point — and the container that lost is still
        reported as having produced its mutation.
        """
        pytest.importorskip("grpc")
        external = _mutation()
        builtin = _mutation()
        entries, _ = merge_registry(CODE_CONTAINERS, [_record("contextual-yield-agent")])
        oapp = _install(
            monkeypatch,
            entries,
            mutations_by_container={
                "yield-optimizer-floor": [builtin],
                "contextual-yield-agent": [external],
            },
        )

        resp = _post_mutations(oapp, REQUEST)
        assert resp.status_code == 200
        body = resp.json()

        # One floor mutation returned, and it is the store container's — the same
        # one last-write-wins would have left in place.
        floors = [m for m in body["mutations"] if m["intent"] == FLOOR_INTENT]
        assert len(floors) == 1

        containers = {c["name"]: c for c in body["metadata"]["containers"]}
        # The built-in still reports its mutation: it really computed one.
        assert len(containers["yield-optimizer-floor"]["mutations"]) == 1
        assert containers["yield-optimizer-floor"]["superseded"] == 1
        assert containers["contextual-yield-agent"]["superseded"] == 0

        conflict = body["metadata"]["conflicts"][0]
        assert conflict["winner"] == "contextual-yield-agent"
        assert conflict["losers"] == ["yield-optimizer-floor"]

    def test_uncontested_mutations_are_all_returned(self, monkeypatch):
        """No contest means no filtering and no conflicts block."""
        pytest.importorskip("grpc")
        entries, _ = merge_registry(
            CODE_CONTAINERS, [_record("metrics-ext", intents=["ADD_METRICS"])]
        )
        oapp = _install(
            monkeypatch,
            entries,
            mutations_by_container={
                "segment-activator": [_mutation("/user/data/segment", 1)],
                "yield-optimizer-floor": [_mutation()],
                "metrics-ext": [_mutation("/imp/imp-1/metric", METRICS_INTENT)],
            },
        )

        body = _post_mutations(oapp, REQUEST).json()
        assert len(body["mutations"]) == 3
        assert body["metadata"]["conflicts"] is None
        for c in body["metadata"]["containers"]:
            assert c["superseded"] == 0


# ---------------------------------------------------------------------------
# Priority actually changes the outcome
# ---------------------------------------------------------------------------

class TestPriorityChangesTheOutcome:
    def test_priority_lets_the_builtin_win(self, monkeypatch):
        """A negative priority on the attached container hands the floor back.

        The built-in cannot be deactivated, so this is the only way to keep its
        floor once an external claimant is active.
        """
        pytest.importorskip("grpc")
        entries, _ = merge_registry(
            CODE_CONTAINERS, [_record("contextual-yield-agent", priority=-1)]
        )
        oapp = _install(
            monkeypatch,
            entries,
            mutations_by_container={
                "yield-optimizer-floor": [_mutation()],
                "contextual-yield-agent": [_mutation()],
            },
        )

        body = _post_mutations(oapp, REQUEST).json()
        conflict = body["metadata"]["conflicts"][0]
        assert conflict["winner"] == "yield-optimizer-floor"
        assert conflict["losers"] == ["contextual-yield-agent"]

        containers = {c["name"]: c for c in body["metadata"]["containers"]}
        assert containers["contextual-yield-agent"]["superseded"] == 1
        assert containers["yield-optimizer-floor"]["superseded"] == 0

    def test_positive_priority_keeps_the_external_winning(self, monkeypatch):
        pytest.importorskip("grpc")
        entries, _ = merge_registry(
            CODE_CONTAINERS, [_record("contextual-yield-agent", priority=10)]
        )
        oapp = _install(
            monkeypatch,
            entries,
            mutations_by_container={
                "yield-optimizer-floor": [_mutation()],
                "contextual-yield-agent": [_mutation()],
            },
        )
        body = _post_mutations(oapp, REQUEST).json()
        assert body["metadata"]["conflicts"][0]["winner"] == "contextual-yield-agent"

    def test_unreachable_external_leaves_the_builtin_intact(self, monkeypatch):
        """An external container that produced nothing cannot displace anything."""
        pytest.importorskip("grpc")
        entries, _ = merge_registry(
            CODE_CONTAINERS, [_record("contextual-yield-agent", priority=10)]
        )
        oapp = _install(
            monkeypatch,
            entries,
            mutations_by_container={
                "yield-optimizer-floor": [_mutation()],
                "contextual-yield-agent": [],
            },
        )
        body = _post_mutations(oapp, REQUEST).json()
        floors = [m for m in body["mutations"] if m["intent"] == FLOOR_INTENT]
        assert len(floors) == 1
        assert body["metadata"]["conflicts"] is None


# ---------------------------------------------------------------------------
# Both response paths (plan step 6.4) — these have drifted before
# ---------------------------------------------------------------------------

class TestBothPathsResolve:
    def test_mcp_tools_call_path_resolves_too(self, monkeypatch):
        """The MCP proxy's tools/call must resolve identically to the REST path.

        The fan-out was unified because these two drifted. Resolution goes through
        the same helper for the same reason, and this asserts it.
        """
        pytest.importorskip("grpc")
        from starlette.applications import Starlette
        from starlette.routing import Route

        entries, _ = merge_registry(CODE_CONTAINERS, [_record("contextual-yield-agent")])
        oapp = _install(
            monkeypatch,
            entries,
            mutations_by_container={
                "yield-optimizer-floor": [_mutation()],
                "contextual-yield-agent": [_mutation()],
            },
        )

        async def run():
            route_app = Starlette(routes=[Route("/mcp", oapp.mcp_proxy, methods=["POST"])])
            transport = httpx.ASGITransport(app=route_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                return await client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "extend_rtb", "arguments": REQUEST},
                    },
                )

        resp = asyncio.run(run())
        assert resp.status_code == 200
        payload = resp.json()

        # The tool result carries the RTBResponse as JSON text.
        text = payload["result"]["content"][0]["text"]
        rtb = json.loads(text)
        floors = [m for m in rtb["mutations"] if m["intent"] == FLOOR_INTENT]
        assert len(floors) == 1, "MCP path did not resolve the contest"
        assert rtb["metadata"]["conflicts"][0]["winner"] == "contextual-yield-agent"


# ---------------------------------------------------------------------------
# priority on GET /v1/containers (plan step 6.3)
# ---------------------------------------------------------------------------

class TestPriorityIsReported:
    def test_entries_report_priority(self, monkeypatch):
        """``GET /v1/containers`` carries ``priority`` on active and inactive alike.

        The active entries here point at ``127.0.0.1:1``, which refuses instantly.
        ``list_containers`` really probes its active containers and its probe
        helpers are inner functions that cannot be patched from outside, so a
        hostname that does not resolve makes this test hang on DNS rather than
        fail. A refused port exercises the same code path in milliseconds.
        """
        pytest.importorskip("grpc")
        from starlette.applications import Starlette
        from starlette.routing import Route

        from orchestrator import app as oapp

        unreachable_code = [
            {**c, "mcp": "http://127.0.0.1:1", "grpc": "127.0.0.1:1"}
            for c in CODE_CONTAINERS
        ]
        entries, _ = merge_registry(
            unreachable_code, [_record("ext", active=False, priority=7)]
        )
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))

        async def run():
            route_app = Starlette(routes=[Route("/v1/containers", oapp.list_containers)])
            transport = httpx.ASGITransport(app=route_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                return await client.get("/v1/containers")

        body = asyncio.run(run()).json()
        containers = {c["name"]: c for c in body["containers"]}
        assert containers["ext"]["priority"] == 7
        # Built-ins are pinned at 0 and are not configurable.
        assert containers["yield-optimizer-floor"]["priority"] == 0
        assert containers["yield-optimizer-floor"]["configurable"] is False


# ---------------------------------------------------------------------------
# Activation consequence (plan step 6.4 / FR-11)
# ---------------------------------------------------------------------------

class TestActivationConsequence:
    def test_contest_is_described_when_an_intent_is_shared(self, monkeypatch):
        from orchestrator import app as oapp

        entries, _ = merge_registry(CODE_CONTAINERS, [_record("contextual-yield-agent")])
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))

        contest = oapp._describe_intent_contest("contextual-yield-agent")
        assert contest is not None
        entry = contest["contested"][0]
        assert entry["intent"] == "ADJUST_DEAL_FLOOR"
        assert entry["winner"] == "contextual-yield-agent"
        assert {c["name"] for c in entry["claimants"]} == {
            "contextual-yield-agent",
            "yield-optimizer-floor",
        }

    def test_contest_names_the_builtin_as_winner_when_it_outranks(self, monkeypatch):
        from orchestrator import app as oapp

        entries, _ = merge_registry(
            CODE_CONTAINERS, [_record("contextual-yield-agent", priority=-5)]
        )
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))

        contest = oapp._describe_intent_contest("contextual-yield-agent")
        assert contest["contested"][0]["winner"] == "yield-optimizer-floor"

    def test_no_contest_when_intent_is_unique(self, monkeypatch):
        from orchestrator import app as oapp

        entries, _ = merge_registry(
            CODE_CONTAINERS, [_record("ext", intents=["ADD_CIDS"])]
        )
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
        assert oapp._describe_intent_contest("ext") is None

    def test_inactive_rival_is_not_a_contest(self, monkeypatch):
        """An inactive container is never called, so it cannot contest anything."""
        from orchestrator import app as oapp

        entries, _ = merge_registry(
            CODE_CONTAINERS,
            [
                _record("a", intents=["ADD_CIDS"], active=True),
                _record("b", intents=["ADD_CIDS"], active=False),
            ],
        )
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
        assert oapp._describe_intent_contest("a") is None

    def test_unknown_container_yields_no_contest(self, monkeypatch):
        from orchestrator import app as oapp

        entries, _ = merge_registry(CODE_CONTAINERS, [])
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
        assert oapp._describe_intent_contest("nope") is None


# ---------------------------------------------------------------------------
# _priorities
# ---------------------------------------------------------------------------

def test_priorities_maps_every_entry(monkeypatch):
    from orchestrator import app as oapp

    entries, _ = merge_registry(CODE_CONTAINERS, [_record("ext", priority=3)])
    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))

    priorities = oapp._priorities()
    assert priorities["ext"] == 3
    assert priorities["yield-optimizer-floor"] == 0
    assert priorities["segment-activator"] == 0
