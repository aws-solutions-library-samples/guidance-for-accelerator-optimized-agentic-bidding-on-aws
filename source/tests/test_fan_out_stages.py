"""The staged fan-out: orchestrator/app._fan_out_staged.

What these pin:

- Containers are called in stage order (enrich, deals, yield, price), and the
  containers of one stage are called together.
- A stage sees the request the previous stage produced: the Deal Scorer's
  payload carries the segments the Audience Activator activated; the Yield
  Optimizer's payload carries the deal the Deal Scorer activated, at the floor
  the Deal Scorer attached.
- A stage running out of budget does not cancel the next stage.
- A request naming two stages' intents runs two stages; an empty
  applicable_intents runs every stage that has a container.
- The bypassed pass (ext.artf.bypass: true) consults no container at all, which
  is the Theater's baseline and must survive staging untouched.
- metadata.stages reports what ran, in order, with per-stage timing.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("grpc")

from orchestrator import app as oapp  # noqa: E402
from orchestrator.container_registry import (  # noqa: E402
    STATUS_NO_MUTATIONS,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
    merge_registry,
)
from shared.artf_types import (  # noqa: E402
    AddMetricsPayload,
    AdjustDealPayload,
    ContainerInvocationModel,
    IDsPayload,
    Intent,
    Metric,
    Mutation,
    Operation,
)

ENVELOPE = {
    "id": "home-001",
    "tmax": 120,
    "applicable_intents": [],
    "bid_request": {
        "id": "auction-home",
        "imp": [{
            "id": "imp-1",
            "banner": {"w": 300, "h": 250},
            "bidfloor": 2.5,
            "bidfloorcur": "USD",
            "pmp": {"private_auction": 0, "deals": [{"id": "deal-remnant-open", "bidfloor": 1.0, "at": 3}]},
        }],
        "site": {"domain": "hearth-and-home.example", "cat": ["283"]},
        "user": {"id": "user-home-01", "data": [{"id": "dmp", "segment": [{"id": "seg-461"}]}]},
    },
}


def _entries():
    entries, _ = merge_registry(oapp.CONTAINERS, [])
    return entries


class Recorder:
    """A fake _call_container_timed that records what each container was sent."""

    def __init__(self, behaviour):
        self.behaviour = behaviour  # name -> callable(payload) -> ContainerInvocationModel | list[Mutation]
        self.seen: list[tuple[str, dict]] = []
        self.batches: list[list[str]] = []
        self._current: list[str] = []

    async def __call__(self, client, container, payload, payload_bytes, timeout_s, headers=None):
        name = container["name"]
        self.seen.append((name, json.loads(payload_bytes)))
        self._current.append(name)
        # Yield once so every container in the gather registers before any returns,
        # which is what lets the batch boundary be observed.
        await asyncio.sleep(0)
        if self._current:
            self.batches.append(sorted(self._current))
            self._current = []
        result = self.behaviour.get(name)
        if callable(result):
            result = result(payload, timeout_s)
        if isinstance(result, ContainerInvocationModel):
            return result
        mutations = result or []
        return ContainerInvocationModel(
            name=name,
            status=STATUS_OK if mutations else STATUS_NO_MUTATIONS,
            latency_ms=1.0,
            mutations=mutations,
            display_name=container.get("display_name", ""),
        )


def _segments(ids):
    return [Mutation(intent=Intent.ACTIVATE_SEGMENTS, op=Operation.ADD, path="/user/data/segment", ids=IDsPayload(id=ids))]


def _metrics():
    return [Mutation(intent=Intent.ADD_METRICS, op=Operation.ADD, path="/imp/imp-1/metric",
                     add_metrics=AddMetricsPayload(metric=[Metric(type="viewability", value=0.8, vendor="v")]))]


def _activate_with_floor(deal, floor):
    return [
        Mutation(intent=Intent.ACTIVATE_DEALS, op=Operation.ADD, path="/imp/imp-1", ids=IDsPayload(id=[deal])),
        Mutation(intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE, path=f"/imp/imp-1/deals/{deal}",
                 adjust_deal=AdjustDealPayload(bidfloor=floor)),
    ]


def _floor(deal, value):
    return [Mutation(intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE, path=f"/imp/imp-1/deals/{deal}",
                     adjust_deal=AdjustDealPayload(bidfloor=value))]


def _run(monkeypatch, recorder, envelope=ENVELOPE, applicable=None, timeout_s=1.0):
    monkeypatch.setattr(oapp, "_effective_registry", lambda: (_entries(), []))
    monkeypatch.setattr(oapp, "_call_container_timed", recorder)
    return asyncio.run(oapp._fan_out_staged(
        envelope, json.dumps(envelope).encode(), applicable, timeout_s=timeout_s,
    ))


def test_containers_are_called_in_stage_order_and_each_stage_together(monkeypatch):
    rec = Recorder({})
    result = _run(monkeypatch, rec)
    called = [name for name, _ in rec.seen]
    assert called == [
        "widedeep-segment-activator", "metrics-enricher",   # stage 1, registry order within
        "ncf-deal-manager",                                   # stage 2
        "yield-optimizer-floor", "yield-optimizer-margin",    # stage 3
        "dlrm-bid-shader",                                    # stage 4
    ]
    # The batches observed at the gather boundary are the stages.
    assert rec.batches[0] == ["metrics-enricher", "widedeep-segment-activator"]
    assert [s.name for s in result.stages] == ["enrich", "deals", "yield", "price"]
    assert [s.containers for s in result.stages] == [
        ["widedeep-segment-activator", "metrics-enricher"],
        ["ncf-deal-manager"],
        ["yield-optimizer-floor", "yield-optimizer-margin"],
        ["dlrm-bid-shader"],
    ]
    assert [i.name for i in result.invocations] == called


def test_the_deal_scorer_sees_the_segments_the_audience_activator_activated(monkeypatch):
    rec = Recorder({"widedeep-segment-activator": _segments(["461", "460"])})
    _run(monkeypatch, rec)
    ncf_payload = dict(rec.seen)["ncf-deal-manager"]
    data = ncf_payload["bid_request"]["user"]["data"]
    assert data[0]["id"] == "dmp", "the publisher's own segments are kept"
    assert data[1] == {"name": "artf", "segment": [{"id": "461"}, {"id": "460"}]}
    # And the stage-1 containers did NOT see them: they ran on the original.
    assert dict(rec.seen)["widedeep-segment-activator"]["bid_request"]["user"]["data"] == ENVELOPE["bid_request"]["user"]["data"]


def test_the_yield_optimizer_sees_the_deal_the_deal_scorer_activated_at_its_library_floor(monkeypatch):
    rec = Recorder({"ncf-deal-manager": _activate_with_floor("deal-home-premium", 3.0)})
    result = _run(monkeypatch, rec)
    for name in ("yield-optimizer-floor", "yield-optimizer-margin"):
        deals = dict(rec.seen)[name]["bid_request"]["imp"][0]["pmp"]["deals"]
        assert [d["id"] for d in deals] == ["deal-remnant-open", "deal-home-premium"]
        assert deals[1]["bidfloor"] == 3.0 and deals[1]["bidfloorcur"] == "USD"
        # imp.bidfloor was raised to the activated deal's floor, so the yield
        # containers price against the floor the host will enforce.
        assert dict(rec.seen)[name]["bid_request"]["imp"][0]["bidfloor"] == 3.0
    deals_stage = result.stages[1]
    assert deals_stage.name == "deals" and deals_stage.applied == 2 and deals_stage.rejected == []
    # The Deal Scorer itself saw the original request (one deal).
    assert len(dict(rec.seen)["ncf-deal-manager"]["bid_request"]["imp"][0]["pmp"]["deals"]) == 1


def test_the_bid_pricer_sees_the_floor_the_yield_optimizer_raised(monkeypatch):
    rec = Recorder({"yield-optimizer-floor": _floor("deal-remnant-open", 2.9)})
    _run(monkeypatch, rec)
    shader_imp = dict(rec.seen)["dlrm-bid-shader"]["bid_request"]["imp"][0]
    assert shader_imp["pmp"]["deals"][0]["bidfloor"] == 2.9
    assert shader_imp["bidfloor"] == 2.9, "raised from 2.5"


def test_a_mutation_the_orchestrator_cannot_apply_is_reported_on_the_stage_and_still_returned(monkeypatch):
    # A floor for a deal that is not on the request: the host will reject it too.
    rec = Recorder({"yield-optimizer-floor": _floor("deal-not-here", 9.0)})
    result = _run(monkeypatch, rec)
    yield_stage = next(s for s in result.stages if s.name == "yield")
    assert yield_stage.applied == 0
    assert len(yield_stage.rejected) == 1
    assert yield_stage.rejected[0].container == "yield-optimizer-floor"
    assert "has no deal 'deal-not-here'" in yield_stage.rejected[0].reason
    # The container's own mutation list is untouched: it did compute it.
    inv = next(i for i in result.invocations if i.name == "yield-optimizer-floor")
    assert len(inv.mutations) == 1


def test_a_stage_that_times_out_does_not_cancel_the_next_stage(monkeypatch):
    def timed_out(payload, timeout_s):
        return ContainerInvocationModel(name="ncf-deal-manager", status=STATUS_TIMEOUT, latency_ms=timeout_s * 1000, mutations=[])

    rec = Recorder({"ncf-deal-manager": timed_out})
    result = _run(monkeypatch, rec, timeout_s=0.05)
    called = [name for name, _ in rec.seen]
    assert "yield-optimizer-floor" in called and "dlrm-bid-shader" in called
    by_name = {i.name: i for i in result.invocations}
    assert by_name["ncf-deal-manager"].status == STATUS_TIMEOUT
    assert len(result.stages) == 4


def test_a_later_stage_is_never_given_less_than_the_floor(monkeypatch):
    budgets: dict[str, float] = {}

    def record_budget(name):
        def inner(payload, timeout_s):
            budgets[name] = timeout_s
            return []
        return inner

    rec = Recorder({n: record_budget(n) for n in
                    ("widedeep-segment-activator", "metrics-enricher", "ncf-deal-manager",
                     "yield-optimizer-floor", "yield-optimizer-margin", "dlrm-bid-shader")})
    # A budget already spent before the first stage: every stage still gets the floor.
    _run(monkeypatch, rec, timeout_s=0.0)
    assert all(b >= oapp.STAGE_MIN_TIMEOUT_S for b in budgets.values())
    assert budgets["dlrm-bid-shader"] == pytest.approx(oapp.STAGE_MIN_TIMEOUT_S)


def test_each_stage_gets_what_remains_of_the_budget(monkeypatch):
    budgets: dict[str, float] = {}

    def record_budget(name):
        def inner(payload, timeout_s):
            budgets[name] = timeout_s
            return []
        return inner

    rec = Recorder({n: record_budget(n) for n in
                    ("widedeep-segment-activator", "metrics-enricher", "ncf-deal-manager",
                     "yield-optimizer-floor", "yield-optimizer-margin", "dlrm-bid-shader")})
    _run(monkeypatch, rec, timeout_s=1.0)
    assert budgets["widedeep-segment-activator"] == pytest.approx(1.0, abs=0.05)
    assert budgets["widedeep-segment-activator"] >= budgets["ncf-deal-manager"] >= budgets["dlrm-bid-shader"]


def test_applicable_intents_naming_two_stages_runs_two_stages(monkeypatch):
    rec = Recorder({})
    result = _run(monkeypatch, rec, applicable=["ACTIVATE_SEGMENTS", "ACTIVATE_DEALS"])
    assert [name for name, _ in rec.seen] == ["widedeep-segment-activator", "ncf-deal-manager"]
    assert [s.name for s in result.stages] == ["enrich", "deals"]
    by_name = {i.name: i for i in result.invocations}
    assert by_name["dlrm-bid-shader"].status == STATUS_SKIPPED
    assert by_name["yield-optimizer-floor"].status == STATUS_SKIPPED
    # Not-called entries come after the called ones, in registry order.
    assert [i.name for i in result.invocations][2:] == [
        "dlrm-bid-shader", "metrics-enricher", "yield-optimizer-floor", "yield-optimizer-margin",
    ]


def test_an_attached_container_runs_in_the_stage_of_its_earliest_intent(monkeypatch):
    from orchestrator.container_registry import REGISTRY_PARTITION
    record = {
        "registry": REGISTRY_PARTITION, "name": "partner-deals", "intents": ["SUPPRESS_DEALS"],
        "endpoint": "http://partner:8081", "active": True,
    }
    entries, _ = merge_registry(oapp.CONTAINERS, [record])
    rec = Recorder({})
    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
    monkeypatch.setattr(oapp, "_call_container_timed", rec)
    result = asyncio.run(oapp._fan_out_staged(ENVELOPE, json.dumps(ENVELOPE).encode(), None, timeout_s=1.0))
    deals_stage = next(s for s in result.stages if s.name == "deals")
    assert deals_stage.containers == ["ncf-deal-manager", "partner-deals"]


def test_the_flattened_response_mutations_are_in_stage_order(monkeypatch):
    rec = Recorder({
        "dlrm-bid-shader": [],
        "widedeep-segment-activator": _segments(["461"]),
        "ncf-deal-manager": _activate_with_floor("deal-home-premium", 3.0),
        "metrics-enricher": _metrics(),
        "yield-optimizer-floor": _floor("deal-home-premium", 3.2),
    })
    result = _run(monkeypatch, rec)
    mutations, _conflicts = oapp._resolve_for_response(result.invocations)
    intents = [Intent(m.intent) for m in mutations]
    # Enrichment first, then the activation, then the floor; the Deal Scorer's own
    # floor for the activated deal lost the (path, intent) contest to the Yield
    # Optimizer's, which is applied last and therefore wins.
    assert intents == [Intent.ACTIVATE_SEGMENTS, Intent.ADD_METRICS, Intent.ACTIVATE_DEALS, Intent.ADJUST_DEAL_FLOOR]
    assert mutations[-1].adjust_deal.bidfloor == 3.2
    assert result.bid_request["imp"][0]["pmp"]["deals"][1]["bidfloor"] == 3.2


def test_a_bypassed_request_consults_no_container(monkeypatch):
    """The Theater's baseline pass. Staging lives inside the fan-out; the bypass
    decision is made before it, so this must be unchanged."""
    rec = Recorder({})
    monkeypatch.setattr(oapp, "_effective_registry", lambda: (_entries(), []))
    monkeypatch.setattr(oapp, "_call_container_timed", rec)
    monkeypatch.setattr(oapp, "emit_bid_outcome", lambda *a, **k: None)
    monkeypatch.setattr(oapp, "emit_deal_yield_outcome", lambda *a, **k: None)
    envelope = json.loads(json.dumps(ENVELOPE))
    envelope["bid_request"]["ext"] = {"artf": {"bypass": True}}
    resp = asyncio.run(oapp._mutations_response(
        envelope, handler_start=0.0, auth_ms=None, fabric_link_id=None,
    ))
    assert rec.seen == []
    assert resp["mutations"] == []
    assert resp["metadata"]["bypassed"] is True
    assert resp["metadata"]["stages"] is None


def test_metadata_stages_is_reported_on_an_ordinary_request(monkeypatch):
    rec = Recorder({"ncf-deal-manager": _activate_with_floor("deal-home-premium", 3.0)})
    monkeypatch.setattr(oapp, "_effective_registry", lambda: (_entries(), []))
    monkeypatch.setattr(oapp, "_call_container_timed", rec)
    monkeypatch.setattr(oapp, "emit_bid_outcome", lambda *a, **k: None)
    monkeypatch.setattr(oapp, "emit_deal_yield_outcome", lambda *a, **k: None)
    resp = asyncio.run(oapp._mutations_response(
        ENVELOPE, handler_start=0.0, auth_ms=None, fabric_link_id=None,
    ))
    stages = resp["metadata"]["stages"]
    assert [s["name"] for s in stages] == ["enrich", "deals", "yield", "price"]
    assert all(s["latency_ms"] >= 0 and s["budget_ms"] > 0 for s in stages)
    assert stages[1]["applied"] == 2
