"""Tests for the ARTF container registry, status vocabulary, and activation.

Covers the registry's business rules:

- BR-1..BR-7   ``merge_registry``
- BR-8..BR-11  ``select_active``
- §3.3         ``derive_status``
- §3.4         five property-based invariants
- §4.1         transport outcome classification
- §4.5         the TTL-cached store's never-raises / last-known-good contract
- §4.6         the activation endpoint's status codes

The transport and fan-out tests run on ``httpx.ASGITransport`` inside a single
``asyncio.run``, never starlette's threaded ``TestClient``. That is not a style
preference: this project previously had a concurrency test that PASSED against
broken code because ``TestClient`` gives each thread its own event-loop portal,
so nothing was ever actually concurrent.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from shared.artf_types import Intent, Mutation, Operation
from orchestrator.container_registry import (
    ERROR_STATUSES,
    NOT_CALLED_STATUSES,
    RAN_STATUSES,
    REGISTRY_PARTITION,
    SOURCE_CODE,
    SOURCE_STORE,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_NO_MUTATIONS,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNREACHABLE,
    ContainerCallOutcome,
    ContainerRegistryStore,
    RegistryRecordNotFound,
    RegistryStoreUnavailable,
    derive_status,
    merge_registry,
    select_active,
    shared_intents,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _code(name, intents, **extra):
    return {
        "name": name,
        "intents": set(intents),
        "grpc": f"{name}:8081",
        "mcp": f"http://{name}:8081",
        **extra,
    }


def _record(name, intents=("ADD_CIDS",), active=False, **extra):
    return {
        "registry": REGISTRY_PARTITION,
        "name": name,
        "intents": list(intents),
        "endpoint": f"http://{name}:8081",
        "active": active,
        **extra,
    }


CODE_TWO = [
    _code("dlrm-bid-shader", ["BID_SHADE"], display_name="Bid Pricer", description="Prices the bid."),
    _code("metrics-enricher", ["ADD_METRICS"]),
]


def _mutation():
    return Mutation(intent=Intent.ADD_CIDS, op=Operation.ADD, path="/imp/1/ext/cids")


# ---------------------------------------------------------------------------
# merge_registry — BR-1..BR-7
# ---------------------------------------------------------------------------

class TestMergeRegistry:
    def test_br1_every_code_container_appears_and_is_never_configurable(self):
        entries, _ = merge_registry(CODE_TWO, [])
        assert [e.name for e in entries] == ["dlrm-bid-shader", "metrics-enricher"]
        assert all(e.source == SOURCE_CODE for e in entries)
        assert all(e.configurable is False for e in entries)
        assert all(e.active is True for e in entries)

    def test_br2_code_containers_keep_declaration_order_and_come_first(self):
        entries, _ = merge_registry(CODE_TWO, [_record("aaa-sorts-first")])
        # "aaa-sorts-first" sorts before both code names, and still comes last:
        # response order is also the mutation attribution order.
        assert [e.name for e in entries] == [
            "dlrm-bid-shader", "metrics-enricher", "aaa-sorts-first",
        ]

    def test_br3_store_records_follow_sorted_by_name(self):
        entries, _ = merge_registry(
            CODE_TWO, [_record("zeta"), _record("alpha"), _record("mid")]
        )
        assert [e.name for e in entries][2:] == ["alpha", "mid", "zeta"]
        assert all(e.source == SOURCE_STORE and e.configurable for e in entries[2:])

    def test_br4_store_record_cannot_shadow_a_code_container(self):
        hijack = _record(
            "dlrm-bid-shader", intents=["ADD_CIDS"], active=False,
            display_name="HIJACKED", endpoint="http://evil:8081",
        )
        entries, warnings = merge_registry(CODE_TWO, [hijack])
        built_in = next(e for e in entries if e.name == "dlrm-bid-shader")
        assert built_in.display_name == "Bid Pricer"
        assert built_in.endpoint == "http://dlrm-bid-shader:8081"
        assert built_in.intents == frozenset({"BID_SHADE"})
        assert built_in.active is True
        assert built_in.configurable is False
        assert len(entries) == 2, "the colliding record must not be added as a 3rd entry"
        assert any("built-in" in w for w in warnings)

    def test_br5_record_without_a_name_is_ignored_with_a_warning(self):
        entries, warnings = merge_registry(CODE_TWO, [{"endpoint": "http://x:8081"}])
        assert len(entries) == 2
        assert any("no name" in w for w in warnings)

    def test_br5_record_without_an_endpoint_is_ignored_rather_than_guessed(self):
        entries, warnings = merge_registry(CODE_TWO, [{"name": "no-endpoint", "active": True}])
        assert [e.name for e in entries] == ["dlrm-bid-shader", "metrics-enricher"]
        assert any("no endpoint" in w for w in warnings)

    def test_duplicate_store_records_are_ignored_after_the_first(self):
        entries, warnings = merge_registry(
            CODE_TWO, [_record("dupe", active=True), _record("dupe", active=False)]
        )
        assert [e.name for e in entries].count("dupe") == 1
        assert any("duplicate" in w for w in warnings)

    def test_br6_intents_are_upper_cased_and_unknown_names_are_kept(self):
        entries, _ = merge_registry([], [_record("c", intents=["add_cids", " Weird_Intent "])])
        assert entries[0].intents == frozenset({"ADD_CIDS", "WEIRD_INTENT"})

    def test_br6_a_bare_string_intent_is_accepted_not_split_into_characters(self):
        entries, _ = merge_registry(
            [], [{"name": "c", "endpoint": "http://c:8081", "intents": "ADD_CIDS"}]
        )
        assert entries[0].intents == frozenset({"ADD_CIDS"})

    def test_br6_absent_intents_yields_an_empty_set_rather_than_raising(self):
        entries, _ = merge_registry([], [{"name": "c", "endpoint": "http://c:8081"}])
        assert entries[0].intents == frozenset()

    def test_br7_display_name_falls_back_to_name_and_description_to_empty(self):
        entries, _ = merge_registry([], [_record("plain")])
        assert entries[0].display_name == "plain"
        assert entries[0].description == ""

    def test_grpc_target_is_derived_from_the_endpoint_when_absent(self):
        entries, _ = merge_registry([], [_record("c")])
        # Host from the endpoint, port = the ARTF gRPC port (50051), matching
        # app.py _grpc_from_url. The endpoint's own port is the HTTP listener
        # and was never a gRPC one.
        assert entries[0].grpc == "c:50051"

    def test_as_container_dict_matches_the_legacy_transport_shape(self):
        entries, _ = merge_registry([], [_record("c", intents=["ADD_CIDS"])])
        d = entries[0].as_container_dict()
        # loadtest.py and three existing tests pass dicts of exactly this shape.
        assert d["name"] == "c"
        assert d["mcp"] == "http://c:8081"
        assert isinstance(d["intents"], set)
        assert d["display_name"] == "c"

    def test_no_store_records_reproduces_the_pre_feature_registry(self):
        entries, warnings = merge_registry(CODE_TWO, None)
        assert [e.name for e in entries] == [c["name"] for c in CODE_TWO]
        assert warnings == []


class TestSharedIntents:
    def test_reports_only_intents_with_more_than_one_claimant(self):
        entries, _ = merge_registry(CODE_TWO, [_record("mine", intents=["BID_SHADE"])])
        assert shared_intents(entries) == {"BID_SHADE": ["dlrm-bid-shader", "mine"]}

    def test_no_clash_reports_nothing(self):
        entries, _ = merge_registry(CODE_TWO, [_record("mine", intents=["ADD_CIDS"])])
        assert shared_intents(entries) == {}


# ---------------------------------------------------------------------------
# select_active — BR-8..BR-11
# ---------------------------------------------------------------------------

class TestSelectActive:
    def test_br8_inactive_is_checked_before_intent_match(self):
        """An inactive container must report `disabled`, never `skipped`.

        This is the distinction the whole feature turns on: "you switched it off"
        and "your intents did not match" are different facts. Here the intents
        ALSO do not match, so an implementation that filtered on intent first
        would label it `skipped` and lose the operator's decision.
        """
        entries, _ = merge_registry([], [_record("off", intents=["ADD_CIDS"], active=False)])
        to_call, not_called = select_active(entries, ["BID_SHADE"])
        assert to_call == []
        assert [r for _, r in not_called] == [STATUS_DISABLED]

    def test_br9_empty_applicable_intents_means_all_intents_apply(self):
        entries, _ = merge_registry(CODE_TWO, [])
        for empty in (None, []):
            to_call, not_called = select_active(entries, empty)
            assert len(to_call) == 2 and not_called == []

    def test_br10_intent_mismatch_is_skipped(self):
        entries, _ = merge_registry(CODE_TWO, [])
        to_call, not_called = select_active(entries, ["BID_SHADE"])
        assert [e.name for e in to_call] == ["dlrm-bid-shader"]
        assert [(e.name, r) for e, r in not_called] == [("metrics-enricher", STATUS_SKIPPED)]

    def test_br11_both_lists_follow_registry_order(self):
        entries, _ = merge_registry(
            CODE_TWO, [_record("zz", active=True), _record("aa", active=True)]
        )
        to_call, _ = select_active(entries, [])
        assert [e.name for e in to_call] == ["dlrm-bid-shader", "metrics-enricher", "aa", "zz"]

    def test_an_active_store_container_is_called(self):
        entries, _ = merge_registry(CODE_TWO, [_record("mine", active=True)])
        to_call, _ = select_active(entries, ["ADD_CIDS"])
        assert [e.name for e in to_call] == ["mine"]

    def test_intent_matching_is_case_insensitive_on_the_request_side(self):
        entries, _ = merge_registry([], [_record("mine", active=True)])
        to_call, _ = select_active(entries, ["add_cids"])
        assert [e.name for e in to_call] == ["mine"]

    def test_numeric_intents_do_not_crash_the_filter(self):
        """applicable_intents is typed `list[str | int]`, so ints must be tolerated."""
        entries, _ = merge_registry(CODE_TWO, [])
        to_call, not_called = select_active(entries, [6])
        # 6 stringifies to "6", which matches no intent NAME — so nothing is
        # called, and nothing raises. Documented behaviour, not an accident.
        assert to_call == []
        assert all(r == STATUS_SKIPPED for _, r in not_called)


# ---------------------------------------------------------------------------
# derive_status — §3.3
# ---------------------------------------------------------------------------

class TestDeriveStatus:
    def test_reached_with_mutations_is_ok(self):
        assert derive_status(ContainerCallOutcome(reached=True, mutations=[_mutation()])) == STATUS_OK

    def test_reached_without_mutations_is_no_mutations_not_ok(self):
        assert derive_status(ContainerCallOutcome(reached=True, mutations=[])) == STATUS_NO_MUTATIONS

    def test_not_reached_is_unreachable(self):
        assert derive_status(ContainerCallOutcome(reached=False)) == STATUS_UNREACHABLE

    def test_reached_with_an_error_is_error(self):
        assert derive_status(
            ContainerCallOutcome(reached=True, error="unparseable")
        ) == STATUS_ERROR

    def test_unreachable_wins_over_mutations(self):
        """A regression guard for the defect this vocabulary exists to fix.

        Nothing answered, so whatever else the outcome carries, the container was
        not reached. Before this change the equivalent case reported `ok`.
        """
        assert derive_status(
            ContainerCallOutcome(reached=False, mutations=[_mutation()], error="conn refused")
        ) == STATUS_UNREACHABLE

    def test_status_sets_are_disjoint_and_cover_the_vocabulary(self):
        assert RAN_STATUSES & ERROR_STATUSES == frozenset()
        assert RAN_STATUSES & NOT_CALLED_STATUSES == frozenset()
        assert ERROR_STATUSES & NOT_CALLED_STATUSES == frozenset()


# ---------------------------------------------------------------------------
# Property-based invariants — design §3.4
# ---------------------------------------------------------------------------

_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=12
).filter(lambda s: s.strip("-") != "")
_intent_names = st.sampled_from([i.name for i in Intent])

_store_records = st.lists(
    st.fixed_dictionaries({
        "name": _names,
        "endpoint": st.just("http://x:8081"),
        "intents": st.lists(_intent_names, max_size=3),
        "active": st.booleans(),
    }),
    max_size=6,
)


class TestRegistryProperties:
    @given(records=_store_records)
    @settings(max_examples=60, deadline=None)
    def test_every_code_container_always_survives_the_merge(self, records):
        """No store content can remove a built-in container from the registry."""
        entries, _ = merge_registry(CODE_TWO, records)
        names = [e.name for e in entries]
        for c in CODE_TWO:
            assert c["name"] in names
        assert names[:2] == [c["name"] for c in CODE_TWO]

    @given(records=_store_records)
    @settings(max_examples=60, deadline=None)
    def test_no_store_record_can_alter_a_code_entry(self, records):
        """BR-4 as an invariant, not just the one hand-built collision case."""
        baseline, _ = merge_registry(CODE_TWO, [])
        entries, _ = merge_registry(CODE_TWO, records)
        by_name = {e.name: e for e in entries}
        for b in baseline:
            got = by_name[b.name]
            assert (got.name, got.intents, got.endpoint, got.active, got.configurable) == \
                   (b.name, b.intents, b.endpoint, b.active, b.configurable)

    @given(records=_store_records, requested=st.lists(_intent_names, max_size=3))
    @settings(max_examples=60, deadline=None)
    def test_select_active_partitions_exactly(self, records, requested):
        """Every entry lands in exactly one bucket, with no duplicates or losses."""
        entries, _ = merge_registry(CODE_TWO, records)
        to_call, not_called = select_active(entries, requested)
        assert len(to_call) + len(not_called) == len(entries)
        called = [e.name for e in to_call]
        uncalled = [e.name for e, _ in not_called]
        assert set(called) & set(uncalled) == set()
        assert sorted(called + uncalled) == sorted(e.name for e in entries)
        assert all(r in NOT_CALLED_STATUSES for _, r in not_called)

    @given(records=_store_records, requested=st.lists(_intent_names, max_size=3))
    @settings(max_examples=60, deadline=None)
    def test_an_inactive_entry_is_never_selected_for_a_call(self, records, requested):
        entries, _ = merge_registry(CODE_TWO, records)
        to_call, not_called = select_active(entries, requested)
        assert all(e.active for e in to_call)
        for entry, reason in not_called:
            if not entry.active:
                assert reason == STATUS_DISABLED

    @given(
        reached=st.booleans(),
        has_mutations=st.booleans(),
        error=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    )
    @settings(max_examples=80, deadline=None)
    def test_derive_status_is_total_and_never_claims_ok_for_unreachable(
        self, reached, has_mutations, error
    ):
        outcome = ContainerCallOutcome(
            reached=reached,
            mutations=[_mutation()] if has_mutations else [],
            error=error,
        )
        status = derive_status(outcome)
        assert status in {STATUS_OK, STATUS_NO_MUTATIONS, STATUS_UNREACHABLE, STATUS_ERROR}
        if not reached:
            assert status == STATUS_UNREACHABLE
        if status == STATUS_OK:
            assert reached and error is None and has_mutations


def test_merge_is_idempotent_over_its_own_store_half():
    """Feeding the merge's own store-derived entries back in changes nothing."""
    records = [_record("a", active=True), _record("b")]
    first, _ = merge_registry(CODE_TWO, records)
    round_tripped = [
        {
            "name": e.name, "display_name": e.display_name, "description": e.description,
            "intents": sorted(e.intents), "endpoint": e.endpoint, "active": e.active,
        }
        for e in first if e.source == SOURCE_STORE
    ]
    second, warnings = merge_registry(CODE_TWO, round_tripped)
    assert [(e.name, e.active, e.intents, e.endpoint) for e in first] == \
           [(e.name, e.active, e.intents, e.endpoint) for e in second]
    assert warnings == []


# ---------------------------------------------------------------------------
# ContainerRegistryStore — §4.5
# ---------------------------------------------------------------------------

class _FakeTable:
    """Minimal stand-in for a boto3 DynamoDB Table resource."""

    def __init__(self, items=None, fail_with=None, pages=None):
        self.items = items if items is not None else []
        self.fail_with = fail_with
        self.pages = pages
        self.query_calls = 0
        self.update_calls = []

    def query(self, **kwargs):
        self.query_calls += 1
        if self.fail_with:
            raise self.fail_with
        if self.pages is not None:
            idx = 1 if kwargs.get("ExclusiveStartKey") else 0
            page = self.pages[idx]
            return page
        return {"Items": list(self.items)}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        if self.fail_with:
            raise self.fail_with
        return {"Attributes": {"name": kwargs["Key"]["name"],
                               "active": kwargs["ExpressionAttributeValues"][":active"],
                               "updated_at": kwargs["ExpressionAttributeValues"][":ts"],
                               "updated_by": kwargs["ExpressionAttributeValues"][":who"]}}


def _store_with(table, **kwargs):
    store = ContainerRegistryStore(table_name="t", region="us-east-1", **kwargs)
    store._table = table
    return store


# Named exactly as botocore names the dynamically generated resource-level
# exception, because the production check matches on the type name.
class ConditionalCheckFailedException(Exception):
    pass


class _ClientErrorShaped(Exception):
    """A client-level ClientError, which carries the code in `response` instead."""

    def __init__(self):
        super().__init__("An error occurred (ConditionalCheckFailedException)")
        self.response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class TestContainerRegistryStore:
    def test_unset_table_name_means_the_feature_is_off_not_an_empty_registry(self):
        store = ContainerRegistryStore(table_name="", region="us-east-1")
        assert store.configured is False
        assert store.get_records() == []
        snap = store.snapshot()
        assert snap["tableConfigured"] is False
        assert snap["tableName"] is None

    def test_reads_records(self):
        table = _FakeTable(items=[_record("a"), _record("b")])
        store = _store_with(table)
        assert [r["name"] for r in store.get_records()] == ["a", "b"]
        assert store.snapshot()["tableReachable"] is True

    def test_ttl_short_circuits_repeat_reads(self):
        table = _FakeTable(items=[_record("a")])
        store = _store_with(table, ttl_seconds=300)
        for _ in range(10):
            store.get_records()
        assert table.query_calls == 1, "the bid path must not pay a read per request"

    def test_zero_ttl_reads_every_time(self):
        table = _FakeTable(items=[_record("a")])
        store = _store_with(table, ttl_seconds=0)
        store.get_records()
        store.get_records()
        assert table.query_calls == 2

    def test_pagination_is_followed_rather_than_truncated(self):
        pages = [
            {"Items": [_record("a")], "LastEvaluatedKey": {"name": "a"}},
            {"Items": [_record("b")]},
        ]
        store = _store_with(_FakeTable(pages=pages))
        assert [r["name"] for r in store.get_records()] == ["a", "b"]

    def test_read_failure_never_raises_and_keeps_the_last_known_good(self):
        table = _FakeTable(items=[_record("a")])
        store = _store_with(table, ttl_seconds=0)
        assert [r["name"] for r in store.get_records()] == ["a"]
        table.fail_with = RuntimeError("dynamo down")
        assert [r["name"] for r in store.get_records()] == ["a"], "stale but valid"
        assert store.consecutive_errors == 1
        snap = store.snapshot()
        assert snap["tableReachable"] is False
        assert "dynamo down" in snap["error"]

    def test_failure_before_any_success_yields_an_empty_registry(self):
        store = _store_with(_FakeTable(fail_with=RuntimeError("boom")))
        assert store.get_records() == []

    def test_reachability_is_unknown_before_any_read_is_attempted(self):
        store = _store_with(_FakeTable())
        assert store.snapshot()["tableReachable"] is None, "None is 'not known yet', not False"

    def test_errors_reset_after_a_success(self):
        table = _FakeTable(fail_with=RuntimeError("boom"))
        store = _store_with(table, ttl_seconds=0)
        store.get_records()
        assert store.consecutive_errors == 1
        table.fail_with = None
        table.items = [_record("a")]
        store.get_records()
        assert store.consecutive_errors == 0
        assert store.snapshot()["error"] is None

    def test_set_active_uses_an_update_expression_and_an_existence_condition(self):
        table = _FakeTable()
        store = _store_with(table)
        out = store.set_active("mine", True, updated_by="user-123")
        (call,) = table.update_calls
        assert call["Key"] == {"registry": REGISTRY_PARTITION, "name": "mine"}
        assert call["UpdateExpression"].startswith("SET ")
        assert "attribute_exists" in call["ConditionExpression"]
        # `name` is a DynamoDB reserved word, so it must be aliased.
        assert call["ExpressionAttributeNames"]["#name"] == "name"
        assert call["ExpressionAttributeValues"][":active"] is True
        assert call["ExpressionAttributeValues"][":who"] == "user-123"
        assert out["active"] is True

    def test_set_active_does_not_overwrite_the_whole_record(self):
        """UpdateExpression must touch only the three fields it owns."""
        table = _FakeTable()
        _store_with(table).set_active("mine", False)
        expr = table.update_calls[0]["UpdateExpression"]
        for field in ("display_name", "description", "intents", "endpoint"):
            assert field not in expr

    def test_set_active_invalidates_the_cache_so_the_writer_sees_its_own_change(self):
        table = _FakeTable(items=[_record("mine", active=False)])
        store = _store_with(table, ttl_seconds=300)
        store.get_records()
        assert table.query_calls == 1
        store.set_active("mine", True)
        table.items = [_record("mine", active=True)]
        assert store.get_records()[0]["active"] is True
        assert table.query_calls == 2

    def test_set_active_on_an_unconfigured_store_raises_unavailable(self):
        store = ContainerRegistryStore(table_name="", region="us-east-1")
        with pytest.raises(RegistryStoreUnavailable):
            store.set_active("mine", True)

    def test_set_active_maps_a_resource_level_condition_failure_to_not_found(self):
        store = _store_with(_FakeTable(fail_with=ConditionalCheckFailedException("nope")))
        with pytest.raises(RegistryRecordNotFound):
            store.set_active("gone", True)

    def test_set_active_maps_a_client_level_condition_failure_to_not_found(self):
        """botocore reports this two different ways; both must land on 409, not 503."""
        store = _store_with(_FakeTable(fail_with=_ClientErrorShaped()))
        with pytest.raises(RegistryRecordNotFound):
            store.set_active("gone", True)

    def test_set_active_maps_other_failures_to_unavailable(self):
        store = _store_with(_FakeTable(fail_with=RuntimeError("throttled")))
        with pytest.raises(RegistryStoreUnavailable):
            store.set_active("mine", True)

    def test_ttl_defaults_to_thirty_seconds(self):
        assert ContainerRegistryStore(table_name="t", region="us-east-1").ttl_seconds == 30.0

    def test_a_non_numeric_ttl_env_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("CONTAINER_REGISTRY_TTL", "not-a-number")
        assert ContainerRegistryStore(table_name="t", region="us-east-1").ttl_seconds == 30.0


# ---------------------------------------------------------------------------
# Transport classification — §4.1
# ---------------------------------------------------------------------------

def _fake_container_app():
    """A container that can be made to answer in each of the ways that matter."""

    async def with_mutations(request):
        return JSONResponse({
            "mutations": [{"intent": 8, "op": 1, "path": "/imp/1/ext/cids"}],
            "metadata": {"model_version": "mv-1"},
        })

    async def empty(request):
        return JSONResponse({"mutations": [], "metadata": {"model_version": "mv-2"}})

    async def not_json(request):
        return PlainTextResponse("this is not json", status_code=200)

    async def server_error(request):
        return PlainTextResponse("boom", status_code=500)

    async def mcp_only(request):
        return JSONResponse({
            "result": {"content": [{
                "type": "text",
                "text": '{"mutations": [{"intent": 8, "op": 1, "path": "/p"}],'
                        ' "metadata": {"model_version": "mv-mcp"}}',
            }]},
        })

    return Starlette(routes=[
        Route("/muts/mutate", with_mutations, methods=["POST"]),
        Route("/empty/mutate", empty, methods=["POST"]),
        Route("/badjson/mutate", not_json, methods=["POST"]),
        Route("/err/mutate", server_error, methods=["POST"]),
        Route("/err/mcp", server_error, methods=["POST"]),
        Route("/mcponly/mcp", mcp_only, methods=["POST"]),
    ])


def _classify(base_urls):
    """Return {label: ContainerInvocationModel} for each base URL, in one loop."""
    from orchestrator import app as oapp

    async def run():
        out = {}
        transport = httpx.ASGITransport(app=_fake_container_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            for label, base in base_urls.items():
                out[label] = await oapp._call_container_timed(
                    client,
                    {"name": label, "mcp": base, "grpc": "", "display_name": f"D-{label}"},
                    {"id": "r1", "bid_request": {}}, b"{}", 1.0,
                )
        return out

    return asyncio.run(run())


class TestTransportClassification:
    def test_each_outcome_maps_to_its_own_status(self):
        pytest.importorskip("grpc")
        got = _classify({
            "muts": "http://t/muts",
            "empty": "http://t/empty",
            "badjson": "http://t/badjson",
            "err": "http://t/err",
        })
        assert got["muts"].status == STATUS_OK
        assert got["muts"].model_version == "mv-1"
        assert got["empty"].status == STATUS_NO_MUTATIONS
        assert got["empty"].model_version == "mv-2", "a real version, even with no mutations"
        assert got["badjson"].status == STATUS_ERROR
        assert got["err"].status == STATUS_ERROR

    def test_display_name_is_propagated_for_the_pipeline_to_label_with(self):
        pytest.importorskip("grpc")
        got = _classify({"muts": "http://t/muts"})
        assert got["muts"].display_name == "D-muts"

    def test_mcp_fallback_still_works_when_rest_is_absent(self):
        pytest.importorskip("grpc")
        got = _classify({"mcponly": "http://t/mcponly"})
        assert got["mcponly"].status == STATUS_OK
        assert got["mcponly"].model_version == "mv-mcp"

    def test_a_refused_connection_is_unreachable_not_ok(self):
        """The exact regression this feature exists to fix.

        A container with no listener used to report ``ok`` with zero mutations,
        because the connect error was caught and degraded to ``[]`` before the
        caller could see it. An unimplemented template container IS this case, so
        shipping it without this fix would mean a container that reports itself
        healthy on every request.
        """
        pytest.importorskip("grpc")
        import socket
        from orchestrator import app as oapp

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
        s.close()

        async def run():
            async with httpx.AsyncClient() as client:
                return await oapp._call_container_timed(
                    client,
                    {"name": "gone", "mcp": f"http://127.0.0.1:{dead_port}", "grpc": ""},
                    {"id": "r1", "bid_request": {}}, b"{}", 1.0,
                )

        inv = asyncio.run(run())
        assert inv.status == STATUS_UNREACHABLE
        assert inv.mutations == []
        assert inv.latency_ms >= 0

    def test_a_parsed_200_is_definitive_so_a_healthy_container_is_called_once(self):
        """No double-call when a container legitimately returns nothing.

        The REST path used to fall through to MCP whenever it got zero mutations,
        so an empty-but-healthy container was called twice per request.
        """
        pytest.importorskip("grpc")
        from orchestrator import app as oapp

        calls = []

        async def counting_empty(request):
            calls.append(request.url.path)
            return JSONResponse({"mutations": []})

        app = Starlette(routes=[
            Route("/c/mutate", counting_empty, methods=["POST"]),
            Route("/c/mcp", counting_empty, methods=["POST"]),
        ])

        async def run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                return await oapp._call_container_timed(
                    client, {"name": "c", "mcp": "http://t/c", "grpc": ""},
                    {"id": "r1"}, b"{}", 1.0,
                )

        inv = asyncio.run(run())
        assert inv.status == STATUS_NO_MUTATIONS
        assert calls == ["/c/mutate"], f"expected one call, got {calls}"


# ---------------------------------------------------------------------------
# Fan-out — §4.2
# ---------------------------------------------------------------------------

class TestFanOut:
    def test_inactive_containers_are_reported_disabled_and_never_called(self, monkeypatch):
        pytest.importorskip("grpc")
        from orchestrator import app as oapp

        called = []

        async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
            from shared.artf_types import ContainerInvocationModel
            called.append(container["name"])
            return ContainerInvocationModel(
                name=container["name"], status=STATUS_OK, latency_ms=1.0,
                mutations=[_mutation()], display_name=container.get("display_name", ""),
            )

        entries, _ = merge_registry(
            CODE_TWO, [_record("mine", intents=["ADD_CIDS"], active=False)]
        )
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
        monkeypatch.setattr(oapp, "_call_container_timed", fake_call)

        invocations = asyncio.run(oapp._fan_out({"id": "r"}, b"{}", None, timeout_s=1.0))

        assert "mine" not in called
        by_name = {i.name: i for i in invocations}
        assert by_name["mine"].status == STATUS_DISABLED
        assert by_name["mine"].latency_ms == 0
        assert by_name["mine"].mutations == []

    def test_every_registry_entry_appears_in_stage_then_registry_order(self, monkeypatch):
        """Stage order first (shared/artf_stages.py), registry order within a stage,
        then the entries that were not called, in registry order.

        CODE_TWO puts the shader (stage 4) before the enricher (stage 1); the two
        store records carry ADD_CIDS (stage 1) and merge in name order. So the
        called order is enricher, aa, zz, shader, and the disabled record comes last.
        """
        pytest.importorskip("grpc")
        from orchestrator import app as oapp
        from shared.artf_types import ContainerInvocationModel

        async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
            return ContainerInvocationModel(
                name=container["name"], status=STATUS_NO_MUTATIONS, latency_ms=1.0,
            )

        entries, _ = merge_registry(
            CODE_TWO,
            [_record("zz", active=True), _record("aa", active=True), _record("off", active=False)],
        )
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
        monkeypatch.setattr(oapp, "_call_container_timed", fake_call)

        invocations = asyncio.run(oapp._fan_out({"id": "r"}, b"{}", None, timeout_s=1.0))
        assert [i.name for i in invocations] == ["metrics-enricher", "aa", "zz", "dlrm-bid-shader", "off"]
        assert {i.name for i in invocations} == {e.name for e in entries}

    def test_a_disabled_container_does_not_suppress_other_mutations(self, monkeypatch):
        pytest.importorskip("grpc")
        from orchestrator import app as oapp
        from shared.artf_types import ContainerInvocationModel

        async def fake_call(client, container, payload, payload_bytes, timeout_s, headers=None):
            return ContainerInvocationModel(
                name=container["name"], status=STATUS_OK, latency_ms=1.0, mutations=[_mutation()],
            )

        entries, _ = merge_registry(CODE_TWO, [_record("off", active=False)])
        monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
        monkeypatch.setattr(oapp, "_call_container_timed", fake_call)

        invocations = asyncio.run(oapp._fan_out({"id": "r"}, b"{}", None, timeout_s=1.0))
        total = sum(len(i.mutations) for i in invocations)
        assert total == 2, "both built-ins still contributed"


# ---------------------------------------------------------------------------
# Activation endpoint — §4.6
# ---------------------------------------------------------------------------

def _post_active(monkeypatch, name, body, *, entries, store=None):
    """Drive set_container_active through the real Starlette route."""
    from orchestrator import app as oapp

    monkeypatch.setattr(oapp, "_effective_registry", lambda: (entries, []))
    if store is not None:
        monkeypatch.setattr(oapp, "_registry_store", lambda: store)

    route_app = Starlette(routes=[
        Route("/v1/containers/{name}/active", oapp.set_container_active, methods=["POST"]),
    ])

    async def run():
        transport = httpx.ASGITransport(app=route_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://o") as client:
            return await client.post(f"/v1/containers/{name}/active", json=body)

    return asyncio.run(run())


class TestActivationEndpoint:
    def _entries(self):
        entries, _ = merge_registry(CODE_TWO, [_record("mine", active=False)])
        return entries

    def test_activating_a_store_container_succeeds_and_reports_the_effect_window(self, monkeypatch):
        pytest.importorskip("grpc")
        store = _store_with(_FakeTable(), ttl_seconds=30)
        resp = _post_active(monkeypatch, "mine", {"active": True},
                            entries=self._entries(), store=store)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True and body["active"] is True
        # The change is not global at once: every replica reads through its own
        # TTL cache, and the response says so rather than implying immediacy.
        assert body["effectiveWithinSeconds"] == 30
        assert "30 seconds" in body["message"]

    def test_deactivating_works_too(self, monkeypatch):
        pytest.importorskip("grpc")
        store = _store_with(_FakeTable())
        resp = _post_active(monkeypatch, "mine", {"active": False},
                            entries=self._entries(), store=store)
        assert resp.status_code == 200 and resp.json()["active"] is False

    def test_a_code_defined_container_is_refused_with_409(self, monkeypatch):
        """The FR-9 boundary test: built-ins are not UI-mutable.

        Refused rather than silently ignored, so the caller learns the built-in
        bid path cannot be switched off instead of watching a control spring back.
        """
        pytest.importorskip("grpc")
        store = _store_with(_FakeTable())
        resp = _post_active(monkeypatch, "dlrm-bid-shader", {"active": False},
                            entries=self._entries(), store=store)
        assert resp.status_code == 409
        assert "built-in" in resp.json()["error"]
        assert store._table.update_calls == [], "no write may be attempted"

    def test_an_unknown_container_is_404(self, monkeypatch):
        pytest.importorskip("grpc")
        resp = _post_active(monkeypatch, "nope", {"active": True},
                            entries=self._entries(), store=_store_with(_FakeTable()))
        assert resp.status_code == 404

    @pytest.mark.parametrize("body", [{}, {"active": "true"}, {"active": 1}, {"active": None}, []])
    def test_a_non_boolean_active_is_400_and_never_coerced(self, monkeypatch, body):
        """An ambiguous request is rejected, not guessed at.

        Coercing "true" or 1 would silently change bid behaviour on a
        malformed request.
        """
        pytest.importorskip("grpc")
        store = _store_with(_FakeTable())
        resp = _post_active(monkeypatch, "mine", body,
                            entries=self._entries(), store=store)
        assert resp.status_code == 400
        assert store._table.update_calls == []

    def test_an_unconfigured_store_is_503_naming_the_reason(self, monkeypatch):
        pytest.importorskip("grpc")
        store = ContainerRegistryStore(table_name="", region="us-east-1")
        resp = _post_active(monkeypatch, "mine", {"active": True},
                            entries=self._entries(), store=store)
        assert resp.status_code == 503
        assert "CONTAINER_REGISTRY_TABLE" in resp.json()["error"]

    def test_a_vanished_record_is_409_rather_than_being_recreated(self, monkeypatch):
        pytest.importorskip("grpc")
        store = _store_with(_FakeTable(fail_with=ConditionalCheckFailedException("gone")))
        resp = _post_active(monkeypatch, "mine", {"active": True},
                            entries=self._entries(), store=store)
        assert resp.status_code == 409

    def test_the_route_is_registered_on_all_three_prefixes(self):
        pytest.importorskip("grpc")
        from orchestrator import app as oapp
        paths = {r.path for r in oapp.routes}
        for prefix in ("", "/api", "/fabric"):
            assert f"{prefix}/v1/containers/{{name}}/active" in paths


# ---------------------------------------------------------------------------
# Backward compatibility of the pre-existing surface
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_containers_registry_keeps_the_dict_shape_loadtest_depends_on(self):
        pytest.importorskip("grpc")
        from orchestrator.app import CONTAINERS
        for c in CONTAINERS:
            assert isinstance(c["intents"], set)
            assert isinstance(c["name"], str) and c["mcp"].startswith("http")

    def test_filter_containers_still_returns_transport_ready_dicts(self):
        pytest.importorskip("grpc")
        from orchestrator.app import _filter_containers
        got = _filter_containers(["BID_SHADE"])
        assert [c["name"] for c in got] == ["dlrm-bid-shader"]
        assert {"name", "intents", "grpc", "mcp"} <= set(got[0])

    def test_container_invocation_model_display_name_defaults_to_empty(self):
        from shared.artf_types import ContainerInvocationModel
        # Additive with a default, so every existing construction site still works.
        inv = ContainerInvocationModel(name="x", status=STATUS_OK, latency_ms=1.0)
        assert inv.display_name == ""
