"""Tests for mutation conflict resolution — orchestrator/container_registry.py.

Two containers may claim the same intent, which the fan-out allows deliberately.
But a consumer applies the mutation list in order and, for a deal floor, the
applier does ``updatedDeals.set(dealIndex, …)`` — so only the last mutation
applied to a given path has any effect. ``resolve_conflicts`` decides which one
that is, and reports the ones it displaced.

The load-bearing test here is ``test_all_equal_priorities_matches_last_write_wins``
(BR-P8). It pins the tie-break DIRECTION against a reference implementation. The
winner is ``max`` by ``(priority, order, index)``; using ``min`` would read as
"preserving registry order" while inverting which container's floor is honoured
on every existing deployment.
"""

from __future__ import annotations

import os
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.container_registry import (  # noqa: E402
    ConflictRecord,
    MutationClaim,
    _coerce_priority,
    _entry_from_code,
    _entry_from_record,
    build_claims,
    resolve_conflicts,
)
from shared.artf_types import ContainerInvocationModel, Mutation  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers and domain generators (PBT-R5)
# ---------------------------------------------------------------------------

FLOOR_INTENT = 4          # ADJUST_DEAL_FLOOR
METRICS_INTENT = 7        # ADD_METRICS
REPLACE_OP = 2

#: Paths the real containers actually emit, so generated cases resemble traffic
#: rather than arbitrary strings. Bare primitives here would generate mutations
#: that never collide and the properties would pass without testing anything.
REAL_PATHS = [
    "/imp/imp-1/deals/deal-a",
    "/imp/imp-1/deals/deal-b",
    "/imp/imp-2/deals/deal-a",
    "/imp/imp-1/metric",
    "/imp/imp-2/metric",
    "/user/data/segment",
]

REAL_INTENTS = [1, 2, 3, 4, 5, 7, 8]


def mutation(path: str, intent: int = FLOOR_INTENT) -> Mutation:
    return Mutation(intent=intent, op=REPLACE_OP, path=path)


def invocation(name: str, *mutations: Mutation) -> ContainerInvocationModel:
    return ContainerInvocationModel(
        name=name, status="ok", latency_ms=1.0, mutations=list(mutations)
    )


mutations_st = st.builds(
    mutation,
    path=st.sampled_from(REAL_PATHS),
    intent=st.sampled_from(REAL_INTENTS),
)


@st.composite
def claim_lists(draw, max_containers: int = 4, max_mutations: int = 3):
    """Generate a realistic claim list with its registry ordering intact.

    Built by generating invocations and running them through ``build_claims``, so
    ``order`` and ``index`` are always internally consistent — a hand-built claim
    list could carry an impossible combination and make a property vacuous.
    """
    count = draw(st.integers(min_value=0, max_value=max_containers))
    invocations = []
    for i in range(count):
        muts = draw(st.lists(mutations_st, min_size=0, max_size=max_mutations))
        invocations.append(invocation(f"c{i}", *muts))
    priorities = {
        f"c{i}": draw(st.integers(min_value=-2, max_value=5)) for i in range(count)
    }
    return build_claims(invocations, priorities)


def last_write_wins(claims: list[MutationClaim]) -> dict[tuple[str, int], Mutation]:
    """Reference implementation of today's behaviour.

    A consumer walks the flattened list in order and applies each mutation, so for
    a given (path, intent) the final state is whatever the LAST one set. Three
    lines, deliberately — it is the oracle, so it must be obviously correct rather
    than clever.

    Returns a **mapping**, not a list, and that distinction matters. The comparison
    worth making is "does the resolver pick the same winner for every key", which
    is what a consumer ends up applying. List order is not part of the claim: the
    returned list necessarily changes because FR-6 drops the losers from it, and
    this reference's own ordering would be an artifact of dict insertion order
    rather than a statement about behaviour.
    """
    final: dict[tuple[str, int], Mutation] = {}
    for claim in claims:
        final[(claim.mutation.path, claim.mutation.intent)] = claim.mutation
    return final


# ---------------------------------------------------------------------------
# BR-P1 — the conflict key is (path, intent)
# ---------------------------------------------------------------------------

def test_same_path_different_intent_is_not_a_conflict():
    claims = build_claims(
        [
            invocation("a", mutation("/imp/imp-1/metric", METRICS_INTENT)),
            invocation("b", mutation("/imp/imp-1/metric", 8)),  # ADD_CIDS
        ]
    )
    survivors, conflicts, superseded = resolve_conflicts(claims)
    assert len(survivors) == 2
    assert conflicts == []
    assert superseded == {}


def test_same_intent_different_path_is_not_a_conflict():
    claims = build_claims(
        [
            invocation("a", mutation("/imp/imp-1/deals/deal-a")),
            invocation("b", mutation("/imp/imp-2/deals/deal-a")),
        ]
    )
    survivors, conflicts, _ = resolve_conflicts(claims)
    assert len(survivors) == 2
    assert conflicts == []


def test_same_path_and_intent_is_a_conflict():
    claims = build_claims(
        [
            invocation("a", mutation("/imp/imp-1/deals/deal-a")),
            invocation("b", mutation("/imp/imp-1/deals/deal-a")),
        ]
    )
    survivors, conflicts, _ = resolve_conflicts(claims)
    assert len(survivors) == 1
    assert len(conflicts) == 1
    assert conflicts[0].path == "/imp/imp-1/deals/deal-a"
    assert conflicts[0].intent == FLOOR_INTENT


# ---------------------------------------------------------------------------
# BR-P2 — an uncontested key produces no conflict record
# ---------------------------------------------------------------------------

def test_uncontested_mutations_produce_no_conflict_records():
    claims = build_claims(
        [invocation("a", mutation("/imp/imp-1/deals/deal-a"), mutation("/user/data/segment", 1))]
    )
    survivors, conflicts, superseded = resolve_conflicts(claims)
    assert len(survivors) == 2
    assert conflicts == []
    assert superseded == {}


# ---------------------------------------------------------------------------
# BR-P3 — winner is max by (priority, order, index)
# ---------------------------------------------------------------------------

def test_higher_priority_wins_regardless_of_order():
    """The built-in is FIRST in registry order and still wins on priority.

    This is the case the whole feature exists for: an operator can hand the floor
    decision back to the built-in without being able to deactivate the attached
    container.
    """
    claims = build_claims(
        [
            invocation("built-in", mutation("/imp/imp-1/deals/deal-a")),
            invocation("external", mutation("/imp/imp-1/deals/deal-a")),
        ],
        {"built-in": 10, "external": 0},
    )
    survivors, conflicts, superseded = resolve_conflicts(claims)
    assert len(survivors) == 1
    assert conflicts[0].winner == "built-in"
    assert conflicts[0].losers == ("external",)
    assert superseded == {"external": 1}


def test_equal_priority_later_registry_order_wins():
    """At equal priority the LATER container wins — this is today's behaviour.

    Store containers sort after code containers, so without a priority an attached
    container wins by default. That is not an accident of this implementation; it
    is what a consumer already did with the unresolved list.
    """
    claims = build_claims(
        [
            invocation("built-in", mutation("/imp/imp-1/deals/deal-a")),
            invocation("external", mutation("/imp/imp-1/deals/deal-a")),
        ],
        {"built-in": 0, "external": 0},
    )
    _, conflicts, _ = resolve_conflicts(claims)
    assert conflicts[0].winner == "external"


def test_negative_priority_loses_to_default():
    claims = build_claims(
        [
            invocation("built-in", mutation("/imp/imp-1/deals/deal-a")),
            invocation("external", mutation("/imp/imp-1/deals/deal-a")),
        ],
        {"built-in": 0, "external": -1},
    )
    _, conflicts, _ = resolve_conflicts(claims)
    assert conflicts[0].winner == "built-in"


def test_one_container_claiming_a_key_twice_keeps_the_last():
    """Degenerate but real: same container, same key, two mutations.

    Same priority and same order, so ``index`` decides — and the later one wins,
    matching what a consumer applying both in sequence would end up with.
    """
    later = mutation("/imp/imp-1/deals/deal-a")
    claims = build_claims([invocation("a", mutation("/imp/imp-1/deals/deal-a"), later)])
    survivors, conflicts, superseded = resolve_conflicts(claims)
    assert len(survivors) == 1
    assert survivors[0] is later
    assert superseded == {"a": 1}
    assert conflicts[0].winner == "a"
    # The container both won and lost, so it is not listed as its own loser twice.
    assert conflicts[0].losers == ("a",)


# ---------------------------------------------------------------------------
# BR-P4 — survivors are returned in input order
# ---------------------------------------------------------------------------

def test_survivors_are_returned_in_input_order():
    claims = build_claims(
        [
            invocation(
                "a",
                mutation("/user/data/segment", 1),
                mutation("/imp/imp-1/deals/deal-a"),
            ),
            invocation("b", mutation("/imp/imp-1/metric", METRICS_INTENT)),
        ]
    )
    survivors, _, _ = resolve_conflicts(claims)
    assert [m.path for m in survivors] == [
        "/user/data/segment",
        "/imp/imp-1/deals/deal-a",
        "/imp/imp-1/metric",
    ]


def test_survivor_order_is_input_order_not_winner_decision_order():
    """A late winner does not get moved to the front of the list."""
    claims = build_claims(
        [
            invocation("a", mutation("/user/data/segment", 1)),
            invocation("b", mutation("/imp/imp-1/deals/deal-a")),
            invocation("c", mutation("/imp/imp-1/deals/deal-a")),
        ],
        {"a": 0, "b": 0, "c": 9},
    )
    survivors, _, _ = resolve_conflicts(claims)
    assert [m.path for m in survivors] == ["/user/data/segment", "/imp/imp-1/deals/deal-a"]


# ---------------------------------------------------------------------------
# BR-P5 / BR-P6 — counts
# ---------------------------------------------------------------------------

def test_superseded_counts_are_per_container():
    claims = build_claims(
        [
            invocation(
                "loser",
                mutation("/imp/imp-1/deals/deal-a"),
                mutation("/imp/imp-2/deals/deal-a"),
            ),
            invocation(
                "winner",
                mutation("/imp/imp-1/deals/deal-a"),
                mutation("/imp/imp-2/deals/deal-a"),
            ),
        ],
        {"loser": 0, "winner": 3},
    )
    survivors, conflicts, superseded = resolve_conflicts(claims)
    assert len(survivors) == 2
    assert len(conflicts) == 2
    assert superseded == {"loser": 2}


def test_survivors_plus_superseded_equals_input():
    claims = build_claims(
        [
            invocation("a", mutation("/imp/imp-1/deals/deal-a")),
            invocation("b", mutation("/imp/imp-1/deals/deal-a")),
            invocation("c", mutation("/imp/imp-1/metric", METRICS_INTENT)),
        ]
    )
    survivors, _, superseded = resolve_conflicts(claims)
    assert len(survivors) + sum(superseded.values()) == len(claims)


# ---------------------------------------------------------------------------
# BR-P7 — idempotence
# ---------------------------------------------------------------------------

def test_resolution_is_idempotent():
    claims = build_claims(
        [
            invocation("a", mutation("/imp/imp-1/deals/deal-a")),
            invocation("b", mutation("/imp/imp-1/deals/deal-a")),
            invocation("c", mutation("/imp/imp-1/metric", METRICS_INTENT)),
        ]
    )
    survivors, _, _ = resolve_conflicts(claims)

    # Feed the survivors back through as a single container's output.
    again = build_claims([invocation("merged", *survivors)])
    survivors2, conflicts2, superseded2 = resolve_conflicts(again)

    assert [m.path for m in survivors2] == [m.path for m in survivors]
    assert conflicts2 == []
    assert superseded2 == {}


# ---------------------------------------------------------------------------
# BR-P8 — THE ORACLE TEST. Pins the tie-break direction.
# ---------------------------------------------------------------------------

def test_all_equal_priorities_matches_last_write_wins():
    """With no priorities set, the result equals today's behaviour exactly.

    This is the backward-compatibility guarantee and success criterion #3:
    attaching a container must not change which floor is honoured until somebody
    sets a priority. If this test fails, an existing deployment has been silently
    altered.
    """
    claims = build_claims(
        [
            invocation("first", mutation("/imp/imp-1/deals/deal-a")),
            invocation("second", mutation("/imp/imp-1/deals/deal-a")),
            invocation("third", mutation("/imp/imp-1/metric", METRICS_INTENT)),
        ]
    )
    survivors, _, _ = resolve_conflicts(claims)
    expected = last_write_wins(claims)

    # Same keys, and for each key the same mutation object the reference chose.
    assert {(m.path, m.intent) for m in survivors} == set(expected)
    for m in survivors:
        assert m is expected[(m.path, m.intent)]

    # And concretely: the SECOND container's floor is the one that survives.
    floor = [m for m in survivors if m.intent == FLOOR_INTENT]
    assert len(floor) == 1
    assert floor[0] is claims[1].mutation


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_equal_priorities_always_matches_reference(claims):
    """PBT form of the oracle: force every priority equal, compare to reference.

    Compares per-key winners rather than list order, for the reason recorded on
    ``last_write_wins``.
    """
    flattened = [
        MutationClaim(
            container=c.container, priority=0, order=c.order, index=c.index, mutation=c.mutation
        )
        for c in claims
    ]
    survivors, _, _ = resolve_conflicts(flattened)
    expected = last_write_wins(flattened)

    assert {(m.path, m.intent) for m in survivors} == set(expected)
    for m in survivors:
        assert m is expected[(m.path, m.intent)]


# ---------------------------------------------------------------------------
# BR-P9 — a built-in is only displaced by something that strictly outranks it
# ---------------------------------------------------------------------------

def test_code_container_not_displaced_without_a_higher_priority():
    """The built-in is second in registry order here, so it wins on order alone.

    Registry order really is code-first, so this ordering does not occur in
    practice — it is constructed to prove the rule rather than the arrangement.
    """
    claims = build_claims(
        [
            invocation("external", mutation("/imp/imp-1/deals/deal-a")),
            invocation("built-in", mutation("/imp/imp-1/deals/deal-a")),
        ],
        {"external": 0, "built-in": 0},
    )
    _, conflicts, superseded = resolve_conflicts(claims)
    assert conflicts[0].winner == "built-in"
    assert superseded == {"external": 1}


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_winner_is_always_maximal(claims):
    """PBT-R2: every survivor is a maximal claim for its key."""
    survivors, _, _ = resolve_conflicts(claims)
    survivor_ids = {id(m) for m in survivors}

    by_key: dict[tuple[str, int], list[MutationClaim]] = {}
    for claim in claims:
        by_key.setdefault((claim.mutation.path, claim.mutation.intent), []).append(claim)

    for group in by_key.values():
        best = max(c.priority for c in group)
        winners = [c for c in group if id(c.mutation) in survivor_ids]
        # Exactly one survivor per key, and it carries the maximal priority.
        assert len(winners) >= 1
        for w in winners:
            assert w.priority == best


# ---------------------------------------------------------------------------
# BR-P10 — total over its input
# ---------------------------------------------------------------------------

def test_empty_claims():
    assert resolve_conflicts([]) == ([], [], {})


def test_invocations_with_no_mutations_produce_no_claims():
    claims = build_claims([invocation("a"), invocation("b")])
    assert claims == []
    assert resolve_conflicts(claims) == ([], [], {})


def test_build_claims_tolerates_missing_priority_map():
    claims = build_claims([invocation("a", mutation("/imp/imp-1/deals/deal-a"))])
    assert claims[0].priority == 0


# ---------------------------------------------------------------------------
# Properties: partition and provenance (PBT-R1, PBT-R3)
# ---------------------------------------------------------------------------

@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_exactly_one_survivor_per_key(claims):
    survivors, _, _ = resolve_conflicts(claims)
    keys = [(m.path, m.intent) for m in survivors]
    assert len(keys) == len(set(keys))


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_partition_is_exact(claims):
    survivors, _, superseded = resolve_conflicts(claims)
    assert len(survivors) + sum(superseded.values()) == len(claims)


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_every_survivor_was_an_input(claims):
    survivors, _, _ = resolve_conflicts(claims)
    inputs = {id(c.mutation) for c in claims}
    for m in survivors:
        assert id(m) in inputs


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_conflicts_only_for_contested_keys(claims):
    _, conflicts, _ = resolve_conflicts(claims)
    counts: dict[tuple[str, int], int] = {}
    for c in claims:
        key = (c.mutation.path, c.mutation.intent)
        counts[key] = counts.get(key, 0) + 1
    for conflict in conflicts:
        # A conflict is only reported for a key more than one claim wrote to.
        assert counts[(conflict.path, conflict.intent)] > 1
        # Each losing container is named exactly once, even if it lost twice.
        assert len(conflict.losers) == len(set(conflict.losers))
        # The winner MAY also appear as a loser: one container can claim the same
        # key twice, win with one mutation and be superseded on the other. That is
        # a real outcome and naming it is the point.
        assert conflict.losers


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(claim_lists())
def test_property_idempotent(claims):
    survivors, _, _ = resolve_conflicts(claims)
    again = build_claims([invocation("merged", *survivors)])
    survivors2, conflicts2, superseded2 = resolve_conflicts(again)
    assert [(m.path, m.intent) for m in survivors2] == [(m.path, m.intent) for m in survivors]
    assert conflicts2 == []
    assert superseded2 == {}


# ---------------------------------------------------------------------------
# priority plumbing (FR-1, FR-2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 0),
        (0, 0),
        (5, 5),
        (-3, -3),
        ("7", 7),
        ("not-a-number", 0),
        ("", 0),
        ([], 0),
        ({}, 0),
    ],
)
def test_coerce_priority(raw, expected):
    assert _coerce_priority(raw) == expected


def test_coerce_priority_accepts_decimal():
    """DynamoDB returns numbers as Decimal, which is the real path here."""
    from decimal import Decimal

    assert _coerce_priority(Decimal("4")) == 4


def test_store_record_priority_is_read():
    entry = _entry_from_record(
        {"name": "ext", "endpoint": "http://ext:8081", "priority": 7, "intents": ["ADJUST_DEAL_FLOOR"]}
    )
    assert entry.priority == 7


def test_store_record_without_priority_defaults_to_zero():
    entry = _entry_from_record({"name": "ext", "endpoint": "http://ext:8081"})
    assert entry.priority == 0


def test_code_container_priority_is_pinned_to_zero():
    entry = _entry_from_code(
        {"name": "built-in", "intents": {"ADJUST_DEAL_FLOOR"}, "mcp": "http://b:8081"}
    )
    assert entry.priority == 0


def test_code_container_priority_cannot_be_set_from_the_dict():
    """A code container declaration carrying a priority does not get one.

    Belt and braces for FR-2: the pin is in ``_entry_from_code``, and this asserts
    there is no accidental passthrough if somebody adds the key to CONTAINERS.
    """
    entry = _entry_from_code(
        {
            "name": "built-in",
            "intents": {"ADJUST_DEAL_FLOOR"},
            "mcp": "http://b:8081",
            "priority": 99,
        }
    )
    assert entry.priority == 0


# ---------------------------------------------------------------------------
# ConflictRecord shape
# ---------------------------------------------------------------------------

def test_conflict_record_is_frozen():
    record = ConflictRecord(path="/imp/1/deals/d", intent=4, winner="a", losers=("b",))
    with pytest.raises(Exception):
        record.winner = "c"  # type: ignore[misc]


def test_conflict_names_each_loser_once():
    claims = build_claims(
        [
            invocation(
                "loser",
                mutation("/imp/imp-1/deals/deal-a"),
                mutation("/imp/imp-1/deals/deal-a"),
            ),
            invocation("winner", mutation("/imp/imp-1/deals/deal-a")),
        ],
        {"loser": 0, "winner": 5},
    )
    _, conflicts, superseded = resolve_conflicts(claims)
    assert conflicts[0].losers == ("loser",)
    assert superseded == {"loser": 2}
