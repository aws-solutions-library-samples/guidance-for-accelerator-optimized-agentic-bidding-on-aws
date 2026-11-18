"""The Deal Scorer's sample interaction generator (training/ncf_sample/generate.py).

Example tests pin the shipped cohort spec; property tests (Hypothesis, PBT-02, 03,
07, 08) draw cohort specs from the seed's own deal ids. The Hypothesis seed is fixed
(HYPOTHESIS_SEED) and printed, so a failure is replayable; shrinking is on.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from collections import Counter

import pytest
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SOURCE)

from shared import ncf_ids  # noqa: E402
from training.ncf_sample import generate as gen  # noqa: E402

HYPOTHESIS_SEED = 20261007
SPEC = gen.load_json(gen.DEFAULT_COHORTS)
SEED_DEALS = gen.seed_deal_ids(gen.load_json(gen.DEFAULT_SEED))


@pytest.fixture(autouse=True, scope="module")
def _print_seeds():
    print(f"hypothesis seed {HYPOTHESIS_SEED}; generator seed {SPEC['generator_seed']}")


def _parse(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


# ---------------------------------------------------------------- the shipped spec


@pytest.fixture(scope="module")
def rows():
    return gen.generate(SPEC, SEED_DEALS)


def test_the_same_seed_writes_a_byte_identical_csv():
    first = gen.to_csv_bytes(gen.generate(SPEC, SEED_DEALS))
    second = gen.to_csv_bytes(gen.generate(SPEC, SEED_DEALS))
    assert first == second


def test_a_different_generator_seed_changes_only_generated_labels():
    other = dict(SPEC, generator_seed=SPEC["generator_seed"] + 1)
    a, b = gen.generate(SPEC, SEED_DEALS), gen.generate(other, SEED_DEALS)
    assert [(r.user_id, r.deal_id) for r in a] == [(r.user_id, r.deal_id) for r in b]
    assert [r.label for r in a if r.is_scenario_user] == [r.label for r in b if r.is_scenario_user]
    assert [r.label for r in a] != [r.label for r in b]


def test_the_header_is_the_documented_schema(rows):
    first_line = gen.to_csv_bytes(rows).split(b"\n", 1)[0]
    assert first_line == b"user_id,user_bucket,deal_id,item_bucket,label,cohort,is_scenario_user,provenance"


def test_every_row_carries_sample_provenance(rows):
    assert {r.provenance for r in rows} == {"sample"}
    assert {d["provenance"] for d in _parse(gen.to_csv_bytes(rows))} == {"sample"}


def test_every_user_is_labelled_against_every_seed_deal_exactly_once(rows):
    pairs = Counter((r.user_id, r.deal_id) for r in rows)
    users = {r.user_id for r in rows}
    assert set(pairs.values()) == {1}
    assert len(pairs) == len(users) * len(SEED_DEALS)
    for user in users:
        assert {d for (u, d) in pairs if u == user} == set(SEED_DEALS)


def test_the_shipped_spec_sizes(rows):
    stats = gen.stats(rows, SPEC)
    cohorts = len(SPEC["cohorts"])
    assert stats["scenario_users"] == sum(len(c["scenario_users"]) for c in SPEC["cohorts"])
    assert stats["generated_members"] == cohorts * SPEC["members_per_cohort"]
    assert stats["rows"] == stats["users"] * len(SEED_DEALS)


def test_scenario_users_carry_their_cohort_targets_exactly(rows):
    by_id = {c["id"]: c for c in SPEC["cohorts"]}
    for r in rows:
        if r.is_scenario_user:
            assert r.label == (1 if r.deal_id in gen.positives(by_id[r.cohort]) else 0), r


def test_the_observed_noise_rate_is_within_bounds(rows):
    stats = gen.stats(rows, SPEC)
    n = stats["generated_rows"]
    rate = SPEC["noise_rate"]
    observed = stats["flipped_generated_labels"] / n
    # Five binomial standard deviations: a draw outside is a generator defect.
    assert abs(observed - rate) <= 5 * math.sqrt(rate * (1 - rate) / n)


def test_no_two_users_share_a_bucket_and_no_scenario_user_collides(rows):
    buckets: dict[int, str] = {}
    for r in rows:
        assert buckets.setdefault(r.user_bucket, r.user_id) == r.user_id
    scenario = {r.user_bucket for r in rows if r.is_scenario_user}
    generated = {r.user_bucket for r in rows if not r.is_scenario_user}
    assert not scenario & generated


def test_buckets_are_the_serving_hash(rows):
    for r in rows[:: max(1, len(rows) // 500)]:
        assert r.user_bucket == ncf_ids.bucket(r.user_id, ncf_ids.NB_USERS)
        assert r.item_bucket == ncf_ids.bucket(r.deal_id, ncf_ids.NB_ITEMS)


def test_generated_member_ids_are_named_by_cohort(rows):
    for r in rows:
        if not r.is_scenario_user:
            assert r.user_id.startswith(f"{gen.MEMBER_PREFIX}{r.cohort}-")


def test_a_spec_naming_an_unknown_deal_is_refused():
    bad = json.loads(json.dumps(SPEC))
    bad["cohorts"][0]["activate"] = ["deal-not-in-the-seed"]
    with pytest.raises(gen.SpecError, match="deal-not-in-the-seed"):
        gen.generate(bad, SEED_DEALS)


def test_a_deal_in_two_target_lists_is_refused():
    bad = json.loads(json.dumps(SPEC))
    cohort = bad["cohorts"][0]
    cohort["suppress"] = list(cohort["activate"])
    with pytest.raises(gen.SpecError, match="both"):
        gen.generate(bad, SEED_DEALS)


def test_scenario_users_sharing_a_bucket_are_refused():
    bad = json.loads(json.dumps(SPEC))
    taken = ncf_ids.bucket(bad["cohorts"][0]["scenario_users"][0])
    twin = next(f"twin-{i}" for i in range(100000) if ncf_ids.bucket(f"twin-{i}") == taken)
    bad["cohorts"][1]["scenario_users"].append(twin)
    with pytest.raises(gen.SpecError, match="share user bucket"):
        gen.generate(bad, SEED_DEALS)


def test_the_cli_writes_the_same_bytes(tmp_path):
    out = tmp_path / "interactions.csv"
    assert gen.main(["--out", str(out)]) == 0
    assert out.read_bytes() == gen.to_csv_bytes(gen.generate(SPEC, SEED_DEALS))


# ------------------------------------------------------------- property tests


@st.composite
def cohort_specs(draw):
    """Cohort specs over the seed's deal ids (PBT-07: domain values, not raw text)."""
    n_cohorts = draw(st.integers(min_value=1, max_value=4))
    cohorts = []
    users_seen: set[int] = set()
    for i in range(n_cohorts):
        deals = draw(st.permutations(SEED_DEALS))
        cuts = sorted(draw(st.lists(st.integers(0, len(deals)), min_size=4, max_size=4)))
        activate, suppress, keep, not_activated = (
            deals[0:cuts[0]], deals[cuts[0]:cuts[1]], deals[cuts[1]:cuts[2]], deals[cuts[2]:cuts[3]]
        )
        n_users = draw(st.integers(min_value=1, max_value=2))
        users = [f"user-{i}-{draw(st.integers(0, 10**6))}-{k}" for k in range(n_users)]
        buckets = {ncf_ids.bucket(u) for u in users}
        assume(len(buckets) == len(users) and not buckets & users_seen)
        users_seen |= buckets
        cohorts.append({
            "id": f"cohort-{i}", "scenario_users": users, "sample": f"cohort-{i}.json",
            "activate": list(activate), "suppress": list(suppress), "keep": list(keep),
            "not_activated": list(not_activated), "winner": None,
        })
    return {
        "format": gen.FORMAT,
        "generator_seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
        "members_per_cohort": draw(st.integers(min_value=0, max_value=6)),
        "noise_rate": draw(st.sampled_from([0.0, 0.03, 0.1, 0.25, 0.45])),
        "cohorts": cohorts,
    }


_PROPERTY = settings(max_examples=60, deadline=None, database=None, print_blob=True)


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(cohort_specs())
def test_generation_is_deterministic(spec):
    assert gen.to_csv_bytes(gen.generate(spec, SEED_DEALS)) == gen.to_csv_bytes(gen.generate(spec, SEED_DEALS))


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(cohort_specs())
def test_the_csv_round_trips(spec):
    """PBT-02: writing rows to CSV and reading them back gives the same rows."""
    rows = gen.generate(spec, SEED_DEALS)
    back = [
        gen.Row(d["user_id"], int(d["user_bucket"]), d["deal_id"], int(d["item_bucket"]),
                int(d["label"]), d["cohort"], int(d["is_scenario_user"]), d["provenance"])
        for d in _parse(gen.to_csv_bytes(rows))
    ]
    assert back == rows


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(cohort_specs())
def test_coverage_labels_and_buckets_hold_for_any_spec(spec):
    """PBT-03: every user x every deal once; scenario labels exact; buckets unique."""
    rows = gen.generate(spec, SEED_DEALS)
    by_id = {c["id"]: c for c in spec["cohorts"]}
    users = {r.user_id for r in rows}
    assert Counter((r.user_id, r.deal_id) for r in rows) == Counter(
        {(u, d): 1 for u in users for d in SEED_DEALS})
    expected_users = sum(len(c["scenario_users"]) for c in spec["cohorts"])
    expected_users += len(spec["cohorts"]) * spec["members_per_cohort"]
    assert len(users) == expected_users
    owner: dict[int, str] = {}
    for r in rows:
        assert owner.setdefault(r.user_bucket, r.user_id) == r.user_id
        if r.is_scenario_user:
            assert r.label == (1 if r.deal_id in gen.positives(by_id[r.cohort]) else 0)


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(cohort_specs())
def test_noise_touches_generated_members_only(spec):
    rows = gen.generate(spec, SEED_DEALS)
    stats = gen.stats(rows, spec)
    if spec["noise_rate"] == 0.0:
        assert stats["flipped_generated_labels"] == 0
    n = stats["generated_rows"]
    if n:
        rate = spec["noise_rate"]
        bound = 5 * math.sqrt(rate * (1 - rate) / n) + 1 / n
        assert abs(stats["flipped_generated_labels"] / n - rate) <= bound
