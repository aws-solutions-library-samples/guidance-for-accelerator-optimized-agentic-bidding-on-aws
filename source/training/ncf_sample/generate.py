"""Deterministic sample user-deal interactions for the Deal Scorer (NCF).

Reads the cohort spec (cohorts.json, format ``ncf-sample-cohorts/1``) and the deal
ids of the deals seed (deployment/seed/deals_and_campaigns.json) and writes one row
per (user, deal) pair:

    user_id,user_bucket,deal_id,item_bucket,label,cohort,is_scenario_user,provenance

``provenance`` is ``sample`` on every row: the rows are generated, not observed.
Labels are binary implicit engagement. A cohort's positives are its ``activate``
and ``keep`` deals; every other deal in the seed is a negative. Each cohort holds
its scenario users (the ids the shipped scenario samples send) with exact labels,
plus ``members_per_cohort`` generated members ``ncf-sample-<cohort>-<n>`` whose
labels are flipped with probability ``noise_rate``. Buckets are
``shared.ncf_ids.bucket``, the hash the serving container uses. Every user in the
output has its own user bucket: a generated id whose bucket is already taken (by a
scenario user or an earlier member of any cohort) is skipped.

Deterministic: one ``random.Random(generator_seed)``, drawn once per generated
member row in cohort, member and seed-deal order. The same spec and seed give a
byte-identical CSV.

Usage:
    python training/ncf_sample/generate.py --out /tmp/interactions.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

_SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from shared import ncf_ids  # noqa: E402

FORMAT = "ncf-sample-cohorts/1"
HEADER = ("user_id", "user_bucket", "deal_id", "item_bucket", "label", "cohort", "is_scenario_user",
          "provenance")
#: The provenance column's value on every generated row.
PROVENANCE = "sample"
MEMBER_PREFIX = "ncf-sample-"
#: The lists a cohort uses to state its targets. Pairwise disjoint.
TARGET_LISTS = ("activate", "suppress", "keep", "not_activated")

DEFAULT_COHORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cohorts.json")
DEFAULT_SEED = os.path.join(os.path.dirname(_SOURCE_ROOT), "deployment", "seed", "deals_and_campaigns.json")

_COHORT_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class SpecError(ValueError):
    """The cohort spec cannot be turned into interactions."""


@dataclass(frozen=True)
class Row:
    user_id: str
    user_bucket: int
    deal_id: str
    item_bucket: int
    label: int
    cohort: str
    is_scenario_user: int
    provenance: str = PROVENANCE

    def as_csv(self) -> tuple:
        return (self.user_id, self.user_bucket, self.deal_id, self.item_bucket,
                self.label, self.cohort, self.is_scenario_user, self.provenance)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def seed_deal_ids(seed: Mapping) -> list[str]:
    """Deal ids in seed order."""
    return [str(d["id"]) for d in seed.get("deals") or ()]


def positives(cohort: Mapping) -> frozenset[str]:
    """The deals a cohort engages with: its activate and keep targets."""
    return frozenset(cohort.get("activate") or ()) | frozenset(cohort.get("keep") or ())


def validate(spec: Mapping, deal_ids: Sequence[str]) -> None:
    """Raise SpecError when the spec is malformed or names a deal the seed lacks."""
    if spec.get("format") != FORMAT:
        raise SpecError(f"cohort spec format is {spec.get('format')!r}, expected {FORMAT!r}")
    seed = spec.get("generator_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SpecError(f"generator_seed must be an integer, got {seed!r}")
    members = spec.get("members_per_cohort")
    if not isinstance(members, int) or isinstance(members, bool) or members < 0:
        raise SpecError(f"members_per_cohort must be a non-negative integer, got {members!r}")
    noise = spec.get("noise_rate")
    if not isinstance(noise, (int, float)) or isinstance(noise, bool) or not 0.0 <= float(noise) < 0.5:
        raise SpecError(f"noise_rate must be a number in [0, 0.5), got {noise!r}")
    cohorts = spec.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise SpecError("cohorts must be a non-empty list")
    if len(set(deal_ids)) != len(deal_ids):
        raise SpecError("the seed lists a deal id twice")
    known = set(deal_ids)
    seen_ids: set[str] = set()
    seen_users: dict[str, str] = {}
    for cohort in cohorts:
        cid = cohort.get("id")
        if not isinstance(cid, str) or not _COHORT_ID.match(cid):
            raise SpecError(f"cohort id {cid!r} must be lowercase letters, digits and dashes")
        if cid in seen_ids:
            raise SpecError(f"cohort id {cid!r} appears twice")
        seen_ids.add(cid)
        sample = cohort.get("sample")
        if not isinstance(sample, str) or not sample.endswith(".json") or "/" in sample:
            raise SpecError(f"cohort {cid}: sample must be a .json file name, got {sample!r}")
        users = cohort.get("scenario_users")
        if not isinstance(users, list) or not users or not all(isinstance(u, str) and u for u in users):
            raise SpecError(f"cohort {cid}: scenario_users must be a non-empty list of ids")
        for user in users:
            if user.startswith(MEMBER_PREFIX):
                raise SpecError(f"cohort {cid}: scenario user {user!r} uses the generated-member prefix")
            if user in seen_users:
                raise SpecError(f"scenario user {user!r} is in cohorts {seen_users[user]} and {cid}")
            seen_users[user] = cid
        lists = {}
        for name in TARGET_LISTS:
            values = cohort.get(name)
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise SpecError(f"cohort {cid}: {name} must be a list of deal ids")
            unknown = [v for v in values if v not in known]
            if unknown:
                raise SpecError(f"cohort {cid}: {name} names deals the seed does not hold: {unknown}")
            if len(set(values)) != len(values):
                raise SpecError(f"cohort {cid}: {name} lists a deal twice")
            lists[name] = set(values)
        for i, a in enumerate(TARGET_LISTS):
            for b in TARGET_LISTS[i + 1:]:
                both = sorted(lists[a] & lists[b])
                if both:
                    raise SpecError(f"cohort {cid}: {both} are in both {a} and {b}")


def assign_users(spec: Mapping) -> list[tuple[str, str, int, int]]:
    """(cohort id, user id, user bucket, is_scenario_user) for every user, in output order.

    Scenario users keep their ids and so their buckets; two scenario users sharing a
    bucket is a SpecError. Generated members take the next free bucket in id order.
    """
    taken: dict[int, str] = {}
    for cohort in spec["cohorts"]:
        for user in cohort["scenario_users"]:
            b = ncf_ids.bucket(user, ncf_ids.NB_USERS)
            if b in taken:
                raise SpecError(f"scenario users {taken[b]!r} and {user!r} share user bucket {b}")
            taken[b] = user
    want = int(spec["members_per_cohort"])
    users: list[tuple[str, str, int, int]] = []
    for cohort in spec["cohorts"]:
        cid = cohort["id"]
        for user in cohort["scenario_users"]:
            users.append((cid, user, ncf_ids.bucket(user, ncf_ids.NB_USERS), 1))
        accepted = 0
        n = 0
        while accepted < want:
            if len(taken) >= ncf_ids.NB_USERS:
                raise SpecError(f"no free user bucket left for cohort {cid} after {len(taken)} users")
            member = f"{MEMBER_PREFIX}{cid}-{n}"
            n += 1
            b = ncf_ids.bucket(member, ncf_ids.NB_USERS)
            if b in taken:
                continue
            taken[b] = member
            users.append((cid, member, b, 0))
            accepted += 1
    return users


def generate(spec: Mapping, deal_ids: Sequence[str]) -> list[Row]:
    """Every user labelled against every seed deal exactly once."""
    validate(spec, deal_ids)
    rng = random.Random(int(spec["generator_seed"]))
    noise = float(spec["noise_rate"])
    by_id = {c["id"]: c for c in spec["cohorts"]}
    item_buckets = [(d, ncf_ids.bucket(d, ncf_ids.NB_ITEMS)) for d in deal_ids]
    rows: list[Row] = []
    for cid, user, ub, scenario in assign_users(spec):
        engaged = positives(by_id[cid])
        for deal, ib in item_buckets:
            label = 1 if deal in engaged else 0
            if not scenario and rng.random() < noise:
                label = 1 - label
            rows.append(Row(user, ub, deal, ib, label, cid, scenario))
    return rows


def to_csv_bytes(rows: Iterable[Row]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    for row in rows:
        writer.writerow(row.as_csv())
    return buffer.getvalue().encode("utf-8")


def stats(rows: Sequence[Row], spec: Mapping) -> dict:
    """Counts for the manifest, including how many generated labels were flipped."""
    by_id = {c["id"]: c for c in spec["cohorts"]}
    generated = [r for r in rows if not r.is_scenario_user]
    flipped = sum(1 for r in generated if r.label != (1 if r.deal_id in positives(by_id[r.cohort]) else 0))
    return {
        "rows": len(rows),
        "users": len({r.user_id for r in rows}),
        "scenario_users": len({r.user_id for r in rows if r.is_scenario_user}),
        "generated_members": len({r.user_id for r in generated}),
        "deals": len({r.deal_id for r in rows}),
        "positive_rows": sum(r.label for r in rows),
        "generated_rows": len(generated),
        "flipped_generated_labels": flipped,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--cohorts", default=DEFAULT_COHORTS)
    parser.add_argument("--seed-file", default=DEFAULT_SEED)
    parser.add_argument("--out", required=True, help="where to write interactions.csv")
    args = parser.parse_args(argv)
    spec = load_json(args.cohorts)
    rows = generate(spec, seed_deal_ids(load_json(args.seed_file)))
    with open(args.out, "wb") as handle:
        handle.write(to_csv_bytes(rows))
    print(f"generator seed {spec['generator_seed']}: wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
