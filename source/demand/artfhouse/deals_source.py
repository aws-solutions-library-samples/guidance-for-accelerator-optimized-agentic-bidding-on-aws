"""Where the demand service's campaign and deal records come from.
Campaigns live in the stack's deals table (``<stack>-deals``, deployed by
deployment/deals_table_cfn.yaml and seeded from deployment/seed/
deals_and_campaigns.json). A campaign is a buyer-side record that references
IAB Tech Lab Deals API v1.0 deals by ``Deal.id``; neither the Deals API nor
OpenRTB defines a campaign object, so its fields are this catalog's own.
Exactly one source is configured through the environment:
  DEALS_TABLE      the DynamoDB table name (the deployed pod)
  DEALS_SEED_FILE  a path to the seed JSON file (unit tests and local runs)
Neither or both is a configuration error. There is no fallback from one to the
other.
These functions are called only by snapshot.py, at startup and from its refresh
thread. The bid path reads the in-memory snapshot and never reaches this module.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any, Iterable, Optional

ENV_TABLE = "DEALS_TABLE"
ENV_SEED_FILE = "DEALS_SEED_FILE"

#: Storage attributes the seeder adds to every item; not part of the record.
STORAGE_KEYS = ("record_type", "seq")

#: Integer-valued campaign fields. DynamoDB returns every number as Decimal.
_INT_FIELDS = frozenset({"width", "height", "duration", "seq"})


def configured_source() -> tuple[str, str]:
    """``("table", name)`` or ``("file", path)``, from the environment."""
    table = os.environ.get(ENV_TABLE, "").strip()
    seed = os.environ.get(ENV_SEED_FILE, "").strip()
    if bool(table) == bool(seed):
        raise ValueError(
            f"set exactly one of {ENV_TABLE} (DynamoDB table name) or "
            f"{ENV_SEED_FILE} (seed JSON path); got "
            f"{ENV_TABLE}={table or '<unset>'} {ENV_SEED_FILE}={seed or '<unset>'}"
        )
    return ("table", table) if table else ("file", seed)


def _plain(value: Any, key: Optional[str] = None) -> Any:
    """Decimal to int or float, recursively; integral int fields to int."""
    if isinstance(value, Decimal):
        if key in _INT_FIELDS and value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {k: _plain(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v, key) for v in value]
    return value


def _strip(items: Iterable[dict]) -> list[dict]:
    ordered = sorted(items, key=lambda item: int(item.get("seq", 0)))
    return [{k: v for k, v in item.items() if k not in STORAGE_KEYS} for item in ordered]


def query_records(table: Any, record_type: str) -> list[dict]:
    """Every item of one record type from a boto3 ``Table``, all pages, in seed order."""
    from boto3.dynamodb.conditions import Key

    kwargs: dict[str, Any] = {"KeyConditionExpression": Key("record_type").eq(record_type)}
    items: list[dict] = []
    while True:
        page = table.query(**kwargs)
        items.extend(page.get("Items") or [])
        last = page.get("LastEvaluatedKey")
        if not last:
            break
        kwargs["ExclusiveStartKey"] = last
    return [_plain(item) for item in _strip(items)]


def _table(name: str) -> Any:
    import boto3

    return boto3.resource("dynamodb").Table(name)


def campaign_records() -> list[dict]:
    """Campaign records from the configured source, in seed order."""
    kind, where = configured_source()
    if kind == "table":
        return query_records(_table(where), "campaign")
    with open(where, encoding="utf-8") as handle:
        seed = json.load(handle)
    return [dict(record) for record in seed["campaigns"]]


def deal_records() -> list[dict]:
    """Deal records from the configured source, in seed order."""
    kind, where = configured_source()
    if kind == "table":
        return query_records(_table(where), "deal")
    with open(where, encoding="utf-8") as handle:
        seed = json.load(handle)
    return [dict(record) for record in seed["deals"]]


def describe_source() -> str:
    """A printable name for the configured source, for log lines."""
    try:
        kind, where = configured_source()
    except ValueError as exc:
        return f"<misconfigured: {exc}>"
    return f"{ENV_TABLE}={where}" if kind == "table" else f"{ENV_SEED_FILE}={where}"
