#!/usr/bin/env python3
"""Validate the deals seed and write it into the deals table.

Run by deploy.sh Phase 1 Step 1.5 after the ``<stack>-deals`` stack is up:

    seed_deals_table.py --table <stack>-deals --role-arn <DealsSeedRoleArn> \\
        --seed deployment/seed/deals_and_campaigns.json --region <region> \\
        --origin-prefix "<stack prefix>" [--campaign-status paused --skip-managed]

The seed holds IAB Tech Lab Deals API v1.0 Deal objects and the demand
endpoint's campaign records. Every deal's ``origin`` is stored in the seed as
the template ``{stackprefix}-prebid.example`` and rendered here from the deploy
prefix. Each record is written as one item with two storage attributes the
readers strip: ``record_type`` (``deal`` or ``campaign``, the partition key) and
``seq`` (its position in the seed, so readers keep the seed's order). The item's
sort key ``id`` is ``Deal.id`` for a deal and ``campaign_id`` for a campaign.

The rules a record must satisfy live in source/control_plane/records.py, which
the campaign/deals control plane applies to every tool call. One validator for
both writers, so an agent cannot write what the seed could not.

Campaign status. ``--campaign-status`` is written onto every campaign item.
deploy.sh passes ``paused`` when the control plane is being deployed, so that
activating a campaign is a step taken through the control plane, and ``active``
otherwise, so a stack without a control plane bids as it always has.

Control-plane edits. With ``--skip-managed`` every put carries the condition
``attribute_not_exists(managed_by)``: an item the control plane has written is
left alone, and the conditional failure is counted as skipped rather than
treated as an error. A batch writer cannot carry a condition, so that path puts
one item at a time; at a few dozen records the difference is immaterial.

The write is otherwise put-only and idempotent: rerunning rewrites the same
items. A record removed from the seed file stays in the table until deleted by
hand, because the seeding role cannot delete.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from decimal import Decimal
from typing import Any, Iterable, Iterator

# The shared record rules live with the control plane under source/, which is
# not on the path when deploy.sh runs this script from deployment/scripts/.
_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from control_plane import records  # noqa: E402

ORIGIN_TEMPLATE = records.ORIGIN_TEMPLATE
MANAGED_BY = records.MANAGED_BY_CONTROL_PLANE


class SeedError(ValueError):
    """The seed file breaks the Deals API schema or its own consistency rules."""


def validate(seed: dict) -> None:
    """Raise SeedError listing every problem; return None when the seed is valid."""
    try:
        records.validate_seed(seed)
    except records.RecordError as exc:
        raise SeedError(str(exc)) from exc


def render_origin(prefix: str) -> str:
    """``origin`` for every deal: ``<prefix>-prebid.example``, or ``prebid.example`` with no prefix."""
    try:
        return records.render_origin(prefix)
    except records.RecordError as exc:
        raise SeedError(str(exc)) from exc


def _decimals(value: Any) -> Any:
    """Floats to Decimal (via their JSON text), as DynamoDB requires."""
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, dict):
        return {k: _decimals(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimals(v) for v in value]
    return value


def items(seed: dict, origin: str, campaign_status: str = records.CAMPAIGN_ACTIVE) -> Iterator[dict]:
    """Every table item: deals then campaigns, each with record_type and seq.

    ``campaign_status`` is written onto every campaign; see the module note.
    """
    if campaign_status not in records.CAMPAIGN_STATUSES:
        raise SeedError(f"campaign status must be one of {list(records.CAMPAIGN_STATUSES)}")
    for seq, deal in enumerate(seed["deals"]):
        record = dict(deal)
        if record.get("origin") == ORIGIN_TEMPLATE:
            record["origin"] = origin
        yield _decimals({**record, "record_type": "deal", "seq": seq})
    for seq, campaign in enumerate(seed["campaigns"]):
        record = {k: v for k, v in campaign.items()}
        record["status"] = campaign_status
        yield _decimals({**record, "record_type": "campaign", "id": campaign["campaign_id"], "seq": seq})


def write(table: Any, rows: Iterable[dict]) -> int:
    """Put every row through a batch writer. Returns the number written."""
    count = 0
    with table.batch_writer() as writer:
        for row in rows:
            writer.put_item(Item=row)
            count += 1
    return count


def write_unmanaged(table: Any, rows: Iterable[dict]) -> tuple[int, int]:
    """Put each row unless the control plane has written that item.

    Returns ``(written, skipped)``. The condition is on ``managed_by``, which only
    the control plane sets, so a seeded item that was never edited keeps receiving
    seed updates and an edited one keeps the edit.
    """
    written = skipped = 0
    for row in rows:
        try:
            table.put_item(Item=row, ConditionExpression="attribute_not_exists(managed_by)")
            written += 1
        except Exception as exc:  # noqa: BLE001 - only the conditional failure is a skip
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code != "ConditionalCheckFailedException":
                raise
            skipped += 1
    return written, skipped


# The seed role is created by the stack seconds before this runs, and IAM can
# refuse a new role or its inline policy until it propagates. Puts are
# idempotent, so a refused attempt is retried whole.
_ACCESS_DENIED = {"AccessDenied", "AccessDeniedException"}
SEED_ATTEMPTS = 6
SEED_RETRY_SECONDS = 10


def _seed_once(args: argparse.Namespace, seed: dict, origin: str) -> tuple[int, int]:
    import boto3

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    creds = session.client("sts").assume_role(
        RoleArn=args.role_arn, RoleSessionName="seed-deals-table",
    )["Credentials"]
    seeded = boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=args.region,
    )
    table = seeded.resource("dynamodb").Table(args.table)
    rows = items(seed, origin, args.campaign_status)
    if args.skip_managed:
        return write_unmanaged(table, rows)
    return write(table, rows), 0


def _seed_with_retry(args: argparse.Namespace, seed: dict, origin: str,
                     attempts: int = SEED_ATTEMPTS, delay: float = SEED_RETRY_SECONDS,
                     sleep=time.sleep):
    for attempt in range(1, attempts + 1):
        try:
            return _seed_once(args, seed, origin)
        except Exception as exc:  # noqa: BLE001 - only access-denied is retried
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in _ACCESS_DENIED or attempt == attempts:
                raise
            print(
                f"seed_deals_table: {code} on attempt {attempt} of {attempts}, "
                f"waiting {delay:g}s for IAM propagation",
                file=sys.stderr,
            )
            sleep(delay)
    raise AssertionError("unreachable")


def load_seed(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", required=True, help="deals table name")
    parser.add_argument("--role-arn", required=True, help="DealsSeedRoleArn from the deals stack")
    parser.add_argument("--seed", required=True, help="path to deals_and_campaigns.json")
    parser.add_argument("--region", required=True)
    parser.add_argument("--origin-prefix", default="", help="deploy prefix used to render Deal.origin")
    parser.add_argument("--profile", default=None, help="AWS profile for the identity that assumes the role")
    parser.add_argument(
        "--campaign-status", choices=list(records.CAMPAIGN_STATUSES), default=records.CAMPAIGN_ACTIVE,
        help="status written onto every seeded campaign (paused when a control plane will activate them)",
    )
    parser.add_argument(
        "--skip-managed", action="store_true",
        help="leave items the control plane has written (managed_by set) untouched",
    )
    args = parser.parse_args(argv)

    try:
        seed = load_seed(args.seed)
        validate(seed)
        origin = render_origin(args.origin_prefix)
    except (OSError, ValueError) as exc:
        print(f"seed_deals_table: {exc}", file=sys.stderr)
        return 1

    try:
        result = _seed_with_retry(args, seed, origin)
    except Exception as exc:  # noqa: BLE001 - any AWS failure fails the deploy step
        print(f"seed_deals_table: writing {args.table} failed: {exc}", file=sys.stderr)
        return 1

    written, skipped = result if isinstance(result, tuple) else (int(result), 0)
    skipped_note = f", {skipped} left to the control plane" if skipped else ""
    print(
        f"seeded {args.table}: {len(seed['deals'])} deals, {len(seed['campaigns'])} campaigns "
        f"({written} items written{skipped_note}), campaigns {args.campaign_status}, origin {origin}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
