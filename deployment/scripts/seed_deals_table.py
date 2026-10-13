#!/usr/bin/env python3
"""Validate the deals seed and write it into the deals table.

Run by deploy.sh Phase 1 Step 1.5 after the ``<stack>-deals`` stack is up:

    seed_deals_table.py --table <stack>-deals --role-arn <DealsSeedRoleArn> \\
        --seed deployment/seed/deals_and_campaigns.json --region <region> \\
        --origin-prefix "<stack prefix>"

The seed holds IAB Tech Lab Deals API v1.0 Deal objects and the demand
endpoint's campaign records. Every deal's ``origin`` is stored in the seed as
the template ``{stackprefix}-prebid.example`` and rendered here from the deploy
prefix. Each record is written as one item with two storage attributes the
readers strip: ``record_type`` (``deal`` or ``campaign``, the partition key) and
``seq`` (its position in the seed, so readers keep the seed's order). The item's
sort key ``id`` is ``Deal.id`` for a deal and ``campaign_id`` for a campaign.

The write is put-only and idempotent: rerunning rewrites the same items. A
record removed from the seed file stays in the table until deleted by hand,
because the seeding role cannot delete.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from decimal import Decimal
from typing import Any, Iterable, Iterator

ORIGIN_TEMPLATE = "{stackprefix}-prebid.example"
_DNS_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

#: Deals API v1.0 field names, by object.
DEAL_FIELDS = frozenset({
    "id", "name", "created", "sellerstatus", "origin", "seller", "desc", "wseat",
    "bseat", "adtypes", "auxdata", "pubcount", "dinventory", "terms", "inventory",
    "curation", "ext",
})
TERMS_FIELDS = frozenset({
    "startdate", "enddate", "countries", "dealfloor", "cur", "guar", "pricetype",
    "units", "totalcost", "ext",
})
INVENTORY_FIELDS = frozenset({
    "inclinventory", "devicetype", "sellerids", "sitedomains", "appbundles", "cat",
    "cattax", "ext",
})
#: The only Terms extension this stack defines.
TERMS_EXT_FIELDS = frozenset({"mindealfloor"})

CAMPAIGN_FIELDS = frozenset({
    "campaign_id", "campaign_name", "adomain", "creative_id", "width", "height",
    "declared_cpm", "deal_ids", "open_market", "target_categories", "media_type",
    "duration", "audience_uplift", "note",
})
CAMPAIGN_REQUIRED = CAMPAIGN_FIELDS - {"note", "duration", "audience_uplift"}


class SeedError(ValueError):
    """The seed file breaks the Deals API schema or its own consistency rules."""


def _int_in(value: Any, allowed) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and allowed(value)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_enum(problems: list[str], where: str, name: str, value: Any, allowed) -> None:
    if not _int_in(value, allowed):
        problems.append(f"{where}: {name}={value!r} is outside the Deals API range")


def _check_enum_list(problems: list[str], where: str, name: str, value: Any, allowed) -> None:
    if not isinstance(value, list) or not value or not all(_int_in(v, allowed) for v in value):
        problems.append(f"{where}: {name}={value!r} must be a non-empty list of values in range")


def validate(seed: dict) -> None:
    """Raise SeedError listing every problem; return None when the seed is valid."""
    problems: list[str] = []
    deals = seed.get("deals")
    campaigns = seed.get("campaigns")
    if not isinstance(deals, list) or not deals:
        raise SeedError("seed has no deals list")
    if not isinstance(campaigns, list):
        raise SeedError("seed has no campaigns list")

    deal_ids: set[str] = set()
    for i, deal in enumerate(deals):
        where = f"deals[{i}] {deal.get('id', '<no id>')}"
        for required in ("id", "origin", "terms"):
            if required not in deal:
                problems.append(f"{where}: missing required {required}")
        unknown = set(deal) - DEAL_FIELDS
        if unknown:
            problems.append(f"{where}: not Deals API Deal fields: {sorted(unknown)}")
        deal_id = deal.get("id")
        if not isinstance(deal_id, str) or not deal_id:
            problems.append(f"{where}: id must be a non-empty string")
        elif deal_id in deal_ids:
            problems.append(f"{where}: duplicate deal id")
        else:
            deal_ids.add(deal_id)
        if len(str(deal.get("desc", ""))) > 250:
            problems.append(f"{where}: desc is over 250 characters")
        if "sellerstatus" in deal:
            _check_enum(problems, where, "sellerstatus", deal["sellerstatus"], lambda v: 0 <= v <= 4)
        if "adtypes" in deal:
            _check_enum_list(problems, where, "adtypes", deal["adtypes"], lambda v: 1 <= v <= 4)
        if "auxdata" in deal:
            _check_enum(problems, where, "auxdata", deal["auxdata"], lambda v: 0 <= v <= 4)
        if "pubcount" in deal:
            _check_enum(problems, where, "pubcount", deal["pubcount"], lambda v: 0 <= v <= 2)
        if "dinventory" in deal:
            _check_enum(problems, where, "dinventory", deal["dinventory"], lambda v: 0 <= v <= 2)

        terms = deal.get("terms")
        if not isinstance(terms, dict):
            problems.append(f"{where}: terms must be an object")
            terms = {}
        unknown = set(terms) - TERMS_FIELDS
        if unknown:
            problems.append(f"{where}: not Deals API Terms fields: {sorted(unknown)}")
        floor = terms.get("dealfloor")
        if not _number(floor) or floor < 0:
            problems.append(f"{where}: terms.dealfloor={floor!r} must be a non-negative number")
        if "pricetype" in terms:
            _check_enum(problems, where, "terms.pricetype", terms["pricetype"], lambda v: 0 <= v <= 3 or v >= 500)
        if "guar" in terms:
            _check_enum(problems, where, "terms.guar", terms["guar"], lambda v: v in (0, 1))
        ext = terms.get("ext")
        if ext is not None:
            if not isinstance(ext, dict) or set(ext) - TERMS_EXT_FIELDS:
                problems.append(f"{where}: terms.ext may hold only {sorted(TERMS_EXT_FIELDS)}")
            else:
                minimum = ext.get("mindealfloor")
                if not _number(minimum) or minimum < 0:
                    problems.append(f"{where}: terms.ext.mindealfloor={minimum!r} must be a non-negative number")
                elif _number(floor) and minimum > floor:
                    problems.append(f"{where}: terms.ext.mindealfloor {minimum} is above terms.dealfloor {floor}")

        inventory = deal.get("inventory")
        if inventory is not None:
            if not isinstance(inventory, dict):
                problems.append(f"{where}: inventory must be an object")
            else:
                unknown = set(inventory) - INVENTORY_FIELDS
                if unknown:
                    problems.append(f"{where}: not Deals API Inventory fields: {sorted(unknown)}")
                if "inclinventory" in inventory:
                    _check_enum_list(problems, where, "inventory.inclinventory", inventory["inclinventory"], lambda v: 1 <= v <= 6)

    held: dict[str, str] = {}
    campaign_ids: set[str] = set()
    for i, campaign in enumerate(campaigns):
        cid = campaign.get("campaign_id")
        where = f"campaigns[{i}] {cid or '<no id>'}"
        missing = CAMPAIGN_REQUIRED - set(campaign)
        if missing:
            problems.append(f"{where}: missing {sorted(missing)}")
        unknown = set(campaign) - CAMPAIGN_FIELDS
        if unknown:
            problems.append(f"{where}: unknown campaign fields {sorted(unknown)}")
        if cid in campaign_ids:
            problems.append(f"{where}: duplicate campaign id")
        campaign_ids.add(cid)
        for deal_id in campaign.get("deal_ids") or []:
            if deal_id not in deal_ids:
                problems.append(f"{where}: holds {deal_id}, which is not a deal in the seed")
            if deal_id in held:
                problems.append(f"{where}: {deal_id} is already held by {held[deal_id]}")
            held[deal_id] = cid

    if problems:
        raise SeedError("deals seed is invalid:\n  " + "\n  ".join(problems))


def render_origin(prefix: str) -> str:
    """``origin`` for every deal: ``<prefix>-prebid.example``, or ``prebid.example`` with no prefix."""
    prefix = (prefix or "").strip().lower()
    label = f"{prefix}-prebid" if prefix else "prebid"
    if not _DNS_LABEL.match(label) or len(label) > 63:
        raise SeedError(f"origin prefix {prefix!r} does not make a valid DNS label ({label!r})")
    return f"{label}.example"


def _decimals(value: Any) -> Any:
    """Floats to Decimal (via their JSON text), as DynamoDB requires."""
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, dict):
        return {k: _decimals(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimals(v) for v in value]
    return value


def items(seed: dict, origin: str) -> Iterator[dict]:
    """Every table item: deals then campaigns, each with record_type and seq."""
    for seq, deal in enumerate(seed["deals"]):
        record = dict(deal)
        if record.get("origin") == ORIGIN_TEMPLATE:
            record["origin"] = origin
        yield _decimals({**record, "record_type": "deal", "seq": seq})
    for seq, campaign in enumerate(seed["campaigns"]):
        record = {k: v for k, v in campaign.items()}
        yield _decimals({**record, "record_type": "campaign", "id": campaign["campaign_id"], "seq": seq})


def write(table: Any, rows: Iterable[dict]) -> int:
    """Put every row through a batch writer. Returns the number written."""
    count = 0
    with table.batch_writer() as writer:
        for row in rows:
            writer.put_item(Item=row)
            count += 1
    return count


# The seed role is created by the stack seconds before this runs, and IAM can
# refuse a new role or its inline policy until it propagates. Puts are
# idempotent, so a refused attempt is retried whole.
_ACCESS_DENIED = {"AccessDenied", "AccessDeniedException"}
SEED_ATTEMPTS = 6
SEED_RETRY_SECONDS = 10


def _seed_once(args: argparse.Namespace, seed: dict, origin: str) -> int:
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
    return write(table, items(seed, origin))


def _seed_with_retry(args: argparse.Namespace, seed: dict, origin: str,
                     attempts: int = SEED_ATTEMPTS, delay: float = SEED_RETRY_SECONDS,
                     sleep=time.sleep) -> int:
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
    args = parser.parse_args(argv)

    try:
        seed = load_seed(args.seed)
        validate(seed)
        origin = render_origin(args.origin_prefix)
    except (OSError, ValueError) as exc:
        print(f"seed_deals_table: {exc}", file=sys.stderr)
        return 1

    try:
        written = _seed_with_retry(args, seed, origin)
    except Exception as exc:  # noqa: BLE001 - any AWS failure fails the deploy step
        print(f"seed_deals_table: writing {args.table} failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"seeded {args.table}: {len(seed['deals'])} deals, {len(seed['campaigns'])} campaigns "
        f"({written} items), origin {origin}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
