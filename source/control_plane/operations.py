"""The control plane's ten operations, over the deals table.

IMPORTS NOTHING FROM ``mcp``, DELIBERATELY. The tool registration lives in server.py,
so every operation here is tested without the SDK and the untested surface is the
registration layer alone.

NO I/O OF ITS OWN. Operations run against a ``Store`` (below). The deployed runtime
uses ``DynamoStore``; tests use an in-memory one. ``DynamoStore`` is the only place
boto3 is touched, and it is constructed by server.py, never at import.

WHAT A WRITE LOOKS LIKE. Every item the control plane writes carries two marks the
readers strip: ``managed_by: control-plane`` (so the seeder's ``--skip-managed`` leaves
it alone) and ``updated_at`` (ISO-8601 UTC). A record created here gets a ``seq`` far
above any seed position, so the readers, which order by ``seq``, list seeded records
first and agent-created ones after, in creation order.

WHAT IS REFUSED, as structured errors naming the rule rather than silent corrections:

  - a record that fails control_plane/records.py (the same rules the seed obeys)
  - a campaign holding a deal another campaign holds (the demand catalog raises on a
    deal claimed twice, which would take the whole snapshot down)
  - a CPM above the price ceiling (Prebid discards a bid above its top price bucket,
    so the campaign would be accepted and never win)
  - activating a campaign that could never offer (no active deal and not open market)
  - any change to an archived deal (Deals API: archived cannot be active again)
  - a caller-supplied ``origin`` on a deal (the stack is the origin; it sets the value)

THERE IS NO DELETE. A campaign is paused; a deal's ``sellerstatus`` is changed. The
runtime's IAM role has no DeleteItem either, so this is not a convention.

PROPAGATION. The demand service and the Deal Scorer refresh their snapshots every
30 seconds; every write response says so (``visible_within_seconds``), because an
agent that reads the record back immediately and sees the change should not assume
the bid path has it yet.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

from . import records
from .scope import Entitlement, visible

#: How long after a write the request-path readers may still serve the old snapshot.
VISIBLE_WITHIN_SECONDS = 30

#: Default ceiling on a campaign's CPM (declared plus uplift), USD. The seed's top
#: ceiling is 13.75; the adapter's bid ceiling must exceed whatever this is.
DEFAULT_PRICE_CEILING = 50.0

RECORD_CAMPAIGN = "campaign"
RECORD_DEAL = "deal"

#: ``seq`` floor for records created here, so they sort after the seed's positions.
CREATED_SEQ_BASE = 1_000_000

#: Campaign fields a caller may set on create or change on update. ``status`` moves
#: only through activate/pause; the table keys and marks are never caller-supplied.
CAMPAIGN_INPUT_FIELDS = records.CAMPAIGN_FIELDS - {"status"}
CAMPAIGN_UPDATABLE_FIELDS = CAMPAIGN_INPUT_FIELDS - {"campaign_id"}

#: Deal fields a caller may set. ``origin`` is the stack's; ``created`` is set here.
DEAL_INPUT_FIELDS = records.DEAL_FIELDS - {"origin", "created"}
DEAL_UPDATABLE_FIELDS = DEAL_INPUT_FIELDS - {"id"}


class InvalidRequest(ValueError):
    """The request cannot be carried out as asked, and says why."""


class NotFound(LookupError):
    """No such record, for a caller entitled to know that."""


class AlreadyExists(ValueError):
    """A record with this id exists; create does not overwrite."""


class Store(Protocol):
    """The table operations the control plane needs. Implemented by DynamoStore and by
    the in-memory store the tests use."""

    def get(self, record_type: str, record_id: str) -> Optional[dict]: ...

    def query(self, record_type: str) -> list[dict]: ...

    def put_new(self, item: Mapping[str, Any]) -> None:
        """Write ``item`` only if no item has its keys; raise AlreadyExists otherwise."""
        ...

    def put(self, item: Mapping[str, Any]) -> None: ...


# ----------------------------------------------------------------- conversions


def to_decimals(value: Any) -> Any:
    """Floats to Decimal via their text, as DynamoDB requires (4.25 stays 4.25)."""
    if isinstance(value, float):
        return Decimal(repr(value))
    if isinstance(value, dict):
        return {k: to_decimals(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_decimals(v) for v in value]
    return value


def to_plain(value: Any) -> Any:
    """Decimal back to int or float, recursively, for JSON responses."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    return value


def public_record(item: Mapping[str, Any]) -> dict:
    """The record as a reader sees it: table keys and marks removed.

    A campaign item's sort key ``id`` duplicates ``campaign_id`` and is dropped; a
    deal's ``id`` is the Deal object's own field and is kept.
    """
    drop = set(records.STORAGE_KEYS)
    if item.get("record_type") == RECORD_CAMPAIGN:
        drop.add("id")
    return to_plain({k: v for k, v in item.items() if k not in drop})


def _now_iso(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock(), tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ------------------------------------------------------------------ operations


class ControlPlaneOperations:
    """The tools' logic, over a Store."""

    def __init__(
        self,
        store: Store,
        *,
        deal_origin: str,
        price_ceiling: float = DEFAULT_PRICE_CEILING,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._origin = deal_origin
        self._ceiling = float(price_ceiling)
        self._clock = clock

    # -- helpers -----------------------------------------------------------

    def _marks(self, *, seq: Optional[int] = None) -> dict:
        marks: dict[str, Any] = {
            "managed_by": records.MANAGED_BY_CONTROL_PLANE,
            "updated_at": _now_iso(self._clock),
        }
        if seq is not None:
            marks["seq"] = seq
        return marks

    def _next_seq(self) -> int:
        return CREATED_SEQ_BASE + int(self._clock() * 1000) % 1_000_000_000

    def _campaign_item(self, entitlement: Entitlement, campaign_id: str) -> dict:
        item = self._store.get(RECORD_CAMPAIGN, campaign_id)
        if item is None:
            raise NotFound(f"no campaign {campaign_id!r}")
        # Entitlement is checked AFTER the load but BEFORE anything is returned, so the
        # answer for "not found" and "not yours" is the same and ids cannot be probed.
        try:
            entitlement.require(str(item.get("adomain") or ""), kind="advertiser")
        except PermissionError:
            raise NotFound(f"no campaign {campaign_id!r}")
        return item

    def _deal_item(self, entitlement: Entitlement, deal_id: str) -> dict:
        item = self._store.get(RECORD_DEAL, deal_id)
        if item is None:
            raise NotFound(f"no deal {deal_id!r}")
        try:
            entitlement.require(str(item.get("seller") or ""), kind="seller")
        except PermissionError:
            raise NotFound(f"no deal {deal_id!r}")
        return item

    def _deal_index(self) -> tuple[set[str], set[str]]:
        """``(all deal ids, active deal ids)`` from the table."""
        deals = self._store.query(RECORD_DEAL)
        ids = {str(d.get("id")) for d in deals}
        active = {str(d.get("id")) for d in deals if records.deal_is_active(d)}
        return ids, active

    def _held_by(self) -> dict[str, str]:
        """Deal id to the campaign id that holds it, across every campaign."""
        held: dict[str, str] = {}
        for campaign in self._store.query(RECORD_CAMPAIGN):
            for deal_id in campaign.get("deal_ids") or ():
                held.setdefault(str(deal_id), str(campaign.get("campaign_id")))
        return held

    @staticmethod
    def _raise_if_problems(problems: list[str]) -> None:
        if problems:
            raise InvalidRequest("; ".join(problems))

    def _write_result(self, record_type: str, item: Mapping[str, Any], **extra: Any) -> dict:
        return {
            record_type: public_record(item),
            "visible_within_seconds": VISIBLE_WITHIN_SECONDS,
            **extra,
        }

    # -- campaigns ---------------------------------------------------------

    def create_campaign(self, entitlement: Entitlement, **fields: Any) -> dict:
        """Create a campaign. Always ``paused``; ``activate_campaign`` makes it biddable."""
        unknown = set(fields) - CAMPAIGN_INPUT_FIELDS
        if unknown:
            raise InvalidRequest(f"unknown campaign fields {sorted(unknown)}")
        record = {k: v for k, v in fields.items() if v is not None}
        record.setdefault("deal_ids", [])
        record.setdefault("open_market", False)
        record.setdefault("target_categories", [])
        record.setdefault("media_type", "banner")
        campaign_id = record.get("campaign_id")
        if not records.valid_record_id(campaign_id):
            raise InvalidRequest(
                f"campaign_id must be 1 to {records.RECORD_ID_MAX} lowercase letters, digits and hyphens"
            )
        # Entitlement before the table is read, so an unentitled caller learns nothing
        # about which deals exist.
        entitlement.require(str(record.get("adomain") or ""), kind="advertiser")
        deal_ids, _active = self._deal_index()
        self._raise_if_problems(records.validate_campaign(
            record, known_deal_ids=deal_ids, held_by=self._held_by(),
            price_ceiling=self._ceiling, strict_id=True,
        ))
        record["status"] = records.CAMPAIGN_PAUSED
        item = to_decimals({
            **record, "record_type": RECORD_CAMPAIGN, "id": campaign_id,
            **self._marks(seq=self._next_seq()),
        })
        try:
            self._store.put_new(item)
        except AlreadyExists:
            raise InvalidRequest(f"a campaign with id {campaign_id!r} already exists")
        return self._write_result(RECORD_CAMPAIGN, item)

    def update_campaign(self, entitlement: Entitlement, *, campaign_id: str, **changes: Any) -> dict:
        """Change the supplied fields of a campaign. Status moves only through activate/pause."""
        unknown = set(changes) - CAMPAIGN_UPDATABLE_FIELDS
        if unknown:
            raise InvalidRequest(f"fields that cannot be updated: {sorted(unknown)}")
        changes = {k: v for k, v in changes.items() if v is not None}
        if not changes:
            raise InvalidRequest("no fields to update were supplied")
        item = self._campaign_item(entitlement, campaign_id)
        if "adomain" in changes:
            # Moving a campaign to another advertiser needs entitlement to BOTH.
            entitlement.require(str(changes["adomain"]), kind="advertiser")
        merged = {**public_record(item), **changes}
        deal_ids, _active = self._deal_index()
        self._raise_if_problems(records.validate_campaign(
            merged, known_deal_ids=deal_ids, held_by=self._held_by(),
            price_ceiling=self._ceiling, strict_id=True,
        ))
        new_item = to_decimals({
            **merged, "record_type": RECORD_CAMPAIGN, "id": campaign_id,
            "seq": item.get("seq", self._next_seq()), **self._marks(),
        })
        self._store.put(new_item)
        return self._write_result(RECORD_CAMPAIGN, new_item, changed=sorted(changes))

    def activate_campaign(self, entitlement: Entitlement, *, campaign_id: str) -> dict:
        """Make a campaign biddable. Refuses, with the reason, one that could never offer."""
        item = self._campaign_item(entitlement, campaign_id)
        _ids, active = self._deal_index()
        held = [str(d) for d in item.get("deal_ids") or ()]
        if not item.get("open_market") and not any(d in active for d in held):
            raise InvalidRequest(
                f"campaign {campaign_id!r} holds no active deal and is not open market, so "
                f"activating it would change nothing; activate one of its deals "
                f"({held or 'none held'}) or set open_market"
            )
        return self._set_campaign_status(item, records.CAMPAIGN_ACTIVE)

    def pause_campaign(self, entitlement: Entitlement, *, campaign_id: str) -> dict:
        """Stop a campaign bidding. The record is kept; there is no delete."""
        item = self._campaign_item(entitlement, campaign_id)
        return self._set_campaign_status(item, records.CAMPAIGN_PAUSED)

    def _set_campaign_status(self, item: dict, status: str) -> dict:
        previous = records.campaign_status(item)
        new_item = {**item, "status": status, **self._marks()}
        self._store.put(new_item)
        return self._write_result(RECORD_CAMPAIGN, new_item, previous_status=previous)

    def get_campaign(self, entitlement: Entitlement, *, campaign_id: str) -> dict:
        item = self._campaign_item(entitlement, campaign_id)
        _ids, active = self._deal_index()
        record = public_record(item)
        record["status"] = records.campaign_status(record)
        return {
            RECORD_CAMPAIGN: record,
            "active_deal_ids": [d for d in record.get("deal_ids") or [] if d in active],
            "updated_at": item.get("updated_at"),
            "managed_by": item.get("managed_by", "seed"),
        }

    def list_campaigns(
        self, entitlement: Entitlement, *, status: Optional[str] = None, advertiser: Optional[str] = None,
    ) -> dict:
        if status is not None and status not in records.CAMPAIGN_STATUSES:
            raise InvalidRequest(f"status must be one of {list(records.CAMPAIGN_STATUSES)}")
        items = visible(entitlement, self._store.query(RECORD_CAMPAIGN), key="adomain")
        out = []
        for item in items:
            record = public_record(item)
            record["status"] = records.campaign_status(record)
            if status and record["status"] != status:
                continue
            if advertiser and str(record.get("adomain", "")).lower() != advertiser.strip().lower():
                continue
            out.append({
                "campaign_id": record["campaign_id"], "campaign_name": record.get("campaign_name"),
                "adomain": record.get("adomain"), "status": record["status"],
                "declared_cpm": record.get("declared_cpm"), "deal_ids": record.get("deal_ids") or [],
                "open_market": bool(record.get("open_market")), "media_type": record.get("media_type"),
                "managed_by": item.get("managed_by", "seed"),
            })
        return {"campaigns": out, "count": len(out)}

    # -- deals -------------------------------------------------------------

    def create_deal(self, entitlement: Entitlement, *, deal: Mapping[str, Any]) -> dict:
        """Create a Deals API v1.0 deal. ``origin`` is the stack's; ``created`` is now."""
        if not isinstance(deal, Mapping):
            raise InvalidRequest("deal must be an object")
        if "origin" in deal:
            raise InvalidRequest(
                f"origin is not caller-settable; this stack writes {self._origin!r}"
            )
        unknown = set(deal) - DEAL_INPUT_FIELDS
        if unknown:
            raise InvalidRequest(f"not Deals API Deal fields, or not settable here: {sorted(unknown)}")
        record = dict(deal)
        deal_id = record.get("id")
        if not records.valid_record_id(deal_id):
            raise InvalidRequest(
                f"id must be 1 to {records.RECORD_ID_MAX} lowercase letters, digits and hyphens"
            )
        seller = str(record.get("seller") or "")
        if not seller:
            # Recommended, not required, by the spec; required here because it is the
            # principal the entitlement is checked against.
            raise InvalidRequest("seller is required: it names who may manage this deal")
        entitlement.require(seller, kind="seller")
        record["origin"] = self._origin
        record["created"] = _now_iso(self._clock)
        record.setdefault("sellerstatus", records.SELLERSTATUS_ACTIVE)
        self._raise_if_problems(records.validate_deal(record, strict_id=True))
        item = to_decimals({**record, "record_type": RECORD_DEAL, **self._marks(seq=self._next_seq())})
        try:
            self._store.put_new(item)
        except AlreadyExists:
            raise InvalidRequest(f"a deal with id {deal_id!r} already exists")
        return self._write_result(RECORD_DEAL, item)

    def update_deal(self, entitlement: Entitlement, *, deal_id: str, changes: Mapping[str, Any]) -> dict:
        """Change the supplied fields of a deal, ``sellerstatus`` included. Archived is final."""
        if not isinstance(changes, Mapping) or not changes:
            raise InvalidRequest("changes must be a non-empty object")
        unknown = set(changes) - DEAL_UPDATABLE_FIELDS
        if unknown:
            raise InvalidRequest(f"fields that cannot be updated: {sorted(unknown)}")
        item = self._deal_item(entitlement, deal_id)
        current = public_record(item)
        if int(current.get("sellerstatus", 0) or 0) == records.SELLERSTATUS_ARCHIVED:
            raise InvalidRequest(f"deal {deal_id!r} is archived (sellerstatus 4) and cannot change")
        if "seller" in changes:
            entitlement.require(str(changes["seller"]), kind="seller")
        merged = {**current, **{k: v for k, v in changes.items() if v is not None}}
        self._raise_if_problems(records.validate_deal(merged, strict_id=True))
        new_item = to_decimals({
            **merged, "record_type": RECORD_DEAL, "seq": item.get("seq", self._next_seq()), **self._marks(),
        })
        self._store.put(new_item)
        return self._write_result(
            RECORD_DEAL, new_item, changed=sorted(changes),
            sellerstatus_name=records.sellerstatus_name(merged.get("sellerstatus", 0)),
        )

    def get_deal(self, entitlement: Entitlement, *, deal_id: str) -> dict:
        item = self._deal_item(entitlement, deal_id)
        record = public_record(item)
        held_by = [c for d, c in self._held_by().items() if d == deal_id]
        return {
            RECORD_DEAL: record,
            "sellerstatus_name": records.sellerstatus_name(record.get("sellerstatus", 0)),
            "active": records.deal_is_active(record),
            "held_by_campaigns": held_by,
            "updated_at": item.get("updated_at"),
            "managed_by": item.get("managed_by", "seed"),
        }

    def list_deals(
        self, entitlement: Entitlement, *, sellerstatus: Optional[int] = None, seller: Optional[str] = None,
    ) -> dict:
        if sellerstatus is not None and not (isinstance(sellerstatus, int) and 0 <= sellerstatus <= 4):
            raise InvalidRequest("sellerstatus must be an integer from 0 to 4")
        items = visible(entitlement, self._store.query(RECORD_DEAL), key="seller")
        held = self._held_by()
        out = []
        for item in items:
            record = public_record(item)
            status = int(record.get("sellerstatus", 0) or 0)
            if sellerstatus is not None and status != sellerstatus:
                continue
            if seller and str(record.get("seller", "")).lower() != seller.strip().lower():
                continue
            terms = record.get("terms") or {}
            inventory = record.get("inventory") or {}
            out.append({
                "id": record["id"], "name": record.get("name"), "seller": record.get("seller"),
                "sellerstatus": status, "sellerstatus_name": records.sellerstatus_name(status),
                "dealfloor": terms.get("dealfloor"), "cur": terms.get("cur", "USD"),
                "pricetype": terms.get("pricetype"), "sitedomains": inventory.get("sitedomains") or [],
                "appbundles": inventory.get("appbundles") or [],
                "held_by_campaign": held.get(record["id"]),
                "managed_by": item.get("managed_by", "seed"),
            })
        return {"deals": out, "count": len(out)}


# --------------------------------------------------------------- the stores


class MemoryStore:
    """A dict-backed Store for tests and local runs. Items are copied on the way in and out."""

    def __init__(self, items: Iterable[Mapping[str, Any]] = ()) -> None:
        self._items: dict[tuple[str, str], dict] = {}
        for item in items:
            self.put(item)

    def _key(self, item: Mapping[str, Any]) -> tuple[str, str]:
        return (str(item["record_type"]), str(item["id"]))

    def get(self, record_type: str, record_id: str) -> Optional[dict]:
        item = self._items.get((record_type, record_id))
        return dict(item) if item is not None else None

    def query(self, record_type: str) -> list[dict]:
        found = [dict(v) for (t, _i), v in self._items.items() if t == record_type]
        return sorted(found, key=lambda i: int(i.get("seq", 0)))

    def put_new(self, item: Mapping[str, Any]) -> None:
        if self._key(item) in self._items:
            raise AlreadyExists(str(item["id"]))
        self._items[self._key(item)] = dict(item)

    def put(self, item: Mapping[str, Any]) -> None:
        self._items[self._key(item)] = dict(item)


class DynamoStore:
    """The Store over a boto3 ``Table``. Constructed by server.py; never at import."""

    def __init__(self, table: Any) -> None:
        self._table = table

    def get(self, record_type: str, record_id: str) -> Optional[dict]:
        response = self._table.get_item(Key={"record_type": record_type, "id": record_id})
        item = response.get("Item")
        return dict(item) if item else None

    def query(self, record_type: str) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        kwargs: dict[str, Any] = {"KeyConditionExpression": Key("record_type").eq(record_type)}
        items: list[dict] = []
        while True:
            page = self._table.query(**kwargs)
            items.extend(dict(i) for i in page.get("Items") or [])
            last = page.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
        return sorted(items, key=lambda i: int(i.get("seq", 0)))

    def put_new(self, item: Mapping[str, Any]) -> None:
        try:
            self._table.put_item(Item=dict(item), ConditionExpression="attribute_not_exists(record_type)")
        except Exception as exc:  # noqa: BLE001 - only the conditional failure is translated
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise AlreadyExists(str(item["id"])) from exc
            raise

    def put(self, item: Mapping[str, Any]) -> None:
        self._table.put_item(Item=dict(item))
