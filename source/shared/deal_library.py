"""The publisher's deal library: the Deal Scorer's deal book, also read by the orchestrator.

ARTF's sell-side ACTIVATE_DEALS means activating a deal the publisher HAS but did
not offer on this request. The request's own ``imp.pmp.deals`` is what the
publisher chose to expose; this module is the rest of the publisher's deal book,
so the scorer has something to activate from.

Two readers. The Deal Scorer (containers/ncf_deal_manager/app.py) picks library
candidates from it. The orchestrator reads each activated deal's auction type
from it (``auction_type``), so the deal shared/artf_applier.py inserts on
ACTIVATE_DEALS carries OpenRTB ``at`` and the yield stage sees a well-formed
Deal. The ARTF IDsPayload carries ids only, so nothing on the mutation can say it.

Where the book comes from. Deals live in the stack's deals table
(``<stack>-deals``) as IAB Tech Lab Deals API v1.0 Deal objects, seeded from
deployment/seed/deals_and_campaigns.json. The same table holds the demand
endpoint's campaigns, which reference deals by ``Deal.id``. Exactly one source
is configured: ``DEALS_TABLE`` (DynamoDB, the deployed pod) or
``DEALS_SEED_FILE`` (the seed JSON, unit tests and local runs).

Off the request path. ARTF containers make no external call while handling a
request, so the book is an in-memory snapshot:

  * ``start()`` primes it before the server starts. For the Deal Scorer a failed
    prime raises, and app.py exits non-zero, so the pod never becomes ready. There
    is no built-in list to fall back to. The orchestrator calls
    ``start(prime_required=False)``: a failed prime is logged, the refresh thread
    keeps trying, and until a snapshot is installed ``auction_type`` raises and the
    applier inserts activated deals id-only, as it did before the orchestrator had
    a deal book.
  * a daemon thread calls ``refresh_once()`` every ``REFRESH_SECONDS``. A failed
    refresh logs the error and keeps the last good snapshot.
  * ``deals_for``, ``not_on_request`` and ``all_deal_ids`` read only the
    installed snapshot, which is swapped in with a single assignment.

Matching a request to a book follows the Deal records' inventory: a deal applies
when ``inventory.sitedomains`` holds the request's ``site.domain`` or
``inventory.appbundles`` holds its ``app.bundle``. When nothing matches, the
request's content categories are looked up in ``inventory.cat``, and the first
deal found names the seller whose whole book applies. Each record carries the
deal's floor and auction type, because an activated deal reaches the seats with
whatever this container says about it: the ARTF IDsPayload carries ids only, so
the floor is emitted as a separate ADJUST_DEAL_FLOOR mutation (see app.py).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Optional

logger = logging.getLogger("ncf.deal_library")

#: Seconds between background refreshes of the snapshot.
REFRESH_SECONDS = 30

ENV_TABLE = "DEALS_TABLE"
ENV_SEED_FILE = "DEALS_SEED_FILE"

#: Attributes the writers add to every item that are not part of the Deal object:
#: the table keys and seed order from the seeder, the provenance marks from the
#: control plane.
STORAGE_KEYS = ("record_type", "seq", "managed_by", "updated_at")

#: Deals API ``sellerstatus`` meaning the deal may transact. Also the spec's default.
SELLERSTATUS_ACTIVE = 0

#: Deals API integer fields (and integer arrays) that DynamoDB returns as Decimal.
_INT_FIELDS = frozenset({
    "seq", "sellerstatus", "adtypes", "auxdata", "pubcount", "dinventory",
    "guar", "pricetype", "units", "inclinventory", "devicetype", "cattax", "curfeetype",
})


@dataclass(frozen=True)
class LibraryDeal:
    """One deal in the publisher's book, in the shape the bid stream needs."""

    id: str
    bidfloor: float
    #: OpenRTB auction type: 1 first price, 2 second price, 3 fixed price.
    at: int
    bidfloorcur: str = "USD"
    guar: int = 0
    wseat: tuple[str, ...] = field(default_factory=tuple)
    #: The lowest floor the seller's optimizer may set (terms.ext.mindealfloor).
    mindealfloor: Optional[float] = None

    def as_openrtb(self) -> dict:
        """The OpenRTB 2.6 Deal object built from the record."""
        deal: dict[str, Any] = {
            "id": self.id,
            "bidfloor": self.bidfloor,
            "bidfloorcur": self.bidfloorcur,
            "at": self.at,
            "guar": self.guar,
        }
        if self.wseat:
            deal["wseat"] = list(self.wseat)
        if self.mindealfloor is not None:
            deal["ext"] = {"mindealfloor": self.mindealfloor}
        return deal


def deal_is_active(record: dict) -> bool:
    """Whether the record's ``sellerstatus`` lets it transact.

    Deals API v1.0: 0 active, 1 paused, 2 pending, 3 complete, 4 archived. A record
    without the field is active, as the spec's default says. Anything else is off
    the book for the request path: the scorer must not activate a deal the seller
    has paused, and the orchestrator must not dress it with an auction type.
    """
    status = record.get("sellerstatus", SELLERSTATUS_ACTIVE)
    try:
        return int(status) == SELLERSTATUS_ACTIVE
    except (TypeError, ValueError):
        return False


def deal_from_record(record: dict) -> LibraryDeal:
    """Map a Deals API v1.0 Deal record to the OpenRTB fields the scorer emits."""
    terms = record.get("terms") or {}
    ext = terms.get("ext") or {}
    minimum = ext.get("mindealfloor") if isinstance(ext, dict) else None
    return LibraryDeal(
        id=str(record["id"]),
        bidfloor=float(terms.get("dealfloor", 0.0) or 0.0),
        at=int(terms.get("pricetype", 2)),
        bidfloorcur=str(terms.get("cur") or "USD"),
        guar=int(terms.get("guar", 0) or 0),
        wseat=tuple(str(s) for s in record.get("wseat") or ()),
        mindealfloor=float(minimum) if minimum is not None else None,
    )


@dataclass(frozen=True)
class _Entry:
    deal: LibraryDeal
    seller: str
    sitedomains: frozenset[str]
    appbundles: frozenset[str]
    cats: tuple[str, ...]
    #: ``sellerstatus`` 0. Only active entries are offered to the request path.
    active: bool = True


class DealBook:
    """An immutable snapshot of every deal, with the request-to-book lookup.

    Every record is kept, so ``all_deal_ids`` and ``__len__`` describe the whole
    table and the cross-check against the demand catalog still sees a deal a
    campaign holds. Only ACTIVE records reach ``deals_for`` and ``by_id``: a deal
    the seller paused (or that is pending, complete or archived) is one the scorer
    must not activate and the orchestrator must not complete with an auction type.
    """

    def __init__(self, records: Iterable[dict]) -> None:
        entries = []
        seen: set[str] = set()
        for record in records:
            deal = deal_from_record(record)
            if deal.id in seen:
                raise ValueError(f"deal {deal.id!r} appears twice in the deal records")
            seen.add(deal.id)
            inventory = record.get("inventory") or {}
            entries.append(_Entry(
                deal=deal,
                seller=str(record.get("seller") or ""),
                sitedomains=frozenset(str(d).lower() for d in inventory.get("sitedomains") or ()),
                appbundles=frozenset(str(b) for b in inventory.get("appbundles") or ()),
                cats=tuple(str(c) for c in inventory.get("cat") or ()),
                active=deal_is_active(record),
            ))
        self._entries = tuple(entries)
        self._active = tuple(e for e in self._entries if e.active)
        self._by_id = {e.deal.id: e.deal for e in self._active}

    def __len__(self) -> int:
        return len(self._entries)

    def by_id(self, deal_id: Any) -> Optional[LibraryDeal]:
        """The ACTIVE deal with this id, or None when the book does not hold one."""
        return self._by_id.get(deal_id) if isinstance(deal_id, str) else None

    def inactive_deal_ids(self) -> frozenset[str]:
        """Deals in the table whose seller has taken them off the market."""
        return frozenset(e.deal.id for e in self._entries if not e.active)

    def deals_for(self, bid_request: Any) -> tuple[LibraryDeal, ...]:
        if not isinstance(bid_request, dict):
            return ()
        domain = _site_domain_of(bid_request)
        bundle = _app_bundle_of(bid_request)
        matched = tuple(
            e.deal for e in self._active
            if (domain and domain in e.sitedomains) or (bundle and bundle in e.appbundles)
        )
        if matched:
            return matched
        for cat in _categories_of(bid_request):
            first = next((e for e in self._active if cat in e.cats), None)
            if first is not None:
                return tuple(e.deal for e in self._active if e.seller == first.seller)
        return ()

    def all_deal_ids(self) -> frozenset[str]:
        return frozenset(e.deal.id for e in self._entries)


def _site_domain_of(bid_request: dict) -> Optional[str]:
    site = bid_request.get("site")
    if isinstance(site, dict):
        domain = site.get("domain")
        if isinstance(domain, str) and domain:
            return domain.lower()
    return None


def _app_bundle_of(bid_request: dict) -> Optional[str]:
    app = bid_request.get("app")
    if isinstance(app, dict):
        bundle = app.get("bundle")
        if isinstance(bundle, str) and bundle:
            return bundle
    return None


def _categories_of(bid_request: dict) -> list[str]:
    for key in ("site", "app"):
        container = bid_request.get(key)
        if isinstance(container, dict):
            cats = container.get("cat")
            if isinstance(cats, list):
                return [str(c) for c in cats]
    return []


# ----------------------------------------------------------------- the source


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


def describe_source() -> str:
    """A printable name for the configured source, for log lines."""
    try:
        kind, where = configured_source()
    except ValueError as exc:
        return f"<misconfigured: {exc}>"
    return f"{ENV_TABLE}={where}" if kind == "table" else f"{ENV_SEED_FILE}={where}"


def _plain(value: Any, key: Optional[str] = None) -> Any:
    if isinstance(value, Decimal):
        if key in _INT_FIELDS and value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {k: _plain(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v, key) for v in value]
    return value


def _table(name: str) -> Any:
    """The boto3 Table. Called only by ``load`` and ``refresh_once``, never by a handler."""
    import boto3

    return boto3.resource("dynamodb").Table(name)


def _query_deal_items(table: Any) -> list[dict]:
    from boto3.dynamodb.conditions import Key

    kwargs: dict[str, Any] = {"KeyConditionExpression": Key("record_type").eq("deal")}
    items: list[dict] = []
    while True:
        page = table.query(**kwargs)
        items.extend(page.get("Items") or [])
        last = page.get("LastEvaluatedKey")
        if not last:
            break
        kwargs["ExclusiveStartKey"] = last
    return items


def read_deal_records() -> list[dict]:
    """Deal records from the configured source, in seed order, storage keys stripped."""
    kind, where = configured_source()
    if kind == "table":
        items = _query_deal_items(_table(where))
    else:
        with open(where, encoding="utf-8") as handle:
            seed = json.load(handle)
        items = [dict(d, seq=i) for i, d in enumerate(seed["deals"])]
    ordered = sorted(items, key=lambda item: int(item.get("seq", 0)))
    return [
        _plain({k: v for k, v in item.items() if k not in STORAGE_KEYS})
        for item in ordered
    ]


# --------------------------------------------------------------- the snapshot

_book: Optional[DealBook] = None
_loaded_at: Optional[datetime] = None
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _build() -> DealBook:
    book = DealBook(read_deal_records())
    # An empty read is treated as a fault, not as an empty book: a seeded table is
    # never empty, so zero deals means the wrong table or a failed seed.
    if not len(book):
        raise ValueError("the source returned no deal records")
    return book


def load() -> DealBook:
    """Read the configured source and install the snapshot. Raises on failure."""
    global _book, _loaded_at
    book = _build()
    _book = book
    _loaded_at = datetime.now(timezone.utc)
    return book


def refresh_once() -> bool:
    """Rebuild and swap in the snapshot; on failure keep the old one and log."""
    try:
        load()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure keeps the last good snapshot
        loaded = _loaded_at.isoformat() if _loaded_at else "never"
        logger.error(
            "deal book refresh from %s failed: %s; keeping snapshot loaded at %s",
            describe_source(), exc, loaded,
        )
        return False


def _refresh_loop(stop: threading.Event) -> None:
    while not stop.wait(REFRESH_SECONDS):
        refresh_once()


def start(prime_required: bool = True) -> Optional[DealBook]:
    """Prime the snapshot, then start the refresh thread once.

    With ``prime_required`` (the Deal Scorer) a failed prime raises and no thread
    starts. Without it (the orchestrator) a failed prime is logged, the thread
    starts anyway and installs the first snapshot it can read, and None is returned.
    """
    global _thread
    try:
        book: Optional[DealBook] = load()
    except Exception as exc:  # noqa: BLE001 - the caller decides whether it is fatal
        if prime_required:
            raise
        logger.warning(
            "deal book prime from %s failed: %s; retrying every %ss",
            describe_source(), exc, REFRESH_SECONDS,
        )
        book = None
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(
            target=_refresh_loop, args=(_stop,), name="deal-book-refresh", daemon=True,
        )
        _thread.start()
    return book


def stop() -> None:
    """Stop the refresh thread (tests only)."""
    global _thread
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)
    _thread = None


def _installed() -> DealBook:
    book = _book
    if book is None:
        raise RuntimeError("deal book not loaded")
    return book


# ------------------------------------------------- the request-path interface


def deals_for(bid_request: dict) -> tuple[LibraryDeal, ...]:
    """The publisher's book for this request, or empty when the publisher is unknown.

    Empty is the correct answer for a publisher not in the library: the scorer then
    has nothing to activate beyond what the request offered. Reads only the
    in-memory snapshot.
    """
    return _installed().deals_for(bid_request)


def not_on_request(library: Iterable[LibraryDeal], request_deals: Iterable[Any]) -> list[LibraryDeal]:
    """Library deals the request did not already offer.

    A deal the publisher offered on the request is scored as a request deal and is
    never duplicated from the library; activating it twice would add nothing and
    suppressing it from the library side would contradict the publisher.
    """
    present = {d.get("id") for d in request_deals if isinstance(d, dict)}
    return [d for d in library if d.id not in present]


def all_deal_ids() -> frozenset[str]:
    """Every deal id in the snapshot, for the cross-check against the demand catalog."""
    return _installed().all_deal_ids()


def auction_type(deal_id: str) -> Optional[int]:
    """The deal's OpenRTB auction type (``at``), or None when the book does not hold it.

    ``at`` is the record's ``terms.pricetype`` unchanged: the Deals API and OpenRTB
    2.6 use the same codes (1 first price, 2 second price, 3 fixed price). Reads only
    the snapshot; raises RuntimeError when none is installed yet.
    """
    deal = _installed().by_id(deal_id)
    return deal.at if deal is not None else None
