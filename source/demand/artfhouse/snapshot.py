"""The demand service's in-memory catalog, refreshed off the bid path.

Same pattern as shared/deal_library.py, which the ARTF containers use: the bid path
makes no external call. The full catalog (campaign records and deal records) is
read from the configured source (``DEALS_TABLE`` or ``DEALS_SEED_FILE``, see
deals_source.py) into an immutable snapshot:

  * ``start()`` loads the first snapshot and starts a daemon thread that reloads it
    every ``REFRESH_SECONDS``. A failed first load does not raise: the thread keeps
    trying, and ``ready()`` stays False until a snapshot is installed, so the pod's
    readiness probe fails and Prebid is never routed to it.
  * a failed refresh logs a warning and keeps serving the last good snapshot.
  * ``service()`` returns the installed DemandDecisionService. A new snapshot is
    swapped in with one assignment, so a bid in flight sees either the old one or
    the new one, never a mix.

Deal records are loaded for two things: to report campaigns that name a deal the
book does not hold (a seed fault worth a log line), and to tell the catalog which
deals the seller has taken off the market (Deals API ``sellerstatus`` not 0), so a
campaign holding only such deals declines with ``deal_inactive``. A request still
carries its own deal floors; the decision never reads the table.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import deals_source
from .catalog import CampaignCatalog, campaigns_from_records
from .service import DemandDecisionService

logger = logging.getLogger("artfhouse.demand.snapshot")

#: Seconds between background refreshes. Matches shared/deal_library.py.
REFRESH_SECONDS = 30


@dataclass(frozen=True)
class Snapshot:
    service: DemandDecisionService
    campaigns: int
    deals: int
    loaded_at: datetime
    paused_campaigns: int = 0
    inactive_deals: int = 0


_snapshot: Optional[Snapshot] = None
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


#: Deals API ``sellerstatus`` meaning the deal may transact; also the spec's default.
_SELLERSTATUS_ACTIVE = 0


def _deal_is_active(record: dict) -> bool:
    """Same rule as shared/deal_library.deal_is_active; restated because the demand
    image carries only the demand package."""
    try:
        return int(record.get("sellerstatus", _SELLERSTATUS_ACTIVE)) == _SELLERSTATUS_ACTIVE
    except (TypeError, ValueError):
        return False


def _build() -> Snapshot:
    campaign_records = deals_source.campaign_records()
    deal_records = deals_source.deal_records()
    # Empty is a fault, not an empty catalog: a seeded table is never empty, so zero
    # records means the wrong table or a failed seed.
    if not campaign_records:
        raise ValueError("the source returned no campaign records")
    inactive = {str(d.get("id")) for d in deal_records if not _deal_is_active(d)}
    catalog = CampaignCatalog(campaigns_from_records(campaign_records), inactive_deal_ids=inactive)
    known = {str(d.get("id")) for d in deal_records}
    unknown = sorted({deal for c in catalog.all() for deal in c.deal_ids} - known)
    if unknown:
        logger.warning(json.dumps({
            "event": "catalog_unknown_deals", "deal_ids": unknown,
        }))
    return Snapshot(
        service=DemandDecisionService(catalog),
        campaigns=len(catalog.all()),  # nosemgrep: len-all-count CampaignCatalog.all() returns a tuple, not a Django queryset
        deals=len(deal_records),
        loaded_at=datetime.now(timezone.utc),
        paused_campaigns=sum(1 for c in catalog.all() if c.paused),
        inactive_deals=len(inactive),
    )


def load() -> Snapshot:
    """Read the source and install a new snapshot. Raises on failure."""
    global _snapshot
    snap = _build()
    _snapshot = snap
    logger.info(json.dumps({
        "event": "catalog_loaded", "campaigns": snap.campaigns, "deals": snap.deals,
        "paused_campaigns": snap.paused_campaigns, "inactive_deals": snap.inactive_deals,
        "source": deals_source.describe_source(),
    }))
    return snap


def refresh_once() -> bool:
    """Reload and swap; on failure keep the last good snapshot and log."""
    try:
        load()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure keeps the last snapshot
        current = _snapshot
        logger.warning(json.dumps({
            "event": "catalog_refresh_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "serving_snapshot_from": current.loaded_at.isoformat() if current else None,
        }))
        return False


def _refresh_loop(stop: threading.Event) -> None:
    while not stop.wait(REFRESH_SECONDS):
        refresh_once()


def start() -> bool:
    """Load the first snapshot (failure is logged) and start the refresh thread once."""
    global _thread
    loaded = refresh_once()
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(
            target=_refresh_loop, args=(_stop,), name="catalog-refresh", daemon=True,
        )
        _thread.start()
    return loaded


def stop() -> None:
    """Stop the refresh thread (tests and shutdown)."""
    global _thread
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)
    _thread = None


def ready() -> bool:
    return _snapshot is not None


def current() -> Optional[Snapshot]:
    return _snapshot


def service() -> DemandDecisionService:
    """The installed service. Reads memory only; raises when nothing is loaded yet."""
    snap = _snapshot
    if snap is None:
        raise RuntimeError("catalog not loaded")
    return snap.service
