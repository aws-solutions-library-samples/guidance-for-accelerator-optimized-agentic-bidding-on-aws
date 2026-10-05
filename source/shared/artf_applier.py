"""Apply ARTF mutations to an OpenRTB bid request (and, for BID_SHADE, a bid response).

This is the Python twin of the Prebid hook's ``ArtfMutationApplier.java`` and of the
frontend's ``utils/artfApplier.js``. The orchestrator uses it between stages
(shared/artf_stages.py) so that stage N+1's containers see what stage N wrote; the
other two use it to apply the final list. All three write to the same five
locations with the same rules, and the test vectors in tests/test_artf_applier.py
are the ones the Java tests pin, so the three cannot drift without a test
noticing.

Targets, by path:

    /user/data/segment            ACTIVATE_SEGMENTS   append a Data{name: artf} block
    /imp/{imp}                    ACTIVATE_DEALS      add Deal{id} to imp.pmp.deals
                                  SUPPRESS_DEALS      mark deal.ext.artf.suppressed
    /imp/{imp}/deals/{deal}       ADJUST_DEAL_FLOOR   deal.bidfloor (+cur), imp.bidfloor
                                  ADJUST_DEAL_MARGIN  = max(existing, new)
    /imp/{imp}/metric             ADD_METRICS/ADD_CIDS append to imp.metric, value in [0,1]
    /seatbid/{seat}/bid/{bid}     BID_SHADE           bid.price on the bid response

Every mutation gets a disposition; a rejection is a normal outcome with a reason.
Each mutation is all-or-nothing: it is applied to a copy, and the copy is adopted
only if every part of it succeeded.

Suppression marks rather than removes, because the demand endpoint and the
Theater both read ``deal.ext.artf.suppressed`` to show that a deal was considered
and suppressed; a removed deal is indistinguishable from one never offered.

A floor written without a currency is ignored by Prebid's enforcer, so the
currency is always written with the floor: the deal's own, then the impression's,
then ``cur[0]``, then USD.

A PERCENT margin's ``value`` is a FRACTION (0.12 is twelve percent). That is what
yield_optimizer_margin emits (shared/yield_exploration.py bounds it to [-0.5, 0.5])
and what deal_yield_feedback and the frontend read.

Pure. No I/O, no clock, no configuration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from shared.artf_types import Intent, MarginCalculationType, Mutation

DEFAULT_CURRENCY = "USD"
EXT_ARTF = "artf"
EXT_SUPPRESSED = "suppressed"
METRIC_MIN = 0.0
METRIC_MAX = 1.0


@dataclass(frozen=True)
class Disposition:
    """What became of one mutation."""

    intent: int
    path: str
    applied: bool
    reason: Optional[str] = None
    #: JSON pointers (from the envelope root: ``/bid_request/...`` or
    #: ``/bid_response/...``) of the nodes this mutation wrote. Empty when
    #: rejected. A renderer highlights these; the mutation's own path is a
    #: semantic address, not a location in the document.
    written: tuple[str, ...] = ()


@dataclass
class ApplyResult:
    bid_request: dict
    bid_response: Optional[dict]
    dispositions: list[Disposition] = field(default_factory=list)

    @property
    def applied_count(self) -> int:
        return sum(1 for d in self.dispositions if d.applied)

    @property
    def rejected(self) -> list[Disposition]:
        return [d for d in self.dispositions if not d.applied]


class _Reject(Exception):
    """Raised inside an applier step to reject the mutation with a reason."""


# ---------------------------------------------------------------------------
# Path resolution (mirrors PathResolver.java)
# ---------------------------------------------------------------------------

def _resolve(path: str) -> tuple[str, tuple[str, ...]]:
    """Return (target kind, captured ids) or raise _Reject."""
    if not path or not path.strip():
        raise _Reject("mutation carries no path")
    parts = [p for p in path.split("/") if p != ""]
    if len(parts) == 3 and parts[:3] == ["user", "data", "segment"]:
        return "user_segments", ()
    if parts and parts[0] == "seatbid":
        if len(parts) == 4 and parts[2] == "bid" and parts[1] and parts[3]:
            return "bid_price", (parts[1], parts[3])
        raise _Reject(
            f"path '{path}' addresses the auction response but does not name "
            "/seatbid/{seat}/bid/{bidId}"
        )
    if parts and parts[0] == "imp":
        if len(parts) == 2 and parts[1]:
            return "imp_deals", (parts[1],)
        if len(parts) == 3 and parts[2] == "metric" and parts[1]:
            return "imp_metrics", (parts[1],)
        if len(parts) == 4 and parts[2] == "deals" and parts[1] and parts[3]:
            return "deal_floor", (parts[1], parts[3])
    raise _Reject(
        f"path '{path}' does not name a location this applier may write. Permitted: "
        "/user/data/segment, /imp/{impId}, /imp/{impId}/metric, "
        "/imp/{impId}/deals/{dealId}, /seatbid/{seat}/bid/{bidId}"
    )


def _imp_index(bid_request: dict, imp_id: str) -> int:
    for i, imp in enumerate(bid_request.get("imp") or []):
        if isinstance(imp, dict) and imp.get("id") == imp_id:
            return i
    return -1


def _deal_index(deals: list, deal_id: str) -> int:
    for i, d in enumerate(deals):
        if isinstance(d, dict) and d.get("id") == deal_id:
            return i
    return -1


def _ids_of(m: Mutation) -> list[str]:
    if m.ids is None or m.ids.id is None:
        return []
    return [s for s in m.ids.id if isinstance(s, str) and s.strip()]


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _currency(bid_request: dict, deal: dict, imp: dict) -> str:
    for candidate in (deal.get("bidfloorcur"), imp.get("bidfloorcur")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    cur = bid_request.get("cur")
    if isinstance(cur, list) and cur and isinstance(cur[0], str) and cur[0].strip():
        return cur[0]
    return DEFAULT_CURRENCY


# ---------------------------------------------------------------------------
# The five targets
# ---------------------------------------------------------------------------

def _apply_user_segments(req: dict, m: Mutation, intent: Intent) -> list[str]:
    if intent != Intent.ACTIVATE_SEGMENTS:
        raise _Reject(f"intent {intent.name} does not write user segments")
    ids = _ids_of(m)
    if not ids:
        raise _Reject("no segment ids supplied")
    user = req.get("user")
    if not isinstance(user, dict):
        user = {}
        req["user"] = user
    data = user.get("data")
    if not isinstance(data, list):
        data = []
        user["data"] = data
    data.append({"name": EXT_ARTF, "segment": [{"id": i} for i in ids]})
    return [f"/bid_request/user/data/{len(data) - 1}"]


def _apply_imp_deals(req: dict, m: Mutation, intent: Intent, imp_id: str) -> list[str]:
    idx = _imp_index(req, imp_id)
    if idx < 0:
        raise _Reject(f"no impression with id '{imp_id}'")
    ids = _ids_of(m)
    if not ids:
        raise _Reject("no deal ids supplied")
    imp = req["imp"][idx]
    pmp = imp.get("pmp")
    if not isinstance(pmp, dict):
        pmp = {}
        imp["pmp"] = pmp
    deals = pmp.get("deals")
    if not isinstance(deals, list):
        deals = []
        pmp["deals"] = deals

    written: list[str] = []
    if intent == Intent.ACTIVATE_DEALS:
        present = {d.get("id") for d in deals if isinstance(d, dict)}
        for i in ids:
            if i not in present:
                deals.append({"id": i})
                present.add(i)
                written.append(f"/bid_request/imp/{idx}/pmp/deals/{len(deals) - 1}")
        return written
    if intent == Intent.SUPPRESS_DEALS:
        for di, d in enumerate(deals):
            if isinstance(d, dict) and d.get("id") in ids:
                ext = d.get("ext")
                if not isinstance(ext, dict):
                    ext = {}
                    d["ext"] = ext
                artf = ext.get(EXT_ARTF)
                if not isinstance(artf, dict):
                    artf = {}
                    ext[EXT_ARTF] = artf
                artf[EXT_SUPPRESSED] = True
                written.append(f"/bid_request/imp/{idx}/pmp/deals/{di}/ext")
        return written
    raise _Reject(f"intent {intent.name} does not write deals")


def _resolve_floor(m: Mutation, deal: dict, intent: Intent) -> Optional[float]:
    payload = m.adjust_deal
    if payload is None:
        return None
    if intent == Intent.ADJUST_DEAL_FLOOR:
        return _num(payload.bidfloor)
    margin = payload.margin
    if margin is None or margin.value is None:
        return _num(payload.bidfloor)
    base = _num(deal.get("bidfloor")) or 0.0
    value = float(margin.value)
    if margin.calculation_type == MarginCalculationType.CPM:
        return base + value
    if margin.calculation_type == MarginCalculationType.PERCENT:
        return base + base * value
    return None


def _apply_deal_floor(req: dict, m: Mutation, intent: Intent, imp_id: str, deal_id: str) -> list[str]:
    if intent not in (Intent.ADJUST_DEAL_FLOOR, Intent.ADJUST_DEAL_MARGIN):
        raise _Reject(f"intent {intent.name} does not write a deal floor")
    idx = _imp_index(req, imp_id)
    if idx < 0:
        raise _Reject(f"no impression with id '{imp_id}'")
    imp = req["imp"][idx]
    pmp = imp.get("pmp")
    deals = pmp.get("deals") if isinstance(pmp, dict) else None
    if not isinstance(deals, list) or not deals:
        raise _Reject(f"impression '{imp_id}' carries no deals")
    di = _deal_index(deals, deal_id)
    if di < 0:
        raise _Reject(f"impression '{imp_id}' has no deal '{deal_id}'")
    deal = deals[di]
    new_floor = _resolve_floor(m, deal, intent)
    if new_floor is None:
        raise _Reject("mutation supplied no usable floor or margin value")
    if new_floor <= 0:
        raise _Reject(
            f"resulting floor {new_floor} is not greater than zero, which Prebid ignores"
        )
    currency = _currency(req, deal, imp)
    deal["bidfloor"] = new_floor
    deal["bidfloorcur"] = currency
    existing = _num(imp.get("bidfloor"))
    imp["bidfloor"] = new_floor if existing is None else max(existing, new_floor)
    imp["bidfloorcur"] = currency
    return [
        f"/bid_request/imp/{idx}/pmp/deals/{di}/bidfloor",
        f"/bid_request/imp/{idx}/bidfloor",
    ]


def _apply_imp_metrics(req: dict, m: Mutation, intent: Intent, imp_id: str) -> list[str]:
    if intent not in (Intent.ADD_METRICS, Intent.ADD_CIDS):
        raise _Reject(f"intent {intent.name} does not write metrics")
    idx = _imp_index(req, imp_id)
    if idx < 0:
        raise _Reject(f"no impression with id '{imp_id}'")
    payload = m.add_metrics
    if payload is None or not payload.metric:
        raise _Reject("no metrics supplied")
    incoming = []
    for metric in payload.metric:
        if metric is None or not metric.type or not metric.type.strip():
            raise _Reject("metric requires a non-blank type and a value")
        value = _num(metric.value)
        if value is None:
            raise _Reject("metric requires a non-blank type and a value")
        if value < METRIC_MIN or value > METRIC_MAX:
            raise _Reject(
                f"metric '{metric.type}' value {value} is outside the "
                f"[{METRIC_MIN}, {METRIC_MAX}] range OpenRTB defines for imp.metric"
            )
        entry = {"type": metric.type, "value": value}
        if metric.vendor is not None:
            entry["vendor"] = metric.vendor
        incoming.append(entry)
    imp = req["imp"][idx]
    metrics = imp.get("metric")
    if not isinstance(metrics, list):
        metrics = []
        imp["metric"] = metrics
    start = len(metrics)
    metrics.extend(incoming)
    return [f"/bid_request/imp/{idx}/metric/{start + i}" for i in range(len(incoming))]


def _apply_bid_price(resp: Optional[dict], m: Mutation, intent: Intent, seat: str, bid_id: str) -> list[str]:
    if intent != Intent.BID_SHADE:
        raise _Reject(f"intent {intent.name} does not write a bid price")
    if not isinstance(resp, dict):
        raise _Reject(
            f"path '/seatbid/{seat}/bid/{bid_id}' addresses the auction response, "
            "and this envelope carries no bid_response"
        )
    payload = m.adjust_bid
    price = _num(payload.price) if payload is not None else None
    if price is None:
        raise _Reject("mutation supplied no price")
    if price <= 0:
        raise _Reject(f"shaded price {price} is not greater than zero")
    for si, seatbid in enumerate(resp.get("seatbid") or []):
        if not isinstance(seatbid, dict) or seatbid.get("seat") != seat:
            continue
        for bi, bid in enumerate(seatbid.get("bid") or []):
            if isinstance(bid, dict) and bid.get("id") == bid_id:
                bid["price"] = price
                return [f"/bid_response/seatbid/{si}/bid/{bi}/price"]
    raise _Reject(f"no bid '{bid_id}' under seat '{seat}' in the bid response")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply(
    bid_request: dict,
    mutations: list[Mutation],
    bid_response: Optional[dict] = None,
) -> ApplyResult:
    """Apply ``mutations`` in order. The inputs are not modified."""
    current_req = copy.deepcopy(bid_request) if isinstance(bid_request, dict) else {}
    current_resp = copy.deepcopy(bid_response) if isinstance(bid_response, dict) else None
    dispositions: list[Disposition] = []

    for m in mutations or []:
        if m is None:
            dispositions.append(Disposition(Intent.UNSPECIFIED, "", False, "mutation was null"))
            continue
        try:
            intent = Intent(m.intent)
        except ValueError:
            dispositions.append(Disposition(m.intent, m.path, False, f"unknown intent {m.intent}"))
            continue

        # All-or-nothing: work on a copy, adopt it only on success.
        candidate_req = copy.deepcopy(current_req)
        candidate_resp = copy.deepcopy(current_resp)
        try:
            kind, captured = _resolve(m.path)
            if kind == "user_segments":
                written = _apply_user_segments(candidate_req, m, intent)
            elif kind == "imp_deals":
                written = _apply_imp_deals(candidate_req, m, intent, captured[0])
            elif kind == "deal_floor":
                written = _apply_deal_floor(candidate_req, m, intent, captured[0], captured[1])
            elif kind == "imp_metrics":
                written = _apply_imp_metrics(candidate_req, m, intent, captured[0])
            else:  # bid_price
                written = _apply_bid_price(candidate_resp, m, intent, captured[0], captured[1])
        except _Reject as rej:
            dispositions.append(Disposition(intent, m.path, False, str(rej)))
            continue
        except Exception as exc:  # noqa: BLE001 - a partial apply must become a rejection
            dispositions.append(
                Disposition(
                    intent, m.path, False,
                    f"mutation could not be applied: {type(exc).__name__}: {exc}",
                )
            )
            continue

        current_req, current_resp = candidate_req, candidate_resp
        dispositions.append(Disposition(intent, m.path, True, None, tuple(written)))

    return ApplyResult(current_req, current_resp, dispositions)
