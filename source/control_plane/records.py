"""Record rules for the deals table: IAB Tech Lab Deals API v1.0 deals and campaigns.

ONE SET OF RULES FOR TWO WRITERS. deployment/scripts/seed_deals_table.py validates
the seed file with these functions before writing it, and the control plane
validates every tool call with the same functions before writing the table. The
seeder used to hold this logic itself; it moved here so an agent cannot write a
record the seed could not.

Standard library only. The seeder runs on the deploying machine and the control
plane runs in an AgentCore microVM; neither should need more than this file to
agree on what a valid record is.

Field names and enum ranges for deals come from the Deals API specification
(https://github.com/IABTechLab/deal-api/blob/main/deal1.0.md). A campaign is this
stack's own record type: neither the Deals API nor OpenRTB defines one.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

ORIGIN_TEMPLATE = "{stackprefix}-prebid.example"
_DNS_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

#: Record identifiers written by a tool call. The seed's own ids all match this
#: already; the rule keeps a caller from writing an id that needs quoting in a
#: shell, a URL or a Kubernetes label.
RECORD_ID = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
RECORD_ID_MAX = 64

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

#: Deals API ``sellerstatus`` values.
SELLERSTATUS_ACTIVE = 0
SELLERSTATUS_PAUSED = 1
SELLERSTATUS_PENDING = 2
SELLERSTATUS_COMPLETE = 3
SELLERSTATUS_ARCHIVED = 4
SELLERSTATUS_NAMES = {
    SELLERSTATUS_ACTIVE: "active",
    SELLERSTATUS_PAUSED: "paused",
    SELLERSTATUS_PENDING: "pending",
    SELLERSTATUS_COMPLETE: "complete",
    SELLERSTATUS_ARCHIVED: "archived",
}

CAMPAIGN_FIELDS = frozenset({
    "campaign_id", "campaign_name", "adomain", "creative_id", "width", "height",
    "declared_cpm", "deal_ids", "open_market", "target_categories", "media_type",
    "duration", "audience_uplift", "note", "status",
})
CAMPAIGN_REQUIRED = CAMPAIGN_FIELDS - {"note", "duration", "audience_uplift", "status"}
MEDIA_TYPES = ("banner", "video")

CAMPAIGN_ACTIVE = "active"
CAMPAIGN_PAUSED = "paused"
CAMPAIGN_STATUSES = (CAMPAIGN_ACTIVE, CAMPAIGN_PAUSED)

#: Attributes the writers add to an item that are not part of the record: the
#: table's keys, the seed order, and the control plane's provenance marks.
STORAGE_KEYS = ("record_type", "seq", "managed_by", "updated_at")
MANAGED_BY_CONTROL_PLANE = "control-plane"

#: Upper bounds on free-text fields, so a tool parameter cannot carry an
#: arbitrarily large value into the table.
NAME_MAX = 200
DESC_MAX = 250
DOMAIN_MAX = 253
LIST_MAX = 100


class RecordError(ValueError):
    """A record breaks the Deals API schema or this stack's own consistency rules.

    ``problems`` lists every violation found, so a caller fixes them in one pass.
    """

    def __init__(self, problems: Iterable[str], *, heading: str = "record is invalid") -> None:
        self.problems = list(problems)
        super().__init__(f"{heading}:\n  " + "\n  ".join(self.problems))


# ------------------------------------------------------------------- helpers


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


def _check_str_list(problems: list[str], where: str, name: str, value: Any, item_max: int) -> None:
    if not isinstance(value, list):
        problems.append(f"{where}: {name} must be a list of strings")
        return
    if len(value) > LIST_MAX:
        problems.append(f"{where}: {name} holds more than {LIST_MAX} entries")
    for item in value:
        if not isinstance(item, str) or not item or len(item) > item_max:
            problems.append(f"{where}: {name} entry {item!r} must be a string of 1 to {item_max} characters")


def valid_record_id(value: Any) -> bool:
    """Whether ``value`` is acceptable as a deal or campaign id written by a tool call."""
    return isinstance(value, str) and 0 < len(value) <= RECORD_ID_MAX and bool(RECORD_ID.match(value))


# --------------------------------------------------------------------- deals


def validate_deal(deal: Mapping[str, Any], *, where: str = "deal", strict_id: bool = False) -> list[str]:
    """Problems with one Deals API v1.0 Deal object; empty when it is valid.

    ``strict_id`` applies the control plane's id rule (``RECORD_ID``). The seed is
    not held to it, so a seed id that predates the rule keeps seeding; every id in
    the shipped seed happens to satisfy it anyway.
    """
    problems: list[str] = []
    if not isinstance(deal, Mapping):
        return [f"{where}: a deal must be an object"]
    for required in ("id", "origin", "terms"):
        if required not in deal:
            problems.append(f"{where}: missing required {required}")
    unknown = set(deal) - DEAL_FIELDS
    if unknown:
        problems.append(f"{where}: not Deals API Deal fields: {sorted(unknown)}")
    deal_id = deal.get("id")
    if not isinstance(deal_id, str) or not deal_id:
        problems.append(f"{where}: id must be a non-empty string")
    elif strict_id and not valid_record_id(deal_id):
        problems.append(
            f"{where}: id must be 1 to {RECORD_ID_MAX} lowercase letters, digits and hyphens"
        )
    for name, limit in (("name", NAME_MAX), ("desc", DESC_MAX), ("seller", DOMAIN_MAX), ("origin", DOMAIN_MAX)):
        if name in deal and (not isinstance(deal[name], str) or len(deal[name]) > limit):
            problems.append(f"{where}: {name} must be a string of at most {limit} characters")
    if "created" in deal and not isinstance(deal["created"], str):
        problems.append(f"{where}: created must be an ISO-8601 string")
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
    for seats in ("wseat", "bseat"):
        if seats in deal:
            _check_str_list(problems, where, seats, deal[seats], NAME_MAX)
    if "wseat" in deal and "bseat" in deal:
        problems.append(f"{where}: at most one of wseat and bseat may be present")

    terms = deal.get("terms")
    if not isinstance(terms, Mapping):
        problems.append(f"{where}: terms must be an object")
        terms = {}
    unknown = set(terms) - TERMS_FIELDS
    if unknown:
        problems.append(f"{where}: not Deals API Terms fields: {sorted(unknown)}")
    floor = terms.get("dealfloor")
    if not _number(floor) or floor < 0:
        problems.append(f"{where}: terms.dealfloor={floor!r} must be a non-negative number")
    if "cur" in terms and (not isinstance(terms["cur"], str) or len(terms["cur"]) != 3):
        problems.append(f"{where}: terms.cur must be a three-letter ISO-4217 code")
    if "pricetype" in terms:
        _check_enum(problems, where, "terms.pricetype", terms["pricetype"], lambda v: 0 <= v <= 3 or v >= 500)
    if "guar" in terms:
        _check_enum(problems, where, "terms.guar", terms["guar"], lambda v: v in (0, 1))
    for name in ("units",):
        if name in terms and not _int_in(terms[name], lambda v: v >= 0):
            problems.append(f"{where}: terms.{name} must be a non-negative integer")
    if "totalcost" in terms and (not _number(terms["totalcost"]) or terms["totalcost"] < 0):
        problems.append(f"{where}: terms.totalcost must be a non-negative number")
    if "countries" in terms:
        _check_str_list(problems, where, "terms.countries", terms["countries"], 3)
    ext = terms.get("ext")
    if ext is not None:
        if not isinstance(ext, Mapping) or set(ext) - TERMS_EXT_FIELDS:
            problems.append(f"{where}: terms.ext may hold only {sorted(TERMS_EXT_FIELDS)}")
        else:
            minimum = ext.get("mindealfloor")
            if not _number(minimum) or minimum < 0:
                problems.append(f"{where}: terms.ext.mindealfloor={minimum!r} must be a non-negative number")
            elif _number(floor) and minimum > floor:
                problems.append(f"{where}: terms.ext.mindealfloor {minimum} is above terms.dealfloor {floor}")

    inventory = deal.get("inventory")
    if inventory is not None:
        if not isinstance(inventory, Mapping):
            problems.append(f"{where}: inventory must be an object")
        else:
            unknown = set(inventory) - INVENTORY_FIELDS
            if unknown:
                problems.append(f"{where}: not Deals API Inventory fields: {sorted(unknown)}")
            if "inclinventory" in inventory:
                _check_enum_list(problems, where, "inventory.inclinventory", inventory["inclinventory"], lambda v: 1 <= v <= 6)
            for name in ("sitedomains", "appbundles", "cat", "sellerids"):
                if name in inventory:
                    _check_str_list(problems, where, f"inventory.{name}", inventory[name], DOMAIN_MAX)
            if "cattax" in inventory and not _int_in(inventory["cattax"], lambda v: v >= 1):
                problems.append(f"{where}: inventory.cattax must be a positive integer")
    return problems


def deal_is_active(record: Mapping[str, Any]) -> bool:
    """Whether a deal may transact: ``sellerstatus`` 0, which is also the spec's default."""
    status = record.get("sellerstatus", SELLERSTATUS_ACTIVE)
    try:
        return int(status) == SELLERSTATUS_ACTIVE
    except (TypeError, ValueError):
        return False


def sellerstatus_name(value: Any) -> str:
    """The spec's word for a ``sellerstatus`` value, or ``unknown``."""
    try:
        return SELLERSTATUS_NAMES.get(int(value), "unknown")
    except (TypeError, ValueError):
        return "unknown"


# ----------------------------------------------------------------- campaigns


def validate_campaign(
    campaign: Mapping[str, Any],
    *,
    where: str = "campaign",
    known_deal_ids: Optional[Iterable[str]] = None,
    held_by: Optional[Mapping[str, str]] = None,
    price_ceiling: Optional[float] = None,
    strict_id: bool = False,
) -> list[str]:
    """Problems with one campaign record; empty when it is valid.

    ``known_deal_ids``: when given, every held deal must be in it.
    ``held_by``: deal id to the campaign id that already holds it; a deal held by a
    different campaign is a problem, because the demand catalog raises on a deal
    claimed twice and that would take the whole snapshot down.
    ``price_ceiling``: when given, ``declared_cpm`` plus any uplift must not exceed it.
    """
    problems: list[str] = []
    if not isinstance(campaign, Mapping):
        return [f"{where}: a campaign must be an object"]
    cid = campaign.get("campaign_id")
    missing = CAMPAIGN_REQUIRED - set(campaign)
    if missing:
        problems.append(f"{where}: missing {sorted(missing)}")
    unknown = set(campaign) - CAMPAIGN_FIELDS
    if unknown:
        problems.append(f"{where}: unknown campaign fields {sorted(unknown)}")
    if not isinstance(cid, str) or not cid:
        problems.append(f"{where}: campaign_id must be a non-empty string")
    elif strict_id and not valid_record_id(cid):
        problems.append(
            f"{where}: campaign_id must be 1 to {RECORD_ID_MAX} lowercase letters, digits and hyphens"
        )
    for name, limit in (("campaign_name", NAME_MAX), ("adomain", DOMAIN_MAX), ("creative_id", NAME_MAX), ("note", DESC_MAX)):
        if name in campaign and (not isinstance(campaign[name], str) or not campaign[name] or len(campaign[name]) > limit):
            problems.append(f"{where}: {name} must be a string of 1 to {limit} characters")
    for name in ("width", "height"):
        if name in campaign and not _int_in(campaign[name], lambda v: 0 < v <= 10000):
            problems.append(f"{where}: {name} must be an integer between 1 and 10000")
    cpm = campaign.get("declared_cpm")
    if "declared_cpm" in campaign and (not _number(cpm) or cpm <= 0):
        problems.append(f"{where}: declared_cpm={cpm!r} must be a positive number")
    uplift_cpm = 0.0
    uplift = campaign.get("audience_uplift")
    if uplift is not None:
        if not isinstance(uplift, Mapping) or set(uplift) - {"segment_ids", "cpm"}:
            problems.append(f"{where}: audience_uplift must be an object with segment_ids and cpm")
        else:
            _check_str_list(problems, where, "audience_uplift.segment_ids", uplift.get("segment_ids"), NAME_MAX)
            if isinstance(uplift.get("segment_ids"), list) and not uplift.get("segment_ids"):
                problems.append(f"{where}: audience_uplift.segment_ids must not be empty")
            if not _number(uplift.get("cpm")) or uplift.get("cpm") < 0:
                problems.append(f"{where}: audience_uplift.cpm must be a non-negative number")
            else:
                uplift_cpm = float(uplift["cpm"])
    if price_ceiling is not None and _number(cpm) and cpm + uplift_cpm > price_ceiling:
        problems.append(
            f"{where}: declared_cpm plus uplift ({cpm + uplift_cpm:g}) is above the price ceiling {price_ceiling:g}"
        )
    media_type = campaign.get("media_type")
    if "media_type" in campaign and media_type not in MEDIA_TYPES:
        problems.append(f"{where}: media_type must be one of {list(MEDIA_TYPES)}")
    duration = campaign.get("duration")
    if duration is not None and not _int_in(duration, lambda v: v > 0):
        problems.append(f"{where}: duration must be a positive integer (seconds)")
    if media_type == "video" and duration is None:
        problems.append(f"{where}: a video campaign needs a duration")
    if "open_market" in campaign and not isinstance(campaign["open_market"], bool):
        problems.append(f"{where}: open_market must be true or false")
    if "target_categories" in campaign:
        _check_str_list(problems, where, "target_categories", campaign["target_categories"], NAME_MAX)
    if "status" in campaign and campaign["status"] not in CAMPAIGN_STATUSES:
        problems.append(f"{where}: status must be one of {list(CAMPAIGN_STATUSES)}")

    deal_ids = campaign.get("deal_ids")
    if "deal_ids" in campaign:
        _check_str_list(problems, where, "deal_ids", deal_ids, RECORD_ID_MAX)
    deal_list = [d for d in deal_ids if isinstance(d, str)] if isinstance(deal_ids, list) else []
    if len(set(deal_list)) != len(deal_list):
        problems.append(f"{where}: deal_ids repeats a deal")
    if "deal_ids" in campaign and "open_market" in campaign and not deal_list and campaign.get("open_market") is False:
        problems.append(
            f"{where}: a campaign must hold at least one deal or set open_market true, or it can never offer"
        )
    known = set(known_deal_ids) if known_deal_ids is not None else None
    for deal_id in deal_list:
        if known is not None and deal_id not in known:
            problems.append(f"{where}: holds {deal_id}, which is not a deal in the seed or the table")
        if held_by and held_by.get(deal_id) not in (None, cid):
            problems.append(f"{where}: {deal_id} is already held by {held_by[deal_id]}")
    return problems


def campaign_status(record: Mapping[str, Any]) -> str:
    """``active`` or ``paused``. A record without the field is active.

    Absence means active so the seeded campaigns, and any table written before the
    field existed, keep bidding exactly as they did. Only an explicit ``paused``
    silences a campaign.
    """
    status = record.get("status")
    return CAMPAIGN_PAUSED if status == CAMPAIGN_PAUSED else CAMPAIGN_ACTIVE


# ---------------------------------------------------------------------- seed


def validate_seed(seed: Mapping[str, Any]) -> None:
    """Raise RecordError listing every problem in a seed file; return None when valid."""
    deals = seed.get("deals")
    campaigns = seed.get("campaigns")
    if not isinstance(deals, list) or not deals:
        raise RecordError(["seed has no deals list"], heading="deals seed is invalid")
    if not isinstance(campaigns, list):
        raise RecordError(["seed has no campaigns list"], heading="deals seed is invalid")

    problems: list[str] = []
    deal_ids: set[str] = set()
    for i, deal in enumerate(deals):
        where = f"deals[{i}] {deal.get('id', '<no id>') if isinstance(deal, Mapping) else '<no id>'}"
        problems.extend(validate_deal(deal, where=where))
        deal_id = deal.get("id") if isinstance(deal, Mapping) else None
        if isinstance(deal_id, str) and deal_id:
            if deal_id in deal_ids:
                problems.append(f"{where}: duplicate deal id")
            deal_ids.add(deal_id)

    held: dict[str, str] = {}
    campaign_ids: set[str] = set()
    for i, campaign in enumerate(campaigns):
        cid = campaign.get("campaign_id") if isinstance(campaign, Mapping) else None
        where = f"campaigns[{i}] {cid or '<no id>'}"
        problems.extend(validate_campaign(campaign, where=where, known_deal_ids=deal_ids, held_by=held))
        if cid in campaign_ids:
            problems.append(f"{where}: duplicate campaign id")
        campaign_ids.add(cid)
        if isinstance(campaign, Mapping):
            for deal_id in campaign.get("deal_ids") or []:
                held.setdefault(deal_id, cid)

    if problems:
        raise RecordError(problems, heading="deals seed is invalid")


def render_origin(prefix: str) -> str:
    """``origin`` for every deal: ``<prefix>-prebid.example``, or ``prebid.example`` with no prefix."""
    prefix = (prefix or "").strip().lower()
    label = f"{prefix}-prebid" if prefix else "prebid"
    if not _DNS_LABEL.match(label) or len(label) > 63:
        raise RecordError(
            [f"origin prefix {prefix!r} does not make a valid DNS label ({label!r})"],
            heading="deals seed is invalid",
        )
    return f"{label}.example"
