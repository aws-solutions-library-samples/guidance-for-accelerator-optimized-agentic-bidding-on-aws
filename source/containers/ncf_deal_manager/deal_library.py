"""The publisher's deal library, as the Deal Scorer knows it.

ARTF's sell-side ACTIVATE_DEALS means activating a deal the publisher HAS but did
not offer on this request. The request's own ``imp.pmp.deals`` is what the
publisher chose to expose; this module is the rest of the publisher's deal book,
so the scorer has something to activate from.

Keyed by publisher domain, with a content-category fallback for a request that
names the category but not the site. Each record carries the deal's floor and
auction type, because an activated deal reaches the seats with whatever this
container says about it: the ARTF IDsPayload carries ids only, so the floor is
emitted as a separate ADJUST_DEAL_FLOOR mutation (see app.py).

The deal ids here mirror the demand side's catalog (source/demand/artfhouse/
catalog.py): every deal below is one an artfhouse campaign transacts on, and
test_ncf_deal_library.py holds the two in step. Floors are the publisher's, taken
from the shipped scenarios that declare these deals, so a deal activated from here
reaches the seat at the same floor it would have had if the request had offered it.
"""
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class LibraryDeal:
    """One deal in the publisher's book."""

    id: str
    bidfloor: float
    #: OpenRTB auction type: 1 first price, 2 second price, 3 fixed price.
    at: int

    def as_openrtb(self) -> dict:
        return {"id": self.id, "bidfloor": self.bidfloor, "at": self.at}


#: Publisher domain -> deals. Floors and auction types are the ones the shipped
#: scenarios for that publisher declare.
_BY_DOMAIN: dict[str, tuple[LibraryDeal, ...]] = {
    "parenting-weekly.example": (
        LibraryDeal("deal-parenting-premium", 3.40, 1),
        LibraryDeal("deal-family-network", 2.10, 2),
        LibraryDeal("deal-remnant-open", 0.85, 3),
    ),
    "market-ledger.example": (
        LibraryDeal("deal-finance-pmp", 3.50, 1),
        LibraryDeal("deal-home-premium", 3.00, 2),
        LibraryDeal("deal-remnant-open", 1.00, 3),
    ),
    "hearth-and-home.example": (
        LibraryDeal("deal-home-premium", 3.00, 2),
        LibraryDeal("deal-retail-run", 2.60, 2),
        LibraryDeal("deal-auto-brand", 1.50, 2),
        LibraryDeal("deal-remnant-open", 1.00, 3),
    ),
    "cnn.com": (
        LibraryDeal("deal-premium-auto", 6.00, 1),
        LibraryDeal("deal-standard-auto", 3.50, 2),
    ),
    "espn.com": (
        LibraryDeal("deal-remnant-open", 1.50, 3),
    ),
    "streaming.example.com": (
        LibraryDeal("deal-premium-video", 10.00, 1),
        LibraryDeal("deal-standard-video", 5.00, 2),
        LibraryDeal("deal-remnant", 0.50, 2),
    ),
    "sportsnetwork.example.com": (
        LibraryDeal("deal-guaranteed-premium", 12.00, 1),
        LibraryDeal("deal-open-midtier", 3.25, 2),
        LibraryDeal("deal-open-remnant", 0.75, 2),
    ),
}

#: Content-category fallback (IAB Content Taxonomy 3.x ids and 2.x codes as the
#: scenarios use them) -> the publisher domain whose book applies. Used only when
#: the request names no site domain.
_BY_CATEGORY: dict[str, str] = {
    "192": "parenting-weekly.example",   # Parenting
    "410": "market-ledger.example",      # Personal Finance
    "283": "hearth-and-home.example",    # Home & Garden
}


def _domain_of(bid_request: dict) -> Optional[str]:
    for key in ("site", "app"):
        container = bid_request.get(key)
        if isinstance(container, dict):
            domain = container.get("domain")
            if isinstance(domain, str) and domain:
                return domain.lower()
    return None


def _categories_of(bid_request: dict) -> list[str]:
    for key in ("site", "app"):
        container = bid_request.get(key)
        if isinstance(container, dict):
            cats = container.get("cat")
            if isinstance(cats, list):
                return [str(c) for c in cats]
    return []


def deals_for(bid_request: dict) -> tuple[LibraryDeal, ...]:
    """The publisher's book for this request, or empty when the publisher is unknown.

    Empty is the correct answer for a publisher not in the library: the scorer then
    has nothing to activate beyond what the request offered, which is today's
    behaviour for every scenario but the ones listed above.
    """
    if not isinstance(bid_request, dict):
        return ()
    domain = _domain_of(bid_request)
    if domain and domain in _BY_DOMAIN:
        return _BY_DOMAIN[domain]
    for cat in _categories_of(bid_request):
        mapped = _BY_CATEGORY.get(cat)
        if mapped:
            return _BY_DOMAIN[mapped]
    return ()


def not_on_request(library: Iterable[LibraryDeal], request_deals: Iterable[Any]) -> list[LibraryDeal]:
    """Library deals the request did not already offer.

    A deal the publisher offered on the request is scored as a request deal and is
    never duplicated from the library; activating it twice would add nothing and
    suppressing it from the library side would contradict the publisher.
    """
    present = {d.get("id") for d in request_deals if isinstance(d, dict)}
    return [d for d in library if d.id not in present]


def all_deal_ids() -> frozenset[str]:
    """Every deal id in the library, for the cross-check against the demand catalog."""
    return frozenset(d.id for deals in _BY_DOMAIN.values() for d in deals)
