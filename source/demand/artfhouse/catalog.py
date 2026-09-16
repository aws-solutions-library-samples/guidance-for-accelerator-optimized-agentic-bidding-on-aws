"""The campaign catalog.

DETERMINISTIC AND DEFINED IN CODE (FR-16). No database, no cache, no network call
on the decision path -- which is what makes the decision a function of the request
and this catalog, and therefore testable without a deployed stack (NFR-4).

THIS IS A DEMONSTRATION CATALOG. It is synthetic, and that is legitimate here for
specific reasons rather than by exemption:
  - it is the HOST'S CONFIGURATION, not a measurement;
  - it is deterministic and inspectable, written here rather than generated;
  - it is the demand side of a demonstration, which has no real advertisers.

What would be fabrication is presenting a computed OUTCOME as real -- an invented
winner, a fake clearing price, or a bid no campaign declared. None of that happens
here or anywhere in this unit: every bid traces to a campaign below, and this
endpoint does not decide the auction at all.

NOT EMBEDDED IN ANY ARTF AGENT IMAGE. The catalog belongs to the host's demand
endpoint. Putting it inside an agent container would make an agent carry its own
demand, which inverts the framework's ownership model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Campaign:
    """A demand-side campaign that may offer on an impression.

    Frozen: the catalog is immutable, so no request handler can mutate a campaign
    and leave the next request seeing different data.
    """

    campaign_id: str
    campaign_name: str
    adomain: str
    creative_id: str
    width: int
    height: int
    #: Declared CPM. THE CATALOG IS THE SINGLE SOURCE OF TRUTH FOR PRICE -- mirroring
    #: it onto the request would create a second source that eventually disagrees.
    declared_cpm: float
    #: Deals this campaign transacts on.
    deal_ids: tuple[str, ...] = field(default_factory=tuple)
    #: Whether it may offer with no deal on the impression.
    open_market: bool = False
    #: Content categories it will bid on. Empty means no category restriction.
    target_categories: tuple[str, ...] = field(default_factory=tuple)


#: The highest CPM any campaign declares. The adapter's bid ceiling MUST exceed
#: this, or an eligible bid is emitted here and then silently discarded by Prebid's
#: price-range filter -- a failure whose symptom is a missing campaign and whose
#: cause is a configuration number (BR-19). Exposed so the ceiling can be derived
#: from it rather than guessed.
def highest_declared_cpm(catalog: "CampaignCatalog") -> float:
    """The maximum declared CPM in the catalog."""
    prices = [c.declared_cpm for c in catalog.all()]
    return max(prices) if prices else 0.0


_CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign(
        campaign_id="camp-cedar",
        campaign_name="Cedar & Co Furnishings",
        adomain="cedarandco.example",
        creative_id="cr-cedar-300x250",
        width=300,
        height=250,
        declared_cpm=6.35,
        deal_ids=("deal-home-premium",),
        target_categories=("home", "lifestyle"),
    ),
    Campaign(
        campaign_id="camp-northlake",
        campaign_name="Northlake Home Goods",
        adomain="northlakehome.example",
        creative_id="cr-northlake-300x250",
        width=300,
        height=250,
        declared_cpm=3.10,
        deal_ids=("deal-retail-run",),
        target_categories=("home", "retail"),
    ),
    Campaign(
        campaign_id="camp-vantage",
        campaign_name="Vantage Motorsport",
        adomain="vantagemotorsport.example",
        creative_id="cr-vantage-300x250",
        width=300,
        height=250,
        declared_cpm=1.80,
        deal_ids=("deal-auto-brand",),
        target_categories=("automotive",),
    ),
    Campaign(
        campaign_id="camp-harbour",
        campaign_name="Harbour Financial",
        adomain="harbourfinancial.example",
        creative_id="cr-harbour-300x250",
        width=300,
        height=250,
        declared_cpm=4.20,
        deal_ids=("deal-finance-pmp",),
        target_categories=("finance",),
    ),
    Campaign(
        campaign_id="camp-openfield",
        campaign_name="Openfield Marketplace",
        adomain="openfield.example",
        creative_id="cr-openfield-300x250",
        width=300,
        height=250,
        declared_cpm=2.05,
        deal_ids=(),
        open_market=True,
    ),
)


class CampaignCatalog:
    """Lookups over the campaigns.

    Returns TUPLES rather than lists so a caller cannot mutate the catalog in
    place. A per-request mutation would make the endpoint stateful and its purity
    claim false.
    """

    def __init__(self, campaigns: tuple[Campaign, ...] = _CAMPAIGNS) -> None:
        self._campaigns = tuple(campaigns)
        by_deal: dict[str, Campaign] = {}
        for campaign in self._campaigns:
            for deal_id in campaign.deal_ids:
                if deal_id in by_deal:
                    # Two campaigns claiming one deal would make the match
                    # ambiguous and the demonstration non-reproducible.
                    raise ValueError(
                        f"deal {deal_id!r} is claimed by both "
                        f"{by_deal[deal_id].campaign_id!r} and {campaign.campaign_id!r}"
                    )
                by_deal[deal_id] = campaign
        self._by_deal = by_deal

    def by_deal(self, deal_id: str) -> Optional[Campaign]:
        """The campaign holding this deal, or None."""
        return self._by_deal.get(deal_id)

    def all(self) -> tuple[Campaign, ...]:
        """Every campaign, immutable."""
        return self._campaigns

    def open_market(self) -> tuple[Campaign, ...]:
        """Campaigns that may offer without a deal."""
        return tuple(c for c in self._campaigns if c.open_market)
