"""The campaign catalog.

Campaigns are loaded from the stack's deals table once per process (see
deals_source.py), so the decision is a pure function of the request and the
loaded catalog: no table read, cache refresh or network call happens while a
request is decided, and unit tests read the same records from the seed file.

The records are authored host configuration: every bid traces to a campaign
record, and this endpoint does not decide the auction. A campaign references
the IAB Tech Lab Deals API v1.0 deals it transacts on by ``Deal.id``; the deal
records themselves (floors, auction types, inventory) live in the same table
and are read by the Deal Scorer, not here.

NOT EMBEDDED IN ANY ARTF AGENT IMAGE. The catalog belongs to the host's demand
endpoint. Putting it inside an agent container would make an agent carry its own
demand, which inverts the framework's ownership model.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import deals_source


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
    #: What the creative IS: "banner" or "video".
    #:
    #: Load-bearing, not decoration. An impression that offers only a video slot
    #: cannot show a banner, so a banner campaign must not offer on it -- and until
    #: this field existed every campaign was implicitly a banner and would offer on
    #: anything. The bid also has to declare it: OpenRTB 2.6 `mtype` is how the
    #: exchange learns a bid's media type, and without it the adapter has to guess.
    media_type: str = "banner"
    #: Creative duration in seconds. Video only; ignored for banner. Video slots
    #: state minduration/maxduration, and a creative outside that range cannot run.
    duration: Optional[int] = None
    #: What this buyer pays extra when the user carries one of its target audience
    #: segments. None means the campaign prices on content only.
    audience_uplift: Optional["AudienceUplift"] = None
    #: Content identifiers it will bid on, compared with the ``cid:`` metric types an
    #: ADD_CIDS container appends to ``imp.metric``. Empty means no content
    #: restriction. Unlike categories, an impression with no content identifier
    #: does NOT match a campaign that names one.
    target_content_ids: tuple[str, ...] = field(default_factory=tuple)

    def price_for(self, segment_ids: tuple[str, ...]) -> float:
        """The CPM this campaign offers to a user carrying ``segment_ids``.

        The declared CPM, plus the uplift when any target segment is present. The
        catalog is still the single source of the price: both numbers are authored
        here, and which applies is decided by the request's own audience data.
        """
        if self.audience_uplift and self.audience_uplift.matches(segment_ids):
            return round(self.declared_cpm + self.audience_uplift.cpm, 4)
        return self.declared_cpm

    @property
    def ceiling_cpm(self) -> float:
        """The most this campaign can ever offer."""
        return round(self.declared_cpm + (self.audience_uplift.cpm if self.audience_uplift else 0.0), 4)


@dataclass(frozen=True)
class AudienceUplift:
    """A CPM premium a buyer pays for a matched audience.

    Segment ids are compared as strings against ``user.data[].segment[].id``, which
    is where both the publisher's DMP and the ARTF Audience Activator place them.
    The ids below are IAB Audience Taxonomy 1.1 identifiers, since that is what the
    Audience Activator emits; a publisher's proprietary segment ids never collide
    with them.
    """

    segment_ids: tuple[str, ...]
    cpm: float

    def matches(self, segment_ids: tuple[str, ...]) -> bool:
        return any(s in self.segment_ids for s in segment_ids)


#: The highest CPM any campaign can offer, uplift included. The adapter's bid
#: ceiling MUST exceed this, or an eligible bid is emitted here and then silently
#: discarded by Prebid's price-range filter -- a failure whose symptom is a missing
#: campaign and whose cause is a configuration number (BR-19). Exposed so the
#: ceiling can be derived from it rather than guessed.
def highest_declared_cpm(catalog: "CampaignCatalog") -> float:
    """The maximum CPM any campaign in the catalog can offer."""
    prices = [c.ceiling_cpm for c in catalog.all()]
    return max(prices) if prices else 0.0


def campaigns_from_records(records: Iterable[dict]) -> tuple[Campaign, ...]:
    """Build frozen campaigns from campaign records, in record order.

    A record's ``note`` is documentation for whoever edits the seed file and is
    not part of the campaign.
    """
    campaigns = []
    for record in records:
        uplift = record.get("audience_uplift")
        duration = record.get("duration")
        campaigns.append(Campaign(
            campaign_id=str(record["campaign_id"]),
            campaign_name=str(record["campaign_name"]),
            adomain=str(record["adomain"]),
            creative_id=str(record["creative_id"]),
            width=int(record["width"]),
            height=int(record["height"]),
            declared_cpm=float(record["declared_cpm"]),
            deal_ids=tuple(str(d) for d in record.get("deal_ids") or ()),
            open_market=bool(record.get("open_market", False)),
            target_categories=tuple(str(c) for c in record.get("target_categories") or ()),
            media_type=str(record.get("media_type") or "banner"),
            duration=int(duration) if duration is not None else None,
            target_content_ids=tuple(str(c) for c in record.get("target_content_ids") or ()),
            audience_uplift=(
                AudienceUplift(
                    segment_ids=tuple(str(s) for s in uplift["segment_ids"]),
                    cpm=float(uplift["cpm"]),
                )
                if uplift else None
            ),
        ))
    return tuple(campaigns)


class CampaignCatalog:
    """Lookups over the campaigns.

    Returns TUPLES rather than lists so a caller cannot mutate the catalog in
    place. A per-request mutation would make the endpoint stateful and its purity
    claim false.
    """

    def __init__(self, campaigns: Optional[Iterable[Campaign]] = None) -> None:
        if campaigns is None:
            campaigns = campaigns_from_records(deals_source.campaign_records())
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
