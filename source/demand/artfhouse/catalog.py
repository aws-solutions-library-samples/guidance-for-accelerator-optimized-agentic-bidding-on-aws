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

    # -----------------------------------------------------------------------
    # CAMPAIGNS FOR THE DEALS THE SHIPPED SCENARIOS ACTUALLY CARRY.
    #
    # The five campaigns above transact on deal-home-premium, deal-retail-run,
    # deal-auto-brand and deal-finance-pmp. NONE of those ids appears in
    # source/frontend-react/public/samples/, which carries deal-premium-auto,
    # deal-parenting-premium, deal-premium-video and so on. So on every shipped
    # scenario this seat could only ever offer camp-openfield at 2.05 open-market
    # and lose to the simulator's 3.25 -- the deal path, which is the point of the
    # ARTF story, never ran outside a hand-built request.
    #
    # Each campaign below transacts on one scenario deal. The prices are authored,
    # like every price in this catalog, and are set ABOVE the binding floor for
    # their impression -- the higher of the impression floor and the deal floor --
    # because a campaign priced under the floor is excluded and demonstrates
    # nothing. The floors they are set against are stated per campaign so the
    # relationship is checkable rather than asserted; test_artf_demand_catalog.py
    # verifies it against the scenario files themselves.
    # -----------------------------------------------------------------------

    # --- isv-ecosystem: banner 970x250, imp floor 4.00, private auction ---
    Campaign(
        campaign_id="camp-autoline-premium",
        campaign_name="Autoline Premium",
        adomain="autoline.example",
        creative_id="cr-autoline-970x250",
        width=970,
        height=250,
        declared_cpm=7.20,          # deal floor 6.00 binds
        deal_ids=("deal-premium-auto",),
        target_categories=("automotive",),
    ),
    Campaign(
        campaign_id="camp-autoline-standard",
        campaign_name="Autoline Standard",
        adomain="autoline.example",
        creative_id="cr-autoline-standard-970x250",
        width=970,
        height=250,
        declared_cpm=4.60,          # imp floor 4.00 binds, above the 3.50 deal floor
        deal_ids=("deal-standard-auto",),
        target_categories=("automotive",),
    ),

    # --- parenting-narrative: banner 300x250, imp floor 2.60, open auction ---
    Campaign(
        campaign_id="camp-brightstart",
        campaign_name="Brightstart Family",
        adomain="brightstart.example",
        creative_id="cr-brightstart-300x250",
        width=300,
        height=250,
        declared_cpm=4.10,          # deal floor 3.40 binds
        deal_ids=("deal-parenting-premium",),
        target_categories=("parenting", "family"),
        # Pays more for a user the Audience Activator places in Parenting (350) or
        # Parenting > Babies and Toddlers (354). 4.50 when matched.
        audience_uplift=AudienceUplift(segment_ids=("350", "354"), cpm=0.40),
    ),
    Campaign(
        campaign_id="camp-familynet",
        campaign_name="Family Network Collective",
        adomain="familynetwork.example",
        creative_id="cr-familynet-300x250",
        width=300,
        height=250,
        declared_cpm=3.05,          # imp floor 2.60 binds, above the 2.10 deal floor
        deal_ids=("deal-family-network",),
        target_categories=("parenting", "family"),
        # Pays more for Parents with Children (98). 3.30 when matched.
        audience_uplift=AudienceUplift(segment_ids=("98",), cpm=0.25),
    ),
    Campaign(
        campaign_id="camp-remnant-open",
        campaign_name="Remnant Open Exchange",
        adomain="remnantopen.example",
        creative_id="cr-remnant-300x250",
        width=300,
        height=250,
        declared_cpm=2.75,          # imp floor 2.60 binds, far above the 0.85 deal floor
        deal_ids=("deal-remnant-open",),
    ),

    # --- video-deals: video 640x480, 15-30s, imp floor 8.00, private auction ---
    Campaign(
        campaign_id="camp-skyline-video",
        campaign_name="Skyline Premium Video",
        adomain="skylinevideo.example",
        creative_id="cr-skyline-640x480",
        width=640,
        height=480,
        declared_cpm=11.50,         # deal floor 10.00 binds
        deal_ids=("deal-premium-video",),
        media_type="video",
        duration=30,
    ),
    Campaign(
        campaign_id="camp-skyline-standard",
        campaign_name="Skyline Standard Video",
        adomain="skylinevideo.example",
        creative_id="cr-skyline-standard-640x480",
        width=640,
        height=480,
        declared_cpm=8.75,          # imp floor 8.00 binds, above the 5.00 deal floor
        deal_ids=("deal-standard-video",),
        media_type="video",
        duration=15,
    ),
    Campaign(
        campaign_id="camp-video-remnant",
        campaign_name="Video Remnant Pool",
        adomain="videoremnant.example",
        creative_id="cr-video-remnant-640x480",
        width=640,
        height=480,
        declared_cpm=8.10,          # imp floor 8.00 binds, far above the 0.50 deal floor
        deal_ids=("deal-remnant",),
        media_type="video",
        duration=15,
    ),

    # --- yield-optimizer: video 1280x720, 15-30s, imp floor 6.00, private auction ---
    Campaign(
        campaign_id="camp-meridian-guaranteed",
        campaign_name="Meridian Guaranteed",
        adomain="meridianmedia.example",
        creative_id="cr-meridian-1280x720",
        width=1280,
        height=720,
        declared_cpm=13.40,         # deal floor 12.00 binds
        deal_ids=("deal-guaranteed-premium",),
        media_type="video",
        duration=30,
    ),
    Campaign(
        campaign_id="camp-meridian-midtier",
        campaign_name="Meridian Mid-tier",
        adomain="meridianmedia.example",
        creative_id="cr-meridian-midtier-1280x720",
        width=1280,
        height=720,
        declared_cpm=6.80,          # imp floor 6.00 binds, above the 3.25 deal floor
        deal_ids=("deal-open-midtier",),
        media_type="video",
        duration=15,
    ),
    Campaign(
        campaign_id="camp-meridian-remnant",
        campaign_name="Meridian Remnant",
        adomain="meridianmedia.example",
        creative_id="cr-meridian-remnant-1280x720",
        width=1280,
        height=720,
        declared_cpm=6.20,          # imp floor 6.00 binds, far above the 0.75 deal floor
        deal_ids=("deal-open-remnant",),
        media_type="video",
        duration=15,
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
