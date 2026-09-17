"""Bid construction. Pure.

One bid per eligible campaign that clears its floor, all within a SINGLE seatbid,
with no seat aliases. Every offer arrives under one seat, which is why the frontend
column is labelled offers rather than bidders.

A bid's price is the campaign's CATALOGUED CPM. It is not the floor, and it is not a
computed clearing price -- this endpoint does not resolve the auction. Prebid does.

Candidates that do not clear their floor become ``below_floor`` exclusions rather
than disappearing, so bids and exclusions PARTITION the candidate set: every
candidate becomes exactly one of the two, and the set covers every campaign
considered.
"""

from dataclasses import dataclass
from typing import Optional

from .eligibility import Candidate
from .exclusion import ExclusionReason
from .floors import BindingFloor

#: The single seat every offer arrives under.
SEAT = "artfhouse"


@dataclass(frozen=True)
class Bid:
    """One offer."""

    bid_id: str
    imp_id: str
    campaign_id: str
    campaign_name: str
    deal_id: Optional[str]
    price: float
    adomain: str
    creative_id: str
    width: int
    height: int
    #: Which floor the price had to clear, for the response's supporting detail.
    binding_floor: float
    floor_bound_by: str
    #: "banner" or "video", carried from the campaign. Emitted as OpenRTB `mtype`.
    media_type: str = "banner"
    #: Creative duration in seconds. Video only.
    duration: Optional[int] = None


@dataclass(frozen=True)
class Exclusion:
    """One campaign that made no offer on one impression, and why.

    PER IMPRESSION, like Bid. A campaign is considered once per impression and can be
    excluded for different reasons on each -- ineligible on a video slot, below floor
    on a banner one -- so an exclusion without its impression is not a whole fact.

    Omitting imp_id made a two-impression request emit the same campaign twice,
    byte-identical: 30 entries for 16 campaigns on the isv-ecosystem scenario, 14 of
    them exact duplicates. The consumer could neither tell them apart nor say which
    impression either referred to, and the offers column keys its rows on campaign
    plus deal, so the two collided on one key as well.
    """

    campaign_id: str
    campaign_name: str
    deal_id: Optional[str]
    reason: ExclusionReason
    imp_id: str


@dataclass(frozen=True)
class BuildResult:
    """Bids and exclusions, which together account for every candidate."""

    bids: tuple[Bid, ...]
    exclusions: tuple[Exclusion, ...]


def build(
    imp_id: str,
    candidates: tuple[Candidate, ...],
    floors: dict[str, BindingFloor],
) -> BuildResult:
    """Emit a bid per clearing candidate; everything else becomes an exclusion.

    ``floors`` is keyed by campaign id, so each candidate is compared against the
    floor resolved for its own deal rather than a single request-wide value.
    """
    bids: list[Bid] = []
    exclusions: list[Exclusion] = []

    for candidate in candidates:
        campaign = candidate.campaign

        if not candidate.eligible:
            exclusions.append(
                Exclusion(
                    campaign_id=campaign.campaign_id,
                    campaign_name=campaign.campaign_name,
                    deal_id=candidate.deal_id,
                    reason=candidate.excluded_because,
                    imp_id=imp_id,
                )
            )
            continue

        floor = floors.get(campaign.campaign_id)
        if floor is not None and not floor.clears(campaign.declared_cpm):
            exclusions.append(
                Exclusion(
                    campaign_id=campaign.campaign_id,
                    campaign_name=campaign.campaign_name,
                    deal_id=candidate.deal_id,
                    reason=ExclusionReason.BELOW_FLOOR,
                    imp_id=imp_id,
                )
            )
            continue

        bids.append(
            Bid(
                bid_id=f"bid-{campaign.campaign_id}",
                imp_id=imp_id,
                campaign_id=campaign.campaign_id,
                campaign_name=campaign.campaign_name,
                deal_id=candidate.deal_id,
                price=campaign.declared_cpm,
                adomain=campaign.adomain,
                creative_id=campaign.creative_id,
                width=campaign.width,
                height=campaign.height,
                binding_floor=floor.value if floor else 0.0,
                floor_bound_by=floor.bound_by.value if floor else "impression",
                media_type=campaign.media_type,
                duration=campaign.duration,
            )
        )

    return BuildResult(bids=tuple(bids), exclusions=tuple(exclusions))


#: OpenRTB 2.6 mtype codes. Only the two this catalog produces are mapped, so an
#: unmapped media type raises here rather than being emitted as a silent default
#: that the exchange would then treat as a banner.
_MTYPE = {"banner": 1, "video": 2}


def video_adm(creative_id: str, duration: int, width: int, height: int) -> str:
    """A VAST document for a video bid.

    WHY A VIDEO BID CANNOT OMIT THIS

    Prebid's ResponseBidValidator rejects a video bid carrying neither ``adm`` nor
    ``nurl``, and says so exactly:

        Bid "bid-camp-skyline-video" with video type missing adm and nurl

    Every video campaign was bid correctly by this endpoint -- right price, right
    mtype, right duration -- and then dropped by the exchange for having no
    creative. The auction reported one seat, and nothing in the seatbid explained
    why the other was absent.

    WHAT THIS DOCUMENT IS, PLAINLY

    A structurally valid VAST wrapper around a MediaFile path that IS NOT SHIPPED.
    No video asset exists in this repository, so there is nothing to point at that
    would play, and inventing a path that looks like a real creative would assert a
    file that does not exist. The AdSystem and AdTitle say what it is, so anyone who
    inspects a winning bid reads "placeholder" rather than a brand name.

    The release's own simulator has the same property -- its MediaFile paths under
    /assets/videos/ are not shipped here either -- so neither seat's video creative
    is playable in this deployment. What is real is the auction: the bid, its price,
    its duration, and which seat won.
    """
    clock = f"00:00:{min(duration, 59):02d}"
    return (
        "<VAST version='3.0'>"
        f"<Ad id='{creative_id}'>"
        "<InLine>"
        "<AdSystem>ARTF demand endpoint (placeholder creative)</AdSystem>"
        f"<AdTitle>{creative_id} -- placeholder, no video asset is shipped</AdTitle>"
        "<Creatives><Creative><Linear>"
        f"<Duration>{clock}</Duration>"
        "<MediaFiles>"
        f"<MediaFile delivery='progressive' type='video/mp4' width='{width}' "
        f"height='{height}'>"
        f"<![CDATA[/assets/videos/not-shipped/{creative_id}.mp4]]>"
        "</MediaFile>"
        "</MediaFiles>"
        "</Linear></Creative></Creatives>"
        "</InLine></Ad></VAST>"
    )


def to_seatbid(bids: tuple[Bid, ...]) -> dict:
    """A single seatbid. No aliases.

    Campaign identity rides in ``ext.prebid.artf`` because OpenRTB's bid object has
    no campaign field, and the frontend needs the identity to label the row.
    """
    return {
        "seat": SEAT,
        "bid": [
            {
                "id": b.bid_id,
                "impid": b.imp_id,
                "price": b.price,
                **({"dealid": b.deal_id} if b.deal_id else {}),
                "adomain": [b.adomain],
                "crid": b.creative_id,
                "w": b.width,
                "h": b.height,
                # OpenRTB 2.6 mtype: 1 banner, 2 video, 3 audio, 4 native. Stated
                # rather than left to the exchange to infer -- an adapter that has
                # to guess will guess one value for every bid, and a video bid
                # reported as a banner cannot render.
                "mtype": _MTYPE[b.media_type],
                # Video only. A slot declares minduration/maxduration, and a bid
                # that does not say how long its creative runs cannot be checked
                # against them.
                **({"dur": b.duration} if b.duration is not None else {}),
                # Video only, and REQUIRED there: Prebid rejects a video bid with
                # neither adm nor nurl. See video_adm for what the document is and
                # what it does not claim.
                **(
                    {"adm": video_adm(b.creative_id, b.duration or 15, b.width, b.height)}
                    if b.media_type == "video"
                    else {}
                ),
                "ext": {
                    "prebid": {
                        "artf": {
                            "campaignId": b.campaign_id,
                            "campaignName": b.campaign_name,
                            "dealId": b.deal_id,
                            "bindingFloor": b.binding_floor,
                            "floorBoundBy": b.floor_bound_by,
                        }
                    }
                },
            }
            for b in bids
        ],
    }


def to_excluded_ext(exclusions: tuple[Exclusion, ...]) -> list[dict]:
    """The ``ext.artf.excluded`` block.

    WHY THIS EXISTS RATHER THAN USING ``ext.seatnonbid``: seatnonbid is a Prebid
    Server construct recording why a SEAT produced no bid for an impression. If this
    seat bids at all, Prebid records no seatnonbid entry for it -- so campaigns
    considered and never offered would be invisible. seatnonbid remains correct for
    Prebid's own rejections (a bid it dropped below floor, code 301); this block
    carries what only the endpoint knows.
    """
    return [
        {
            "campaignId": e.campaign_id,
            "campaignName": e.campaign_name,
            "dealId": e.deal_id,
            "exclusionReason": e.reason.value,
            # The impression this exclusion is about. Without it a multi-impression
            # request emits the same campaign once per impression with nothing to
            # distinguish the entries.
            "impId": e.imp_id,
        }
        for e in exclusions
    ]
