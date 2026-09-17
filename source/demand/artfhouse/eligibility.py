"""Eligibility. Pure.

EVERY campaign in the catalog is CONSIDERED for every impression, and each
consideration yields exactly one Candidate -- eligible, or excluded with a reason.

This completeness is the most consequential property of the unit. A campaign that
simply vanished from the output would be indistinguishable from one never
considered, and the interface would then have to render an ABSENCE rather than a
REASON. Those two read completely differently: an omitted row says "this campaign
was not involved", a row reading "suppressed by the Deal Scorer" says the feature
worked.

The below-floor decision is NOT made here -- it needs the binding floor, so it is
made when bids are built. A candidate leaving this stage as eligible has passed
deal matching and targeting only.
"""

from dataclasses import dataclass
from typing import Optional

from .catalog import Campaign, CampaignCatalog
from .exclusion import ExclusionReason


@dataclass(frozen=True)
class Candidate:
    """One campaign's consideration for one impression."""

    campaign: Campaign
    #: The deal it would transact on, or None for an open-market offer.
    deal: Optional[dict]
    #: None when eligible; otherwise why it made no offer.
    excluded_because: Optional[ExclusionReason] = None

    @property
    def eligible(self) -> bool:
        return self.excluded_because is None

    @property
    def deal_id(self) -> Optional[str]:
        return (self.deal or {}).get("id") if self.deal else None


def _deals_on(imp: dict) -> tuple[dict, ...]:
    return tuple((imp or {}).get("pmp", {}).get("deals", []) or ())


def _is_suppressed(deal: dict) -> bool:
    """A deal the Deal Scorer suppressed.

    Suppression is carried on the deal itself, since the ARTF containers mutate
    ``imp.pmp.deals`` in place and the hook applies those mutations before the
    bidder fan-out.
    """
    ext = (deal or {}).get("ext") or {}
    if ext.get("suppressed") is True:
        return True
    artf = ext.get("artf") or {}
    return artf.get("suppressed") is True


def _categories_of(imp: dict) -> tuple[str, ...]:
    """Content categories on the impression, if any.

    TWO locations are read, and both are load-bearing:

      imp.ext.data.artf.categories  -- the only one that SURVIVES Prebid Server.
          Prebid rewrites imp.ext per bidder and removes keys it does not recognise,
          treating an unknown key as a bidder name: a request carrying
          imp.ext.artf came back with "request.imp[0].ext.prebid.bidder.artf was
          dropped ... contains unknown bidder: artf". imp.ext.data is first-party
          data, which Prebid preserves and forwards, so that is where categories
          have to live to reach this endpoint through an auction.

      imp.ext.artf.categories       -- kept for callers that reach this endpoint
          DIRECTLY, without Prebid in the path, which is how the demand endpoint is
          exercised in isolation.

    Read in that order, so the Prebid-safe location wins when both are present.
    """
    ext = (imp or {}).get("ext") or {}

    data_artf = ((ext.get("data") or {}).get("artf")) or {}
    cats = data_artf.get("categories") or []
    if not cats:
        cats = (ext.get("artf") or {}).get("categories") or []

    return tuple(str(c) for c in cats)


def _targets(campaign: Campaign, categories: tuple[str, ...]) -> bool:
    """Whether the campaign's targeting matches.

    An empty target list means no restriction. An impression with no declared
    categories matches everything -- absence of a signal is not treated as a
    mismatch, since that would exclude campaigns for a reason the request never
    stated.
    """
    if not campaign.target_categories:
        return True
    if not categories:
        return True
    return any(c in campaign.target_categories for c in categories)


def _media_types_of(imp: dict) -> tuple[str, ...]:
    """The media types the impression actually offers a slot for.

    Read from the OpenRTB objects themselves -- an imp declares a slot by carrying
    `banner`, `video`, `audio` or `native`. An imp with none of them offers nothing
    placeable, which is treated as no restriction rather than as a silent exclusion
    of the whole catalog, so a malformed request fails visibly elsewhere instead of
    presenting as "no demand".
    """
    present = tuple(k for k in ("banner", "video", "audio", "native") if (imp or {}).get(k))
    return present


def _placeable(campaign: Campaign, media_types: tuple[str, ...]) -> bool:
    """Whether the impression has a slot this campaign's creative could fill."""
    if not media_types:
        return True
    return campaign.media_type in media_types


def evaluate(imp: dict, catalog: CampaignCatalog) -> tuple[Candidate, ...]:
    """One Candidate per considered campaign, in catalog order.

    Order is stable so the same request yields the same response (BR-4).
    """
    deals = _deals_on(imp)
    deals_by_id = {d.get("id"): d for d in deals if d.get("id")}
    categories = _categories_of(imp)
    media_types = _media_types_of(imp)

    candidates: list[Candidate] = []

    for campaign in catalog.all():
        # Checked FIRST, and before the deal lookup, because it is the most
        # fundamental reason: a creative that cannot be placed in any slot on this
        # impression is not a near miss on price or targeting. Reporting
        # "below_floor" for a video campaign on a banner-only slot would name a
        # reason that had nothing to do with why it lost.
        if not _placeable(campaign, media_types):
            candidates.append(
                Candidate(campaign, None, ExclusionReason.MEDIA_TYPE_UNSUPPORTED)
            )
            continue

        matching = [deals_by_id[d] for d in campaign.deal_ids if d in deals_by_id]

        if not matching:
            if campaign.open_market:
                if not _targets(campaign, categories):
                    candidates.append(
                        Candidate(campaign, None, ExclusionReason.NOT_TARGETED)
                    )
                else:
                    candidates.append(Candidate(campaign, None))
            else:
                candidates.append(
                    Candidate(campaign, None, ExclusionReason.NO_DEAL_ON_IMPRESSION)
                )
            continue

        # A campaign may hold several deals on one impression. It transacts on the
        # first live one; if all are suppressed, the suppression is the reason.
        live = [d for d in matching if not _is_suppressed(d)]
        if not live:
            candidates.append(
                Candidate(campaign, matching[0], ExclusionReason.DEAL_SUPPRESSED)
            )
            continue

        deal = live[0]
        if not _targets(campaign, categories):
            candidates.append(Candidate(campaign, deal, ExclusionReason.NOT_TARGETED))
            continue

        candidates.append(Candidate(campaign, deal))

    return tuple(candidates)
