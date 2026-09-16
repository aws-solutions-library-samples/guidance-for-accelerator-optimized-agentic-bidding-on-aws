"""Why a campaign made no offer.

A CLOSED set of exactly four members. Closed rather than a free string so the
distinction between a suppressed deal and a below-floor rejection cannot be lost
to a typo, and so the frontend can render every case exhaustively instead of
falling back to an "other" bucket.

TRANSPORT FAILURE IS DELIBERATELY NOT A MEMBER. If the endpoint is unreachable,
no campaign was considered and no campaign declined -- there is simply no answer.
Admitting transport failure here would let "the network broke" render as though a
campaign had chosen not to bid, which is a different claim entirely.
"""

from enum import Enum


class ExclusionReason(str, Enum):
    """The four reasons a considered campaign produced no offer."""

    DEAL_SUPPRESSED = "deal_suppressed"
    """The Deal Scorer suppressed the deal this campaign would have used."""

    BELOW_FLOOR = "below_floor"
    """The campaign's CPM did not clear the binding floor."""

    NOT_TARGETED = "not_targeted"
    """The campaign's targeting did not match the impression."""

    NO_DEAL_ON_IMPRESSION = "no_deal_on_impression"
    """No deal this campaign holds was present, and it may not offer without one."""


#: Every member, for exhaustiveness checks in tests and consumers.
ALL_EXCLUSION_REASONS = tuple(ExclusionReason)
