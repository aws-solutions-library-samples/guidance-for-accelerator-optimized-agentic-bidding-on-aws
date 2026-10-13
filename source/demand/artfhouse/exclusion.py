"""Why a campaign made no offer.

A CLOSED set of exactly five members. Closed rather than a free string so the
distinction between a suppressed deal and a below-floor rejection cannot be lost
to a typo, and so the frontend can render every case exhaustively instead of
falling back to an "other" bucket.

ADDING A MEMBER IS A TWO-SIDED CHANGE. The frontend renders these by name, from
its own map in `frontend-react/src/utils/outcomeClassifier.js`, and a reason it
does not know falls to a catch-all that prints "No offer -- no reason reported" --
so a campaign whose reason WAS reported renders as though none was. That is what
happened when MEDIA_TYPE_UNSUPPORTED was added here and not there: 12 of 30
excluded campaigns on the isv-ecosystem scenario claimed no reason existed.
`test_exclusion_reason_parity.py` now fails if the two sides drift again.

TRANSPORT FAILURE IS DELIBERATELY NOT A MEMBER. If the endpoint is unreachable,
no campaign was considered and no campaign declined -- there is simply no answer.
Admitting transport failure here would let "the network broke" render as though a
campaign had chosen not to bid, which is a different claim entirely.
"""

from enum import Enum


class ExclusionReason(str, Enum):
    """The reasons a considered campaign produced no offer."""

    DEAL_SUPPRESSED = "deal_suppressed"
    """The Deal Scorer suppressed the deal this campaign would have used."""

    MEDIA_TYPE_UNSUPPORTED = "media_type_unsupported"
    """The impression offers no slot this campaign's creative could fill.

    A video-only slot cannot show a banner, and a banner-only slot cannot play
    video. Before this reason existed every campaign was implicitly a banner and
    offered on anything, so a banner creative was bid into video slots -- an offer
    that could never have rendered.
    """

    BELOW_FLOOR = "below_floor"
    """The campaign's CPM did not clear the binding floor."""

    NOT_TARGETED = "not_targeted"
    """The campaign's targeting did not match the impression."""

    NO_DEAL_ON_IMPRESSION = "no_deal_on_impression"
    """No deal this campaign holds was present, and it may not offer without one."""


#: Every member, for exhaustiveness checks in tests and consumers.
ALL_EXCLUSION_REASONS = tuple(ExclusionReason)
