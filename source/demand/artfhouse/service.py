"""Turn an enriched bid request into a bid response.

Sequences the pure modules and assembles the response. Holds no decision logic of
its own:

  1. read ``imp.pmp.deals`` and the user's audience segments
  2. evaluate eligibility against the catalog
  3. resolve the binding floor per candidate
  4. build bids (catalogued CPM plus any audience uplift); below-floor candidates
     become exclusions
  5. assemble a single seatbid plus ``ext.artf.excluded``

This endpoint computes NO winner and NO clearing price. Prebid resolves the auction;
the response carries offers and reasons.
"""

from typing import Optional

from . import bids as bid_builder
from . import floors as floor_resolver
from .catalog import CampaignCatalog
from .eligibility import evaluate


class MalformedRequest(ValueError):
    """The request could not be read as an OpenRTB bid request.

    Distinct from "no campaign wished to offer". An empty bid response asserts
    something substantive about demand, so it must never stand in for a request that
    could not be parsed.
    """


def validate(bid_request: Optional[dict]) -> list[str]:
    """Structural errors, empty when the request is acceptable."""
    errors: list[str] = []

    if not isinstance(bid_request, dict):
        return ["request is not a JSON object"]

    if not bid_request.get("id"):
        errors.append("request has no id")

    imps = bid_request.get("imp")
    if not isinstance(imps, list) or not imps:
        errors.append("request has no impressions")
    else:
        for index, imp in enumerate(imps):
            if not isinstance(imp, dict):
                errors.append(f"imp[{index}] is not an object")
            elif not imp.get("id"):
                errors.append(f"imp[{index}] has no id")

    return errors


def user_segment_ids(bid_request: dict) -> tuple[str, ...]:
    """Every segment id on ``user.data[].segment[]``, as strings, in request order.

    One flat tuple across providers: the publisher's DMP and the ARTF Audience
    Activator both write here, under different ``data[].name`` values, and a buyer
    pricing on an audience does not care which provider asserted it.
    """
    user = bid_request.get("user")
    if not isinstance(user, dict):
        return ()
    out: list[str] = []
    for provider in user.get("data") or []:
        if not isinstance(provider, dict):
            continue
        for segment in provider.get("segment") or []:
            if isinstance(segment, dict) and segment.get("id") is not None:
                out.append(str(segment["id"]))
    return tuple(out)


class DemandDecisionService:
    """One pass over a bid request."""

    def __init__(self, catalog: Optional[CampaignCatalog] = None) -> None:
        self._catalog = catalog or CampaignCatalog()

    def decide(self, bid_request: dict) -> dict:
        """The bid response.

        Raises MalformedRequest when the request cannot be read, and CurrencyMismatch
        when it asks for a currency this endpoint does not price in. Neither is
        reported as an empty bid response.
        """
        errors = validate(bid_request)
        if errors:
            raise MalformedRequest("; ".join(errors))

        floor_resolver.assert_supported_currency(bid_request.get("cur"))

        all_bids: list[bid_builder.Bid] = []
        all_exclusions: list[bid_builder.Exclusion] = []
        segment_ids = user_segment_ids(bid_request)

        for imp in bid_request["imp"]:
            candidates = evaluate(imp, self._catalog)

            resolved = {
                c.campaign.campaign_id: floor_resolver.resolve(imp, c.deal)
                for c in candidates
            }

            result = bid_builder.build(imp["id"], candidates, resolved, segment_ids)
            all_bids.extend(result.bids)
            all_exclusions.extend(result.exclusions)

        response: dict = {
            "id": bid_request["id"],
            "cur": floor_resolver.SUPPORTED_CURRENCY,
        }

        # A seatbid with no bids is omitted rather than sent empty: OpenRTB treats an
        # absent seatbid as "no bids", and an empty array invites a consumer to render
        # a seat that offered nothing.
        if all_bids:
            response["seatbid"] = [bid_builder.to_seatbid(tuple(all_bids))]

        if all_exclusions:
            response["ext"] = {
                "artf": {"excluded": bid_builder.to_excluded_ext(tuple(all_exclusions))}
            }

        return response
