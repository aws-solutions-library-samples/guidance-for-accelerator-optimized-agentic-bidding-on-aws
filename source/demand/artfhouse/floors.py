"""Floor resolution. Pure.

The binding floor is the HIGHER of the impression's own floor and the
ARTF-resolved deal floor. The resolver reports both the value and WHICH OF THE TWO
BOUND.

Reporting which bound is not decoration. When a container adjusts a deal floor, the
visible evidence that the adjustment mattered is that the deal floor became the
binding one. Without that, an effective adjustment and a coincidence look identical.

Two semantics are fixed here because leaving them implicit invites divergence:

  - A CPM exactly EQUAL to the binding floor CLEARS. The floor is an inclusive
    minimum. This matches how Prebid's bidder treats its price range as inclusive of
    ``imp.bidfloor``, so a bid at the floor is not silently dropped downstream.

  - All comparisons are CPM in a SINGLE CURRENCY. No implicit conversion happens
    anywhere. A request in another currency is a validation error, not a rescale --
    an implicit conversion would make every floor comparison wrong in a way nothing
    surfaces.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class FloorSource(str, Enum):
    """Which floor bound."""

    IMPRESSION = "impression"
    DEAL = "deal"


@dataclass(frozen=True)
class BindingFloor:
    """The price a candidate must clear, and where it came from."""

    value: float
    bound_by: FloorSource

    def clears(self, cpm: float) -> bool:
        """Whether this CPM clears the floor. Equal clears."""
        return cpm >= self.value


class CurrencyMismatch(ValueError):
    """Raised when a request asks for a currency this endpoint does not price in."""


#: The only currency this endpoint prices in.
SUPPORTED_CURRENCY = "USD"


def assert_supported_currency(currency: Optional[str]) -> None:
    """Reject a currency we cannot compare against, rather than rescaling silently.

    An absent currency is accepted as the default, matching OpenRTB, where ``cur``
    is optional and USD is the assumed default.
    """
    if currency is None:
        return
    if currency.upper() != SUPPORTED_CURRENCY:
        raise CurrencyMismatch(
            f"this endpoint prices in {SUPPORTED_CURRENCY}; request asked for {currency!r}"
        )


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def resolve(imp: dict, deal: Optional[dict]) -> BindingFloor:
    """The binding floor for one impression and one deal.

    ``deal`` may be None, for an open-market candidate. The impression floor then
    binds by definition.
    """
    imp_floor = _as_float((imp or {}).get("bidfloor")) or 0.0
    deal_floor = _as_float((deal or {}).get("bidfloor")) if deal else None

    if deal_floor is not None and deal_floor > imp_floor:
        return BindingFloor(value=deal_floor, bound_by=FloorSource.DEAL)
    return BindingFloor(value=imp_floor, bound_by=FloorSource.IMPRESSION)
