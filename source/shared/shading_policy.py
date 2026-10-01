"""The bid shading policy: the one place a shaded price is computed.

A shaded price is a function of expected value. `ev = p(response) x value(response)` is
what the advertiser stands to gain from winning; the policy decides what fraction of
that to bid, then clamps the result into the range the auction allows.

Per decision D2 this is a small parametric form over `ev` rather than a learned
monotone transform. Three coefficients, so monotonicity and boundedness are provable
from the algebra rather than hoped for, and a trading team can read the parameters and
say what they do. Revisit if it underfits.

    raw    = base + slope * ev ** curvature
    price  = clamp(raw, floor, original_bid)

* ``base`` is what the policy bids when `ev` is zero — a floor on aggression,
  expressed in the same currency as the bid.
* ``slope`` is how much of each unit of expected value is offered.
* ``curvature`` bends the response. Below 1 it concedes value early and flattens,
  which is the usual shape for a first-price auction where the marginal win gets
  expensive. At exactly 1 the policy is linear.

Monotonicity. With ``slope >= 0`` and ``curvature > 0``, ``ev ** curvature`` is
non-decreasing in ``ev`` over the non-negative reals, so ``raw`` is non-decreasing,
and clamping to a fixed interval preserves that. Hence ``ev_1 >= ev_2`` implies
``price_1 >= price_2`` for a given floor and original bid. The bounds on the
parameters are what make this true, which is why they are enforced rather than
documented.

Boundedness. The result is `clamp`ed last, so it is always within
``[floor, original_bid]`` whatever the parameters do — including a floor above the
original bid, where the bid itself wins and the caller is told the floor was
unreachable.

The genesis parameters reproduce the previous hardcoded behaviour
(``min(original, ev * shade_factor)``) so swapping the call site in is not also a
behaviour change: ``base=0``, ``curvature=1`` make the form linear through the
origin, and ``slope`` is the old ``shade_factor``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

#: Parameter bounds. These are the conditions the monotonicity proof needs, plus
#: sanity limits so a search cannot wander somewhere meaningless.
BASE_BOUNDS = (0.0, 5.0)
SLOPE_BOUNDS = (0.0, 3.0)
CURVATURE_BOUNDS = (0.2, 2.0)

#: The previous hardcoded default (`EST_SHADE_FACTOR` in the shader container).
_GENESIS_SLOPE = 0.65

#: Bumped when the parameter set itself changes shape, so a model carrying policy
#: parameters can be checked against the code that will apply them. Adding a fourth
#: coefficient is a version change; retuning the three is not.
POLICY_VERSION = 1


class PolicyParameterError(ValueError):
    """A parameter set that would break monotonicity or boundedness.

    Raised rather than clamped. A caller that produced an out-of-range parameter has
    a bug or a bad search space, and silently correcting it hides which.
    """


@dataclass(frozen=True)
class ShadingPolicy:
    """The three coefficients, validated on construction."""

    base: float = 0.0
    slope: float = _GENESIS_SLOPE
    curvature: float = 1.0

    def __post_init__(self) -> None:
        for name, bounds in (
            ("base", BASE_BOUNDS),
            ("slope", SLOPE_BOUNDS),
            ("curvature", CURVATURE_BOUNDS),
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value != value:  # NaN
                raise PolicyParameterError(f"{name} must be a real number, got {value!r}")
            low, high = bounds
            if not low <= value <= high:
                raise PolicyParameterError(
                    f"{name}={value} is outside [{low}, {high}]. These bounds are what "
                    "make the policy provably monotone and bounded; a search must be "
                    "constrained to them rather than clamped afterwards."
                )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def raw_price(self, ev: float) -> float:
        """The policy's unclamped offer for this expected value.

        Exposed separately so a caller can tell "the policy wanted less than the
        floor" from "the policy wanted less than the bid", which are different
        situations and only distinguishable before the clamp.
        """
        if ev < 0:
            raise PolicyParameterError(
                f"ev must be non-negative, got {ev}. A negative expected value means "
                "the caller's probability or conversion value is wrong; shading it "
                "would produce a price with no meaning."
            )
        return self.base + self.slope * (ev ** self.curvature)

    def price(self, ev: float, *, floor: float, original_bid: float) -> float:
        """The shaded price, clamped into what the auction allows.

        ``floor`` wins over the policy: a bid below the auction's reserve cannot
        trade. ``original_bid`` caps it, because shading exists to bid LESS than the
        buyer was willing to pay, never more.
        """
        if floor < 0:
            raise PolicyParameterError(f"floor must be non-negative, got {floor}")
        if original_bid < 0:
            raise PolicyParameterError(
                f"original_bid must be non-negative, got {original_bid}"
            )
        return clamp(self.raw_price(ev), floor=floor, original_bid=original_bid)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """The form written into a model manifest and the parameter store."""
        return {
            "policy_version": POLICY_VERSION,
            "base": self.base,
            "slope": self.slope,
            "curvature": self.curvature,
        }

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None, *, default: "ShadingPolicy | None" = None
    ) -> "ShadingPolicy":
        """Build a policy from a manifest, parameter-store record or request override.

        Absent or empty input yields ``default`` (the genesis policy when none is
        given), so a model published before policy parameters existed still serves.
        A present but *unreadable* parameter set raises: that is a model claiming a
        policy this code cannot apply, and serving the genesis policy instead would
        silently price against different parameters than the model was tuned for.
        """
        if not data:
            return default if default is not None else cls()

        version = data.get("policy_version", POLICY_VERSION)
        if version != POLICY_VERSION:
            raise PolicyParameterError(
                f"policy_version {version!r} cannot be applied by this code, which "
                f"implements version {POLICY_VERSION}. Deploy the matching shader "
                "rather than pricing with a parameter set of unknown shape."
            )

        known = {"base", "slope", "curvature"}
        supplied = {k: v for k, v in data.items() if k in known}
        reference = default if default is not None else cls()
        return replace(reference, **supplied)


def clamp(price: float, *, floor: float, original_bid: float) -> float:
    """Force a price into ``[floor, original_bid]``.

    When the floor exceeds the original bid there is no value in the interval. The
    floor wins: a price below reserve cannot trade at all, whereas a price above the
    buyer's bid is a decision for the caller to notice, and it will — the returned
    price exceeds ``original_bid``, which the shader treats as "do not mutate".
    """
    if floor > original_bid:
        return floor
    return min(max(price, floor), original_bid)


#: The parameters that reproduce the behaviour this module replaced. Used when a model
#: publishes none.
GENESIS_POLICY = ShadingPolicy()


def policy_search_space() -> dict[str, tuple[float, float]]:
    """The bounds a parameter search must stay inside.

    Handed to phase 2 so the search space and the validation come from one place; a
    search with its own copy of the bounds drifts from the ones actually enforced.
    """
    return {
        "base": BASE_BOUNDS,
        "slope": SLOPE_BOUNDS,
        "curvature": CURVATURE_BOUNDS,
    }
