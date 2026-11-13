"""Outcome simulator — SYNTHETIC auction outcomes for a deployment with no signal feed.

Everything this module produces is synthetic. It exists because the bid-shading
training loop needs win/impression/click/conversion labels, and in a deployment with
no exchange win notice and no advertiser pixel there is nothing to report them. The
simulator stands in for that feed so the loop can be exercised end to end.

Every signal it emits carries ``provenance="simulated"``, which travels into the
enriched outcome event, into the training record, and into the trained model's
manifest. A model trained on simulated outcomes is therefore always identifiable as
one, and can never be presented as trained on observed behaviour.

Two properties make this a labelled test affordance rather than fabricated data:

* It is **off by default**. Nothing runs unless ``OUTCOME_SIMULATOR_ENABLED`` is set
  to a truthy value.
* It is **deterministic and inspectable**. Outcomes are derived from a hash of the
  ``request_id``, not from ``random``. The same request always yields the same
  outcome, so any run can be reproduced and any label can be recomputed by hand.

What it deliberately does NOT model: conversion lag. Signals are applied
immediately, so a simulated dataset carries no delay between bid and conversion.
The ETL's aggregation window is a separate concern.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

from shared.feedback_models import BidShadingOutcomeEvent
from shared.signal_associator import DownstreamSignal, SignalType

logger = logging.getLogger(__name__)

#: Environment variable gating the whole module. Absent or falsey → nothing runs.
ENABLE_ENV_VAR = "OUTCOME_SIMULATOR_ENABLED"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Default conditional rates. Each is conditional on the previous stage having
# occurred, which is how the funnel actually composes: a click cannot precede an
# impression. Chosen to sit in the range commonly seen in display advertising so a
# simulated dataset is not trivially separable, NOT calibrated against any real
# campaign.
_DEFAULT_WIN_RATE = 0.40
_DEFAULT_IMPRESSION_RATE = 0.95
_DEFAULT_CLICK_RATE = 0.02
_DEFAULT_CONVERSION_RATE = 0.05
_DEFAULT_CONVERSION_VALUE = 25.0


def _env_float(name: str, default: float) -> float:
    """Read a rate from the environment, falling back on an unparseable value."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class OutcomeSimulatorConfig:
    """Conditional funnel rates for the simulator.

    Each rate is conditional on the preceding stage: ``click_rate`` is the
    probability of a click GIVEN an impression, not given a bid.
    """

    win_rate: float = _DEFAULT_WIN_RATE
    impression_rate: float = _DEFAULT_IMPRESSION_RATE
    click_rate: float = _DEFAULT_CLICK_RATE
    conversion_rate: float = _DEFAULT_CONVERSION_RATE
    conversion_value: float = _DEFAULT_CONVERSION_VALUE

    def __post_init__(self) -> None:
        for field_name in (
            "win_rate",
            "impression_rate",
            "click_rate",
            "conversion_rate",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be in [0.0, 1.0], got {value}"
                )
        if self.conversion_value < 0.0:
            raise ValueError(
                f"conversion_value must be >= 0.0, got {self.conversion_value}"
            )

    @classmethod
    def from_env(cls) -> "OutcomeSimulatorConfig":
        """Build a config from ``OUTCOME_SIMULATOR_*`` environment variables."""
        return cls(
            win_rate=_env_float("OUTCOME_SIMULATOR_WIN_RATE", _DEFAULT_WIN_RATE),
            impression_rate=_env_float(
                "OUTCOME_SIMULATOR_IMPRESSION_RATE", _DEFAULT_IMPRESSION_RATE
            ),
            click_rate=_env_float("OUTCOME_SIMULATOR_CLICK_RATE", _DEFAULT_CLICK_RATE),
            conversion_rate=_env_float(
                "OUTCOME_SIMULATOR_CONVERSION_RATE", _DEFAULT_CONVERSION_RATE
            ),
            conversion_value=_env_float(
                "OUTCOME_SIMULATOR_CONVERSION_VALUE", _DEFAULT_CONVERSION_VALUE
            ),
        )


@dataclass(frozen=True)
class SimulatedOutcome:
    """The synthetic outcome for one bid.

    ``won`` is tri-state-free here — the simulator always decides — but the signals
    derived from it are only the ones that actually occurred, so a lost bid produces
    no signals at all and its outcome event stays ``unresolved``. That is deliberate:
    a real deployment cannot observe an impression for a bid it did not win either.
    """

    won: bool
    impression: bool
    click: bool
    conversion: bool
    conversion_value: Optional[float]


def _uniform(request_id: str, stage: str) -> float:
    """A stable pseudo-uniform draw in [0.0, 1.0) for one request and stage.

    Deterministic by construction: the same (request_id, stage) always yields the
    same value, in this process and any other. Distinct stage labels decorrelate
    the four funnel decisions for a single request, so a request is not simply
    "lucky" or "unlucky" at every stage at once.
    """
    digest = hashlib.sha256(f"{request_id}:{stage}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def simulate_outcome(
    request_id: str, config: Optional[OutcomeSimulatorConfig] = None
) -> SimulatedOutcome:
    """Derive the synthetic outcome for ``request_id``.

    Pure and deterministic — no clock, no randomness, no I/O. Call it twice and get
    the same answer; that is what makes a simulated dataset reproducible.
    """
    cfg = config or OutcomeSimulatorConfig()

    won = _uniform(request_id, "won") < cfg.win_rate
    impression = won and _uniform(request_id, "impression") < cfg.impression_rate
    click = impression and _uniform(request_id, "click") < cfg.click_rate
    conversion = click and _uniform(request_id, "conversion") < cfg.conversion_rate

    return SimulatedOutcome(
        won=won,
        impression=impression,
        click=click,
        conversion=conversion,
        conversion_value=cfg.conversion_value if conversion else None,
    )


def signals_for(
    request_id: str,
    timestamp: float,
    config: Optional[OutcomeSimulatorConfig] = None,
) -> list[DownstreamSignal]:
    """The signals a simulated outcome would have generated, in funnel order.

    Only stages that occurred produce a signal. A lost bid produces none — there is
    nothing to report, and emitting a "no impression" signal would be inventing an
    observation. Every signal is labelled ``provenance="simulated"``.
    """
    outcome = simulate_outcome(request_id, config)
    cfg = config or OutcomeSimulatorConfig()

    signals: list[DownstreamSignal] = []
    if outcome.impression:
        signals.append(
            DownstreamSignal(
                request_id=request_id,
                signal_type=SignalType.IMPRESSION,
                timestamp=timestamp,
                provenance="simulated",
            )
        )
    if outcome.click:
        signals.append(
            DownstreamSignal(
                request_id=request_id,
                signal_type=SignalType.CLICK,
                timestamp=timestamp,
                provenance="simulated",
            )
        )
    if outcome.conversion:
        signals.append(
            DownstreamSignal(
                request_id=request_id,
                signal_type=SignalType.CONVERSION,
                timestamp=timestamp,
                conversion_value=cfg.conversion_value,
                provenance="simulated",
            )
        )
    return signals


def is_enabled() -> bool:
    """Whether the simulator is switched on for this process.

    Read at call time rather than import time so a test can toggle it.
    """
    return os.environ.get(ENABLE_ENV_VAR, "").strip().lower() in _TRUTHY


class OutcomeSimulator:
    """Applies synthetic signals for a bid through a ``SignalAssociator``.

    Holds no state beyond its config and the associator it writes to. Signals go
    through the same ``handle_signal`` path a real downstream signal takes, so the
    enrichment, the re-emit to Kinesis and the ETL join are all exercised for real —
    only the outcome itself is synthetic.
    """

    def __init__(
        self,
        associator,
        config: Optional[OutcomeSimulatorConfig] = None,
    ) -> None:
        self._associator = associator
        self._config = config or OutcomeSimulatorConfig.from_env()

    @property
    def config(self) -> OutcomeSimulatorConfig:
        return self._config

    async def apply(self, event: BidShadingOutcomeEvent) -> int:
        """Emit this bid's simulated signals. Returns the number applied.

        Returns 0 for a simulated loss, which is a real result and not a failure.
        Never raises: a simulator fault must not take down the bid path.
        """
        applied = 0
        try:
            for signal in signals_for(
                event.request_id, event.timestamp, self._config
            ):
                if await self._associator.handle_signal(signal):
                    applied += 1
        except Exception:
            logger.warning(
                "Outcome simulator failed for request_id=%s",
                event.request_id,
                exc_info=True,
            )
        return applied
