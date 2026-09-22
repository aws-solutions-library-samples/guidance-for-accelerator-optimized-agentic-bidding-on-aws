"""ARTF message types — Python equivalents of the protobuf definitions.

These mirror ``agenticrtbframework.proto`` from the IAB Tech Lab ARTF v1.0
spec so containers can speak the same language without requiring protobuf
compilation.  The orchestrator and each container import these directly.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Lifecycle(IntEnum):
    UNSPECIFIED = 0
    PUBLISHER_BID_REQUEST = 1
    DSP_BID_RESPONSE = 2


class Intent(IntEnum):
    UNSPECIFIED = 0
    ACTIVATE_SEGMENTS = 1
    ACTIVATE_DEALS = 2
    SUPPRESS_DEALS = 3
    ADJUST_DEAL_FLOOR = 4
    ADJUST_DEAL_MARGIN = 5
    BID_SHADE = 6
    ADD_METRICS = 7
    ADD_CIDS = 8


class Operation(IntEnum):
    UNSPECIFIED = 0
    ADD = 1
    REMOVE = 2
    REPLACE = 3


class MarginCalculationType(IntEnum):
    """Margin.CalculationType from the ARTF proto."""
    CPM = 0       # Absolute margin adjustment
    PERCENT = 1   # Relative margin adjustment (percentage)


# ---------------------------------------------------------------------------
# Payload types
# ---------------------------------------------------------------------------

class IDsPayload(BaseModel):
    id: list[str] = Field(default_factory=list)


class Margin(BaseModel):
    """Mirrors the ARTF ``Margin`` message (value + calculation_type)."""
    value: float | None = None
    calculation_type: int = MarginCalculationType.CPM


class AdjustDealPayload(BaseModel):
    bidfloor: float | None = None
    margin: Margin | None = None


class AdjustBidPayload(BaseModel):
    price: float


class Metric(BaseModel):
    type: str
    value: float
    vendor: str | None = None


class AddMetricsPayload(BaseModel):
    metric: list[Metric] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

class Mutation(BaseModel):
    intent: int  # Intent enum value
    op: int  # Operation enum value
    path: str
    ids: IDsPayload | None = None
    adjust_deal: AdjustDealPayload | None = None
    adjust_bid: AdjustBidPayload | None = None
    add_metrics: AddMetricsPayload | None = None


# ---------------------------------------------------------------------------
# Originator
# ---------------------------------------------------------------------------

class Originator(BaseModel):
    type: str = "TYPE_UNSPECIFIED"
    id: str = ""


# ---------------------------------------------------------------------------
# RTBRequest / RTBResponse
# ---------------------------------------------------------------------------

class RTBRequest(BaseModel):
    """Mirrors the protobuf RTBRequest."""
    # ``model_params`` is exposed as a property backed by ``ext`` (below),
    # which lives in the ``model_`` namespace pydantic reserves — opt out so
    # the accessor is allowed.
    model_config = ConfigDict(protected_namespaces=())

    id: str
    lifecycle: str | int = "LIFECYCLE_PUBLISHER_BID_REQUEST"
    tmax: int = 100
    bid_request: dict[str, Any] = Field(default_factory=dict)
    bid_response: dict[str, Any] | None = None
    originator: Originator | None = None
    applicable_intents: list[str | int] = Field(default_factory=list)
    # ARTF ``ext`` object (proto field 99, ``extensions 500 to max``) — the
    # spec-sanctioned channel for nonstandard signaling. Demo model overrides
    # from the frontend sliders travel as ``ext.model_params``.
    ext: dict[str, Any] | None = Field(default=None)

    @property
    def model_params(self) -> dict[str, Any] | None:
        """Non-standard model overrides carried in ``ext.model_params``."""
        if self.ext:
            return self.ext.get("model_params")
        return None


class ContainerInvocationModel(BaseModel):
    """Per-container invocation record for demo flow visualization.

    Captures which container was invoked during orchestration, its completion
    status, observed latency, and any mutations it contributed. Surfaced via
    ``Metadata.containers`` so UI clients can render the ARTF flow graph.

    ``status`` is deliberately an unconstrained ``str`` rather than a Literal so
    the vocabulary can widen without breaking older clients. The values the
    orchestrator produces are defined in
    ``orchestrator/container_registry.py``: ok, no_mutations, unreachable,
    error, timeout, disabled, skipped.

    ``display_name`` carries the container's human label so a consumer does not
    have to resolve it from a build-time lookup table. That matters for
    store-defined containers, whose names are not known when the frontend is
    built — without it a user's own container cannot be labelled at all. Empty
    for callers that do not set it, so existing readers are unaffected.

    ``superseded`` counts this container's mutations that lost a contest for a
    ``(path, intent)`` to a higher-priority container. ``mutations`` still lists
    them: the container did compute them, and a reader needs to be able to tell
    "ran and was overridden" from "ran and produced nothing". See
    ``orchestrator/container_registry.resolve_conflicts``.
    """
    name: str
    status: str
    latency_ms: float
    mutations: list[Mutation] = []
    model_version: str = ""
    display_name: str = ""
    superseded: int = 0


class ConflictModel(BaseModel):
    """One (path, intent) claimed by more than one container.

    Two containers may legitimately claim the same intent — the fan-out calls
    both and both return mutations. But a consumer applies mutations in order,
    so for a given path only the last one it applies has any effect. Reporting
    the contest is what turns "two mutations, one of which quietly did nothing"
    into something a reader can act on.

    ``losers`` names the containers whose mutation for this key was not the one
    returned. Their mutation is still present on their own
    ``ContainerInvocationModel.mutations``, because it was really computed.
    """

    path: str
    intent: int
    winner: str
    losers: list[str] = []


class Metadata(BaseModel):
    api_version: str = "1.0"
    model_version: str = ""
    containers: list[ContainerInvocationModel] | None = None
    # None rather than [] when nothing was contested, so "no conflicts" stays
    # distinguishable from "this orchestrator does not report conflicts" for a
    # client older than this field.
    conflicts: list[ConflictModel] | None = None


class RTBResponse(BaseModel):
    """Mirrors the protobuf RTBResponse."""
    id: str
    mutations: list[Mutation] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)


def intent_applicable(intent: Intent, applicable: list[str | int]) -> bool:
    """Check if *intent* is allowed by the applicable_intents list.

    An empty list means all intents are applicable (per ARTF spec).
    """
    if not applicable:
        return True
    for a in applicable:
        if isinstance(a, int) and a == intent:
            return True
        if isinstance(a, str):
            # Accept both "ACTIVATE_SEGMENTS" and "1"
            if a == intent.name or a == str(intent.value):
                return True
    return False
