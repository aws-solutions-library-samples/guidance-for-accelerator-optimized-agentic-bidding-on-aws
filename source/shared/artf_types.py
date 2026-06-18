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
    """
    name: str
    status: str
    latency_ms: float
    mutations: list[Mutation] = []


class Metadata(BaseModel):
    api_version: str = "1.0"
    model_version: str = ""
    containers: list[ContainerInvocationModel] | None = None


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
