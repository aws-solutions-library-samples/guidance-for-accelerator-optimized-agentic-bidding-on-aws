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
    # The container's own account of why it produced no mutations, when the reason
    # is not simply "nothing to change" — read from its response metadata and passed
    # through unaltered. None for a container that mutated, or that declined for
    # ordinary reasons.
    #
    # ``status: no_mutations`` on its own cannot distinguish a model that found no
    # reason to act from one that could not obtain a prediction. The bid shader's
    # abstention is the second, and reporting it as the first is what let a broken
    # inference path read as healthy.
    abstained_reason: str | None = None
    # The container's reported metadata.timing, passed through unaltered so the
    # orchestrator's response carries both its own measurement of this container
    # (latency_ms, which includes the network and client stack) and the
    # container's measurement of itself. None when not reported.
    timing: dict[str, float] | None = None


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


class RejectedMutationModel(BaseModel):
    """A mutation the orchestrator could not apply to its working request."""

    container: str
    intent: int
    path: str
    reason: str


class StageModel(BaseModel):
    """One stage of the orchestrator's sequenced fan-out (shared/artf_stages.py).

    ``latency_ms`` is the stage's wall clock: the slowest container in it plus
    the time to apply its mutations to the working request. Summed across stages
    it is the agent time a consumer should report; a single parallel ceiling
    would understate it by the other stages' time.

    ``applied`` and ``rejected`` describe what the orchestrator did with the
    stage's mutations on its own working copy, which is what the NEXT stage saw.
    The host applies the same list itself; a rejection here is the one it will
    reproduce.
    """

    stage: int
    name: str
    containers: list[str] = []
    latency_ms: float = 0.0
    budget_ms: float = 0.0
    applied: int = 0
    rejected: list[RejectedMutationModel] = []


class Metadata(BaseModel):
    api_version: str = "1.0"
    model_version: str = ""
    containers: list[ContainerInvocationModel] | None = None
    # The stages the fan-out ran, in order. None when the request was bypassed or
    # the server predates staging.
    stages: list[StageModel] | None = None
    # None rather than [] when nothing was contested, so "no conflicts" stays
    # distinguishable from "this orchestrator does not report conflicts" for a
    # client older than this field.
    conflicts: list[ConflictModel] | None = None
    # Why a container that was invoked produced no mutation, when the reason is
    # something other than "nothing to change". None means the container either
    # mutated, or declined for ordinary reasons.
    #
    # An empty mutation list is otherwise ambiguous: a model that saw no reason to
    # act and a model that could not be reached look identical, and the second was
    # being served as the first.
    abstained_reason: str | None = None
    # Where the time went, in milliseconds, keyed by segment name. A container
    # reports parse/queue/mutate/triton; the orchestrator reports
    # auth/parse/fan_out/merge/emit. Absent (None) when the server did not time
    # the request. The segment set is defined in shared/hop_timing.py.
    timing: dict[str, float] | None = None


class RTBResponse(BaseModel):
    """Mirrors the protobuf RTBResponse."""
    id: str
    mutations: list[Mutation] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)


#: The per-request marker that asks the extension point to propose nothing. It
#: lives at `bid_request.ext.artf.bypass` -- the TOP-LEVEL ext, not `ext.prebid`.
#: Prebid Server parses `ext.prebid` into a typed model and drops keys it does not
#: know, so a marker under `ext.prebid.artf` never reached the hook's envelope
#: (verified live: the response said "bypassed" while every container was still
#: consulted). Top-level `ext` is a flexible extension Prebid carries through
#: unchanged. Set by the orchestrator's auction endpoint for the Theater's baseline
#: pass and by nothing else; there is no global switch on purpose.
ARTF_BYPASS_KEY = "bypass"


def is_artf_bypass(bid_request: Any) -> bool:
    """True only when the request carries `ext.artf.bypass: true` exactly.

    `is True`, not truthiness: a string "true", a 1, or any other value is not the
    marker. The cost of a false positive is an auction silently run without ARTF,
    which is the one outcome the per-request design exists to prevent.
    """
    if not isinstance(bid_request, dict):
        return False
    ext = bid_request.get("ext")
    if not isinstance(ext, dict):
        return False
    artf = ext.get("artf")
    if not isinstance(artf, dict):
        return False
    return artf.get(ARTF_BYPASS_KEY) is True


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
