"""The order in which the orchestrator consults the ARTF containers.

Five stages, each run to completion before the next is called, and each stage's
mutations applied to a working copy of the bid request (shared/artf_applier.py)
before the next stage sees it:

    0. content  ADD_CIDS
    1. enrich   ACTIVATE_SEGMENTS, ADD_METRICS
    2. deals    ACTIVATE_DEALS, SUPPRESS_DEALS
    3. yield    ADJUST_DEAL_FLOOR, ADJUST_DEAL_MARGIN
    4. price    BID_SHADE

The order is the data dependency between the intents. ADD_CIDS runs first so
the content ids it adds (``cid:`` metrics on ``imp.metric``) are on the request
when the Audience Activator runs, which lets segment activation take content
into account. No shipped container claims ADD_CIDS; the stage runs only when a
registry container claiming it (the ARTF template by default) is active and the
request's applicable_intents allow it, like every other stage. The Deal Scorer scores
deals against the user's segments, so it runs after the Audience Activator has
written them. The yield containers price the deals on the impression, so they
run after the Deal Scorer has activated or suppressed any. The Bid Pricer clamps
a shaded price to the impression floor, so it runs after the yield containers
have moved it. Within a stage the containers are independent and run in
parallel.

A container's stage is derived from the intents it registers: the earliest stage
any of its intents belongs to. A container registering no recognised intent runs
in stage 1 (enrich), as it did before stage 0 existed; stage 0 is kept to
ADD_CIDS.

Pure. No I/O.
"""

from __future__ import annotations

from typing import Iterable, Sequence, TypeVar

from shared.artf_types import Intent

STAGE_CONTENT = 0
STAGE_ENRICH = 1
STAGE_DEALS = 2
STAGE_YIELD = 3
STAGE_PRICE = 4

STAGE_NAMES: dict[int, str] = {
    STAGE_CONTENT: "content",
    STAGE_ENRICH: "enrich",
    STAGE_DEALS: "deals",
    STAGE_YIELD: "yield",
    STAGE_PRICE: "price",
}

STAGE_ORDER: tuple[int, ...] = (
    STAGE_CONTENT, STAGE_ENRICH, STAGE_DEALS, STAGE_YIELD, STAGE_PRICE,
)

INTENT_STAGE: dict[Intent, int] = {
    Intent.ADD_CIDS: STAGE_CONTENT,
    Intent.ACTIVATE_SEGMENTS: STAGE_ENRICH,
    Intent.ADD_METRICS: STAGE_ENRICH,
    Intent.ACTIVATE_DEALS: STAGE_DEALS,
    Intent.SUPPRESS_DEALS: STAGE_DEALS,
    Intent.ADJUST_DEAL_FLOOR: STAGE_YIELD,
    Intent.ADJUST_DEAL_MARGIN: STAGE_YIELD,
    Intent.BID_SHADE: STAGE_PRICE,
}


def _to_intent(value: object) -> Intent | None:
    """An Intent from a wire name ("ACTIVATE_DEALS"), a numeric string, or an int."""
    if isinstance(value, Intent):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        try:
            return Intent(value)
        except ValueError:
            return None
    if isinstance(value, str):
        name = value.strip().upper()
        if name in Intent.__members__:
            return Intent[name]
        if name.isdigit():
            try:
                return Intent(int(name))
            except ValueError:
                return None
    return None


def stage_of_intent(intent: object) -> int | None:
    """The stage one intent belongs to, or None for an unrecognised intent."""
    resolved = _to_intent(intent)
    if resolved is None:
        return None
    return INTENT_STAGE.get(resolved)


def stage_of(intents: Iterable[object]) -> int:
    """The stage a container with these intents runs in: the earliest of them.

    Earliest, not latest, so a container that registers intents across stages
    runs as soon as its first dependency is satisfied and nothing downstream
    waits on it longer than it must. ``STAGE_ENRICH`` when no intent is
    recognised, since an unknown intent can have no dependency this table knows
    about.
    """
    stages = [s for s in (stage_of_intent(i) for i in intents) if s is not None]
    return min(stages) if stages else STAGE_ENRICH


T = TypeVar("T")


def group_by_stage(entries: Sequence[T], intents_of) -> list[tuple[int, list[T]]]:
    """Partition ``entries`` into ``(stage, [entries])`` in stage order.

    ``intents_of(entry)`` returns the entry's intents. Order within a stage is the
    input order, which for the orchestrator is registry order, so the mutation
    attribution order inside a stage is unchanged from before staging existed.
    Stages with no entries are omitted.
    """
    buckets: dict[int, list[T]] = {s: [] for s in STAGE_ORDER}
    for entry in entries:
        buckets[stage_of(intents_of(entry))].append(entry)
    return [(s, buckets[s]) for s in STAGE_ORDER if buckets[s]]
