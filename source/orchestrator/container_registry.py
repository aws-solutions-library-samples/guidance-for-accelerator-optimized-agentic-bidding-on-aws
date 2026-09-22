"""ARTF container registry — code-defined containers plus store-defined ones.

The orchestrator's container list used to be a single hardcoded Python literal.
This module keeps that literal as the *code half* of the registry and adds a
*store half* read from DynamoDB, so a container can be described and switched on
or off at runtime without rebuilding the orchestrator image or redeploying the
stack.

Two halves, deliberately asymmetric:

- **Code-defined** containers (the six ARTF recommenders) are always present,
  always active, and never configurable. There is no code path by which a stored
  record can rename, re-target, re-intent or deactivate one of them — see
  ``merge_registry``'s collision rule. The live bid path is not UI-mutable.
- **Store-defined** containers are described entirely by their record and carry
  an ``active`` flag the UI can toggle.

The pure functions at the top (``merge_registry``, ``select_active``,
``derive_status``) hold all the logic and perform no I/O, so the routing rules
are testable without AWS or a cluster. ``ContainerRegistryStore`` at the bottom
is the only part that talks to DynamoDB, and it is modelled on
``shared/parameter_cache.py``: TTL-cached, never raises on read, and degrades to
the last known good value.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.artf_types import Mutation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
#
# Before this module, a container that could not be reached at all reported
# status "ok" with zero mutations: the connect error was caught inside
# _call_container / _call_mcp and degraded to an empty mutation list, so the
# caller's success branch ran. "failed" was effectively unreachable code. That
# made a missing container indistinguishable from a healthy one that chose not
# to mutate, which matters most for exactly the case this feature introduces —
# a template container a user has not implemented or deployed yet.
#
# These seven values are the real outcomes, kept as plain strings because
# ContainerInvocationModel.status is an unconstrained str and widening it must
# stay wire-compatible.

STATUS_OK = "ok"                      # reached, returned one or more mutations
STATUS_NO_MUTATIONS = "no_mutations"   # reached, deliberately returned none
STATUS_UNREACHABLE = "unreachable"     # nothing answered on the wire
STATUS_ERROR = "error"                 # answered, but the response was unusable
STATUS_TIMEOUT = "timeout"             # exceeded the request's tmax budget
STATUS_DISABLED = "disabled"           # registered but inactive — never called
STATUS_SKIPPED = "skipped"             # intents did not match applicable_intents

#: Statuses that mean the container ran and its answer was usable.
RAN_STATUSES = frozenset({STATUS_OK, STATUS_NO_MUTATIONS})

#: Statuses that mean the call went wrong. Used by the load test's error
#: counter — without unreachable/error in this set the counter would be blind
#: to precisely the failures it exists to catch.
ERROR_STATUSES = frozenset({STATUS_UNREACHABLE, STATUS_ERROR, STATUS_TIMEOUT, "failed"})

#: Statuses that mean the container was intentionally not invoked.
NOT_CALLED_STATUSES = frozenset({STATUS_DISABLED, STATUS_SKIPPED})

SOURCE_CODE = "code"
SOURCE_STORE = "store"

#: Constant partition key. One Query on this value returns every record, so no
#: Scan is ever needed — Scan is not granted to the orchestrator's role, and
#: this keeps it that way.
REGISTRY_PARTITION = "artf-containers"

DEFAULT_TTL_SECONDS = 30.0

#: Reused from parameter_cache.py's contract: report sustained failure once
#: rather than on every read.
SUSTAINED_ERROR_THRESHOLD = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContainerCallOutcome:
    """What actually happened on one container call.

    This type exists because the information needed to tell "nothing answered"
    from "answered with nothing" used to be destroyed inside the transport
    helpers before the caller could see it. ``derive_status`` is a pure function
    of this observation.

    ``reached`` means something answered on the wire — not that the answer was
    usable. A container that returns an unparseable 200 was reached and also
    produced an ``error``.
    """

    reached: bool
    mutations: list[Mutation] = field(default_factory=list)
    model_version: str = ""
    error: str | None = None


@dataclass(frozen=True)
class RegistryEntry:
    """One container in the effective registry.

    ``configurable`` is not a preference — it is False for every code-defined
    container and there is no way to set it True for one. The activation
    endpoint refuses to write to a non-configurable entry, so FR-9 is enforced
    at both the merge and the boundary.
    """

    name: str
    display_name: str
    description: str
    intents: frozenset[str]
    endpoint: str
    grpc: str
    active: bool
    source: str
    configurable: bool
    priority: int = 0

    def as_container_dict(self) -> dict:
        """The legacy container-dict shape the transport helpers expect.

        ``_call_container_timed`` and ``loadtest.py`` both take a plain dict with
        ``name``/``intents``/``grpc``/``mcp`` keys, and three existing tests
        monkeypatch that boundary with hand-built dicts. Converting here rather
        than changing those signatures keeps the live bid path's call contract
        byte-identical and leaves those tests valid.
        """
        return {
            "name": self.name,
            "intents": set(self.intents),
            "grpc": self.grpc,
            "mcp": self.endpoint,
            "display_name": self.display_name,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Pure function: merge
# ---------------------------------------------------------------------------

def _normalise_intents(raw) -> frozenset[str]:
    """Coerce an intents value to a set of upper-case names.

    Unknown intent names are kept rather than dropped: this module does not own
    the Intent enum's future, and silently discarding an intent a user typed
    would make their container mysteriously never get called.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    out = set()
    for item in raw:
        if isinstance(item, bytes):
            item = item.decode("utf-8", "replace")
        if not isinstance(item, str):
            item = str(item)
        item = item.strip().upper()
        if item:
            out.add(item)
    return frozenset(out)


def _coerce_priority(raw) -> int:
    """Coerce a stored priority to an int, defaulting to 0.

    DynamoDB numbers arrive as ``Decimal``, and a hand-written record can carry a
    string or nothing at all. An unparseable value becomes 0 rather than raising:
    this runs on the bid path, and a typo in one registry record must not stop
    every container being called.
    """
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Registry record has an unparseable priority %r; treating it as 0.", raw)
        return 0


def _entry_from_code(container: dict) -> RegistryEntry:
    name = container["name"]
    return RegistryEntry(
        name=name,
        display_name=container.get("display_name") or name,
        description=container.get("description") or "",
        intents=_normalise_intents(container.get("intents")),
        endpoint=container.get("mcp") or "",
        grpc=container.get("grpc") or "",
        # Code-defined containers are always active. Their availability is a
        # deployment fact reported by the probe, not a stored preference.
        active=True,
        source=SOURCE_CODE,
        configurable=False,
        # Pinned, with no code path to change it — same reasoning as `active`
        # and `configurable`. A store record cannot promote or demote a built-in;
        # to outrank one, an attached container must be given a priority above 0
        # deliberately.
        priority=0,
    )


def _entry_from_record(record: dict) -> RegistryEntry:
    name = str(record.get("name") or "").strip()
    endpoint = str(record.get("endpoint") or "").strip()
    display_name = str(record.get("display_name") or "").strip() or name
    # Falls back to empty, never to invented prose.
    description = str(record.get("description") or "")
    grpc = str(record.get("grpc") or "").strip()
    if not grpc and endpoint:
        # Mirror the code half's convention: the gRPC target is the same
        # host:port with the scheme stripped.
        grpc = endpoint.replace("http://", "").replace("https://", "").rstrip("/")
    return RegistryEntry(
        name=name,
        display_name=display_name,
        description=description,
        intents=_normalise_intents(record.get("intents")),
        endpoint=endpoint,
        grpc=grpc,
        active=bool(record.get("active", False)),
        source=SOURCE_STORE,
        configurable=True,
        priority=_coerce_priority(record.get("priority")),
    )


def merge_registry(
    code_containers: list[dict],
    store_records: list[dict] | None = None,
) -> tuple[list[RegistryEntry], list[str]]:
    """Merge the code-defined and store-defined halves of the registry.

    Returns ``(entries, warnings)``. Warnings are returned rather than only
    logged so ``GET /v1/containers`` can show the operator why a record they
    wrote is not taking effect.

    Rules:

    - BR-1/BR-2: every code container appears, in declaration order, first.
      Keeping them first and ordered preserves today's response ordering, which
      is also the mutation attribution order.
    - BR-3: store records follow, sorted by name, so the order is deterministic
      across replicas and across cache refreshes.
    - BR-4: a store record colliding with a code container is ignored. This is
      the structural guarantee that the six built-ins are immutable from the UI.
    - BR-5: a record with no name or no endpoint is ignored — it cannot be
      routed to, and inventing an endpoint would be fabrication.
    - BR-6/BR-7: intents normalised; display name falls back to the internal
      name; description falls back to empty.
    """
    warnings: list[str] = []
    entries: list[RegistryEntry] = [_entry_from_code(c) for c in code_containers]
    code_names = {e.name for e in entries}

    store_entries: list[RegistryEntry] = []
    seen_store_names: set[str] = set()

    for record in store_records or []:
        entry = _entry_from_record(record)

        if not entry.name:
            warnings.append("Ignored a registry record with no name.")
            continue
        if entry.name in code_names:
            warnings.append(
                f"Ignored registry record '{entry.name}': that name belongs to a "
                f"built-in container, which cannot be reconfigured."
            )
            continue
        if entry.name in seen_store_names:
            warnings.append(f"Ignored duplicate registry record '{entry.name}'.")
            continue
        if not entry.endpoint:
            warnings.append(
                f"Ignored registry record '{entry.name}': no endpoint, so there is "
                f"nowhere to send its requests."
            )
            continue

        seen_store_names.add(entry.name)
        store_entries.append(entry)

    store_entries.sort(key=lambda e: e.name)
    entries.extend(store_entries)
    return entries, warnings


def shared_intents(entries: list[RegistryEntry]) -> dict[str, list[str]]:
    """Intents claimed by more than one container, mapped to the claimants.

    Two containers on one intent is allowed — the fanout calls both and merges
    both sets of mutations, which is legitimate for comparing implementations.
    But the merge order then decides which value survives downstream, so the UI
    warns. This function is what it warns from.
    """
    by_intent: dict[str, list[str]] = {}
    for entry in entries:
        for intent in sorted(entry.intents):
            by_intent.setdefault(intent, []).append(entry.name)
    return {i: names for i, names in by_intent.items() if len(names) > 1}


# ---------------------------------------------------------------------------
# Pure function: select
# ---------------------------------------------------------------------------

def select_active(
    entries: list[RegistryEntry],
    applicable_intents: list | None = None,
) -> tuple[list[RegistryEntry], list[tuple[RegistryEntry, str]]]:
    """Partition the registry into containers to call and containers not to.

    Returns ``(to_call, not_called)`` where each ``not_called`` element carries
    the reason as a status string.

    The inactive check runs **before** the intent check, so a deactivated
    container is never reported as ``skipped``. "You turned it off" and "your
    intents did not match" are different facts and must not share a label.

    An empty or absent ``applicable_intents`` means all intents apply, which is
    the existing ARTF semantics and the behaviour the previous
    ``_filter_containers`` had.
    """
    requested: set[str] = set()
    for item in applicable_intents or []:
        if isinstance(item, str):
            requested.add(item.strip().upper())
        elif item is not None:
            requested.add(str(item).strip().upper())

    to_call: list[RegistryEntry] = []
    not_called: list[tuple[RegistryEntry, str]] = []

    for entry in entries:
        if not entry.active:
            not_called.append((entry, STATUS_DISABLED))
            continue
        if requested and not (entry.intents & requested):
            not_called.append((entry, STATUS_SKIPPED))
            continue
        to_call.append(entry)

    return to_call, not_called


# ---------------------------------------------------------------------------
# Pure function: status
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationClaim:
    """One mutation, plus who produced it and where they sit in the registry.

    ``order`` is the producing container's index in registry order and ``index``
    is the mutation's position in the flattened list. Both are needed because the
    tie-break has to reproduce the existing behaviour exactly — see
    ``resolve_conflicts``.
    """

    container: str
    priority: int
    order: int
    index: int
    mutation: Mutation


@dataclass(frozen=True)
class ConflictRecord:
    """One ``(path, intent)`` contested by more than one container."""

    path: str
    intent: int
    winner: str
    losers: tuple[str, ...]


def build_claims(
    invocations,
    priority_by_name: dict[str, int] | None = None,
) -> list[MutationClaim]:
    """Flatten per-container mutations into claims, preserving registry order.

    *invocations* is the list ``_fan_out`` returns, which is **already in registry
    order** — so a container's index in it is its registry position and no second
    registry lookup is needed to establish ordering. Only the priority has to be
    supplied, and that comes from the TTL-cached registry, so this adds no I/O to
    the bid path.
    """
    priorities = priority_by_name or {}
    claims: list[MutationClaim] = []
    index = 0
    for order, inv in enumerate(invocations):
        for mutation in getattr(inv, "mutations", None) or []:
            claims.append(
                MutationClaim(
                    container=inv.name,
                    priority=priorities.get(inv.name, 0),
                    order=order,
                    index=index,
                    mutation=mutation,
                )
            )
            index += 1
    return claims


def resolve_conflicts(
    claims: list[MutationClaim],
) -> tuple[list[Mutation], list[ConflictRecord], dict[str, int]]:
    """Pick one mutation per ``(path, intent)``.

    Returns ``(survivors, conflicts, superseded_by_container)``. Survivors are in
    **input order**, so the response's mutation ordering stays stable and still
    reflects registry order rather than the order winners happened to be decided.

    Why this exists
    ---------------
    Two containers may claim the same intent, which the fan-out allows on purpose.
    But a consumer applies the mutation list in order, and for a deal floor the
    applier does ``updatedDeals.set(dealIndex, …)`` — so only the **last** mutation
    applied to a given path has any effect. Before this function, the earlier one
    was still returned, still reported ``ok``, and silently did nothing.

    The tie-break direction
    -----------------------
    The winner is ``max`` by ``(priority, order, index)``. The ``max`` matters:
    with every priority equal it reduces to "highest order, then highest index",
    which is the **last** mutation in the flattened list — exactly the
    last-write-wins outcome a consumer produces today. Using ``min`` would look
    like it was "preserving registry order" while inverting which container's
    value is honoured on every existing deployment.

    That is the whole backward-compatibility guarantee: set no priorities and the
    surviving set is byte-identical to what was returned before this function
    existed. Attaching a container therefore changes nothing until someone gives
    it a priority deliberately.
    """
    if not claims:
        return [], [], {}

    grouped: dict[tuple[str, int], list[MutationClaim]] = {}
    for claim in claims:
        grouped.setdefault((claim.mutation.path, claim.mutation.intent), []).append(claim)

    winners: set[int] = set()
    conflicts: list[ConflictRecord] = []
    superseded: dict[str, int] = {}

    for (path, intent), group in grouped.items():
        winner = max(group, key=lambda c: (c.priority, c.order, c.index))
        winners.add(winner.index)

        if len(group) == 1:
            # Uncontested. A conflict record here would be noise.
            continue

        losers = [c for c in group if c.index != winner.index]
        for loser in losers:
            superseded[loser.container] = superseded.get(loser.container, 0) + 1

        # Losing container names, de-duplicated but order-preserving: one
        # container can lose twice on the same key and should be named once.
        loser_names: list[str] = []
        for loser in losers:
            if loser.container not in loser_names:
                loser_names.append(loser.container)

        conflicts.append(
            ConflictRecord(
                path=path,
                intent=intent,
                winner=winner.container,
                losers=tuple(loser_names),
            )
        )

    survivors = [c.mutation for c in claims if c.index in winners]
    return survivors, conflicts, superseded


def derive_status(outcome: ContainerCallOutcome) -> str:
    """Map a real call observation to a status.

    Total over its input. ``timeout`` is produced by the caller's wait_for
    wrapper and ``disabled``/``skipped`` by ``select_active``, so this function
    needs no catch-all branch.
    """
    if not outcome.reached:
        return STATUS_UNREACHABLE
    if outcome.error is not None:
        return STATUS_ERROR
    if outcome.mutations:
        return STATUS_OK
    return STATUS_NO_MUTATIONS


# ---------------------------------------------------------------------------
# Store errors
# ---------------------------------------------------------------------------

class RegistryStoreUnavailable(RuntimeError):
    """The registry table is not configured, or could not be written to."""


class RegistryRecordNotFound(LookupError):
    """No store record exists with that name."""


def _is_conditional_check_failure(exc: Exception) -> bool:
    """Whether *exc* is DynamoDB rejecting our attribute_exists condition.

    Checked two ways because botocore surfaces this differently depending on how
    the call was made: a resource-level ``update_item`` raises a dynamically
    generated class named ``ConditionalCheckFailedException``, while a
    client-level call raises ``ClientError`` carrying the code in
    ``response["Error"]["Code"]``. Matching only one of the two would silently
    reclassify a vanished record as a store outage.
    """
    if "ConditionalCheckFailedException" in type(exc).__name__:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict) and error.get("Code") == "ConditionalCheckFailedException":
            return True
    return False


# ---------------------------------------------------------------------------
# DynamoDB-backed store
# ---------------------------------------------------------------------------

class ContainerRegistryStore:
    """TTL-cached reader and conditional writer for registry records.

    Read contract, copied from ``ParameterCache`` because the bid path has the
    same requirement: ``get_records`` **never raises** and never blocks on a
    retry. On failure it keeps serving the last known good records; if it has
    never succeeded it serves an empty list, which merges to the six
    code-defined containers. A registry outage therefore degrades routing to
    exactly today's behaviour rather than failing a bid.

    The TTL is what keeps this off the per-request path: at most one DynamoDB
    Query per TTL window per replica, regardless of request rate.
    """

    def __init__(
        self,
        table_name: str | None = None,
        region: str | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._table_name = table_name if table_name is not None else os.environ.get("CONTAINER_REGISTRY_TABLE", "")
        self._region = region or os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        if ttl_seconds is None:
            try:
                ttl_seconds = float(os.environ.get("CONTAINER_REGISTRY_TTL", DEFAULT_TTL_SECONDS))
            except ValueError:
                ttl_seconds = DEFAULT_TTL_SECONDS
        self._ttl_seconds = max(0.0, ttl_seconds)

        self._records: list[dict] = []
        self._last_refresh: float | None = None
        self._consecutive_errors = 0
        self._alarm_emitted = False
        self._last_error: str | None = None
        self._table = None

    # -- introspection ---------------------------------------------------

    @property
    def configured(self) -> bool:
        """Whether a table name is set. Unset means the feature is off, and is
        reported as such rather than as an empty registry."""
        return bool(self._table_name)

    @property
    def table_name(self) -> str | None:
        return self._table_name or None

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    def snapshot(self) -> dict:
        """Evidence about the registry's own state, for the API to surface.

        ``tableReachable`` is None when no read has been attempted — a real
        "not known yet", not a claim either way.
        """
        age = None
        if self._last_refresh is not None:
            age = round(max(0.0, time.monotonic() - self._last_refresh), 1)

        if not self.configured:
            reachable = False
        elif self._last_refresh is None and self._consecutive_errors == 0:
            reachable = None
        else:
            reachable = self._consecutive_errors == 0

        return {
            "tableName": self.table_name,
            "tableConfigured": self.configured,
            "tableReachable": reachable,
            "ttlSeconds": self._ttl_seconds,
            "ageSeconds": age,
            "consecutiveErrors": self._consecutive_errors,
            "error": self._last_error,
        }

    # -- read ------------------------------------------------------------

    def get_records(self) -> list[dict]:
        """Return the cached registry records. Never raises."""
        self._refresh_if_stale()
        return list(self._records)

    def invalidate(self) -> None:
        """Force the next read to hit DynamoDB.

        Called after a successful write so the replica that handled the toggle
        reflects it immediately. Other replicas converge within the TTL, which
        is why the endpoint reports an effect window instead of implying the
        change is global at once.
        """
        self._last_refresh = None

    def _refresh_if_stale(self) -> None:
        if not self.configured:
            return

        now = time.monotonic()
        if self._last_refresh is not None and (now - self._last_refresh) < self._ttl_seconds:
            return

        try:
            from boto3.dynamodb.conditions import Key

            table = self._get_table()
            response = table.query(
                KeyConditionExpression=Key("registry").eq(REGISTRY_PARTITION),
                ConsistentRead=False,
            )
            items = response.get("Items", []) or []
            # Single constant partition, so pagination is theoretical at this
            # size — handled anyway rather than silently truncating.
            while response.get("LastEvaluatedKey"):
                response = table.query(
                    KeyConditionExpression=Key("registry").eq(REGISTRY_PARTITION),
                    ConsistentRead=False,
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []) or [])

            self._records = [dict(item) for item in items]
            self._last_refresh = now
            self._consecutive_errors = 0
            self._alarm_emitted = False
            self._last_error = None

        except Exception as exc:
            self._consecutive_errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "ContainerRegistryStore refresh failed (consecutive_errors=%d): %s",
                self._consecutive_errors,
                exc,
            )
            if self._consecutive_errors >= SUSTAINED_ERROR_THRESHOLD and not self._alarm_emitted:
                self._emit_sustained_error_metric()
                self._alarm_emitted = True

    # -- write -----------------------------------------------------------

    def set_active(self, name: str, active: bool, updated_by: str = "unknown") -> dict:
        """Flip a store record's ``active`` flag.

        Uses UpdateItem with a SET expression rather than PutItem so a toggle
        cannot clobber a display name or description written concurrently by
        another replica. The ``attribute_exists`` condition means a vanished
        record surfaces as a real conflict instead of being silently recreated
        with invented values.

        Every attribute name is aliased because ``name`` is a DynamoDB reserved
        word.

        Raises ``RegistryStoreUnavailable`` or ``RegistryRecordNotFound``;
        never returns a fabricated success.
        """
        if not self.configured:
            raise RegistryStoreUnavailable(
                "CONTAINER_REGISTRY_TABLE is not set, so container activation state "
                "cannot be stored."
            )

        timestamp = _now_iso()
        try:
            table = self._get_table()
            result = table.update_item(
                Key={"registry": REGISTRY_PARTITION, "name": name},
                UpdateExpression="SET #active = :active, #updated_at = :ts, #updated_by = :who",
                ConditionExpression="attribute_exists(#name)",
                ExpressionAttributeNames={
                    "#name": "name",
                    "#active": "active",
                    "#updated_at": "updated_at",
                    "#updated_by": "updated_by",
                },
                ExpressionAttributeValues={
                    ":active": bool(active),
                    ":ts": timestamp,
                    ":who": updated_by or "unknown",
                },
                ReturnValues="ALL_NEW",
            )
        except Exception as exc:
            if _is_conditional_check_failure(exc):
                raise RegistryRecordNotFound(
                    f"No registry record named '{name}'."
                ) from exc
            raise RegistryStoreUnavailable(
                f"Could not update registry record '{name}': {type(exc).__name__}: {exc}"
            ) from exc

        # The writing replica reflects its own change at once.
        self.invalidate()
        return dict(result.get("Attributes") or {})

    # -- internals -------------------------------------------------------

    def _get_table(self):
        if self._table is None:
            import boto3

            dynamodb = boto3.resource("dynamodb", region_name=self._region)
            self._table = dynamodb.Table(self._table_name)
        return self._table

    def _emit_sustained_error_metric(self) -> None:
        """Report sustained read failure once. Must never block bidding."""
        try:
            import boto3

            cloudwatch = boto3.client("cloudwatch", region_name=self._region)
            cloudwatch.put_metric_data(
                Namespace="ARTF/ContainerRegistry",
                MetricData=[{
                    "MetricName": "SustainedReadErrors",
                    "Value": 1.0,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "TableName", "Value": self._table_name}],
                }],
            )
            logger.error(
                "ContainerRegistryStore: %d consecutive read failures — metric emitted",
                self._consecutive_errors,
            )
        except Exception as exc:
            logger.error("Failed to emit registry error metric: %s", exc)
