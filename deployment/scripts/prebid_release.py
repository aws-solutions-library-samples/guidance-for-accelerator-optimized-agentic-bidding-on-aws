"""Pure helpers for the Prebid stack deployment.

Everything here is a function of its arguments: no AWS calls, no filesystem, no
clock. That is deliberate -- a deployment script is otherwise almost impossible to
test, and these three pieces are where the mistakes actually happen:

  1. archive layout      a wrong path assumption fails later and less legibly
  2. stack status         conflating "wait" with "delete and retry" is destructive
  3. cost disclosure      a figure that silently omits its dominant term is worse
                          than no figure

Used by deploy_prebid.sh via `python3 -c` / `-m`, and unit-tested directly.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --------------------------------------------------------------------- version

#: The pinned upstream release. Verified to exist; its assets contain NO pre-built
#: container image, which is why a from-source build is mandatory rather than an
#: optimisation.
PINNED_VERSION = "v1.4.0"

#: True for the releases in scope. Modelled as data rather than an assumption in the
#: build logic, so a future release that publishes an image changes this and nothing
#: else.
RELEASE_CARRIES_PREBUILT_IMAGE = False


# ------------------------------------------------------------- archive layout


def normalise_archive_root(names: Iterable[str]) -> str:
    """The directory to treat as the checkout root, given an archive's entry names.

    A versioned archive nests everything under one top-level directory named for the
    version; an unversioned one places files at the root. Detected rather than
    assumed, because a wrong guess fails during placement or build with an error that
    points at the wrong thing.

    Returns "." for a flat archive, or the single top-level directory name.
    """
    # Materialised FIRST, because this function is documented as taking any Iterable
    # and the caller passes a generator over a file. It reads the names twice -- once
    # for the top-level set, once to decide whether that entry is a directory -- and
    # against a generator the second pass saw nothing, so every nested archive was
    # misreported as flat. The resulting failure surfaced two steps later as "the
    # release layout has changed", which is the wrong thing to go and check.
    entries = [_strip_leading_dot_slash(name) for name in names]
    entries = [entry for entry in entries if entry not in ("", ".")]

    tops = {entry.split("/", 1)[0] for entry in entries}
    tops.discard("")
    tops.discard(".")

    if len(tops) != 1:
        return "."

    only = next(iter(tops))
    # A single top-level entry that is a FILE is a flat archive of one file, not a
    # nested layout. Distinguished by whether anything lives beneath it.
    nested = any(
        posixpath.dirname(entry).split("/", 1)[0] == only for entry in entries if "/" in entry
    )
    return only if nested else "."


def _strip_leading_dot_slash(name: str) -> str:
    """Normalise one archive entry name.

    Only a leading ``./`` is removed. ``str.lstrip("./")`` would also eat the leading
    dot of ``.github/workflows``, turning a hidden directory into a different name.
    """
    text = str(name).strip()
    while text.startswith("./"):
        text = text[2:]
    return text


# --------------------------------------------------------------- stack status

#: Statuses from which CloudFormation cannot move forward: an update is rejected, so
#: the remnant must be deleted before a create can succeed.
TERMINAL_FAILURE_STATUSES = frozenset(
    {"ROLLBACK_COMPLETE", "ROLLBACK_FAILED", "CREATE_FAILED", "DELETE_FAILED"}
)

#: Statuses that mean a deployment is mid-flight. Distinct from terminal failure:
#: deleting a stack that is merely in progress is destructive.
IN_PROGRESS_SUFFIX = "_IN_PROGRESS"

#: Statuses that mean the stack is healthy and reconcilable.
HEALTHY_STATUSES = frozenset({"CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"})


def classify_stack_status(status: Optional[str]) -> str:
    """One of: absent, healthy, failed_terminal, in_progress, unknown.

    ``unknown`` is a real answer, not a failure: acting on a status we do not
    recognise is how a stack gets deleted by accident.
    """
    if status is None or status == "":
        return "absent"
    if status in TERMINAL_FAILURE_STATUSES:
        return "failed_terminal"
    if status.endswith(IN_PROGRESS_SUFFIX):
        return "in_progress"
    if status in HEALTHY_STATUSES:
        return "healthy"
    return "unknown"


def is_no_op_error(message: str) -> bool:
    """Whether a CloudFormation error means "nothing to change".

    A reconciliation that finds nothing to do is a SUCCESS, not a failure.
    """
    return "No updates are to be performed" in (message or "")


# ------------------------------------------------------------------ idle cost


@dataclass(frozen=True)
class CostComponent:
    """One line of the idle-cost disclosure."""

    name: str
    monthly_usd: Optional[float]
    basis: str
    #: Where the number comes from. Present so a reader can check it.
    source: str


@dataclass(frozen=True)
class CostDisclosure:
    """What the operator is told before anything is provisioned."""

    components: tuple[CostComponent, ...] = field(default_factory=tuple)
    currency: str = "USD"

    @property
    def known_total(self) -> float:
        return sum(c.monthly_usd for c in self.components if c.monthly_usd is not None)

    @property
    def unknown_components(self) -> tuple[CostComponent, ...]:
        return tuple(c for c in self.components if c.monthly_usd is None)

    @property
    def is_complete(self) -> bool:
        """Whether every component has a figure."""
        return not self.unknown_components


def build_cost_disclosure(additional_node_monthly_usd: Optional[float]) -> CostDisclosure:
    """The marginal monthly cost of adding this stack to an existing cluster.

    ``additional_node_monthly_usd`` is the cost of any node capacity the Prebid pods
    force. It is REQUIRED to be supplied and is NOT defaulted, because on a cluster
    without headroom it is the dominant term -- and a total that silently omitted it
    would understate the real figure by an order of magnitude. Pass None when it is
    genuinely not yet known; the disclosure then reports itself incomplete rather than
    printing a total that looks authoritative.

    The upstream guidance's published figure of roughly $241.50/month describes its
    own ECS Fargate deployment. It does NOT describe this one, which runs as pods on an
    existing EKS cluster, so it is deliberately not used here.
    """
    return CostDisclosure(
        components=(
            CostComponent(
                name="Additional EKS node capacity",
                monthly_usd=additional_node_monthly_usd,
                basis="only if the Prebid pods do not fit existing headroom",
                source="depends on instance type and region; supplied by the caller",
            ),
            CostComponent(
                name="Secrets Manager secret",
                monthly_usd=0.40,
                basis="one secret",
                source="AWS Secrets Manager list price, per secret per month",
            ),
            CostComponent(
                name="Cognito domain, resource servers, app client",
                monthly_usd=0.0,
                basis="no standing charge for these resources",
                source="AWS Cognito pricing: no charge for a user pool domain",
            ),
            CostComponent(
                name="ECR image storage",
                monthly_usd=None,
                basis="per GB-month; depends on the built image size",
                source="AWS ECR list price; image size not known until built",
            ),
            CostComponent(
                name="CodeBuild",
                monthly_usd=0.0,
                basis="per build-minute, not a standing cost",
                source="charged per build; zero when idle",
            ),
            CostComponent(
                name="API Gateway + Lambda (demand endpoint)",
                monthly_usd=0.0,
                basis="per request; no standing charge",
                source="charged per request; zero when idle",
            ),
        )
    )


def format_cost_disclosure(disclosure: CostDisclosure) -> str:
    """Human-readable disclosure.

    States what is known, what is not, and where the numbers come from. Never prints a
    bare total: a figure without its provenance cannot be checked, and this one drives
    a spending decision.
    """
    lines = ["Idle monthly cost of adding the Prebid stack to your existing cluster:", ""]
    for component in disclosure.components:
        if component.monthly_usd is None:
            amount = "not yet known"
        elif component.monthly_usd == 0.0:
            amount = "no standing charge"
        else:
            amount = f"${component.monthly_usd:.2f}"
        lines.append(f"  {component.name}: {amount}")
        lines.append(f"      {component.basis}")
        lines.append(f"      source: {component.source}")
    lines.append("")
    lines.append(f"  Known standing total: ${disclosure.known_total:.2f}/month")

    if not disclosure.is_complete:
        lines.append("")
        lines.append("  INCOMPLETE. These components have no figure yet:")
        for component in disclosure.unknown_components:
            lines.append(f"    - {component.name} ({component.basis})")
        lines.append("  The total above therefore UNDERSTATES the real cost.")

    lines.append("")
    lines.append(
        "  Note: the upstream guidance publishes roughly $241.50/month for its own"
    )
    lines.append(
        "  ECS Fargate deployment. That is not this deployment and is not included."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- capacity

#: What one Prebid pod requests, from eks/prebid-server-deployment.yaml. One container
#: only: the config fetch is performed by the image's own entrypoint, so there is no
#: init container to account for. Kept here so the preflight capacity check and the
#: manifest cannot drift apart silently.
POD_CPU_REQUEST_M = 500
POD_MEMORY_REQUEST_MI = 1024


def parse_cpu_millis(value: str) -> int:
    """Kubernetes CPU quantity to millicores.

    ``"2"`` is 2000m; ``"1470m"`` is 1470m. Anything else raises, rather than being
    guessed at -- a silently wrong capacity check is worse than a failed one.
    """
    text = str(value).strip()
    if not text:
        raise ValueError("empty CPU quantity")
    if text.endswith("m"):
        return int(float(text[:-1]))
    return int(float(text) * 1000)


_MEMORY_SUFFIXES = (
    ("Ki", 1.0 / 1024.0),
    ("Mi", 1.0),
    ("Gi", 1024.0),
    ("Ti", 1024.0 * 1024.0),
)


def parse_memory_mib(value: str) -> int:
    """Kubernetes memory quantity to MiB.

    Handles the binary suffixes Kubernetes actually reports for node allocatable, and
    treats a bare number as bytes.
    """
    text = str(value).strip()
    if not text:
        raise ValueError("empty memory quantity")
    for suffix, multiplier in _MEMORY_SUFFIXES:
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(float(text) / (1024.0 * 1024.0))


def node_fits(
    nodes: Iterable[dict],
    need_cpu_m: int = POD_CPU_REQUEST_M,
    need_memory_mi: int = POD_MEMORY_REQUEST_MI,
) -> Optional[bool]:
    """Whether any single node is large enough to hold the pod.

    Takes the ``items`` of ``kubectl get nodes -o json``.

    Returns True, False, or **None for "could not tell"** -- an unreadable or empty
    node list is not evidence of a fit, and reporting it as one would let the
    deployment claim capacity it never measured.

    This tests ALLOCATABLE, which is an upper bound: it does not subtract what other
    pods have already requested. So True means "not obviously impossible", not
    "guaranteed to schedule". The caller must say so.
    """
    seen = False
    for node in nodes:
        allocatable = (node or {}).get("status", {}).get("allocatable") or {}
        cpu_raw = allocatable.get("cpu")
        mem_raw = allocatable.get("memory")
        if cpu_raw is None or mem_raw is None:
            continue
        try:
            cpu = parse_cpu_millis(cpu_raw)
            memory = parse_memory_mib(mem_raw)
        except (TypeError, ValueError):
            continue
        seen = True
        if cpu >= need_cpu_m and memory >= need_memory_mi:
            return True
    return False if seen else None
