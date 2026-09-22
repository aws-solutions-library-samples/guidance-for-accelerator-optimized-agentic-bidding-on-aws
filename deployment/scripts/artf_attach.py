"""Attach an externally prepared ARTF container to this stack's orchestrator.

The container's *producer* already emits everything that describes it: a
``artf-registry-record.json`` carrying the registry row plus the image
coordinates, and a ``register-artf-container.sh`` that writes the row with
``aws dynamodb put-item``. This module is the consuming half. It does not invent
a record format — it reads theirs and does the work that only this repository can
do:

- resolve the registry table from **this** stack rather than the producer's
  default, which names a table that does not exist here;
- refuse an endpoint that is not in-cluster, because ARTF conformance requires
  the container be reachable only inside the cluster;
- check the container's intents against the Prebid hook's configured ceiling,
  which is an allowlist that narrows every request and will silently never ask
  for an intent it does not list;
- warn when an intent is already claimed by a built-in container, and say which
  one would win;
- render the producer's Kubernetes example against this repository's conventions.

Everything here is pure: no ``boto3``, no ``kubectl``, no network, no environment
reads at import time. The AWS and cluster calls live in
``deployment/attach_artf_container.sh``, which calls into this. That split is
what makes the validation testable without an account or a cluster.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

#: The container's mutate port. Fixed by the ARTF contract and by the producer's
#: own registration module, which declares it non-configurable.
MUTATE_PORT = 8081

#: Constant partition key of the registry table.
REGISTRY_PARTITION = "artf-containers"

#: Intents the Prebid hook can apply at ``processed-auction-request``. BID_SHADE
#: is absent because it addresses the auction RESPONSE, where no ``seatbid``
#: exists at that stage — a mutation for it can only ever be rejected.
REQUEST_SIDE_INTENTS = frozenset({
    "ACTIVATE_SEGMENTS",
    "ACTIVATE_DEALS",
    "SUPPRESS_DEALS",
    "ADJUST_DEAL_FLOOR",
    "ADJUST_DEAL_MARGIN",
    "ADD_METRICS",
    "ADD_CIDS",
})

#: A cluster-internal host: either a fully qualified Service DNS name or a bare
#: Service name. Anything else is refused.
_CLUSTER_DNS = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$", re.IGNORECASE)


class AttachError(ValueError):
    """The attachment cannot proceed, with a reason fit to print."""


@dataclass(frozen=True)
class ExternalContainer:
    """A container described by the producer's record."""

    name: str
    display_name: str
    description: str
    intents: tuple[str, ...]
    endpoint: str
    image: str
    image_digest: str
    namespace: str
    serving_modes: tuple[str, ...] = ()
    gpu_architectures: tuple[str, ...] = ()
    carries: dict = field(default_factory=dict)

    @property
    def supports_gpu(self) -> bool:
        return "gpu" in {m.lower() for m in self.serving_modes}


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def parse_record(raw: str | dict) -> ExternalContainer:
    """Read the producer's ``artf-registry-record.json``.

    Reads the plain ``record`` block rather than ``dynamodb_item``. The producer
    ships both deliberately, noting that the typed form belongs to a table whose
    shape lives in a repository they do not control and could drift — so the
    plain form is the one meant for a consumer to interpret.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise AttachError("The record is not a JSON object.")

    record = data.get("record")
    if not isinstance(record, dict):
        raise AttachError(
            "The record has no 'record' object. Expected the file produced by the "
            "container's own deploy step, which contains 'record', 'image' and "
            "'image_digest'."
        )

    name = str(record.get("name") or "").strip()
    if not name:
        raise AttachError("The record has no container name.")

    endpoint = str(record.get("endpoint") or "").strip()
    if not endpoint:
        raise AttachError(
            f"The record for '{name}' has no endpoint, so there is nowhere to send "
            f"its requests."
        )

    intents = record.get("intents")
    if not isinstance(intents, list) or not intents:
        raise AttachError(
            f"The record for '{name}' claims no intents. A container with no intent "
            f"is never called, so registering it would do nothing."
        )
    normalised = tuple(str(i).strip().upper() for i in intents if str(i).strip())
    if not normalised:
        raise AttachError(f"The record for '{name}' has an empty intent list.")

    image = str(data.get("image") or "").strip()
    if not image:
        raise AttachError(
            f"The record for '{name}' carries no image reference. It lives outside "
            f"the DynamoDB row, in the record file's top-level 'image' key."
        )

    return ExternalContainer(
        name=name,
        display_name=str(record.get("display_name") or "").strip() or name,
        description=str(record.get("description") or ""),
        intents=normalised,
        endpoint=endpoint,
        image=image,
        image_digest=str(data.get("image_digest") or "").strip(),
        namespace=str(data.get("namespace") or "default").strip() or "default",
        serving_modes=tuple(str(m) for m in (data.get("serving_modes") or [])),
        gpu_architectures=tuple(str(a) for a in (data.get("gpu_architectures") or [])),
        carries=data.get("carries") or {},
    )


# ---------------------------------------------------------------------------
# Endpoint enforcement (in-cluster only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointCheck:
    ok: bool
    host: str = ""
    port: int = 0
    namespace: str = ""
    reason: str = ""


def check_endpoint(endpoint: str) -> EndpointCheck:
    """Require a cluster-internal ``http://host:port`` endpoint.

    This is a refusal, not a warning. An ARTF container is expected to be
    reachable only from inside the cluster; an endpoint that leaves it is a
    different deployment model with a different security posture, and quietly
    accepting one would misrepresent what was attached.
    """
    value = (endpoint or "").strip()
    if not value:
        return EndpointCheck(False, reason="The endpoint is empty.")

    if value.startswith("https://"):
        return EndpointCheck(
            False,
            reason=(
                "The endpoint uses https, which means it is not a cluster-internal "
                "Service address. ARTF containers are reached over plain HTTP on "
                "cluster DNS; an external TLS endpoint is a different deployment "
                "model and is not supported here."
            ),
        )
    if not value.startswith("http://"):
        return EndpointCheck(
            False,
            reason=(
                f"The endpoint '{value}' has no http:// scheme, so it cannot be "
                f"called as it stands."
            ),
        )

    rest = value[len("http://"):].rstrip("/")
    if "/" in rest:
        rest, path = rest.split("/", 1)
        if path:
            return EndpointCheck(
                False,
                reason=(
                    f"The endpoint carries a path ('/{path}'). The orchestrator "
                    f"appends /mutate itself, so the record must name host and port "
                    f"only."
                ),
            )

    if ":" not in rest:
        return EndpointCheck(
            False,
            reason=(
                f"The endpoint '{value}' names no port. It must be explicit, and for "
                f"an ARTF container it is {MUTATE_PORT}."
            ),
        )

    host, _, port_text = rest.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return EndpointCheck(False, reason=f"'{port_text}' is not a port number.")

    if not host or not _CLUSTER_DNS.match(host):
        return EndpointCheck(False, reason=f"'{host}' is not a valid Service host name.")

    # An IP address is reachable but is not a Service, so it cannot be resolved
    # or re-scheduled, and it says nothing about what is actually listening.
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
        return EndpointCheck(
            False,
            reason=(
                f"'{host}' is an IP address. The endpoint must name a Kubernetes "
                f"Service so it survives the pod being rescheduled."
            ),
        )

    if host.endswith(".svc.cluster.local"):
        labels = host.split(".")
        if len(labels) < 5:
            return EndpointCheck(
                False,
                reason=(
                    f"'{host}' looks like cluster DNS but is not "
                    f"<service>.<namespace>.svc.cluster.local."
                ),
            )
        namespace = labels[1]
    elif "." in host:
        # service.namespace — the short form Kubernetes also resolves.
        namespace = host.split(".")[1]
    else:
        namespace = ""

    return EndpointCheck(True, host=host, port=port, namespace=namespace)


def service_name(endpoint: str) -> str:
    """The Service name an endpoint refers to, or "" if it is not in-cluster."""
    check = check_endpoint(endpoint)
    if not check.ok:
        return ""
    return check.host.split(".")[0]


def retarget_endpoint(endpoint: str, namespace: str) -> str:
    """Rewrite an endpoint's namespace.

    The producer's endpoint hardcodes whatever namespace was chosen when the
    record was generated, and the record is written before the workload exists,
    so a mismatch would not surface until activation — at which point the
    container reports unreachable for a reason that looks like a network problem.
    """
    check = check_endpoint(endpoint)
    if not check.ok:
        raise AttachError(check.reason)
    return f"http://{service_name(endpoint)}.{namespace}.svc.cluster.local:{check.port}"


# ---------------------------------------------------------------------------
# Table resolution
# ---------------------------------------------------------------------------

def resolve_table(stack_name: str, override: str | None = None) -> str:
    """This stack's registry table.

    The producer's script defaults to ``artf-container-registry``, which is not
    this repository's name for it. Taking their default would write a row into a
    table nothing reads.
    """
    if override:
        return override
    stack = (stack_name or "").strip()
    if not stack:
        raise AttachError(
            "No stack name. The registry table is ${STACK_NAME}-container-registry, "
            "so the stack name is required to find it."
        )
    return f"{stack}-container-registry"


# ---------------------------------------------------------------------------
# DynamoDB item
# ---------------------------------------------------------------------------

def build_item(
    container: ExternalContainer,
    *,
    endpoint: str | None = None,
    priority: int = 0,
    updated_at: str = "",
    updated_by: str = "",
) -> dict:
    """The typed DynamoDB item, written inactive.

    ``active`` is always False. Installing a container must not change what the
    bid path does; activating it is a separate decision, and for a container that
    sets a price it is a decision about money.

    ``priority`` defaults to 0, which means an attached container does not
    displace a built-in by being attached — the precedence rule falls back to
    registry order, which is the behaviour that predates precedence existing.

    ``intents`` is a list of strings (``L`` of ``S``), never a string set. A
    ``SS`` would sort and deduplicate, losing the order the producer wrote.
    """
    item = {
        "registry": {"S": REGISTRY_PARTITION},
        "name": {"S": container.name},
        "display_name": {"S": container.display_name},
        "description": {"S": container.description},
        "intents": {"L": [{"S": i} for i in container.intents]},
        "endpoint": {"S": endpoint or container.endpoint},
        "active": {"BOOL": False},
        "priority": {"N": str(int(priority))},
    }
    if updated_at:
        item["updated_at"] = {"S": updated_at}
    if updated_by:
        item["updated_by"] = {"S": updated_by}
    return item


def item_to_record(item: dict) -> dict:
    """Read a typed item back to plain values. Inverse of ``build_item``."""

    def scalar(key, default=""):
        cell = item.get(key) or {}
        if "S" in cell:
            return cell["S"]
        if "N" in cell:
            return int(cell["N"])
        if "BOOL" in cell:
            return cell["BOOL"]
        return default

    return {
        "registry": scalar("registry"),
        "name": scalar("name"),
        "display_name": scalar("display_name"),
        "description": scalar("description"),
        "intents": [c.get("S", "") for c in (item.get("intents") or {}).get("L", [])],
        "endpoint": scalar("endpoint"),
        "active": scalar("active", False),
        "priority": scalar("priority", 0),
    }


# ---------------------------------------------------------------------------
# Prebid ceiling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CeilingCheck:
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


def check_prebid_ceiling(
    intents: tuple[str, ...] | list[str],
    configured: tuple[str, ...] | list[str] | None,
) -> CeilingCheck:
    """Compare a container's intents against the hook's configured allowlist.

    The hook intersects its configured list with each request's
    ``applicable_intents``, so the configured list is a ceiling: an intent absent
    from it is never asked for, and the container is simply never called on the
    Prebid path. That failure is silent — the container looks healthy and does
    nothing.

    ``unknown`` names intents outside the request-side set. Reported rather than
    treated as missing, because the configured list is not validated against the
    enum and a new intent name can legitimately appear there before any code
    knows it.
    """
    wanted = [str(i).strip().upper() for i in intents if str(i).strip()]
    if configured is None:
        # Not knowing is different from being absent. The caller decides whether
        # an unreadable config is fatal; inventing a ceiling here would be worse.
        return CeilingCheck(covered=(), missing=(), unknown=tuple(
            i for i in wanted if i not in REQUEST_SIDE_INTENTS
        ))

    allowed = {str(i).strip().upper() for i in configured}
    covered = tuple(i for i in wanted if i in allowed)
    missing = tuple(i for i in wanted if i not in allowed)
    unknown = tuple(i for i in wanted if i not in REQUEST_SIDE_INTENTS)
    return CeilingCheck(covered=covered, missing=missing, unknown=unknown)


def ceiling_remedy(missing: tuple[str, ...] | list[str], config_key: str) -> str:
    """What to do about a missing intent, as instructions rather than a warning."""
    if not missing:
        return ""
    lines = [
        "These intents are not in the Prebid hook's configured list, so the hook "
        "will never ask for them and this container will never be called on the "
        "Prebid path:",
        "",
    ]
    lines += [f"  - {i}" for i in missing]
    lines += [
        "",
        "Add them under hooks.artf-orchestrator.intents in the config object:",
        "",
        f"  aws s3 cp s3://<bucket>/{config_key} /tmp/prebid-config.yaml",
        "  # add the intents above to the 'intents:' list",
        f"  aws s3 cp /tmp/prebid-config.yaml s3://<bucket>/{config_key}",
        "  kubectl rollout restart deployment/prebid-server",
        "",
        "The config is read at container start, so the restart is required — "
        "editing the object alone changes nothing.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intent contention
# ---------------------------------------------------------------------------

def describe_contention(
    container: ExternalContainer,
    existing: list[dict],
    *,
    priority: int = 0,
) -> list[str]:
    """Warnings about intents already claimed by an active container.

    Mirrors the orchestrator's rule: higher priority wins, and equal priorities
    fall back to registry order, where store-defined containers follow
    code-defined ones. So an attached container at the default priority of 0 wins
    against a built-in — which is worth saying out loud at attach time rather
    than leaving to be discovered from a bid response.
    """
    warnings: list[str] = []
    for intent in container.intents:
        rivals = [
            e for e in existing
            if e.get("active") and intent in {str(i).upper() for i in (e.get("intents") or [])}
            and e.get("name") != container.name
        ]
        if not rivals:
            continue
        for rival in rivals:
            rival_priority = int(rival.get("priority") or 0)
            rival_name = rival.get("name")
            if priority > rival_priority:
                outcome = f"'{container.name}' would win on priority ({priority} > {rival_priority})"
            elif priority < rival_priority:
                outcome = (
                    f"'{rival_name}' would win on priority ({rival_priority} > {priority}); "
                    f"'{container.name}' would compute a mutation that is then discarded"
                )
            elif rival.get("source") == "code":
                outcome = (
                    f"priorities are equal, so registry order decides and store-defined "
                    f"containers follow built-in ones — '{container.name}' would win, and "
                    f"the built-in '{rival_name}' would compute a mutation that is then "
                    f"discarded. Set a negative priority to keep the built-in's value"
                )
            else:
                outcome = (
                    f"priorities are equal, so registry order decides (alphabetical among "
                    f"store containers)"
                )
            warnings.append(f"{intent} is also claimed by '{rival_name}': {outcome}.")
    return warnings


# ---------------------------------------------------------------------------
# Manifest rendering
# ---------------------------------------------------------------------------

def split_image(image: str) -> tuple[str, str]:
    """Split ``registry/repo:tag`` into ``(registry/repo, tag)``.

    Splits on the last colon only when it follows the last slash, so a registry
    that carries a port is not mistaken for a tag.
    """
    value = (image or "").strip()
    if not value:
        raise AttachError("No image reference to split.")
    if "@" in value:
        repo, _, digest = value.partition("@")
        return repo, digest
    slash = value.rfind("/")
    colon = value.rfind(":")
    if colon > slash:
        return value[:colon], value[colon + 1:]
    return value, "latest"


def render_manifest(
    template: str,
    *,
    image: str,
    namespace: str,
    pull_policy: str = "Always",
) -> str:
    """Substitute the producer's placeholders and apply this repo's conventions.

    The producer ships the manifest as an example with ``__REGISTRY__`` and
    ``__TAG__`` placeholders and does not apply it themselves. Rendering theirs
    rather than shipping a competing manifest keeps one source of truth for the
    shape — including the constraints recorded in its comments, which a
    re-implementation would lose.

    ``imagePullPolicy`` is rewritten to match the rest of this cluster's ARTF
    containers. Theirs is ``IfNotPresent``, which is reasonable for an
    immutably-tagged image but inconsistent here.
    """
    if "__REGISTRY__" not in template and "__TAG__" not in template:
        raise AttachError(
            "The manifest template has no __REGISTRY__ or __TAG__ placeholder. "
            "Expected the producer's artf-container.example.yaml."
        )
    repo, tag = split_image(image)
    rendered = template.replace("__REGISTRY__", repo).replace("__TAG__", tag)
    rendered = re.sub(
        r"imagePullPolicy:\s*\S+", f"imagePullPolicy: {pull_policy}", rendered
    )
    if namespace:
        rendered = _ensure_namespace(rendered, namespace)
    return rendered


def _ensure_namespace(manifest: str, namespace: str) -> str:
    """Set ``metadata.namespace`` on every document that lacks one.

    Only touches uncommented lines: the producer's GPU shape is a commented block
    in the same file, and rewriting it would produce a manifest that silently
    disagrees with the variant actually selected.
    """
    out: list[str] = []
    for line in manifest.splitlines():
        out.append(line)
        stripped = line.strip()
        if stripped == "metadata:" and not stripped.startswith("#"):
            indent = len(line) - len(line.lstrip())
            out.append(" " * (indent + 2) + f"namespace: {namespace}")
    return "\n".join(out) + ("\n" if manifest.endswith("\n") else "")


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------

#: Compute capabilities the producer's GPU image is built for.
GPU_SUPPORTED_ARCHES = frozenset({"sm_80", "sm_86", "sm_87", "sm_90"})

#: What this cluster's GPU nodegroup actually is, by instance family.
INSTANCE_FAMILY_ARCH = {
    "g5": "sm_86",   # A10G — supported
    "g6": "sm_89",   # L4 — NOT supported by HierarchicalKV
    "p4d": "sm_80",  # A100
    "p5": "sm_90",   # H100
}


def gpu_guard(
    container: ExternalContainer,
    *,
    instance_types: list[str] | tuple[str, ...],
    image_exists: bool,
) -> tuple[bool, str]:
    """Whether the GPU variant may be rendered, and why not if it may not.

    Three conditions, all required. The label check is the one that matters most:
    a manifest can be rendered for a GPU shape that the published image does not
    actually support, and the failure would then appear at runtime as a startup
    refusal rather than here as a refusal to attach.
    """
    if not image_exists:
        return False, (
            f"The GPU image for '{container.name}' was not found. Only the CPU image "
            f"appears to be published — check the producer's build before requesting "
            f"--variant gpu."
        )
    if not container.supports_gpu:
        return False, (
            f"The record for '{container.name}' advertises serving modes "
            f"{list(container.serving_modes) or '[]'}, which does not include 'gpu'. "
            f"The image's own agent-manifest label is the authority on this, and it "
            f"says this build cannot serve on a GPU."
        )

    arches = {a.lower() for a in container.gpu_architectures} or set(GPU_SUPPORTED_ARCHES)
    cluster_arches = set()
    for instance in instance_types:
        family = str(instance).split(".")[0].lower()
        arch = INSTANCE_FAMILY_ARCH.get(family)
        if arch:
            cluster_arches.add(arch)

    if not cluster_arches:
        return False, (
            f"Could not determine the GPU architecture of {list(instance_types)}, so "
            f"whether this image can run here is unknown. Refusing rather than "
            f"guessing."
        )

    usable = arches & cluster_arches
    if not usable:
        return False, (
            f"This cluster's GPU nodes are {sorted(cluster_arches)} and the image "
            f"builds for {sorted(arches)}. They do not overlap, so the container "
            f"would refuse to start."
        )
    return True, f"GPU variant supported: cluster {sorted(cluster_arches)} matches {sorted(usable)}."


# ---------------------------------------------------------------------------
# Probe payload
# ---------------------------------------------------------------------------

def probe_payload(container: ExternalContainer, request_id: str = "attach-probe") -> dict:
    """A minimal ARTF request for verifying the container actually answers.

    Deliberately probes ``/mutate`` and not ``/health/ready``: the shared server
    answers readiness with an unconditional ok even when its data has not loaded,
    so a health probe here would be a check that cannot fail. A ``/mutate`` call
    shows whether the container returns mutations and what model version it
    reports.
    """
    return {
        "id": request_id,
        "lifecycle": 1,
        "tmax": 100,
        "applicable_intents": list(container.intents),
        "bid_request": {
            "id": request_id,
            "imp": [
                {
                    "id": "imp-1",
                    "bidfloor": 1.0,
                    "pmp": {"deals": [{"id": "deal-1", "bidfloor": 1.0}]},
                }
            ],
            "site": {"page": "https://example.com/attach-probe"},
        },
    }


def summarise_probe(response: dict | None, *, error: str = "") -> str:
    """Describe a probe result as what it is, including 'not deployed yet'."""
    if error:
        return (
            f"Could not reach the container: {error}. That is expected if it has not "
            f"been deployed yet — the registry row is written inactive, so nothing "
            f"is affected until you activate it."
        )
    if not isinstance(response, dict):
        return "The container answered, but the response was not a JSON object."

    mutations = response.get("mutations")
    count = len(mutations) if isinstance(mutations, list) else 0
    version = ((response.get("metadata") or {}).get("model_version")) or "(none reported)"
    if count == 0:
        return (
            f"The container answered with 0 mutations, reporting model_version "
            f"{version}. It is reachable and running; whether it should have produced "
            f"a mutation for this probe depends on its own logic."
        )
    return (
        f"The container answered with {count} mutation(s), reporting model_version "
        f"{version}."
    )
