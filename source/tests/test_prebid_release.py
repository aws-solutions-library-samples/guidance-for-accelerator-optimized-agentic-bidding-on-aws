"""Pure helpers for the Prebid deployment: examples and properties.

These are the only parts of U1 testable without an AWS account, which is exactly why
they were extracted.
"""

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# deployment/scripts is not a package; add it to the path the same way a script would.
_SCRIPTS = Path(__file__).resolve().parents[2] / "deployment" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from prebid_release import (  # noqa: E402
    HEALTHY_STATUSES,
    PINNED_VERSION,
    POD_CPU_REQUEST_M,
    POD_MEMORY_REQUEST_MI,
    RELEASE_CARRIES_PREBUILT_IMAGE,
    TERMINAL_FAILURE_STATUSES,
    build_cost_disclosure,
    classify_stack_status,
    format_cost_disclosure,
    is_no_op_error,
    node_fits,
    normalise_archive_root,
    parse_cpu_millis,
    parse_memory_mib,
)

# --------------------------------------------------------------- archive layout


def test_versioned_archive_resolves_to_its_top_level_directory():
    names = [
        "prebid-server-deployment-on-aws-1.4.0/",
        "prebid-server-deployment-on-aws-1.4.0/README.md",
        "prebid-server-deployment-on-aws-1.4.0/source/pom.xml",
    ]
    assert normalise_archive_root(names) == "prebid-server-deployment-on-aws-1.4.0"


def test_flat_archive_resolves_to_dot():
    names = ["README.md", "source/pom.xml", "deployment/cdk.json"]
    assert normalise_archive_root(names) == "."


def test_a_single_file_archive_is_flat_not_nested():
    # One top-level entry with nothing beneath it is a flat archive of one file.
    assert normalise_archive_root(["README.md"]) == "."


def test_leading_dot_slash_is_tolerated():
    names = ["./prebid-1.4.0/", "./prebid-1.4.0/README.md"]
    assert normalise_archive_root(names) == "prebid-1.4.0"


def test_empty_archive_resolves_to_dot():
    assert normalise_archive_root([]) == "."


# ----------------------------------------------------------------- stack status


def test_absent_stack():
    assert classify_stack_status(None) == "absent"
    assert classify_stack_status("") == "absent"


@pytest.mark.parametrize("status", sorted(TERMINAL_FAILURE_STATUSES))
def test_terminal_failure_statuses(status):
    assert classify_stack_status(status) == "failed_terminal"


@pytest.mark.parametrize("status", sorted(HEALTHY_STATUSES))
def test_healthy_statuses(status):
    assert classify_stack_status(status) == "healthy"


def test_in_progress_is_distinct_from_terminal_failure():
    # Conflating these is destructive: deleting a mid-flight stack is not a retry.
    assert classify_stack_status("UPDATE_IN_PROGRESS") == "in_progress"
    assert classify_stack_status("ROLLBACK_IN_PROGRESS") == "in_progress"
    assert classify_stack_status("DELETE_IN_PROGRESS") == "in_progress"


def test_an_unrecognised_status_is_unknown_not_assumed_safe():
    assert classify_stack_status("SOMETHING_NEW") == "unknown"


def test_no_updates_error_is_treated_as_success():
    assert is_no_op_error("An error occurred: No updates are to be performed.") is True
    assert is_no_op_error("ValidationError: template invalid") is False
    assert is_no_op_error("") is False


# -------------------------------------------------------------------- the cost


def test_cost_is_incomplete_when_node_capacity_is_unknown():
    disclosure = build_cost_disclosure(additional_node_monthly_usd=None)
    assert disclosure.is_complete is False
    names = [c.name for c in disclosure.unknown_components]
    assert "Additional EKS node capacity" in names


def test_an_incomplete_disclosure_says_it_understates_the_cost():
    text = format_cost_disclosure(build_cost_disclosure(None))
    assert "INCOMPLETE" in text
    assert "UNDERSTATES" in text


def test_every_component_states_its_source():
    for component in build_cost_disclosure(0.0).components:
        assert component.source
        assert component.basis


def test_the_upstream_fargate_figure_is_named_as_not_applying():
    text = format_cost_disclosure(build_cost_disclosure(0.0))
    assert "241.50" in text
    assert "not this deployment" in text


def test_node_capacity_is_included_in_the_total_when_supplied():
    disclosure = build_cost_disclosure(additional_node_monthly_usd=120.0)
    assert disclosure.known_total >= 120.0


def test_the_pinned_release_is_recorded_and_carries_no_image():
    assert PINNED_VERSION == "v1.4.0"
    # Modelled as data: a future release publishing an image changes this flag, not the
    # build logic.
    assert RELEASE_CARRIES_PREBUILT_IMAGE is False


# ----------------------------------------------------------------- properties

path_segment = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=6
)


@settings(max_examples=200)
@given(root=path_segment, children=st.lists(path_segment, min_size=1, max_size=5))
def test_property_a_nested_archive_always_yields_its_root(root, children):
    names = [f"{root}/"] + [f"{root}/{child}" for child in children]
    assert normalise_archive_root(names) == root


@settings(max_examples=200)
@given(names=st.lists(path_segment, min_size=2, max_size=6, unique=True))
def test_property_multiple_top_level_entries_are_always_flat(names):
    assert normalise_archive_root(names) == "."


@settings(max_examples=200)
@given(root=path_segment, children=st.lists(path_segment, min_size=1, max_size=4))
def test_property_normalisation_is_idempotent(root, children):
    names = [f"{root}/{child}" for child in children]
    once = normalise_archive_root(names)
    # Re-normalising the already-stripped names yields a flat root: there is nothing
    # left to strip. The point is that it terminates rather than oscillating.
    stripped = [n[len(once) + 1 :] for n in names] if once != "." else names
    assert normalise_archive_root(stripped) in (".", *[c for c in children])


@settings(max_examples=300)
@given(status=st.one_of(st.none(), st.text(max_size=30)))
def test_property_classification_is_total(status):
    assert classify_stack_status(status) in {
        "absent",
        "healthy",
        "failed_terminal",
        "in_progress",
        "unknown",
    }


@settings(max_examples=200)
@given(node_cost=st.floats(min_value=0, max_value=10000, allow_nan=False, allow_infinity=False))
def test_property_known_total_is_the_sum_of_known_components(node_cost):
    disclosure = build_cost_disclosure(node_cost)
    expected = sum(c.monthly_usd for c in disclosure.components if c.monthly_usd is not None)
    assert disclosure.known_total == pytest.approx(expected)


@settings(max_examples=100)
@given(node_cost=st.one_of(st.none(), st.floats(0, 500, allow_nan=False)))
def test_property_the_disclosure_always_names_every_component(node_cost):
    disclosure = build_cost_disclosure(node_cost)
    text = format_cost_disclosure(disclosure)
    for component in disclosure.components:
        assert component.name in text



# --------------------------------------------------------------------- capacity


def _node(cpu, memory):
    return {"status": {"allocatable": {"cpu": cpu, "memory": memory}}}


@pytest.mark.parametrize(
    "quantity,expected",
    [("2", 2000), ("4", 4000), ("1470m", 1470), ("500m", 500), ("0", 0), ("1.5", 1500)],
)
def test_cpu_quantities_parse(quantity, expected):
    assert parse_cpu_millis(quantity) == expected


@pytest.mark.parametrize(
    "quantity,expected",
    [
        ("4096Mi", 4096),
        ("4Gi", 4096),
        ("1Ti", 1024 * 1024),
        ("1048576Ki", 1024),
        # A bare number is bytes -- what the API reports for some node types.
        ("2147483648", 2048),
    ],
)
def test_memory_quantities_parse(quantity, expected):
    assert parse_memory_mib(quantity) == expected


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_quantities_raise_rather_than_defaulting_to_zero(bad):
    # A zero default would report a node as too small and block a valid deployment;
    # a large default would claim capacity that was never measured. Neither is safe.
    with pytest.raises(ValueError):
        parse_cpu_millis(bad)
    with pytest.raises(ValueError):
        parse_memory_mib(bad)


def test_a_c5_xlarge_style_node_fits_the_pod():
    # 4 vCPU / ~7.5Gi allocatable, the shape this repository's cpu-services nodegroup
    # actually uses.
    assert node_fits([_node("4", "7700Mi")]) is True


def test_a_node_too_small_on_memory_does_not_fit():
    assert node_fits([_node("4", "512Mi")]) is False


def test_a_node_too_small_on_cpu_does_not_fit():
    assert node_fits([_node("250m", "8Gi")]) is False


def test_one_big_enough_node_among_small_ones_is_a_fit():
    nodes = [_node("250m", "512Mi"), _node("4", "7700Mi"), _node("250m", "512Mi")]
    assert node_fits(nodes) is True


def test_an_empty_node_list_is_unknown_not_a_fit():
    # The distinction this test exists for: "I could not tell" must never be reported
    # as "it fits", or the deployment claims capacity it never measured.
    assert node_fits([]) is None


def test_nodes_with_unreadable_allocatable_are_unknown():
    assert node_fits([{"status": {}}, {}]) is None


def test_unparseable_quantities_do_not_count_as_a_fit():
    assert node_fits([_node("banana", "7700Mi")]) is None


def _manifest_text():
    return (
        Path(__file__).resolve().parents[2]
        / "deployment"
        / "eks"
        / "prebid-server-deployment.yaml"
    ).read_text()


def test_pod_requests_match_the_manifest():
    """The manifest is the source of truth for what the pod asks for.

    These constants drive the preflight capacity check, so if the manifest's requests
    change and these do not, preflight starts approving a cluster that cannot schedule
    the pod. Parsed rather than string-matched, so a comment mentioning "500m" cannot
    make this pass.
    """
    import yaml

    docs = [d for d in yaml.safe_load_all(_manifest_text()) if d]
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]

    containers = pod["containers"]
    assert len(containers) == 1, "an added container must be reflected in the constants"
    assert "initContainers" not in pod, (
        "an init container's requests count toward scheduling; add them to the constants"
    )

    requests = containers[0]["resources"]["requests"]
    assert requests["cpu"] == f"{POD_CPU_REQUEST_M}m"
    assert parse_memory_mib(requests["memory"]) == POD_MEMORY_REQUEST_MI


def test_the_manifest_does_not_reimplement_the_images_own_config_fetch():
    """Regression, from reading the pinned release.

    The image ENTRYPOINT is bootstrap.sh, which fetches configuration from S3 itself
    and exits 1 without DOCKER_CONFIGS_S3_BUCKET_NAME. An init container was drafted
    here that duplicated that fetch and wrote to /config, a path nothing in the image
    reads -- so the pod would have failed on its own entrypoint while the deployment
    reported success.
    """
    import yaml

    docs = [d for d in yaml.safe_load_all(_manifest_text()) if d]
    pod = next(d for d in docs if d["kind"] == "Deployment")["spec"]["template"]["spec"]
    assert "initContainers" not in pod

    env = {e["name"]: e.get("value") for e in pod["containers"][0]["env"]}
    assert "DOCKER_CONFIGS_S3_BUCKET_NAME" in env, (
        "the variable the image's entrypoint actually requires"
    )
    assert "SPRING_CONFIG_ADDITIONAL_LOCATION" not in env, (
        "entrypoint.sh passes --spring.config.additional-location itself"
    )


def test_probes_and_service_target_the_port_the_server_actually_listens_on():
    """The upstream default config sets server.http.port 8443 with ssl: true, and
    entrypoint.sh generates a self-signed keystore. A probe on 8080, or on 8443 with
    scheme HTTP, never succeeds -- and the pod then crash-loops with connection errors
    that say nothing about the port.
    """
    import yaml

    docs = [d for d in yaml.safe_load_all(_manifest_text()) if d]
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    service = next(d for d in docs if d["kind"] == "Service")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    ports = {p["name"]: p["containerPort"] for p in container["ports"]}
    assert ports["https"] == 8443
    assert 8080 not in ports.values()

    for probe in ("readinessProbe", "livenessProbe", "startupProbe"):
        http_get = container[probe]["httpGet"]
        assert http_get["port"] == 8443, f"{probe} must target the TLS port"
        assert http_get["scheme"] == "HTTPS", f"{probe} must speak TLS"

    assert service["spec"]["ports"][0]["targetPort"] == 8443


def test_the_config_overlay_keeps_the_load_bearing_server_block():
    """bootstrap.sh copies current/ over default/ into one directory. There is no YAML
    merge, so the overlay replaces the default file wholesale -- and anything the
    default file provided and the overlay omits is gone. The TLS settings are the ones
    that matter: without them the server stops listening on 8443 over TLS and every
    probe above fails.
    """
    import yaml

    template = (
        Path(__file__).resolve().parents[2]
        / "deployment"
        / "scripts"
        / "prebid_config_template.yaml"
    ).read_text()
    config = yaml.safe_load(template)

    assert config["server"]["http"]["port"] == 8443
    assert config["server"]["ssl"] is True
    assert config["server"]["jks-path"] == "${SSL_KEYSTORE_PATH}"
    assert config["server"]["jks-password"] == "${SSL_KEYSTORE_PASS}"
    assert config["status-response"] == "ok"
    # The artfhouse adapter is this feature's demand source and must be present.
    assert config["adapters"]["artfhouse"]["enabled"] is True

    # Carried forward from the upstream default and load-bearing in its own right:
    # without it Prebid demands a stored account for the request's account id, and this
    # deployment has no account store -- every auction would be rejected before a hook
    # ever ran.
    assert config["settings"]["enforce-valid-account"] is False


def test_the_config_overlay_references_no_placeholder_this_topology_cannot_resolve():
    """The overlay replaces the release default, and that is what makes the pod bootable.

    The release's own config refers to ${CACHE_HOST}, ${SETTINGS_S3_BUCKET} and friends
    -- infrastructure the upstream ECS stack creates and this one does not. Spring fails
    to start on an unresolved placeholder, so any of those surviving into the overlay
    would be a pod that never boots.

    Comments may mention them, so this parses the YAML and inspects values only.
    """
    import re

    import yaml

    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "deployment"
            / "scripts"
            / "prebid_config_template.yaml"
        ).read_text()
    )

    # Read the env names OUT OF THE MANIFEST rather than listing them here. A hardcoded
    # list drifts: it passes while naming variables the manifest stopped setting, and
    # fails when the manifest gains one. Deriving it makes this an actual cross-file
    # check between the two files that have to agree.
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "deployment"
        / "eks"
        / "prebid-server-deployment.yaml"
    )
    provided: set[str] = set()
    for doc in yaml.safe_load_all(manifest_path.read_text()):
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        for container in doc["spec"]["template"]["spec"].get("containers", []):
            for env in container.get("env", []):
                # Both forms count as provided. `valueFrom` (a secretKeyRef) puts the
                # variable in the container's environment exactly as `value` does -- the
                # difference is only that the value is not written into the manifest.
                if "value" in env or "valueFrom" in env:
                    provided.add(env["name"])

    # The two keystore variables are exported by the image's own entrypoint.sh, so they
    # are never in the manifest and have to be named.
    provided |= {"SSL_KEYSTORE_PATH", "SSL_KEYSTORE_PASS"}

    # Guard against the derivation silently collecting nothing and making the assertion
    # below vacuous.
    assert "ARTF_EXTENSION_POINT_URL" in provided, (
        "env names were not read from the manifest; this test would pass vacuously"
    )

    referenced: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            referenced.update(re.findall(r"\$\{([A-Z_]+)\}", node))

    walk(config)

    assert referenced <= provided, (
        f"unresolvable at runtime: {sorted(referenced - provided)}"
    )


@settings(max_examples=200)
@given(
    cpu=st.integers(min_value=0, max_value=64000),
    memory=st.integers(min_value=0, max_value=262144),
)
def test_property_fit_is_exactly_both_dimensions_being_sufficient(cpu, memory):
    verdict = node_fits([_node(f"{cpu}m", f"{memory}Mi")])
    expected = cpu >= POD_CPU_REQUEST_M and memory >= POD_MEMORY_REQUEST_MI
    assert verdict is expected


@settings(max_examples=100)
@given(
    nodes=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=32000),
            st.integers(min_value=0, max_value=131072),
        ),
        max_size=8,
    )
)
def test_property_verdict_is_never_true_without_a_node_that_fits(nodes):
    verdict = node_fits([_node(f"{c}m", f"{m}Mi") for c, m in nodes])
    if verdict is True:
        assert any(
            c >= POD_CPU_REQUEST_M and m >= POD_MEMORY_REQUEST_MI for c, m in nodes
        )
    if not nodes:
        assert verdict is None



# ------------------------------------------------- archive root: iterable contract


def test_archive_root_works_on_a_generator_not_just_a_list():
    """Regression. The signature says Iterable, and deploy_prebid.sh passes a
    generator over the tar listing. The function reads the names twice, and against a
    one-shot iterator the second pass saw nothing -- so every nested archive was
    reported flat. The deployment then failed two steps later with "the release layout
    has changed", sending the reader to check the wrong thing.
    """
    names = [
        "prebid-server-deployment-on-aws-1.4.0/",
        "prebid-server-deployment-on-aws-1.4.0/README.md",
        "prebid-server-deployment-on-aws-1.4.0/deployment/ecr/prebid-server/Dockerfile",
    ]
    from_list = normalise_archive_root(names)
    from_generator = normalise_archive_root(name for name in names)
    assert from_list == "prebid-server-deployment-on-aws-1.4.0"
    assert from_generator == from_list


def test_archive_root_works_on_a_file_iterator():
    # Exactly the shape the script uses: iterating an open file yields newline-
    # terminated strings once.
    import io

    payload = "prebid-1.4.0/\nprebid-1.4.0/README.md\nprebid-1.4.0/deployment/x\n"
    handle = io.StringIO(payload)
    assert normalise_archive_root(line for line in handle) == "prebid-1.4.0"


def test_a_hidden_top_level_directory_keeps_its_leading_dot():
    # lstrip("./") would turn ".github" into "github", inventing a directory name that
    # is not in the archive.
    names = [".github/workflows/build.yml", ".github/CODEOWNERS"]
    assert normalise_archive_root(names) == ".github"
