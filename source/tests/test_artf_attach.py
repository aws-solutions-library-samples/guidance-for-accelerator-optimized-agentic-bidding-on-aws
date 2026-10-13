"""Tests for the external ARTF container attach helper.

The helper reads the *producer's* record format. So the most valuable test here is
``test_parses_the_real_producer_record``, which runs against the actual file the
producing repository generated — a hand-written fixture could drift from it
without anyone noticing, and then the helper would be verified against a format
nobody emits.

Two rules encoded here that are easy to get wrong later:

- ``intents`` must serialise as an ordered list (``L`` of ``S``), never a string
  set. A ``SS`` sorts and deduplicates, silently losing the order the producer
  wrote.
- ``check_endpoint`` is a **refusal**, not a warning. ARTF conformance requires
  the container be reachable only inside the cluster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_SCRIPTS = Path(__file__).resolve().parents[2] / "deployment" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from artf_attach import (  # noqa: E402
    GPU_SUPPORTED_ARCHES,
    MUTATE_PORT,
    REQUEST_SIDE_INTENTS,
    AttachError,
    ExternalContainer,
    build_item,
    ceiling_remedy,
    check_endpoint,
    check_prebid_ceiling,
    describe_contention,
    gpu_guard,
    item_to_record,
    parse_record,
    probe_payload,
    render_manifest,
    resolve_table,
    retarget_endpoint,
    service_name,
    split_image,
    summarise_probe,
)

#: The producing repository, if it happens to be checked out beside this one.
PRODUCER_RECORD = (
    Path(__file__).resolve().parents[3]
    / "guidance-for-containerized-semantic-context-on-aws"
    / "deployment"
    / "artf-registration"
    / "artf-registry-record.json"
)

MINIMAL = {
    "record": {
        "registry": "artf-containers",
        "name": "contextual-yield-agent",
        "display_name": "Contextual Yield Agent",
        "description": "A contextual yield and supply agent.",
        "intents": ["ADJUST_DEAL_FLOOR"],
        "endpoint": "http://contextual-yield-agent.default.svc.cluster.local:8081",
        "active": False,
    },
    "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/semantic-context-engine-artf:src-abc-cache-def",
    "image_digest": "sha256:1871ed",
    "namespace": "default",
    "serving_modes": ["cpu"],
    "gpu_architectures": [],
}


def _container(**overrides) -> ExternalContainer:
    data = json.loads(json.dumps(MINIMAL))
    for key, value in overrides.items():
        if key in data["record"]:
            data["record"][key] = value
        else:
            data[key] = value
    return parse_record(data)


# ---------------------------------------------------------------------------
# Parsing — including against the real artifact
# ---------------------------------------------------------------------------

class TestParseRecord:
    @pytest.mark.skipif(
        not PRODUCER_RECORD.exists(),
        reason="the producing repository is not checked out beside this one",
    )
    def test_parses_the_real_producer_record(self):
        """Runs against the file the producer actually generated.

        If their format changes, this fails here rather than at attach time.
        """
        container = parse_record(PRODUCER_RECORD.read_text())
        assert container.name == "contextual-yield-agent"
        assert "ADJUST_DEAL_FLOOR" in container.intents
        assert check_endpoint(container.endpoint).ok
        assert container.image
        assert container.image_digest.startswith("sha256:")
        # The published build is CPU-only; the GPU guard depends on this.
        assert container.supports_gpu is False

    def test_parses_minimal_record(self):
        c = parse_record(MINIMAL)
        assert c.name == "contextual-yield-agent"
        assert c.display_name == "Contextual Yield Agent"
        assert c.intents == ("ADJUST_DEAL_FLOOR",)
        assert c.namespace == "default"

    def test_accepts_a_json_string(self):
        assert parse_record(json.dumps(MINIMAL)).name == "contextual-yield-agent"

    def test_display_name_falls_back_to_name(self):
        assert _container(display_name="").display_name == "contextual-yield-agent"

    def test_intents_are_upper_cased(self):
        assert _container(intents=["adjust_deal_floor"]).intents == ("ADJUST_DEAL_FLOOR",)

    def test_rejects_a_non_object(self):
        with pytest.raises(AttachError, match="not a JSON object"):
            parse_record("[]")

    def test_rejects_a_missing_record_block(self):
        with pytest.raises(AttachError, match="no 'record' object"):
            parse_record({"image": "x"})

    def test_rejects_a_missing_name(self):
        with pytest.raises(AttachError, match="no container name"):
            parse_record({"record": {"endpoint": "http://x:8081", "intents": ["A"]}, "image": "i"})

    def test_rejects_a_missing_endpoint(self):
        with pytest.raises(AttachError, match="no endpoint"):
            parse_record({"record": {"name": "x", "intents": ["A"]}, "image": "i"})

    def test_rejects_an_empty_intent_list(self):
        """A container with no intent is never called, so registering it is a no-op."""
        with pytest.raises(AttachError, match="claims no intents"):
            parse_record({"record": {"name": "x", "endpoint": "http://x:8081", "intents": []}, "image": "i"})

    def test_rejects_a_missing_image(self):
        with pytest.raises(AttachError, match="no image reference"):
            parse_record({"record": MINIMAL["record"]})


# ---------------------------------------------------------------------------
# Endpoint enforcement
# ---------------------------------------------------------------------------

class TestCheckEndpoint:
    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://contextual-yield-agent.default.svc.cluster.local:8081",
            "http://contextual-yield-agent.artf.svc.cluster.local:8081",
            "http://contextual-yield-agent:8081",
            "http://svc.namespace:8081",
        ],
    )
    def test_accepts_cluster_endpoints(self, endpoint):
        assert check_endpoint(endpoint).ok, check_endpoint(endpoint).reason

    def test_rejects_https(self):
        check = check_endpoint("https://agent.example.com:443")
        assert not check.ok
        assert "not a cluster-internal" in check.reason

    def test_rejects_a_missing_scheme(self):
        assert not check_endpoint("agent:8081").ok

    def test_rejects_a_missing_port(self):
        check = check_endpoint("http://agent")
        assert not check.ok
        assert str(MUTATE_PORT) in check.reason

    def test_rejects_a_path(self):
        check = check_endpoint("http://agent:8081/mutate")
        assert not check.ok
        assert "appends /mutate itself" in check.reason

    def test_rejects_an_ip_address(self):
        """An IP is reachable but is not a Service, so it cannot survive a reschedule."""
        check = check_endpoint("http://10.0.1.5:8081")
        assert not check.ok
        assert "Service" in check.reason

    def test_rejects_an_empty_endpoint(self):
        assert not check_endpoint("").ok

    def test_rejects_a_non_numeric_port(self):
        assert not check_endpoint("http://agent:mutate").ok

    def test_rejects_a_truncated_cluster_dns_name(self):
        assert not check_endpoint("http://agent.svc.cluster.local:8081").ok

    def test_extracts_namespace_and_port(self):
        check = check_endpoint("http://agent.artf.svc.cluster.local:8081")
        assert check.namespace == "artf"
        assert check.port == 8081

    def test_service_name(self):
        assert service_name("http://agent.artf.svc.cluster.local:8081") == "agent"
        assert service_name("https://nope.example.com") == ""


class TestRetargetEndpoint:
    def test_rewrites_the_namespace(self):
        """The producer's endpoint hardcodes the namespace chosen at generation."""
        assert retarget_endpoint(
            "http://agent.default.svc.cluster.local:8081", "artf"
        ) == "http://agent.artf.svc.cluster.local:8081"

    def test_preserves_the_port(self):
        assert retarget_endpoint("http://agent.default.svc.cluster.local:9090", "x").endswith(":9090")

    def test_refuses_an_invalid_endpoint(self):
        with pytest.raises(AttachError):
            retarget_endpoint("https://agent.example.com", "artf")


# ---------------------------------------------------------------------------
# Table resolution
# ---------------------------------------------------------------------------

class TestResolveTable:
    def test_uses_this_stacks_name(self):
        """Not the producer's 'artf-container-registry' default."""
        assert resolve_table("nv5") == "nv5-container-registry"

    def test_override_wins(self):
        assert resolve_table("nv5", "other-table") == "other-table"

    def test_requires_a_stack_name(self):
        with pytest.raises(AttachError, match="No stack name"):
            resolve_table("")


# ---------------------------------------------------------------------------
# DynamoDB item and the round trip (PBT-R4)
# ---------------------------------------------------------------------------

class TestBuildItem:
    def test_written_inactive(self):
        """Installing must not change what the bid path does."""
        assert build_item(_container())["active"] == {"BOOL": False}

    def test_priority_defaults_to_zero(self):
        assert build_item(_container())["priority"] == {"N": "0"}

    def test_priority_is_written_as_a_number(self):
        assert build_item(_container(), priority=-3)["priority"] == {"N": "-3"}

    def test_intents_are_an_ordered_list_not_a_string_set(self):
        """A SS would sort and deduplicate, losing the producer's order."""
        c = _container(intents=["ADD_METRICS", "ADJUST_DEAL_FLOOR"])
        item = build_item(c)
        assert "SS" not in item["intents"]
        assert item["intents"] == {"L": [{"S": "ADD_METRICS"}, {"S": "ADJUST_DEAL_FLOOR"}]}

    def test_endpoint_override_is_used(self):
        item = build_item(_container(), endpoint="http://agent.artf.svc.cluster.local:8081")
        assert item["endpoint"]["S"].endswith(".artf.svc.cluster.local:8081")

    def test_optional_audit_fields_are_omitted_when_empty(self):
        item = build_item(_container())
        assert "updated_at" not in item
        assert "updated_by" not in item

    def test_round_trip_preserves_the_record(self):
        c = _container(intents=["ADD_METRICS", "ADJUST_DEAL_FLOOR"])
        back = item_to_record(build_item(c, priority=4))
        assert back["name"] == c.name
        assert back["display_name"] == c.display_name
        assert back["intents"] == list(c.intents)
        assert back["endpoint"] == c.endpoint
        assert back["active"] is False
        assert back["priority"] == 4


INTENT_ST = st.sampled_from(sorted(REQUEST_SIDE_INTENTS))


@settings(max_examples=200)
@given(
    name=st.from_regex(r"[a-z][a-z0-9-]{0,20}", fullmatch=True),
    display=st.text(min_size=0, max_size=40),
    description=st.text(min_size=0, max_size=80),
    intents=st.lists(INTENT_ST, min_size=1, max_size=5),
    priority=st.integers(min_value=-50, max_value=50),
)
def test_property_item_round_trip(name, display, description, intents, priority):
    """PBT-R4: record -> typed item -> record is lossless for the fields we write."""
    container = ExternalContainer(
        name=name,
        display_name=display or name,
        description=description,
        intents=tuple(intents),
        endpoint=f"http://{name}.default.svc.cluster.local:{MUTATE_PORT}",
        image="r/i:t",
        image_digest="sha256:x",
        namespace="default",
    )
    back = item_to_record(build_item(container, priority=priority))
    assert back["name"] == container.name
    assert back["display_name"] == container.display_name
    assert back["description"] == container.description
    # Order preserved, duplicates preserved — the L-of-S guarantee.
    assert back["intents"] == list(container.intents)
    assert back["priority"] == priority
    assert back["active"] is False


# ---------------------------------------------------------------------------
# Prebid ceiling
# ---------------------------------------------------------------------------

class TestPrebidCeiling:
    def test_covered_when_the_ceiling_lists_the_intent(self):
        check = check_prebid_ceiling(["ADJUST_DEAL_FLOOR"], sorted(REQUEST_SIDE_INTENTS))
        assert check.ok
        assert check.missing == ()

    def test_missing_when_the_ceiling_omits_it(self):
        check = check_prebid_ceiling(["ADJUST_DEAL_FLOOR"], ["ADD_METRICS"])
        assert not check.ok
        assert check.missing == ("ADJUST_DEAL_FLOOR",)

    def test_bid_shade_is_flagged_as_outside_the_request_side_set(self):
        """BID_SHADE addresses the response, where the hook has no seatbid."""
        check = check_prebid_ceiling(["BID_SHADE"], sorted(REQUEST_SIDE_INTENTS))
        assert "BID_SHADE" in check.unknown

    def test_unreadable_config_is_not_reported_as_missing(self):
        """Not knowing the ceiling differs from knowing the intent is absent."""
        check = check_prebid_ceiling(["ADJUST_DEAL_FLOOR"], None)
        assert check.missing == ()
        assert check.covered == ()

    def test_remedy_names_the_restart(self):
        text = ceiling_remedy(["ADD_CIDS"], "prebid-server/current/prebid-config.yaml")
        assert "ADD_CIDS" in text
        assert "rollout restart" in text
        assert "read at container start" in text

    def test_remedy_is_empty_when_nothing_is_missing(self):
        assert ceiling_remedy([], "k") == ""


# ---------------------------------------------------------------------------
# Contention
# ---------------------------------------------------------------------------

class TestDescribeContention:
    BUILTIN = [{
        "name": "yield-optimizer-floor",
        "active": True,
        "intents": ["ADJUST_DEAL_FLOOR"],
        "source": "code",
        "priority": 0,
    }]

    def test_default_priority_beats_a_builtin_and_says_so(self):
        warnings = describe_contention(_container(), self.BUILTIN, priority=0)
        assert len(warnings) == 1
        assert "would win" in warnings[0]
        assert "negative priority" in warnings[0]

    def test_negative_priority_keeps_the_builtin(self):
        warnings = describe_contention(_container(), self.BUILTIN, priority=-1)
        assert "'yield-optimizer-floor' would win on priority" in warnings[0]

    def test_higher_priority_wins_explicitly(self):
        warnings = describe_contention(_container(), self.BUILTIN, priority=5)
        assert "would win on priority (5 > 0)" in warnings[0]

    def test_no_warning_when_nothing_is_claimed(self):
        assert describe_contention(_container(intents=["ADD_CIDS"]), self.BUILTIN) == []

    def test_inactive_rival_is_not_contention(self):
        inactive = [{**self.BUILTIN[0], "active": False}]
        assert describe_contention(_container(), inactive) == []

    def test_a_container_does_not_contend_with_itself(self):
        same = [{
            "name": "contextual-yield-agent",
            "active": True,
            "intents": ["ADJUST_DEAL_FLOOR"],
            "source": "store",
            "priority": 0,
        }]
        assert describe_contention(_container(), same) == []


# ---------------------------------------------------------------------------
# Image and manifest
# ---------------------------------------------------------------------------

class TestSplitImage:
    def test_splits_repo_and_tag(self):
        assert split_image("r.example.com/repo:tag-1") == ("r.example.com/repo", "tag-1")

    def test_registry_port_is_not_mistaken_for_a_tag(self):
        assert split_image("localhost:5000/repo") == ("localhost:5000/repo", "latest")

    def test_digest_reference(self):
        repo, digest = split_image("r/repo@sha256:abc")
        assert repo == "r/repo"
        assert digest == "sha256:abc"

    def test_rejects_empty(self):
        with pytest.raises(AttachError):
            split_image("")


TEMPLATE = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: contextual-yield-agent
spec:
  template:
    spec:
      containers:
        - name: agent
          image: __REGISTRY__:__TAG__
          imagePullPolicy: IfNotPresent
---
apiVersion: v1
kind: Service
metadata:
  name: contextual-yield-agent
# kind: Deployment       <- commented GPU shape
#         image: __REGISTRY__:__TAG__
"""


class TestRenderManifest:
    def test_substitutes_registry_and_tag(self):
        out = render_manifest(TEMPLATE, image="acct.ecr/repo:v1", namespace="")
        assert "image: acct.ecr/repo:v1" in out
        assert "__REGISTRY__" not in out
        assert "__TAG__" not in out

    def test_rewrites_pull_policy_to_this_repos_convention(self):
        out = render_manifest(TEMPLATE, image="r/i:t", namespace="")
        assert "imagePullPolicy: Always" in out
        assert "IfNotPresent" not in out

    def test_sets_the_namespace_on_uncommented_documents(self):
        out = render_manifest(TEMPLATE, image="r/i:t", namespace="artf")
        assert out.count("namespace: artf") == 2

    def test_does_not_touch_commented_blocks(self):
        """The GPU shape is commented in the same file; rewriting it would make the
        rendered manifest disagree with the variant actually selected."""
        out = render_manifest(TEMPLATE, image="r/i:t", namespace="artf")
        for line in out.splitlines():
            if line.strip().startswith("#"):
                assert "namespace: artf" not in line

    def test_rejects_a_template_with_no_placeholders(self):
        with pytest.raises(AttachError, match="no __REGISTRY__"):
            render_manifest("kind: Deployment\n", image="r/i:t", namespace="")


# ---------------------------------------------------------------------------
# GPU guard
# ---------------------------------------------------------------------------

class TestGpuGuard:
    def test_refuses_when_the_image_is_absent(self):
        ok, reason = gpu_guard(_container(), instance_types=["g5.xlarge"], image_exists=False)
        assert not ok
        assert "not found" in reason

    def test_refuses_when_the_label_does_not_advertise_gpu(self):
        """The image's own label is the authority, not the manifest shape."""
        ok, reason = gpu_guard(_container(), instance_types=["g5.xlarge"], image_exists=True)
        assert not ok
        assert "does not include 'gpu'" in reason

    def test_allows_on_g5_which_is_sm86(self):
        c = _container(serving_modes=["cpu", "gpu"], gpu_architectures=["sm_86"])
        ok, reason = gpu_guard(c, instance_types=["g5.xlarge", "g5.2xlarge"], image_exists=True)
        assert ok, reason
        assert "sm_86" in reason

    def test_refuses_on_g6_which_is_sm89(self):
        """L4 is sm_89 and HierarchicalKV does not build for it."""
        c = _container(serving_modes=["gpu"], gpu_architectures=sorted(GPU_SUPPORTED_ARCHES))
        ok, reason = gpu_guard(c, instance_types=["g6.xlarge"], image_exists=True)
        assert not ok
        assert "do not overlap" in reason

    def test_refuses_an_unrecognised_instance_family(self):
        c = _container(serving_modes=["gpu"], gpu_architectures=["sm_86"])
        ok, reason = gpu_guard(c, instance_types=["m5.large"], image_exists=True)
        assert not ok
        assert "Refusing rather than" in reason


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class TestProbe:
    def test_payload_carries_the_containers_intents(self):
        payload = probe_payload(_container())
        assert payload["applicable_intents"] == ["ADJUST_DEAL_FLOOR"]
        # A deal with a floor, because the floor container mutates only a deal
        # that already has one.
        assert payload["bid_request"]["imp"][0]["pmp"]["deals"][0]["bidfloor"] == 1.0

    def test_summarise_reports_mutations_and_version(self):
        text = summarise_probe({"mutations": [{}], "metadata": {"model_version": "v9"}})
        assert "1 mutation" in text
        assert "v9" in text

    def test_summarise_reports_zero_mutations_without_calling_it_a_failure(self):
        text = summarise_probe({"mutations": [], "metadata": {"model_version": "v9"}})
        assert "0 mutations" in text
        assert "reachable and running" in text

    def test_summarise_explains_an_unreachable_container(self):
        text = summarise_probe(None, error="connection refused")
        assert "not been deployed yet" in text
        assert "inactive" in text

    def test_summarise_handles_a_non_object(self):
        assert "not a JSON object" in summarise_probe("nope")
