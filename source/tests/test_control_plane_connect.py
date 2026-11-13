"""The pure parts of deployment/scripts/control_plane_connect.py: URL and region
resolution and argument parsing. The network and MCP parts run only against a live
runtime (the user-run live gate)."""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..", "..")
_SCRIPT = os.path.join(_REPO, "deployment", "scripts", "control_plane_connect.py")
ARN = "arn:aws:bedrock-agentcore:eu-west-1:123456789012:runtime/nv5_nvidia_artf_recommenders_control-abc123"


@pytest.fixture(scope="module")
def connect():
    spec = importlib.util.spec_from_file_location("control_plane_connect_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_region_comes_from_the_arn(connect):
    assert connect.region_from_arn(ARN) == "eu-west-1"
    assert connect.region_from_arn("not-an-arn") is None
    region, how = connect.resolve_region(None, ARN)
    assert (region, how) == ("eu-west-1", "the runtime ARN")


def test_an_explicit_region_wins_but_a_mismatch_is_reported(connect, capsys):
    region, how = connect.resolve_region("us-east-1", ARN)
    assert (region, how) == ("us-east-1", "--region")
    assert "disagrees" in capsys.readouterr().err


def test_mcp_url_percent_encodes_the_arn_into_the_path(connect):
    url = connect.mcp_url(ARN, "eu-west-1")
    assert url.startswith("https://bedrock-agentcore.eu-west-1.amazonaws.com/runtimes/")
    assert ":" not in url.split("/runtimes/")[1].split("/invocations")[0]
    assert url.endswith("/invocations?qualifier=DEFAULT")
    assert "arn%3Aaws%3Abedrock-agentcore%3Aeu-west-1" in url
    assert "runtime%2Fnv5_nvidia_artf_recommenders_control-abc123" in url


def test_tool_args_merge_json_and_pairs_with_json_typed_values(connect):
    args = connect.parse_tool_args('{"status": "active"}', ["campaign_id=camp-cedar", "open_market=true", 'deal_ids=["deal-a"]', "declared_cpm=4.5"])
    assert args == {"status": "active", "campaign_id": "camp-cedar", "open_market": True, "deal_ids": ["deal-a"], "declared_cpm": 4.5}


def test_tool_args_reject_malformed_input(connect):
    with pytest.raises(ValueError):
        connect.parse_tool_args("[]", [])
    with pytest.raises(ValueError):
        connect.parse_tool_args("{}", ["no-equals-sign"])


def test_default_scopes_are_the_four_control_scopes(connect):
    scopes = set(connect.DEFAULT_SCOPES.split())
    assert scopes == {
        "artf-control/campaigns:read", "artf-control/campaigns:write",
        "artf-control/deals:read", "artf-control/deals:write",
    }


def test_decode_claims_never_raises(connect):
    assert connect.decode_claims("garbage") == {}
    assert connect.decode_claims("a.eyJjbGllbnRfaWQiOiAieCJ9.c") == {"client_id": "x"}
