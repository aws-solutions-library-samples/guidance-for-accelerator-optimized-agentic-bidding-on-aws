"""The governance agent's proxy read timeout must outlast the proxy Lambda.

A RequestResponse lambda:InvokeFunction holds the connection until the function
returns, so a client read timeout below the Lambda's own Timeout makes the caller
abandon a call that is still running and would still have succeeded.

That is exactly what rejected dlrm-bid-shader version 2 on the live stack: the
client default was 600s (read timeout 630s) while vpc_proxy_cfn.yaml sets the
Lambda's Timeout to 900s. A TensorRT optimize ran 846s, the client gave up first,
and the agent recorded:

    Model optimization failed: ... status=502, body=proxy invoke failed:
    Read timeout on endpoint URL: ".../nv5-vpc-optimizer-proxy/invocations"

so no canary was ever staged and the A/B test never ran. The compile had not
failed.

These tests pin the ordering (client >= Lambda) rather than the specific number,
so raising one without the other fails here instead of in production.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

VPC_PROXY_CFN = REPO_ROOT / "deployment" / "vpc_proxy_cfn.yaml"
DEPLOY_CLOSED_LOOP = REPO_ROOT / "deployment" / "deploy_closed_loop.sh"

# The duration actually observed on the live stack for a TensorRT optimize.
OBSERVED_OPTIMIZE_SECONDS = 846


def _proxy_lambda_timeout() -> int:
    """The proxy Lambda's Timeout as declared in its CloudFormation template."""
    text = VPC_PROXY_CFN.read_text(encoding="utf-8")
    matches = [int(m) for m in re.findall(r"^\s*Timeout:\s*(\d+)\s*$", text, re.M)]
    assert matches, "no Timeout found in vpc_proxy_cfn.yaml"
    return max(matches)


def _client(**kwargs):
    import agents.governance.integrations as integrations

    importlib.reload(integrations)
    return integrations.LambdaProxyHttpClient(
        "arn:aws:lambda:us-east-1:1:function:p", region="us-east-1", **kwargs
    )


def test_client_timeout_is_not_below_the_proxy_lambda_timeout():
    """The invariant. Violating it means abandoning in-flight, succeeding calls."""
    assert _client()._timeout >= _proxy_lambda_timeout()


def test_client_timeout_covers_the_observed_optimize_duration():
    """A real 846s compile must not trip the client. The +30 is the margin
    _lambda_client() adds on top of _timeout for the read timeout."""
    assert int(_client()._timeout) + 30 > OBSERVED_OPTIMIZE_SECONDS


def test_env_var_overrides_the_default(monkeypatch):
    """Deployment sets this (deploy_closed_loop.sh), so both sides can be raised
    together without a code change."""
    monkeypatch.setenv("VPC_PROXY_TIMEOUT_SECONDS", "1200")
    assert _client()._timeout == 1200.0


def test_explicit_argument_still_wins():
    """Callers and tests can pin a short timeout without touching the env."""
    assert _client(timeout=42.0)._timeout == 42.0


def test_deploy_script_passes_the_timeout_to_the_agent_runtime():
    """A default that only exists in Python would drift from the deployed agent.
    The value has to be handed to the runtime alongside the proxy ARN.
    """
    text = DEPLOY_CLOSED_LOOP.read_text(encoding="utf-8")
    assert "VPC_PROXY_TIMEOUT_SECONDS=" in text
    assert '--environment "VPC_PROXY_TIMEOUT_SECONDS=' in text


def test_deploy_script_default_matches_the_lambda_timeout():
    """The shell default and the Lambda ceiling must agree, or the deployed agent
    silently reverts to the too-short behavior this fixes."""
    text = DEPLOY_CLOSED_LOOP.read_text(encoding="utf-8")
    m = re.search(r'VPC_PROXY_TIMEOUT_SECONDS="\$\{VPC_PROXY_TIMEOUT_SECONDS:-(\d+)\}"', text)
    assert m, "no shell default found for VPC_PROXY_TIMEOUT_SECONDS"
    assert int(m.group(1)) >= _proxy_lambda_timeout()


def test_lambda_ceiling_headroom_is_documented_as_thin():
    """Guard rather than assertion of comfort: AWS caps Lambda at 900s, and the
    observed compile already used 846s. If a compile exceeds the ceiling the
    Lambda itself times out and no client setting can help -- that needs an
    async launch-and-poll proxy, not a bigger timeout.
    """
    ceiling = _proxy_lambda_timeout()
    assert ceiling <= 900, "900s is the AWS Lambda maximum; above this is not deployable"
    headroom = ceiling - OBSERVED_OPTIMIZE_SECONDS
    assert headroom > 0, (
        f"observed optimize ({OBSERVED_OPTIMIZE_SECONDS}s) already exceeds the "
        f"Lambda ceiling ({ceiling}s) — a synchronous proxy cannot work"
    )
