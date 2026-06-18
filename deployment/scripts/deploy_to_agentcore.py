"""Deploy or destroy an AgentCore runtime for the Accelerator-optimized Agentic Bidding.

Called by ``deploy-agentcore.sh``.  Uses boto3 to interact with the
Bedrock AgentCore control plane.

Actions:
  deploy  — create or update the runtime, wait for READY
  destroy — delete the runtime (endpoints first)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import boto3
from botocore.exceptions import ClientError

_LOG = logging.getLogger("deploy_to_agentcore")


def _get_client(region: str):
    """Get the AgentCore control-plane client."""
    try:
        return boto3.client("bedrock-agentcore-control", region_name=region)
    except Exception:
        # Fallback — some boto3 versions use a different service name
        return boto3.client("bedrock-agentcore", region_name=region)


def _find_runtime(client, name: str) -> dict | None:
    """Find an existing runtime by name."""
    try:
        runtimes = client.list_agent_runtimes(maxResults=100).get("agentRuntimes", [])
        for r in runtimes:
            if r.get("agentRuntimeName") == name:
                return r
    except ClientError as exc:
        _LOG.error("list_agent_runtimes failed: %s", exc)
        raise
    return None


def _wait_ready(client, runtime_id: str, timeout: int = 600) -> str:
    """Poll until the runtime reaches READY or a terminal failure state."""
    for _ in range(timeout // 10):
        desc = client.get_agent_runtime(agentRuntimeId=runtime_id)
        status = desc.get("status", "UNKNOWN")
        _LOG.info("Runtime %s status: %s", runtime_id, status)
        if status == "READY":
            return desc.get("agentRuntimeArn", "")
        if "FAILED" in status:
            reason = desc.get("failureReason", "unknown")
            raise RuntimeError(f"Runtime entered {status}: {reason}")
        time.sleep(10)  # nosemgrep: arbitrary-sleep  — polling for runtime readiness
    raise TimeoutError(f"Runtime {runtime_id} did not reach READY within {timeout}s")


def deploy(client, *, name: str, role_arn: str, container_uri: str) -> None:
    """Create or update the AgentCore runtime."""
    existing = _find_runtime(client, name)

    if existing is None:
        _LOG.info("Creating AgentCore runtime %s", name)
        resp = client.create_agent_runtime(
            agentRuntimeName=name,
            description="Accelerator-optimized Agentic Bidding — DLRM, Wide&Deep, NCF, Metrics",
            roleArn=role_arn,
            agentRuntimeArtifact={
                "containerConfiguration": {"containerUri": container_uri},
            },
            networkConfiguration={"networkMode": "PUBLIC"},
            protocolConfiguration={"serverProtocol": "MCP"},
        )
        runtime_id = resp["agentRuntimeId"]
        _LOG.info("Created runtime %s (id=%s)", name, runtime_id)
    else:
        runtime_id = existing["agentRuntimeId"]
        _LOG.info("Updating AgentCore runtime %s (id=%s)", name, runtime_id)
        client.update_agent_runtime(
            agentRuntimeId=runtime_id,
            roleArn=role_arn,
            agentRuntimeArtifact={
                "containerConfiguration": {"containerUri": container_uri},
            },
            networkConfiguration={"networkMode": "PUBLIC"},
            protocolConfiguration={"serverProtocol": "MCP"},
        )

    arn = _wait_ready(client, runtime_id)
    _LOG.info("Runtime is READY: %s", arn)
    _LOG.info("")
    _LOG.info("=== Invoke with ===")
    _LOG.info("  aws bedrock-agentcore invoke-agent-runtime \\")
    _LOG.info("    --agent-runtime-arn %s \\", arn)
    _LOG.info("    --payload '{\"method\":\"tools/call\",\"params\":{\"name\":\"extend_rtb\",\"arguments\":{...}}}'")
    _LOG.info("")
    _LOG.info("Or via the MCP endpoint at the runtime's URL + /mcp")


def destroy(client, *, name: str) -> None:
    """Delete the AgentCore runtime."""
    existing = _find_runtime(client, name)
    if existing is None:
        _LOG.info("Runtime %s not found — nothing to destroy", name)
        return

    runtime_id = existing["agentRuntimeId"]

    # Delete non-DEFAULT endpoints first
    try:
        endpoints = client.list_agent_runtime_endpoints(agentRuntimeId=runtime_id).get("agentRuntimeEndpoints", [])
        for ep in endpoints:
            ep_name = ep.get("name", "")
            if ep_name != "DEFAULT":
                _LOG.info("Deleting endpoint %s", ep_name)
                client.delete_agent_runtime_endpoint(agentRuntimeId=runtime_id, endpointName=ep_name)
    except ClientError as exc:
        _LOG.warning("Could not list/delete endpoints: %s", exc)

    _LOG.info("Deleting runtime %s (id=%s)", name, runtime_id)
    client.delete_agent_runtime(agentRuntimeId=runtime_id)
    _LOG.info("Runtime deleted.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=["deploy", "destroy"])
    parser.add_argument("--runtime-name", required=True)
    parser.add_argument("--role-arn", default="")
    parser.add_argument("--container-uri", default="")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    client = _get_client(args.region)

    if args.action == "deploy":
        if not args.role_arn or not args.container_uri:
            _LOG.error("--role-arn and --container-uri are required for deploy")
            return 1
        deploy(client, name=args.runtime_name, role_arn=args.role_arn, container_uri=args.container_uri)
    elif args.action == "destroy":
        destroy(client, name=args.runtime_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
