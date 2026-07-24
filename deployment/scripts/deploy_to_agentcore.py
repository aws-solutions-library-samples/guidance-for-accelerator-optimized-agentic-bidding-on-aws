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


def _build_network_configuration(
    network_mode: str,
    subnets: list[str] | None,
    security_groups: list[str] | None,
) -> dict:
    """Build the AgentCore networkConfiguration.

    PUBLIC -> {"networkMode": "PUBLIC"}
    VPC    -> {"networkMode": "VPC", "networkModeConfig": {"subnets": [...],
               "securityGroups": [...]}}  (verified against the CreateAgentRuntime API)

    VPC mode governs the runtime's OUTBOUND traffic only (inbound invocations are
    unaffected), which is what lets the runtime reach in-cluster services (the
    TensorRT optimizer + Triton) via their internal NLBs.
    """
    mode = (network_mode or "PUBLIC").upper()
    if mode == "VPC":
        if not subnets:
            raise ValueError("VPC network mode requires at least one subnet")
        return {
            "networkMode": "VPC",
            "networkModeConfig": {
                "subnets": list(subnets),
                "securityGroups": list(security_groups or []),
            },
        }
    return {"networkMode": "PUBLIC"}


def deploy(
    client,
    *,
    name: str,
    role_arn: str,
    container_uri: str,
    protocol: str = "MCP",
    environment: dict[str, str] | None = None,
    description: str | None = None,
    network_mode: str = "PUBLIC",
    subnets: list[str] | None = None,
    security_groups: list[str] | None = None,
) -> str:
    """Create or update the AgentCore runtime.

    Args:
        protocol: AgentCore server protocol — "MCP" or "HTTP".
        environment: Optional environment variables to set in the runtime
            (top-level ``environmentVariables`` map per the CreateAgentRuntime API).
        network_mode: "PUBLIC" (default) or "VPC".
        subnets: VPC subnet IDs (required for VPC mode). Must be in AgentCore's
            supported AZs for the region.
        security_groups: VPC security group IDs applied to the runtime ENIs.
    Returns:
        The runtime ARN once READY.
    """
    existing = _find_runtime(client, name)
    protocol = (protocol or "MCP").upper()
    env = dict(environment or {})
    net_cfg = _build_network_configuration(network_mode, subnets, security_groups)
    desc = description or (
        "Accelerator-optimized Agentic Bidding — DLRM, Wide&Deep, NCF, Metrics"
    )

    if existing is None:
        _LOG.info("Creating AgentCore runtime %s (protocol=%s)", name, protocol)
        create_kwargs = dict(
            agentRuntimeName=name,
            description=desc,
            roleArn=role_arn,
            agentRuntimeArtifact={
                "containerConfiguration": {"containerUri": container_uri},
            },
            networkConfiguration=net_cfg,
            protocolConfiguration={"serverProtocol": protocol},
        )
        if env:
            # environmentVariables is a top-level {string: string} map on
            # CreateAgentRuntime (verified against the control-plane API).
            create_kwargs["environmentVariables"] = env
        resp = client.create_agent_runtime(**create_kwargs)
        runtime_id = resp["agentRuntimeId"]
        _LOG.info("Created runtime %s (id=%s)", name, runtime_id)
    else:
        runtime_id = existing["agentRuntimeId"]
        _LOG.info("Updating AgentCore runtime %s (id=%s, protocol=%s)", name, runtime_id, protocol)
        update_kwargs = dict(
            agentRuntimeId=runtime_id,
            roleArn=role_arn,
            agentRuntimeArtifact={
                "containerConfiguration": {"containerUri": container_uri},
            },
            networkConfiguration=net_cfg,
            protocolConfiguration={"serverProtocol": protocol},
        )
        if env:
            update_kwargs["environmentVariables"] = env
        client.update_agent_runtime(**update_kwargs)

    arn = _wait_ready(client, runtime_id)
    _LOG.info("Runtime is READY: %s", arn)
    _LOG.info("")
    _LOG.info("=== Invoke with ===")
    _LOG.info("  aws bedrock-agentcore invoke-agent-runtime \\")
    _LOG.info("    --agent-runtime-arn %s \\", arn)
    if protocol == "MCP":
        _LOG.info("    --payload '{\"method\":\"tools/call\",\"params\":{\"name\":\"extend_rtb\",\"arguments\":{...}}}'")
        _LOG.info("")
        _LOG.info("Or via the MCP endpoint at the runtime's URL + /mcp")
    else:
        _LOG.info("    --payload '{...}'   # POSTed to the runtime's /invocations")
    return arn


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
    parser.add_argument(
        "--protocol",
        default="MCP",
        choices=["MCP", "HTTP"],
        help="AgentCore server protocol (default: MCP).",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for the runtime; repeatable (e.g. --environment PARAMETER_STORE_TABLE=foo).",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional runtime description.",
    )
    parser.add_argument(
        "--network-mode",
        default="PUBLIC",
        choices=["PUBLIC", "VPC"],
        help="AgentCore network mode (default: PUBLIC). VPC lets the runtime reach in-cluster services.",
    )
    parser.add_argument(
        "--subnets",
        default="",
        help="Comma-separated VPC subnet IDs (required for --network-mode VPC).",
    )
    parser.add_argument(
        "--security-groups",
        default="",
        help="Comma-separated VPC security group IDs (for --network-mode VPC).",
    )
    parser.add_argument(
        "--print-arn",
        action="store_true",
        help="Print the runtime ARN to stdout on success (for shell capture).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    client = _get_client(args.region)

    if args.action == "deploy":
        if not args.role_arn or not args.container_uri:
            _LOG.error("--role-arn and --container-uri are required for deploy")
            return 1
        # Parse KEY=VALUE environment entries.
        environment: dict[str, str] = {}
        for entry in args.environment:
            if "=" not in entry:
                _LOG.error("Invalid --environment entry (expected KEY=VALUE): %s", entry)
                return 1
            key, _, value = entry.partition("=")
            key = key.strip()
            if not key:
                _LOG.error("Invalid --environment entry (empty key): %s", entry)
                return 1
            environment[key] = value
        subnets = [s.strip() for s in args.subnets.split(",") if s.strip()]
        security_groups = [s.strip() for s in args.security_groups.split(",") if s.strip()]
        if args.network_mode == "VPC" and not subnets:
            _LOG.error("--network-mode VPC requires --subnets")
            return 1
        arn = deploy(
            client,
            name=args.runtime_name,
            role_arn=args.role_arn,
            container_uri=args.container_uri,
            protocol=args.protocol,
            environment=environment,
            description=args.description or None,
            network_mode=args.network_mode,
            subnets=subnets,
            security_groups=security_groups,
        )
        if args.print_arn:
            print(arn)
    elif args.action == "destroy":
        destroy(client, name=args.runtime_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
