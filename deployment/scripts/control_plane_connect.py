#!/usr/bin/env python3
"""Connect to the campaign/deals MCP control plane as an external agent, over OAuth 2.0
client_credentials.

WHAT THIS PROVES, and why each step is separate. Four things have to line up before an
agent can manage a campaign or a deal, and when they do not, the symptom is the same
opaque failure:

  1. Cognito issues a token for the control client and the requested scopes
  2. AgentCore's inbound JWT authorizer accepts it: right discovery URL, allowed client,
     allowed scope
  3. the container is running, which needs ARTF_CONTROL_TOKEN_ISSUER, ARTF_CONTROL_TABLE
     and ARTF_CONTROL_DEAL_ORIGIN to have been set on the runtime
  4. the container can read Authorization (it has to be on the runtime's request-header
     allowlist) and the caller's client id is in the entitlement map

So each step is reported on its own, and a failure names WHICH of the four broke.

This is also a worked example: the token fetch and the streamable-HTTP connection here
are exactly what an agent framework does, so the code can be lifted into one. Ported
from the embedded DSP guidance's agent_connect.py.

    python3 control_plane_connect.py \\
      --runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/nv5_nvidia_artf_recommenders_control-abc \\
      --token-endpoint https://nv5-xxxx.auth.us-east-1.amazoncognito.com/oauth2/token \\
      --client-id 1example23456789 --client-secret "$SECRET"

No --region is needed: it is taken from the runtime ARN, which is the one place it cannot
disagree with the endpoint. Add ``--call-tool list_campaigns`` to exercise a tool, and
``--arg campaign_id=camp-cedar`` (repeatable) or ``--tool-args '{...}'`` for its arguments.

Needs the ``mcp`` package (pip install mcp). Nothing here uses AWS credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

#: All four control scopes. Requesting every one means a failure here is a CONFIGURATION
#: problem rather than an agent that happened to ask for too little. Narrow with --scopes.
DEFAULT_SCOPES = (
    "artf-control/campaigns:read artf-control/campaigns:write "
    "artf-control/deals:read artf-control/deals:write"
)


def hr(char: str = "-") -> str:
    return char * 78


def step(number: int, text: str) -> None:
    print(f"\n[{number}/4] {text}")


def tls_context() -> ssl.SSLContext:
    """A verifying TLS context for the token request.

    A python.org build of Python on macOS ships no CA bundle until its "Install
    Certificates" step runs, so the default context fails every HTTPS request with
    CERTIFICATE_VERIFY_FAILED. The MCP leg already trusts certifi's bundle (through
    httpx), so the token leg uses it too. SSL_CERT_FILE or SSL_CERT_DIR, when set,
    win, so a corporate proxy's CA still works.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_token(token_endpoint: str, client_id: str, client_secret: str, scopes: str) -> dict:
    """An access token from the Cognito hosted token endpoint.

    HTTP Basic for the client credential, which is what Cognito's client_credentials grant
    expects: the secret goes in the Authorization header, never in the body or a query
    string where it would reach access logs.
    """
    body = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": scopes}).encode()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    request = urllib.request.Request(
        token_endpoint,
        data=body,
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15, context=tls_context()) as response:  # noqa: S310 - https endpoint from the caller
        return json.loads(response.read())


def decode_claims(token: str) -> dict:
    """The token's claims, WITHOUT verifying it. For display only: verification is
    AgentCore's job on the way in and the control plane's job again inside."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:  # noqa: BLE001
        return {}


def region_from_arn(runtime_arn: str) -> Optional[str]:
    """``arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/x``: the fourth field."""
    parts = runtime_arn.split(":")
    if len(parts) < 5 or parts[0] != "arn":
        return None
    return parts[3] or None


def mcp_url(runtime_arn: str, region: str, qualifier: str = "DEFAULT") -> str:
    """The AgentCore streamable-HTTP invocation URL for an MCP runtime.

    The ARN is percent-encoded into the PATH: the colons and the slash all have to be
    escaped or the request 404s. The endpoint's region must match the ARN's.
    """
    encoded = runtime_arn.replace(":", "%3A").replace("/", "%2F")
    return f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded}/invocations?qualifier={qualifier}"


def resolve_region(explicit: Optional[str], runtime_arn: str) -> tuple[str, str]:
    """(region, how it was resolved): --region, then the ARN, then the environment, then
    the AWS CLI's configured region."""
    from_arn = region_from_arn(runtime_arn)
    if explicit:
        if from_arn and from_arn != explicit:
            print(
                f"      WARNING: --region {explicit} disagrees with the ARN's region {from_arn}. "
                f"Using {explicit}; a mismatch returns 404.",
                file=sys.stderr,
            )
        return explicit, "--region"
    if from_arn:
        return from_arn, "the runtime ARN"
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        value = os.environ.get(name)
        if value:
            return value, name
    try:
        result = subprocess.run(["aws", "configure", "get", "region"], capture_output=True, text=True, timeout=10)
        configured = result.stdout.strip()
        if configured:
            return configured, "aws configure"
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit("could not determine a region. Pass --region or set AWS_REGION.")


def parse_tool_args(raw_json: str, pairs: list[str]) -> dict:
    """Tool arguments from ``--tool-args`` JSON merged with ``--arg key=value`` pairs.

    A pair's value is parsed as JSON when it is one (so ``--arg open_market=true`` and
    ``--arg deal_ids='["deal-a"]'`` work) and kept as a string otherwise.
    """
    args: dict[str, Any] = json.loads(raw_json or "{}")
    if not isinstance(args, dict):
        raise ValueError("--tool-args must be a JSON object")
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--arg expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        try:
            args[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            args[key.strip()] = value
    return args


async def connect_and_list(url: str, token: str, call_tool: Optional[str], tool_args: dict) -> int:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with streamablehttp_client(url, headers, timeout=120, terminate_on_close=False) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            step(3, "Initialising the MCP session")
            await session.initialize()
            print("      session established: the container is running and reachable")

            step(4, "Listing tools")
            listed = await session.list_tools()
            tools = list(listed.tools)
            print(f"      {len(tools)} tool(s):")
            for tool in tools:
                summary = (tool.description or "").strip().splitlines()
                first = summary[0] if summary else ""
                print(f"        {tool.name:<18} {first[:56]}")

            if call_tool:
                print()
                print(hr())
                print(f"  calling {call_tool} with {json.dumps(tool_args) if tool_args else '{}'}")
                print(hr())
                result = await session.call_tool(call_tool, tool_args)
                payload_error = False
                for item in result.content:
                    text = getattr(item, "text", None)
                    print(f"    {text if text is not None else item}")
                    try:
                        payload_error = payload_error or bool(json.loads(text or "").get("error"))
                    except (ValueError, AttributeError):
                        pass
                if result.isError or payload_error:
                    print(
                        "\n    The tool returned an ERROR.\n"
                        "      unauthorized  -> the container refused the token: Authorization is not on the\n"
                        "                       runtime's requestHeaderAllowlist, or the scope is missing\n"
                        "      forbidden     -> this client id is not in ARTF_CONTROL_CLIENT_ENTITLEMENTS for\n"
                        "                       that advertiser/seller; redeploy with --control-advertisers\n"
                        "      not_found     -> no such record, or one this client may not see\n"
                        "      invalid_request -> the message names the record rule that was broken\n"
                        "    Inspect the runtime: aws bedrock-agentcore-control get-agent-runtime "
                        "--agent-runtime-id ID --region REGION"
                    )
                    return 1
            return 0


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """The innermost exceptions of a (possibly nested) exception group, in order."""
    nested = getattr(exc, "exceptions", None)
    if not nested:
        return [exc]
    leaves: list[BaseException] = []
    for inner in nested:
        leaves.extend(_leaf_exceptions(inner))
    return leaves


def explain_http_error(exc: urllib.error.HTTPError) -> None:
    body = exc.read().decode(errors="replace")[:400]
    print(f"      HTTP {exc.code}: {body}", file=sys.stderr)
    if exc.code == 400 and "invalid_scope" in body:
        print(
            "\n      invalid_scope means this client is not allowed the scopes requested.\n"
            "      The control client holds the four artf-control scopes. Check:\n"
            "        aws cognito-idp describe-user-pool-client --user-pool-id POOL \\\n"
            "          --client-id CLIENT --query 'UserPoolClient.AllowedOAuthScopes'",
            file=sys.stderr,
        )
    elif exc.code in (400, 401):
        print(
            "\n      Check the client id and secret, and that the token endpoint belongs to\n"
            "      the same user pool. A client without the client_credentials grant also\n"
            "      fails here.",
            file=sys.stderr,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-arn", required=True, help="AgentCore MCP runtime ARN")
    parser.add_argument("--token-endpoint", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True, help="Read it from the Cognito app client; never commit it")
    parser.add_argument("--region", help="Defaults to the region in the runtime ARN")
    parser.add_argument("--scopes", default=DEFAULT_SCOPES)
    parser.add_argument("--qualifier", default="DEFAULT")
    parser.add_argument("--call-tool", help="Also invoke this tool, e.g. list_campaigns")
    parser.add_argument("--tool-args", default="{}", help='JSON arguments for --call-tool, e.g. \'{"status":"active"}\'')
    parser.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE", help="One argument for --call-tool; repeatable")
    args = parser.parse_args(argv)

    secret = (args.client_secret or "").strip()
    if not secret or secret == "None":
        print(
            f"error: --client-secret is {'empty' if not secret else 'the string None'}. The shell variable\n"
            "  holding it was never set, or the AWS CLI call that read it had no credentials for the\n"
            "  account (it prints None when the client has no secret visible to it). Read it with:\n"
            "    export CLIENT_SECRET=$(aws cognito-idp describe-user-pool-client --user-pool-id POOL \\\n"
            "      --client-id CLIENT --region REGION --query UserPoolClient.ClientSecret --output text)\n"
            "  adding --profile NAME if your credentials are not the default profile.",
            file=sys.stderr,
        )
        return 2
    args.client_secret = secret

    try:
        tool_args = parse_tool_args(args.tool_args, args.arg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(hr("="))
    print("  Connecting to the campaign/deals MCP control plane as an external agent")
    print("  OAuth 2.0 client_credentials; no AWS credentials are used")
    print(hr("="))

    step(1, "Requesting a token from Cognito")
    try:
        payload = fetch_token(args.token_endpoint, args.client_id, args.client_secret, args.scopes)
    except urllib.error.HTTPError as exc:
        explain_http_error(exc)
        return 1
    except urllib.error.URLError as exc:
        print(f"      could not reach {args.token_endpoint}: {exc.reason}", file=sys.stderr)
        if "CERTIFICATE_VERIFY_FAILED" in str(exc.reason):
            print(
                "\n      This Python has no CA bundle it can verify Cognito's certificate with.\n"
                "      pip install certifi (or run the deploy venv's python, which has it), or\n"
                "      point SSL_CERT_FILE at a CA bundle. On a python.org macOS install, running\n"
                "      'Install Certificates.command' in its Applications folder also fixes it.",
                file=sys.stderr,
            )
        return 1
    token = payload.get("access_token")
    if not token:
        print(f"      no access_token in the response: {payload}", file=sys.stderr)
        return 1
    claims = decode_claims(token)
    print(f"      token obtained, expires in {payload.get('expires_in')}s")
    print(f"      client_id : {claims.get('client_id', '(absent)')}")
    print(f"      scope     : {claims.get('scope', '(absent)')}")
    print(f"      iss       : {claims.get('iss', '(absent)')}")
    if "custom:advertisers" not in claims:
        print(
            "      no custom:advertisers claim, as expected for a machine token. The runtime's\n"
            "      ARTF_CONTROL_CLIENT_ENTITLEMENTS map is what grants this client its principals."
        )

    region, region_source = resolve_region(args.region, args.runtime_arn)
    url = mcp_url(args.runtime_arn, region, args.qualifier)
    step(2, "Invoking AgentCore; its JWT authorizer validates the token")
    print(f"      region: {region} (from {region_source})")
    print(f"      {url[:96]}{'...' if len(url) > 96 else ''}")

    try:
        return asyncio.run(connect_and_list(url, token, args.call_tool, tool_args))
    except ImportError:
        print("error: the mcp package is required. pip install mcp", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        # The MCP client runs inside an anyio TaskGroup, so a failure arrives as an
        # ExceptionGroup whose own text says nothing. Report the leaves instead.
        leaves = _leaf_exceptions(exc)
        for leaf in leaves:
            print(f"      connection failed: {type(leaf).__name__}: {str(leaf)[:300]}", file=sys.stderr)
        message = " ".join(f"{type(leaf).__name__} {leaf}" for leaf in leaves)
        if "401" in message or "Unauthorized" in message:
            print(
                "\n      A 401 from AgentCore means its inbound authorizer rejected the token.\n"
                "      Confirm the runtime is configured for OAuth rather than IAM:\n"
                "        aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id ID \\\n"
                "          --region REGION --query authorizerConfiguration\n"
                "      An empty result means the runtime defaults to IAM SigV4 and never accepts\n"
                "      a bearer token.",
                file=sys.stderr,
            )
        elif "403" in message:
            print(
                "\n      A 403 usually means the client or scope is not in allowedClients /\n"
                "      allowedScopes on the runtime's authorizerConfiguration.",
                file=sys.stderr,
            )
        elif "404" in message:
            print(
                "\n      A 404 usually means the runtime ARN is wrong, or the qualifier does not\n"
                "      exist. List them: aws bedrock-agentcore-control list-agent-runtimes --region REGION",
                file=sys.stderr,
            )
        elif "timeout" in message.lower() or "ReadTimeout" in message:
            print(
                "\n      A timeout with a valid token usually means the CONTAINER is not serving.\n"
                "      The usual cause is a missing required environment variable (ARTF_CONTROL_TABLE,\n"
                "      ARTF_CONTROL_TOKEN_ISSUER, ARTF_CONTROL_DEAL_ORIGIN), which makes it exit at\n"
                "      startup. Check CloudWatch under /aws/bedrock-agentcore/runtimes/.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
