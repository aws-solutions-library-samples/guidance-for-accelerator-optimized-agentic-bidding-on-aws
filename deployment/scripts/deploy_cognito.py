"""Create or update Cognito auth for the ARTF demo.

Provisions three things and writes them to ``.cognito-outputs.json`` for the
frontend build (``deploy_frontend.py``) and the orchestrator to consume:

1. A **User Pool** + SPA **App Client** (interactive sign-in; ID token used as the
   bearer for the orchestrator API through CloudFront).
2. A **Cognito Identity Pool** federated to that User Pool. The browser exchanges
   the User Pool ID token for temporary SigV4 credentials so it can invoke the
   closed-loop AgentCore runtimes DIRECTLY (never through the orchestrator), the
   same SigV4 inbound auth the EventBridge invokers use.
3. An **authenticated IAM role** attached to the Identity Pool. Its inline policy
   grants ``bedrock-agentcore:InvokeAgentRuntime`` scoped to the two closed-loop
   runtime ARNs (and their DEFAULT endpoint sub-resource) — least privilege, no
   wildcard principal or resource, and no ``InvokeAgentRuntimeForUser`` (the UI
   invokes with pool credentials, not on behalf of a user id).

Ordering note: the Identity Pool + role are created here (before the AgentCore
runtimes exist), but the InvokeAgentRuntime grant needs the runtime ARNs. So the
grant is applied via the idempotent ``grant-agent-invoke`` action, which
``deploy_closed_loop.sh`` calls AFTER it has the runtime ARNs. ``deploy`` will also
apply the grant inline if the ARNs are passed to it.

Usage:
    python3 scripts/deploy_cognito.py --action deploy \
        --stack-name nvidia-artf-recommenders \
        --region us-east-1 \
        --cloudfront-domain <CLOUDFRONT_DOMAIN>

    # After the AgentCore runtimes are deployed (idempotent, scoped grant):
    python3 scripts/deploy_cognito.py --action grant-agent-invoke \
        --stack-name nvidia-artf-recommenders --region us-east-1 \
        --adaptive-runtime-arn <ARN> --governance-runtime-arn <ARN>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

_LOG = logging.getLogger("deploy_cognito")

_OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), "..", ".cognito-outputs.json")


def _uid(stack_name: str, account_id: str, region: str) -> str:
    return hashlib.sha256(f"{stack_name}:{account_id}:{region}".encode()).hexdigest()[:8]


def _auth_role_name(stack_name: str) -> str:
    """Authenticated-role name for the Identity Pool (kept < 64 chars)."""
    name = f"{stack_name}-cl-auth-role"
    return name if len(name) <= 64 else f"{stack_name[:50]}-cl-auth-role"


_INVOKE_POLICY_NAME = "closed-loop-agent-invoke"
_CAPTION_POLICY_NAME = "caption-model-invoke"
_UI_API_POLICY_NAME = "ui-api-proxy-invoke"

# Every inline policy this script may attach to the authenticated role.
# ``destroy`` iterates this list: IAM refuses DeleteRole while any inline policy
# remains, and the delete_role call below only warns on failure, so a policy
# added without a matching entry here would leave an orphaned role behind
# silently. Adding to this list is the whole registration step.
_ROLE_INLINE_POLICIES = (_INVOKE_POLICY_NAME, _CAPTION_POLICY_NAME, _UI_API_POLICY_NAME)


# ---------------------------------------------------------------------------
# User Pool helpers
# ---------------------------------------------------------------------------


def _find_pool(client, pool_name: str) -> str | None:
    """Find existing user pool by name."""
    paginator = client.get_paginator("list_user_pools")
    for page in paginator.paginate(MaxResults=60):
        for pool in page["UserPools"]:
            if pool["Name"] == pool_name:
                return pool["Id"]
    return None


def _find_client(client, pool_id: str, client_name: str) -> str | None:
    """Find existing app client by name."""
    paginator = client.get_paginator("list_user_pool_clients")
    for page in paginator.paginate(UserPoolId=pool_id, MaxResults=60):
        for c in page["UserPoolClients"]:
            if c["ClientName"] == client_name:
                return c["ClientId"]
    return None


# ---------------------------------------------------------------------------
# Identity Pool + authenticated IAM role helpers
# ---------------------------------------------------------------------------


def _provider_name(region: str, user_pool_id: str) -> str:
    return f"cognito-idp.{region}.amazonaws.com/{user_pool_id}"


def _find_identity_pool(identity_client, pool_name: str) -> str | None:
    paginator = identity_client.get_paginator("list_identity_pools")
    for page in paginator.paginate(MaxResults=60):
        for p in page["IdentityPools"]:
            if p["IdentityPoolName"] == pool_name:
                return p["IdentityPoolId"]
    return None


def _ensure_identity_pool(
    identity_client, *, pool_name: str, region: str, user_pool_id: str, client_id: str
) -> str:
    """Create or update an Identity Pool federated to the User Pool app client."""
    providers = [
        {
            "ProviderName": _provider_name(region, user_pool_id),
            "ClientId": client_id,
            "ServerSideTokenCheck": True,
        }
    ]
    identity_pool_id = _find_identity_pool(identity_client, pool_name)
    if identity_pool_id is None:
        _LOG.info("Creating Cognito Identity Pool: %s", pool_name)
        resp = identity_client.create_identity_pool(
            IdentityPoolName=pool_name,
            AllowUnauthenticatedIdentities=False,
            CognitoIdentityProviders=providers,
        )
        identity_pool_id = resp["IdentityPoolId"]
        _LOG.info("  Created identity pool: %s", identity_pool_id)
    else:
        _LOG.info("Identity Pool exists: %s (%s)", pool_name, identity_pool_id)
        identity_client.update_identity_pool(
            IdentityPoolId=identity_pool_id,
            IdentityPoolName=pool_name,
            AllowUnauthenticatedIdentities=False,
            CognitoIdentityProviders=providers,
        )
    return identity_pool_id


def _ensure_auth_role(iam, *, role_name: str, identity_pool_id: str) -> str:
    """Create or repair the authenticated role trusted by the Identity Pool."""
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Federated": "cognito-identity.amazonaws.com"},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "cognito-identity.amazonaws.com:aud": identity_pool_id
                    },
                    "ForAnyValue:StringLike": {
                        "cognito-identity.amazonaws.com:amr": "authenticated"
                    },
                },
            }
        ],
    }
    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Closed-loop demo: authenticated Identity Pool role (direct AgentCore invocation via SigV4).",
            MaxSessionDuration=3600,
        )
        role_arn = resp["Role"]["Arn"]
        _LOG.info("Created authenticated role: %s", role_arn)
        # New IAM roles can be eventually consistent for set-identity-pool-roles.
        time.sleep(8)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        _LOG.info("Authenticated role exists: %s (updating trust policy)", role_name)
        iam.update_assume_role_policy(
            RoleName=role_name, PolicyDocument=json.dumps(trust_policy)
        )
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
    return role_arn


def _set_identity_pool_roles(identity_client, *, identity_pool_id: str, auth_role_arn: str) -> None:
    identity_client.set_identity_pool_roles(
        IdentityPoolId=identity_pool_id,
        Roles={"authenticated": auth_role_arn},
    )
    _LOG.info("Attached authenticated role to identity pool.")


def _invoke_resources(runtime_arns: list[str]) -> list[str]:
    """Expand each runtime ARN to the runtime + its endpoint sub-resource.

    InvokeAgentRuntime is authorized at BOTH the runtime and the endpoint level;
    granting only the runtime ARN is denied at the endpoint. Scoped to the specific
    runtimes (no wildcard resource).
    """
    resources: list[str] = []
    for arn in runtime_arns:
        arn = arn.strip()
        if not arn:
            continue
        resources.append(arn)
        resources.append(f"{arn}/runtime-endpoint/*")
    return resources


def _put_invoke_policy(iam, *, role_name: str, runtime_arns: list[str]) -> None:
    """Attach/replace the scoped InvokeAgentRuntime inline policy on the role."""
    resources = _invoke_resources(runtime_arns)
    if not resources:
        _LOG.warning("No runtime ARNs provided — skipping InvokeAgentRuntime grant.")
        return
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeClosedLoopRuntimes",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeAgentRuntime",
                "Resource": resources,
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=_INVOKE_POLICY_NAME,
        PolicyDocument=json.dumps(policy),
    )
    _LOG.info("Granted InvokeAgentRuntime on %d scoped resource(s).", len(resources))


def _put_caption_policy(iam, *, role_name: str) -> None:
    """Attach/replace the Bedrock invoke policy used by the Auction Theater captions.

    The document is deliberately STATIC -- no account id, no region, no model id --
    so this function makes no ``sts`` or ``bedrock`` calls and is region-independent.

    The wildcard resource is a deliberate exception to the project's
    least-privilege rule. It is what makes the ``global.`` cross-Region
    inference profile usable without the three-statement conditional policy pattern
    (profile ARN, in-Region model ARN, and the global model ARN with a
    ``aws:RequestedRegion: unspecified`` condition). Scoping it instead required
    deriving a geography prefix from the deployment region and verifying the profile
    at deploy time -- machinery whose only purpose was keeping a narrow grant correct
    on a demo stack.

    Blast radius: the browser role of a signed-in demo user can invoke any Bedrock
    model in the account, bounded by that account's Bedrock quotas. No data access,
    no write path, no other service.

    ``bedrock:Invoke*`` covers Converse, ConverseStream, InvokeModel and
    InvokeModelWithResponseStream, so moving the client to streaming later is a code
    change rather than an IAM change.
    """
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeCaptionModel",
                "Effect": "Allow",
                "Action": "bedrock:Invoke*",
                "Resource": "*",
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=_CAPTION_POLICY_NAME,
        PolicyDocument=json.dumps(policy),
    )
    _LOG.info("Granted bedrock:Invoke* for Auction Theater captions.")


def _put_ui_api_policy(iam, *, role_name: str, function_arn: str) -> None:
    """Attach/replace the lambda:InvokeFunction grant for the UI API proxy.

    The browser invokes ``<prefix>-ui-api-proxy`` directly with the Identity Pool
    credentials this role issues; the function forwards to the orchestrator's
    ClusterIP Service inside the VPC. One action, one function ARN (plus its
    qualified form for versions/aliases). The orchestrator still validates the
    Cognito bearer token the browser puts in the forwarded request, so this grant
    is the outer gate, not the only one. See private-ui-api-express.md FR-3.
    """
    if not function_arn or not function_arn.startswith("arn:aws:lambda:"):
        raise ValueError(f"--ui-api-proxy-arn must be a Lambda function ARN, got {function_arn!r}")
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeUiApiProxy",
                "Effect": "Allow",
                "Action": "lambda:InvokeFunction",
                "Resource": [function_arn, f"{function_arn}:*"],
            }
        ],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=_UI_API_POLICY_NAME,
        PolicyDocument=json.dumps(policy),
    )
    _LOG.info("Granted lambda:InvokeFunction on %s.", function_arn)


def _write_outputs(outputs: dict) -> None:
    with open(_OUTPUTS_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    _LOG.info("Cognito outputs written to .cognito-outputs.json")


def _read_outputs() -> dict:
    if os.path.exists(_OUTPUTS_PATH):
        with open(_OUTPUTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def deploy(
    *,
    stack_name: str,
    region: str,
    cloudfront_domain: str,
    adaptive_runtime_arn: str | None = None,
    governance_runtime_arn: str | None = None,
) -> dict:
    """Create/update the User Pool, App Client, Identity Pool, and auth role.

    If the runtime ARNs are supplied, the scoped InvokeAgentRuntime grant is applied
    inline; otherwise apply it later via ``grant-agent-invoke``.
    """
    cognito = boto3.client("cognito-idp", region_name=region)
    identity_client = boto3.client("cognito-identity", region_name=region)
    iam = boto3.client("iam", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    _ = _uid(stack_name, account_id, region)  # reserved for future disambiguation
    pool_name = f"{stack_name}-users"
    client_name = f"{stack_name}-frontend"
    identity_pool_name = f"{stack_name}-identity"

    # --- User Pool ---
    pool_id = _find_pool(cognito, pool_name)
    if pool_id is None:
        _LOG.info("Creating Cognito User Pool: %s", pool_name)
        resp = cognito.create_user_pool(
            PoolName=pool_name,
            AutoVerifiedAttributes=["email"],
            UsernameAttributes=["email"],
            Policies={
                "PasswordPolicy": {
                    "MinimumLength": 8,
                    "RequireUppercase": True,
                    "RequireLowercase": True,
                    "RequireNumbers": True,
                    "RequireSymbols": False,
                }
            },
            Schema=[
                {"Name": "email", "Required": True, "Mutable": True, "AttributeDataType": "String"},
            ],
            AdminCreateUserConfig={
                "AllowAdminCreateUserOnly": True,  # No self-signup — admin creates users
            },
        )
        pool_id = resp["UserPool"]["Id"]
        _LOG.info("  Created pool: %s", pool_id)
    else:
        _LOG.info("Cognito User Pool exists: %s (%s)", pool_name, pool_id)

    # --- App Client (SPA — no secret, ALLOW_USER_SRP_AUTH) ---
    client_id = _find_client(cognito, pool_id, client_name)
    callback_urls = [
        f"https://{cloudfront_domain}/",
        "http://localhost:5173/",  # local dev
    ]
    if client_id is None:
        _LOG.info("Creating App Client: %s", client_name)
        resp = cognito.create_user_pool_client(
            UserPoolId=pool_id,
            ClientName=client_name,
            GenerateSecret=False,
            ExplicitAuthFlows=[
                "ALLOW_USER_SRP_AUTH",
                "ALLOW_REFRESH_TOKEN_AUTH",
            ],
            SupportedIdentityProviders=["COGNITO"],
            CallbackURLs=callback_urls,
            LogoutURLs=callback_urls,
            AllowedOAuthFlows=["implicit"],
            AllowedOAuthScopes=["openid", "email", "profile"],
            AllowedOAuthFlowsUserPoolClient=True,
            PreventUserExistenceErrors="ENABLED",
            AccessTokenValidity=1,  # 1 hour
            IdTokenValidity=1,
            RefreshTokenValidity=30,  # 30 days
            TokenValidityUnits={
                "AccessToken": "hours",
                "IdToken": "hours",
                "RefreshToken": "days",
            },
        )
        client_id = resp["UserPoolClient"]["ClientId"]
        _LOG.info("  Created client: %s", client_id)
    else:
        _LOG.info("App Client exists: %s (%s)", client_name, client_id)
        # Update callback URLs in case CloudFront domain changed
        cognito.update_user_pool_client(
            UserPoolId=pool_id,
            ClientId=client_id,
            ClientName=client_name,
            ExplicitAuthFlows=["ALLOW_USER_SRP_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"],
            SupportedIdentityProviders=["COGNITO"],
            CallbackURLs=callback_urls,
            LogoutURLs=callback_urls,
            AllowedOAuthFlows=["implicit"],
            AllowedOAuthScopes=["openid", "email", "profile"],
            AllowedOAuthFlowsUserPoolClient=True,
            PreventUserExistenceErrors="ENABLED",
        )

    # --- Identity Pool + authenticated role (for SigV4 AgentCore invocation) ---
    identity_pool_id = _ensure_identity_pool(
        identity_client,
        pool_name=identity_pool_name,
        region=region,
        user_pool_id=pool_id,
        client_id=client_id,
    )
    role_name = _auth_role_name(stack_name)
    auth_role_arn = _ensure_auth_role(iam, role_name=role_name, identity_pool_id=identity_pool_id)
    _set_identity_pool_roles(
        identity_client, identity_pool_id=identity_pool_id, auth_role_arn=auth_role_arn
    )

    # Apply the scoped invoke grant now if we already know the runtime ARNs.
    arns = [a for a in (adaptive_runtime_arn, governance_runtime_arn) if a]
    if arns:
        _put_invoke_policy(iam, role_name=role_name, runtime_arns=arns)

    outputs = {
        "UserPoolId": pool_id,
        "ClientId": client_id,
        "Region": region,
        "IdentityPoolId": identity_pool_id,
        "AuthRoleArn": auth_role_arn,
        "AuthRoleName": role_name,
    }
    _write_outputs(outputs)
    _LOG.info(
        "  Pool: %s  Client: %s  IdentityPool: %s  Region: %s",
        pool_id, client_id, identity_pool_id, region,
    )
    return outputs


def grant_agent_invoke(
    *, stack_name: str, region: str, runtime_arns: list[str]
) -> dict:
    """Idempotently attach the scoped InvokeAgentRuntime policy to the auth role.

    Called after the AgentCore runtimes are deployed and their ARNs are known.
    """
    iam = boto3.client("iam", region_name=region)
    role_name = _auth_role_name(stack_name)
    # Verify the role exists (deploy must have run first).
    iam.get_role(RoleName=role_name)
    _put_invoke_policy(iam, role_name=role_name, runtime_arns=runtime_arns)
    outputs = _read_outputs()
    outputs["InvokeGrantedRuntimeArns"] = [a for a in runtime_arns if a]
    _write_outputs(outputs)
    return outputs


def grant_caption_invoke(*, stack_name: str, region: str) -> dict:
    """Idempotently attach the Bedrock caption-invoke policy to the auth role.

    Unlike ``grant_agent_invoke`` this has no dependency on the AgentCore runtimes,
    so it runs in the base deploy immediately after the role is created. Gating it
    behind ``--with-retraining`` would ship a theater whose captions never generate,
    with no error anywhere but the browser console.
    """
    iam = boto3.client("iam", region_name=region)
    role_name = _auth_role_name(stack_name)
    # Verify the role exists (deploy must have run first).
    iam.get_role(RoleName=role_name)
    _put_caption_policy(iam, role_name=role_name)
    outputs = _read_outputs()
    outputs["CaptionInvokeGranted"] = True
    _write_outputs(outputs)
    return outputs


def grant_ui_api_invoke(*, stack_name: str, region: str, function_arn: str) -> dict:
    """Idempotently attach the UI API proxy invoke policy to the auth role.

    Runs in the base deploy right after the proxy stack is created (Phase 3), so
    the Phase 4 frontend build ships against a role that can already call it.
    """
    iam = boto3.client("iam", region_name=region)
    role_name = _auth_role_name(stack_name)
    # Verify the role exists (deploy must have run first).
    iam.get_role(RoleName=role_name)
    _put_ui_api_policy(iam, role_name=role_name, function_arn=function_arn)
    outputs = _read_outputs()
    outputs["UiApiProxyArn"] = function_arn
    _write_outputs(outputs)
    return outputs


def destroy(*, stack_name: str, region: str) -> None:
    """Delete the Identity Pool, authenticated role, and User Pool (best effort)."""
    cognito = boto3.client("cognito-idp", region_name=region)
    identity_client = boto3.client("cognito-identity", region_name=region)
    iam = boto3.client("iam", region_name=region)

    # Identity Pool
    identity_pool_id = _find_identity_pool(identity_client, f"{stack_name}-identity")
    if identity_pool_id:
        _LOG.info("Deleting Identity Pool: %s", identity_pool_id)
        try:
            identity_client.delete_identity_pool(IdentityPoolId=identity_pool_id)
        except ClientError as exc:
            _LOG.warning("Could not delete identity pool: %s", exc)

    # Authenticated role (delete inline policy first)
    role_name = _auth_role_name(stack_name)
    try:
        # DeleteRole fails while ANY inline policy remains, and the delete_role
        # call below only warns, so every policy this script can attach must be
        # deleted here or the role is orphaned silently.
        for policy_name in _ROLE_INLINE_POLICIES:
            try:
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            except ClientError:
                pass
        iam.delete_role(RoleName=role_name)
        _LOG.info("Deleted authenticated role: %s", role_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            _LOG.warning("Could not delete role %s: %s", role_name, exc)

    # User Pool
    pool_name = f"{stack_name}-users"
    pool_id = _find_pool(cognito, pool_name)
    if pool_id:
        _LOG.info("Deleting Cognito User Pool: %s (%s)", pool_name, pool_id)
        cognito.delete_user_pool(UserPoolId=pool_id)
    else:
        _LOG.info("No Cognito pool found for %s", pool_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        required=True,
        choices=["deploy", "destroy", "grant-agent-invoke", "grant-caption-invoke", "grant-ui-api-invoke"],
    )
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE") or None,
                        help="AWS CLI profile for every call (default: AWS_PROFILE, else the SDK default chain)")
    parser.add_argument("--cloudfront-domain", default="localhost")
    parser.add_argument("--adaptive-runtime-arn", default="")
    parser.add_argument("--governance-runtime-arn", default="")
    parser.add_argument("--ui-api-proxy-arn", default="",
                        help="Lambda ARN of the UI API proxy (grant-ui-api-invoke)")
    args = parser.parse_args(argv)
    if args.profile:
        # One place for both credential paths: boto3 clients created below, and any
        # subprocess (aws/kubectl) that reads AWS_PROFILE from the environment.
        os.environ["AWS_PROFILE"] = args.profile
        boto3.setup_default_session(profile_name=args.profile, region_name=args.region)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.action == "deploy":
        deploy(
            stack_name=args.stack_name,
            region=args.region,
            cloudfront_domain=args.cloudfront_domain,
            adaptive_runtime_arn=args.adaptive_runtime_arn or None,
            governance_runtime_arn=args.governance_runtime_arn or None,
        )
    elif args.action == "grant-caption-invoke":
        grant_caption_invoke(stack_name=args.stack_name, region=args.region)
    elif args.action == "grant-ui-api-invoke":
        grant_ui_api_invoke(
            stack_name=args.stack_name, region=args.region, function_arn=args.ui_api_proxy_arn
        )
    elif args.action == "grant-agent-invoke":
        grant_agent_invoke(
            stack_name=args.stack_name,
            region=args.region,
            runtime_arns=[args.adaptive_runtime_arn, args.governance_runtime_arn],
        )
    else:
        destroy(stack_name=args.stack_name, region=args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
