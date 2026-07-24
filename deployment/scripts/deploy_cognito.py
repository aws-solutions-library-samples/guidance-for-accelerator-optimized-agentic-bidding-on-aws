"""Create or update Cognito auth for the ARTF demo.

Provisions three things and writes them to ``.cognito-outputs.json`` for the
frontend build (``deploy_frontend.py``) and the orchestrator to consume:

1. A **User Pool** + SPA **App Client** (interactive sign-in; ID token used as the
   bearer for the orchestrator API through CloudFront).
2. A **Cognito Identity Pool** federated to that User Pool. The browser exchanges
   the User Pool ID token for temporary SigV4 credentials so it can invoke the
   closed-loop AgentCore runtimes DIRECTLY (never through the orchestrator). See
   aidlc-docs/inception/requirements/agentcore-verification-findings.md.
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
        try:
            iam.delete_role_policy(RoleName=role_name, PolicyName=_INVOKE_POLICY_NAME)
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
        "--action", required=True, choices=["deploy", "destroy", "grant-agent-invoke"]
    )
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--cloudfront-domain", default="localhost")
    parser.add_argument("--adaptive-runtime-arn", default="")
    parser.add_argument("--governance-runtime-arn", default="")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.action == "deploy":
        deploy(
            stack_name=args.stack_name,
            region=args.region,
            cloudfront_domain=args.cloudfront_domain,
            adaptive_runtime_arn=args.adaptive_runtime_arn or None,
            governance_runtime_arn=args.governance_runtime_arn or None,
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
