"""Create or update a Cognito User Pool + App Client for the ARTF demo.

Called by deploy.sh.  Outputs a JSON file with pool ID, client ID, and
domain that the frontend and orchestrator consume.

Usage:
    python3 scripts/deploy_cognito.py \
        --stack-name nvidia-artf-recommenders \
        --region us-east-1 \
        --cloudfront-domain <CLOUDFRONT_DOMAIN>
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


def _uid(stack_name: str, account_id: str, region: str) -> str:
    return hashlib.sha256(f"{stack_name}:{account_id}:{region}".encode()).hexdigest()[:8]


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


def deploy(*, stack_name: str, region: str, cloudfront_domain: str) -> dict:
    """Create or update Cognito User Pool and App Client."""
    cognito = boto3.client("cognito-idp", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    uid = _uid(stack_name, account_id, region)
    pool_name = f"{stack_name}-users"
    client_name = f"{stack_name}-frontend"

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

    outputs = {
        "UserPoolId": pool_id,
        "ClientId": client_id,
        "Region": region,
    }

    outputs_path = os.path.join(os.path.dirname(__file__), "..", ".cognito-outputs.json")
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    _LOG.info("Cognito outputs written to .cognito-outputs.json")
    _LOG.info("  Pool: %s  Client: %s  Region: %s", pool_id, client_id, region)
    return outputs


def destroy(*, stack_name: str, region: str) -> None:
    """Delete the Cognito User Pool."""
    cognito = boto3.client("cognito-idp", region_name=region)
    pool_name = f"{stack_name}-users"
    pool_id = _find_pool(cognito, pool_name)
    if pool_id:
        _LOG.info("Deleting Cognito User Pool: %s (%s)", pool_name, pool_id)
        cognito.delete_user_pool(UserPoolId=pool_id)
    else:
        _LOG.info("No Cognito pool found for %s", pool_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=["deploy", "destroy"])
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--cloudfront-domain", default="localhost")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.action == "deploy":
        deploy(stack_name=args.stack_name, region=args.region, cloudfront_domain=args.cloudfront_domain)
    else:
        destroy(stack_name=args.stack_name, region=args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
