"""Deploy the ARTF React testing frontend to S3 + CloudFront.

The deployment manages a single CloudFront distribution:
  - PRIMARY (Comment = "<stack-name>") → serves the React UI

Available actions:
  --action deploy    Deploy the React UI to the primary distribution.
  --action destroy   Disable the distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

_LOG = logging.getLogger("deploy_frontend")

# Distribution comment used to look up the existing CF distribution.
PRIMARY_COMMENT_SUFFIX = ""              # primary CF Comment is just the stack name


def _uid(stack_name: str, account_id: str, region: str) -> str:
    """Deterministic 8-char hex suffix unique to this stack+account+region."""
    return hashlib.sha256(f"{stack_name}:{account_id}:{region}".encode()).hexdigest()[:8]


# =============================================================================
# Shared helpers
# =============================================================================

def _ensure_bucket(s3, bucket_name: str, region: str) -> None:
    """Create bucket if missing and lock down public access."""
    _LOG.info("Ensuring S3 bucket: %s", bucket_name)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
    s3.put_public_access_block(Bucket=bucket_name, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
    })


def _ensure_oac(cf, oac_name: str) -> str:
    """Get or create an S3 Origin Access Control."""
    try:
        oac = cf.create_origin_access_control(OriginAccessControlConfig={
            "Name": oac_name, "OriginAccessControlOriginType": "s3",
            "SigningBehavior": "always", "SigningProtocol": "sigv4",
        })
        return oac["OriginAccessControl"]["Id"]
    except ClientError:
        oacs = cf.list_origin_access_controls()["OriginAccessControlList"]["Items"]
        return next(o["Id"] for o in oacs if o["Name"] == oac_name)


def _ensure_strip_api_function(cf, function_name: str) -> str:
    """Get or create+publish the CloudFront Function that strips the /api prefix."""
    code = """function handler(event) {
  var request = event.request;
  request.uri = request.uri.replace(/^\\/api/, '');
  if (request.uri === '') request.uri = '/';
  return request;
}"""
    try:
        existing = cf.describe_function(Name=function_name)
        return existing["FunctionSummary"]["FunctionMetadata"]["FunctionARN"]
    except ClientError as e:
        if "NoSuchFunctionExists" not in str(e):
            raise
        resp = cf.create_function(
            Name=function_name,
            FunctionConfig={"Comment": "Strip /api prefix for ALB origin", "Runtime": "cloudfront-js-2.0"},
            FunctionCode=code.encode(),
        )
        cf.publish_function(Name=function_name, IfMatch=resp["ETag"])
        return resp["FunctionSummary"]["FunctionMetadata"]["FunctionARN"]


def _api_cache_behavior(fn_arn: str) -> dict:
    return {
        "PathPattern": "/api/*",
        "TargetOriginId": "alb-api",
        "ViewerProtocolPolicy": "redirect-to-https",
        "AllowedMethods": {
            "Quantity": 7,
            "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
            "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
        },
        "Compress": False,
        "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",         # AWS managed: CachingDisabled
        "OriginRequestPolicyId": "216adef6-5c7f-47e4-b989-5492eafa07d3", # AWS managed: AllViewer
        "SmoothStreaming": False,
        "FieldLevelEncryptionId": "",
        "LambdaFunctionAssociations": {"Quantity": 0},
        "FunctionAssociations": {"Quantity": 1, "Items": [{
            "FunctionARN": fn_arn,
            "EventType": "viewer-request",
        }]},
    }


def _alb_origin(alb_host: str) -> dict:
    return {
        "Id": "alb-api",
        "DomainName": alb_host,
        "OriginPath": "",
        "CustomHeaders": {"Quantity": 0},
        "CustomOriginConfig": {
            "HTTPPort": 80, "HTTPSPort": 443,
            "OriginProtocolPolicy": "http-only",
            "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
            "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 30,
        },
    }


def _bucket_policy(bucket: str, dist_id: str, account_id: str) -> str:
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowCloudFrontServicePrincipal",
            "Effect": "Allow",
            "Principal": {"Service": "cloudfront.amazonaws.com"},
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{bucket}/*",
            "Condition": {"StringEquals": {"AWS:SourceArn": f"arn:aws:cloudfront::{account_id}:distribution/{dist_id}"}},
        }],
    })


def _build_and_upload_react(s3, bucket_name: str, react_dir: Path) -> None:
    """Run `npm run build` for the React app and upload `dist/` to the bucket."""
    _LOG.info("Building React frontend (%s)...", react_dir)
    r = subprocess.run(["npm", "run", "build"], cwd=str(react_dir), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"React build failed: {r.stderr}")
    dist_dir = react_dir / "dist"
    _LOG.info("React build complete; uploading to %s", bucket_name)
    for path in dist_dir.rglob("*"):
        if not path.is_file():
            continue
        key = str(path.relative_to(dist_dir))
        ct = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            ct = "text/javascript"
        elif path.suffix == ".css":
            ct = "text/css"
        s3.upload_file(str(path), bucket_name, key, ExtraArgs={"ContentType": ct})


def _find_distribution(cf, *, primary_comment: str, fallback_comments: tuple[str, ...] = ()) -> dict | None:
    """Locate an existing CloudFront distribution by Comment.

    Tries `primary_comment` first, then any `fallback_comments`. Returns the
    distribution summary or None.
    """
    dists = cf.list_distributions().get("DistributionList", {}).get("Items", []) or []
    for comment in (primary_comment, *fallback_comments):
        match = next((d for d in dists if d.get("Comment", "") == comment), None)
        if match:
            return match
    return None


def _ensure_distribution(
    *,
    cf, s3,
    canonical_comment: str,      # CloudFront distribution Comment to set/use
    fallback_comments: tuple[str, ...],
    bucket_name: str,            # S3 bucket that holds the UI assets
    s3_origin_id: str,           # logical Origin Id for the S3 origin
    oac_name: str,               # Origin Access Control name
    region: str,
    account_id: str,
    orchestrator_url: str,
    function_name: str,          # CloudFront Function name for /api stripping
) -> tuple[str, str]:
    """Create or update a CloudFront distribution. Returns (distribution_id, domain)."""

    alb_host = urlparse(orchestrator_url).hostname or "localhost"
    has_alb = alb_host != "localhost"
    s3_origin_domain = f"{bucket_name}.s3.{region}.amazonaws.com"

    origins = [{
        "Id": s3_origin_id,
        "DomainName": s3_origin_domain,
        "OriginPath": "",
        "CustomHeaders": {"Quantity": 0},
        "S3OriginConfig": {"OriginAccessIdentity": ""},
    }]
    if has_alb:
        origins.append(_alb_origin(alb_host))

    fn_arn = _ensure_strip_api_function(cf, function_name) if has_alb else None
    cache_behaviors = {"Quantity": 0, "Items": []}
    if has_alb:
        cache_behaviors = {"Quantity": 1, "Items": [_api_cache_behavior(fn_arn)]}

    existing = _find_distribution(cf, primary_comment=canonical_comment, fallback_comments=fallback_comments)

    oac_id = _ensure_oac(cf, oac_name)
    origins[0]["OriginAccessControlId"] = oac_id

    if existing:
        dist_id = existing["Id"]
        cf_domain = existing["DomainName"]
        _LOG.info("CF distribution exists (%s, %s) — updating origin -> %s",
                  canonical_comment, cf_domain, s3_origin_domain)

        config_resp = cf.get_distribution_config(Id=dist_id)
        etag = config_resp["ETag"]
        dc = config_resp["DistributionConfig"]

        # Normalise the Comment to the canonical value (idempotent).
        dc["Comment"] = canonical_comment
        # Replace origins outright so the S3 origin can change to a different bucket cleanly.
        dc["Origins"] = {"Quantity": len(origins), "Items": origins}
        dc["DefaultCacheBehavior"]["TargetOriginId"] = s3_origin_id
        dc["CacheBehaviors"] = cache_behaviors

        cf.update_distribution(Id=dist_id, DistributionConfig=dc, IfMatch=etag)
        cf.create_invalidation(DistributionId=dist_id, InvalidationBatch={
            "Paths": {"Quantity": 1, "Items": ["/*"]},
            "CallerReference": str(time.time()),
        })
    else:
        dist = cf.create_distribution(DistributionConfig={
            "Comment": canonical_comment,
            "Enabled": True,
            "DefaultRootObject": "index.html",
            "Origins": {"Quantity": len(origins), "Items": origins},
            "DefaultCacheBehavior": {
                "TargetOriginId": s3_origin_id,
                "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"],
                                   "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
                "ForwardedValues": {"QueryString": False, "Cookies": {"Forward": "none"}},
                "Compress": True,
                "MinTTL": 0, "DefaultTTL": 86400, "MaxTTL": 31536000,
            },
            "CacheBehaviors": cache_behaviors,
            "CallerReference": str(time.time()),
        })
        dist_id = dist["Distribution"]["Id"]
        cf_domain = dist["Distribution"]["DomainName"]
        _LOG.info("Created CF distribution (%s): %s", canonical_comment, cf_domain)

    s3.put_bucket_policy(Bucket=bucket_name, Policy=_bucket_policy(bucket_name, dist_id, account_id))
    return dist_id, cf_domain


# =============================================================================
# Public actions
# =============================================================================

def deploy(*, stack_name: str, region: str, orchestrator_url: str) -> dict:
    """Deploy the React UI to the primary CloudFront distribution."""
    s3 = boto3.client("s3", region_name=region)
    cf = boto3.client("cloudfront", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    uid = _uid(stack_name, account_id, region)
    bucket_name = f"{stack_name}-frontend-{uid}"
    react_dir = Path(__file__).parent.parent.parent / "source" / "frontend-react"

    _ensure_bucket(s3, bucket_name, region)
    _build_and_upload_react(s3, bucket_name, react_dir)

    dist_id, cf_domain = _ensure_distribution(
        cf=cf, s3=s3,
        canonical_comment=stack_name,
        fallback_comments=(),
        bucket_name=bucket_name,
        s3_origin_id="s3-frontend",
        oac_name=f"{stack_name}-oac-{uid}",
        region=region, account_id=account_id,
        orchestrator_url=orchestrator_url,
        function_name=f"{stack_name}-strip-api-prefix",
    )

    outputs = {"CloudFrontDomain": cf_domain, "DistributionId": dist_id, "BucketName": bucket_name}
    outputs_path = os.path.join(os.path.dirname(__file__), "..", ".frontend-outputs.json")
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    _LOG.info("React Frontend (primary): https://%s", cf_domain)
    return outputs


def deploy_vanilla(*, stack_name: str, region: str, orchestrator_url: str) -> dict:
    """Deploy the vanilla UI to the secondary CloudFront distribution."""
    s3 = boto3.client("s3", region_name=region)
    cf = boto3.client("cloudfront", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    uid = _uid(stack_name, account_id, region)
    bucket_name = f"{stack_name}-vanilla-{uid}"
    frontend_dir = Path(__file__).parent.parent / "frontend"

    _ensure_bucket(s3, bucket_name, region)
    _upload_vanilla(s3, bucket_name, frontend_dir)

    canonical_comment = f"{stack_name}{SECONDARY_COMMENT_SUFFIX}"
    fallback_comments = tuple(f"{stack_name}{s}" for s in SECONDARY_LEGACY_COMMENT_SUFFIXES)

    dist_id, cf_domain = _ensure_distribution(
        cf=cf, s3=s3,
        canonical_comment=canonical_comment,
        fallback_comments=fallback_comments,
        bucket_name=bucket_name,
        s3_origin_id="s3-vanilla",
        oac_name=f"{stack_name}-vanilla-oac-{uid}",
        region=region, account_id=account_id,
        orchestrator_url=orchestrator_url,
        function_name=f"{stack_name}-vanilla-strip-api",
    )

    outputs = {"CloudFrontDomain": cf_domain, "DistributionId": dist_id, "BucketName": bucket_name}
    outputs_path = os.path.join(os.path.dirname(__file__), "..", ".frontend-vanilla-outputs.json")
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    _LOG.info("Vanilla Frontend (secondary): https://%s", cf_domain)
    return outputs


def deploy_vanilla(*, stack_name: str, region: str, orchestrator_url: str) -> dict:
    """Deploy the vanilla UI to the secondary CloudFront distribution."""
    s3 = boto3.client("s3", region_name=region)
    cf = boto3.client("cloudfront", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    uid = _uid(stack_name, account_id, region)
    bucket_name = f"{stack_name}-vanilla-{uid}"
    frontend_dir = Path(__file__).parent.parent / "frontend"

    _ensure_bucket(s3, bucket_name, region)
    _upload_vanilla(s3, bucket_name, frontend_dir)

    canonical_comment = f"{stack_name}{SECONDARY_COMMENT_SUFFIX}"
    fallback_comments = tuple(f"{stack_name}{s}" for s in SECONDARY_LEGACY_COMMENT_SUFFIXES)

    dist_id, cf_domain = _ensure_distribution(
        cf=cf, s3=s3,
        canonical_comment=canonical_comment,
        fallback_comments=fallback_comments,
        bucket_name=bucket_name,
        s3_origin_id="s3-vanilla",
        oac_name=f"{stack_name}-vanilla-oac-{uid}",
        region=region, account_id=account_id,
        orchestrator_url=orchestrator_url,
        function_name=f"{stack_name}-vanilla-strip-api",
    )

    outputs = {"CloudFrontDomain": cf_domain, "DistributionId": dist_id, "BucketName": bucket_name}
    outputs_path = os.path.join(os.path.dirname(__file__), "..", ".frontend-vanilla-outputs.json")
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    _LOG.info("Vanilla Frontend (secondary): https://%s", cf_domain)
    return outputs


def destroy(*, stack_name: str, region: str) -> None:
    cf = boto3.client("cloudfront", region_name=region)
    _LOG.info("Disabling CloudFront distribution for %s", stack_name)
    targets = {stack_name}
    dists = cf.list_distributions().get("DistributionList", {}).get("Items", [])
    for d in dists or []:
        if d.get("Comment", "") in targets and d.get("Enabled"):
            _LOG.info("  Disabling distribution %s (%s)", d["Id"], d.get("Comment"))
            config = cf.get_distribution_config(Id=d["Id"])
            etag = config["ETag"]
            dc = config["DistributionConfig"]
            dc["Enabled"] = False
            cf.update_distribution(Id=d["Id"], DistributionConfig=dc, IfMatch=etag)
            _LOG.info("  Distribution disabled. Delete manually once status is Deployed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=["deploy", "destroy"],
        help="deploy: React UI -> primary CF; destroy: disable the distribution.",
    )
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--orchestrator-url", default="http://localhost:8080")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.action == "deploy":
        deploy(stack_name=args.stack_name, region=args.region, orchestrator_url=args.orchestrator_url)
    else:
        destroy(stack_name=args.stack_name, region=args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
