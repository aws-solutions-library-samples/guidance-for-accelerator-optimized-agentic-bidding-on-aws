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


# Name of the CloudFront Function earlier releases attached to a "/api/*" cache
# behaviour in front of the orchestrator load balancer. The orchestrator now has
# no public address (the UI invokes the ui-api-proxy Lambda instead), so the
# behaviour is gone and a leftover function is deleted on update.
def _legacy_strip_api_function_name(stack_name: str) -> str:
    return f"{stack_name}-strip-api-prefix"


def _delete_function_if_present(cf, function_name: str) -> None:
    try:
        desc = cf.describe_function(Name=function_name)
    except ClientError as e:
        if "NoSuchFunctionExists" in str(e):
            return
        raise
    try:
        cf.delete_function(Name=function_name, IfMatch=desc["ETag"])
        _LOG.info("Deleted legacy CloudFront Function %s (no longer referenced)", function_name)
    except ClientError as e:
        # FunctionInUse means a distribution still references it; the update
        # that removes the reference has to propagate first. Harmless to leave.
        _LOG.warning("Could not delete CloudFront Function %s yet: %s", function_name, e)


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
    # Ensure dependencies (incl. vite) are installed before building.
    if not (react_dir / "node_modules").is_dir():
        install_cmd = ["npm", "ci"] if (react_dir / "package-lock.json").is_file() else ["npm", "install"]
        _LOG.info("Installing React dependencies (%s)...", " ".join(install_cmd))
        ri = subprocess.run(install_cmd, cwd=str(react_dir), capture_output=True, text=True)
        if ri.returncode != 0:
            raise RuntimeError(f"npm install failed: {ri.stderr}")

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
        s3.upload_file(
            str(path), bucket_name, key,
            ExtraArgs={"ContentType": ct, "CacheControl": _cache_control(key)},
        )

    _prune_stale(s3, bucket_name, dist_dir)


#: Prefixes pruned after upload. Only PATH-STABLE prefixes belong here.
#
# `assets/` is deliberately NOT pruned. Vite content-hashes those filenames, so
# every past deploy's bundle is still present and still reachable by the
# index.html that named it -- a client holding a cached index.html (which is
# served no-cache but may still be in flight) would 404 on its own bundle if the
# old asset were removed.
#
# `samples/` is the opposite case: the filename is the scenario's identity, so a
# removed scenario leaves a payload behind at a stable URL, and that file goes on
# being served as if the app still ships it. It also contradicts
# test_scenario_card_fixture_wiring's no-orphaned-payloads assertion, which can
# only see the working tree.
_PRUNED_PREFIXES = ("samples/",)


def _prune_stale(s3, bucket_name: str, dist_dir: Path) -> None:
    """Delete objects under _PRUNED_PREFIXES that the new build does not contain."""
    built = {
        str(p.relative_to(dist_dir))
        for p in dist_dir.rglob("*")
        if p.is_file()
    }

    stale: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for prefix in _PRUNED_PREFIXES:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                if obj["Key"] not in built:
                    stale.append({"Key": obj["Key"]})

    if not stale:
        return

    _LOG.info("Removing %d object(s) the build no longer contains: %s",
              len(stale), ", ".join(o["Key"] for o in stale))
    # Batched, and scoped to the keys listed above -- never a prefix wildcard.
    for i in range(0, len(stale), 1000):
        s3.delete_objects(
            Bucket=bucket_name, Delete={"Objects": stale[i:i + 1000], "Quiet": True}
        )


def _cache_control(key: str) -> str:
    """Cache-Control for a dist object.

    index.html must revalidate on every load. It is the only file that names the
    content-hashed bundle, so a browser holding a cached copy keeps loading the
    previous bundle after a deploy — a CloudFront invalidation clears the edge
    but not the client. Vite emits `assets/*` with a content hash in the
    filename, so those are safe to pin immutable.
    """
    if key == "index.html" or key.endswith("/index.html"):
        return "no-cache, must-revalidate"
    if key.startswith("assets/"):
        return "public, max-age=31536000, immutable"
    return "public, max-age=3600"


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
) -> tuple[str, str]:
    """Create or update a static-only CloudFront distribution. Returns (distribution_id, domain).

    The distribution has exactly one origin, the S3 bucket. API traffic never
    touches CloudFront: the browser invokes the ui-api-proxy Lambda directly
    (see source/frontend-react/src/authFetch.js), so there is no "/api/*"
    behaviour and no second origin. On update the origins and cache behaviours
    are replaced outright, which also strips the ALB origin earlier releases
    configured.
    """
    s3_origin_domain = f"{bucket_name}.s3.{region}.amazonaws.com"

    origins = [{
        "Id": s3_origin_id,
        "DomainName": s3_origin_domain,
        "OriginPath": "",
        "CustomHeaders": {"Quantity": 0},
        "S3OriginConfig": {"OriginAccessIdentity": ""},
    }]
    cache_behaviors = {"Quantity": 0, "Items": []}

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

    # Invalidate the cache on every deploy so freshly uploaded assets are served
    # immediately instead of from CloudFront's edge cache. Runs for both the
    # create and update paths (harmless on a brand-new distribution).
    try:
        inv = cf.create_invalidation(
            DistributionId=dist_id,
            InvalidationBatch={
                "Paths": {"Quantity": 1, "Items": ["/*"]},
                "CallerReference": str(time.time()),
            },
        )
        _LOG.info("Created CloudFront invalidation %s (/*) on %s",
                  inv["Invalidation"]["Id"], dist_id)
    except ClientError as e:
        _LOG.warning("CloudFront invalidation failed on %s: %s — assets may serve stale until TTL", dist_id, e)

    return dist_id, cf_domain


# =============================================================================
# Public actions
# =============================================================================

def deploy(*, stack_name: str, region: str) -> dict:
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
    )
    _delete_function_if_present(cf, _legacy_strip_api_function_name(stack_name))

    outputs = {"CloudFrontDomain": cf_domain, "DistributionId": dist_id, "BucketName": bucket_name}
    outputs_path = os.path.join(os.path.dirname(__file__), "..", ".frontend-outputs.json")
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    _LOG.info("React Frontend (primary): https://%s", cf_domain)
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
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE") or None,
                        help="AWS CLI profile for every call (default: AWS_PROFILE, else the SDK default chain)")
    args = parser.parse_args(argv)
    if args.profile:
        # One place for both credential paths: boto3 clients created below, and any
        # subprocess (aws/kubectl) that reads AWS_PROFILE from the environment.
        os.environ["AWS_PROFILE"] = args.profile
        boto3.setup_default_session(profile_name=args.profile, region_name=args.region)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.action == "deploy":
        deploy(stack_name=args.stack_name, region=args.region)
    else:
        destroy(stack_name=args.stack_name, region=args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
