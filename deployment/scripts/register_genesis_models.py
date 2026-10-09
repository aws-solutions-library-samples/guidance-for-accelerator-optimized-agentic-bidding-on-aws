"""Idempotently register the "genesis" (v1, unretrained-starter) model version
in each SageMaker Model Package Group.

closed_loop_cfn.yaml creates the Model Package Groups (artf-dlrm-bid-shader /
artf-ncf-deal-manager / artf-deal-yield-manager-floor /
artf-deal-yield-manager-margin) empty. Nothing else ever calls
CreateModelPackage, so without this script the registry starts with zero
versions and TrainingJobConfig.base_model_version (a required field, no
default) has nothing real to reference for the very first retraining job.

This script registers the ONNX artifact already uploaded by deploy.sh Step 3a
(s3://<bucket>/onnx-source/<model>/model.onnx - the same seeded-weights export
the ARTF containers use for inference today) as ModelPackageVersion 1,
ModelApprovalStatus=Approved (it IS the version currently serving production
traffic via Triton). The registration is tagged genesis=true in
CustomerMetadataProperties so it is honestly distinguishable from an actually
trained version - this is NOT a trained result and must never be presented as
one.

Idempotent: if a Model Package Group already has any registered version, that
group is skipped (re-running deploy never duplicates or overwrites a version).

ncf_deal_manager is the exception. Its ONNX is no longer seeded weights: deploy.sh
Step 2 trains it on sample interactions generated from the deal book and writes a
manifest.json beside it whose fingerprint identifies that build. A new ncf version
is registered whenever the most recent version that records a `fingerprint` does
not record this one (or no version records one), with CustomerMetadataProperties
training_data, label (the manifest's target), generator, seed (the generator seed)
and fingerprint (12 hex) copied from the manifest. The same fingerprint is never
registered twice. Its ModelDataUrl is the build's own copy,
onnx-source/ncf_deal_manager/builds/<fingerprint>/model.onnx, which only a build
with that fingerprint writes; onnx-source/ncf_deal_manager/model.onnx is
overwritten by every build, so a version pointing there would name a later
build's bytes.

--only <model_type> limits a run to the named model types; deploy.sh uses
`--only ncf_deal_manager` right after Step 3a uploads a new build.

This does NOT emit a ModelPackageGroupChanged event - genesis registration is
not a governance/promotion action; there is no prior version to A/B test
against yet.

Usage:
    python3 scripts/register_genesis_models.py \
        --model-bucket dv2-triton-models-abc123 \
        --region us-east-1 \
        --dlrm-package-group dv2-artf-dlrm-bid-shader \
        --ncf-package-group dv2-artf-ncf-deal-manager \
        --yield-floor-package-group dv2-artf-deal-yield-manager-floor \
        --yield-margin-package-group dv2-artf-deal-yield-manager-margin \
        --training-image-repository artf-nemo-rl-training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import boto3
from botocore.exceptions import ClientError

_LOG = logging.getLogger("register_genesis_models")

# model_type -> (Model Package Group CLI flag value, ECR image tag pushed by
# deploy_closed_loop.sh Step 4b). Must match RECOMMENDER_MODELS in deploy.sh
# and ModelType in source/training/pipeline.py.
#
# deal_yield_manager_floor / deal_yield_manager_margin are trained via a
# SEPARATE path (XGBoost via source/training/xgboost_pipeline.py, not
# NeMo-RL) but their genesis artifacts are registered through this SAME
# script/convention -- each genesis seed is converted to ONNX at export time
# (onnx-source/deal_yield_manager_floor/model.onnx,
# onnx-source/deal_yield_manager_margin/model.onnx) purely for Model
# Registry bookkeeping consistency across all model types. Triton's FIL
# backend does NOT read this ONNX form -- it loads a separate, native
# XGBoost JSON artifact staged directly to each model's own Triton model
# repository path (triton-models/deal_yield_manager_floor/1/xgboost.json,
# .../deal_yield_manager_margin/1/xgboost.json), which this script does not
# touch. Two single-output models rather than one two-output model because
# Triton's FIL backend does not support multi-output regression; see the FIL
# correction note for why this is two independent model types, not one.
_MODEL_TYPES: list[tuple[str, str]] = [
    ("dlrm_bid_shader", "dlrm"),
    ("ncf_deal_manager", "ncf"),
    ("deal_yield_manager_floor", "yield-floor"),
    ("deal_yield_manager_margin", "yield-margin"),
]

# model_type values that use SageMaker's built-in XGBoost algorithm image
# (not this project's own NeMo-RL ECR image).
_XGBOOST_MODEL_TYPES = frozenset({"deal_yield_manager_floor", "deal_yield_manager_margin"})

# SageMaker's CustomerMetadataProperties values must match
# ([\p{L}\p{Z}\p{N}_.:\/=+\-@]*){1,256} - no semicolons, commas, or
# parentheses. Keep this note plain prose within that character set.
_GENESIS_METADATA_NOTE = (
    "Unretrained starter model - seeded weights exported from the same "
    "source used by the ARTF Triton containers at inference time. This is "
    "NOT a trained result - it exists so the first retraining job has a real "
    "base model version to fine-tune from."
)

# The Deal Scorer's build, registered per fingerprint rather than once.
_NCF_MODEL_TYPE = "ncf_deal_manager"
_NCF_MANIFEST_KEY = f"onnx-source/{_NCF_MODEL_TYPE}/manifest.json"
# deploy.sh upload_ncf_source also keeps each build under <prefix>/<fingerprint>/.
_NCF_BUILDS_PREFIX = f"onnx-source/{_NCF_MODEL_TYPE}/builds"
_NCF_SAMPLE_DESCRIPTION = (
    "Deal Scorer trained on sample interactions generated from the deal book, "
    "not observed outcomes"
)
# CustomerMetadataProperties value charset (see the note above) and the 12-hex
# fingerprint form the registry records.
_METADATA_VALUE_RE = re.compile(r"^[\w .:/=+\-@]{1,256}$")
_FP12_RE = re.compile(r"^[0-9a-f]{12}$")


class ManifestError(ValueError):
    """The ncf manifest is missing a field the registration needs, or it is malformed."""


def _ncf_build_identity(s3_client, bucket: str) -> dict[str, str] | None:
    """training_data, label, generator, seed and fingerprint (12 hex) from the ncf manifest in S3.

    label is the manifest's ``target``, generator its ``generator.path`` and seed its
    ``generator_seed``. None when there is no manifest. Raises ManifestError when the
    manifest cannot supply the five values within the metadata charset, and
    ClientError for any S3 error other than a missing object.
    """
    try:
        body = s3_client.get_object(Bucket=bucket, Key=_NCF_MANIFEST_KEY)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code", "") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    try:
        manifest = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"s3://{bucket}/{_NCF_MANIFEST_KEY} is not JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"s3://{bucket}/{_NCF_MANIFEST_KEY} is not a JSON object")
    fp12 = manifest.get("fingerprint12")
    if not fp12 and isinstance(manifest.get("fingerprint"), str):
        fp12 = manifest["fingerprint"][:12]
    if not isinstance(fp12, str) or not _FP12_RE.match(fp12):
        raise ManifestError(f"{_NCF_MANIFEST_KEY} has no 12-hex fingerprint (got {fp12!r})")
    generator = manifest.get("generator")
    seed = manifest.get("generator_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ManifestError(f"{_NCF_MANIFEST_KEY} field generator_seed={seed!r} is missing or not an integer")
    identity = {
        "training_data": manifest.get("training_data"),
        "label": manifest.get("target"),
        "generator": generator.get("path") if isinstance(generator, dict) else None,
        "seed": str(seed),
        "fingerprint": fp12,
    }
    sources = {"training_data": "training_data", "label": "target", "generator": "generator.path"}
    for field, source in sources.items():
        value = identity[field]
        if not isinstance(value, str) or not _METADATA_VALUE_RE.match(value):
            raise ManifestError(f"{_NCF_MANIFEST_KEY} field {source}={value!r} is missing or outside the metadata charset")
    return identity


def _latest_recorded_fingerprint(sagemaker_client, model_package_group: str) -> str | None:
    """The `fingerprint` metadata of the newest version that records one, or None.

    Versions without it (the seeded-weights genesis, retrained versions registered by
    the governance pipeline) are passed over, so a retrained version on top does not
    make the same build register again.
    """
    kwargs = {
        "ModelPackageGroupName": model_package_group,
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
        "MaxResults": 100,
    }
    while True:
        response = sagemaker_client.list_model_packages(**kwargs)
        for summary in response.get("ModelPackageSummaryList", []):
            detail = sagemaker_client.describe_model_package(
                ModelPackageName=summary["ModelPackageArn"]
            )
            recorded = (detail.get("CustomerMetadataProperties") or {}).get("fingerprint")
            if recorded:
                return recorded
        token = response.get("NextToken")
        if not token:
            return None
        kwargs["NextToken"] = token


def _resolve_account_id(sts_client) -> str:
    return sts_client.get_caller_identity()["Account"]


def _has_existing_version(sagemaker_client, model_package_group: str) -> bool:
    """Return True if the group already has at least one registered version."""
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group,
        MaxResults=1,
    )
    return len(response.get("ModelPackageSummaryList", [])) > 0


def _onnx_artifact_exists(s3_client, bucket: str, key: str) -> bool:
    """Verify the genesis ONNX artifact was actually uploaded before claiming
    a real registration. No fabricated success - if the artifact is missing,
    the caller must skip this model type and report it honestly.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _training_image_exists(ecr_client, repository: str, tag: str) -> bool:
    """Verify the training image tag this registration will reference is
    actually present in ECR. The NeMo-RL training image builds asynchronously
    (CodeBuild, 15-50 min) and deploy_closed_loop.sh does not block on it, so
    on a fresh deploy this image usually does not exist yet when genesis
    registration first runs. CreateModelPackage validates the referenced
    image and fails otherwise - checking first lets the caller skip honestly
    instead of attempting (and failing) a doomed API call.
    """
    try:
        ecr_client.describe_images(repositoryName=repository, imageIds=[{"imageTag": tag}])
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in ("ImageNotFoundException", "RepositoryNotFoundException"):
            return False
        raise


def _resolve_inference_image_uri(
    model_type: str, *, account_id: str, region: str, training_image_repository: str, image_tag: str
) -> str:
    """Resolves the InferenceSpecification container image for this model
    type's genesis registration.

    deal_yield_manager_floor / deal_yield_manager_margin are trained via a
    separate path (XGBoost, see source/training/xgboost_pipeline.py) that
    does NOT build a custom image into this project's own NeMo-RL ECR repo
    -- they use SageMaker's first-party built-in XGBoost algorithm
    container instead, which always exists and is never subject to the "is
    our own async CodeBuild image ready yet" check that applies to
    dlrm_bid_shader/ncf_deal_manager.
    """
    if model_type in _XGBOOST_MODEL_TYPES:
        try:
            from sagemaker import image_uris  # type: ignore

            return image_uris.retrieve(framework="xgboost", region=region, version="1.7-1")
        except ImportError as exc:
            # SageMaker's built-in XGBoost algorithm image is hosted in an
            # AWS-owned account that differs per region (NOT this caller's
            # own account_id, and not a value safe to guess/hardcode here).
            # Fail honestly rather than fabricate a URI that would silently
            # point at the wrong account.
            raise RuntimeError(
                f"The 'sagemaker' SDK is required to resolve the built-in "
                f"XGBoost training image URI for {model_type} "
                f"(sagemaker.image_uris.retrieve). Install it with "
                f"'pip install sagemaker' and re-run this script."
            ) from exc

    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/{training_image_repository}:{image_tag}"


def register_genesis_models(
    *,
    model_bucket: str,
    region: str,
    package_groups: dict[str, str],
    training_image_repository: str,
    account_id: str | None = None,
    sagemaker_client=None,
    s3_client=None,
    sts_client=None,
    ecr_client=None,
    model_types=None,
) -> dict[str, str]:
    """Register the genesis version for each model type that has none yet, and a
    new ncf_deal_manager version for each new build fingerprint.

    Args:
        model_bucket: S3 bucket holding onnx-source/<model>/model.onnx.
        region: AWS region.
        package_groups: model_type -> Model Package Group name.
        training_image_repository: ECR repo name for the training image
            (used as the InferenceSpecification container image reference,
            mirroring TrainingPipeline.register_model's pattern). Each model
            type's own tag (from _MODEL_TYPES, e.g. "dlrm"/"ncf") is checked
            for existence before registering that model type.
        account_id: AWS account id. Resolved via STS if not supplied.
        sagemaker_client / s3_client / sts_client / ecr_client: optional
            pre-built boto3 clients (for testing).
        model_types: model types to process; None processes all of them.

    Returns:
        Dict of model_type -> outcome string ("registered:<arn>",
        "skipped:exists", "skipped:missing-artifact",
        "skipped:missing-manifest", "skipped:invalid-manifest", or
        "skipped:training-image-not-ready").
    """
    sagemaker_client = sagemaker_client or boto3.client("sagemaker", region_name=region)
    s3_client = s3_client or boto3.client("s3", region_name=region)
    sts_client = sts_client or boto3.client("sts", region_name=region)
    ecr_client = ecr_client or boto3.client("ecr", region_name=region)

    if account_id is None:
        account_id = _resolve_account_id(sts_client)

    selected = None if model_types is None else set(model_types)
    results: dict[str, str] = {}

    for model_type, image_tag in _MODEL_TYPES:
        if selected is not None and model_type not in selected:
            continue
        package_group = package_groups.get(model_type)
        if not package_group:
            _LOG.warning("No Model Package Group configured for %s - skipping.", model_type)
            results[model_type] = "skipped:no-package-group"
            continue

        ncf_identity = None
        if model_type == _NCF_MODEL_TYPE:
            try:
                ncf_identity = _ncf_build_identity(s3_client, model_bucket)
            except ManifestError as exc:
                _LOG.error("  %s: %s - not registering.", model_type, exc)
                results[model_type] = "skipped:invalid-manifest"
                continue
            if ncf_identity is None:
                _LOG.error(
                    "  %s: no manifest at s3://%s/%s - did deploy.sh Step 3a run? "
                    "Not registering a version whose training data cannot be stated.",
                    model_type,
                    model_bucket,
                    _NCF_MANIFEST_KEY,
                )
                results[model_type] = "skipped:missing-manifest"
                continue
            recorded = _latest_recorded_fingerprint(sagemaker_client, package_group)
            if recorded == ncf_identity["fingerprint"]:
                _LOG.info(
                    "  %s (%s): fingerprint %s is already registered - skipping.",
                    model_type,
                    package_group,
                    recorded,
                )
                results[model_type] = "skipped:exists"
                continue
            _LOG.info(
                "  %s (%s): build fingerprint %s, last registered %s - registering.",
                model_type,
                package_group,
                ncf_identity["fingerprint"],
                recorded or "none",
            )
        elif _has_existing_version(sagemaker_client, package_group):
            _LOG.info(
                "  %s (%s): already has a registered version - skipping (idempotent).",
                model_type,
                package_group,
            )
            results[model_type] = "skipped:exists"
            continue

        if ncf_identity is not None:
            # The build's own copy: the shared key is overwritten by the next build.
            onnx_key = f"{_NCF_BUILDS_PREFIX}/{ncf_identity['fingerprint']}/model.onnx"
        else:
            onnx_key = f"onnx-source/{model_type}/model.onnx"
        if not _onnx_artifact_exists(s3_client, model_bucket, onnx_key):
            _LOG.error(
                "  %s: genesis ONNX artifact not found at s3://%s/%s - "
                "did deploy.sh Step 3a run? Skipping (not registering a "
                "fabricated version).",
                model_type,
                model_bucket,
                onnx_key,
            )
            results[model_type] = "skipped:missing-artifact"
            continue

        # deal_yield_manager_floor/margin use SageMaker's first-party
        # built-in XGBoost image (always exists) rather than this project's
        # own async-built NeMo-RL ECR image -- the "is our CodeBuild image
        # ready yet" check only applies to the NeMo-RL-trained model types.
        if model_type not in _XGBOOST_MODEL_TYPES and not _training_image_exists(
            ecr_client, training_image_repository, image_tag
        ):
            _LOG.warning(
                "  %s: training image %s:%s not found in ECR yet - the NeMo-RL "
                "training container builds asynchronously (15-50 min) and this "
                "registration references it as the InferenceSpecification "
                "container. Skipping honestly (not attempting a call that would "
                "fail validation). Re-run this script (or ./check_builds.sh) once "
                "the training image has finished building.",
                model_type,
                training_image_repository,
                image_tag,
            )
            results[model_type] = "skipped:training-image-not-ready"
            continue

        model_data_url = f"s3://{model_bucket}/{onnx_key}"
        training_image_uri = _resolve_inference_image_uri(
            model_type,
            account_id=account_id,
            region=region,
            training_image_repository=training_image_repository,
            image_tag=image_tag,
        )

        create_params = {
            "ModelPackageGroupName": package_group,
            "ModelPackageDescription": (
                f"Genesis (v1) unretrained starter model for {model_type} - "
                f"seeded weights exported at deploy time, not a trained result."
            ),
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": training_image_uri,
                        "ModelDataUrl": model_data_url,
                    }
                ],
                "SupportedContentTypes": ["application/octet-stream"],
                "SupportedResponseMIMETypes": ["application/octet-stream"],
                "SupportedRealtimeInferenceInstanceTypes": [
                    "ml.g5.xlarge",
                    "ml.p4d.24xlarge",
                ],
            },
            "ModelApprovalStatus": "Approved",
            "CustomerMetadataProperties": {
                "genesis": "true",
                "source": "onnx-export",
                "note": _GENESIS_METADATA_NOTE,
            },
        }
        if ncf_identity is not None:
            # Trained, not seeded: the genesis note and description would be false.
            # Approved because it is the build deploy.sh just uploaded for serving.
            create_params["ModelPackageDescription"] = (
                _NCF_SAMPLE_DESCRIPTION
                if ncf_identity["training_data"] == "sample"
                else f"Deal Scorer build {ncf_identity['fingerprint']} exported at deploy time"
            )
            create_params["CustomerMetadataProperties"] = {
                "source": "onnx-export",
                "training_data": ncf_identity["training_data"],
                "label": ncf_identity["label"],
                "generator": ncf_identity["generator"],
                "seed": ncf_identity["seed"],
                "fingerprint": ncf_identity["fingerprint"],
            }

        response = sagemaker_client.create_model_package(**create_params)
        model_package_arn = response["ModelPackageArn"]
        _LOG.info(
            "  %s (%s): registered %s - %s",
            model_type,
            package_group,
            f"build {ncf_identity['fingerprint']} ({ncf_identity['training_data']})"
            if ncf_identity is not None
            else "genesis version",
            model_package_arn,
        )
        results[model_type] = f"registered:{model_package_arn}"

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bucket", required=True, help="S3 bucket holding onnx-source/<model>/model.onnx")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE") or None,
                        help="AWS CLI profile for every call (default: AWS_PROFILE, else the SDK default chain)")
    # Each group flag is required for the model types this run processes (all of
    # them unless --only narrows it); checked after parsing.
    parser.add_argument("--dlrm-package-group", default=None, help="Model Package Group name for dlrm_bid_shader")
    parser.add_argument("--ncf-package-group", default=None, help="Model Package Group name for ncf_deal_manager")
    parser.add_argument("--yield-floor-package-group", default=None, help="Model Package Group name for deal_yield_manager_floor")
    parser.add_argument("--yield-margin-package-group", default=None, help="Model Package Group name for deal_yield_manager_margin")
    parser.add_argument(
        "--only",
        action="append",
        choices=[model_type for model_type, _ in _MODEL_TYPES],
        default=None,
        help="Process only this model type (repeatable). Default: all model types.",
    )
    parser.add_argument(
        "--training-image-repository",
        default="artf-nemo-rl-training",
        help="ECR repository name for the NeMo-RL training container (default: artf-nemo-rl-training)",
    )
    parser.add_argument("--account-id", default="", help="AWS account id (resolved via STS if omitted)")
    args = parser.parse_args(argv)
    group_flags = {
        "dlrm_bid_shader": ("dlrm_package_group", "--dlrm-package-group"),
        "ncf_deal_manager": ("ncf_package_group", "--ncf-package-group"),
        "deal_yield_manager_floor": ("yield_floor_package_group", "--yield-floor-package-group"),
        "deal_yield_manager_margin": ("yield_margin_package_group", "--yield-margin-package-group"),
    }
    selected = args.only or [model_type for model_type, _ in _MODEL_TYPES]
    missing_flags = [
        flag for model_type, (dest, flag) in group_flags.items()
        if model_type in selected and getattr(args, dest) is None
    ]
    if missing_flags:
        parser.error(f"the following arguments are required: {', '.join(missing_flags)}")
    if args.profile:
        # One place for both credential paths: boto3 clients created below, and any
        # subprocess (aws/kubectl) that reads AWS_PROFILE from the environment.
        os.environ["AWS_PROFILE"] = args.profile
        boto3.setup_default_session(profile_name=args.profile, region_name=args.region)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    package_groups = {
        "dlrm_bid_shader": args.dlrm_package_group,
        "ncf_deal_manager": args.ncf_package_group,
        "deal_yield_manager_floor": args.yield_floor_package_group,
        "deal_yield_manager_margin": args.yield_margin_package_group,
    }

    _LOG.info(
        "Registering genesis models (bucket=%s, region=%s, groups=%s)",
        args.model_bucket,
        args.region,
        package_groups,
    )

    results = register_genesis_models(
        model_bucket=args.model_bucket,
        region=args.region,
        package_groups=package_groups,
        training_image_repository=args.training_image_repository,
        account_id=args.account_id or None,
        model_types=args.only,
    )

    missing = [
        m for m, outcome in results.items()
        if outcome in ("skipped:missing-artifact", "skipped:missing-manifest", "skipped:invalid-manifest")
    ]
    not_ready = [m for m, outcome in results.items() if outcome == "skipped:training-image-not-ready"]
    _LOG.info("Genesis registration complete: %s", results)
    if not_ready:
        _LOG.warning(
            "%d model type(s) are waiting on the NeMo-RL training image to finish "
            "building: %s. Re-run this script (or ./check_builds.sh) once it is ready.",
            len(not_ready),
            not_ready,
        )
    if missing:
        _LOG.error(
            "%d model type(s) had no registrable artifact or manifest: %s",
            len(missing),
            missing,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
