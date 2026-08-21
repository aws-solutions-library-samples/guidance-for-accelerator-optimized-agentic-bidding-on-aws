"""Unit tests for deployment/scripts/register_genesis_models.py.

Tests cover:
- Idempotency: a Model Package Group with an existing version is skipped
- Registration: an empty Model Package Group gets a genesis version registered
  with the correct fields (ModelApprovalStatus=Approved, genesis metadata tag)
- Honest failure: a missing ONNX artifact is skipped, not fabricated
- CLI exit code reflects whether any model type was skipped for a missing artifact

No fabricated success/failure values in surfaces presented as real (see
source/tests/README.md conventions and the workspace no-fabricated-data policy).
All assertions are against real (mocked) boto3 call arguments and return values.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "deployment", "scripts")
)

import register_genesis_models as rgm  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_PACKAGE_GROUPS = {
    "dlrm_bid_shader": "artf-dlrm-bid-shader",
    "ncf_deal_manager": "artf-ncf-deal-manager",
}


def _make_sts_client(account_id: str = "123456789012") -> MagicMock:
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": account_id}
    return sts


def _make_s3_client(artifact_exists: bool = True) -> MagicMock:
    s3 = MagicMock()
    if artifact_exists:
        s3.head_object.return_value = {}
    else:
        def _raise(Bucket, Key):
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )

        s3.head_object.side_effect = _raise
    return s3


def _make_sagemaker_client(existing_groups: set[str] | None = None) -> MagicMock:
    """SageMaker client where groups in `existing_groups` already have a version."""
    existing_groups = existing_groups or set()
    sm = MagicMock()

    def _list_model_packages(ModelPackageGroupName, MaxResults):
        if ModelPackageGroupName in existing_groups:
            return {"ModelPackageSummaryList": [{"ModelPackageArn": "existing-arn"}]}
        return {"ModelPackageSummaryList": []}

    sm.list_model_packages.side_effect = _list_model_packages
    sm.create_model_package.return_value = {
        "ModelPackageArn": (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/"
            "artf-dlrm-bid-shader/1"
        )
    }
    return sm


# ---------------------------------------------------------------------------
# Tests: registration (happy path)
# ---------------------------------------------------------------------------


class TestRegisterGenesisModels:
    def test_registers_both_model_types_when_groups_empty(self):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        assert results["dlrm_bid_shader"].startswith("registered:")
        assert results["ncf_deal_manager"].startswith("registered:")
        assert sm.create_model_package.call_count == 2

    def test_registration_uses_approved_status_and_genesis_metadata(self):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        call_kwargs = sm.create_model_package.call_args_list[0][1]
        assert call_kwargs["ModelApprovalStatus"] == "Approved"
        metadata = call_kwargs["CustomerMetadataProperties"]
        assert metadata["genesis"] == "true"
        assert metadata["source"] == "onnx-export"
        assert "note" in metadata

    def test_registration_uses_real_account_and_region_in_image_uri(self):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client(account_id="999988887777")

        rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="eu-west-1",
            package_groups={"dlrm_bid_shader": "artf-dlrm-bid-shader"},
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        call_kwargs = sm.create_model_package.call_args_list[0][1]
        image = call_kwargs["InferenceSpecification"]["Containers"][0]["Image"]
        # Must be a real, resolved ECR URI - never the unformatted template
        # string bug present in source/training/pipeline.py's _TRAINING_IMAGE_MAP.
        assert image == "999988887777.dkr.ecr.eu-west-1.amazonaws.com/artf-nemo-rl-training:dlrm"
        assert "{account}" not in image
        assert "{region}" not in image

    def test_model_data_url_points_to_onnx_source_artifact(self):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        rgm.register_genesis_models(
            model_bucket="my-model-bucket",
            region="us-east-1",
            package_groups={"ncf_deal_manager": "artf-ncf-deal-manager"},
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        call_kwargs = sm.create_model_package.call_args_list[0][1]
        model_data_url = call_kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        assert model_data_url == "s3://my-model-bucket/onnx-source/ncf_deal_manager/model.onnx"


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_skips_group_with_existing_version(self):
        sm = _make_sagemaker_client(existing_groups={"artf-dlrm-bid-shader"})
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        assert results["dlrm_bid_shader"] == "skipped:exists"
        assert results["ncf_deal_manager"].startswith("registered:")
        # Only the empty group gets a create_model_package call.
        assert sm.create_model_package.call_count == 1

    def test_skips_both_groups_when_both_already_registered(self):
        sm = _make_sagemaker_client(
            existing_groups={"artf-dlrm-bid-shader", "artf-ncf-deal-manager"}
        )
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        assert results["dlrm_bid_shader"] == "skipped:exists"
        assert results["ncf_deal_manager"] == "skipped:exists"
        assert sm.create_model_package.call_count == 0

    def test_rerunning_after_registration_is_a_noop(self):
        """Simulates running the script twice: first call registers, second
        call (with a fresh client reflecting the now-existing version) skips.
        """
        sm_first = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        first_results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm_first,
            s3_client=s3,
            sts_client=sts,
        )
        assert all(v.startswith("registered:") for v in first_results.values())

        # Second run: both groups now report an existing version.
        sm_second = _make_sagemaker_client(
            existing_groups={"artf-dlrm-bid-shader", "artf-ncf-deal-manager"}
        )
        second_results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm_second,
            s3_client=s3,
            sts_client=sts,
        )
        assert all(v == "skipped:exists" for v in second_results.values())
        assert sm_second.create_model_package.call_count == 0


# ---------------------------------------------------------------------------
# Tests: honest failure (no fabrication)
# ---------------------------------------------------------------------------


class TestNoFabrication:
    def test_missing_onnx_artifact_is_skipped_not_fabricated(self):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=False)
        sts = _make_sts_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        assert results["dlrm_bid_shader"] == "skipped:missing-artifact"
        assert results["ncf_deal_manager"] == "skipped:missing-artifact"
        assert sm.create_model_package.call_count == 0

    def test_missing_artifact_for_one_model_type_does_not_block_the_other(self):
        s3 = MagicMock()

        def _head_object(Bucket, Key):
            if "ncf_deal_manager" in Key:
                raise ClientError(
                    {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
                )
            return {}

        s3.head_object.side_effect = _head_object
        sm = _make_sagemaker_client(existing_groups=set())
        sts = _make_sts_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        assert results["dlrm_bid_shader"].startswith("registered:")
        assert results["ncf_deal_manager"] == "skipped:missing-artifact"
        assert sm.create_model_package.call_count == 1

    def test_non_404_s3_error_propagates_instead_of_being_swallowed(self):
        """A permissions error (403) or other unexpected S3 error must not be
        silently treated as 'artifact missing' - that would misreport the
        actual problem. Only 404/NoSuchKey/NotFound means 'missing'.
        """
        s3 = MagicMock()

        def _head_object(Bucket, Key):
            raise ClientError(
                {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
            )

        s3.head_object.side_effect = _head_object
        sm = _make_sagemaker_client(existing_groups=set())
        sts = _make_sts_client()

        with pytest.raises(ClientError):
            rgm.register_genesis_models(
                model_bucket="test-bucket",
                region="us-east-1",
                package_groups={"dlrm_bid_shader": "artf-dlrm-bid-shader"},
                training_image_repository="artf-nemo-rl-training",
                sagemaker_client=sm,
                s3_client=s3,
                sts_client=sts,
            )

    def test_missing_package_group_config_is_skipped_honestly(self):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups={"dlrm_bid_shader": "artf-dlrm-bid-shader", "ncf_deal_manager": ""},
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        assert results["ncf_deal_manager"] == "skipped:no-package-group"
        assert sm.create_model_package.call_count == 1


# ---------------------------------------------------------------------------
# Tests: CLI entry point
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_main_returns_zero_on_full_success(self, monkeypatch):
        sm = _make_sagemaker_client(existing_groups=set())
        s3 = _make_s3_client(artifact_exists=True)
        sts = _make_sts_client()

        monkeypatch.setattr(rgm, "register_genesis_models", lambda **kwargs: {
            "dlrm_bid_shader": "registered:arn1",
            "ncf_deal_manager": "registered:arn2",
        })

        exit_code = rgm.main(
            [
                "--model-bucket", "test-bucket",
                "--region", "us-east-1",
                "--dlrm-package-group", "artf-dlrm-bid-shader",
                "--ncf-package-group", "artf-ncf-deal-manager",
                "--yield-floor-package-group", "artf-deal-yield-manager-floor",
                "--yield-margin-package-group", "artf-deal-yield-manager-margin",
            ]
        )
        assert exit_code == 0

    def test_main_returns_nonzero_when_artifact_missing(self, monkeypatch):
        monkeypatch.setattr(rgm, "register_genesis_models", lambda **kwargs: {
            "dlrm_bid_shader": "skipped:missing-artifact",
            "ncf_deal_manager": "registered:arn2",
        })

        exit_code = rgm.main(
            [
                "--model-bucket", "test-bucket",
                "--region", "us-east-1",
                "--dlrm-package-group", "artf-dlrm-bid-shader",
                "--ncf-package-group", "artf-ncf-deal-manager",
                "--yield-floor-package-group", "artf-deal-yield-manager-floor",
                "--yield-margin-package-group", "artf-deal-yield-manager-margin",
            ]
        )
        assert exit_code == 1

    def test_main_requires_both_package_group_args(self):
        with pytest.raises(SystemExit):
            rgm.main(["--model-bucket", "test-bucket"])
