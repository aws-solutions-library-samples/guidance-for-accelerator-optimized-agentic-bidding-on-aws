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

import io
import json
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

# The fields deploy.sh Step 2's ncf manifest carries that the registration reads
# (source/training/ncf_sample/build.py writes them).
_NCF_FP = "9509e1edf0273ebbb9efa03b17fcbe4ad9c1264119dd120dedff93f40f41fc48"
_NCF_MANIFEST = {
    "model_type": "ncf_deal_manager",
    "model_version": f"ncf-neumf-sample-{_NCF_FP[:12]}",
    "fingerprint": _NCF_FP,
    "fingerprint12": _NCF_FP[:12],
    "training_data": "sample",
    "target": "implicit-engagement-sample",
    "generator_seed": 20261007,
    "generator": {"path": "source/training/ncf_sample", "format": "ncf-sample-cohorts/1"},
}


@pytest.fixture(autouse=True)
def _no_live_aws(monkeypatch):
    """Every client a test does not pass in is a stub, so no test reaches AWS.

    The registration checks the training image in ECR; the stub reports it present.
    """
    ecr = MagicMock()
    ecr.describe_images.return_value = {"imageDetails": [{}]}

    def _client(service, *args, **kwargs):
        if service == "ecr":
            return ecr
        raise AssertionError(f"test created a live boto3 {service} client")

    monkeypatch.setattr(rgm.boto3, "client", _client)


def _make_sts_client(account_id: str = "123456789012") -> MagicMock:
    sts = MagicMock()
    sts.get_caller_identity.return_value = {"Account": account_id}
    return sts


def _attach_manifest(s3: MagicMock, manifest: dict | None = _NCF_MANIFEST) -> MagicMock:
    """get_object serves the ncf manifest (or 404s when `manifest` is None)."""
    def _get_object(Bucket, Key):
        if Key == "onnx-source/ncf_deal_manager/manifest.json" and manifest is not None:
            return {"Body": io.BytesIO(json.dumps(manifest).encode())}
        raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")

    s3.get_object.side_effect = _get_object
    return s3


def _make_s3_client(artifact_exists: bool = True, manifest: dict | None = _NCF_MANIFEST) -> MagicMock:
    s3 = MagicMock()
    if artifact_exists:
        s3.head_object.return_value = {}
    else:
        def _raise(Bucket, Key):
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )

        s3.head_object.side_effect = _raise
    return _attach_manifest(s3, manifest)


def _make_sagemaker_client(
    existing_groups: set[str] | None = None,
    recorded: dict[str, list[str | None]] | None = None,
) -> MagicMock:
    """SageMaker client where groups in `existing_groups` already have a version.

    `recorded` maps a group to its versions newest first, each the `fingerprint`
    metadata that version records (None for none). A group in `existing_groups`
    without an entry has one version that records none.
    """
    existing_groups = existing_groups or set()
    recorded = recorded or {}
    sm = MagicMock()

    def _versions(group):
        if group in recorded:
            return recorded[group]
        return [None] if group in existing_groups else []

    def _list_model_packages(ModelPackageGroupName, MaxResults, **kwargs):
        versions = _versions(ModelPackageGroupName)
        return {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": f"{ModelPackageGroupName}/v{len(versions) - i}"}
                for i in range(len(versions))
            ][:MaxResults]
        }

    def _describe_model_package(ModelPackageName):
        group, version = ModelPackageName.rsplit("/v", 1)
        versions = _versions(group)
        fp = versions[len(versions) - int(version)]
        return {"CustomerMetadataProperties": {"fingerprint": fp} if fp else {}}

    sm.list_model_packages.side_effect = _list_model_packages
    sm.describe_model_package.side_effect = _describe_model_package
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
            package_groups={"dlrm_bid_shader": "artf-dlrm-bid-shader"},
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=s3,
            sts_client=sts,
        )

        call_kwargs = sm.create_model_package.call_args_list[0][1]
        model_data_url = call_kwargs["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        assert model_data_url == "s3://my-model-bucket/onnx-source/dlrm_bid_shader/model.onnx"

    def test_ncf_model_data_url_is_the_builds_own_copy(self):
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
        assert model_data_url == (
            f"s3://my-model-bucket/onnx-source/ncf_deal_manager/builds/{_NCF_FP[:12]}/model.onnx"
        )
        s3.head_object.assert_called_once_with(
            Bucket="my-model-bucket",
            Key=f"onnx-source/ncf_deal_manager/builds/{_NCF_FP[:12]}/model.onnx",
        )


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
        # ncf counts as registered when a version records this build's fingerprint.
        sm = _make_sagemaker_client(
            existing_groups={"artf-dlrm-bid-shader"},
            recorded={"artf-ncf-deal-manager": [_NCF_FP[:12]]},
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
        # Only the two configured groups; the yield types report no-package-group.
        assert all(first_results[m].startswith("registered:") for m in _PACKAGE_GROUPS)

        # Second run: both groups now report an existing version, the ncf one
        # recording the fingerprint the first run registered.
        registered_ncf = sm_first.create_model_package.call_args_list[1][1]
        sm_second = _make_sagemaker_client(
            existing_groups={"artf-dlrm-bid-shader"},
            recorded={
                "artf-ncf-deal-manager": [
                    registered_ncf["CustomerMetadataProperties"]["fingerprint"]
                ]
            },
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
        assert all(second_results[m] == "skipped:exists" for m in _PACKAGE_GROUPS)
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
        s3 = _attach_manifest(MagicMock())

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
# Tests: the Deal Scorer is registered per build fingerprint
# ---------------------------------------------------------------------------

_NCF_ONLY = {"ncf_deal_manager": "artf-ncf-deal-manager"}


def _register_ncf(sm, s3=None):
    return rgm.register_genesis_models(
        model_bucket="test-bucket",
        region="us-east-1",
        package_groups=_NCF_ONLY,
        training_image_repository="artf-nemo-rl-training",
        sagemaker_client=sm,
        s3_client=s3 or _make_s3_client(),
        sts_client=_make_sts_client(),
    )


class TestNcfFingerprintRule:
    def test_a_different_fingerprint_registers_a_sample_version(self):
        sm = _make_sagemaker_client(recorded={"artf-ncf-deal-manager": ["0123456789ab"]})

        results = _register_ncf(sm)

        assert results["ncf_deal_manager"].startswith("registered:")
        params = sm.create_model_package.call_args_list[0][1]
        assert params["CustomerMetadataProperties"] == {
            "source": "onnx-export",
            "training_data": "sample",
            "label": "implicit-engagement-sample",
            "generator": "source/training/ncf_sample",
            "seed": "20261007",
            "fingerprint": _NCF_FP[:12],
        }
        assert params["ModelPackageDescription"] == (
            "Deal Scorer trained on sample interactions generated from the deal book, "
            "not observed outcomes"
        )
        assert params["ModelApprovalStatus"] == "Approved"
        assert params["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == (
            f"s3://test-bucket/onnx-source/ncf_deal_manager/builds/{_NCF_FP[:12]}/model.onnx"
        )

    def test_each_build_version_names_its_own_artifact(self):
        """Two builds in turn: each version's URL carries the fingerprint it records."""
        second_fp = "0e76145b9ea17c263e64ce611993a67adccb61e9f33eed1682393ff7a6f368d6"
        second = {**_NCF_MANIFEST, "fingerprint": second_fp, "fingerprint12": second_fp[:12],
                  "model_version": f"ncf-neumf-sample-{second_fp[:12]}"}
        sm_first = _make_sagemaker_client()
        _register_ncf(sm_first)
        sm_second = _make_sagemaker_client(recorded={"artf-ncf-deal-manager": [_NCF_FP[:12]]})
        _register_ncf(sm_second, _make_s3_client(manifest=second))

        urls = {}
        for sm in (sm_first, sm_second):
            params = sm.create_model_package.call_args_list[0][1]
            fp12 = params["CustomerMetadataProperties"]["fingerprint"]
            urls[fp12] = params["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        assert set(urls) == {_NCF_FP[:12], second_fp[:12]}
        for fp12, url in urls.items():
            assert url == f"s3://test-bucket/onnx-source/ncf_deal_manager/builds/{fp12}/model.onnx"

    def test_a_build_without_its_own_copy_is_not_registered(self):
        """Only the shared key exists (an upload that predates builds/): no version names it."""
        s3 = _attach_manifest(MagicMock())

        def _head_object(Bucket, Key):
            if Key == "onnx-source/ncf_deal_manager/model.onnx":
                return {}
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

        s3.head_object.side_effect = _head_object
        sm = _make_sagemaker_client()

        results = _register_ncf(sm, s3)

        assert results["ncf_deal_manager"] == "skipped:missing-artifact"
        assert sm.create_model_package.call_count == 0

    def test_the_metadata_stays_within_the_registry_charset(self):
        """CreateModelPackage rejects values outside [\\p{L}\\p{Z}\\p{N}_.:/=+-@]."""
        sm = _make_sagemaker_client()
        _register_ncf(sm)
        for value in sm.create_model_package.call_args_list[0][1]["CustomerMetadataProperties"].values():
            assert rgm._METADATA_VALUE_RE.match(value), value

    def test_the_same_fingerprint_is_not_registered_again(self):
        sm = _make_sagemaker_client(recorded={"artf-ncf-deal-manager": [_NCF_FP[:12]]})

        results = _register_ncf(sm)

        assert results["ncf_deal_manager"] == "skipped:exists"
        assert sm.create_model_package.call_count == 0

    def test_a_newer_version_without_a_fingerprint_does_not_cause_a_duplicate(self):
        """A retrained version on top records no fingerprint; the build is still registered."""
        sm = _make_sagemaker_client(recorded={"artf-ncf-deal-manager": [None, _NCF_FP[:12], None]})

        results = _register_ncf(sm)

        assert results["ncf_deal_manager"] == "skipped:exists"
        assert sm.create_model_package.call_count == 0

    def test_a_genesis_only_group_gets_the_sample_version(self):
        """An existing stack: the seeded-weights genesis is the only ncf version."""
        sm = _make_sagemaker_client(existing_groups={"artf-ncf-deal-manager"})

        results = _register_ncf(sm)

        assert results["ncf_deal_manager"].startswith("registered:")
        assert "genesis" not in sm.create_model_package.call_args_list[0][1]["CustomerMetadataProperties"]

    def test_dlrm_keeps_skip_if_any(self):
        sm = _make_sagemaker_client(existing_groups={"artf-dlrm-bid-shader"})

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups={"dlrm_bid_shader": "artf-dlrm-bid-shader"},
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=_make_s3_client(),
            sts_client=_make_sts_client(),
        )

        assert results["dlrm_bid_shader"] == "skipped:exists"

    def test_a_missing_manifest_is_not_registered(self):
        sm = _make_sagemaker_client()

        results = _register_ncf(sm, _make_s3_client(manifest=None))

        assert results["ncf_deal_manager"] == "skipped:missing-manifest"
        assert sm.create_model_package.call_count == 0

    @pytest.mark.parametrize(
        "manifest",
        [
            {**_NCF_MANIFEST, "fingerprint12": "not-hex", "fingerprint": "zz"},
            {k: v for k, v in _NCF_MANIFEST.items() if k != "training_data"},
            {**_NCF_MANIFEST, "target": "implicit, engagement"},
            {k: v for k, v in _NCF_MANIFEST.items() if k != "generator_seed"},
            {**_NCF_MANIFEST, "generator_seed": "20261007"},
            {k: v for k, v in _NCF_MANIFEST.items() if k != "generator"},
            {**_NCF_MANIFEST, "generator": {"path": "source/training/(ncf_sample)"}},
        ],
        ids=["bad-fingerprint", "no-training-data", "comma-in-target", "no-seed", "seed-not-int",
             "no-generator", "generator-outside-charset"],
    )
    def test_an_invalid_manifest_is_not_registered(self, manifest):
        sm = _make_sagemaker_client()

        results = _register_ncf(sm, _make_s3_client(manifest=manifest))

        assert results["ncf_deal_manager"] == "skipped:invalid-manifest"
        assert sm.create_model_package.call_count == 0

    def test_only_limits_the_run(self):
        sm = _make_sagemaker_client()

        results = rgm.register_genesis_models(
            model_bucket="test-bucket",
            region="us-east-1",
            package_groups=_PACKAGE_GROUPS,
            training_image_repository="artf-nemo-rl-training",
            sagemaker_client=sm,
            s3_client=_make_s3_client(),
            sts_client=_make_sts_client(),
            model_types=["ncf_deal_manager"],
        )

        assert list(results) == ["ncf_deal_manager"]


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

    def test_only_ncf_needs_only_the_ncf_group(self, monkeypatch):
        seen = {}

        def _fake(**kwargs):
            seen.update(kwargs)
            return {"ncf_deal_manager": "skipped:exists"}

        monkeypatch.setattr(rgm, "register_genesis_models", _fake)

        exit_code = rgm.main(
            [
                "--model-bucket", "test-bucket",
                "--region", "us-east-1",
                "--ncf-package-group", "artf-ncf-deal-manager",
                "--only", "ncf_deal_manager",
            ]
        )

        assert exit_code == 0
        assert seen["model_types"] == ["ncf_deal_manager"]

    def test_only_ncf_without_the_ncf_group_is_an_error(self):
        with pytest.raises(SystemExit):
            rgm.main(["--model-bucket", "test-bucket", "--only", "ncf_deal_manager"])

    def test_a_missing_manifest_fails_the_run(self, monkeypatch):
        monkeypatch.setattr(rgm, "register_genesis_models", lambda **kwargs: {
            "ncf_deal_manager": "skipped:missing-manifest",
        })

        exit_code = rgm.main(
            [
                "--model-bucket", "test-bucket",
                "--ncf-package-group", "artf-ncf-deal-manager",
                "--only", "ncf_deal_manager",
            ]
        )

        assert exit_code == 1
