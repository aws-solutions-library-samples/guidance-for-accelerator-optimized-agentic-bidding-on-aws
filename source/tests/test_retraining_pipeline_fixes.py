"""Retraining failures on tr3 that were not about the training data.

Seen in CloudWatch on 2026-11-06 (aidlc-docs/construction/retraining-pipeline-fixes):

1. The registration Lambda passed window_days='' for XGBoost jobs, and
   CreateModelPackage rejects an empty CustomerMetadataProperties value, so both
   completed deal-yield jobs were never registered.
2. The scheduled trigger role granted ListModelPackages only on the
   model-package-group ARN; IAM checks model-package/<group>/*, so every
   scheduled run failed AccessDenied.
3. The registration role could not read the AWS-owned sagemaker-xgboost image
   that CreateModelPackage validates with the caller's credentials.
4. train.py never wrote /opt/ml/output/failure, so FailureReason was empty.

The Lambda code is inline in the template, so the registration tests extract the
ZipFile text and exec it with boto3 stubbed.
"""

from __future__ import annotations

import os
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "deployment" / "governance_eventbridge_cfn.yaml"

PACKAGE_GROUPS = {
    "DLRM_PACKAGE_GROUP": "tr3-artf-dlrm-bid-shader",
    "NCF_PACKAGE_GROUP": "tr3-artf-ncf-deal-manager",
    "YIELD_FLOOR_PACKAGE_GROUP": "tr3-artf-deal-yield-manager-floor",
    "YIELD_MARGIN_PACKAGE_GROUP": "tr3-artf-deal-yield-manager-margin",
}
XGB_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that keeps CloudFormation short-form intrinsics as {tag: value}."""


def _cfn_tag(loader, tag_suffix, node):
    name = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"
    if isinstance(node, yaml.ScalarNode):
        return {name: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {name: loader.construct_sequence(node)}
    return {name: loader.construct_mapping(node)}


_CfnLoader.add_multi_constructor("!", _cfn_tag)


def _template():
    loader = _CfnLoader(TEMPLATE.read_text(encoding="utf-8"))
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def _statements(role_name, policy_name):
    role = _template()["Resources"][role_name]["Properties"]
    for policy in role["Policies"]:
        if policy["PolicyName"] == policy_name:
            return policy["PolicyDocument"]["Statement"]
    raise AssertionError(f"{role_name} has no policy {policy_name}")


def _resources_for(statements, action):
    found = []
    for stmt in statements:
        actions = stmt["Action"] if isinstance(stmt["Action"], list) else [stmt["Action"]]
        if action in actions:
            res = stmt["Resource"] if isinstance(stmt["Resource"], list) else [stmt["Resource"]]
            found.extend(r["Fn::Sub"] if isinstance(r, dict) else r for r in res)
    return found


# ---------------------------------------------------------------------------
# Registration Lambda, executed from the template
# ---------------------------------------------------------------------------

def _run_registration(monkeypatch, job_name, *, hyperparameters, tags, metrics):
    sm = mock.Mock()
    sm.describe_training_job.return_value = {
        "TrainingJobArn": f"arn:aws:sagemaker:us-east-1:1:training-job/{job_name}",
        "ModelArtifacts": {"S3ModelArtifacts": f"s3://mb/models/x/{job_name}/output/model.tar.gz"},
        "FinalMetricDataList": metrics,
        "HyperParameters": hyperparameters,
    }
    sm.list_tags.return_value = {"Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}
    sm.create_model_package.return_value = {"ModelPackageArn": "arn:aws:sagemaker:us-east-1:1:model-package/g/2"}
    events = mock.Mock()
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda name, **_: sm if name == "sagemaker" else events
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    for key, value in {**PACKAGE_GROUPS, "TRAINING_IMAGE_REGISTRY": "reg",
                       "XGBOOST_TRAINING_IMAGE_URI": XGB_IMAGE}.items():
        monkeypatch.setenv(key, value)

    code = _template()["Resources"]["RetrainingRegistrationFunction"]["Properties"]["Code"]["ZipFile"]
    ns: dict = {}
    exec(compile(code, "retraining_registration_index.py", "exec"), ns)
    result = ns["handler"]({"detail": {"TrainingJobName": job_name, "TrainingJobStatus": "Completed"}}, None)
    return result, sm


def test_xgboost_job_registers_without_window_days(monkeypatch):
    """The 2026-10-28 margin job: XGBoost-only hyperparameters, provenance in tags."""
    result, sm = _run_registration(
        monkeypatch,
        "deal-yield-manager-margin-1793224287-29db5e5c",
        hyperparameters={"eta": "0.3", "max_depth": "6", "num_round": "100", "objective": "reg:squarederror"},
        tags={"base_model_version": "arn:base/1", "load_test_run_id": "lt-abc"},
        metrics=[{"MetricName": "train:rmse", "Value": 0.00167}],
    )
    assert result["status"] == "registered"
    metadata = sm.create_model_package.call_args.kwargs["CustomerMetadataProperties"]
    assert "window_days" not in metadata
    assert metadata["base_model_version"] == "arn:base/1"
    assert metadata["load_test_run_id"] == "lt-abc"
    assert metadata["metric_train:rmse"] == "0.00167"
    assert sm.create_model_package.call_args.kwargs["ModelPackageGroupName"] == PACKAGE_GROUPS["YIELD_MARGIN_PACKAGE_GROUP"]


def test_no_metadata_value_is_empty(monkeypatch):
    """A scheduled job: no load_test_run_id anywhere."""
    _, sm = _run_registration(
        monkeypatch,
        "deal-yield-manager-floor-1-abcdef12",
        hyperparameters={"eta": "0.3"},
        tags={"base_model_version": "arn:base/1"},
        metrics=[],
    )
    metadata = sm.create_model_package.call_args.kwargs["CustomerMetadataProperties"]
    assert "load_test_run_id" not in metadata
    assert all(isinstance(v, str) and v for v in metadata.values()), metadata


def test_nemo_rl_job_keeps_window_days(monkeypatch):
    _, sm = _run_registration(
        monkeypatch,
        "dlrm-bid-shader-1-abcdef12",
        hyperparameters={"model_type": "dlrm_bid_shader", "base_model_version": "arn:base/1",
                         "window_days": "7", "load_test_run_id": "lt-abc"},
        tags={},
        metrics=[],
    )
    metadata = sm.create_model_package.call_args.kwargs["CustomerMetadataProperties"]
    assert metadata["window_days"] == "7"
    assert metadata["load_test_run_id"] == "lt-abc"


# ---------------------------------------------------------------------------
# IAM
# ---------------------------------------------------------------------------

def test_trigger_role_lists_model_packages_on_the_package_arn():
    group_params = sorted(p for p in _template()["Parameters"] if p.endswith("ModelPackageGroupName"))
    assert len(group_params) == 4, group_params
    resources = _resources_for(
        _statements("RetrainingTriggerFunctionRole", "StartTrainingJobs"), "sagemaker:ListModelPackages"
    )
    for group_param in group_params:
        expected = (
            "arn:${AWS::Partition}:sagemaker:${AWS::Region}:${AWS::AccountId}:"
            f"model-package/${{{group_param}}}/*"
        )
        assert expected in resources, group_param


def test_registration_role_can_read_the_xgboost_image():
    resources = _resources_for(
        _statements("RetrainingRegistrationFunctionRole", "ReadTrainingImageForRegistration"),
        "ecr:BatchGetImage",
    )
    assert "arn:${AWS::Partition}:ecr:${AWS::Region}:*:repository/sagemaker-xgboost" in resources


# ---------------------------------------------------------------------------
# train.py failure reason
# ---------------------------------------------------------------------------

@pytest.fixture
def train_module():
    # Appended for the same reason as test_train_container.py: prepending would
    # shadow source/shared with the container's staged copy.
    sys.path.append(os.path.join(REPO_ROOT, "source", "training", "container"))
    import train

    return train


def test_unhandled_error_is_written_to_the_failure_file(train_module, tmp_path, monkeypatch):
    failure = tmp_path / "output" / "failure"
    monkeypatch.setattr(train_module, "SM_FAILURE_FILE", str(failure))

    def _fail():
        raise train_module.DatasetUnfitError("'label_conversion' has no positives across 2756 labelled rows")

    monkeypatch.setattr(train_module, "main", _fail)

    assert train_module.run() == 1
    assert failure.read_text(encoding="utf-8") == (
        "DatasetUnfitError: 'label_conversion' has no positives across 2756 labelled rows"
    )


def test_success_writes_no_failure_file(train_module, tmp_path, monkeypatch):
    failure = tmp_path / "failure"
    monkeypatch.setattr(train_module, "SM_FAILURE_FILE", str(failure))
    monkeypatch.setattr(train_module, "main", lambda: None)
    assert train_module.run() == 0
    assert not failure.exists()
