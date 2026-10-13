"""A trained model version records which load-test run it was triggered from.

The picker let an operator choose a run, but `/train` posted only
`{model_type, confirmed}` — the selection never left the browser. So a resulting
model version had no link back to its originating run, and a comparison had no
way to pick "the run that trained this challenger" as its control.

The link travels: UI -> `/train` `run_id` -> a `load_test_run_id` hyperparameter
on the SageMaker job -> copied onto the model package by the registration Lambda
(governance_eventbridge_cfn.yaml) -> read back by `read_model_versions`.

Provenance only, and the tests assert the reason: both training shapes read an
ENTIRE S3 prefix holding every swept run, so this identifies the run the job was
triggered FROM, never the only data it learned from.
"""

from __future__ import annotations

import re
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

from closed_loop_demo.readers import read_model_versions  # noqa: E402
from orchestrator.training_trigger import _build_training_job_params  # noqa: E402

GOVERNANCE_CFN = REPO_ROOT / "deployment" / "governance_eventbridge_cfn.yaml"

XGB_IMAGE = "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


def _params(model_type, *, run_id=None):
    kwargs = dict(
        sagemaker_role_arn="r",
        model_bucket="mb",
        training_data_bucket="tb",
        training_image_registry="reg",
        xgboost_training_image_uri=XGB_IMAGE,
    )
    if run_id is not None:
        kwargs["load_test_run_id"] = run_id
    return _build_training_job_params(model_type, "job-1", "arn:base", **kwargs)


# ---------------------------------------------------------------------------
# The hyperparameter, for both training shapes
# ---------------------------------------------------------------------------

ALL_SHAPES = ["dlrm_bid_shader", "ncf_deal_manager", "deal_yield_manager_floor", "deal_yield_manager_margin"]
NEMO_RL = ["dlrm_bid_shader", "ncf_deal_manager"]
XGBOOST = ["deal_yield_manager_floor", "deal_yield_manager_margin"]


def _tags(params):
    return {t["Key"]: t["Value"] for t in params.get("Tags", [])}


@pytest.mark.parametrize("model_type", ALL_SHAPES)
def test_run_id_and_base_version_are_recorded_as_tags(model_type):
    """Tags are the carrier both shapes share, so the registration Lambda has
    one read path for every model."""
    tags = _tags(_params(model_type, run_id="lt-abc123"))
    assert tags["load_test_run_id"] == "lt-abc123"
    assert tags["base_model_version"] == "arn:base"


@pytest.mark.parametrize("model_type", NEMO_RL)
def test_nemo_rl_also_keeps_them_as_hyperparameters(model_type):
    """The custom NeMo-RL container reads them from its hyperparameters."""
    hp = _params(model_type, run_id="lt-abc123")["HyperParameters"]
    assert hp["load_test_run_id"] == "lt-abc123"
    assert hp["base_model_version"] == "arn:base"


@pytest.mark.parametrize("model_type", XGBOOST)
def test_xgboost_sends_only_xgboost_hyperparameters(model_type):
    """The built-in XGBoost container fails the job with "Extraneous
    hyperparameter found" for any key outside its own list."""
    hp = _params(model_type, run_id="lt-abc123")["HyperParameters"]
    assert set(hp) == {"max_depth", "eta", "num_round", "objective"}


@pytest.mark.parametrize("model_type", ["dlrm_bid_shader", "deal_yield_manager_floor"])
def test_key_is_absent_rather_than_empty_when_no_run_was_selected(model_type):
    """Scheduled retraining has no originating run. An empty string would read as
    "triggered from a run whose id we lost"."""
    params = _params(model_type)
    assert "load_test_run_id" not in params["HyperParameters"]
    assert "load_test_run_id" not in _tags(params)


# ---------------------------------------------------------------------------
# Input channel name, per shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", XGBOOST)
def test_xgboost_channel_is_train(model_type):
    """The built-in algorithm reads SM_CHANNEL_TRAIN in algorithm mode. A
    channel named "training" fails the job with KeyError: 'SM_CHANNEL_TRAIN'."""
    assert _params(model_type)["InputDataConfig"][0]["ChannelName"] == "train"


@pytest.mark.parametrize("model_type", NEMO_RL)
def test_nemo_rl_channel_is_training(model_type):
    """source/training/container/train.py reads SM_CHANNEL_TRAINING."""
    assert _params(model_type)["InputDataConfig"][0]["ChannelName"] == "training"


@pytest.mark.parametrize(
    ("model_type", "expected_prefix"),
    [
        ("dlrm_bid_shader", "s3://tb/training-data/"),
        ("deal_yield_manager_floor", "s3://tb/training-data-deal-yield-floor/"),
        ("deal_yield_manager_margin", "s3://tb/training-data-deal-yield-margin/"),
    ],
)
def test_provenance_does_not_scope_the_training_input(model_type, expected_prefix):
    """The load-bearing caveat. Recording a run id must not be mistaken for
    training on only that run -- the input stays an entire prefix, so anything
    presenting this has to say "triggered from", not "trained on"."""
    params = _params(model_type, run_id="lt-abc123")
    uri = params["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"]
    assert uri == expected_prefix
    assert "lt-abc123" not in uri


def test_existing_hyperparameters_are_untouched():
    params = _params("dlrm_bid_shader", run_id="lt-abc123")["HyperParameters"]
    assert params["model_type"] == "dlrm_bid_shader"
    assert params["base_model_version"] == "arn:base"
    assert params["triggered_by"] == "governance_ui_on_demand"


# ---------------------------------------------------------------------------
# Reading it back off the model version
# ---------------------------------------------------------------------------

def _sagemaker(metadata):
    client = mock.Mock()
    client.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageArn": "arn:aws:sagemaker:us-east-1:1:model-package/g/2",
                "ModelPackageVersion": 2,
                "ModelApprovalStatus": "Approved",
                "ModelPackageStatus": "Completed",
                "CreationTime": "2026-08-28T10:00:00",
            }
        ]
    }
    client.describe_model_package.return_value = {"CustomerMetadataProperties": metadata}
    return client


def test_training_run_id_is_returned_for_the_version():
    versions = read_model_versions(_sagemaker({"load_test_run_id": "lt-abc123"}), "g")
    assert versions[0]["training_run_id"] == "lt-abc123"


def test_missing_provenance_reads_as_none():
    """Scheduled retraining, and versions registered before this existed."""
    versions = read_model_versions(_sagemaker({}), "g")
    assert versions[0]["training_run_id"] is None


def test_empty_provenance_normalises_to_none():
    versions = read_model_versions(_sagemaker({"load_test_run_id": ""}), "g")
    assert versions[0]["training_run_id"] is None


def test_describe_failure_does_not_break_the_listing():
    client = _sagemaker({})
    client.describe_model_package.side_effect = RuntimeError("AccessDenied")
    versions = read_model_versions(client, "g")
    assert len(versions) == 1
    assert versions[0]["training_run_id"] is None
    assert versions[0]["approval_status"] == "Approved"


def test_a_sample_build_reports_its_training_data_and_fingerprint():
    """What register_genesis_models.py records on a deploy-time Deal Scorer build."""
    versions = read_model_versions(
        _sagemaker({"training_data": "sample", "label": "implicit-engagement-sample",
                    "generator": "source/training/ncf_sample", "seed": "20261007",
                    "fingerprint": "9509e1edf027", "source": "onnx-export"}),
        "g",
    )
    assert versions[0]["training_data"] == "sample"
    assert versions[0]["fingerprint"] == "9509e1edf027"


def test_a_version_that_records_neither_reads_as_none():
    versions = read_model_versions(_sagemaker({"load_test_run_id": "lt-abc123"}), "g")
    assert versions[0]["training_data"] is None
    assert versions[0]["fingerprint"] is None


# ---------------------------------------------------------------------------
# The deployment half of the chain
# ---------------------------------------------------------------------------

def test_registration_lambda_copies_the_run_id_onto_the_model_package():
    """Provenance nothing reads back is not provenance. The Lambda that registers
    the version has to carry it across, from hyperparameters (NeMo-RL) or Tags
    (XGBoost, where hyperparameters cannot hold it)."""
    text = GOVERNANCE_CFN.read_text(encoding="utf-8")
    assert re.search(r"sm\.list_tags\(ResourceArn=job\['TrainingJobArn'\]\)", text), (
        "registration Lambda does not read the training job's tags"
    )
    assert re.search(
        r"return hyperparams\.get\(key\) or tags\.get\(key\) or ''", text
    ), "registration Lambda does not fall back from hyperparameters to tags"
    for key in ("load_test_run_id", "base_model_version"):
        assert re.search(rf"'{key}':\s*_provenance\('{key}'\)", text), (
            f"registration Lambda does not copy {key} into CustomerMetadataProperties"
        )


def _iam_actions_for(text, policy_name):
    block = text[text.index(f"PolicyName: {policy_name}"):]
    block = block[: block.index("PolicyName:", 1)] if "PolicyName:" in block[1:] else block
    return block


def test_roles_carry_the_tag_permissions():
    """CreateTrainingJob with Tags is authorized as sagemaker:AddTags, and the
    registration Lambda's ListTags read needs its own grant."""
    text = GOVERNANCE_CFN.read_text(encoding="utf-8")
    assert "'sagemaker:AddTags'" in _iam_actions_for(text, "StartTrainingJobs")
    assert "'sagemaker:ListTags'" in _iam_actions_for(text, "DescribeAndRegisterTrainedModels")
    deploy = (REPO_ROOT / "deployment" / "deploy.sh").read_text(encoding="utf-8")
    sid = deploy[deploy.index('TrainingTriggerFromGovernanceUI'):]
    sid = sid[: sid.index("]}")]
    assert "sagemaker:AddTags" in sid


def test_scheduled_trigger_uses_train_channel_for_xgboost():
    """Scheduled yield retraining had the same channel bug as the UI trigger."""
    text = GOVERNANCE_CFN.read_text(encoding="utf-8")
    assert "channel_name = 'train'" in text
    assert "channel_name = 'training'" in text
    assert "'ChannelName': channel_name" in text
    assert "'ChannelName': 'training'" not in text


def test_frontend_sends_the_selected_run():
    """The selection existed in component state but was never in the request
    body, which is why no version had provenance."""
    panel = (
        REPO_ROOT / "source" / "frontend-react" / "src" / "components" / "GovernancePanel.jsx"
    ).read_text(encoding="utf-8")
    assert "run_id: selectedTrainingRun" in panel
