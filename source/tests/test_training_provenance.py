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

XGB_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"


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

@pytest.mark.parametrize(
    "model_type",
    ["dlrm_bid_shader", "ncf_deal_manager", "deal_yield_manager_floor", "deal_yield_manager_margin"],
)
def test_run_id_is_recorded_as_a_hyperparameter(model_type):
    params = _params(model_type, run_id="lt-abc123")
    assert params["HyperParameters"]["load_test_run_id"] == "lt-abc123"


@pytest.mark.parametrize("model_type", ["dlrm_bid_shader", "deal_yield_manager_floor"])
def test_key_is_absent_rather_than_empty_when_no_run_was_selected(model_type):
    """Scheduled retraining has no originating run. An empty string would read as
    "triggered from a run whose id we lost"."""
    assert "load_test_run_id" not in _params(model_type)["HyperParameters"]


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


# ---------------------------------------------------------------------------
# The deployment half of the chain
# ---------------------------------------------------------------------------

def test_registration_lambda_copies_the_run_id_onto_the_model_package():
    """A hyperparameter nothing reads back is not provenance. The Lambda that
    registers the version has to carry it across."""
    text = GOVERNANCE_CFN.read_text(encoding="utf-8")
    assert re.search(
        r"'load_test_run_id':\s*hyperparams\.get\('load_test_run_id',\s*''\)", text
    ), "registration Lambda does not copy load_test_run_id into CustomerMetadataProperties"


def test_frontend_sends_the_selected_run():
    """The selection existed in component state but was never in the request
    body, which is why no version had provenance."""
    panel = (
        REPO_ROOT / "source" / "frontend-react" / "src" / "components" / "GovernancePanel.jsx"
    ).read_text(encoding="utf-8")
    assert "run_id: selectedTrainingRun" in panel
