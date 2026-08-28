"""Tests that read_model_versions surfaces the governance agent's stated reason.

Why this needs a second API call: ModelPackageSummaryList (from
list_model_packages) carries ModelApprovalStatus but NOT ApprovalDescription, so
the reason a version was rejected is simply absent from the list response. Only
DescribeModelPackage returns it.

The reason text used here is the real ApprovalDescription read from the deployed
registry, where dlrm-bid-shader version 2 was rejected because the TensorRT
optimization step timed out against the VPC optimizer proxy — a failed pipeline
step, not a model that lost its A/B test.
"""

from __future__ import annotations

import datetime
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop_demo.readers import read_model_versions  # noqa: E402

REAL_OPTIMIZATION_FAILURE = (
    "Model optimization failed: Model optimization failed for dlrm_bid_shader: "
    "status=502, body=proxy invoke failed: Read timeout on endpoint URL: "
    '"https://lambda.us-east-1.amazonaws.com/2015-03-31/functions/'
    'arn%3Aaws%3Alambda%3Aus-east-1%3A960328030835%3Afunction%3Anv5-vpc-optimizer-proxy/invocations"'
)


def _client(*, describe_side_effect=None):
    sm = mock.Mock()
    sm.list_model_packages.return_value = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageArn": "arn:aws:sagemaker:us-east-1:1:model-package/g/2",
                "ModelPackageVersion": 2,
                "ModelApprovalStatus": "Rejected",
                "ModelPackageStatus": "Completed",
                "CreationTime": datetime.datetime(2026, 8, 27, 22, 6, 57),
            },
            {
                "ModelPackageArn": "arn:aws:sagemaker:us-east-1:1:model-package/g/1",
                "ModelPackageVersion": 1,
                "ModelApprovalStatus": "Approved",
                "ModelPackageStatus": "Completed",
                "CreationTime": datetime.datetime(2026, 8, 22, 2, 47, 37),
            },
        ]
    }
    if describe_side_effect is not None:
        sm.describe_model_package.side_effect = describe_side_effect
    else:
        sm.describe_model_package.side_effect = lambda ModelPackageName: (
            {"ApprovalDescription": REAL_OPTIMIZATION_FAILURE}
            if ModelPackageName.endswith("/2")
            else {}
        )
    return sm


def test_rejection_reason_is_returned():
    versions = read_model_versions(_client(), "g")
    rejected = next(v for v in versions if v["version"] == 2)
    assert rejected["approval_description"] == REAL_OPTIMIZATION_FAILURE


def test_version_without_a_reason_reports_none_not_a_placeholder():
    versions = read_model_versions(_client(), "g")
    approved = next(v for v in versions if v["version"] == 1)
    assert approved["approval_description"] is None


def test_empty_approval_description_normalises_to_none():
    """An empty string must read as "no reason recorded", not as a blank reason."""
    sm = _client(describe_side_effect=lambda ModelPackageName: {"ApprovalDescription": ""})
    versions = read_model_versions(sm, "g")
    assert all(v["approval_description"] is None for v in versions)


def test_describe_failure_does_not_break_the_listing():
    """The registry listing is the caller's primary content; one failed
    per-version describe must not take it down."""
    sm = _client(describe_side_effect=RuntimeError("AccessDeniedException"))
    versions = read_model_versions(sm, "g")
    assert len(versions) == 2
    assert all(v["approval_description"] is None for v in versions)
    # Statuses still come through from the list call.
    assert {v["approval_status"] for v in versions} == {"Rejected", "Approved"}


def test_existing_fields_are_unchanged():
    """Additive change — nothing the UI already rendered may shift."""
    versions = read_model_versions(_client(), "g")
    v2 = next(v for v in versions if v["version"] == 2)
    assert v2["approval_status"] == "Rejected"
    assert v2["status"] == "Completed"
    assert v2["model_package_arn"].endswith("/2")
    assert v2["created_at"] == "2026-08-27T22:06:57"


def test_describe_is_called_once_per_version():
    sm = _client()
    read_model_versions(sm, "g")
    assert sm.describe_model_package.call_count == 2
