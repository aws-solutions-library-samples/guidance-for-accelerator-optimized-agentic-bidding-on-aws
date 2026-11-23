"""POST /v1/gpu/start and /v1/gpu/stop: node counts the README cost table depends on."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from orchestrator import app as oapp


@pytest.fixture
def eks(monkeypatch):
    client = MagicMock()
    client.describe_nodegroup.return_value = {
        "nodegroup": {"scalingConfig": {"minSize": 0, "maxSize": 3, "desiredSize": 0}}
    }
    fake_boto3 = types.SimpleNamespace(client=MagicMock(return_value=client))
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return client


def _call(handler):
    resp = asyncio.run(handler(MagicMock()))
    return resp.status_code, json.loads(resp.body)


def test_start_scales_to_one_node_and_keeps_max(eks):
    status, body = _call(oapp.gpu_start)
    assert status == 200
    assert body["desiredSize"] == 1
    eks.update_nodegroup_config.assert_called_once()
    scaling = eks.update_nodegroup_config.call_args.kwargs["scalingConfig"]
    assert scaling == {"minSize": 1, "maxSize": 3, "desiredSize": 1}


def test_stop_scales_to_zero_and_keeps_max(eks):
    status, _ = _call(oapp.gpu_stop)
    assert status == 200
    scaling = eks.update_nodegroup_config.call_args.kwargs["scalingConfig"]
    assert scaling == {"minSize": 0, "maxSize": 3, "desiredSize": 0}


def test_start_reports_an_eks_error(eks):
    eks.describe_nodegroup.side_effect = RuntimeError("AccessDenied")
    status, body = _call(oapp.gpu_start)
    assert status == 500
    assert "AccessDenied" in body["error"]
    eks.update_nodegroup_config.assert_not_called()


_ICE = ("Could not launch On-Demand Instances. InsufficientInstanceCapacity - We currently do "
        "not have sufficient g6.xlarge capacity in the Availability Zone you requested "
        "(us-west-2a). Our system will be working on provisioning additional capacity.")


def _status_with(eks, monkeypatch, *, desired, issues):
    eks.describe_nodegroup.return_value = {"nodegroup": {
        "status": "DEGRADED" if issues else "ACTIVE",
        "scalingConfig": {"minSize": 1, "maxSize": 3, "desiredSize": desired},
        "health": {"issues": issues},
        "modifiedAt": 0,
    }}
    monkeypatch.setenv("TRITON_URL", "127.0.0.1:9")  # nothing listens: Triton not ready
    return _call(oapp.gpu_status)


def test_status_reports_no_capacity_with_the_ec2_reason(eks, monkeypatch):
    status, body = _status_with(eks, monkeypatch, desired=1, issues=[
        {"code": "AsgInstanceLaunchFailures", "message": _ICE}])
    assert status == 200
    assert body["tritonState"] == "no_capacity"
    assert "sufficient g6.xlarge capacity" in body["tritonDetail"]
    assert "(us-west-2a)" in body["tritonDetail"]
    assert "Our system will be working" not in body["tritonDetail"]
    assert "Re-running deploy.sh" in body["tritonDetail"]


def test_status_without_a_capacity_issue_is_not_no_capacity(eks, monkeypatch):
    _, body = _status_with(eks, monkeypatch, desired=1, issues=[])
    assert body["tritonState"] in ("starting", "blocked")


def test_a_stopped_group_with_an_old_issue_is_stopped(eks, monkeypatch):
    _, body = _status_with(eks, monkeypatch, desired=0, issues=[
        {"code": "AsgInstanceLaunchFailures", "message": _ICE}])
    assert body["tritonState"] == "stopped"
