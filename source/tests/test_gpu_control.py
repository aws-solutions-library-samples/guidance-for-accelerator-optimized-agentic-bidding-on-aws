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
