"""Unit tests for orchestrator.demo_dataset_api (FR1.1, FR1.2, FR1.5, FR1.6, FR1.10, FR1.11).

A minimal Starlette TestClient app with the three routes and a stubbed S3
client injected through demo_dataset._s3_client.
"""
from __future__ import annotations

import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Route  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from orchestrator.demo_dataset_api import (  # noqa: E402
    estimate_handler,
    generate_handler,
    status_handler,
)

_app = Starlette(routes=[
    Route("/v1/governance/demo-dataset", status_handler, methods=["GET"]),
    Route("/v1/governance/demo-dataset", generate_handler, methods=["POST"]),
    Route("/v1/governance/demo-dataset/estimate", estimate_handler, methods=["GET"]),
])
client = TestClient(_app)


@pytest.fixture
def bucket(monkeypatch):
    monkeypatch.setenv("TRAINING_DATA_BUCKET", "artf-training-data-bucket")


@pytest.fixture
def s3():
    stub = MagicMock()
    stub.list_objects_v2.return_value = {"Contents": [], "IsTruncated": False}
    with patch("orchestrator.demo_dataset._s3_client", return_value=stub):
        yield stub


class TestGenerate:
    def test_unconfirmed_request_is_refused_and_writes_nothing(self, bucket, s3):
        resp = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "deal_yield_manager_floor", "params": {"clear_share": 0.6}},
        )
        assert resp.status_code == 422
        assert resp.json()["reason"] == "not_confirmed"
        s3.put_object.assert_not_called()
        s3.delete_objects.assert_not_called()

    def test_confirmed_request_writes_under_the_model_prefix(self, bucket, s3):
        resp = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "deal_yield_manager_margin", "params": {"clear_share": 0.6},
                  "seed": 11, "confirmed": True},
        )
        assert resp.status_code == 201
        payload = resp.json()
        assert payload["exists"] is True
        assert payload["demonstration_only"] is True
        assert payload["metadata"]["seed"] == 11
        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert keys == [
            "demo-training-data-deal-yield-margin/part-00000.parquet",
            "demo-training-data-deal-yield-margin/_demo_metadata.json",
        ]
        assert all(c.kwargs["Bucket"] == "artf-training-data-bucket" for c in s3.put_object.call_args_list)

    def test_out_of_bound_rate_returns_422_naming_the_field(self, bucket, s3):
        resp = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "dlrm_bid_shader", "params": {"win_rate": 0.05}, "confirmed": True},
        )
        assert resp.status_code == 422
        assert resp.json()["field"] == "win_rate"
        s3.put_object.assert_not_called()

    @pytest.mark.parametrize("model_type", ["ncf_deal_manager", "widedeep_segment_activator", "bogus"])
    def test_untrainable_model_type_returns_422(self, bucket, s3, model_type):
        resp = client.post(
            "/v1/governance/demo-dataset", json={"model_type": model_type, "confirmed": True},
        )
        assert resp.status_code == 422
        assert resp.json()["reason"] == "unsupported_model_type"

    def test_string_rate_and_unknown_field_are_refused(self, bucket, s3):
        string_rate = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "dlrm_bid_shader", "params": {"win_rate": "0.9"}, "confirmed": True},
        )
        extra = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "dlrm_bid_shader", "confirmed": True, "overwrite": True},
        )
        assert string_rate.status_code == 422
        assert extra.status_code == 422
        s3.put_object.assert_not_called()

    def test_unset_bucket_returns_503(self, monkeypatch, s3):
        monkeypatch.delenv("TRAINING_DATA_BUCKET", raising=False)
        resp = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "deal_yield_manager_floor", "confirmed": True},
        )
        assert resp.status_code == 503
        s3.put_object.assert_not_called()

    def test_s3_failure_returns_json_500(self, bucket, s3):
        s3.put_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject"
        )
        resp = client.post(
            "/v1/governance/demo-dataset",
            json={"model_type": "deal_yield_manager_floor", "confirmed": True},
        )
        assert resp.status_code == 500
        assert resp.json()["reason"] == "storage_error"

    def test_unfit_dataset_returns_422_and_writes_nothing(self, bucket, s3):
        from orchestrator import demo_dataset

        with patch.object(
            demo_dataset, "generate_demo_dataset",
            side_effect=demo_dataset.DemoDatasetUnfitError("'label_conversion' has no positives"),
        ):
            resp = client.post(
                "/v1/governance/demo-dataset",
                json={"model_type": "dlrm_bid_shader", "confirmed": True},
            )
        assert resp.status_code == 422
        assert resp.json()["reason"] == "dataset_unfit"
        assert "no positives" in resp.json()["error"]
        s3.put_object.assert_not_called()


class TestStatusAndEstimate:
    def test_status_without_dataset_reports_not_exists(self, bucket, s3):
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
        )
        resp = client.get("/v1/governance/demo-dataset?model_type=dlrm_bid_shader")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["exists"] is False
        assert payload["metadata"] is None
        assert payload["prefix"] == "demo-training-data/"
        assert [p["name"] for p in payload["params"]] == ["win_rate", "click_rate", "conversion_rate"]

    def test_status_with_dataset_returns_metadata(self, bucket, s3):
        s3.get_object.return_value = {
            "Body": io.BytesIO(json.dumps({"generated_at": "2026-11-15T10:00:00Z", "rows_written": 5}).encode())
        }
        resp = client.get("/v1/governance/demo-dataset?model_type=deal_yield_manager_floor")
        assert resp.status_code == 200
        assert resp.json()["exists"] is True
        assert resp.json()["metadata"]["rows_written"] == 5

    def test_estimate_returns_default_sizing(self):
        resp = client.get("/v1/governance/demo-dataset/estimate?model_type=dlrm_bid_shader")
        assert resp.status_code == 200
        sizing = resp.json()["sizing"]
        assert sizing["rows"] == 55_556
        assert round(sizing["expected_positives"]) == 50

    def test_estimate_rejects_non_number(self):
        resp = client.get(
            "/v1/governance/demo-dataset/estimate?model_type=deal_yield_manager_floor&clear_share=abc"
        )
        assert resp.status_code == 422
        assert resp.json()["field"] == "clear_share"
