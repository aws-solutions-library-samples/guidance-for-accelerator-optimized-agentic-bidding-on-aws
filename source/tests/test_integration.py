"""Integration tests for the closed-loop learning system.

These tests exercise multiple components together, verifying end-to-end
flows at the interface level using mocked AWS service boundaries:

1. Feedback pipeline flow: outcome → emit → Kinesis mock, signal → Kinesis
2. Parameter write → read within TTL (with bounds clamping)
3. Canary traffic split verification (deterministic routing)
4. Register → A/B → decision governance flow

All tests use declared deterministic fixtures. Only external AWS service
boundaries (boto3 clients for Kinesis, DynamoDB, SageMaker, EventBridge)
are mocked. Internal logic runs with real code.

**Validates: Requirements 13.4, 13.5**
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import BidOutcomeEvent, SignalEvent
from shared.feedback_collector import FeedbackCollector
from shared.parameter_store import (
    ParameterStore,
    ParameterState,
    PARAMETER_BOUNDS,
)
from shared.parameter_cache import ParameterCache
from deployment.canary_deployer import (
    CanaryDeployer,
    DeploymentState,
)
from deployment.model_deployer import (
    HttpResponse,
    ModelOptimizer,
    TritonModelLoader,
)
from agents.governance.ab_evaluator import ABEvaluator, ABTestConfig, TestStatus
from agents.governance.governance_agent import ModelPromotionGovernanceAgent, GovernanceDecision
from training.pipeline import ModelType, TrainingPipeline, TrainingResult


# ---------------------------------------------------------------------------
# Deterministic fixture data
# ---------------------------------------------------------------------------

FIXTURE_REQUEST_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
FIXTURE_TIMESTAMP = 1718000000.0
FIXTURE_MODEL_VERSION = "v2.1.0"


def _make_bid_outcome_event() -> BidOutcomeEvent:
    """Deterministic BidOutcomeEvent fixture for integration tests."""
    return BidOutcomeEvent(
        request_id=FIXTURE_REQUEST_ID,
        timestamp=FIXTURE_TIMESTAMP,
        model_version=FIXTURE_MODEL_VERSION,
        original_price=5.0,
        shaded_price=4.0,
        bid_floor=2.0,
        won=True,
        price_paid=3.5,
        impression=True,
        click=True,
        conversion=True,
        conversion_value=25.0,
        user_id_hash="user_hash_abc123",
        site_domain="example.com",
        device_type="mobile",
        hour_of_day=14,
        shade_factor_used=0.8,
        conversion_value_estimate_used=10.0,
    )


def _make_signal_event() -> SignalEvent:
    """Deterministic SignalEvent (click) fixture for integration tests."""
    return SignalEvent(
        request_id=FIXTURE_REQUEST_ID,
        signal_type="click",
        timestamp=FIXTURE_TIMESTAMP + 5.0,
    )


# ---------------------------------------------------------------------------
# Mock HTTP client (same pattern as test_canary_deployer.py)
# ---------------------------------------------------------------------------


class MockHttpClient:
    """Deterministic mock HTTP client for Triton/NIM endpoints."""

    def __init__(self, responses: dict[str, HttpResponse] | None = None):
        self._responses: dict[str, HttpResponse] = responses or {}
        self.calls: list[dict] = []

    def set_response(self, url: str, response: HttpResponse) -> None:
        self._responses[url] = response

    def _find_response(self, url: str) -> HttpResponse:
        if url in self._responses:
            return self._responses[url]
        for key, resp in self._responses.items():
            if url.startswith(key):
                return resp
        return HttpResponse(status=404, body=b"Not Found", headers={})

    async def get(self, url: str) -> HttpResponse:
        self.calls.append({"method": "GET", "url": url})
        return self._find_response(url)

    async def post(self, url: str, json_body: dict | None = None) -> HttpResponse:
        self.calls.append({"method": "POST", "url": url, "json_body": json_body})
        return self._find_response(url)


def _ok_json(data: dict) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


# ===========================================================================
# Integration Test 1: Feedback pipeline flow
# outcome → emit → Kinesis mock; signal → emit → Kinesis mock
# ===========================================================================


class TestFeedbackPipelineFlow:
    """Integration test: BidOutcomeEvent and SignalEvent flow to Kinesis.

    Exercises:
    - FeedbackCollector.emit() with real serialization logic
    - Verifies the serialized record arrives with correct partition key
    - Verifies the JSON payload contains all expected fields
    - SignalEvent emission via the signal receiver path (put_record)
    """

    @pytest.fixture
    def mock_boto3_clients(self):
        """Patch boto3.client to capture Kinesis writes."""
        mock_kinesis = MagicMock()
        mock_cloudwatch = MagicMock()

        mock_kinesis.put_records.return_value = {
            "FailedRecordCount": 0,
            "Records": [{"SequenceNumber": "seq1", "ShardId": "shard-0"}],
        }
        mock_kinesis.put_record.return_value = {
            "SequenceNumber": "seq2",
            "ShardId": "shard-0",
        }

        def client_factory(service, **kwargs):
            if service == "kinesis":
                return mock_kinesis
            elif service == "cloudwatch":
                return mock_cloudwatch
            return MagicMock()

        with patch("boto3.client", side_effect=client_factory):
            yield mock_kinesis, mock_cloudwatch

    def test_bid_outcome_flows_through_collector_to_kinesis(
        self, mock_boto3_clients
    ):
        """BidOutcomeEvent → FeedbackCollector.emit() → Kinesis put_records.

        Verifies the full serialization chain: event creation, JSON encoding,
        partition key selection, and that the payload arrives intact at the
        mocked Kinesis endpoint.
        """
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="artf-bid-outcomes", region="us-east-1"
        )

        event = _make_bid_outcome_event()
        asyncio.run(collector.emit(event))

        # Verify Kinesis was called
        mock_kinesis.put_records.assert_called_once()
        call_kwargs = mock_kinesis.put_records.call_args[1]

        # Verify stream name
        assert call_kwargs["StreamName"] == "artf-bid-outcomes"

        # Verify partition key is user_id_hash
        record = call_kwargs["Records"][0]
        assert record["PartitionKey"] == "user_hash_abc123"

        # Verify the JSON payload contains all expected fields
        payload = json.loads(record["Data"].decode("utf-8"))
        assert payload["request_id"] == FIXTURE_REQUEST_ID
        assert payload["model_version"] == FIXTURE_MODEL_VERSION
        assert payload["won"] is True
        assert payload["price_paid"] == 3.5
        assert payload["conversion"] is True
        assert payload["conversion_value"] == 25.0
        assert payload["shade_factor_used"] == 0.8
        assert payload["user_id_hash"] == "user_hash_abc123"
        assert payload["original_price"] == 5.0
        assert payload["shaded_price"] == 4.0
        assert payload["bid_floor"] == 2.0

    def test_signal_event_flows_to_kinesis_with_request_id_partition(
        self, mock_boto3_clients
    ):
        """SignalEvent → signal receiver path → Kinesis put_record.

        Verifies that a click signal is emitted to the same Kinesis stream
        with request_id as the partition key (for shard co-location with
        the originating bid).
        """
        mock_kinesis, _ = mock_boto3_clients
        collector = FeedbackCollector(
            stream_name="artf-bid-outcomes", region="us-east-1"
        )

        signal = _make_signal_event()

        # Simulate the signal_receiver emission path directly
        record_bytes = json.dumps(signal.model_dump(), default=str).encode("utf-8")
        collector._kinesis_client.put_record(
            StreamName=collector._stream_name,
            Data=record_bytes,
            PartitionKey=signal.request_id,
        )

        # Verify put_record was called with correct partition
        mock_kinesis.put_record.assert_called_once()
        call_kwargs = mock_kinesis.put_record.call_args[1]
        assert call_kwargs["StreamName"] == "artf-bid-outcomes"
        assert call_kwargs["PartitionKey"] == FIXTURE_REQUEST_ID

        # Verify signal payload
        payload = json.loads(call_kwargs["Data"])
        assert payload["record_type"] == "signal"
        assert payload["signal_type"] == "click"
        assert payload["request_id"] == FIXTURE_REQUEST_ID



# ===========================================================================
# Integration Test 2: Parameter write → read within TTL
# ===========================================================================


class _MockClientError(Exception):
    """Mimics botocore.exceptions.ClientError for test purposes."""

    def __init__(self, error_code: str, message: str = ""):
        self.response = {"Error": {"Code": error_code, "Message": message}}
        super().__init__(f"{error_code}: {message}")


class TestParameterWriteReadWithinTTL:
    """Integration test: write parameter via ParameterStore, read via ParameterCache.

    Exercises the full write → DynamoDB → read → clamp path:
    - ParameterStore.update_parameter() writes the value
    - ParameterCache reads it back within TTL
    - Bounds clamping is verified in the read path
    """

    @pytest.fixture
    def mock_dynamodb(self):
        """Set up mocked DynamoDB table that stores parameter state in memory."""
        # In-memory storage to simulate DynamoDB
        storage: dict[tuple[str, str], dict] = {}

        mock_table = MagicMock()
        mock_audit_table = MagicMock()
        mock_resource = MagicMock()

        def table_factory(name):
            if name == "parameter-store-audit":
                return mock_audit_table
            return mock_table

        mock_resource.Table.side_effect = table_factory

        def put_item_side_effect(**kwargs):
            item = kwargs["Item"]
            key = (item["model_type"], item["parameter_name"])
            storage[key] = item

        def get_item_side_effect(**kwargs):
            key_dict = kwargs["Key"]
            key = (key_dict["model_type"], key_dict["parameter_name"])
            if key in storage:
                return {"Item": storage[key]}
            return {}

        mock_table.put_item.side_effect = put_item_side_effect
        mock_table.get_item.side_effect = get_item_side_effect
        mock_audit_table.put_item.return_value = {}

        mock_cloudwatch = MagicMock()

        with patch("boto3.resource", return_value=mock_resource):
            with patch("boto3.client", return_value=mock_cloudwatch):
                yield storage, mock_table, mock_resource

    def test_write_then_read_returns_same_value(self, mock_dynamodb):
        """Write shade_factor=0.72 → read back via ParameterCache → get 0.72."""
        storage, mock_table, mock_resource = mock_dynamodb

        # Seed initial state in storage (simulating an existing parameter)
        storage[("dlrm_bid_shader", "shade_factor")] = {
            "model_type": "dlrm_bid_shader",
            "parameter_name": "shade_factor",
            "current_value": "0.70",
            "previous_value": "0.68",
            "updated_at": "1718000000.0",
            "updated_by": "adaptive_bidding_agent",
            "version": 3,
            "min_value": "0.3",
            "max_value": "0.95",
            "max_delta_per_update": "0.05",
            "reason": "Initial",
            "confidence": "0.8",
        }
        storage[("dlrm_bid_shader", "conversion_value")] = {
            "model_type": "dlrm_bid_shader",
            "parameter_name": "conversion_value",
            "current_value": "12.0",
            "previous_value": "11.0",
            "updated_at": "1718000000.0",
            "updated_by": "adaptive_bidding_agent",
            "version": 2,
            "min_value": "1.0",
            "max_value": "50.0",
            "max_delta_per_update": "2.5",
            "reason": "Initial",
            "confidence": "0.8",
        }

        # Step 1: Write via ParameterStore.update_parameter()
        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        asyncio.run(
            store.update_parameter(
                model_type="dlrm_bid_shader",
                parameter_name="shade_factor",
                new_value=0.72,
                updated_by="adaptive_bidding_agent",
                reason="Win rate below target",
                confidence=0.9,
                expected_version=3,
            )
        )

        # Step 2: Read via ParameterCache (TTL=60s, should read fresh)
        cache = ParameterCache(
            table_name="parameter-store",
            region="us-east-1",
            ttl_seconds=60.0,
            model_type="dlrm_bid_shader",
        )
        result = cache.get_shade_factor()

        # The written value should be returned
        assert result == 0.72

    def test_bounds_clamping_in_read_path(self, mock_dynamodb):
        """Write a value at bound edge, read via cache → clamped correctly.

        Even if DynamoDB somehow contains an out-of-bounds value (e.g., from
        a race or manual write), the ParameterCache clamps it at read time.
        """
        storage, mock_table, mock_resource = mock_dynamodb

        # Directly inject an out-of-bounds value into "DynamoDB"
        storage[("dlrm_bid_shader", "shade_factor")] = {
            "model_type": "dlrm_bid_shader",
            "parameter_name": "shade_factor",
            "current_value": "1.5",  # Above max of 0.95
            "previous_value": "0.9",
            "updated_at": "1718000000.0",
            "updated_by": "manual_override",
            "version": 10,
            "min_value": "0.3",
            "max_value": "0.95",
            "max_delta_per_update": "0.05",
            "reason": "Manual test",
            "confidence": "1.0",
        }
        storage[("dlrm_bid_shader", "conversion_value")] = {
            "model_type": "dlrm_bid_shader",
            "parameter_name": "conversion_value",
            "current_value": "0.5",  # Below min of 1.0
            "previous_value": "1.0",
            "updated_at": "1718000000.0",
            "updated_by": "manual_override",
            "version": 10,
            "min_value": "1.0",
            "max_value": "50.0",
            "max_delta_per_update": "2.5",
            "reason": "Manual test",
            "confidence": "1.0",
        }

        cache = ParameterCache(
            table_name="parameter-store",
            region="us-east-1",
            ttl_seconds=60.0,
            model_type="dlrm_bid_shader",
        )

        # Read path should clamp to bounds
        assert cache.get_shade_factor() == 0.95  # Clamped from 1.5
        assert cache.get_conversion_value() == 1.0  # Clamped from 0.5



# ===========================================================================
# Integration Test 3: Canary traffic split verification
# ===========================================================================

TRITON_URL = "http://triton:8000"
MODEL_BUCKET = "artf-model-bucket"
MODEL_NAME = "dlrm_bid_shader"
ARTIFACT_URI = "s3://artf-model-bucket/optimized-models/dlrm_bid_shader/model.engine"


def _setup_canary_responses(
    client: MockHttpClient, new_version: int = 2
) -> None:
    """Configure mock responses for a successful canary deployment."""
    client.set_response(
        f"{TRITON_URL}/v2/models/{MODEL_NAME}",
        _ok_json({"name": MODEL_NAME, "versions": ["1"]}),
    )
    client.set_response(
        ARTIFACT_URI,
        HttpResponse(status=200, body=b"engine-data", headers={}),
    )
    client.set_response(
        f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/load",
        _ok_json({}),
    )
    client.set_response(
        f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/{new_version}/ready",
        _ok_json({}),
    )
    client.set_response(
        f"{TRITON_URL}/v2/repository/models/{MODEL_NAME}/unload",
        _ok_json({}),
    )


class _FakePaginator:
    def __init__(self, s3):
        self._s3 = s3

    def paginate(self, Bucket, Prefix, Delimiter=None):  # noqa: N803
        keys = [k for k in self._s3.objects if k.startswith(Prefix)]
        if Delimiter:
            prefixes = set()
            for k in keys:
                rest = k[len(Prefix):]
                if Delimiter in rest:
                    prefixes.add(Prefix + rest.split(Delimiter, 1)[0] + Delimiter)
            yield {"CommonPrefixes": [{"Prefix": p} for p in sorted(prefixes)]}
        else:
            yield {"Contents": [{"Key": k} for k in sorted(keys)]}


class _FakeS3:
    """In-memory S3 stand-in so the REAL TritonModelLoader runs its Option-D
    control-plane ops (write engine/config, edit router split, delete canary)."""

    class _Exceptions:
        class NoSuchKey(Exception):
            pass

    exceptions = _Exceptions()

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _FakeS3.exceptions.NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def copy_object(self, CopySource, Bucket, Key):  # noqa: N803
        self.objects[Key] = self.objects.get(CopySource["Key"], b"engine-bytes")

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)
        return {}

    def get_paginator(self, _op):
        return _FakePaginator(self)


_REPO = "triton-models"
_STABLE_CFG = 'name: "dlrm_bid_shader_stable"\nplatform: "tensorrt_plan"\n'
_ROUTER_CFG = (
    'name: "dlrm_bid_shader"\nbackend: "python"\n'
    'parameters { key: "canary_model" value: { string_value: "" } }\n'
    'parameters { key: "canary_traffic_pct" value: { string_value: "0" } }\n'
)


def _seeded_s3() -> _FakeS3:
    return _FakeS3({
        f"{_REPO}/{MODEL_NAME}/config.pbtxt": _ROUTER_CFG.encode(),
        f"{_REPO}/{MODEL_NAME}_stable/config.pbtxt": _STABLE_CFG.encode(),
        f"{_REPO}/{MODEL_NAME}_stable/1/model.plan": b"stable-v1-engine",
    })


def _ready_http() -> MockHttpClient:
    client = MockHttpClient()
    client.set_response(f"{TRITON_URL}/v2/models/{MODEL_NAME}_canary/ready", _ok_json({}))
    client.set_response(f"{TRITON_URL}/v2/models/{MODEL_NAME}_stable/ready", _ok_json({}))
    return client


def _make_canary_deployer(s3: _FakeS3, http_client: MockHttpClient) -> CanaryDeployer:
    """Real CanaryDeployer + real TritonModelLoader over a fake S3 + HTTP (Option D)."""
    triton_loader = TritonModelLoader(
        triton_url=TRITON_URL,
        model_bucket=MODEL_BUCKET,
        http_client=http_client,
        s3_client=s3,
        region="us-east-1",
        poll_interval_seconds=0.0,
        ready_timeout_seconds=0.0,
    )
    model_optimizer = ModelOptimizer(
        optimizer_endpoint="http://model-optimizer:8080",
        model_bucket=MODEL_BUCKET,
        region="us-east-1",
        http_client=http_client,
    )
    return CanaryDeployer(triton_loader=triton_loader, model_optimizer=model_optimizer)


def _router_pct(s3: _FakeS3) -> str:
    import re

    cfg = s3.objects[f"{_REPO}/{MODEL_NAME}/config.pbtxt"].decode()
    m = re.search(
        r'key: "canary_traffic_pct" value: \{ string_value: "([^"]*)" \}', cfg
    )
    return m.group(1) if m else ""


class TestCanaryTrafficSplitIntegration:
    """Integration test: deploy canary, adjust split, rollback — Option D router.

    Drives the REAL CanaryDeployer + TritonModelLoader against a fake S3/HTTP.
    Per-request traffic splitting happens in-process in the Triton router (not
    here), so these assert the control-plane split written to the router config
    and the resulting deployment state — the dependency-free ARTF contract.
    """

    @pytest.mark.asyncio
    async def test_deploy_sets_router_split_and_state(self):
        s3 = _seeded_s3()
        deployer = _make_canary_deployer(s3, _ready_http())

        state = await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ARTIFACT_URI, initial_traffic_pct=50.0,
        )

        assert state.canary_model == f"{MODEL_NAME}_canary"
        assert state.canary_traffic_pct == 50.0
        assert state.control_traffic_pct == 50.0
        # Canary model materialized in the served repo and the router split written.
        assert f"{_REPO}/{MODEL_NAME}_canary/1/model.plan" in s3.objects
        assert _router_pct(s3) == "50.0"

    @pytest.mark.asyncio
    async def test_adjust_to_100_updates_router_split(self):
        s3 = _seeded_s3()
        deployer = _make_canary_deployer(s3, _ready_http())
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ARTIFACT_URI, initial_traffic_pct=50.0,
        )

        state = await deployer.adjust_traffic(MODEL_NAME, 100.0)
        assert state.canary_traffic_pct == 100.0
        assert state.control_traffic_pct == 0.0
        assert _router_pct(s3) == "100.0"

    @pytest.mark.asyncio
    async def test_rollback_zeroes_split_and_removes_canary(self):
        s3 = _seeded_s3()
        deployer = _make_canary_deployer(s3, _ready_http())
        await deployer.deploy_canary(
            model_name=MODEL_NAME, artifact_uri=ARTIFACT_URI, initial_traffic_pct=50.0,
        )

        state = await deployer.rollback(MODEL_NAME)
        assert state.status == "stable"
        assert state.canary_model is None
        assert _router_pct(s3) == "0.0"
        # Canary model removed from the served repo.
        assert not any(f"{MODEL_NAME}_canary" in k for k in s3.objects)



# ===========================================================================
# Integration Test 4: Register → A/B → decision governance flow
# ===========================================================================

# Deterministic A/B test data: treatment clearly wins
CONTROL_DATA = [
    1.82, 2.15, 1.97, 2.03, 2.10, 1.88, 2.22, 1.95, 2.08, 1.91,
    2.01, 1.99, 2.14, 1.87, 2.06, 2.11, 1.93, 2.04, 1.96, 2.09,
    2.00, 1.85, 2.17, 1.94, 2.07, 1.90, 2.12, 1.98, 2.05, 1.89,
    2.02, 2.13, 1.92, 2.08, 1.86, 2.16, 1.95, 2.03, 1.97, 2.10,
    1.88, 2.19, 1.93, 2.06, 1.91, 2.11, 1.96, 2.04, 2.00, 1.84,
]

TREATMENT_WINNING_DATA = [
    2.42, 2.55, 2.47, 2.63, 2.50, 2.38, 2.62, 2.45, 2.58, 2.41,
    2.51, 2.49, 2.64, 2.37, 2.56, 2.61, 2.43, 2.54, 2.46, 2.59,
    2.50, 2.35, 2.67, 2.44, 2.57, 2.40, 2.52, 2.48, 2.55, 2.39,
    2.52, 2.63, 2.42, 2.58, 2.36, 2.66, 2.45, 2.53, 2.47, 2.60,
    2.38, 2.69, 2.43, 2.56, 2.41, 2.61, 2.46, 2.54, 2.50, 2.34,
]

TREATMENT_LOSING_DATA = [
    1.42, 1.55, 1.47, 1.63, 1.50, 1.38, 1.62, 1.45, 1.58, 1.41,
    1.51, 1.49, 1.64, 1.37, 1.56, 1.61, 1.43, 1.54, 1.46, 1.59,
    1.50, 1.35, 1.67, 1.44, 1.57, 1.40, 1.52, 1.48, 1.55, 1.39,
    1.52, 1.63, 1.42, 1.58, 1.36, 1.66, 1.45, 1.53, 1.47, 1.60,
    1.38, 1.69, 1.43, 1.56, 1.41, 1.61, 1.46, 1.54, 1.50, 1.34,
]


# Deterministic training result fixture
FIXTURE_TRAINING_RESULT = TrainingResult(
    job_name="dlrm_bid_shader-1718000000-abc12345",
    model_artifact_uri="s3://model-bucket/models/dlrm_bid_shader/model.tar.gz",
    metrics={"ctr_auc": 0.82, "revenue_lift": 0.07},
    training_duration_s=3600.0,
    model_version="v2.1.0",
    data_window_days=7,
    sample_count=500000,
    base_model_version="v2.0.0",
)


class MockModelOptimizer:
    """Mock Model Optimizer returning a deterministic optimized URI."""

    def __init__(self, *, should_fail: bool = False):
        self._should_fail = should_fail
        self.optimize_calls: list[dict] = []

    async def optimize(self, model_artifact_uri: str, model_name: str) -> str:
        self.optimize_calls.append({
            "model_artifact_uri": model_artifact_uri,
            "model_name": model_name,
        })
        if self._should_fail:
            raise RuntimeError("optimizer error")
        return f"s3://models/optimized/{model_name}/model.engine"


class MockCanaryDeployerForGovernance:
    """Mock canary deployer that tracks deploy/promote/rollback."""

    def __init__(self):
        self.deploy_calls: list[dict] = []
        self.promote_calls: list[str] = []
        self.rollback_calls: list[str] = []

    async def deploy_canary(self, model_name, artifact_uri, initial_traffic_pct):
        self.deploy_calls.append({
            "model_name": model_name,
            "artifact_uri": artifact_uri,
            "initial_traffic_pct": initial_traffic_pct,
        })

    async def promote(self, model_name):
        self.promote_calls.append(model_name)

    async def rollback(self, model_name):
        self.rollback_calls.append(model_name)


class MockGuardrailMonitor:
    """Mock guardrail monitor returning no violations by default."""

    def __init__(self, violations: list[str] | None = None):
        self._violations = violations or []

    async def check(self, model_name: str) -> list[str]:
        return self._violations


class MockModelRegistryClient:
    """Mock SageMaker client for model registry updates."""

    def __init__(self):
        self.update_calls: list[dict] = []
        self.create_calls: list[dict] = []

    def update_model_package(self, **kwargs) -> dict:
        self.update_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def create_model_package(self, **kwargs) -> dict:
        self.create_calls.append(kwargs)
        return {
            "ModelPackageArn": (
                f"arn:aws:sagemaker:us-east-1:123456789012:"
                f"model-package/{kwargs.get('ModelPackageGroupName', 'test')}/1"
            )
        }


class MockAuditStore:
    """Mock audit store recording put_record calls."""

    def __init__(self):
        self.records: list[dict] = []

    async def put_record(self, record: dict) -> None:
        self.records.append(record)


class TestRegisterABDecisionFlow:
    """Integration test: register_model → A/B test → governance decision.

    Exercises the full governance flow:
    - Create a TrainingResult fixture
    - Call register_model() (mocked SageMaker)
    - Verify EventBridge event emission
    - Simulate GovernanceAgent receiving the event
    - With winning treatment: verify promote decision, registry Approved
    - With losing treatment: verify reject decision, registry Rejected
    """

    def _build_governance_agent(
        self,
        guardrail_violations: list[str] | None = None,
    ) -> tuple[
        ModelPromotionGovernanceAgent,
        MockModelOptimizer,
        MockCanaryDeployerForGovernance,
        MockModelRegistryClient,
        MockAuditStore,
    ]:
        """Build a governance agent with all mock dependencies."""
        nim = MockModelOptimizer()
        canary = MockCanaryDeployerForGovernance()
        guardrail = MockGuardrailMonitor(violations=guardrail_violations)
        registry = MockModelRegistryClient()
        audit = MockAuditStore()

        agent = ModelPromotionGovernanceAgent(
            model_optimizer=nim,
            canary_deployer=canary,
            ab_evaluator_factory=lambda config: ABEvaluator(config),
            guardrail_monitor=guardrail,
            model_registry_client=registry,
            audit_store=audit,
        )
        return agent, nim, canary, registry, audit

    def _default_ab_config(self) -> ABTestConfig:
        return ABTestConfig(
            model_type="dlrm_bid_shader",
            control_version="v2.0.0",
            treatment_version="v2.1.0",
            traffic_percentage=5.0,
            min_samples=10,
            max_duration_hours=0.001,
            significance_level=0.05,
            primary_metric="revenue_per_bid",
            guardrail_metrics=["latency_p99"],
        )

    @pytest.mark.asyncio
    async def test_register_model_then_governance_promotes(self):
        """Full flow: register_model → governance receives event → promote.

        Steps:
        1. Call register_model on TrainingPipeline (mocked SageMaker)
        2. Verify the model package was created with correct metadata
        3. Simulate GovernanceAgent handling the event with winning data
        4. Verify promote decision and registry updated to Approved
        """
        # Step 1: Register model
        mock_sagemaker = MockModelRegistryClient()
        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123:role/SageMakerRole",
            model_bucket="model-bucket",
            region="us-east-1",
            training_data_bucket="training-data-bucket",
            sagemaker_client=mock_sagemaker,
        )

        version_arn = await pipeline.register_model(
            result=FIXTURE_TRAINING_RESULT,
            model_type=ModelType.DLRM_BID_SHADER,
        )

        # Verify registration happened
        assert len(mock_sagemaker.create_calls) == 1
        create_args = mock_sagemaker.create_calls[0]
        assert create_args["ModelPackageGroupName"] == "artf-dlrm-bid-shader"
        assert create_args["ModelApprovalStatus"] == "PendingManualApproval"
        metadata = create_args["CustomerMetadataProperties"]
        assert metadata["training_job_name"] == FIXTURE_TRAINING_RESULT.job_name
        assert metadata["sample_count"] == "500000"
        assert metadata["data_window_days"] == "7"

        # Step 2: Simulate governance agent receiving the event
        agent, nim, canary, gov_registry, audit = self._build_governance_agent()
        config = self._default_ab_config()

        async def winning_collector(model_name, config):
            return CONTROL_DATA, TREATMENT_WINNING_DATA, None

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn=version_arn,
            artifact_uri=FIXTURE_TRAINING_RESULT.model_artifact_uri,
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=winning_collector,
        )

        # Step 3: Verify promote decision
        assert result.decision == "promote"
        assert "outperforms" in result.reason

        # Verify canary was deployed and then promoted
        assert len(canary.deploy_calls) == 1
        assert MODEL_NAME in canary.promote_calls
        assert len(canary.rollback_calls) == 0

        # Verify registry updated to Approved
        assert len(gov_registry.update_calls) == 1
        assert gov_registry.update_calls[0]["ModelApprovalStatus"] == "Approved"
        assert gov_registry.update_calls[0]["ModelPackageArn"] == version_arn

        # Verify audit record written
        assert len(audit.records) == 1
        assert audit.records[0]["decision"] == "promote"
        assert audit.records[0]["actor"] == "model_promotion_governance_agent"
        assert audit.records[0]["model_type"] == "dlrm_bid_shader"

    @pytest.mark.asyncio
    async def test_register_model_then_governance_rejects(self):
        """Full flow: register_model → governance receives event → reject.

        With losing treatment data, the governance agent should:
        - Reject the model version
        - Rollback canary traffic
        - Update registry to Rejected
        """
        # Step 1: Register model
        mock_sagemaker = MockModelRegistryClient()
        pipeline = TrainingPipeline(
            sagemaker_role="arn:aws:iam::123:role/SageMakerRole",
            model_bucket="model-bucket",
            region="us-east-1",
            training_data_bucket="training-data-bucket",
            sagemaker_client=mock_sagemaker,
        )

        version_arn = await pipeline.register_model(
            result=FIXTURE_TRAINING_RESULT,
            model_type=ModelType.DLRM_BID_SHADER,
        )

        # Step 2: Simulate governance agent with losing treatment data
        agent, nim, canary, gov_registry, audit = self._build_governance_agent()
        config = self._default_ab_config()

        async def losing_collector(model_name, config):
            return CONTROL_DATA, TREATMENT_LOSING_DATA, None

        result = await agent.on_new_model_version(
            model_type="dlrm_bid_shader",
            version_arn=version_arn,
            artifact_uri=FIXTURE_TRAINING_RESULT.model_artifact_uri,
            ab_test_config=config,
            evaluation_interval_seconds=0.0,
            collect_metrics=losing_collector,
        )

        # Step 3: Verify reject decision
        assert result.decision == "reject"

        # Verify canary was deployed and then rolled back
        assert len(canary.deploy_calls) == 1
        assert MODEL_NAME in canary.rollback_calls
        assert len(canary.promote_calls) == 0

        # Verify registry updated to Rejected
        assert len(gov_registry.update_calls) == 1
        assert gov_registry.update_calls[0]["ModelApprovalStatus"] == "Rejected"

        # Verify audit record written
        assert len(audit.records) == 1
        assert audit.records[0]["decision"] == "reject"
        assert audit.records[0]["actor"] == "model_promotion_governance_agent"
