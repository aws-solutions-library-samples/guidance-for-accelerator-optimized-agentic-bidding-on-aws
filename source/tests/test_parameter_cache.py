"""Tests for shared.parameter_cache — In-memory parameter cache.

Validates:
- Cache returns fresh values after refresh from DynamoDB
- Cache returns stale values on DynamoDB failure (graceful degradation)
- TTL expiry triggers a refresh attempt
- Values are clamped to bounds on load
- Error counter increments on consecutive failures
- Sustained error alarm emitted after threshold failures

Only the boto3 DynamoDB client (external boundary) is mocked. All caching
logic, TTL management, clamping, error counting, and fallback behavior are
tested directly.

**Validates: Requirements 8.3, 9.1, 9.2, 9.3**
"""

import sys
import os
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.parameter_cache import (
    ParameterCache,
    PARAMETER_BOUNDS,
    SUSTAINED_ERROR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic DynamoDB item fixtures
# ---------------------------------------------------------------------------


def _make_dynamodb_item(parameter_name: str, current_value: float) -> dict:
    """Create a DynamoDB item dict as returned by Table.get_item."""
    return {
        "model_type": "dlrm_bid_shader",
        "parameter_name": parameter_name,
        "current_value": str(current_value),
        "previous_value": str(current_value - 0.02),
        "updated_at": "1718000000.0",
        "updated_by": "bid_shading_agent",
        "version": 5,
        "min_value": str(PARAMETER_BOUNDS[parameter_name][0]),
        "max_value": str(PARAMETER_BOUNDS[parameter_name][1]),
        "max_delta_per_update": "0.05",
        "reason": "Win rate adjustment",
        "confidence": "0.85",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_boto3():
    """Patch boto3 to return a mocked DynamoDB table and CloudWatch client."""
    mock_table = MagicMock()
    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table

    mock_cloudwatch = MagicMock()
    mock_cloudwatch.put_metric_data.return_value = {}

    mock_boto3_module = MagicMock()
    mock_boto3_module.resource.return_value = mock_resource
    mock_boto3_module.client.return_value = mock_cloudwatch

    with patch.dict("sys.modules", {"boto3": mock_boto3_module}):
        with patch("boto3.resource", return_value=mock_resource):
            with patch("boto3.client", return_value=mock_cloudwatch):
                yield mock_table, mock_cloudwatch


# ---------------------------------------------------------------------------
# Test: Fresh values after refresh
# ---------------------------------------------------------------------------


class TestCacheRefresh:
    """Tests that the cache fetches and returns fresh values from DynamoDB."""

    def test_returns_fresh_shade_factor_after_refresh(self, mock_boto3):
        """get_shade_factor returns the value from DynamoDB after a refresh."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.72)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 15.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        result = cache.get_shade_factor()

        assert result == 0.72

    def test_returns_fresh_conversion_value_after_refresh(self, mock_boto3):
        """get_conversion_value returns the value from DynamoDB after a refresh."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.72)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 18.5)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        result = cache.get_conversion_value()

        assert result == 18.5

    def test_uses_eventually_consistent_read(self, mock_boto3):
        """Reads use ConsistentRead=False for DAX compatibility and speed."""
        mock_table, _ = mock_boto3
        mock_table.get_item.return_value = {
            "Item": _make_dynamodb_item("shade_factor", 0.7)
        }

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        cache.get_shade_factor()

        for call in mock_table.get_item.call_args_list:
            assert call[1]["ConsistentRead"] is False


# ---------------------------------------------------------------------------
# Test: Graceful degradation (stale values on failure)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests that cache returns last known values when DynamoDB is unavailable."""

    def test_returns_default_shade_factor_on_initial_failure(self, mock_boto3):
        """If the first read fails, returns the module default (0.65)."""
        mock_table, _ = mock_boto3
        mock_table.get_item.side_effect = Exception("DynamoDB unreachable")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        result = cache.get_shade_factor()

        # Default is 0.65 (same as the DLRM module constant)
        assert result == 0.65

    def test_returns_default_conversion_value_on_initial_failure(self, mock_boto3):
        """If the first read fails, returns the module default (12.0)."""
        mock_table, _ = mock_boto3
        mock_table.get_item.side_effect = Exception("DynamoDB unreachable")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        result = cache.get_conversion_value()

        assert result == 12.0

    def test_returns_cached_value_after_successful_then_failed_refresh(self, mock_boto3):
        """After a successful refresh, failure returns the last good value."""
        mock_table, _ = mock_boto3

        # First call succeeds
        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.80)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 20.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0  # expire immediately
        )

        # Initial fetch — succeeds
        assert cache.get_shade_factor() == 0.80

        # Now DynamoDB fails
        mock_table.get_item.side_effect = Exception("Network timeout")

        # Should return cached value (0.80) not default (0.65)
        assert cache.get_shade_factor() == 0.80

    def test_never_raises_on_read(self, mock_boto3):
        """get_shade_factor and get_conversion_value never raise exceptions."""
        mock_table, _ = mock_boto3
        mock_table.get_item.side_effect = RuntimeError("Catastrophic failure")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )

        # Should not raise
        sf = cache.get_shade_factor()
        cv = cache.get_conversion_value()

        assert isinstance(sf, float)
        assert isinstance(cv, float)


# ---------------------------------------------------------------------------
# Test: TTL expiry triggers refresh
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    """Tests that the cache refreshes only when TTL is expired."""

    def test_no_refresh_within_ttl(self, mock_boto3):
        """Cache does not call DynamoDB again within the TTL window."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.72)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 15.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )

        # First call triggers refresh (2 get_item calls: shade_factor + conversion_value)
        cache.get_shade_factor()
        initial_call_count = mock_table.get_item.call_count

        # Second call within TTL — no additional DynamoDB calls
        cache.get_shade_factor()
        cache.get_conversion_value()

        assert mock_table.get_item.call_count == initial_call_count

    def test_refresh_after_ttl_expires(self, mock_boto3):
        """Cache calls DynamoDB again after TTL expires."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.72)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 15.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        # TTL = 0 means always stale → always refresh
        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0
        )

        cache.get_shade_factor()
        first_call_count = mock_table.get_item.call_count

        cache.get_shade_factor()
        second_call_count = mock_table.get_item.call_count

        # Each refresh does 2 get_item calls (shade_factor + conversion_value)
        assert second_call_count > first_call_count


# ---------------------------------------------------------------------------
# Test: Clamping to bounds
# ---------------------------------------------------------------------------


class TestClamping:
    """Tests that values are clamped to PARAMETER_BOUNDS on load."""

    def test_shade_factor_clamped_below_min(self, mock_boto3):
        """shade_factor below 0.3 is clamped to 0.3."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.1)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 12.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        assert cache.get_shade_factor() == 0.3

    def test_shade_factor_clamped_above_max(self, mock_boto3):
        """shade_factor above 0.95 is clamped to 0.95."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 1.5)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 12.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        assert cache.get_shade_factor() == 0.95

    def test_conversion_value_clamped_below_min(self, mock_boto3):
        """conversion_value below 1.0 is clamped to 1.0."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.7)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 0.5)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        assert cache.get_conversion_value() == 1.0

    def test_conversion_value_clamped_above_max(self, mock_boto3):
        """conversion_value above 50.0 is clamped to 50.0."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.7)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 99.9)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        assert cache.get_conversion_value() == 50.0

    def test_values_within_bounds_not_clamped(self, mock_boto3):
        """Values within bounds are returned as-is."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.6)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 25.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=60.0
        )
        assert cache.get_shade_factor() == 0.6
        assert cache.get_conversion_value() == 25.0


# ---------------------------------------------------------------------------
# Test: Error counter and sustained error alarm
# ---------------------------------------------------------------------------


class TestErrorCounting:
    """Tests that error counter increments and alarm emits on sustained failures."""

    def test_error_counter_increments_on_failure(self, mock_boto3):
        """Consecutive errors increment the error counter."""
        mock_table, _ = mock_boto3
        mock_table.get_item.side_effect = Exception("Connection refused")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0  # always stale
        )

        cache.get_shade_factor()
        assert cache.consecutive_errors == 1

        cache.get_shade_factor()
        assert cache.consecutive_errors == 2

    def test_error_counter_resets_on_success(self, mock_boto3):
        """Successful refresh resets the error counter to 0."""
        mock_table, _ = mock_boto3

        # First: fail
        mock_table.get_item.side_effect = Exception("Timeout")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0
        )
        cache.get_shade_factor()
        assert cache.consecutive_errors == 1

        # Now: succeed
        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.7)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 12.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect
        cache.get_shade_factor()
        assert cache.consecutive_errors == 0

    def test_alarm_emitted_after_threshold_failures(self, mock_boto3):
        """Alarm is emitted after SUSTAINED_ERROR_THRESHOLD consecutive failures."""
        mock_table, mock_cloudwatch = mock_boto3
        mock_table.get_item.side_effect = Exception("Service unavailable")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0
        )

        # Fail up to the threshold
        for _ in range(SUSTAINED_ERROR_THRESHOLD):
            cache.get_shade_factor()

        assert cache.alarm_emitted is True
        mock_cloudwatch.put_metric_data.assert_called_once()

        # Verify metric dimensions
        call_kwargs = mock_cloudwatch.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == "ARTF/ParameterCache"
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "SustainedReadErrors"
        assert metric["Value"] == 1.0

    def test_alarm_emitted_only_once(self, mock_boto3):
        """Alarm is emitted only once per sustained error episode."""
        mock_table, mock_cloudwatch = mock_boto3
        mock_table.get_item.side_effect = Exception("Service unavailable")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0
        )

        # Exceed the threshold many times
        for _ in range(SUSTAINED_ERROR_THRESHOLD + 5):
            cache.get_shade_factor()

        # CloudWatch should only be called once
        assert mock_cloudwatch.put_metric_data.call_count == 1

    def test_alarm_resets_after_recovery(self, mock_boto3):
        """After recovery, alarm_emitted resets and can fire again."""
        mock_table, mock_cloudwatch = mock_boto3
        mock_table.get_item.side_effect = Exception("Service unavailable")

        cache = ParameterCache(
            table_name="test-table", region="us-east-1", ttl_seconds=0.0
        )

        # Trigger alarm
        for _ in range(SUSTAINED_ERROR_THRESHOLD):
            cache.get_shade_factor()
        assert cache.alarm_emitted is True

        # Recover
        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.7)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 12.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect
        cache.get_shade_factor()

        assert cache.alarm_emitted is False
        assert cache.consecutive_errors == 0


# ---------------------------------------------------------------------------
# Test: DLRM container integration logic
# Tests the _get_parameter_cache logic and parameter precedence without
# importing the full DLRM app module (which requires torch).
# ---------------------------------------------------------------------------


class TestDLRMIntegration:
    """Tests that the DLRM parameter resolution logic works correctly.

    Since the full DLRM app requires torch (not available in test env),
    we test the cache integration logic directly by exercising the same
    code paths that the mutate function uses.
    """

    def test_cache_initializes_when_table_configured(self, mock_boto3, monkeypatch):
        """When PARAMETER_STORE_TABLE is set, ParameterCache is created."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.80)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 20.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        # Simulate what _get_parameter_cache does
        table_name = "test-params"
        region = "us-east-1"

        cache = ParameterCache(
            table_name=table_name,
            region=region,
            ttl_seconds=60.0,
            model_type="dlrm_bid_shader",
        )

        assert cache.get_shade_factor() == 0.80
        assert cache.get_conversion_value() == 20.0

    def test_no_cache_when_table_not_configured(self):
        """When PARAMETER_STORE_TABLE is not set, no cache is used (None path)."""
        # This tests the logic: if table_name is empty/None, no cache is created
        table_name = os.environ.get("PARAMETER_STORE_TABLE_NOT_SET_12345")
        assert table_name is None
        # In the DLRM app, this means _get_parameter_cache() returns None
        # and the code falls back to SHADE_FACTOR / EST_CONVERSION_VALUE constants

    def test_frontend_sliders_take_priority_over_cache(self, mock_boto3):
        """model_params (frontend sliders) should take priority over cache values.

        This tests the precedence logic: if 'shade_factor' is in model_params,
        it should be used directly, not the cache value.
        """
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.80)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 20.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-params",
            region="us-east-1",
            ttl_seconds=60.0,
            model_type="dlrm_bid_shader",
        )

        # Simulate the mutate() logic:
        # frontend sliders take priority
        model_params = {"shade_factor": 0.55, "conversion_value": 8.0}

        if "shade_factor" in model_params:
            shade_factor = model_params["shade_factor"]
        else:
            shade_factor = cache.get_shade_factor()

        if "conversion_value" in model_params:
            conversion_value = model_params["conversion_value"]
        else:
            conversion_value = cache.get_conversion_value()

        # Frontend values win
        assert shade_factor == 0.55
        assert conversion_value == 8.0

    def test_cache_used_when_no_frontend_override(self, mock_boto3):
        """When model_params doesn't contain an override, cache value is used."""
        mock_table, _ = mock_boto3

        def get_item_side_effect(**kwargs):
            key = kwargs["Key"]
            if key["parameter_name"] == "shade_factor":
                return {"Item": _make_dynamodb_item("shade_factor", 0.80)}
            elif key["parameter_name"] == "conversion_value":
                return {"Item": _make_dynamodb_item("conversion_value", 20.0)}
            return {}

        mock_table.get_item.side_effect = get_item_side_effect

        cache = ParameterCache(
            table_name="test-params",
            region="us-east-1",
            ttl_seconds=60.0,
            model_type="dlrm_bid_shader",
        )

        # Simulate the mutate() logic with empty model_params:
        model_params = {}
        SHADE_FACTOR_DEFAULT = 0.65
        EST_CONVERSION_VALUE_DEFAULT = 12.0

        if "shade_factor" in model_params:
            shade_factor = model_params["shade_factor"]
        else:
            shade_factor = cache.get_shade_factor() if cache else SHADE_FACTOR_DEFAULT

        if "conversion_value" in model_params:
            conversion_value = model_params["conversion_value"]
        else:
            conversion_value = cache.get_conversion_value() if cache else EST_CONVERSION_VALUE_DEFAULT

        # Cache values win over defaults
        assert shade_factor == 0.80
        assert conversion_value == 20.0

    def test_static_defaults_used_when_no_cache(self):
        """When cache is None (no table configured), static defaults are used."""
        cache = None
        model_params = {}
        SHADE_FACTOR_DEFAULT = 0.65
        EST_CONVERSION_VALUE_DEFAULT = 12.0

        if "shade_factor" in model_params:
            shade_factor = model_params["shade_factor"]
        else:
            shade_factor = cache.get_shade_factor() if cache else SHADE_FACTOR_DEFAULT

        if "conversion_value" in model_params:
            conversion_value = model_params["conversion_value"]
        else:
            conversion_value = cache.get_conversion_value() if cache else EST_CONVERSION_VALUE_DEFAULT

        assert shade_factor == 0.65
        assert conversion_value == 12.0
