"""Tests for shared.parameter_store — DynamoDB Parameter Store.

Validates:
- Read parameter returns correct ParameterState
- Write parameter within bounds succeeds and increments version
- Write parameter out of bounds raises ParameterBoundsError
- Write parameter exceeding max_delta raises ParameterBoundsError
- Concurrent write (version conflict) raises OptimisticLockError
- DynamoDB conditional expression is correctly applied
- Audit trail is written on successful updates

**Validates: Requirements 8.1, 8.2, 8.4, 10.2, 12.3**
"""

import sys
import os
import asyncio
import time
from unittest.mock import MagicMock, patch, PropertyMock
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.parameter_store import (
    ParameterStore,
    ParameterState,
    ParameterBounds,
    ParameterBoundsError,
    OptimisticLockError,
    StaleVersionError,
    PARAMETER_BOUNDS,
)


# ---------------------------------------------------------------------------
# Helpers — deterministic fixtures
# ---------------------------------------------------------------------------


def _make_state(
    model_type: str = "dlrm_bid_shader",
    parameter_name: str = "shade_factor",
    current_value: float = 0.7,
    previous_value: float = 0.68,
    version: int = 1,
    updated_by: str = "bid_shading_agent",
    reason: str = "Win rate below target",
    confidence: float = 0.85,
) -> ParameterState:
    """Create a valid ParameterState fixture."""
    bounds = PARAMETER_BOUNDS[parameter_name]
    return ParameterState(
        model_type=model_type,
        parameter_name=parameter_name,
        current_value=current_value,
        previous_value=previous_value,
        updated_at=1718000000.0,
        updated_by=updated_by,
        version=version,
        min_value=bounds.min_value,
        max_value=bounds.max_value,
        max_delta_per_update=bounds.max_delta_per_update,
        reason=reason,
        confidence=confidence,
    )


def _make_dynamodb_item(state: ParameterState) -> dict:
    """Convert a ParameterState into a DynamoDB item dict (as returned by get_item)."""
    return {
        "model_type": state.model_type,
        "parameter_name": state.parameter_name,
        "current_value": str(state.current_value),
        "previous_value": str(state.previous_value),
        "updated_at": str(state.updated_at),
        "updated_by": state.updated_by,
        "version": state.version,
        "min_value": str(state.min_value),
        "max_value": str(state.max_value),
        "max_delta_per_update": str(state.max_delta_per_update),
        "reason": state.reason,
        "confidence": str(state.confidence),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dynamodb():
    """Patch boto3.resource and boto3.dynamodb.conditions to return mocks."""
    mock_table = MagicMock()
    mock_audit_table = MagicMock()

    mock_resource = MagicMock()

    def table_factory(name):
        if name == "parameter-store-audit":
            return mock_audit_table
        return mock_table

    mock_resource.Table.side_effect = table_factory

    # Create a mock Attr class that produces a mock condition expression
    mock_attr_instance = MagicMock()
    mock_attr_instance.eq.return_value = MagicMock()
    mock_attr_instance.not_exists.return_value = MagicMock()
    # The | operator on the condition returns a mock
    mock_attr_instance.eq.return_value.__or__ = MagicMock(return_value=MagicMock())

    mock_attr_cls = MagicMock(return_value=mock_attr_instance)

    # Mock Key class for read_all_parameters
    mock_key_instance = MagicMock()
    mock_key_instance.eq.return_value = MagicMock()
    mock_key_cls = MagicMock(return_value=mock_key_instance)

    # Create a mock boto3 module structure
    import types
    mock_boto3 = MagicMock()
    mock_boto3.resource.return_value = mock_resource

    # Set up boto3.dynamodb.conditions.Attr and Key
    mock_conditions = MagicMock()
    mock_conditions.Attr = mock_attr_cls
    mock_conditions.Key = mock_key_cls

    mock_dynamodb_pkg = MagicMock()
    mock_dynamodb_pkg.conditions = mock_conditions

    mock_boto3.dynamodb = mock_dynamodb_pkg

    with patch.dict("sys.modules", {
        "boto3": mock_boto3,
        "boto3.dynamodb": mock_dynamodb_pkg,
        "boto3.dynamodb.conditions": mock_conditions,
    }):
        with patch("boto3.resource", return_value=mock_resource):
            yield mock_table, mock_audit_table


class _MockClientError(Exception):
    """Mimics botocore.exceptions.ClientError for test purposes."""

    def __init__(self, error_code: str, message: str = ""):
        self.response = {"Error": {"Code": error_code, "Message": message}}
        super().__init__(f"{error_code}: {message}")


# ---------------------------------------------------------------------------
# Test: Read parameter
# ---------------------------------------------------------------------------


class TestReadParameter:
    """Tests for ParameterStore.read_parameter."""

    def test_read_existing_parameter_returns_correct_state(self, mock_dynamodb):
        """read_parameter returns a ParameterState with correct field values."""
        mock_table, _ = mock_dynamodb
        state = _make_state()
        mock_table.get_item.return_value = {"Item": _make_dynamodb_item(state)}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(store.read_parameter("dlrm_bid_shader", "shade_factor"))

        assert result.model_type == "dlrm_bid_shader"
        assert result.parameter_name == "shade_factor"
        assert result.current_value == 0.7
        assert result.previous_value == 0.68
        assert result.version == 1
        assert result.updated_by == "bid_shading_agent"
        assert result.reason == "Win rate below target"
        assert result.confidence == 0.85

    def test_read_uses_consistent_read(self, mock_dynamodb):
        """read_parameter uses ConsistentRead=True for strong consistency."""
        mock_table, _ = mock_dynamodb
        state = _make_state()
        mock_table.get_item.return_value = {"Item": _make_dynamodb_item(state)}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        asyncio.run(store.read_parameter("dlrm_bid_shader", "shade_factor"))

        call_kwargs = mock_table.get_item.call_args[1]
        assert call_kwargs["ConsistentRead"] is True

    def test_read_nonexistent_parameter_raises_key_error(self, mock_dynamodb):
        """read_parameter raises KeyError when item does not exist."""
        mock_table, _ = mock_dynamodb
        mock_table.get_item.return_value = {}  # No "Item" key

        store = ParameterStore(table_name="parameter-store", region="us-east-1")

        with pytest.raises(KeyError, match="Parameter not found"):
            asyncio.run(store.read_parameter("dlrm_bid_shader", "shade_factor"))

    def test_read_parameter_correct_key(self, mock_dynamodb):
        """read_parameter queries with the correct partition and sort key."""
        mock_table, _ = mock_dynamodb
        state = _make_state(model_type="ncf_deal_manager", parameter_name="conversion_value")
        mock_table.get_item.return_value = {"Item": _make_dynamodb_item(state)}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        asyncio.run(store.read_parameter("ncf_deal_manager", "conversion_value"))

        call_kwargs = mock_table.get_item.call_args[1]
        assert call_kwargs["Key"] == {
            "model_type": "ncf_deal_manager",
            "parameter_name": "conversion_value",
        }


# ---------------------------------------------------------------------------
# Test: Write parameter — bounds enforcement
# ---------------------------------------------------------------------------


class TestWriteParameterBounds:
    """Tests for bounds validation in ParameterStore.write_parameter."""

    def test_write_within_bounds_succeeds(self, mock_dynamodb):
        """Write with value within bounds and delta within limit succeeds."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70)

        result = asyncio.run(store.write_parameter(state))

        assert result.current_value == 0.72
        assert result.version == 2  # incremented from 1
        mock_table.put_item.assert_called_once()

    def test_write_below_min_raises_bounds_error(self, mock_dynamodb):
        """Write with current_value below min_value raises ParameterBoundsError."""
        mock_table, _ = mock_dynamodb

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # shade_factor min is 0.3
        state = _make_state(current_value=0.2, previous_value=0.3)

        with pytest.raises(ParameterBoundsError, match="outside bounds"):
            asyncio.run(store.write_parameter(state))

        # DynamoDB should not have been called
        mock_table.put_item.assert_not_called()

    def test_write_above_max_raises_bounds_error(self, mock_dynamodb):
        """Write with current_value above max_value raises ParameterBoundsError."""
        mock_table, _ = mock_dynamodb

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # shade_factor max is 0.95
        state = _make_state(current_value=0.99, previous_value=0.94)

        with pytest.raises(ParameterBoundsError, match="outside bounds"):
            asyncio.run(store.write_parameter(state))

        mock_table.put_item.assert_not_called()

    def test_write_exceeding_max_delta_raises_bounds_error(self, mock_dynamodb):
        """Write with delta exceeding max_delta_per_update raises ParameterBoundsError."""
        mock_table, _ = mock_dynamodb

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # shade_factor max_delta is 0.05; delta here is 0.10
        state = _make_state(current_value=0.80, previous_value=0.70)

        with pytest.raises(ParameterBoundsError, match="exceeds max_delta_per_update"):
            asyncio.run(store.write_parameter(state))

        mock_table.put_item.assert_not_called()

    def test_write_at_boundary_min_succeeds(self, mock_dynamodb):
        """Write with current_value exactly at min_value succeeds."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # shade_factor min is 0.3; previous 0.32, delta = 0.02 (within 0.05)
        state = _make_state(current_value=0.3, previous_value=0.32)

        result = asyncio.run(store.write_parameter(state))
        assert result.current_value == 0.3

    def test_write_at_boundary_max_succeeds(self, mock_dynamodb):
        """Write with current_value exactly at max_value succeeds."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # shade_factor max is 0.95; previous 0.93, delta = 0.02 (within 0.05)
        state = _make_state(current_value=0.95, previous_value=0.93)

        result = asyncio.run(store.write_parameter(state))
        assert result.current_value == 0.95

    def test_write_at_exact_max_delta_succeeds(self, mock_dynamodb):
        """Write with delta exactly at max_delta_per_update succeeds."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # shade_factor max_delta is 0.05; previous 0.70, current 0.75 → delta = 0.05
        state = _make_state(current_value=0.75, previous_value=0.70)

        result = asyncio.run(store.write_parameter(state))
        assert result.current_value == 0.75

    def test_write_conversion_value_within_bounds(self, mock_dynamodb):
        """Write conversion_value within its bounds [1.0, 50.0] succeeds."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(
            parameter_name="conversion_value",
            current_value=12.5,
            previous_value=11.0,
        )

        result = asyncio.run(store.write_parameter(state))
        assert result.current_value == 12.5

    def test_write_conversion_value_exceeding_delta(self, mock_dynamodb):
        """Write conversion_value with delta > 2.5 raises ParameterBoundsError."""
        mock_table, _ = mock_dynamodb

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        # conversion_value max_delta is 2.5; delta here is 5.0
        state = _make_state(
            parameter_name="conversion_value",
            current_value=15.0,
            previous_value=10.0,
        )

        with pytest.raises(ParameterBoundsError, match="exceeds max_delta_per_update"):
            asyncio.run(store.write_parameter(state))


# ---------------------------------------------------------------------------
# Test: Write parameter — optimistic locking
# ---------------------------------------------------------------------------


class TestWriteParameterOptimisticLocking:
    """Tests for optimistic concurrency control in ParameterStore.write_parameter."""

    def test_version_conflict_raises_optimistic_lock_error(self, mock_dynamodb):
        """ConditionalCheckFailedException from DynamoDB raises OptimisticLockError."""
        mock_table, _ = mock_dynamodb

        # Simulate a version conflict
        mock_table.put_item.side_effect = _MockClientError("ConditionalCheckFailedException", "Conflict")

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70, version=3)

        with pytest.raises(OptimisticLockError) as exc_info:
            asyncio.run(store.write_parameter(state))

        assert exc_info.value.model_type == "dlrm_bid_shader"
        assert exc_info.value.parameter_name == "shade_factor"
        assert exc_info.value.expected_version == 3

    def test_conditional_expression_uses_version(self, mock_dynamodb):
        """put_item is called with a ConditionExpression referencing the version."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70, version=5)

        asyncio.run(store.write_parameter(state))

        call_kwargs = mock_table.put_item.call_args[1]
        # Verify ConditionExpression is present
        assert "ConditionExpression" in call_kwargs
        # The item should have new version = 6
        assert call_kwargs["Item"]["version"] == 6

    def test_write_increments_version(self, mock_dynamodb):
        """Successful write returns state with version incremented by 1."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70, version=7)

        result = asyncio.run(store.write_parameter(state))
        assert result.version == 8

    def test_unexpected_client_error_propagates(self, mock_dynamodb):
        """Non-condition-check ClientErrors propagate as-is."""
        mock_table, _ = mock_dynamodb

        mock_table.put_item.side_effect = _MockClientError("InternalServerError", "Oops")

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70)

        with pytest.raises(_MockClientError) as exc_info:
            asyncio.run(store.write_parameter(state))

        assert "InternalServerError" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: Audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    """Tests for audit trail writes on parameter updates."""

    def test_successful_write_creates_audit_record(self, mock_dynamodb):
        """Successful write_parameter also writes to the audit table."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70, version=2)

        asyncio.run(store.write_parameter(state))

        mock_audit_table.put_item.assert_called_once()
        audit_item = mock_audit_table.put_item.call_args[1]["Item"]
        assert audit_item["model_type"] == "dlrm_bid_shader"
        assert audit_item["parameter_name"] == "shade_factor"
        assert audit_item["old_value"] == "0.7"
        assert audit_item["new_value"] == "0.72"
        assert audit_item["updated_by"] == "bid_shading_agent"
        assert audit_item["version"] == 3

    def test_audit_failure_does_not_block_write(self, mock_dynamodb):
        """If audit table write fails, the primary write still succeeds."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.side_effect = _MockClientError("InternalServerError", "Audit down")

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(current_value=0.72, previous_value=0.70)

        # Should not raise despite audit failure
        result = asyncio.run(store.write_parameter(state))
        assert result.version == 2

    def test_audit_record_contains_reason(self, mock_dynamodb):
        """Audit record includes the reason for the parameter change."""
        mock_table, mock_audit_table = mock_dynamodb
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        state = _make_state(
            current_value=0.72,
            previous_value=0.70,
            reason="ROI suboptimal, increasing aggressiveness",
        )

        asyncio.run(store.write_parameter(state))

        audit_item = mock_audit_table.put_item.call_args[1]["Item"]
        assert audit_item["reason"] == "ROI suboptimal, increasing aggressiveness"


# ---------------------------------------------------------------------------
# Test: Read all parameters
# ---------------------------------------------------------------------------


class TestReadAllParameters:
    """Tests for ParameterStore.read_all_parameters."""

    def test_read_all_returns_dict_by_parameter_name(self, mock_dynamodb):
        """read_all_parameters returns a dict keyed by parameter_name."""
        mock_table, _ = mock_dynamodb

        shade_state = _make_state(parameter_name="shade_factor", current_value=0.7)
        cv_state = _make_state(
            parameter_name="conversion_value",
            current_value=12.0,
            previous_value=11.0,
        )

        mock_table.query.return_value = {
            "Items": [
                _make_dynamodb_item(shade_state),
                _make_dynamodb_item(cv_state),
            ]
        }

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(store.read_all_parameters("dlrm_bid_shader"))

        assert "shade_factor" in result
        assert "conversion_value" in result
        assert result["shade_factor"].current_value == 0.7
        assert result["conversion_value"].current_value == 12.0

    def test_read_all_empty_returns_empty_dict(self, mock_dynamodb):
        """read_all_parameters returns empty dict when no items exist."""
        mock_table, _ = mock_dynamodb
        mock_table.query.return_value = {"Items": []}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(store.read_all_parameters("dlrm_bid_shader"))

        assert result == {}

    def test_read_all_uses_consistent_read(self, mock_dynamodb):
        """read_all_parameters uses ConsistentRead=True."""
        mock_table, _ = mock_dynamodb
        mock_table.query.return_value = {"Items": []}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        asyncio.run(store.read_all_parameters("dlrm_bid_shader"))

        call_kwargs = mock_table.query.call_args[1]
        assert call_kwargs["ConsistentRead"] is True


# ---------------------------------------------------------------------------
# Test: Default bounds
# ---------------------------------------------------------------------------


class TestDefaultBounds:
    """Tests for the PARAMETER_BOUNDS constants."""

    def test_shade_factor_bounds(self):
        """shade_factor bounds are [0.3, 0.95] with max_delta 0.05."""
        b = PARAMETER_BOUNDS["shade_factor"]
        assert b.min_value == 0.3
        assert b.max_value == 0.95
        assert b.max_delta_per_update == 0.05

    def test_conversion_value_bounds(self):
        """conversion_value bounds are [1.0, 50.0] with max_delta 2.5."""
        b = PARAMETER_BOUNDS["conversion_value"]
        assert b.min_value == 1.0
        assert b.max_value == 50.0
        assert b.max_delta_per_update == 2.5


# ---------------------------------------------------------------------------
# Test: Initialize parameter
# ---------------------------------------------------------------------------


class TestInitializeParameter:
    """Tests for ParameterStore.initialize_parameter."""

    def test_initialize_creates_parameter_with_version_zero(self, mock_dynamodb):
        """initialize_parameter creates a new parameter with version=0."""
        mock_table, _ = mock_dynamodb
        mock_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(
            store.initialize_parameter("dlrm_bid_shader", "shade_factor", 0.7)
        )

        assert result.version == 0
        assert result.current_value == 0.7
        assert result.previous_value == 0.7
        assert result.model_type == "dlrm_bid_shader"
        assert result.parameter_name == "shade_factor"
        assert result.updated_by == "system"
        assert result.reason == "Parameter initialized"

    def test_initialize_sets_bounds_from_parameter_bounds(self, mock_dynamodb):
        """initialize_parameter sets min/max from PARAMETER_BOUNDS."""
        mock_table, _ = mock_dynamodb
        mock_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(
            store.initialize_parameter("dlrm_bid_shader", "shade_factor", 0.6)
        )

        assert result.min_value == 0.3
        assert result.max_value == 0.95

    def test_initialize_uses_conditional_put(self, mock_dynamodb):
        """initialize_parameter uses attribute_not_exists condition."""
        mock_table, _ = mock_dynamodb
        mock_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        asyncio.run(
            store.initialize_parameter("dlrm_bid_shader", "shade_factor", 0.7)
        )

        call_kwargs = mock_table.put_item.call_args[1]
        assert "ConditionExpression" in call_kwargs
        assert "attribute_not_exists" in call_kwargs["ConditionExpression"]

    def test_initialize_existing_parameter_raises_error(self, mock_dynamodb):
        """initialize_parameter raises OptimisticLockError if parameter already exists."""
        mock_table, _ = mock_dynamodb
        mock_table.put_item.side_effect = _MockClientError(
            "ConditionalCheckFailedException", "Already exists"
        )

        store = ParameterStore(table_name="parameter-store", region="us-east-1")

        with pytest.raises(OptimisticLockError):
            asyncio.run(
                store.initialize_parameter("dlrm_bid_shader", "shade_factor", 0.7)
            )

    def test_initialize_out_of_bounds_raises_bounds_error(self, mock_dynamodb):
        """initialize_parameter rejects initial_value outside global bounds."""
        mock_table, _ = mock_dynamodb

        store = ParameterStore(table_name="parameter-store", region="us-east-1")

        with pytest.raises(ParameterBoundsError, match="outside bounds"):
            asyncio.run(
                store.initialize_parameter("dlrm_bid_shader", "shade_factor", 0.1)
            )

        mock_table.put_item.assert_not_called()

    def test_initialize_unknown_parameter_raises_value_error(self, mock_dynamodb):
        """initialize_parameter rejects unknown parameter names."""
        mock_table, _ = mock_dynamodb

        store = ParameterStore(table_name="parameter-store", region="us-east-1")

        with pytest.raises(ValueError, match="Unknown parameter_name"):
            asyncio.run(
                store.initialize_parameter("dlrm_bid_shader", "unknown_param", 1.0)
            )

    def test_initialize_conversion_value_within_bounds(self, mock_dynamodb):
        """initialize_parameter works for conversion_value within [1.0, 50.0]."""
        mock_table, _ = mock_dynamodb
        mock_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(
            store.initialize_parameter("dlrm_bid_shader", "conversion_value", 10.0)
        )

        assert result.current_value == 10.0
        assert result.min_value == 1.0
        assert result.max_value == 50.0


# ---------------------------------------------------------------------------
# Test: update_parameter (convenience API)
# ---------------------------------------------------------------------------


class TestUpdateParameter:
    """Tests for ParameterStore.update_parameter convenience method."""

    def test_update_parameter_reads_then_writes(self, mock_dynamodb):
        """update_parameter reads current state then writes the update."""
        mock_table, mock_audit_table = mock_dynamodb
        # Set up read
        existing_state = _make_state(current_value=0.70, previous_value=0.68, version=3)
        mock_table.get_item.return_value = {"Item": _make_dynamodb_item(existing_state)}
        # Set up write
        mock_table.put_item.return_value = {}
        mock_audit_table.put_item.return_value = {}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")
        result = asyncio.run(
            store.update_parameter(
                model_type="dlrm_bid_shader",
                parameter_name="shade_factor",
                new_value=0.73,
                updated_by="bid_shading_agent",
                reason="Win rate below target",
                confidence=0.9,
                expected_version=3,
            )
        )

        assert result.current_value == 0.73
        assert result.previous_value == 0.70
        assert result.version == 4

    def test_update_parameter_out_of_bounds_raises(self, mock_dynamodb):
        """update_parameter raises ParameterBoundsError for out-of-bounds value."""
        mock_table, _ = mock_dynamodb
        existing_state = _make_state(current_value=0.90, previous_value=0.88, version=5)
        mock_table.get_item.return_value = {"Item": _make_dynamodb_item(existing_state)}

        store = ParameterStore(table_name="parameter-store", region="us-east-1")

        with pytest.raises(ParameterBoundsError):
            asyncio.run(
                store.update_parameter(
                    model_type="dlrm_bid_shader",
                    parameter_name="shade_factor",
                    new_value=0.99,  # above max 0.95
                    updated_by="bid_shading_agent",
                    reason="test",
                    confidence=0.5,
                    expected_version=5,
                )
            )


# ---------------------------------------------------------------------------
# Test: StaleVersionError alias
# ---------------------------------------------------------------------------


class TestStaleVersionErrorAlias:
    """Verify StaleVersionError is an alias for OptimisticLockError."""

    def test_stale_version_error_is_optimistic_lock_error(self):
        """StaleVersionError and OptimisticLockError are the same class."""
        assert StaleVersionError is OptimisticLockError

    def test_stale_version_error_can_be_caught(self):
        """StaleVersionError instances are catchable as OptimisticLockError."""
        exc = StaleVersionError("model", "param", 1)
        assert isinstance(exc, OptimisticLockError)
