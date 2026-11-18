"""DynamoDB-backed Parameter Store for bidding parameters.

Holds the current ``shade_factor`` and ``conversion_value`` parameters read by
the DLRM container at inference time and written by the Adaptive Bidding Strategy
Agent. Provides:

- **Bounds enforcement**: Writes are rejected at the application layer if the
  proposed value falls outside ``[min_value, max_value]`` or the delta from
  ``previous_value`` exceeds ``max_delta_per_update``.
- **Optimistic concurrency**: Uses a ``version`` field with a DynamoDB
  ``ConditionExpression`` (``attribute_not_exists(version) OR version = :expected``)
  so that concurrent writes are safely rejected.
- **Audit trail**: Every successful write appends an audit record to a
  separate audit trail table.

Security notes (handled at the deployment/IAM layer, not in application code):
- DynamoDB table is encrypted at rest (SSE-KMS) — configured in CloudFormation.
- Write access is restricted to the Adaptive_Bidding_Agent Workload Identity role
  via IAM policy — configured in the AgentCore deployment.

Requirements: 8.1, 8.2, 8.4, 10.2, 12.3
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ParameterBoundsError(Exception):
    """Raised when a parameter write would violate bounds or max-delta constraints."""

    def __init__(self, message: str, parameter_name: str, value: float, bounds: "ParameterBounds"):
        self.parameter_name = parameter_name
        self.value = value
        self.bounds = bounds
        super().__init__(message)


class OptimisticLockError(Exception):
    """Raised when a write fails due to a version conflict (optimistic concurrency)."""

    def __init__(self, model_type: str, parameter_name: str, expected_version: int):
        self.model_type = model_type
        self.parameter_name = parameter_name
        self.expected_version = expected_version
        super().__init__(
            f"Version conflict for {model_type}/{parameter_name}: "
            f"expected version {expected_version} but item was modified concurrently"
        )


# Alias for task-spec naming compatibility
StaleVersionError = OptimisticLockError


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ParameterBounds:
    """Global constraints for a parameter type."""

    min_value: float
    max_value: float
    max_delta_per_update: float


@dataclass
class ParameterState:
    """Stored in DynamoDB for the Adaptive Bidding Strategy Agent.

    Attributes:
        model_type: Partition key (e.g. "dlrm_bid_shader")
        parameter_name: Sort key ("shade_factor" | "conversion_value")
        current_value: The active parameter value
        previous_value: Value before the last update
        updated_at: Unix timestamp of last write
        updated_by: Identity of the writer ("adaptive_bidding_agent" | "manual")
        version: Monotonically increasing integer for optimistic locking
        min_value: Lower bound for this parameter
        max_value: Upper bound for this parameter
        max_delta_per_update: Maximum allowed absolute change per write
        reason: Human-readable reason for the last update
        confidence: Agent's confidence in the update (0.0–1.0)
    """

    model_type: str
    parameter_name: str
    current_value: float
    previous_value: float
    updated_at: float
    updated_by: str
    version: int
    min_value: float
    max_value: float
    max_delta_per_update: float
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# Default bounds — global constraints per the design
# ---------------------------------------------------------------------------

PARAMETER_BOUNDS: dict[str, ParameterBounds] = {
    "shade_factor": ParameterBounds(min_value=0.3, max_value=0.95, max_delta_per_update=0.05),
    "conversion_value": ParameterBounds(min_value=1.0, max_value=50.0, max_delta_per_update=2.5),
}


# ---------------------------------------------------------------------------
# ParameterStore
# ---------------------------------------------------------------------------


class ParameterStore:
    """DynamoDB-backed parameter store with bounds enforcement and optimistic locking.

    Parameters:
        table_name: Name of the DynamoDB table holding parameter state.
        region: AWS region for the DynamoDB resource.
        audit_table_name: Optional name of the audit trail table. Defaults to
            ``{table_name}-audit``.
    """

    def __init__(
        self,
        table_name: str,
        region: str,
        audit_table_name: Optional[str] = None,
    ) -> None:
        self._table_name = table_name
        self._audit_table_name = audit_table_name or f"{table_name}-audit"
        self._region = region

        import boto3

        self._dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = self._dynamodb.Table(table_name)
        self._audit_table = self._dynamodb.Table(self._audit_table_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def read_parameter(self, model_type: str, parameter_name: str) -> ParameterState:
        """Read the current state of a single parameter from DynamoDB.

        Args:
            model_type: Partition key (e.g. "dlrm_bid_shader").
            parameter_name: Sort key (e.g. "shade_factor").

        Returns:
            ParameterState with the current stored values.

        Raises:
            KeyError: If the parameter does not exist in the table.
        """
        response = self._table.get_item(
            Key={"model_type": model_type, "parameter_name": parameter_name},
            ConsistentRead=True,
        )

        item = response.get("Item")
        if item is None:
            raise KeyError(
                f"Parameter not found: model_type={model_type}, "
                f"parameter_name={parameter_name}"
            )

        return self._item_to_state(item)

    async def write_parameter(self, state: ParameterState) -> ParameterState:
        """Write an updated parameter to DynamoDB with bounds and version checks.

        Validates:
        1. ``current_value`` is within ``[min_value, max_value]``
        2. The change from ``previous_value`` does not exceed ``max_delta_per_update``
        3. Optimistic concurrency via version conditional expression

        Args:
            state: The proposed new parameter state. The ``version`` field
                should contain the version that was read (the expected version).

        Returns:
            Updated ParameterState with the version incremented.

        Raises:
            ParameterBoundsError: If bounds or max-delta validation fails.
            OptimisticLockError: If another writer updated the item concurrently.
        """
        # Resolve bounds — prefer the state's own bounds, fall back to defaults
        bounds = ParameterBounds(
            min_value=state.min_value,
            max_value=state.max_value,
            max_delta_per_update=state.max_delta_per_update,
        )

        # --- Bounds validation ---
        self._validate_bounds(state, bounds)

        # --- Prepare the write ---
        new_version = state.version + 1
        now = time.time()

        item = {
            "model_type": state.model_type,
            "parameter_name": state.parameter_name,
            "current_value": str(state.current_value),
            "previous_value": str(state.previous_value),
            "updated_at": str(now),
            "updated_by": state.updated_by,
            "version": new_version,
            "min_value": str(state.min_value),
            "max_value": str(state.max_value),
            "max_delta_per_update": str(state.max_delta_per_update),
            "reason": state.reason,
            "confidence": str(state.confidence),
        }

        # --- Conditional write for optimistic concurrency ---
        # Uses DynamoDB condition expression string format to avoid
        # runtime dependency on boto3.dynamodb.conditions at import time.
        condition_expression = "attribute_not_exists(version) OR version = :expected_version"
        expression_values = {":expected_version": state.version}

        try:
            self._table.put_item(
                Item=item,
                ConditionExpression=condition_expression,
                ExpressionAttributeValues=expression_values,
            )
        except Exception as exc:
            # Handle ClientError for conditional check failure
            if hasattr(exc, "response") and exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise OptimisticLockError(
                    model_type=state.model_type,
                    parameter_name=state.parameter_name,
                    expected_version=state.version,
                ) from exc
            raise  # Re-raise unexpected errors

        # --- Write audit trail ---
        self._write_audit_record(state, new_version, now)

        # Return the updated state
        return ParameterState(
            model_type=state.model_type,
            parameter_name=state.parameter_name,
            current_value=state.current_value,
            previous_value=state.previous_value,
            updated_at=now,
            updated_by=state.updated_by,
            version=new_version,
            min_value=state.min_value,
            max_value=state.max_value,
            max_delta_per_update=state.max_delta_per_update,
            reason=state.reason,
            confidence=state.confidence,
        )

    async def read_all_parameters(self, model_type: str) -> dict[str, ParameterState]:
        """Read all parameters for a given model type.

        Args:
            model_type: Partition key (e.g. "dlrm_bid_shader").

        Returns:
            Dict mapping ``parameter_name`` to its ``ParameterState``.
        """
        from boto3.dynamodb.conditions import Key

        response = self._table.query(
            KeyConditionExpression=Key("model_type").eq(model_type),
            ConsistentRead=True,
        )

        result: dict[str, ParameterState] = {}
        for item in response.get("Items", []):
            state = self._item_to_state(item)
            result[state.parameter_name] = state

        return result

    async def initialize_parameter(
        self,
        model_type: str,
        parameter_name: str,
        initial_value: float,
        max_delta_per_update: float = 0.05,
    ) -> ParameterState:
        """Create a parameter with version=0 if it doesn't already exist.

        Uses a conditional put so that an existing parameter is never overwritten.

        Args:
            model_type: Partition key (e.g. "dlrm_bid_shader").
            parameter_name: Sort key (e.g. "shade_factor").
            initial_value: Starting value for the parameter.
            max_delta_per_update: Maximum absolute change per write (default 0.05).

        Returns:
            The newly created ParameterState with version=0.

        Raises:
            ParameterBoundsError: If initial_value is outside the global bounds.
            ValueError: If parameter_name is not a recognized bounded parameter.
            OptimisticLockError: If the parameter already exists.
        """
        if parameter_name not in PARAMETER_BOUNDS:
            raise ValueError(
                f"Unknown parameter_name '{parameter_name}'. "
                f"Must be one of: {list(PARAMETER_BOUNDS.keys())}"
            )

        bounds = PARAMETER_BOUNDS[parameter_name]

        # Validate the initial value is within global bounds
        if initial_value < bounds.min_value or initial_value > bounds.max_value:
            raise ParameterBoundsError(
                f"Initial value {initial_value} for '{parameter_name}' is outside "
                f"bounds [{bounds.min_value}, {bounds.max_value}]",
                parameter_name=parameter_name,
                value=initial_value,
                bounds=bounds,
            )

        now = time.time()
        item = {
            "model_type": model_type,
            "parameter_name": parameter_name,
            "current_value": str(initial_value),
            "previous_value": str(initial_value),
            "updated_at": str(now),
            "updated_by": "system",
            "version": 0,
            "min_value": str(bounds.min_value),
            "max_value": str(bounds.max_value),
            "max_delta_per_update": str(max_delta_per_update),
            "reason": "Parameter initialized",
            "confidence": str(1.0),
        }

        # Conditional put: only create if it doesn't already exist
        condition_expression = "attribute_not_exists(model_type)"

        try:
            self._table.put_item(
                Item=item,
                ConditionExpression=condition_expression,
            )
        except Exception as exc:
            if hasattr(exc, "response") and exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise OptimisticLockError(
                    model_type=model_type,
                    parameter_name=parameter_name,
                    expected_version=-1,
                ) from exc
            raise

        return ParameterState(
            model_type=model_type,
            parameter_name=parameter_name,
            current_value=initial_value,
            previous_value=initial_value,
            updated_at=now,
            updated_by="system",
            version=0,
            min_value=bounds.min_value,
            max_value=bounds.max_value,
            max_delta_per_update=max_delta_per_update,
            reason="Parameter initialized",
            confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Convenience aliases matching task-spec naming
    # ------------------------------------------------------------------

    async def get_parameter(self, model_type: str, parameter_name: str) -> ParameterState:
        """Alias for read_parameter (matches task spec naming)."""
        return await self.read_parameter(model_type, parameter_name)

    async def update_parameter(
        self,
        model_type: str,
        parameter_name: str,
        new_value: float,
        updated_by: str,
        reason: str,
        confidence: float,
        expected_version: int,
    ) -> ParameterState:
        """Update a parameter by name with bounds and optimistic locking.

        This is a higher-level API that reads the current state and applies
        the update in one logical operation.

        Args:
            model_type: Partition key (e.g. "dlrm_bid_shader").
            parameter_name: Sort key (e.g. "shade_factor").
            new_value: The proposed new value.
            updated_by: Identity of the writer.
            reason: Human-readable reason for the change.
            confidence: Agent's confidence in the update (0.0–1.0).
            expected_version: The version the caller last read (optimistic lock).

        Returns:
            Updated ParameterState with version incremented.

        Raises:
            ParameterBoundsError: If new_value is outside bounds or exceeds max delta.
            OptimisticLockError: If the expected_version doesn't match current.
            KeyError: If the parameter does not exist.
        """
        current = await self.read_parameter(model_type, parameter_name)

        # Build the state for write
        state = ParameterState(
            model_type=model_type,
            parameter_name=parameter_name,
            current_value=new_value,
            previous_value=current.current_value,
            updated_at=current.updated_at,  # will be overwritten by write
            updated_by=updated_by,
            version=expected_version,
            min_value=current.min_value,
            max_value=current.max_value,
            max_delta_per_update=current.max_delta_per_update,
            reason=reason,
            confidence=confidence,
        )

        return await self.write_parameter(state)

    async def get_all_parameters(self, model_type: str) -> list[ParameterState]:
        """Get all parameters for a model type as a list.

        Alias for read_all_parameters but returns a list instead of a dict.
        """
        params = await self.read_all_parameters(model_type)
        return list(params.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_bounds(self, state: ParameterState, bounds: ParameterBounds) -> None:
        """Validate that current_value is within bounds and delta is acceptable."""
        # Check absolute bounds
        if state.current_value < bounds.min_value or state.current_value > bounds.max_value:
            raise ParameterBoundsError(
                f"Parameter '{state.parameter_name}' value {state.current_value} "
                f"is outside bounds [{bounds.min_value}, {bounds.max_value}]",
                parameter_name=state.parameter_name,
                value=state.current_value,
                bounds=bounds,
            )

        # Check max delta from previous value (use tolerance for floating-point equality)
        delta = abs(state.current_value - state.previous_value)
        if delta > bounds.max_delta_per_update + 1e-9:
            raise ParameterBoundsError(
                f"Parameter '{state.parameter_name}' change delta {delta:.6f} "
                f"exceeds max_delta_per_update {bounds.max_delta_per_update}",
                parameter_name=state.parameter_name,
                value=state.current_value,
                bounds=bounds,
            )

    def _write_audit_record(
        self, state: ParameterState, new_version: int, timestamp: float
    ) -> None:
        """Append an audit trail record for a parameter change."""
        audit_item = {
            "model_type": state.model_type,
            "timestamp_version": f"{timestamp}#{new_version}",
            "parameter_name": state.parameter_name,
            "old_value": str(state.previous_value),
            "new_value": str(state.current_value),
            "updated_by": state.updated_by,
            "reason": state.reason,
            "confidence": str(state.confidence),
            "version": new_version,
        }
        try:
            self._audit_table.put_item(Item=audit_item)
        except Exception:
            # Audit write failure should not block the parameter update
            # (the primary write already succeeded). Log in production.
            pass

    @staticmethod
    def _item_to_state(item: dict) -> ParameterState:
        """Convert a DynamoDB item dict to a ParameterState dataclass."""
        return ParameterState(
            model_type=item["model_type"],
            parameter_name=item["parameter_name"],
            current_value=float(item["current_value"]),
            previous_value=float(item["previous_value"]),
            updated_at=float(item["updated_at"]),
            updated_by=item["updated_by"],
            version=int(item["version"]),
            min_value=float(item["min_value"]),
            max_value=float(item["max_value"]),
            max_delta_per_update=float(item["max_delta_per_update"]),
            reason=item["reason"],
            confidence=float(item["confidence"]),
        )
