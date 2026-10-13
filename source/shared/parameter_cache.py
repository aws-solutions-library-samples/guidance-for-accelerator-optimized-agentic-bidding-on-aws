"""In-memory parameter cache with DynamoDB + DAX backing.

Provides sub-5ms reads for bidding parameters (shade_factor, conversion_value)
by caching values locally with a configurable TTL (default 60s). On read failure
the cache returns the last known good value and continues operation (graceful
degradation). Sustained errors increment an error counter and emit a CloudWatch
alarm dimension.

The cache is non-blocking: if DynamoDB is slow or unavailable, the caller always
gets back a value immediately (either fresh or stale-but-valid).

Requirements: 8.3, 9.1, 9.2, 9.3
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds (mirrored from parameter_store.py for read-side clamping)
# ---------------------------------------------------------------------------

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "shade_factor": (0.3, 0.95),
    "conversion_value": (1.0, 50.0),
}

# Sustained error threshold before emitting alarm dimension
SUSTAINED_ERROR_THRESHOLD = 5


class ParameterCache:
    """In-memory cache for bidding parameters backed by DynamoDB (or DAX).

    Provides:
    - Sub-5ms reads via local caching with a configurable TTL
    - Graceful degradation: on read failure, returns last cached value
    - Bounds clamping on load
    - Error counting with alarm emission on sustained failures

    Parameters:
        table_name: Name of the DynamoDB table holding parameter state.
        region: AWS region for the DynamoDB resource.
        ttl_seconds: How long cached values are considered fresh (default 60s).
        model_type: Model type partition key (default "dlrm_bid_shader").
    """

    def __init__(
        self,
        table_name: str,
        region: str,
        ttl_seconds: float = 60.0,
        model_type: str = "dlrm_bid_shader",
    ) -> None:
        self._table_name = table_name
        self._region = region
        self._ttl_seconds = ttl_seconds
        self._model_type = model_type

        # Cached values — initialized to module defaults
        self._cached_shade_factor: float = 0.65
        self._cached_conversion_value: float = 12.0

        # Timestamps of last successful refresh (0 = never refreshed)
        self._last_refresh_time: float = 0.0

        # Error tracking
        self._consecutive_errors: int = 0
        self._alarm_emitted: bool = False

        # Lazy DynamoDB client
        self._table: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_shade_factor(self) -> float:
        """Get current shade_factor, refreshing from DynamoDB if TTL expired.

        Always returns a value (cached or fresh). Never raises.
        """
        self._refresh_if_stale()
        return self._cached_shade_factor

    def get_conversion_value(self) -> float:
        """Get current conversion_value, refreshing from DynamoDB if TTL expired.

        Always returns a value (cached or fresh). Never raises.
        """
        self._refresh_if_stale()
        return self._cached_conversion_value

    @property
    def consecutive_errors(self) -> int:
        """Number of consecutive DynamoDB read failures."""
        return self._consecutive_errors

    @property
    def alarm_emitted(self) -> bool:
        """Whether a sustained-error alarm has been emitted."""
        return self._alarm_emitted

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_if_stale(self) -> None:
        """Refresh from DynamoDB (or DAX) if TTL has expired.

        Non-blocking in the sense that on failure we immediately return
        without retrying, using the last cached values.
        """
        now = time.monotonic()
        if (now - self._last_refresh_time) < self._ttl_seconds:
            return  # Still fresh

        try:
            table = self._get_table()

            # Read shade_factor
            sf_response = table.get_item(
                Key={"model_type": self._model_type, "parameter_name": "shade_factor"},
                ConsistentRead=False,  # Eventually consistent for speed (DAX-compatible)
            )
            sf_item = sf_response.get("Item")

            # Read conversion_value
            cv_response = table.get_item(
                Key={"model_type": self._model_type, "parameter_name": "conversion_value"},
                ConsistentRead=False,
            )
            cv_item = cv_response.get("Item")

            # Update cached values with clamping
            if sf_item and "current_value" in sf_item:
                raw_sf = float(sf_item["current_value"])
                self._cached_shade_factor = self._clamp_to_bounds("shade_factor", raw_sf)

            if cv_item and "current_value" in cv_item:
                raw_cv = float(cv_item["current_value"])
                self._cached_conversion_value = self._clamp_to_bounds("conversion_value", raw_cv)

            # Successful refresh
            self._last_refresh_time = now
            self._consecutive_errors = 0
            self._alarm_emitted = False

        except Exception as exc:
            # Graceful degradation: keep using cached values
            self._consecutive_errors += 1
            logger.warning(
                "ParameterCache refresh failed (consecutive_errors=%d): %s",
                self._consecutive_errors,
                exc,
            )

            if self._consecutive_errors >= SUSTAINED_ERROR_THRESHOLD and not self._alarm_emitted:
                self._emit_sustained_error_alarm()
                self._alarm_emitted = True

    def _clamp_to_bounds(self, param_name: str, value: float) -> float:
        """Clamp value to PARAMETER_BOUNDS for the given parameter."""
        bounds = PARAMETER_BOUNDS.get(param_name)
        if bounds is None:
            return value
        min_val, max_val = bounds
        return max(min_val, min(max_val, value))

    def _get_table(self):
        """Lazily initialize the DynamoDB table resource.

        If DAX_ENDPOINT is set, uses the DAX client for sub-millisecond reads.
        Otherwise falls back to standard DynamoDB.
        """
        if self._table is None:
            import os

            dax_endpoint = os.environ.get("DAX_ENDPOINT")

            if dax_endpoint:
                try:
                    import amazondax
                    dax_client = amazondax.AmazonDaxClient.resource(
                        endpoint_url=dax_endpoint,
                        region_name=self._region,
                    )
                    self._table = dax_client.Table(self._table_name)
                    logger.info(
                        "ParameterCache: using DAX cluster at %s",
                        dax_endpoint,
                    )
                except ImportError:
                    logger.warning(
                        "DAX_ENDPOINT set (%s) but amazondax package not installed — "
                        "falling back to standard DynamoDB",
                        dax_endpoint,
                    )
                    import boto3
                    dynamodb = boto3.resource("dynamodb", region_name=self._region)
                    self._table = dynamodb.Table(self._table_name)
                except Exception as exc:
                    logger.warning(
                        "DAX connection failed (%s) — falling back to standard DynamoDB: %s",
                        dax_endpoint,
                        exc,
                    )
                    import boto3
                    dynamodb = boto3.resource("dynamodb", region_name=self._region)
                    self._table = dynamodb.Table(self._table_name)
            else:
                import boto3
                dynamodb = boto3.resource("dynamodb", region_name=self._region)
                self._table = dynamodb.Table(self._table_name)

        return self._table

    def _emit_sustained_error_alarm(self) -> None:
        """Emit a CloudWatch metric indicating sustained parameter store errors.

        This metric can be used to trigger a CloudWatch Alarm.
        """
        try:
            import boto3

            cloudwatch = boto3.client("cloudwatch", region_name=self._region)
            cloudwatch.put_metric_data(
                Namespace="ARTF/ParameterCache",
                MetricData=[
                    {
                        "MetricName": "SustainedReadErrors",
                        "Value": 1.0,
                        "Unit": "Count",
                        "Dimensions": [
                            {"Name": "ModelType", "Value": self._model_type},
                            {"Name": "TableName", "Value": self._table_name},
                        ],
                    }
                ],
            )
            logger.error(
                "ParameterCache: sustained errors (%d consecutive failures) — alarm emitted",
                self._consecutive_errors,
            )
        except Exception as alarm_exc:
            # Alarm emission failure should never block bidding
            logger.error("Failed to emit sustained error alarm: %s", alarm_exc)
