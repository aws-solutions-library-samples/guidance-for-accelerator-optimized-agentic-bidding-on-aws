"""Tests for the Glue ETL feature engineering and deduplication logic.

Tests use a local PySpark session (no Glue dependencies required) to validate:
- De-duplication by request_id (keep latest timestamp)
- Feature engineering (ROI, shade_ratio, label, win_rate_bucket)
- PII detection and filtering

**Validates: Requirements 2.2, 2.3, 2.4, 2.6**
"""

import os
import sys

import pytest

# Add source root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try importing PySpark — skip tests gracefully if not available
pyspark = pytest.importorskip("pyspark", reason="PySpark required for Glue ETL tests")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from etl.glue_feature_engineering import (
    _contains_pii_pattern,
    _is_plausible_hash,
    compute_win_rate_buckets,
    deduplicate_by_request_id,
    engineer_features,
    validate_no_raw_pii,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Schema matching the Glue catalog table (raw_bid_outcomes) -- these columns
# match shared/feedback_models.py's BidOutcomeEvent field names/types
# exactly, since that's what Firehose's JSON->Parquet conversion actually
# populates (see feedback_pipeline_cfn.yaml's raw_bid_outcomes table).
_RAW_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("timestamp", DoubleType(), False),
        StructField("model_version", StringType(), False),
        StructField("source", StringType(), False),
        StructField("model_type", StringType(), True),
        StructField("original_price", DoubleType(), False),
        StructField("shaded_price", DoubleType(), False),
        StructField("bid_floor", DoubleType(), False),
        StructField("won", BooleanType(), False),
        StructField("price_paid", DoubleType(), True),
        StructField("impression", BooleanType(), False),
        StructField("click", BooleanType(), False),
        StructField("conversion", BooleanType(), False),
        StructField("conversion_value", DoubleType(), True),
        StructField("user_id_hash", StringType(), False),
        StructField("site_domain", StringType(), False),
        StructField("device_type", StringType(), False),
        StructField("hour_of_day", IntegerType(), False),
        StructField("shade_factor_used", DoubleType(), False),
        StructField("conversion_value_estimate_used", DoubleType(), False),
    ]
)


def _make_record(
    request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    timestamp=1718000000.0,
    model_version="v1.0.0",
    source="live",
    model_type="dlrm_bid_shader",
    original_price=5.0,
    shaded_price=4.0,
    bid_floor=2.0,
    won=True,
    price_paid=3.5,
    impression=True,
    click=False,
    conversion=False,
    conversion_value=None,
    user_id_hash="abc123def456789012345678abcdef01",
    site_domain="espn.com",
    device_type="mobile",
    hour_of_day=14,
    shade_factor_used=0.8,
    conversion_value_estimate_used=10.0,
):
    """Create a test record tuple matching _RAW_SCHEMA."""
    return (
        request_id,
        timestamp,
        model_version,
        source,
        model_type,
        original_price,
        shaded_price,
        bid_floor,
        won,
        price_paid,
        impression,
        click,
        conversion,
        conversion_value,
        user_id_hash,
        site_domain,
        device_type,
        hour_of_day,
        shade_factor_used,
        conversion_value_estimate_used,
    )


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test_glue_etl")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Pure-Python PII helper tests (no Spark needed)
# ---------------------------------------------------------------------------


class TestPIIHelpers:
    """Tests for PII detection helper functions (pure Python, no Spark)."""

    def test_hex_hash_is_plausible(self):
        """32-char hex string is recognized as a plausible hash."""
        assert _is_plausible_hash("abc123def456789012345678abcdef01") is True

    def test_short_string_not_a_hash(self):
        """Short strings are not recognized as hashes."""
        assert _is_plausible_hash("abc") is False

    def test_empty_string_not_a_hash(self):
        """Empty string is not a hash."""
        assert _is_plausible_hash("") is False

    def test_none_not_a_hash(self):
        """None is not a hash."""
        assert _is_plausible_hash(None) is False

    def test_email_is_pii(self):
        """Email addresses are detected as PII."""
        assert _contains_pii_pattern("user@example.com") is True

    def test_hex_hash_not_pii(self):
        """Hex hash string does not contain PII patterns."""
        assert _contains_pii_pattern("abc123def456789012345678abcdef01") is False

    def test_empty_not_pii(self):
        """Empty string does not contain PII patterns."""
        assert _contains_pii_pattern("") is False

    def test_base64_hash_is_plausible(self):
        """Base64-like strings of sufficient length are plausible hashes."""
        assert _is_plausible_hash("dGhpcyBpcyBhIGJhc2U2NCBzdHJpbmc=") is True


# ---------------------------------------------------------------------------
# Deduplication Tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for deduplicate_by_request_id."""

    def test_keeps_latest_by_timestamp(self, spark):
        """When multiple records share request_id, only the latest is kept."""
        records = [
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                timestamp=1000.0,
                click=False,
                conversion=False,
            ),
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                timestamp=2000.0,
                click=False,
                conversion=False,
            ),
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                timestamp=3000.0,
                click=True,
                conversion=False,
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = deduplicate_by_request_id(df)

        assert result.count() == 1
        row = result.collect()[0]
        assert row["timestamp"] == 3000.0
        assert row["click"] is True

    def test_distinct_request_ids_preserved(self, spark):
        """Records with different request_ids are all preserved."""
        records = [
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                timestamp=1000.0,
            ),
            _make_record(
                request_id="bbbb1111-2222-3333-4444-555566667777",
                timestamp=2000.0,
            ),
            _make_record(
                request_id="cccc1111-2222-3333-4444-555566667777",
                timestamp=3000.0,
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = deduplicate_by_request_id(df)

        assert result.count() == 3

    def test_single_record_unchanged(self, spark):
        """A single record passes through unchanged."""
        records = [_make_record()]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = deduplicate_by_request_id(df)

        assert result.count() == 1


# ---------------------------------------------------------------------------
# Feature Engineering Tests
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    """Tests for engineer_features."""

    def test_roi_computed_for_winning_conversion(self, spark):
        """ROI = (conversion_value - price_paid) / price_paid for winning conversions."""
        records = [
            _make_record(
                won=True,
                impression=True,
                click=True,
                conversion=True,
                price_paid=2.0,
                conversion_value=10.0,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        # ROI = (10.0 - 2.0) / 2.0 = 4.0
        assert abs(row["roi"] - 4.0) < 1e-6

    def test_roi_null_when_no_price_paid(self, spark):
        """ROI is null when price_paid is null (lost bid)."""
        records = [
            _make_record(
                won=False,
                price_paid=None,
                impression=False,
                click=False,
                conversion=False,
                conversion_value=None,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        assert row["roi"] is None

    def test_roi_null_when_no_conversion_value(self, spark):
        """ROI is null when conversion_value is null (won but no conversion)."""
        records = [
            _make_record(
                won=True,
                price_paid=3.0,
                impression=True,
                click=False,
                conversion=False,
                conversion_value=None,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        assert row["roi"] is None

    def test_shade_ratio_computed(self, spark):
        """shade_ratio = shaded_price / original_price."""
        records = [_make_record(shaded_price=4.0, original_price=5.0)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        # shade_ratio = 4.0 / 5.0 = 0.8
        assert abs(row["shade_ratio"] - 0.8) < 1e-6

    def test_label_1_for_profitable_win(self, spark):
        """Label is 1 when won=True AND conversion_value > price_paid."""
        records = [
            _make_record(
                won=True,
                impression=True,
                click=True,
                conversion=True,
                price_paid=2.0,
                conversion_value=10.0,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        assert row["label"] == 1

    def test_label_0_for_lost_bid(self, spark):
        """Label is 0 for a lost bid."""
        records = [
            _make_record(
                won=False,
                price_paid=None,
                impression=False,
                click=False,
                conversion=False,
                conversion_value=None,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        assert row["label"] == 0

    def test_label_0_for_unprofitable_win(self, spark):
        """Label is 0 when won but conversion_value < price_paid (unprofitable)."""
        records = [
            _make_record(
                won=True,
                impression=True,
                click=True,
                conversion=True,
                price_paid=10.0,
                conversion_value=5.0,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        assert row["label"] == 0

    def test_label_0_for_win_without_conversion(self, spark):
        """Label is 0 when won but no conversion (no revenue signal)."""
        records = [
            _make_record(
                won=True,
                impression=True,
                click=False,
                conversion=False,
                price_paid=3.0,
                conversion_value=None,
            )
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = engineer_features(df)

        row = result.collect()[0]
        assert row["label"] == 0


# ---------------------------------------------------------------------------
# Win Rate Bucket Tests
# ---------------------------------------------------------------------------


class TestWinRateBuckets:
    """Tests for compute_win_rate_buckets."""

    def test_all_wins_bucket_9(self, spark):
        """When all records in a group are wins, bucket should be 9."""
        records = [
            _make_record(hour_of_day=10, device_type="mobile", won=True),
            _make_record(
                request_id="bbbb1111-2222-3333-4444-555566667777",
                hour_of_day=10,
                device_type="mobile",
                won=True,
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = compute_win_rate_buckets(df)

        rows = result.collect()
        for row in rows:
            assert row["win_rate_bucket"] == 9

    def test_no_wins_bucket_0(self, spark):
        """When no records in a group are wins, bucket should be 0."""
        records = [
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                hour_of_day=10,
                device_type="desktop",
                won=False,
                price_paid=None,
                impression=False,
                click=False,
                conversion=False,
                conversion_value=None,
            ),
            _make_record(
                request_id="bbbb1111-2222-3333-4444-555566667777",
                hour_of_day=10,
                device_type="desktop",
                won=False,
                price_paid=None,
                impression=False,
                click=False,
                conversion=False,
                conversion_value=None,
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = compute_win_rate_buckets(df)

        rows = result.collect()
        for row in rows:
            assert row["win_rate_bucket"] == 0

    def test_mixed_wins_intermediate_bucket(self, spark):
        """50% win rate should produce bucket 5."""
        records = [
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                hour_of_day=12,
                device_type="tablet",
                won=True,
            ),
            _make_record(
                request_id="bbbb1111-2222-3333-4444-555566667777",
                hour_of_day=12,
                device_type="tablet",
                won=False,
                price_paid=None,
                impression=False,
                click=False,
                conversion=False,
                conversion_value=None,
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = compute_win_rate_buckets(df)

        rows = result.collect()
        for row in rows:
            # 50% win rate -> floor(0.5 * 10) = 5
            assert row["win_rate_bucket"] == 5


# ---------------------------------------------------------------------------
# PII Validation Tests (Spark-based)
# ---------------------------------------------------------------------------


class TestPIIValidation:
    """Tests for validate_no_raw_pii."""

    def test_valid_hashes_pass(self, spark):
        """Records with valid hex hashes pass through unchanged."""
        records = [
            _make_record(user_id_hash="abc123def456789012345678abcdef01"),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = validate_no_raw_pii(df)

        assert result.count() == 1

    def test_email_in_user_id_hash_dropped(self, spark):
        """Record with email pattern in user_id_hash is dropped."""
        records = [
            _make_record(user_id_hash="user@example.com"),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = validate_no_raw_pii(df)

        assert result.count() == 0

    def test_short_non_hash_string_dropped(self, spark):
        """Record with a short non-hex string in hash column is dropped."""
        records = [
            _make_record(user_id_hash="john"),  # Too short, not a hash
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = validate_no_raw_pii(df)

        assert result.count() == 0

    def test_mixed_valid_and_invalid_records(self, spark):
        """Only valid records are kept when mix of valid/invalid present."""
        records = [
            _make_record(
                request_id="aaaa1111-2222-3333-4444-555566667777",
                user_id_hash="abc123def456789012345678abcdef01",
            ),
            _make_record(
                request_id="bbbb1111-2222-3333-4444-555566667777",
                user_id_hash="plaintext_user_name@mail.org",
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = validate_no_raw_pii(df)

        assert result.count() == 1
        row = result.collect()[0]
        assert row["request_id"] == "aaaa1111-2222-3333-4444-555566667777"
