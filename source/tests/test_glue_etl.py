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
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

from spark_java import SKIP_REASON, resolve_java_home

_SOURCE_ETL = Path(__file__).resolve().parents[1] / "etl"

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
    _add_response_label,
    _contains_pii_pattern,
    _is_plausible_hash,
    compute_win_rate_buckets,
    deduplicate_by_request_id,
    engineer_features,
    ensure_outcome_provenance,
    validate_no_raw_pii,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Schema matching the Glue catalog table (raw_bid_outcomes) -- these columns
# match shared/feedback_models.py's BidShadingOutcomeEvent field names/types
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
        # Nullable: the outcome signals are tri-state, and NULL means "not reported
        # yet" -- a bid-time row has no outcome at all until a signal arrives. The
        # live Glue table allows null on every non-partition column; declaring these
        # NOT NULL here made the fixture unable to represent the rows the pipeline
        # actually carries.
        StructField("won", BooleanType(), True),
        StructField("price_paid", DoubleType(), True),
        StructField("impression", BooleanType(), True),
        StructField("click", BooleanType(), True),
        StructField("conversion", BooleanType(), True),
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
    if resolve_java_home() is None:
        pytest.skip(SKIP_REASON)

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
# Outcome Provenance Tests
# ---------------------------------------------------------------------------


class TestOutcomeProvenance:
    """The training dataset must say where its outcomes came from.

    The trainer reports this mix and writes it into the model manifest, so a model
    trained on simulated outcomes stays identifiable as one. A partition that
    predates the column reports "unresolved" rather than claiming its outcomes were
    observed.
    """

    _PROVENANCE_SCHEMA = StructType(
        _RAW_SCHEMA.fields
        + [StructField("outcome_provenance", StringType(), True)]
    )

    def test_missing_column_is_added_as_unresolved(self, spark):
        df = spark.createDataFrame([_make_record()], schema=_RAW_SCHEMA)
        result = engineer_features(df)

        assert "outcome_provenance" in result.columns
        assert result.collect()[0]["outcome_provenance"] == "unresolved"

    def test_existing_label_is_preserved(self, spark):
        df = spark.createDataFrame(
            [_make_record() + ("simulated",)], schema=self._PROVENANCE_SCHEMA
        )
        result = engineer_features(df)
        assert result.collect()[0]["outcome_provenance"] == "simulated"

    def test_observed_label_is_preserved(self, spark):
        df = spark.createDataFrame(
            [_make_record() + ("observed",)], schema=self._PROVENANCE_SCHEMA
        )
        result = engineer_features(df)
        assert result.collect()[0]["outcome_provenance"] == "observed"

    def test_null_becomes_unresolved_not_observed(self, spark):
        df = spark.createDataFrame(
            [_make_record() + (None,)], schema=self._PROVENANCE_SCHEMA
        )
        result = engineer_features(df)
        assert result.collect()[0]["outcome_provenance"] == "unresolved"

    def test_enriched_row_wins_dedup_at_equal_timestamps(self, spark):
        """Regression: the enriched row shares its bid's timestamp, so they tie.

        An enriched outcome event keeps the timestamp of the bid it belongs to,
        because the processing window filters on that column. Ordering by timestamp
        alone therefore picks arbitrarily between the unlabelled bid row and its
        labelled replacement, discarding roughly half of all labels with no error
        anywhere. The tie-break is how many outcome signals the row carries.
        """
        rid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        unresolved = _make_record(
            request_id=rid, timestamp=1718000000.0, won=None,
            price_paid=None, impression=None, click=None, conversion=None,
            conversion_value=None,
        ) + ("unresolved",)
        enriched = _make_record(
            request_id=rid, timestamp=1718000000.0, won=True,
            price_paid=3.5, impression=True, click=True, conversion=True,
            conversion_value=25.0,
        ) + ("simulated",)

        for rows in ([unresolved, enriched], [enriched, unresolved]):
            df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
            kept = deduplicate_by_request_id(df).collect()
            assert len(kept) == 1
            assert kept[0]["outcome_provenance"] == "simulated"
            assert kept[0]["conversion"] is True

    def test_dedup_still_prefers_a_later_timestamp(self, spark):
        """Resolvedness is only the tie-break; a genuinely later row still wins."""
        rid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        older_resolved = _make_record(
            request_id=rid, timestamp=1718000000.0, won=True, price_paid=3.0,
            impression=True, click=True, conversion=True, conversion_value=10.0,
        ) + ("observed",)
        newer_unresolved = _make_record(
            request_id=rid, timestamp=1718009999.0, won=None, price_paid=None,
            impression=None, click=None, conversion=None, conversion_value=None,
        ) + ("unresolved",)
        df = spark.createDataFrame(
            [older_resolved, newer_unresolved], schema=self._PROVENANCE_SCHEMA
        )
        kept = deduplicate_by_request_id(df).collect()
        assert len(kept) == 1
        assert kept[0]["timestamp"] == 1718009999.0

    def test_raw_read_merges_schemas_across_file_versions(self, spark, tmp_path):
        """Regression: an old 20-column file erased four columns from the whole run.

        The raw prefix accumulates Parquet written by different versions of the event.
        Without mergeSchema, Spark infers from one arbitrary file, so a pre-upgrade
        partition silently drops every newer column — confirmed live on an output
        dataset that had no provenance and no geo at all while the raw rows had both.
        """
        old = spark.createDataFrame([_make_record()], schema=_RAW_SCHEMA)
        new = spark.createDataFrame(
            [_make_record() + ("simulated",)], schema=self._PROVENANCE_SCHEMA
        )
        root = str(tmp_path / "raw")
        old.write.mode("append").parquet(root)
        new.write.mode("append").parquet(root)

        without = spark.read.format("parquet").load(root)
        merged = spark.read.format("parquet").option("mergeSchema", "true").load(root)

        assert "outcome_provenance" in merged.columns
        values = {r["outcome_provenance"] for r in merged.collect()}
        assert "simulated" in values
        # The unmerged read is allowed to be either schema — the point is only that
        # it is not guaranteed, which is why the job must not rely on it.
        assert len(merged.columns) >= len(without.columns)

    def test_etl_sets_merge_schema_on_the_raw_read(self):
        source = (
            _SOURCE_ETL / "glue_feature_engineering.py"
        ).read_text()
        assert '.option("mergeSchema", "true")' in source

    def test_dedup_drops_its_helper_column(self, spark):
        df = spark.createDataFrame(
            [_make_record() + ("simulated",)], schema=self._PROVENANCE_SCHEMA
        )
        assert "_resolved_rank" not in deduplicate_by_request_id(df).columns

    def test_absent_response_is_null_before_the_attribution_deadline(self, spark):
        """Not yet had its chance to convert — unlabelled, not a negative."""
        rows = [
            _make_record(
                timestamp=1718000000.0, won=True, price_paid=3.0, impression=True,
                click=None, conversion=None, conversion_value=None,
            ) + ("simulated",)
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        out = engineer_features(df, attribution_deadline_epoch=1717000000.0).collect()
        assert out[0]["label_conversion"] is None
        assert out[0]["label_click"] is None

    def test_absent_response_is_zero_after_the_attribution_deadline(self, spark):
        """Had its full window and reported nothing — that is a real negative.

        Without this, every labelled row is a 1: downstream signals only report events
        that happened, so nothing ever says "no click". Confirmed on a live dataset —
        37 click labels and 11 conversion labels, all 1, no zeros.
        """
        rows = [
            _make_record(
                timestamp=1718000000.0, won=True, price_paid=3.0, impression=True,
                click=None, conversion=None, conversion_value=None,
            ) + ("simulated",)
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        out = engineer_features(df, attribution_deadline_epoch=1719000000.0).collect()
        assert out[0]["label_conversion"] == 0
        assert out[0]["label_click"] == 0

    def test_unresolved_row_stays_null_even_past_the_deadline(self, spark):
        """Nothing reported on this bid at all, so there is no impression to follow."""
        rows = [
            _make_record(
                timestamp=1718000000.0, won=None, price_paid=None, impression=None,
                click=None, conversion=None, conversion_value=None,
            ) + ("unresolved",)
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        out = engineer_features(df, attribution_deadline_epoch=1719000000.0).collect()
        assert out[0]["label_conversion"] is None
        assert out[0]["label_click"] is None

    def test_reported_response_is_one_regardless_of_the_deadline(self, spark):
        rows = [
            _make_record(
                timestamp=1718000000.0, won=True, price_paid=3.0, impression=True,
                click=True, conversion=True, conversion_value=20.0,
            ) + ("simulated",)
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        for deadline in (1717000000.0, 1719000000.0):
            out = engineer_features(df, attribution_deadline_epoch=deadline).collect()
            assert out[0]["label_conversion"] == 1
            assert out[0]["label_click"] == 1

    def test_no_deadline_preserves_the_previous_behaviour(self, spark):
        rows = [
            _make_record(
                timestamp=1718000000.0, won=True, price_paid=3.0, impression=True,
                click=None, conversion=None, conversion_value=None,
            ) + ("simulated",)
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        out = engineer_features(df).collect()
        assert out[0]["label_conversion"] is None
        assert out[0]["label_click"] is None

    def test_deadline_produces_both_label_classes(self, spark):
        """The property the dataset gate actually needs."""
        rows = [
            _make_record(
                request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                timestamp=1718000000.0, won=True, price_paid=3.0, impression=True,
                click=True, conversion=True, conversion_value=20.0,
            ) + ("simulated",),
            _make_record(
                request_id="b1b2c3d4-e5f6-7890-abcd-ef1234567890",
                timestamp=1718000001.0, won=True, price_paid=3.0, impression=True,
                click=None, conversion=None, conversion_value=None,
            ) + ("simulated",),
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        out = engineer_features(df, attribution_deadline_epoch=1719000000.0)
        labels = {r["label_click"] for r in out.collect()}
        assert labels == {0, 1}

    def test_labelled_row_survives_the_full_pipeline(self, spark):
        """End to end: two rows in, one labelled row out."""
        rid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        rows = [
            _make_record(
                request_id=rid, timestamp=1718000000.0, won=None, price_paid=None,
                impression=None, click=None, conversion=None, conversion_value=None,
            ) + ("unresolved",),
            _make_record(
                request_id=rid, timestamp=1718000000.0, won=True, price_paid=3.5,
                impression=True, click=True, conversion=True, conversion_value=25.0,
            ) + ("simulated",),
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        out = engineer_features(deduplicate_by_request_id(df)).collect()
        assert len(out) == 1
        assert out[0]["label_conversion"] == 1
        assert out[0]["label_click"] == 1
        assert out[0]["outcome_provenance"] == "simulated"

    def test_mixed_provenance_survives_row_by_row(self, spark):
        rows = [
            _make_record(request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890")
            + ("simulated",),
            _make_record(request_id="b1b2c3d4-e5f6-7890-abcd-ef1234567890")
            + ("observed",),
            _make_record(request_id="c1b2c3d4-e5f6-7890-abcd-ef1234567890")
            + (None,),
        ]
        df = spark.createDataFrame(rows, schema=self._PROVENANCE_SCHEMA)
        result = engineer_features(df)

        by_id = {
            r["request_id"][0]: r["outcome_provenance"] for r in result.collect()
        }
        assert by_id == {"a": "simulated", "b": "observed", "c": "unresolved"}


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


# ---------------------------------------------------------------------------
# Attribution-deadline labelling (task 6.1b)
# ---------------------------------------------------------------------------


class TestAttributionDeadlineLabelling:
    """Runs the three-state label rule instead of asserting its source text.

    `test_label_and_calibration.py` greps `glue_feature_engineering.py` for this rule,
    because that file predates PySpark being available to the suite. Grepping proves the
    code SAYS the rule. Task 6.1b changed what an absent signal MEANS, so the rule needs
    to be executed, not read.

    The deadline is an absolute epoch, so these cases are fully determined by the
    (timestamp, won, signal) triple and need no clock.
    """

    _DEADLINE = 1000.0

    _SCHEMA = StructType(
        [
            StructField("request_id", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("won", BooleanType(), True),
            StructField("conversion", BooleanType(), True),
        ]
    )

    # (case, timestamp, won, conversion, expected_with_deadline, expected_without)
    _CASES = [
        # Nothing has reported on the bid. There is no impression for a response to
        # have followed, so this is unlabelled however long ago it happened.
        ("unresolved-before-deadline", 900.0, None, None, None, None),
        ("unresolved-after-deadline", 1100.0, None, None, None, None),
        # Won, nothing reported, window closed -> absence is now evidence.
        ("won-absent-past-deadline", 900.0, True, None, 0, None),
        # Won, nothing reported, window still open -> too early to call it a negative.
        ("won-absent-window-open", 1100.0, True, None, None, None),
        # The boundary is inclusive.
        ("won-absent-at-deadline", 1000.0, True, None, 0, None),
        # A lost bid cannot have produced a response: known, not missing.
        ("lost", 900.0, False, None, 0, 0),
        ("converted", 900.0, True, True, 1, 1),
        ("explicit-negative", 900.0, True, False, 0, 0),
    ]

    def _labels(self, spark, deadline):
        rows = [(c[0], c[1], c[2], c[3]) for c in self._CASES]
        df = spark.createDataFrame(rows, self._SCHEMA)
        args = (df, "label_conversion", "conversion")
        labelled = (
            _add_response_label(*args, deadline)
            if deadline is not None
            else _add_response_label(*args)
        )
        return {r["request_id"]: r["label_conversion"] for r in labelled.collect()}

    def test_every_case_labels_as_specified_with_a_deadline(self, spark) -> None:
        got = self._labels(spark, self._DEADLINE)
        expected = {c[0]: c[4] for c in self._CASES}
        assert got == expected

    def test_no_deadline_reproduces_the_previous_behaviour(self, spark) -> None:
        """A caller that has not opted in must be completely unaffected by 6.1b."""
        got = self._labels(spark, None)
        expected = {c[0]: c[5] for c in self._CASES}
        assert got == expected

    def test_an_unresolved_row_is_never_labelled_zero(self, spark) -> None:
        """The guard that matters, asserted on its own so a regression names itself.

        A 0 here would make an unlabelled dataset look like a dataset of negatives and
        train the model to predict zero at a healthy-looking loss.
        """
        unresolved = [c[0] for c in self._CASES if c[2] is None]
        assert unresolved, "fixture must contain at least one unresolved row"
        for deadline in (self._DEADLINE, None):
            got = self._labels(spark, deadline)
            for case in unresolved:
                assert got[case] is None, (
                    "%s became %r with deadline=%r; an unresolved row must stay NULL"
                    % (case, got[case], deadline)
                )

    def test_the_deadline_only_ever_adds_zeros(self, spark) -> None:
        """Supplying a deadline must not change a 1 into anything, or invent a 1.

        It resolves absences. Any other difference between the two modes would mean the
        deadline is altering labels that were already known.
        """
        with_deadline = self._labels(spark, self._DEADLINE)
        without = self._labels(spark, None)
        for case, before in without.items():
            after = with_deadline[case]
            if before is None:
                assert after in (None, 0)
            else:
                assert after == before, (
                    "%s was %r without a deadline and %r with one" % (case, before, after)
                )
