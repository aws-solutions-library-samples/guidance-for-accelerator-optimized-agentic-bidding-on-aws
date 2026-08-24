"""Tests for the deal yield Glue ETL feature engineering and labeling logic.

Mirrors test_glue_etl.py's structure/fixture conventions exactly (local
PySpark session, pytest.importorskip gate), for
etl.glue_deal_yield_feature_engineering instead of etl.glue_feature_engineering.

Validates: FR-12, FR-13 (deal-floor-margin-requirements.md).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pyspark = pytest.importorskip("pyspark", reason="PySpark required for Glue ETL tests")

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from etl.glue_deal_yield_feature_engineering import (
    FEATURE_COLUMNS,
    compute_deal_yield_features,
    deduplicate_by_deal_event,
    select_floor_training_rows,
    select_margin_training_rows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Schema matching feedback_pipeline_cfn.yaml's DealYieldOutcomeGlueTable
# column list exactly (shared/feedback_models.py's DealYieldOutcomeEvent
# field names/types).
_RAW_SCHEMA = StructType(
    [
        StructField("request_id", StringType(), False),
        StructField("timestamp", DoubleType(), False),
        StructField("model_version", StringType(), False),
        StructField("source", StringType(), False),
        StructField("imp_id", StringType(), False),
        StructField("deal_id", StringType(), False),
        StructField("intent", StringType(), False),
        StructField("original_bidfloor", DoubleType(), False),
        StructField("adjusted_bidfloor", DoubleType(), True),
        StructField("margin_value", DoubleType(), True),
        StructField("margin_calculation_type", IntegerType(), True),
        StructField("won", BooleanType(), False),
        StructField("price_paid", DoubleType(), True),
        StructField("auction_type", IntegerType(), True),
        StructField("category_tier", DoubleType(), False),
        StructField("hour_of_day", IntegerType(), False),
        StructField("day_of_week", IntegerType(), False),
    ]
)


def _make_record(
    request_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    timestamp=1718000000.0,
    model_version="v1.0.0",
    source="live",
    imp_id="imp-1",
    deal_id="deal-1",
    intent="ADJUST_DEAL_FLOOR",
    original_bidfloor=2.0,
    adjusted_bidfloor=2.5,
    margin_value=None,
    margin_calculation_type=None,
    won=True,
    price_paid=2.6,
    auction_type=1,
    category_tier=1.0,
    hour_of_day=14,
    day_of_week=2,
):
    """Create a test record tuple matching _RAW_SCHEMA."""
    return (
        request_id,
        timestamp,
        model_version,
        source,
        imp_id,
        deal_id,
        intent,
        original_bidfloor,
        adjusted_bidfloor,
        margin_value,
        margin_calculation_type,
        won,
        price_paid,
        auction_type,
        category_tier,
        hour_of_day,
        day_of_week,
    )


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test_glue_deal_yield_etl")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Deduplication Tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for deduplicate_by_deal_event."""

    def test_keeps_latest_by_timestamp_for_same_key(self, spark):
        """Same (request_id, deal_id, intent) -> only the latest timestamp survives."""
        records = [
            _make_record(timestamp=1000.0, adjusted_bidfloor=2.1),
            _make_record(timestamp=2000.0, adjusted_bidfloor=2.2),
            _make_record(timestamp=3000.0, adjusted_bidfloor=2.3),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = deduplicate_by_deal_event(df)

        assert result.count() == 1
        row = result.collect()[0]
        assert row["timestamp"] == 3000.0
        assert row["adjusted_bidfloor"] == 2.3

    def test_floor_and_margin_on_same_deal_both_preserved(self, spark):
        """A single deal can carry BOTH a floor and a margin mutation --
        these are genuinely distinct events (BR-4/BR-5), not duplicates,
        even though request_id and deal_id are identical."""
        records = [
            _make_record(intent="ADJUST_DEAL_FLOOR", adjusted_bidfloor=2.5, margin_value=None),
            _make_record(intent="ADJUST_DEAL_MARGIN", adjusted_bidfloor=None, margin_value=0.1),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = deduplicate_by_deal_event(df)

        assert result.count() == 2

    def test_distinct_deals_preserved(self, spark):
        """Records for different deal_ids are all preserved."""
        records = [
            _make_record(deal_id="deal-1", timestamp=1000.0),
            _make_record(deal_id="deal-2", timestamp=2000.0),
            _make_record(deal_id="deal-3", timestamp=3000.0),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        result = deduplicate_by_deal_event(df)

        assert result.count() == 3


# ---------------------------------------------------------------------------
# Feature Reconstruction Tests
# ---------------------------------------------------------------------------


class TestComputeDealYieldFeatures:
    """Tests for compute_deal_yield_features."""

    def test_first_price_onehot(self, spark):
        records = [_make_record(auction_type=1)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["is_first_price"] == 1.0
        assert row["is_second_price"] == 0.0

    def test_second_price_onehot(self, spark):
        records = [_make_record(auction_type=2)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["is_first_price"] == 0.0
        assert row["is_second_price"] == 1.0

    def test_unrecognized_auction_type_is_neither(self, spark):
        """Missing/unrecognized auction_type -> (0.0, 0.0), a real
        "neither", matching features.py's _auction_type_onehot()."""
        records = [_make_record(auction_type=None)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["is_first_price"] == 0.0
        assert row["is_second_price"] == 0.0

    def test_bidfloor_tier_remnant(self, spark):
        records = [_make_record(original_bidfloor=0.5)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["bidfloor"] == 0.5
        assert row["bidfloor_tier"] == 0.0

    def test_bidfloor_tier_mid(self, spark):
        records = [_make_record(original_bidfloor=3.0)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["bidfloor_tier"] == 1.0

    def test_bidfloor_tier_premium(self, spark):
        records = [_make_record(original_bidfloor=7.5)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["bidfloor_tier"] == 2.0

    def test_hour_and_weekday_normalization(self, spark):
        records = [_make_record(hour_of_day=12, day_of_week=3)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert abs(row["hour_norm"] - 0.5) < 1e-9
        assert abs(row["weekday_norm"] - (3 / 7.0)) < 1e-9

    def test_category_tier_passed_through_unchanged(self, spark):
        """category_tier is not recomputed -- deal_yield_feedback.py
        already computed it at emission time."""
        records = [_make_record(category_tier=2.0)]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        row = compute_deal_yield_features(df).collect()[0]
        assert row["category_tier"] == 2.0


# ---------------------------------------------------------------------------
# Labeling Tests
# ---------------------------------------------------------------------------


class TestSelectFloorTrainingRows:
    """Tests for select_floor_training_rows."""

    def test_won_floor_row_labeled_with_multiplier(self, spark):
        records = [
            _make_record(
                intent="ADJUST_DEAL_FLOOR",
                won=True,
                original_bidfloor=2.0,
                adjusted_bidfloor=2.5,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_floor_training_rows(df)

        assert result.count() == 1
        row = result.collect()[0]
        # label = adjusted_bidfloor / original_bidfloor = 2.5 / 2.0 = 1.25
        assert abs(row["label"] - 1.25) < 1e-9
        assert set(result.columns) == {"label", *FEATURE_COLUMNS}

    def test_lost_deal_excluded(self, spark):
        """A deal that did not win produces no floor training row --
        won/price_paid unknown outcomes cannot honestly be labeled."""
        records = [
            _make_record(
                intent="ADJUST_DEAL_FLOOR",
                won=False,
                original_bidfloor=2.0,
                adjusted_bidfloor=2.5,
                price_paid=None,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_floor_training_rows(df)

        assert result.count() == 0

    def test_margin_intent_excluded_from_floor_rows(self, spark):
        records = [
            _make_record(
                intent="ADJUST_DEAL_MARGIN",
                won=True,
                adjusted_bidfloor=None,
                margin_value=0.1,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_floor_training_rows(df)

        assert result.count() == 0

    def test_zero_original_bidfloor_excluded(self, spark):
        """original_bidfloor == 0 would divide-by-zero -- excluded as an
        unusable row rather than fabricating a label."""
        records = [
            _make_record(
                intent="ADJUST_DEAL_FLOOR",
                won=True,
                original_bidfloor=0.0,
                adjusted_bidfloor=0.5,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_floor_training_rows(df)

        assert result.count() == 0


class TestSelectMarginTrainingRows:
    """Tests for select_margin_training_rows."""

    def test_won_margin_row_labeled_with_margin_value(self, spark):
        records = [
            _make_record(
                intent="ADJUST_DEAL_MARGIN",
                won=True,
                adjusted_bidfloor=None,
                margin_value=0.15,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_margin_training_rows(df)

        assert result.count() == 1
        row = result.collect()[0]
        assert abs(row["label"] - 0.15) < 1e-9
        assert set(result.columns) == {"label", *FEATURE_COLUMNS}

    def test_lost_deal_excluded(self, spark):
        records = [
            _make_record(
                intent="ADJUST_DEAL_MARGIN",
                won=False,
                margin_value=0.15,
                price_paid=None,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_margin_training_rows(df)

        assert result.count() == 0

    def test_floor_intent_excluded_from_margin_rows(self, spark):
        records = [
            _make_record(
                intent="ADJUST_DEAL_FLOOR",
                won=True,
                original_bidfloor=2.0,
                adjusted_bidfloor=2.5,
                margin_value=None,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_margin_training_rows(df)

        assert result.count() == 0

    def test_null_margin_value_excluded(self, spark):
        records = [
            _make_record(
                intent="ADJUST_DEAL_MARGIN",
                won=True,
                margin_value=None,
            )
        ]
        df = compute_deal_yield_features(spark.createDataFrame(records, schema=_RAW_SCHEMA))
        result = select_margin_training_rows(df)

        assert result.count() == 0


# ---------------------------------------------------------------------------
# End-to-end (dedup -> features -> select) mixed-batch test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Exercises the full transform chain against a mixed batch, matching
    how main() actually chains these functions."""

    def test_mixed_batch_produces_correct_floor_and_margin_counts(self, spark):
        records = [
            # Deal A: won floor mutation -> 1 floor row
            _make_record(
                request_id="req-a", deal_id="deal-a", intent="ADJUST_DEAL_FLOOR",
                won=True, original_bidfloor=2.0, adjusted_bidfloor=2.4, margin_value=None,
            ),
            # Deal A also has a margin mutation on the same request/deal -> 1 margin row
            _make_record(
                request_id="req-a", deal_id="deal-a", intent="ADJUST_DEAL_MARGIN",
                won=True, adjusted_bidfloor=None, margin_value=0.2,
            ),
            # Deal B: lost -> excluded from both
            _make_record(
                request_id="req-b", deal_id="deal-b", intent="ADJUST_DEAL_FLOOR",
                won=False, original_bidfloor=3.0, adjusted_bidfloor=3.3,
                margin_value=None, price_paid=None,
            ),
            # Deal C: duplicate floor event (re-emitted), only latest kept
            _make_record(
                request_id="req-c", deal_id="deal-c", intent="ADJUST_DEAL_FLOOR",
                won=True, original_bidfloor=4.0, adjusted_bidfloor=4.0,
                margin_value=None, timestamp=1000.0,
            ),
            _make_record(
                request_id="req-c", deal_id="deal-c", intent="ADJUST_DEAL_FLOOR",
                won=True, original_bidfloor=4.0, adjusted_bidfloor=4.8,
                margin_value=None, timestamp=2000.0,
            ),
        ]
        df = spark.createDataFrame(records, schema=_RAW_SCHEMA)
        df = deduplicate_by_deal_event(df)
        df = compute_deal_yield_features(df)

        floor_rows = select_floor_training_rows(df).collect()
        margin_rows = select_margin_training_rows(df).collect()

        assert len(floor_rows) == 2  # deal-a, deal-c (latest only)
        assert len(margin_rows) == 1  # deal-a's margin mutation

        deal_c_label = [r["label"] for r in floor_rows if abs(r["label"] - 1.2) < 1e-9]
        assert deal_c_label, "expected deal-c's latest (4.8/4.0=1.2) label to survive dedup"
