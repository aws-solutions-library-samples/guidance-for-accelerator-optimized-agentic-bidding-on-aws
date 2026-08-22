"""AWS Glue ETL job for feature engineering and labeling of deal yield outcomes.

Reads raw deal floor/margin outcome records from the Glue catalog
(feedback_pipeline.raw_deal_yield_outcomes), de-duplicates by
(request_id, deal_id, intent) (keeping the latest by timestamp), reconstructs
the same feature vector deal_yield_manager's own
containers/deal_yield_manager/features.py.build_feature_vector() computes at
inference time, and writes TWO labeled Parquet datasets -- one per XGBoost
target (floor, margin) -- for XGBoostTrainingPipeline to train on. See
source/training/xgboost_pipeline.py's TARGET_FLOOR/TARGET_MARGIN: Triton's
FIL backend does not support multi-output regression, so the two targets
are independently-trained single-output models with their own training data.

Feature reconstruction is duplicated here rather than imported from
containers.deal_yield_manager.features -- consistent with the
"no cross-unit code dependency" boundary
aidlc-docs/construction/deal-yield-outcome-capture/functional-design/
business-logic-model.md documents for deal_yield_feedback.py's own
_classify_category_tier() mirror, and also a practical necessity: a Glue job
script has no access to the containers/ package (it is not bundled into the
Glue script upload). is_first_price/is_second_price/bidfloor_tier are
recomputed here from auction_type/original_bidfloor (the same real fields
DealYieldOutcomeEvent already carries); category_tier, hour_of_day, and
day_of_week are read directly off the event, since deal_yield_feedback.py
already computed and stored them at emission time -- no need to recompute.

Labeling function: filters to won=True rows (a deal that actually cleared)
and uses the *action value the container actually took* (floor_multiplier =
adjusted_bidfloor / original_bidfloor, or margin_value directly) as the
regression label. This is a real, non-fabricated regression target derived
entirely from observed outcomes -- not synthesized here -- but it does mean
training data only accumulates from deals with a *known* won value. Per
deal-yield-outcome-capture/functional-design/business-rules.md BR-7, no
downstream-signal-association path exists yet for deal-level outcomes on
live traffic, so won/price_paid are real "unknown" (False/None) defaults
there today; only source="load_test" events (deliberately, honestly
synthetic per NFR-2) and the deal_yield_manager container's own bounded
exploration (source/containers/deal_yield_manager/exploration.py, disclosed
via a ":explore" model_version suffix) currently produce the variance this
labeling function needs. This is a documented limitation, not a hidden
assumption -- extending won/price_paid capture to live traffic is future
work (a live downstream-signal path), out of scope here.

Job Parameters:
    --window_start: Optional. ISO 8601 timestamp for the start of the
                    processing window (inclusive). See
                    glue_feature_engineering.py's docstring for the
                    window_start/window_end/window_hours contract this
                    mirrors exactly.
    --window_end:   Optional. ISO 8601 timestamp for the end of the window
                    (exclusive). Must be provided together with
                    --window_start, or not at all.
    --window_hours: Optional (default: 6). Self-computed rolling window
                    when window_start/window_end are not both provided.
    --output_bucket: S3 bucket for labeled training output.
    --database_name: Glue catalog database name (default: feedback_pipeline).
    --table_name:    Glue catalog table name (default: raw_deal_yield_outcomes).

Requirements: FR-12, FR-13 (deal-floor-margin-requirements.md).
"""

import logging
import os
import sys

import boto3
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

try:
    # Local/test execution: etl/ is a real package on the path (source/ is
    # the pytest rootdir -- see test_glue_deal_yield_etl.py).
    from etl.glue_feature_engineering import _resolve_window
except ImportError:
    # Glue runtime: --extra-py-files delivers glue_feature_engineering.py as
    # a flat top-level module (Glue does not preserve package structure for
    # --extra-py-files entries, and does not add ScriptLocation's own
    # directory to sys.path either -- confirmed against AWS's own
    # --extra-py-files docs). See glue_etl_cfn.yaml's
    # DealYieldFeatureEngineeringJob for the matching --extra-py-files
    # argument that makes this import resolve at run time.
    from glue_feature_engineering import _resolve_window  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("glue_deal_yield_feature_engineering")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constants -- mirror containers/deal_yield_manager/features.py's thresholds
# exactly, so the reconstructed feature vector matches what the model was
# actually served at inference time.
# ---------------------------------------------------------------------------
TARGET_FLOOR = "floor"
TARGET_MARGIN = "margin"

_REMNANT_BIDFLOOR_MAX = 1.0
_PREMIUM_BIDFLOOR_MIN = 5.0

INTENT_FLOOR = "ADJUST_DEAL_FLOOR"
INTENT_MARGIN = "ADJUST_DEAL_MARGIN"

# Output column order: label first (SageMaker's built-in XGBoost algorithm
# assumes the target variable is the first column for columnar input --
# see AWS docs "How to use SageMaker AI XGBoost"), then the same 7 features
# in the exact order build_feature_vector() returns them.
FEATURE_COLUMNS = [
    "is_first_price",
    "is_second_price",
    "bidfloor",
    "bidfloor_tier",
    "category_tier",
    "hour_norm",
    "weekday_norm",
]


# ---------------------------------------------------------------------------
# Core ETL functions (exported for unit testing)
# ---------------------------------------------------------------------------


def deduplicate_by_deal_event(df: DataFrame) -> DataFrame:
    """De-duplicate records by (request_id, deal_id, intent), keeping the
    latest by timestamp.

    Unlike glue_feature_engineering.py's deduplicate_by_request_id() (keyed
    on request_id alone), a single bid request/deal can carry BOTH an
    ADJUST_DEAL_FLOOR and an ADJUST_DEAL_MARGIN mutation for the same deal
    (business-rules.md BR-4/BR-5 -- two independent, atomic mutations) --
    keying on request_id alone would incorrectly collapse those two
    genuinely-distinct events into one.
    """
    window = Window.partitionBy("request_id", "deal_id", "intent").orderBy(
        F.col("timestamp").desc()
    )
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


def compute_deal_yield_features(df: DataFrame) -> DataFrame:
    """Reconstructs the inference-time feature vector's columns.

    Adds: is_first_price, is_second_price (one-hot from auction_type),
    bidfloor (renamed from original_bidfloor -- the deal's floor BEFORE
    this event's adjustment, matching feature[2] at prediction time),
    bidfloor_tier (ordinal from bidfloor), hour_norm, weekday_norm
    (normalized from hour_of_day/day_of_week). category_tier is used
    as-is -- deal_yield_feedback.py already computed and stored it at
    emission time (same classify_content_tier() logic), so it needs no
    recomputation here.
    """
    df = df.withColumn(
        "is_first_price",
        F.when(F.col("auction_type") == 1, F.lit(1.0)).otherwise(F.lit(0.0)),
    ).withColumn(
        "is_second_price",
        F.when(F.col("auction_type") == 2, F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    df = df.withColumn("bidfloor", F.col("original_bidfloor").cast(DoubleType()))

    df = df.withColumn(
        "bidfloor_tier",
        F.when(F.col("bidfloor") <= _REMNANT_BIDFLOOR_MAX, F.lit(0.0))
        .when(F.col("bidfloor") >= _PREMIUM_BIDFLOOR_MIN, F.lit(2.0))
        .otherwise(F.lit(1.0)),
    )

    df = df.withColumn("hour_norm", F.col("hour_of_day").cast(DoubleType()) / 24.0)
    df = df.withColumn("weekday_norm", F.col("day_of_week").cast(DoubleType()) / 7.0)

    return df


def select_floor_training_rows(df: DataFrame) -> DataFrame:
    """Filters to labeled ADJUST_DEAL_FLOOR rows for the floor target.

    Label = floor_multiplier, the real action value the container took
    (adjusted_bidfloor / original_bidfloor), computed only where
    original_bidfloor > 0 (an unusable row otherwise -- a real "can't
    compute this label", never a fabricated one) and only for deals that
    actually won (see module docstring's labeling-function rationale).
    """
    floor_df = df.filter(
        (F.col("intent") == INTENT_FLOOR)
        & (F.col("won") == True)  # noqa: E712
        & (F.col("original_bidfloor") > 0)
        & (F.col("adjusted_bidfloor").isNotNull())
    )
    floor_df = floor_df.withColumn(
        "label", F.col("adjusted_bidfloor") / F.col("original_bidfloor")
    )
    return floor_df.select("label", *FEATURE_COLUMNS)


def select_margin_training_rows(df: DataFrame) -> DataFrame:
    """Filters to labeled ADJUST_DEAL_MARGIN rows for the margin target.

    Label = margin_value, the real action value the container took,
    for deals that actually won (see module docstring).
    """
    margin_df = df.filter(
        (F.col("intent") == INTENT_MARGIN)
        & (F.col("won") == True)  # noqa: E712
        & (F.col("margin_value").isNotNull())
    )
    margin_df = margin_df.withColumn("label", F.col("margin_value"))
    return margin_df.select("label", *FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Glue Job Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the AWS Glue ETL job."""
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    # Only JOB_NAME and output_bucket are required -- see module docstring
    # and glue_feature_engineering.py's identical convention for why the
    # rest are parsed separately as optional.
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "output_bucket"])

    def _optional_arg(name: str) -> str | None:
        flag = f"--{name}"
        if flag in sys.argv:
            return sys.argv[sys.argv.index(flag) + 1]
        return None

    job_name = args["JOB_NAME"]
    output_bucket = args["output_bucket"]
    database_name = _optional_arg("database_name") or "feedback_pipeline"
    table_name = _optional_arg("table_name") or "raw_deal_yield_outcomes"

    start_dt, end_dt = _resolve_window(
        _optional_arg("window_start"), _optional_arg("window_end"), _optional_arg("window_hours")
    )
    window_start = start_dt.isoformat()
    window_end = end_dt.isoformat()

    logger.info(
        "Starting ETL job %s: window=[%s, %s), database=%s, table=%s",
        job_name,
        window_start,
        window_end,
        database_name,
        table_name,
    )

    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(job_name, args)

    try:
        window_start_epoch = start_dt.timestamp()
        window_end_epoch = end_dt.timestamp()

        # Read directly from the table's S3 location, bypassing partition
        # metadata -- see glue_feature_engineering.py's identical comment
        # for why (nothing in this pipeline registers partition_date/
        # partition_hour, so spark.read.table() would always return 0 rows).
        boto3_glue = boto3.client("glue", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        table_location = boto3_glue.get_table(
            DatabaseName=database_name, Name=table_name
        )["Table"]["StorageDescriptor"]["Location"]

        logger.info("Reading from S3 location: %s (table=%s.%s)", table_location, database_name, table_name)

        df = spark.read.format("parquet").load(table_location)
        df = df.filter(
            (F.col("timestamp") >= window_start_epoch)
            & (F.col("timestamp") < window_end_epoch)
        )

        record_count = df.count()
        logger.info("Read %d records within time window", record_count)

        if record_count == 0:
            logger.info("No records in window -- nothing to process.")
            job.commit()
            return

        # No PII validation step here -- unlike raw_bid_outcomes,
        # raw_deal_yield_outcomes has no hash-bearing or otherwise
        # PII-shaped column at all (see feedback_pipeline_cfn.yaml's
        # DealYieldOutcomeGlueTable column list / BR-1's distinct-fields
        # rule) -- there is nothing to check.
        df = deduplicate_by_deal_event(df)
        deduped_count = df.count()
        logger.info(
            "After de-duplication: %d records (%d duplicates removed)",
            deduped_count,
            record_count - deduped_count,
        )

        df = compute_deal_yield_features(df)

        floor_df = select_floor_training_rows(df)
        margin_df = select_margin_training_rows(df)

        floor_output_path = (
            f"s3://{output_bucket}/training-data-deal-yield-{TARGET_FLOOR}/"
            f"window_start={window_start}/window_end={window_end}/"
        )
        margin_output_path = (
            f"s3://{output_bucket}/training-data-deal-yield-{TARGET_MARGIN}/"
            f"window_start={window_start}/window_end={window_end}/"
        )

        floor_count = floor_df.count()
        margin_count = margin_df.count()
        logger.info(
            "Writing %d labeled floor rows to %s, %d labeled margin rows to %s",
            floor_count,
            floor_output_path,
            margin_count,
            margin_output_path,
        )

        if floor_count > 0:
            floor_df.write.mode("overwrite").parquet(floor_output_path)
        else:
            logger.info("No won ADJUST_DEAL_FLOOR rows in window -- skipping floor write.")

        if margin_count > 0:
            margin_df.write.mode("overwrite").parquet(margin_output_path)
        else:
            logger.info("No won ADJUST_DEAL_MARGIN rows in window -- skipping margin write.")

    except Exception:
        logger.exception("ETL job failed")
        raise
    finally:
        job.commit()
        logger.info("Job committed.")


if __name__ == "__main__":
    main()
