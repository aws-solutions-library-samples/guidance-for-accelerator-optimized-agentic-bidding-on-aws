"""AWS Glue ETL job for feature engineering and labeling of bid outcomes.

This script runs in AWS Glue's managed Spark environment (Glue 4.0, Spark 3.3).
It reads raw bid outcome records and signal events from the Glue Data Catalog,
joins signals to bid outcomes by request_id, windows by event_timestamp,
de-duplicates by request_id, computes training labels, and writes labeled
Parquet datasets for SageMaker model retraining.

Job Parameters (passed via Glue job default arguments or overrides):
    --JOB_NAME:         Glue job name (auto-injected by Glue)
    --start_timestamp:  Start of processing window in epoch milliseconds (inclusive)
    --end_timestamp:    End of processing window in epoch milliseconds (exclusive)
    --output_path:      S3 path for labeled training data output
    --database_name:    Glue catalog database name (default: feedback_pipeline)
    --table_name:       Glue catalog table name (default: raw_bid_outcomes)

Requirements: 2.2, 2.3, 2.4, 2.6

Only hashed identifiers (user_id_hash, site_domain_hash) are retained in output.
Raw PII is never persisted in labeled datasets.
"""

import logging
import sys

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, BooleanType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("feedback_etl")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Columns that are safe to include in output (hashed identifiers only, no raw PII)
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    # Primary key
    "request_id",
    "event_timestamp",
    # Bid context
    "model_type",
    "model_version",
    "intent",
    # Pricing
    "original_price",
    "shaded_price",
    "bid_floor",
    "price_paid",
    # Outcome signals
    "won",
    "impression",
    "click",
    "conversion",
    "conversion_value",
    # Hashed identifiers only (no raw PII)
    "user_id_hash",
    "site_domain_hash",
    # Context features (non-PII)
    "device_type",
    "geo_country",
    "hour_of_day",
    "day_of_week",
    "has_video",
    "iab_categories",
    # Parameters at bid time
    "shade_factor_used",
    "conversion_value_estimate",
    # Computed labels
    "label_won",
    "label_ctr",
    "label_conversion_rate",
    "label_roi",
]

# Columns that must NEVER appear in output (raw PII)
PII_COLUMNS_TO_DROP = [
    "user_id",
    "email",
    "name",
    "first_name",
    "last_name",
    "phone",
    "address",
    "site_domain",
    "ip_address",
]


# ---------------------------------------------------------------------------
# Core ETL functions
# ---------------------------------------------------------------------------


def filter_by_time_window(
    df: DataFrame, start_timestamp_millis: int, end_timestamp_millis: int
) -> DataFrame:
    """Filter records to include only those within the configured time window.

    Args:
        df: Input DataFrame with event_timestamp column (epoch millis).
        start_timestamp_millis: Inclusive start of window (epoch millis).
        end_timestamp_millis: Exclusive end of window (epoch millis).

    Returns:
        Filtered DataFrame containing only records within [start, end).
    """
    return df.filter(
        (F.col("event_timestamp") >= start_timestamp_millis)
        & (F.col("event_timestamp") < end_timestamp_millis)
    )


def join_signals_to_outcomes(
    bid_outcomes_df: DataFrame, signals_df: DataFrame
) -> DataFrame:
    """Join signal events (impression, click, conversion) to bid outcomes by request_id.

    Signal events arrive after the initial bid outcome. This join updates the
    outcome record with the latest signal information. When multiple signal
    records exist for the same request_id, we take the latest one (by event_timestamp)
    since later signals accumulate prior signals (conversion implies click implies impression).

    Args:
        bid_outcomes_df: DataFrame of initial bid outcome records.
        signals_df: DataFrame of signal events (impression/click/conversion updates).

    Returns:
        DataFrame with signal fields updated from the latest signal event per request_id.
    """
    if signals_df is None or signals_df.rdd.isEmpty():
        return bid_outcomes_df

    # De-duplicate signals: keep the latest signal per request_id
    signal_window = Window.partitionBy("request_id").orderBy(
        F.col("event_timestamp").desc()
    )
    latest_signals = (
        signals_df.withColumn("_sig_row", F.row_number().over(signal_window))
        .filter(F.col("_sig_row") == 1)
        .drop("_sig_row")
    )

    # Select signal columns to join
    signal_cols = ["request_id", "impression", "click", "conversion", "conversion_value"]
    available_signal_cols = [c for c in signal_cols if c in latest_signals.columns]
    latest_signals = latest_signals.select(available_signal_cols)

    # Rename signal columns to avoid ambiguity during join
    for col_name in available_signal_cols:
        if col_name != "request_id":
            latest_signals = latest_signals.withColumnRenamed(
                col_name, f"_sig_{col_name}"
            )

    # Left join: keep all bid outcomes, update signals where available
    joined = bid_outcomes_df.join(latest_signals, on="request_id", how="left")

    # Coalesce signal fields: prefer signal update over original
    signal_fields = [c for c in available_signal_cols if c != "request_id"]
    for field in signal_fields:
        sig_col = f"_sig_{field}"
        if sig_col in joined.columns:
            joined = joined.withColumn(
                field,
                F.coalesce(F.col(sig_col), F.col(field)),
            ).drop(sig_col)

    return joined


def deduplicate_by_request_id(df: DataFrame) -> DataFrame:
    """De-duplicate records by request_id, keeping the latest by event_timestamp.

    When the same request_id appears multiple times (e.g., re-emitted with
    signal updates), we keep only the record with the highest event_timestamp.

    Args:
        df: Input DataFrame with request_id and event_timestamp columns.

    Returns:
        De-duplicated DataFrame with one record per request_id.
    """
    window = Window.partitionBy("request_id").orderBy(
        F.col("event_timestamp").desc()
    )
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


def compute_labels(df: DataFrame) -> DataFrame:
    """Compute training labels for model retraining.

    Labels computed:
        - label_won: boolean — did the bid win the auction?
        - label_ctr: 1.0 if click occurred, else 0.0
        - label_conversion_rate: 1.0 if conversion occurred, else 0.0
        - label_roi: (conversion_value - price_paid) / price_paid
                     if won and price_paid > 0, else null

    Args:
        df: Input DataFrame with outcome signal columns.

    Returns:
        DataFrame with label columns appended.
    """
    # label_won: boolean indicating auction win
    df = df.withColumn(
        "label_won",
        F.col("won").cast(BooleanType()),
    )

    # label_ctr: 1.0 if click, else 0.0
    df = df.withColumn(
        "label_ctr",
        F.when(F.col("click") == True, F.lit(1.0)).otherwise(F.lit(0.0)).cast(
            DoubleType()
        ),
    )

    # label_conversion_rate: 1.0 if conversion, else 0.0
    df = df.withColumn(
        "label_conversion_rate",
        F.when(F.col("conversion") == True, F.lit(1.0))
        .otherwise(F.lit(0.0))
        .cast(DoubleType()),
    )

    # label_roi: (conversion_value - price_paid) / price_paid
    # Only computed when won=True and price_paid > 0
    df = df.withColumn(
        "label_roi",
        F.when(
            (F.col("won") == True)
            & (F.col("price_paid").isNotNull())
            & (F.col("price_paid") > 0),
            (
                F.coalesce(F.col("conversion_value"), F.lit(0.0))
                - F.col("price_paid")
            )
            / F.col("price_paid"),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    return df


def strip_pii_columns(df: DataFrame) -> DataFrame:
    """Remove any raw PII columns that may exist in the source data.

    Only hashed identifiers (user_id_hash, site_domain_hash) are retained.
    Any column matching known PII patterns is dropped.

    Args:
        df: Input DataFrame potentially containing PII columns.

    Returns:
        DataFrame with PII columns removed.
    """
    columns_to_drop = [col for col in PII_COLUMNS_TO_DROP if col in df.columns]
    if columns_to_drop:
        logger.info("Dropping PII columns from output: %s", columns_to_drop)
        df = df.drop(*columns_to_drop)
    return df


def select_output_columns(df: DataFrame) -> DataFrame:
    """Select only the columns defined for output, in the canonical order.

    This ensures no unexpected columns (especially PII) leak into the output.

    Args:
        df: Input DataFrame after all transformations.

    Returns:
        DataFrame with only the defined output columns.
    """
    available_output_cols = [col for col in OUTPUT_COLUMNS if col in df.columns]
    return df.select(available_output_cols)


# ---------------------------------------------------------------------------
# Glue Job Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the AWS Glue ETL job.

    This function is called when the script runs in the Glue managed Spark
    environment. It reads job parameters, executes the ETL pipeline, and
    commits the job bookmark.
    """
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    # Parse job arguments
    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "start_timestamp",
            "end_timestamp",
            "output_path",
            "database_name",
            "table_name",
        ],
    )

    job_name = args["JOB_NAME"]
    start_timestamp = int(args["start_timestamp"])
    end_timestamp = int(args["end_timestamp"])
    output_path = args["output_path"]
    database_name = args.get("database_name", "feedback_pipeline")
    table_name = args.get("table_name", "raw_bid_outcomes")

    logger.info(
        "Starting feedback ETL job '%s': window=[%d, %d) millis, "
        "database=%s, table=%s, output=%s",
        job_name,
        start_timestamp,
        end_timestamp,
        database_name,
        table_name,
        output_path,
    )

    # Validate parameters
    if end_timestamp <= start_timestamp:
        raise ValueError(
            f"end_timestamp ({end_timestamp}) must be greater than "
            f"start_timestamp ({start_timestamp})"
        )

    # Initialize Glue/Spark context
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(job_name, args)

    try:
        # ------------------------------------------------------------------
        # Step 1: Read raw bid outcomes from the Glue catalog
        # ------------------------------------------------------------------
        logger.info("Reading from catalog: %s.%s", database_name, table_name)
        raw_df = spark.read.format("parquet").table(f"{database_name}.{table_name}")

        # ------------------------------------------------------------------
        # Step 2: Window by event_timestamp
        # ------------------------------------------------------------------
        windowed_df = filter_by_time_window(raw_df, start_timestamp, end_timestamp)
        record_count = windowed_df.count()
        logger.info("Records within time window: %d", record_count)

        if record_count == 0:
            logger.info("No records in window. Nothing to process.")
            job.commit()
            return

        # ------------------------------------------------------------------
        # Step 3: Separate bid outcomes from signal events and join
        # ------------------------------------------------------------------
        # Bid outcomes have pricing data (original_price > 0 indicates an actual bid)
        # Signal events are updates with impression/click/conversion without full bid data
        bid_outcomes = windowed_df.filter(
            F.col("original_price").isNotNull() & (F.col("original_price") > 0)
        )
        signal_events = windowed_df.filter(
            F.col("original_price").isNull() | (F.col("original_price") == 0)
        )

        bid_count = bid_outcomes.count()
        signal_count = signal_events.count()
        logger.info(
            "Bid outcomes: %d, Signal events: %d", bid_count, signal_count
        )

        # Join signals to bid outcomes by request_id
        joined_df = join_signals_to_outcomes(bid_outcomes, signal_events)

        # ------------------------------------------------------------------
        # Step 4: De-duplicate by request_id (keep latest event_timestamp)
        # ------------------------------------------------------------------
        deduped_df = deduplicate_by_request_id(joined_df)
        deduped_count = deduped_df.count()
        logger.info(
            "After de-duplication: %d records (%d duplicates removed)",
            deduped_count,
            bid_count - deduped_count,
        )

        # ------------------------------------------------------------------
        # Step 5: Compute training labels
        # ------------------------------------------------------------------
        labeled_df = compute_labels(deduped_df)

        # ------------------------------------------------------------------
        # Step 6: Strip raw PII and select output columns
        # ------------------------------------------------------------------
        safe_df = strip_pii_columns(labeled_df)
        output_df = select_output_columns(safe_df)

        # ------------------------------------------------------------------
        # Step 7: Write labeled output as Parquet
        # ------------------------------------------------------------------
        logger.info("Writing labeled dataset to: %s", output_path)
        output_df.write.mode("overwrite").parquet(output_path)

        final_count = output_df.count()
        logger.info("Wrote %d labeled records to output.", final_count)

    except Exception:
        logger.exception("Feedback ETL job failed")
        raise
    finally:
        job.commit()
        logger.info("Job committed.")


if __name__ == "__main__":
    main()
