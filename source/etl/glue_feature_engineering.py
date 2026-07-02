"""AWS Glue ETL job for feature engineering and labeling of bid outcomes.

Reads raw bid outcome records from the Glue catalog (feedback_pipeline.raw_bid_outcomes),
de-duplicates by request_id (keeping the latest by event_timestamp), engineers training
features, and writes labeled Parquet datasets for model retraining.

Job Parameters:
    --window_start: ISO 8601 timestamp for the start of the processing window (inclusive)
    --window_end:   ISO 8601 timestamp for the end of the processing window (exclusive)
    --output_bucket: S3 bucket for labeled training output
    --database_name: Glue catalog database name (default: feedback_pipeline)
    --table_name:    Glue catalog table name (default: raw_bid_outcomes)

Requirements: 2.2, 2.3, 2.4, 2.6
"""

import logging
import re
import sys
from datetime import datetime

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("glue_feature_engineering")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# PII detection patterns — flag records containing raw PII
# ---------------------------------------------------------------------------
# Matches common email patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Matches potential raw user IDs that are NOT hex hashes (hashes are 32+ hex chars)
_RAW_ID_RE = re.compile(r"^(?![\da-fA-F]{32,}$).+$")

# Columns that should only contain hashed values (hex strings of sufficient length)
_HASH_COLUMNS = ("user_id_hash", "site_domain_hash")
_MIN_HASH_LENGTH = 8  # Minimum length for a value to be considered a plausible hash


def _is_plausible_hash(value: str) -> bool:
    """Check if a string value looks like a hashed identifier.

    A plausible hash is at least _MIN_HASH_LENGTH characters and consists
    entirely of hexadecimal characters, or is a common hash format (sha256, md5, etc.).
    """
    if not value or len(value) < _MIN_HASH_LENGTH:
        return False
    # Accept hex strings (md5=32, sha1=40, sha256=64, etc.)
    if re.fullmatch(r"[0-9a-fA-F]+", value):
        return True
    # Accept base64-like patterns (url-safe base64)
    if re.fullmatch(r"[A-Za-z0-9+/=_\-]+", value) and len(value) >= 16:
        return True
    return False


def _contains_pii_pattern(value: str) -> bool:
    """Check if a string contains patterns suggesting raw PII."""
    if not value:
        return False
    if _EMAIL_RE.search(value):
        return True
    return False


# ---------------------------------------------------------------------------
# Core ETL functions (exported for unit testing)
# ---------------------------------------------------------------------------


def deduplicate_by_request_id(df: DataFrame) -> DataFrame:
    """De-duplicate records by request_id, keeping the latest by event_timestamp.

    When the same request_id appears multiple times (e.g., re-emitted with
    signal updates like impression/click/conversion arriving later), we keep
    only the record with the highest event_timestamp.
    """
    window = Window.partitionBy("request_id").orderBy(F.col("event_timestamp").desc())
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


def engineer_features(df: DataFrame) -> DataFrame:
    """Compute derived features for model training.

    Features added:
        - roi: (conversion_value - price_paid) / price_paid (null when price_paid is 0 or null)
        - shade_ratio: shaded_price / original_price
        - label: 1 if profitable win (won=True AND conversion_value > price_paid), 0 otherwise
    """
    # ROI: only meaningful for records with a price_paid > 0 and conversion_value present
    df = df.withColumn(
        "roi",
        F.when(
            (F.col("price_paid").isNotNull()) & (F.col("price_paid") > 0) & (F.col("conversion_value").isNotNull()),
            (F.col("conversion_value") - F.col("price_paid")) / F.col("price_paid"),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # Shade ratio: shaded_price / original_price (original_price should always be > 0)
    df = df.withColumn(
        "shade_ratio",
        F.when(
            F.col("original_price") > 0,
            F.col("shaded_price") / F.col("original_price"),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # Label: 1 if profitable win (won AND revenue > cost)
    # Revenue = conversion_value (when available), Cost = price_paid
    df = df.withColumn(
        "label",
        F.when(
            (F.col("won") == True)
            & (F.col("conversion_value").isNotNull())
            & (F.col("price_paid").isNotNull())
            & (F.col("conversion_value") > F.col("price_paid")),
            F.lit(1),
        ).otherwise(F.lit(0)).cast(IntegerType()),
    )

    return df


def compute_win_rate_buckets(df: DataFrame) -> DataFrame:
    """Compute win_rate_bucket: binned win rate by (hour_of_day, device_type).

    The win rate is computed as the fraction of records where won=True within
    each (hour_of_day, device_type) group, then binned into discrete buckets:
        0: [0.0, 0.1)
        1: [0.1, 0.2)
        ...
        9: [0.9, 1.0]
    """
    # Compute win rate per (hour_of_day, device_type) group
    win_rate_window = Window.partitionBy("hour_of_day", "device_type")
    df = df.withColumn(
        "_group_win_rate",
        F.avg(F.col("won").cast(IntegerType())).over(win_rate_window),
    )

    # Bin into 10 buckets (0-9)
    df = df.withColumn(
        "win_rate_bucket",
        F.when(F.col("_group_win_rate") >= 1.0, F.lit(9))
        .otherwise(F.floor(F.col("_group_win_rate") * 10).cast(IntegerType())),
    )

    return df.drop("_group_win_rate")


def validate_no_raw_pii(df: DataFrame) -> DataFrame:
    """Validate that hash columns contain only hashed identifiers.

    Drops any record where user_id_hash or site_domain_hash appears to
    contain raw PII (email addresses or non-hash strings). Logs warnings
    for dropped records.

    Returns the filtered DataFrame (records with PII removed).
    """
    # UDF to check if a value is a plausible hash (not raw PII)
    @F.udf("boolean")
    def is_valid_hash(value):
        if value is None:
            return True
        if _contains_pii_pattern(value):
            return False
        return _is_plausible_hash(value)

    # Apply checks to each hash column
    valid_mask = F.lit(True)
    for col_name in _HASH_COLUMNS:
        valid_mask = valid_mask & is_valid_hash(F.col(col_name))

    # Count and log dropped records
    total_count = df.count()
    valid_df = df.filter(valid_mask)
    valid_count = valid_df.count()
    dropped_count = total_count - valid_count

    if dropped_count > 0:
        logger.warning(
            "Dropped %d records (%.2f%%) containing raw PII patterns in hash columns",
            dropped_count,
            (dropped_count / total_count * 100) if total_count > 0 else 0,
        )

    return valid_df


# ---------------------------------------------------------------------------
# Glue Job Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the AWS Glue ETL job."""
    # Import Glue-specific modules only at runtime (not needed for unit tests)
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    # Parse job arguments
    args = getResolvedOptions(
        sys.argv,
        [
            "JOB_NAME",
            "window_start",
            "window_end",
            "output_bucket",
            "database_name",
            "table_name",
        ],
    )

    job_name = args["JOB_NAME"]
    window_start = args["window_start"]
    window_end = args["window_end"]
    output_bucket = args["output_bucket"]
    database_name = args.get("database_name", "feedback_pipeline")
    table_name = args.get("table_name", "raw_bid_outcomes")

    logger.info(
        "Starting ETL job %s: window=[%s, %s), database=%s, table=%s",
        job_name,
        window_start,
        window_end,
        database_name,
        table_name,
    )

    # Parse timestamps for partition filtering
    start_dt = datetime.fromisoformat(window_start)
    end_dt = datetime.fromisoformat(window_end)

    # Validate window
    if end_dt <= start_dt:
        raise ValueError(
            f"window_end ({window_end}) must be after window_start ({window_start})"
        )

    # Initialize Glue context
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(job_name, args)

    try:
        # ------------------------------------------------------------------
        # Step 1: Read from Glue catalog with push-down predicate
        # ------------------------------------------------------------------
        # Convert window to event_timestamp millis for filtering
        window_start_millis = int(start_dt.timestamp() * 1000)
        window_end_millis = int(end_dt.timestamp() * 1000)

        logger.info("Reading from catalog: %s.%s", database_name, table_name)

        # Read as Spark DataFrame for more control over filtering
        df = spark.read.format("parquet").table(f"{database_name}.{table_name}")

        # Filter by event_timestamp window
        df = df.filter(
            (F.col("event_timestamp") >= window_start_millis)
            & (F.col("event_timestamp") < window_end_millis)
        )

        record_count = df.count()
        logger.info("Read %d records within time window", record_count)

        if record_count == 0:
            logger.info("No records in window — nothing to process.")
            job.commit()
            return

        # ------------------------------------------------------------------
        # Step 2: Validate PII — drop records with raw PII
        # ------------------------------------------------------------------
        df = validate_no_raw_pii(df)

        # ------------------------------------------------------------------
        # Step 3: De-duplicate by request_id (keep latest event_timestamp)
        # ------------------------------------------------------------------
        df = deduplicate_by_request_id(df)
        deduped_count = df.count()
        logger.info(
            "After de-duplication: %d records (%d duplicates removed)",
            deduped_count,
            record_count - deduped_count,
        )

        # ------------------------------------------------------------------
        # Step 4: Feature engineering
        # ------------------------------------------------------------------
        df = engineer_features(df)
        df = compute_win_rate_buckets(df)

        # ------------------------------------------------------------------
        # Step 5: Write labeled output to S3
        # ------------------------------------------------------------------
        output_path = (
            f"s3://{output_bucket}/training-data/"
            f"window_start={window_start}/window_end={window_end}/"
        )
        logger.info("Writing labeled dataset to %s", output_path)

        df.write.mode("overwrite").parquet(output_path)

        final_count = df.count()
        logger.info("Wrote %d labeled records to output", final_count)

    except Exception:
        logger.exception("ETL job failed")
        raise
    finally:
        job.commit()
        logger.info("Job committed.")


if __name__ == "__main__":
    main()
