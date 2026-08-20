"""AWS Glue ETL job for feature engineering and labeling of bid outcomes.

Reads raw bid outcome records from the Glue catalog (feedback_pipeline.raw_bid_outcomes),
de-duplicates by request_id (keeping the latest by event_timestamp), engineers training
features, and writes labeled Parquet datasets for model retraining.

Job Parameters:
    --window_start: Optional. ISO 8601 timestamp for the start of the processing
                    window (inclusive). Use for an explicit, one-off window (e.g.
                    manual backfill runs).
    --window_end:   Optional. ISO 8601 timestamp for the end of the processing
                    window (exclusive). Must be provided together with
                    --window_start, or not at all.
    --window_hours: Optional (default: 6). When --window_start/--window_end are
                    NOT both provided, the job self-computes a rolling window of
                    [now - window_hours, now) at execution time. This is what the
                    scheduled trigger (glue_etl_cfn.yaml's FeatureEngineeringSchedule)
                    uses — CloudFormation cannot compute a relative "now" at
                    template-render time, only a static duration, so the window
                    itself must be computed here, at run time, not in the trigger.
    --output_bucket: S3 bucket for labeled training output
    --database_name: Glue catalog database name (default: feedback_pipeline)
    --table_name:    Glue catalog table name (default: raw_bid_outcomes)

Requirements: 2.2, 2.3, 2.4, 2.6
"""

import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import boto3
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

# Columns that should only contain hashed values (hex strings of sufficient
# length). site_domain is intentionally excluded: it's a literal domain name
# (e.g. "espn.com"), not a hash -- see shared/feedback_models.py's
# BidOutcomeEvent.site_domain. Checking it against _is_plausible_hash would
# misclassify every real record as suspected PII and drop it.
_HASH_COLUMNS = ("user_id_hash",)
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
    """De-duplicate records by request_id, keeping the latest by timestamp.

    When the same request_id appears multiple times (e.g., re-emitted with
    signal updates like impression/click/conversion arriving later), we keep
    only the record with the highest timestamp (Unix seconds -- see
    shared/feedback_models.py's BidOutcomeEvent.timestamp, the field this
    column is actually populated from).
    """
    window = Window.partitionBy("request_id").orderBy(F.col("timestamp").desc())
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


def _resolve_window(
    window_start: str | None, window_end: str | None, window_hours: str | None
) -> tuple[datetime, datetime]:
    """Resolve the [start, end) processing window.

    If both window_start and window_end are provided (explicit, one-off run —
    e.g. a manual backfill), use them as-is. Otherwise self-compute a rolling
    window [now - window_hours, now) at execution time — this is the path the
    scheduled trigger uses, since CloudFormation can only pass a static
    window_hours duration, not a computed "now" or "N hours ago" timestamp.
    """
    if window_start and window_end:
        start_dt = datetime.fromisoformat(window_start)
        end_dt = datetime.fromisoformat(window_end)
    elif window_start or window_end:
        raise ValueError(
            "window_start and window_end must be provided together, or not at all "
            f"(got window_start={window_start!r}, window_end={window_end!r})"
        )
    else:
        hours = float(window_hours) if window_hours else 6.0
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=hours)

    if end_dt <= start_dt:
        raise ValueError(
            f"window_end ({end_dt.isoformat()}) must be after "
            f"window_start ({start_dt.isoformat()})"
        )
    return start_dt, end_dt


def main():
    """Entry point for the AWS Glue ETL job."""
    # Import Glue-specific modules only at runtime (not needed for unit tests)
    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    # Only JOB_NAME and output_bucket are required; window_start/window_end/
    # window_hours/database_name/table_name are all optional (see module
    # docstring and _resolve_window()). getResolvedOptions treats every name
    # in this list as required, so optional args are parsed separately below.
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "output_bucket"])

    def _optional_arg(name: str) -> str | None:
        flag = f"--{name}"
        if flag in sys.argv:
            return sys.argv[sys.argv.index(flag) + 1]
        return None

    job_name = args["JOB_NAME"]
    output_bucket = args["output_bucket"]
    database_name = _optional_arg("database_name") or "feedback_pipeline"
    table_name = _optional_arg("table_name") or "raw_bid_outcomes"

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

    # Initialize Glue context
    sc = SparkContext()
    glue_context = GlueContext(sc)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(job_name, args)

    try:
        # ------------------------------------------------------------------
        # Step 1: Read directly from the table's S3 location
        # ------------------------------------------------------------------
        # Window bounds as Unix seconds -- matches BidOutcomeEvent.timestamp,
        # the real column this table's "timestamp" field is populated from
        # (a prior version of this filter used a non-existent
        # "event_timestamp" millis column, which was always null).
        window_start_epoch = start_dt.timestamp()
        window_end_epoch = end_dt.timestamp()

        # Resolve the table's S3 location via the Glue API and read Parquet
        # directly from it, rather than through spark.read.table(). The
        # latter only returns rows for partitions actually REGISTERED in the
        # Glue catalog (partition_date/partition_hour) -- nothing in this
        # pipeline ever registers those partitions (no crawler, no MSCK
        # REPAIR, no BatchCreatePartition call), so spark.read.table() always
        # silently returned 0 rows regardless of how much real data existed
        # in S3 (confirmed live). Reading the S3 location directly sidesteps
        # partition metadata entirely; this script already does its own
        # timestamp-based windowing below, so partition pruning was never
        # required for correctness -- only for read efficiency, which an
        # unpartitioned read still gets from Parquet's own predicate pushdown
        # on the timestamp column.
        boto3_glue = boto3.client("glue", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        table_location = boto3_glue.get_table(
            DatabaseName=database_name, Name=table_name
        )["Table"]["StorageDescriptor"]["Location"]

        logger.info("Reading from S3 location: %s (table=%s.%s)", table_location, database_name, table_name)

        # Read as Spark DataFrame for more control over filtering
        df = spark.read.format("parquet").load(table_location)

        # Filter by timestamp window
        df = df.filter(
            (F.col("timestamp") >= window_start_epoch)
            & (F.col("timestamp") < window_end_epoch)
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
