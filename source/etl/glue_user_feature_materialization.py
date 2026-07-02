"""AWS Glue ETL job for user feature materialization into DynamoDB.

Reads bid outcome records from the Glue catalog, aggregates per-user features
over a rolling 7-day window, and writes the feature vectors to the DynamoDB
user-features table for real-time serving in the bid path.

This is the materialization path from the offline feature store (S3/Glue catalog)
to the online feature store (DynamoDB). Runs on a schedule (e.g., every 4 hours)
to keep features fresh.

Job Parameters:
    --user_feature_table: DynamoDB table name for user features (default: user-features)
    --window_days: Rolling window for feature aggregation (default: 7)
    --database_name: Glue catalog database name (default: feedback_pipeline)
    --table_name: Glue catalog table name (default: raw_bid_outcomes)
    --region: AWS region (default: us-east-1)

Requirements: 2.5, 8.4, 9.4
"""

import logging
import sys
import time
from datetime import datetime, timedelta, timezone

import boto3
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("glue_user_feature_materialization")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_TTL_DAYS = 14  # Features auto-expire from DynamoDB after this many days
BATCH_SIZE = 25  # DynamoDB batch_write_item max


def get_job_parameters() -> dict:
    """Parse Glue job parameters."""
    from awsglue.utils import getResolvedOptions

    args = getResolvedOptions(sys.argv, [
        "JOB_NAME",
        "user_feature_table",
        "window_days",
        "database_name",
        "table_name",
        "region",
    ])
    return {
        "job_name": args["JOB_NAME"],
        "user_feature_table": args.get("user_feature_table", "user-features"),
        "window_days": int(args.get("window_days", "7")),
        "database_name": args.get("database_name", "feedback_pipeline"),
        "table_name": args.get("table_name", "raw_bid_outcomes"),
        "region": args.get("region", "us-east-1"),
    }


def load_bid_outcomes(spark: SparkSession, params: dict) -> DataFrame:
    """Load bid outcome records from the Glue catalog within the time window."""
    now = datetime.now(tz=timezone.utc)
    window_start = now - timedelta(days=params["window_days"])

    df = spark.read.format("parquet").table(
        f"{params['database_name']}.{params['table_name']}"
    )

    # Filter to the rolling window
    df = df.filter(F.col("event_timestamp") >= window_start.isoformat())
    logger.info("Loaded %d records from %s.%s (window: %d days)",
                df.count(), params["database_name"], params["table_name"], params["window_days"])
    return df


def compute_user_features(df: DataFrame) -> DataFrame:
    """Aggregate per-user features from bid outcome records.

    Output columns:
        user_id: hashed user identifier
        bid_count_7d: total bids in the window
        win_rate_7d: wins / total bids
        avg_ctr_7d: average click-through rate
        avg_spend_7d: average price paid on wins
        top_domain_hash: most-frequently-bid domain (mode)
        device_preference: most-used device type
    """
    # Aggregate per user
    user_agg = df.groupBy("user_id").agg(
        F.count("*").alias("bid_count_7d"),
        F.avg(F.col("won").cast("double")).alias("win_rate_7d"),
        F.avg(F.col("click").cast("double")).alias("avg_ctr_7d"),
        F.avg(
            F.when(F.col("won") == True, F.col("price_paid")).otherwise(None)
        ).alias("avg_spend_7d"),
    )

    # Top domain per user (mode)
    domain_window = Window.partitionBy("user_id").orderBy(F.desc("domain_count"))
    domain_mode = (
        df.groupBy("user_id", "domain_hash")
        .agg(F.count("*").alias("domain_count"))
        .withColumn("rank", F.row_number().over(domain_window))
        .filter(F.col("rank") == 1)
        .select("user_id", F.col("domain_hash").alias("top_domain_hash"))
    )

    # Device preference per user (mode)
    device_window = Window.partitionBy("user_id").orderBy(F.desc("device_count"))
    device_mode = (
        df.groupBy("user_id", "device_type")
        .agg(F.count("*").alias("device_count"))
        .withColumn("rank", F.row_number().over(device_window))
        .filter(F.col("rank") == 1)
        .select("user_id", F.col("device_type").alias("device_preference"))
    )

    # Join all features
    features = (
        user_agg
        .join(domain_mode, "user_id", "left")
        .join(device_mode, "user_id", "left")
    )

    # Fill nulls
    features = features.fillna({
        "win_rate_7d": 0.0,
        "avg_ctr_7d": 0.0,
        "avg_spend_7d": 0.0,
        "top_domain_hash": "unknown",
        "device_preference": "unknown",
    })

    logger.info("Computed features for %d users", features.count())
    return features


def write_to_dynamodb(features_df: DataFrame, table_name: str, region: str) -> int:
    """Write user feature vectors to DynamoDB in batches.

    Uses batch_write_item for throughput. Each item includes a TTL attribute
    (feature_expiry_epoch) so stale features auto-expire.

    Returns:
        Number of items written.
    """
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    now_epoch = int(time.time())
    expiry_epoch = now_epoch + (FEATURE_TTL_DAYS * 86400)

    # Collect to driver — for production, use foreachPartition for scalability
    rows = features_df.collect()
    written = 0

    with table.batch_writer() as batch:
        for row in rows:
            item = {
                "user_id": str(row["user_id"]),
                "bid_count_7d": int(row["bid_count_7d"]),
                "win_rate_7d": str(round(row["win_rate_7d"], 4)),
                "avg_ctr_7d": str(round(row["avg_ctr_7d"], 4)),
                "avg_spend_7d": str(round(row["avg_spend_7d"] or 0.0, 4)),
                "top_domain_hash": str(row["top_domain_hash"]),
                "device_preference": str(row["device_preference"]),
                "last_updated": str(now_epoch),
                "feature_expiry_epoch": expiry_epoch,
            }
            batch.put_item(Item=item)
            written += 1

    logger.info("Wrote %d user feature items to %s", written, table_name)
    return written


def main():
    """Glue job entrypoint."""
    params = get_job_parameters()
    logger.info("User Feature Materialization starting: %s", params)

    spark = SparkSession.builder.appName(params["job_name"]).getOrCreate()

    # Load bid outcomes from catalog
    df = load_bid_outcomes(spark, params)

    if df.count() == 0:
        logger.warning("No bid outcome records in window — nothing to materialize")
        return

    # Compute aggregated user features
    features = compute_user_features(df)

    # Write to DynamoDB online feature store
    n_written = write_to_dynamodb(features, params["user_feature_table"], params["region"])

    logger.info("Materialization complete: %d users updated", n_written)
    spark.stop()


if __name__ == "__main__":
    main()
