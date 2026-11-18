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
    --conversion_lag_hours: Optional (default: 0). How far in the past the rolling
                    window ends, so a bid is only processed once its downstream
                    signals have had time to arrive. The window filters on the BID's
                    timestamp, and a conversion arriving after that bid's window was
                    processed can never be joined — see _resolve_window. Set this to
                    the longest signal lag you expect. Cannot be combined with an
                    explicit --window_start/--window_end.
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
from pyspark.sql.types import IntegerType, DoubleType, StringType

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
# BidShadingOutcomeEvent.site_domain. Checking it against _is_plausible_hash would
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
    """De-duplicate records by request_id, keeping the most resolved version.

    When the same request_id appears multiple times (re-emitted with signal updates
    as impression/click/conversion arrive), keep one row: the highest timestamp
    (Unix seconds -- see shared/feedback_models.py's
    BidShadingOutcomeEvent.timestamp, the field this column is populated from), and
    among equal timestamps the row carrying the most reported outcome signals.
    """
    # Timestamp alone is not enough to order these rows. An enriched outcome event
    # KEEPS the timestamp of the bid it belongs to (see
    # SignalAssociator._enrich_event) -- it has to, because the processing window
    # filters on that column and a re-timestamped row could fall outside it. So the
    # bid-time row and its enriched replacement tie exactly, and row_number() picks
    # between them arbitrarily: roughly half of all labelled rows would be discarded
    # in favour of the unlabelled version of the same bid, silently.
    #
    # The tie-break is how many outcome signals the row actually carries. More
    # reported signals means a later, more complete version of the same bid, so the
    # enriched row wins every time.
    resolved_rank = sum(
        F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))
        for c in ("won", "impression", "click", "conversion")
        if c in df.columns
    )
    df = df.withColumn("_resolved_rank", resolved_rank)

    window = Window.partitionBy("request_id").orderBy(
        F.col("timestamp").desc(), F.col("_resolved_rank").desc()
    )
    return (
        df.withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num", "_resolved_rank")
    )


def engineer_features(
    df: DataFrame, attribution_deadline_epoch: float | None = None
) -> DataFrame:
    """Compute derived features for model training.

    Features added:
        - roi: (conversion_value - price_paid) / price_paid (null when price_paid is 0 or null)
        - shade_ratio: shaded_price / original_price
        - label: 1 if profitable win (won=True AND conversion_value > price_paid), 0 otherwise.
          Retained unchanged for continuity; see the note on response labels below.
        - label_conversion: 1 if the impression was won and converted, else 0; NULL when no
          conversion signal exists for the row
        - label_click: 1 if the impression was won and was clicked, else 0; NULL when no click
          signal exists for the row

    On labels. `label` is a PROFITABLE WIN -- it folds the response and the price paid into one
    bit, so a model trained on it learns "will this bid be cheap enough to be worth winning"
    rather than "how likely is the response". Expected value needs the second: `ev = p x
    conversion_value` only holds when p is the probability of the response the advertiser pays
    for. `label_conversion` and `label_click` are those responses, for a CPA and a CPC objective
    respectively, and the trainer selects one by name.

    NULL, not 0, when the signal is missing. A row with no conversion signal is unlabelled; a row
    with a conversion signal saying "no conversion" is a real negative. Encoding both as 0 makes
    an unlabelled dataset look like a dataset of negatives, which trains a model to predict zero
    and reports a healthy-looking loss while doing it. The trainer's dataset gate refuses a run
    whose label column is entirely NULL or entirely one class.
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

    # Response labels: the event the advertiser actually pays for.
    df = _add_response_label(
        df, "label_conversion", "conversion", attribution_deadline_epoch
    )
    df = _add_response_label(df, "label_click", "click", attribution_deadline_epoch)

    # Where the outcomes came from, carried through to the trainer.
    df = ensure_outcome_provenance(df)

    return df


def ensure_outcome_provenance(df: DataFrame) -> DataFrame:
    """Guarantee an ``outcome_provenance`` column on the training dataset.

    The column records whether each row's outcome was observed, produced by the
    outcome simulator, or never reported. The trainer reports the mix and writes it
    into the model manifest, so a model trained on simulated outcomes stays
    identifiable as one.

    A partition written before the column existed has no provenance to report, and
    gets ``"unresolved"`` rather than a guess. Calling this on a DataFrame that
    already has the column only fills NULLs; an existing label is never overwritten.
    """
    if "outcome_provenance" not in df.columns:
        return df.withColumn(
            "outcome_provenance", F.lit("unresolved").cast(StringType())
        )

    return df.withColumn(
        "outcome_provenance",
        F.when(
            F.col("outcome_provenance").isNull(), F.lit("unresolved")
        ).otherwise(F.col("outcome_provenance")).cast(StringType()),
    )


def _add_response_label(
    df: DataFrame,
    column: str,
    signal: str,
    attribution_deadline_epoch: float | None = None,
) -> DataFrame:
    """Add a response label from a boolean outcome signal.

    Three states, not two:

    * `won=False` is a real 0. A lost impression cannot have produced a response.
    * The signal reported True is a 1.
    * The signal absent is UNLABELLED -- NULL -- *until the attribution window for that row
      has closed*. After that, absence is evidence: a won impression that has had its full
      attribution window and reported no conversion did not convert, and that is a real 0.

    That third clause is what makes a dataset trainable. Downstream signals only ever report
    events that HAPPENED -- there is no "no click" message, from a real pixel or from the
    outcome simulator, because a non-event cannot be observed directly. Without an attribution
    deadline every labelled row is therefore a 1, the label has a single class, and the
    trainer's dataset gate refuses the run. Confirmed on a live dataset: 37 click labels and
    11 conversion labels, all of them 1, no zeros anywhere.

    The deadline is `row timestamp + conversion_lag_hours`, compared against the window end.
    `conversion_lag_hours` is the same knob that holds the processing window back (see
    _resolve_window), and it carries the same claim in both places: how long a response may
    take to arrive. Declaring 0 asserts responses are immediate, which makes every won row
    with no response an immediate negative. Passing no deadline keeps the previous
    behaviour -- NULL for every absent signal -- so a caller that has not opted in is
    unaffected.

    The signal column being absent from the schema entirely is a real state: the outcome table
    has grown over time and an older partition may predate a signal. The column is still
    emitted, all NULL, so the trainer's dataset gate reports "no labelled rows" rather than the
    trainer failing on a missing column and leaving the cause to be guessed.
    """
    if signal not in df.columns:
        return df.withColumn(column, F.lit(None).cast(IntegerType()))

    if attribution_deadline_epoch is None:
        absent = F.lit(None).cast(IntegerType())
    else:
        # Absent AND the window has closed AND the bid was won -> a real 0.
        # Absent on a row whose outcome is still unresolved stays NULL: nothing has
        # reported on that bid at all, so there is no impression for a response to
        # have followed.
        absent = (
            F.when(
                (F.col("won") == True)
                & (F.col("timestamp") <= F.lit(attribution_deadline_epoch)),
                F.lit(0),
            ).otherwise(F.lit(None).cast(IntegerType()))
        )

    return df.withColumn(
        column,
        F.when(F.col("won") == False, F.lit(0))
        .when(F.col(signal).isNull(), absent)
        .when(F.col(signal) == True, F.lit(1))
        .otherwise(F.lit(0))
        .cast(IntegerType()),
    )


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
    window_start: str | None,
    window_end: str | None,
    window_hours: str | None,
    conversion_lag_hours: str | None = None,
) -> tuple[datetime, datetime]:
    """Resolve the [start, end) processing window.

    If both window_start and window_end are provided (explicit, one-off run —
    e.g. a manual backfill), use them as-is. Otherwise self-compute a rolling
    window at execution time — this is the path the scheduled trigger uses, since
    CloudFormation can only pass a static duration, not a computed "now".

    On conversion lag. The window filters on ``timestamp``, which is the BID's
    timestamp: an enriched outcome event re-emitted when a conversion arrives keeps
    the timestamp of the bid it belongs to (see SignalAssociator._enrich_event), so
    the bid row and its enriched replacement always fall in the same window and the
    de-duplication step can pick the later one. That only works if the window is
    processed after the signals have arrived. A bid at time T whose conversion arrives
    at T+20h is enriched in S3 at T+20h, long after the window containing T was
    processed — and no later window contains T, so that conversion is never joined.

    ``conversion_lag_hours`` is the fix: the rolling window ends that far in the past,
    so a bid is only processed once its signals have had time to land. It defaults to
    0, which preserves the previous behaviour of processing right up to now, and logs
    a warning saying what that costs.
    """
    lag_hours = float(conversion_lag_hours) if conversion_lag_hours else 0.0
    if lag_hours < 0:
        raise ValueError(
            f"conversion_lag_hours must be >= 0, got {lag_hours}"
        )

    if window_start and window_end:
        if lag_hours:
            raise ValueError(
                "conversion_lag_hours cannot be combined with an explicit "
                "window_start/window_end: a backfill names the window it wants, and "
                "silently shifting it would process a different range than asked for. "
                "Subtract the lag from the explicit bounds instead."
            )
        start_dt = datetime.fromisoformat(window_start)
        end_dt = datetime.fromisoformat(window_end)
    elif window_start or window_end:
        raise ValueError(
            "window_start and window_end must be provided together, or not at all "
            f"(got window_start={window_start!r}, window_end={window_end!r})"
        )
    else:
        hours = float(window_hours) if window_hours else 6.0
        end_dt = datetime.now(timezone.utc) - timedelta(hours=lag_hours)
        start_dt = end_dt - timedelta(hours=hours)

        if lag_hours == 0.0:
            logger.warning(
                "conversion_lag_hours=0: this run processes bids up to the present "
                "moment, so any impression, click or conversion signal that arrives "
                "after its bid's window has been processed will never be joined and "
                "that row stays unlabelled. Set conversion_lag_hours to the longest "
                "lag you expect signals to have."
            )
        else:
            logger.info(
                "Rolling window lags real time by %.2fh to wait for downstream "
                "signals; processing [%s, %s)",
                lag_hours,
                start_dt.isoformat(),
                end_dt.isoformat(),
            )

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
        _optional_arg("window_start"),
        _optional_arg("window_end"),
        _optional_arg("window_hours"),
        _optional_arg("conversion_lag_hours"),
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
        # Window bounds as Unix seconds -- matches BidShadingOutcomeEvent.timestamp,
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

        # Read as Spark DataFrame for more control over filtering.
        # pathGlobFilter restricts the read to actual *.parquet objects --
        # without it, Spark tries to infer a schema from EVERY object under
        # table_location, including the zero-byte ".keep" marker
        # deploy_closed_loop.sh now writes so this prefix exists before
        # Firehose's first delivery (see that script's comment). A prefix
        # containing only that marker (no real data yet) fails outright
        # with "Unable to infer schema for Parquet" instead of reading 0
        # rows -- confirmed live on the deal-yield ETL job's equivalent
        # prefix, which had no real Parquet data yet.
        # mergeSchema is load-bearing, not an optimisation toggle. Without it Spark
        # infers the schema from a single arbitrary file, and this prefix accumulates
        # files written over time by different versions of the event: a partition
        # written before `outcome_provenance` / `day_of_week` / `geo_country` /
        # `has_video` existed has 20 columns, a current one has 24. Picking the older
        # file silently drops all four columns from every row in the run -- confirmed
        # live, where an output dataset had no provenance and no geo at all even
        # though the raw rows carried both. Merging unions the schemas and leaves the
        # absent values NULL, which `ensure_outcome_provenance` and
        # `_add_response_label` already handle.
        df = (
            spark.read.format("parquet")
            .option("pathGlobFilter", "*.parquet")
            .option("mergeSchema", "true")
            .load(table_location)
        )

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
        # A row whose attribution window closed before the end of this processing
        # window has had its full chance to report a response, so an absent response
        # is a real negative rather than an unlabelled row. See _add_response_label.
        lag_seconds = (
            float(_optional_arg("conversion_lag_hours") or 0.0) * 3600.0
        )
        attribution_deadline_epoch = window_end_epoch - lag_seconds
        logger.info(
            "Attribution deadline: rows with timestamp <= %.0f (window end %.0f minus "
            "%.0fs lag) label an absent response as 0 rather than NULL",
            attribution_deadline_epoch,
            window_end_epoch,
            lag_seconds,
        )
        df = engineer_features(df, attribution_deadline_epoch)
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
