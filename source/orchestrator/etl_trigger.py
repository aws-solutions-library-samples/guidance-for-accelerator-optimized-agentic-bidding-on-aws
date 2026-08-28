"""On-demand Glue ETL sweep trigger for load-test outcome data.

A load-test run only becomes selectable as a "Train from load test" input
once the Glue job that sweeps raw outcomes into training-data/ has completed
a run covering that run's timestamp (see loadtest_eligibility._is_trainable).
Those jobs are otherwise only started by a schedule -- every 6 hours, per
glue_etl_cfn.yaml's FeatureEngineeringSchedule/DealYieldFeatureEngineeringSchedule
(``cron(0 */6 * * ? *)``). Without an on-demand sweep, a run recorded just
after a scheduled sweep completes stays untrainable for nearly 6 hours, which
reads as a broken feature rather than a pipeline latency.

Timing constraint (why the sweep is delayed rather than immediate):
outcomes reach S3 through Kinesis Data Firehose, which buffers up to
``IntervalInSeconds`` before writing a Parquet object -- 300s in
feedback_pipeline_cfn.yaml's delivery streams. Starting a sweep the instant a
load test finishes would read S3 *before* that run's outcomes have landed. The
sweep would still succeed, and its CompletedOn would then satisfy the
trainability gate, marking the run trainable while its data is genuinely
absent from training-data/ -- producing exactly the empty-input training
failure the gate exists to prevent. So the sweep waits
ETL_SWEEP_DELAY_SECONDS (default 360 = the 300s buffer plus a 60s margin).

This module performs no I/O at import time and depends on no other
orchestrator module, so it is safe to import from both loadtest.py and
governance_api.py (which deliberately avoid importing each other eagerly).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

# How long to wait after a load test completes before starting a sweep, so
# Firehose has flushed that run's outcomes to S3. See the module docstring --
# a shorter delay risks a sweep that marks a run trainable without its data.
SWEEP_DELAY_SECONDS = int(os.environ.get("ETL_SWEEP_DELAY_SECONDS", "360"))

# Which Glue job labels model_type's training data -- matches deploy.sh's
# GLUE_JOB_NAME/DEAL_YIELD_GLUE_JOB_NAME env var naming convention
# (glue_etl_cfn.yaml's FeatureEngineeringJob/DealYieldFeatureEngineeringJob).
# Both deal_yield_manager_floor and deal_yield_manager_margin are labeled by
# the SAME Glue job (glue_deal_yield_feature_engineering.py writes both output
# prefixes in one run).
#
# This is the single source of truth for the mapping; governance_api.py imports
# it here rather than keeping a second copy, so a model type added in one place
# cannot silently resolve to "no Glue job configured" in the other.
GLUE_JOB_ENV_VAR_BY_MODEL_TYPE = {
    "dlrm_bid_shader": "GLUE_JOB_NAME",
    "deal_yield_manager_floor": "DEAL_YIELD_GLUE_JOB_NAME",
    "deal_yield_manager_margin": "DEAL_YIELD_GLUE_JOB_NAME",
}


def resolve_glue_job_name(model_type: str) -> str:
    """Returns the configured Glue job name that labels model_type's training
    data, or "" when the model type is unknown or its env var is unset."""
    env_var = GLUE_JOB_ENV_VAR_BY_MODEL_TYPE.get(model_type)
    if not env_var:
        return ""
    return os.environ.get(env_var, "")


def trigger_etl_sweep(model_type: str) -> str | None:
    """Start a Glue run that sweeps model_type's raw outcomes into
    training-data/. Returns the JobRunId when a run was started, else None.

    Never raises -- this runs on the load-test completion path, which must not
    fail because a best-effort sweep could not be started (mirrors the
    emit_bid_outcome fire-and-forget contract).

    A None return is a real "no sweep started", not a silent success:

    - unknown model type, or its job-name env var is unset
    - ConcurrentRunsExceededException: the Glue jobs are MaxConcurrentRuns: 1
      (glue_etl_cfn.yaml). Note this case does NOT guarantee coverage -- the
      in-flight sweep may have started before this run's outcomes landed in
      S3, in which case the run stays untrainable until the next sweep. Callers
      must not treat it as equivalent to a started run.
    - any other API/credential error, logged with the job name
    """
    job_name = resolve_glue_job_name(model_type)
    if not job_name:
        logger.warning(
            "[etl_trigger] no Glue job configured for model_type=%r "
            "(env var %r unset) — skipping on-demand sweep",
            model_type,
            GLUE_JOB_ENV_VAR_BY_MODEL_TYPE.get(model_type, "<unmapped>"),
        )
        return None

    try:
        import boto3

        glue = boto3.client("glue", region_name=_REGION)
        run_id = glue.start_job_run(JobName=job_name)["JobRunId"]
        logger.info(
            "[etl_trigger] started Glue sweep job=%s run=%s for model_type=%s",
            job_name, run_id, model_type,
        )
        return run_id
    except Exception as exc:  # noqa: BLE001 — fire-and-forget, never raises
        if type(exc).__name__ == "ConcurrentRunsExceededException":
            logger.info(
                "[etl_trigger] Glue job %s already running — not starting another. "
                "If that run began before this load test's outcomes reached S3, "
                "the run stays untrainable until the next sweep.",
                job_name,
            )
        else:
            logger.warning(
                "[etl_trigger] could not start Glue sweep job=%s for model_type=%s: %s",
                job_name, model_type, exc,
            )
        return None
