"""Idempotently seed the closed-loop bidding parameters in the DynamoDB
Parameter Store.

The Adaptive Bidding agent only READS/UPDATES parameters — it never creates them.
In the demo, the orchestrator's ``/api/v1/closed-loop/generate`` endpoint seeds the
table (via ``ensure_parameters_initialized``). But the production/scheduled path —
EventBridge -> InvokeAgentRuntime — never calls ``/generate``, so without a one-time
seed the agent would read an empty table and have nothing to act on.

This script performs that one-time, idempotent seed at deploy time. It uses the same
initial values and per-update deltas as the orchestrator's initializer, and relies on
``ParameterStore.initialize_parameter``'s conditional put — an already-existing
parameter is left untouched (OptimisticLockError is swallowed), so it is safe to run
on every deploy.

Usage:
    python3 scripts/init_parameter_store.py \
        --table-name dv2-parameter-store \
        --region us-east-1 \
        --model-types dlrm_bid_shader
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Make source/ importable so we can reuse the real ParameterStore (no logic fork).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source"))

from shared.parameter_store import ParameterStore, OptimisticLockError  # noqa: E402

_LOG = logging.getLogger("init_parameter_store")

# Must match closed_loop_demo.invoker._INIT_DEFAULTS so the deploy-time seed and the
# orchestrator's runtime seed agree on initial values and per-update delta caps.
_INIT_DEFAULTS = {
    "shade_factor": {"initial_value": 0.65, "max_delta_per_update": 0.05},
    "conversion_value": {"initial_value": 10.0, "max_delta_per_update": 2.5},
}


async def _seed(table_name: str, region: str, model_types: list[str]) -> int:
    store = ParameterStore(table_name=table_name, region=region)
    created = 0
    for model_type in model_types:
        for name, cfg in _INIT_DEFAULTS.items():
            try:
                await store.initialize_parameter(
                    model_type=model_type,
                    parameter_name=name,
                    initial_value=cfg["initial_value"],
                    max_delta_per_update=cfg["max_delta_per_update"],
                )
                created += 1
                _LOG.info("  initialized %s/%s", model_type, name)
            except OptimisticLockError:
                _LOG.info("  exists (unchanged) %s/%s", model_type, name)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-name", required=True, help="Parameter store DynamoDB table name")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument(
        "--model-types",
        default="dlrm_bid_shader",
        help="Comma-separated ARTF model types to seed (default: dlrm_bid_shader — the type the agent reads).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    _LOG.info("Seeding parameter store %s (region=%s) for: %s", args.table_name, args.region, model_types)
    created = asyncio.run(_seed(args.table_name, args.region, model_types))
    _LOG.info("Parameter store seed complete (%d newly created).", created)
    return 0


if __name__ == "__main__":
    sys.exit(main())
