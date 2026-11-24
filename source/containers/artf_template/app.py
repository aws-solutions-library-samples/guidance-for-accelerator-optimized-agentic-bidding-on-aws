"""ARTF Template Container: the starting point for your own ARTF container.

This container is deployed and wired to the orchestrator but ships inactive and
returns no mutations. The gRPC/MCP/health server, the orchestrator registration,
the health panel row and the activation switch are already done; ``mutate()``
below is the only thing to write. README.md in this directory covers the
rebuild, restart and activation steps, the registry record that routes requests
here, and the status vocabulary.

The orchestrator calls ``mutate(RTBRequest) -> RTBResponse`` and reads
``mutations`` and ``metadata.model_version`` off the response. Returning zero
mutations is a legitimate answer, reported as ``no_mutations``.

Where to look:

- ``shared/artf_types.py``: ``Mutation``, ``Intent``, ``Operation`` and the
  payload types (``IDsPayload``, ``AdjustBidPayload``, ``AdjustDealPayload``,
  ``AddMetricsPayload``).
- ``shared/artf_applier.py``: the paths and payloads the appliers accept per
  intent. A mutation outside that table is rejected with a reason.
- ``containers/metrics_enricher/app.py``: the simplest real container,
  rule-based, no model.
- ``containers/dlrm_bid_shader/app.py``: a container that calls a model on
  Triton and reads runtime parameters from DynamoDB.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (  # noqa: E402
    Intent,
    Metadata,
    Mutation,          # noqa: F401  (unused until you build one; see below)
    Operation,         # noqa: F401
    RTBRequest,
    RTBResponse,
    intent_applicable,
)

logger = logging.getLogger("artf.template")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ADD_CIDS is the default because no other container in this repo implements
# it. Override with ARTF_TEMPLATE_INTENT, and update the registry record's
# `intents` to match (see README.md).
_DEFAULT_INTENT = "ADD_CIDS"
_INTENT_NAME = os.environ.get("ARTF_TEMPLATE_INTENT", _DEFAULT_INTENT).strip().upper()

try:
    INTENT: Intent = Intent[_INTENT_NAME]
except KeyError:
    # An unknown name is reported and then ignored in favour of the default.
    # Crashing would take the container out of the cluster for a typo in an env
    # var, and silently serving intent 0 (UNSPECIFIED) would be worse.
    logger.error(
        "ARTF_TEMPLATE_INTENT=%r is not a valid ARTF intent; falling back to %s. "
        "Valid names: %s",
        _INTENT_NAME, _DEFAULT_INTENT, ", ".join(i.name for i in Intent),
    )
    INTENT = Intent[_DEFAULT_INTENT]

# Reported in every bid response. Change it when you implement mutate() so
# responses identify your version rather than the unimplemented template.
MODEL_VERSION = "artf-template-unimplemented-v1"


# ---------------------------------------------------------------------------
# The mutation handler
# ---------------------------------------------------------------------------

def mutate(req: RTBRequest) -> RTBResponse:
    """Produce ARTF mutations for one bid request.

    ``req.bid_request`` is the OpenRTB bid request; ``req.bid_response`` is the
    bid response and is present only on the DSP-response lifecycle (it is None
    for a publisher bid request, so guard before reading it).
    """
    # The orchestrator already filters on the registry record's intents, so this
    # guard is for direct callers that bypass it. Every container in this repo
    # has it; keep it.
    if not intent_applicable(INTENT, req.applicable_intents):
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    # =======================================================================
    # >>> IMPLEMENT YOUR LOGIC HERE <<<
    # =======================================================================
    #
    # Read what you need off req.bid_request, then build one Mutation per change.
    # The path and payload field depend on the intent (shared/artf_applier.py):
    #
    #     ACTIVATE_SEGMENTS               /user/data/segment      ids=IDsPayload(id=[...])
    #     ACTIVATE_DEALS / SUPPRESS_DEALS /imp/{imp}              ids=IDsPayload(id=[...])
    #     ADJUST_DEAL_FLOOR / _MARGIN     /imp/{imp}/deals/{deal} adjust_deal=AdjustDealPayload(...)
    #     ADD_METRICS / ADD_CIDS          /imp/{imp}/metric       add_metrics=AddMetricsPayload(metric=[...])
    #     BID_SHADE                       /seatbid/{seat}/bid/{bid} adjust_bid=AdjustBidPayload(price=...)
    #
    # A worked example for ADD_CIDS, publishing content identifiers on each
    # impression as metrics with value 1.0 (the appliers accept ADD_CIDS only on
    # /imp/{imp}/metric with an add_metrics payload):
    #
    #     from shared.artf_types import AddMetricsPayload, Metric
    #
    #     mutations = []
    #     for imp in req.bid_request.get("imp", []):
    #         content_ids = your_lookup(req.bid_request, imp)   # your logic
    #         if not content_ids:
    #             continue          # nothing to say about this impression
    #         mutations.append(Mutation(
    #             intent=INTENT,
    #             op=Operation.ADD,
    #             path=f"/imp/{imp.get('id', '')}/metric",
    #             add_metrics=AddMetricsPayload(metric=[
    #                 Metric(type=f"cid:{cid}", value=1.0, vendor="your-vendor")
    #                 for cid in content_ids
    #             ]),
    #         ))
    #     return RTBResponse(
    #         id=req.id,
    #         mutations=mutations,
    #         metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION),
    #     )
    #
    # Calling a model on Triton? Copy the pattern in containers/dlrm_bid_shader:
    # a triton_inference.py beside this file, gated on USE_TRITON, reading
    # TRITON_URL, with the model registered in orchestrator/app.py's
    # `container_to_model` so the health panel can probe it.
    #
    # This runs on the live bid path inside the request's `tmax` budget (100ms
    # by default, shared with the other containers, which run concurrently).
    # Exceed it and the orchestrator reports `timeout` and drops your mutations.
    # If you cannot compute a value, return fewer mutations rather than a guess.
    # =======================================================================

    # Shipped behaviour: no mutations, reported as `no_mutations`.
    return RTBResponse(
        id=req.id,
        mutations=[],
        metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION),
    )


if __name__ == "__main__":
    from shared.server import run_artf_server

    logging.basicConfig(level=logging.INFO)
    logger.info(
        "ARTF template container starting: intent=%s model_version=%s",
        INTENT.name, MODEL_VERSION,
    )
    run_artf_server(
        mutate,
        agent_name="artf-template",
        grpc_port=50051,
        mcp_port=8081,
        health_port=8080,
    )
