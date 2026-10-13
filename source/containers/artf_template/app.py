"""ARTF Template Container — the starting point for your own ARTF container.

This container is deployed and wired to the orchestrator but ships **inactive**
and returns **no mutations**. It is a working skeleton, not a stub: everything
around ``mutate()`` — the gRPC/MCP/health server, the orchestrator registration,
the health panel row, the activation switch — is already done. The only thing
missing is your logic.

===============================================================================
FOUR STEPS TO MAKE THIS YOURS
===============================================================================

1. **Write your logic** in ``mutate()`` below, at the marker. Return a
   ``RTBResponse`` carrying the ``Mutation`` objects you want applied.

2. **Rebuild just this image** (not the whole stack)::

       cd deployment
       ./deploy.sh --start-at 2          # image build + push

   or build it directly::

       cd source
       docker build --build-arg CONTAINER=containers/artf_template \
         -t <registry>/<stack>-artf-template:<tag> .
       docker push <registry>/<stack>-artf-template:<tag>

3. **Restart just this Deployment**. This step is not optional: ``deploy.sh``
   reuses the previous image tag when the registry already has it, so
   ``kubectl apply`` is a no-op when only the image *content* changed and the
   old pod keeps running your old code::

       kubectl rollout restart deployment/artf-template
       kubectl rollout status  deployment/artf-template

4. **Activate it in the UI**: open the Container Health panel and switch this
   container on. That writes to the registry table; no redeploy, and every
   orchestrator replica picks it up within the registry cache TTL (30s by
   default). Switch it off the same way — an inactive container is never called
   and reports ``disabled``, which is distinct from a failure.

===============================================================================
WHAT THE ORCHESTRATOR EXPECTS FROM YOU
===============================================================================

``mutate(RTBRequest) -> RTBResponse``, and nothing else. The orchestrator calls
``POST /mutate`` on port 8081 with the RTBRequest as JSON, and reads
``mutations`` and ``metadata.model_version`` off your response.

Returning zero mutations is a legitimate answer and is reported as
``no_mutations``, which is deliberately distinguishable from ``unreachable``
(nothing answered) and ``error`` (answered unusably). You do not need to fake a
mutation to look healthy.

Your intent is declared in **two** places and they should agree:

- The registry record's ``intents`` — what the orchestrator filters on. This is
  authoritative: if a request's ``applicable_intents`` does not overlap it, you
  are never called at all.
- ``ARTF_TEMPLATE_INTENT`` on this container — what the guard below checks. It
  matters when something calls this container directly, bypassing the
  orchestrator.

Changing the intent means updating the registry record. Changing only the env
var will make the orchestrator keep filtering on the old one.

Two containers may claim the same intent; both are called and both sets of
mutations are merged, which is a legitimate way to compare an implementation
against a built-in. The UI flags a shared intent because merge order then
decides which value survives downstream.

===============================================================================
WHERE TO LOOK
===============================================================================

- ``shared/artf_types.py``  — ``Mutation``, ``Intent``, ``Operation`` and every
  payload type (``IDsPayload``, ``AdjustBidPayload``, ``AdjustDealPayload``,
  ``AddMetricsPayload``). Start here.
- ``containers/metrics_enricher/app.py`` — the simplest real container.
  Rule-based, no model, ~100 lines. The closest thing to this file that does
  real work.
- ``containers/dlrm_bid_shader/app.py`` — a real container that calls a model on
  Triton, reads runtime parameters from DynamoDB, and reports the served model
  version per request.
- ``shared/server.py`` — the server this container runs on. You should not need
  to change it.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (  # noqa: E402
    Intent,
    Metadata,
    Mutation,          # noqa: F401  (unused until you build one — see below)
    Operation,         # noqa: F401
    RTBRequest,
    RTBResponse,
    intent_applicable,
)

logger = logging.getLogger("artf.template")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ADD_CIDS is the default because it is the one intent in the ARTF enum that no
# container in this repo implements, so the template fills a real gap instead of
# shadowing a working container. Override with ARTF_TEMPLATE_INTENT, and update
# the registry record's `intents` to match (see the module docstring).
_DEFAULT_INTENT = "ADD_CIDS"
_INTENT_NAME = os.environ.get("ARTF_TEMPLATE_INTENT", _DEFAULT_INTENT).strip().upper()

try:
    INTENT: Intent = Intent[_INTENT_NAME]
except KeyError:
    # An unknown name is reported and then ignored in favour of the default. The
    # alternative — crashing on startup — would take the container out of the
    # cluster for a typo in an env var, and silently serving intent 0
    # (UNSPECIFIED) would be worse still.
    logger.error(
        "ARTF_TEMPLATE_INTENT=%r is not a valid ARTF intent; falling back to %s. "
        "Valid names: %s",
        _INTENT_NAME, _DEFAULT_INTENT, ", ".join(i.name for i in Intent),
    )
    INTENT = Intent[_DEFAULT_INTENT]

# Names the unimplemented state on purpose. If you see this string in a bid
# response, you are looking at the template rather than your model — change it
# when you implement mutate() so responses identify your version.
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
    # Read whatever you need off the request:
    #
    #     imps  = req.bid_request.get("imp", [])
    #     site  = req.bid_request.get("site", req.bid_request.get("app", {}))
    #     user  = req.bid_request.get("user", {})
    #     device = req.bid_request.get("device", {})
    #
    # Then build one Mutation per change you want applied. A worked example for
    # ADD_CIDS — attaching content IDs to an impression:
    #
    #     from shared.artf_types import IDsPayload
    #
    #     mutations = []
    #     for imp in imps:
    #         content_ids = your_lookup(site.get("domain"), imp)   # your logic
    #         if not content_ids:
    #             continue          # nothing to say about this impression
    #         mutations.append(Mutation(
    #             intent=INTENT,
    #             op=Operation.ADD,
    #             path=f"/imp/{imp.get('id', '')}/ext/cids",
    #             ids=IDsPayload(id=content_ids),
    #         ))
    #     return RTBResponse(
    #         id=req.id,
    #         mutations=mutations,
    #         metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION),
    #     )
    #
    # Which payload field to set depends on your intent:
    #
    #     ACTIVATE_SEGMENTS / ACTIVATE_DEALS / SUPPRESS_DEALS / ADD_CIDS
    #                                     -> ids=IDsPayload(id=[...])
    #     BID_SHADE                       -> adjust_bid=AdjustBidPayload(price=...)
    #     ADJUST_DEAL_FLOOR / _MARGIN     -> adjust_deal=AdjustDealPayload(...)
    #     ADD_METRICS                     -> add_metrics=AddMetricsPayload(metric=[...])
    #
    # Calling a model on Triton instead of writing rules? Copy the pattern in
    # containers/dlrm_bid_shader/app.py: a `triton_inference.py` beside this
    # file, gated on a USE_TRITON env var, reading TRITON_URL. Add your model to
    # source/triton/model_repository/ and register the container-to-model
    # mapping in orchestrator/app.py's `container_to_model` so the health panel
    # can probe your model's readiness too.
    #
    # Keep it fast. This runs on the live bid path inside the request's `tmax`
    # budget (100ms by default, shared with every other container since they
    # run concurrently). mutate() is synchronous and is offloaded to a thread
    # pool by shared/server.py, so blocking I/O here is safe but slow — exceed
    # tmax and the orchestrator reports you as `timeout` and drops your
    # mutations.
    #
    # Do not invent values. If you cannot compute something, return fewer
    # mutations. A fabricated score is worse than an absent one, and the status
    # vocabulary already has a truthful way to say "I ran and had nothing to
    # add".
    # =======================================================================

    # Shipped behaviour: no mutations. Reported as `no_mutations` — reached,
    # ran, produced nothing — which is exactly what an unimplemented template
    # should say about itself.
    return RTBResponse(
        id=req.id,
        mutations=[],
        metadata=Metadata(api_version="1.0", model_version=MODEL_VERSION),
    )


if __name__ == "__main__":
    from shared.server import run_artf_server

    logging.basicConfig(level=logging.INFO)
    logger.info(
        "ARTF template container starting — intent=%s model_version=%s "
        "(returns no mutations until mutate() is implemented)",
        INTENT.name, MODEL_VERSION,
    )
    run_artf_server(
        mutate,
        agent_name="artf-template",
        grpc_port=50051,
        mcp_port=8081,
        health_port=8080,
    )
