"""NCF Deal Manager: ARTF container for ACTIVATE_DEALS / SUPPRESS_DEALS.

Implements Neural Collaborative Filtering (He et al. 2017) following
NVIDIA's DeepLearningExamples NCF implementation.

- NVIDIA NCF: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
- Paper: https://arxiv.org/abs/1708.05031

When USE_TRITON=true, inference is delegated to NVIDIA Triton Inference
Server via tritonclient.http.  Otherwise, PyTorch runs inline (CPU).

The id hash, the thresholds and the decision rule come from shared/ncf_ids.py and
the model definition from shared/ncf_model.py, the same files the trainer
(training/ncf_sample/build.py) reads.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    AdjustDealPayload, IDsPayload, Intent, Metadata, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)

# The deal book lives in shared/ because the orchestrator reads it too. The image
# copies shared/ to /app/shared/ (see source/Dockerfile); the tests reach it
# through the sys.path entry above.
from shared import deal_library  # noqa: E402
from shared import ncf_ids  # noqa: E402

USE_TRITON = os.environ.get("USE_TRITON", "").lower() in ("1", "true", "yes")

ACTIVATE_THRESHOLD = ncf_ids.ACTIVATE_THRESHOLD
SUPPRESS_THRESHOLD = ncf_ids.SUPPRESS_THRESHOLD

#: Env var carrying the served model's version label (set by deploy.sh from the
#: build manifest). Required on the Triton path.
MODEL_LABEL_ENV = "NCF_MODEL_LABEL"


def _model_label_from_env() -> str:
    """The served model's label from NCF_MODEL_LABEL. Raises when it is unset or malformed."""
    label = os.environ.get(MODEL_LABEL_ENV, "").strip()
    if not ncf_ids.is_model_label(label):
        raise RuntimeError(
            f"{MODEL_LABEL_ENV}={label or '<unset>'} does not match {ncf_ids.MODEL_LABEL_RE}; "
            "the Deal Scorer will not serve a model it cannot name"
        )
    return label


# ---------------------------------------------------------------------------
# Inference backend: Triton or PyTorch
# ---------------------------------------------------------------------------

if USE_TRITON:
    import numpy as np
    from container.triton_inference import predict_relevance as _triton_predict_relevance

    # Reported when the router has not resolved a served_model_version. The
    # per-request value comes from _score_deals's served_model_version (FR-2 / Q1=A).
    MODEL_VERSION = _model_label_from_env()

    def _score_deals(uid_hash: int, deal_ids: list[str],
                     activate_threshold: float = ACTIVATE_THRESHOLD,
                     suppress_threshold: float = SUPPRESS_THRESHOLD) -> tuple[list[str], list[str], str, str]:
        """Returns (to_activate, to_suppress, served_variant, served_model_version).

        Reads the load-test-only target-variant override from
        shared.load_test_context, which is None for all live traffic.
        """
        from shared.load_test_context import get_target_variant

        user_arr = np.array([uid_hash] * len(deal_ids), dtype=np.int64)
        deal_arr = np.array([_h(d, ncf_ids.NB_ITEMS) for d in deal_ids], dtype=np.int64)
        scores, served_variant, served_model_version = _triton_predict_relevance(
            user_arr, deal_arr, target_variant=get_target_variant()
        )
        to_act, to_sup = ncf_ids.classify(deal_ids, [float(s) for s in scores],
                                          activate_threshold, suppress_threshold)
        return to_act, to_sup, served_variant, served_model_version

else:
    import torch

    from shared.ncf_model import NeuMF

    # Untrained weights. The trained model is served by Triton (USE_TRITON=true).
    _model = NeuMF()
    _model.eval()
    MODEL_VERSION = "ncf-neumf-v1"

    def _score_deals(uid_hash: int, deal_ids: list[str],
                     activate_threshold: float = ACTIVATE_THRESHOLD,
                     suppress_threshold: float = SUPPRESS_THRESHOLD) -> tuple[list[str], list[str], str, str]:
        """Returns (to_activate, to_suppress, served_variant, served_model_version).

        The non-Triton (inline PyTorch) path has no canary/variant concept:
        served_variant/served_model_version are always "" (a real "unknown"),
        matching this container's static MODEL_VERSION.
        """
        deal_hashes = torch.tensor([_h(d, ncf_ids.NB_ITEMS) for d in deal_ids])
        user_tensor = torch.tensor([uid_hash] * len(deal_ids))
        with torch.no_grad():
            scores = _model(user_tensor, deal_hashes)
        to_act, to_sup = ncf_ids.classify(deal_ids, [float(s) for s in scores.reshape(-1).tolist()],
                                          activate_threshold, suppress_threshold)
        return to_act, to_sup, "", ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: The id hash training and serving share (md5 mod 2000).
_h = ncf_ids.bucket


# ---------------------------------------------------------------------------
# ARTF mutate
# ---------------------------------------------------------------------------

def mutate(req: RTBRequest) -> RTBResponse:
    act_ok = intent_applicable(Intent.ACTIVATE_DEALS, req.applicable_intents)
    sup_ok = intent_applicable(Intent.SUPPRESS_DEALS, req.applicable_intents)
    # The floor that completes a library activation is its own intent, so a request
    # that narrows ADJUST_DEAL_FLOOR away gets the activation without the floor
    # (the deal then binds at the impression floor) rather than a mutation it
    # did not ask for.
    floor_ok = intent_applicable(Intent.ADJUST_DEAL_FLOOR, req.applicable_intents)
    if not act_ok and not sup_ok:
        return RTBResponse(id=req.id, metadata=Metadata(model_version=MODEL_VERSION))

    # Read model parameter overrides from the request (frontend sliders)
    params = req.model_params or {}
    activate_threshold = params.get('activate_threshold', ACTIVATE_THRESHOLD)
    suppress_threshold = params.get('suppress_threshold', SUPPRESS_THRESHOLD)

    uid = _h(req.bid_request.get("user", {}).get("id", "unknown"))
    mutations: list[Mutation] = []
    # Per-request resolved model_version, updated as imps with deals are
    # scored (applies to ALL traffic per Q1=A). Starts at the static
    # constant so an imp-less/deal-less request still reports something
    # real rather than an undefined value.
    resolved_model_version = MODEL_VERSION

    # The publisher's deal book beyond what this request offered. Scored with
    # the same model as the request's own deals; a library deal that scores
    # high enough is ACTIVATED onto the impression, one that does not is left
    # alone -- never SUPPRESSED, since the publisher never offered it here.
    library = deal_library.deals_for(req.bid_request)

    for imp in req.bid_request.get("imp", []):
        imp_id = imp.get("id", "")
        pmp = imp.get("pmp")
        deals = (pmp or {}).get("deals", []) or []
        # Library deals go only onto impressions the publisher opened a private
        # marketplace on (a `pmp` object, deals or not). An impression with no
        # `pmp` was never offered to deal buyers, and activating onto it would put
        # a deal-bound creative in a slot it was not sold for. The isv-ecosystem
        # scenario's 300x600 rail is the shipped case: a request-level library
        # applied to every imp would fill it with a 970x250 deal.
        candidates = deal_library.not_on_request(library, deals) if pmp is not None else []
        if not deals and not candidates:
            continue

        deal_ids = [d.get("id", f"deal-{i}") for i, d in enumerate(deals)]
        candidate_ids = [d.id for d in candidates]
        to_act, to_sup, served_variant, served_model_version = _score_deals(
            uid, deal_ids + candidate_ids,
            activate_threshold=activate_threshold,
            suppress_threshold=suppress_threshold,
        )
        if served_model_version:
            resolved_model_version = served_model_version

        # Activation applies to library candidates only (an offered deal is already
        # on the impression) and suppression to offered deals only.
        to_act, to_sup = ncf_ids.restrict(deal_ids, candidate_ids, to_act, to_sup)
        activated_from_library = [c for c in candidates if c.id in to_act]

        if to_act and act_ok:
            mutations.append(Mutation(
                intent=Intent.ACTIVATE_DEALS, op=Operation.ADD,
                path=f"/imp/{imp_id}", ids=IDsPayload(id=to_act),
            ))
            # ACTIVATE_DEALS carries ids only (ARTF IDsPayload), so an activated
            # library deal would reach the seats with no floor and bind at the
            # impression floor. Its floor from the publisher's book follows as a
            # separate ADJUST_DEAL_FLOOR on the deal's own path. Emitted after the
            # activation, which the hook applies in order, so the deal exists when
            # the floor is written.
            for lib_deal in activated_from_library if floor_ok else []:
                mutations.append(Mutation(
                    intent=Intent.ADJUST_DEAL_FLOOR, op=Operation.REPLACE,
                    path=f"/imp/{imp_id}/deals/{lib_deal.id}",
                    adjust_deal=AdjustDealPayload(bidfloor=lib_deal.bidfloor),
                ))
        if to_sup and sup_ok:
            mutations.append(Mutation(
                intent=Intent.SUPPRESS_DEALS, op=Operation.REMOVE,
                path=f"/imp/{imp_id}", ids=IDsPayload(id=to_sup),
            ))

    return RTBResponse(id=req.id, mutations=mutations,
                       metadata=Metadata(api_version="1.0", model_version=resolved_model_version))


# ---------------------------------------------------------------------------
# Readiness warm-up (shared.server.WarmupGate)
# ---------------------------------------------------------------------------

#: Shaped like the home-lifestyle scenario: one impression with a private
#: marketplace and one offered deal, so mutate() scores at least that deal (plus
#: any deal-book candidates) through Triton. The user id is not a scenario user.
#: Run inside the pod only; nothing is emitted.
WARMUP_REQUEST: dict = {
    "id": "artf-warmup-ncf-deal-manager",
    "tmax": 120,
    "lifecycle": "LIFECYCLE_PUBLISHER_BID_REQUEST",
    "applicable_intents": ["ACTIVATE_DEALS", "SUPPRESS_DEALS"],
    "bid_request": {
        "id": "artf-warmup-auction",
        "imp": [{
            "id": "imp-1",
            "banner": {"w": 300, "h": 250, "pos": 1},
            "bidfloor": 2.50,
            "bidfloorcur": "USD",
            "pmp": {
                "private_auction": 0,
                "deals": [{"id": "deal-remnant-open-hearth", "bidfloor": 1.00, "at": 3}],
            },
        }],
        "site": {"id": "artf-warmup-site", "domain": "hearth-and-home.example", "cat": ["283"], "cattax": 9},
        "user": {"id": "artf-warmup-user"},
        "device": {"ua": "Mozilla/5.0", "geo": {"country": "USA", "region": "WA"}},
    },
}


def warmup_check(resp: RTBResponse) -> None:
    """On the Triton path, require the scorer model to report ready after the call.

    predict_relevance answers a Triton-reported error with 0.5 scores, so the
    response alone cannot show that the inference succeeded.
    """
    if USE_TRITON:
        from container.triton_inference import is_triton_ready
        if not is_triton_ready():
            raise RuntimeError("Triton or the ncf_deal_manager model is not ready")


if __name__ == "__main__":
    # Prime the deal book before serving. A book that cannot be read must stop the
    # process here: the readiness probe then never passes and the pod never takes
    # traffic.
    try:
        deal_library.start()
    except Exception as exc:  # noqa: BLE001 - any prime failure is fatal
        print(
            f"deal book prime failed from {deal_library.describe_source()}: {exc}",
            file=sys.stderr, flush=True,
        )
        sys.exit(1)
    from shared.server import run_artf_server, warmup_from_request
    run_artf_server(
        mutate, agent_name="ncf-deal-manager", grpc_port=50051, mcp_port=8081, health_port=8080,
        warmup=warmup_from_request(mutate, WARMUP_REQUEST, require_triton=USE_TRITON, check=warmup_check),
    )
