"""NCF Deal Manager — ARTF container for ACTIVATE_DEALS / SUPPRESS_DEALS.

Implements Neural Collaborative Filtering (He et al. 2017) following
NVIDIA's DeepLearningExamples NCF implementation.

- NVIDIA NCF: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/NCF
- Paper: https://arxiv.org/abs/1708.05031

When USE_TRITON=true, inference is delegated to NVIDIA Triton Inference
Server via tritonclient.http.  Otherwise, PyTorch runs inline (CPU).
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.artf_types import (
    AdjustDealPayload, IDsPayload, Intent, Metadata, Mutation, Operation,
    RTBRequest, RTBResponse, intent_applicable,
)

# The image copies this directory to /app/container/ (see source/Dockerfile); the
# tests import app.py by path. Both must find the library.
try:
    from container import deal_library  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by the test import path
    # By file path, not by adding this directory to sys.path: the other containers
    # also have an `app.py`, and a path entry here would shadow theirs for any
    # test that imports more than one container.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ncf_deal_manager_deal_library",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deal_library.py"),
    )
    deal_library = _ilu.module_from_spec(_spec)  # type: ignore[assignment]
    sys.modules[_spec.name] = deal_library
    _spec.loader.exec_module(deal_library)

USE_TRITON = os.environ.get("USE_TRITON", "").lower() in ("1", "true", "yes")

ACTIVATE_THRESHOLD = 0.499
SUPPRESS_THRESHOLD = 0.497


# ---------------------------------------------------------------------------
# Inference backend — Triton or PyTorch
# ---------------------------------------------------------------------------

if USE_TRITON:
    import numpy as np
    from container.triton_inference import predict_relevance as _triton_predict_relevance

    # Static placeholder — used ONLY as a fallback when the router hasn't
    # resolved a served_model_version. Per-request accurate value comes from
    # _score_deals's served_model_version return value (see FR-2 / Q1=A).
    MODEL_VERSION = "ncf-neumf-triton-v1"

    def _score_deals(uid_hash: int, deal_ids: list[str],
                     activate_threshold: float = ACTIVATE_THRESHOLD,
                     suppress_threshold: float = SUPPRESS_THRESHOLD) -> tuple[list[str], list[str], str, str]:
        """Returns (to_activate, to_suppress, served_variant, served_model_version).

        Reads the load-test-only target-variant override from
        shared.load_test_context, which is None for all live traffic.
        """
        from shared.load_test_context import get_target_variant

        user_arr = np.array([uid_hash] * len(deal_ids), dtype=np.int64)
        deal_arr = np.array([_h(d) for d in deal_ids], dtype=np.int64)
        scores, served_variant, served_model_version = _triton_predict_relevance(
            user_arr, deal_arr, target_variant=get_target_variant()
        )
        to_act = [deal_ids[j] for j in range(len(deal_ids)) if scores[j] >= activate_threshold]
        to_sup = [deal_ids[j] for j in range(len(deal_ids)) if scores[j] < suppress_threshold]
        return to_act, to_sup, served_variant, served_model_version

else:
    import torch
    import torch.nn as nn

    class NeuMF(nn.Module):
        """NeuMF following NVIDIA's DeepLearningExamples NCF implementation."""

        def __init__(self, nb_users=2000, nb_items=2000, mf_dim=64, mlp_layer_sizes=None):
            super().__init__()
            if mlp_layer_sizes is None:
                mlp_layer_sizes = [256, 128, 64]
            self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
            self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
            mlp_embed_dim = mlp_layer_sizes[0] // 2
            self.mlp_user_embed = nn.Embedding(nb_users, mlp_embed_dim)
            self.mlp_item_embed = nn.Embedding(nb_items, mlp_embed_dim)
            mlp_layers = []
            input_size = mlp_layer_sizes[0]
            for output_size in mlp_layer_sizes[1:]:
                mlp_layers.append(nn.Linear(input_size, output_size))
                mlp_layers.append(nn.ReLU())
                input_size = output_size
            self.mlp = nn.Sequential(*mlp_layers)
            self.final = nn.Sequential(nn.Linear(mf_dim + mlp_layer_sizes[-1], 1), nn.Sigmoid())
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, user_ids, item_ids):
            mf_user = self.mf_user_embed(user_ids)
            mf_item = self.mf_item_embed(item_ids)
            gmf_out = mf_user * mf_item
            mlp_user = self.mlp_user_embed(user_ids)
            mlp_item = self.mlp_item_embed(item_ids)
            mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)
            mlp_out = self.mlp(mlp_in)
            concat = torch.cat([gmf_out, mlp_out], dim=-1)
            return self.final(concat).squeeze(-1)

    _model = NeuMF()
    _model.eval()
    MODEL_VERSION = "ncf-neumf-v1"

    def _score_deals(uid_hash: int, deal_ids: list[str],
                     activate_threshold: float = ACTIVATE_THRESHOLD,
                     suppress_threshold: float = SUPPRESS_THRESHOLD) -> tuple[list[str], list[str], str, str]:
        """Returns (to_activate, to_suppress, served_variant, served_model_version).

        The non-Triton (inline PyTorch) path has no canary/variant concept —
        served_variant/served_model_version are always "" (a real "unknown"),
        matching this container's static MODEL_VERSION.
        """
        deal_hashes = torch.tensor([_h(d) for d in deal_ids])
        user_tensor = torch.tensor([uid_hash] * len(deal_ids))
        with torch.no_grad():
            scores = _model(user_tensor, deal_hashes)
        to_act = [deal_ids[j] for j in range(len(deal_ids)) if scores[j].item() >= activate_threshold]
        to_sup = [deal_ids[j] for j in range(len(deal_ids)) if scores[j].item() < suppress_threshold]
        return to_act, to_sup, "", ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _h(v: str, n: int = 2000) -> int:
    return int(hashlib.md5(v.encode(), usedforsecurity=False).hexdigest(), 16) % n  # nosec B324


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

        # Suppression applies to offered deals only.
        to_sup = [d for d in to_sup if d in deal_ids]
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


if __name__ == "__main__":
    from shared.server import run_artf_server
    run_artf_server(mutate, agent_name="ncf-deal-manager", grpc_port=50051, mcp_port=8081, health_port=8080)
