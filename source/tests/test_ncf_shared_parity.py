"""Training and serving share one NCF id hash, one model and one decision rule.

- the container hashes with shared.ncf_ids.bucket and decides with its thresholds
  (0.60 / 0.30) and classify/restrict;
- the container and the exporter use shared.ncf_model.NeuMF, and no second copy
  remains;
- offered deals never appear in ACTIVATE_DEALS, library deals never in
  SUPPRESS_DEALS, on every shipped scenario sample;
- on the Triton path the container reports NCF_MODEL_LABEL and refuses to load
  without a valid one.

Property tests use Hypothesis with a fixed, printed seed (HYPOTHESIS_SEED).
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import sys
import types

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SOURCE)

from shared import ncf_ids, ncf_model  # noqa: E402
from shared.artf_types import Intent, RTBRequest  # noqa: E402
from training.ncf_sample import generate as gen  # noqa: E402

HYPOTHESIS_SEED = 20261008
_APP = os.path.join(_SOURCE, "containers", "ncf_deal_manager", "app.py")
_SAMPLES = os.path.join(_SOURCE, "frontend-react", "public", "samples")
SEED_DEALS = gen.seed_deal_ids(gen.load_json(gen.DEFAULT_SEED))
_PROPERTY = settings(max_examples=200, deadline=None, database=None, print_blob=True)
_counter = itertools.count()


@pytest.fixture(autouse=True, scope="module")
def _print_seed():
    print(f"hypothesis seed {HYPOTHESIS_SEED}")


def _load_app(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_{next(_counter)}", _APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    module = _load_app("ncf_parity_inline")
    module.deal_library.load()
    return module


def _reference_md5_bucket(value: str, n: int) -> int:
    return int(hashlib.md5(value.encode(), usedforsecurity=False).hexdigest(), 16) % n


# --------------------------------------------------------------- hash parity


def test_the_container_hash_is_the_shared_hash(app):
    assert app._h is ncf_ids.bucket


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(st.one_of(st.sampled_from(SEED_DEALS), st.text(min_size=0, max_size=40)))
def test_bucket_matches_md5_mod_2000(value):
    """PBT-05 oracle: the documented hash, md5 of the UTF-8 id modulo the table size."""
    assert ncf_ids.bucket(value) == _reference_md5_bucket(value, 2000)
    assert 0 <= ncf_ids.bucket(value, ncf_ids.NB_ITEMS) < ncf_ids.NB_ITEMS


def test_the_container_thresholds_are_the_shared_ones(app):
    assert (app.ACTIVATE_THRESHOLD, app.SUPPRESS_THRESHOLD) == (0.60, 0.30)
    assert (ncf_ids.ACTIVATE_THRESHOLD, ncf_ids.SUPPRESS_THRESHOLD) == (0.60, 0.30)


def test_one_neumf_definition(app):
    assert app.NeuMF is ncf_model.NeuMF
    export_src = open(os.path.join(_SOURCE, "triton", "export_models.py"), encoding="utf-8").read()
    app_src = open(_APP, encoding="utf-8").read()
    assert "class NeuMF" not in export_src
    assert "class NeuMF" not in app_src
    assert not os.path.exists(os.path.join(_SOURCE, "training", "container", "models", "ncf.py"))


def test_nothing_imports_the_removed_training_model():
    hits = []
    for root, _, files in os.walk(_SOURCE):
        if "node_modules" in root or "__pycache__" in root:
            continue
        for name in files:
            if name.endswith(".py") and name != os.path.basename(__file__):
                text = open(os.path.join(root, name), encoding="utf-8", errors="ignore").read()
                if "models.ncf" in text or "NCFModel" in text:
                    hits.append(os.path.join(root, name))
    assert hits == []


# ------------------------------------------------------- classify and restrict


scored = st.lists(
    st.tuples(st.sampled_from(SEED_DEALS), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    max_size=24, unique_by=lambda t: t[0],
)
thresholds = st.tuples(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
).map(lambda t: (max(t), min(t)))


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(scored, thresholds)
def test_classify_partitions_by_the_thresholds(pairs, bounds):
    """PBT-03: disjoint, order-preserving, and exactly the ids on each side of the band."""
    activate, suppress = bounds
    ids = [i for i, _ in pairs]
    scores = [s for _, s in pairs]
    to_act, to_sup = ncf_ids.classify(ids, scores, activate, suppress)
    assert not set(to_act) & set(to_sup)
    assert to_act == [i for i, s in pairs if s >= activate]
    assert to_sup == [i for i, s in pairs if s < suppress]
    assert len(to_act) + len(to_sup) <= len(ids)


def test_classify_boundaries_use_the_shared_thresholds():
    ids = ["a", "b", "c", "d"]
    scores = [0.60, 0.5999, 0.30, 0.2999]
    assert ncf_ids.classify(ids, scores) == (["a"], ["d"])


@seed(HYPOTHESIS_SEED)
@_PROPERTY
@given(
    st.lists(st.sampled_from(SEED_DEALS), unique=True, max_size=8),
    st.lists(st.sampled_from(SEED_DEALS), unique=True, max_size=8),
    st.lists(st.sampled_from(SEED_DEALS), unique=True, max_size=12),
    st.lists(st.sampled_from(SEED_DEALS), unique=True, max_size=12),
)
def test_restrict_keeps_only_candidates_and_offered(offered, candidates, to_act, to_sup):
    """PBT-03: activations are library candidates, suppressions offered deals; idempotent."""
    act, sup = ncf_ids.restrict(offered, candidates, to_act, to_sup)
    assert act == [d for d in to_act if d in candidates]
    assert sup == [d for d in to_sup if d in offered]
    assert ncf_ids.restrict(offered, candidates, act, sup) == (act, sup)


def test_model_label_pattern():
    assert ncf_ids.is_model_label("ncf-neumf-sample-0123456789ab")
    for bad in ("ncf-neumf-triton-v1", "ncf-neumf-sample-0123456789AB", "ncf-neumf-sample-012345",
                "ncf-neumf-sample-0123456789abc", "", None):
        assert not ncf_ids.is_model_label(bad)


# ---------------------------------------------- the container's mutate rule


def _samples():
    out = []
    for name in sorted(os.listdir(_SAMPLES)):
        if name.endswith(".json"):
            with open(os.path.join(_SAMPLES, name), encoding="utf-8") as handle:
                sample = json.load(handle)
            if isinstance(sample.get("bid_request"), dict):
                out.append((name, sample))
    return out


@pytest.mark.parametrize("name,sample", _samples(), ids=[n for n, _ in _samples()])
def test_offered_deals_are_never_activated_and_library_deals_never_suppressed(app, monkeypatch, name, sample):
    def everything(uid_hash, deal_ids, activate_threshold=0.0, suppress_threshold=0.0):
        return list(deal_ids), list(deal_ids), "", ""

    monkeypatch.setattr(app, "_score_deals", everything)
    request = RTBRequest(id="t", bid_request=sample["bid_request"],
                         applicable_intents=["ACTIVATE_DEALS", "SUPPRESS_DEALS", "ADJUST_DEAL_FLOOR"])
    response = app.mutate(request)
    offered = {
        d.get("id")
        for imp in sample["bid_request"].get("imp", [])
        for d in ((imp.get("pmp") or {}).get("deals") or [])
    }
    for m in response.mutations:
        if m.intent == Intent.ACTIVATE_DEALS:
            assert not set(m.ids.id) & offered, (name, m.ids.id)
        if m.intent == Intent.SUPPRESS_DEALS:
            assert set(m.ids.id) <= offered, (name, m.ids.id)


# ------------------------------------------------------- the Triton label


@pytest.fixture
def triton_env(monkeypatch):
    """Load app.py on its Triton branch with a stub Triton client."""
    calls = []

    def predict_relevance(user_ids, item_ids, target_variant=None):
        calls.append((list(user_ids), list(item_ids)))
        return [1.0] * len(item_ids), "stable", ""

    package = types.ModuleType("container")
    package.__path__ = []
    stub = types.ModuleType("container.triton_inference")
    stub.predict_relevance = predict_relevance
    monkeypatch.setitem(sys.modules, "container", package)
    monkeypatch.setitem(sys.modules, "container.triton_inference", stub)
    monkeypatch.setenv("USE_TRITON", "true")
    return calls


def test_the_triton_path_refuses_to_load_without_a_label(triton_env, monkeypatch):
    monkeypatch.delenv("NCF_MODEL_LABEL", raising=False)
    with pytest.raises(RuntimeError, match="NCF_MODEL_LABEL"):
        _load_app("ncf_parity_triton_unset")


def test_the_triton_path_refuses_a_malformed_label(triton_env, monkeypatch):
    monkeypatch.setenv("NCF_MODEL_LABEL", "ncf-neumf-triton-v1")
    with pytest.raises(RuntimeError, match="ncf-neumf-triton-v1"):
        _load_app("ncf_parity_triton_bad")


def test_the_triton_path_reports_the_label_and_restricts(triton_env, monkeypatch):
    monkeypatch.setenv("NCF_MODEL_LABEL", "ncf-neumf-sample-0123456789ab")
    module = _load_app("ncf_parity_triton_ok")
    module.deal_library.load()
    assert module.MODEL_VERSION == "ncf-neumf-sample-0123456789ab"
    with open(os.path.join(_SAMPLES, "banner-basic.json"), encoding="utf-8") as handle:
        bid_request = json.load(handle)["bid_request"]
    response = module.mutate(RTBRequest(id="t", bid_request=bid_request,
                                        applicable_intents=["ACTIVATE_DEALS", "SUPPRESS_DEALS"]))
    assert response.metadata.model_version == "ncf-neumf-sample-0123456789ab"
    activations = [m for m in response.mutations if m.intent == Intent.ACTIVATE_DEALS]
    assert [m.ids.id for m in activations] == [["deal-sports-premium-espn"]]
    user_bucket = ncf_ids.bucket(bid_request["user"]["id"])
    assert triton_env == [([user_bucket, user_bucket],
                           [ncf_ids.bucket("deal-remnant-open-espn"), ncf_ids.bucket("deal-sports-premium-espn")])]
