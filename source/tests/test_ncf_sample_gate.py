"""The Deal Scorer sample build: training, ONNX export, the acceptance gate and the cache.

- the exported ONNX equals the trained torch model within 1e-5 (PBT-02, Hypothesis
  over id pairs, fixed printed seed);
- the ONNX inputs and output match ncf_deal_manager_stable/config.pbtxt by name,
  dtype and rank;
- the gate passes on the built model for every cohort, and FAILS (GateFailed, exit
  1, nothing written) when a cohort target is flipped, both on the built model and
  through a full retrain;
- the container's own mutate(), scoring through onnxruntime on the trained model,
  emits each cohort's ACTIVATE_DEALS and SUPPRESS_DEALS, so the gate's copy of the
  container loop cannot drift from the container;
- a cache entry with an unreadable manifest, or one for another build, is rebuilt;
- the manifest supplies every field the registry records;
- the fingerprint excludes prices and covers the cohort spec; a cache hit reuses the
  model and re-runs the gate;
- triton/export_models.py --model ncf exits 0, and 1 on a flipped target.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

_SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SOURCE)

from shared import ncf_ids, ncf_model  # noqa: E402
from training.ncf_sample import build  # noqa: E402
from training.ncf_sample import generate as gen  # noqa: E402

HYPOTHESIS_SEED = 20261009
_CONFIG = Path(_SOURCE) / "triton" / "model_repository" / "ncf_deal_manager_stable" / "config.pbtxt"
SEED_DEALS = gen.seed_deal_ids(gen.load_json(gen.DEFAULT_SEED))
SPEC = gen.load_json(gen.DEFAULT_COHORTS)
SCENARIO_USERS = [u for c in SPEC["cohorts"] for u in c["scenario_users"]]


@pytest.fixture(autouse=True, scope="module")
def _print_seeds():
    print(f"hypothesis seed {HYPOTHESIS_SEED}; generator seed {SPEC['generator_seed']}")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One training run on the shipped spec: (torch model, onnx path, inputs)."""
    out = tmp_path_factory.mktemp("ncf-trained")
    rows = gen.generate(SPEC, SEED_DEALS)
    model, report = build.train(rows)
    path = build.export(model, str(out / "model.onnx"))
    return model, path, build.load_inputs(), report


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A full build() into a temp output and a temp cache."""
    root = tmp_path_factory.mktemp("ncf-built")
    cache = root / "cache"
    result = build.build(str(root / "out"), cache_dir=str(cache), log=lambda _: None)
    return result, root / "out", cache


def _flipped(**changes) -> dict:
    spec = copy.deepcopy(SPEC)
    for cohort_id, fields in changes.items():
        cohort = next(c for c in spec["cohorts"] if c["id"] == cohort_id)
        cohort.update(fields)
    return spec


def _inputs_with(spec: dict) -> build.Inputs:
    inputs = build.load_inputs()
    inputs.cohorts = spec
    return inputs


# ------------------------------------------------------------- the export


_TENSOR = re.compile(
    r'name:\s*"(?P<name>[^"]+)"\s*data_type:\s*(?P<dtype>\w+)\s*dims:\s*\[(?P<dims>[^\]]*)\]'
    r'(?:\s*reshape:\s*\{\s*shape:\s*\[(?P<reshape>[^\]]*)\]\s*\})?'
)


def _config_tensors(text: str, block: str) -> list[dict]:
    """Name, dtype and model-side rank of each tensor in a config.pbtxt block.

    With max_batch_size > 0 Triton adds the batch dimension, and ``reshape``
    replaces ``dims`` as the shape the model sees, so the ONNX rank is
    1 + len(reshape or dims).
    """
    body = text[text.index(f"{block} ["):]
    body = body[: body.index("\n]")]
    tensors = []
    for m in _TENSOR.finditer(body):
        shape = m.group("dims") if m.group("reshape") is None else m.group("reshape")
        dims = [d for d in shape.split(",") if d.strip()]
        tensors.append({"name": m.group("name"), "dtype": m.group("dtype"), "rank": 1 + len(dims)})
    return tensors


def test_the_onnx_signature_matches_the_stable_config(trained):
    import onnx

    _, path, _, _ = trained
    text = _CONFIG.read_text()
    assert re.search(r"max_batch_size:\s*\d+", text)
    graph = onnx.load(path).graph
    onnx_dtype = {onnx.TensorProto.INT64: "TYPE_INT64", onnx.TensorProto.FLOAT: "TYPE_FP32"}

    def described(values):
        return [{"name": v.name, "dtype": onnx_dtype[v.type.tensor_type.elem_type],
                 "rank": len(v.type.tensor_type.shape.dim)} for v in values]

    assert described(graph.input) == _config_tensors(text, "input")
    assert described(graph.output) == _config_tensors(text, "output")
    assert [i.name for i in graph.input] == list(ncf_model.INPUT_NAMES)
    assert [o.name for o in graph.output] == [ncf_model.OUTPUT_NAME]
    # Dynamic batch on every tensor.
    for v in [*graph.input, *graph.output]:
        assert v.type.tensor_type.shape.dim[0].dim_param == "batch"


@seed(HYPOTHESIS_SEED)
@settings(max_examples=60, deadline=None, database=None, print_blob=True)
@given(st.lists(
    st.tuples(st.one_of(st.sampled_from(SCENARIO_USERS), st.text(min_size=1, max_size=20)),
              st.one_of(st.sampled_from(SEED_DEALS), st.text(min_size=1, max_size=20))),
    min_size=1, max_size=64,
))
def test_the_onnx_export_equals_the_torch_model(trained, pairs):
    """PBT-02: export then run equals the trained model, on scenario and unseen ids."""
    import onnxruntime as ort
    import torch

    model, path, _, _ = trained
    users = np.array([ncf_ids.bucket(u, ncf_ids.NB_USERS) for u, _ in pairs], dtype=np.int64)
    items = np.array([ncf_ids.bucket(d, ncf_ids.NB_ITEMS) for _, d in pairs], dtype=np.int64)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    (onnx_out,) = session.run(["relevance_scores"], {"user_ids": users, "item_ids": items})
    with torch.no_grad():
        torch_out = model(torch.from_numpy(users), torch.from_numpy(items)).numpy()
    assert onnx_out.dtype == np.float32
    assert onnx_out.shape == (len(pairs),)
    np.testing.assert_allclose(onnx_out, torch_out, rtol=0, atol=1e-5)


# --------------------------------------------------------------- the gate


def test_the_gate_passes_for_every_cohort(trained):
    _, path, inputs, _ = trained
    report = build.gate(path, inputs)
    assert report.passed, build.format_table(report)
    assert [s.cohort for s in report.scenarios] == [c["id"] for c in SPEC["cohorts"]]
    for s, cohort in zip(report.scenarios, SPEC["cohorts"]):
        assert sorted(d.split(":", 1)[1] for d in s.activated) == sorted(cohort["activate"])
        assert sorted(d.split(":", 1)[1] for d in s.suppressed) == sorted(cohort["suppress"])


def test_every_gated_score_keeps_the_margin(trained):
    _, path, inputs, _ = trained
    report = build.gate(path, inputs)
    for s in report.scenarios:
        for c in s.checks:
            if c.role == "activate":
                assert c.score >= 0.75
            elif c.role == "suppress":
                assert c.score < 0.15
            elif c.role == "keep":
                assert c.score >= 0.45
            elif c.role == "candidate":
                assert c.score < 0.45


@pytest.fixture(scope="module")
def container(trained):
    """The Deal Scorer app with _score_deals scoring through onnxruntime on the trained model.

    The scorer mirrors the Triton path's _score_deals (hashed user, deal buckets,
    shared classify), so mutate() runs its own per-impression loop on trained scores.
    """
    import onnxruntime as ort

    path = os.path.join(_SOURCE, "containers", "ncf_deal_manager", "app.py")
    spec = importlib.util.spec_from_file_location("ncf_deal_manager_gate_targets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.deal_library.load()
    session = ort.InferenceSession(trained[1], providers=["CPUExecutionProvider"])

    def _score(uid_hash, deal_ids, activate_threshold=ncf_ids.ACTIVATE_THRESHOLD,
               suppress_threshold=ncf_ids.SUPPRESS_THRESHOLD):
        users = np.array([uid_hash] * len(deal_ids), dtype=np.int64)
        items = np.array([ncf_ids.bucket(d, ncf_ids.NB_ITEMS) for d in deal_ids], dtype=np.int64)
        (out,) = session.run(["relevance_scores"], {"user_ids": users, "item_ids": items})
        to_act, to_sup = ncf_ids.classify(deal_ids, [float(v) for v in np.asarray(out).reshape(-1)],
                                          activate_threshold, suppress_threshold)
        return to_act, to_sup, "", ""

    module._score_deals = _score
    return module


@pytest.mark.parametrize("cohort", SPEC["cohorts"], ids=lambda c: c["id"])
def test_the_container_reproduces_each_cohort_target(container, cohort):
    """mutate() on the cohort's sample, with trained scores, emits exactly the cohort's decisions."""
    from shared.artf_types import Intent, RTBRequest

    sample = gen.load_json(os.path.join(build.DEFAULT_SAMPLES, cohort["sample"]))
    request = RTBRequest(**sample)
    response = container.mutate(request)
    activated = [i for m in response.mutations if m.intent == Intent.ACTIVATE_DEALS for i in m.ids.id]
    suppressed = [i for m in response.mutations if m.intent == Intent.SUPPRESS_DEALS for i in m.ids.id]
    assert sorted(activated) == sorted(cohort["activate"])
    assert sorted(suppressed) == sorted(cohort["suppress"])
    assert not set(activated) & set(cohort["not_activated"])


def test_the_gate_fails_when_an_activate_target_is_flipped_on_the_built_model(trained):
    """The built model activates deal-finance-pmp; a spec that says not to must fail."""
    _, path, _, _ = trained
    spec = _flipped(**{"finance-ledger": {"activate": [], "not_activated": ["deal-home-premium-ledger", "deal-finance-pmp"]}})
    report = build.gate(path, _inputs_with(spec))
    assert not report.passed
    failed = [s for s in report.scenarios if not s.passed]
    assert [s.cohort for s in failed] == ["finance-ledger"]
    assert any("deal-finance-pmp" in f and "needs < 0.45" in f for f in failed[0].failures)
    table = build.format_table(report)
    assert "FAIL" in table and "deal-finance-pmp" in table


def test_the_gate_fails_when_a_keep_target_is_flipped_to_suppress_on_the_built_model(trained):
    _, path, _, _ = trained
    spec = _flipped(**{"sports-espn": {"keep": [], "suppress": ["deal-remnant-open-espn"]}})
    report = build.gate(path, _inputs_with(spec))
    failed = [s for s in report.scenarios if not s.passed]
    assert [s.cohort for s in failed] == ["sports-espn"]
    assert any("deal-remnant-open-espn" in f and "needs < 0.15" in f for f in failed[0].failures)


def test_the_gate_fails_on_a_missing_sample(trained):
    _, path, _, _ = trained
    spec = _flipped(**{"yield-sports": {"sample": "no-such-sample.json"}})
    report = build.gate(path, _inputs_with(spec))
    failed = [s for s in report.scenarios if not s.passed]
    assert "yield-sports" in [s.cohort for s in failed]
    assert any("missing" in f for s in failed for f in s.failures)


def test_the_gate_fails_when_the_sample_user_is_not_a_scenario_user(trained):
    _, path, _, _ = trained
    spec = _flipped(**{"auto-cnn": {"scenario_users": ["user-somebody-else"]}})
    report = build.gate(path, _inputs_with(spec))
    assert not report.passed
    assert any("not a scenario user" in f for s in report.scenarios for f in s.failures)


def test_an_offered_deal_left_out_of_the_spec_fails(trained):
    _, path, _, _ = trained
    spec = _flipped(**{"video-streaming": {"keep": []}})
    report = build.gate(path, _inputs_with(spec))
    assert any("in neither keep nor suppress" in f for s in report.scenarios for f in s.failures)


def test_a_retrain_on_a_flipped_target_fails_and_writes_nothing(tmp_path):
    """activate -> suppress for a library deal: training follows the spec, the gate does not pass."""
    spec = _flipped(**{"sports-espn": {"activate": [], "suppress": ["deal-sports-premium-espn"]}})
    cohorts = tmp_path / "cohorts.json"
    cohorts.write_text(json.dumps(spec))
    out = tmp_path / "out"
    with pytest.raises(build.GateFailed) as failed:
        build.build(str(out), cohorts_path=str(cohorts), cache_dir=str(tmp_path / "cache"), log=lambda _: None)
    bad = [s for s in failed.value.report.scenarios if not s.passed]
    assert [s.cohort for s in bad] == ["sports-espn"]
    assert any("not offered on the request" in f for f in bad[0].failures)
    assert not out.exists() or not any(out.iterdir())
    # A failing build is not cached.
    assert not any(p for p in (tmp_path / "cache").iterdir() if not p.name.startswith("."))
    assert build.main(["--output-dir", str(out), "--cohorts", str(cohorts), "--cache-dir", str(tmp_path / "cache")]) == 1


# --------------------------------------------------- manifest, fingerprint, cache


def test_the_build_writes_the_three_outputs_and_the_manifest(built):
    result, out, cache = built
    assert result.report.passed
    assert sorted(p.name for p in out.iterdir()) == sorted(build.OUTPUT_FILES)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model_version"] == f"ncf-neumf-sample-{result.fingerprint[:12]}"
    assert ncf_ids.is_model_label(manifest["model_version"])
    assert manifest["fingerprint"] == result.fingerprint
    assert manifest["training_data"] == "sample"
    assert manifest["target"] == "implicit-engagement-sample"
    assert manifest["generator_seed"] == SPEC["generator_seed"]
    assert manifest["generator"]["path"] == build.GENERATOR_PATH == "source/training/ncf_sample"
    assert manifest["gate"]["result"] == "pass"
    assert [s["result"] for s in manifest["gate"]["scenarios"]] == ["PASS"] * len(SPEC["cohorts"])
    assert manifest["triton_input_names"] == ["user_ids", "item_ids"]
    assert (out / "interactions.csv").read_bytes() == gen.to_csv_bytes(gen.generate(SPEC, SEED_DEALS))
    entry = cache / result.fingerprint
    assert sorted(p.name for p in entry.iterdir()) == sorted([*build.OUTPUT_FILES, build.GATE_FILE])


def test_the_registry_reads_every_field_from_the_built_manifest(built):
    """deployment/scripts/register_genesis_models.py on the manifest this build writes."""
    scripts = str(Path(_SOURCE).parent / "deployment" / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    rgm = importlib.import_module("register_genesis_models")
    result, out, _ = built

    class _S3:
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO((out / "manifest.json").read_bytes())}

    assert rgm._ncf_build_identity(_S3(), "bucket") == {
        "training_data": "sample",
        "label": "implicit-engagement-sample",
        "generator": "source/training/ncf_sample",
        "seed": str(SPEC["generator_seed"]),
        "fingerprint": result.fingerprint[:12],
    }


def test_a_cache_hit_reuses_the_model_and_reruns_the_gate(built, tmp_path, monkeypatch):
    result, out, cache = built
    calls = []
    real_gate = build.gate

    def no_training(*args, **kwargs):
        raise AssertionError("a cache hit trained a model")

    def counting_gate(*args, **kwargs):
        calls.append(1)
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(build, "train", no_training)
    monkeypatch.setattr(build, "gate", counting_gate)
    again = build.build(str(tmp_path / "again"), cache_dir=str(cache), log=lambda _: None)
    assert again.cache_hit and calls == [1]
    assert again.fingerprint == result.fingerprint
    assert again.report.passed
    assert (tmp_path / "again" / "model.onnx").read_bytes() == (out / "model.onnx").read_bytes()


@pytest.mark.parametrize("damage", ["truncated", "other-fingerprint"])
def test_a_damaged_cache_entry_is_discarded_and_rebuilt(built, tmp_path, damage):
    result, out, cache = built
    copy_cache = tmp_path / "cache"
    shutil.copytree(cache, copy_cache)
    manifest_path = copy_cache / result.fingerprint / build.MANIFEST_FILE
    if damage == "truncated":
        manifest_path.write_text(manifest_path.read_text()[:40])
    else:
        record = json.loads(manifest_path.read_text())
        manifest_path.write_text(json.dumps({**record, "fingerprint": "0" * 64}))
    lines = []
    again = build.build(str(tmp_path / "out"), cache_dir=str(copy_cache), log=lines.append)
    assert not again.cache_hit and again.report.passed
    assert any("discarding it and rebuilding" in line for line in lines)
    rebuilt = json.loads(manifest_path.read_text())
    assert rebuilt["fingerprint"] == result.fingerprint
    assert (tmp_path / "out" / "model.onnx").read_bytes() == (out / "model.onnx").read_bytes()


def test_json_writes_leave_no_temp_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}\n")
    build._write_json(str(path), {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_the_fingerprint_excludes_prices(tmp_path):
    seed_doc = gen.load_json(gen.DEFAULT_SEED)
    base = build.fingerprint(build.load_inputs())
    priced = copy.deepcopy(seed_doc)
    for deal in priced["deals"]:
        deal["terms"]["dealfloor"] = round(float(deal["terms"]["dealfloor"]) + 1.0, 2)
    for campaign in priced["campaigns"]:
        campaign["declared_cpm"] = round(float(campaign["declared_cpm"]) * 2, 2)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(priced))
    assert build.fingerprint(build.load_inputs(seed_path=str(path))) == base


def test_the_fingerprint_covers_eligibility_and_the_cohort_spec(tmp_path):
    base = build.fingerprint(build.load_inputs())
    moved = gen.load_json(gen.DEFAULT_SEED)
    moved["deals"][0]["inventory"]["sitedomains"] = ["elsewhere.example"]
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(moved))
    assert build.fingerprint(build.load_inputs(seed_path=str(path))) != base
    assert build.fingerprint(_inputs_with(_flipped(**{"finance-ledger": {"activate": []}}))) != base
    assert re.fullmatch(r"[0-9a-f]{64}", base)


def test_fingerprint_only_prints_the_fingerprint(capsys):
    assert build.main(["--fingerprint-only"]) == 0
    assert capsys.readouterr().out.strip() == build.fingerprint(build.load_inputs())


# --------------------------------------------------------- export_models.py


@pytest.fixture(scope="module")
def export_models():
    sys.path.insert(0, os.path.join(_SOURCE, "triton"))
    return importlib.import_module("export_models")


def test_export_models_ncf_writes_the_gated_model(export_models, built, tmp_path):
    _, _, cache = built
    out = tmp_path / "x"
    assert export_models.main(["--model", "ncf", "--output-dir", str(out), "--ncf-cache-dir", str(cache)]) == 0
    model_dir = out / "ncf_deal_manager" / "1"
    assert sorted(p.name for p in model_dir.iterdir()) == sorted(build.OUTPUT_FILES)
    assert not hasattr(export_models, "NeuMF")


def test_export_models_ncf_exits_1_on_a_flipped_target(export_models, tmp_path):
    spec = _flipped(**{"sports-espn": {"activate": [], "suppress": ["deal-sports-premium-espn"]}})
    cohorts = tmp_path / "cohorts.json"
    cohorts.write_text(json.dumps(spec))
    out = tmp_path / "x"
    code = export_models.main(["--model", "ncf", "--output-dir", str(out), "--ncf-cohorts", str(cohorts),
                               "--ncf-cache-dir", str(tmp_path / "cache")])
    assert code == 1
    assert not (out / "ncf_deal_manager" / "1" / "model.onnx").exists()
    shutil.rmtree(tmp_path / "cache", ignore_errors=True)
