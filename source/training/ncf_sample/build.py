"""Build the Deal Scorer model from sample interactions: fingerprint, generate, train, export, gate.

The single entry point the deploy uses (through triton/export_models.py --model ncf):

1. ``fingerprint``: sha256 over the training inputs (seed deal ids and eligibility
   fields, the cohort spec, each scenario sample's user id and offered deal ids, the
   training parameters, the bytes of this file, generate.py, shared/ncf_ids.py and
   shared/ncf_model.py, and the torch major.minor version). Prices are not inputs, so
   a CPM edit in the seed keeps the fingerprint.
2. ``generate``: the deterministic interactions of generate.py.
3. ``train``: NeuMF on CPU, one thread, deterministic algorithms, fixed epochs, BCE
   on the model's sigmoid output.
4. ``export``: ONNX with the Triton signature (shared/ncf_model.export_onnx).
5. ``gate``: for every scenario sample whose user is in the cohort spec, score the
   exact candidate set the container scores with onnxruntime on the exported model,
   apply the container's decision rule, and require every cohort target with a 0.15
   margin from each threshold. Any failure prints the per-scenario table and exits 1.
   There is no fallback model.

Outputs model.onnx, manifest.json and interactions.csv in --output-dir. A passing
build is cached under deployment/.ncf-cache/<fingerprint>/ (also gate.json); a cache
hit re-runs the gate on the cached model and skips training. An entry whose manifest
is unreadable or records another fingerprint is discarded and rebuilt. JSON files are
written through a temp file and a rename.

Usage:
    python training/ncf_sample/build.py --output-dir /tmp/ncf-x
    python training/ncf_sample/build.py --fingerprint-only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
REPO_ROOT = os.path.dirname(SOURCE_ROOT)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from shared import ncf_ids  # noqa: E402
from shared.deal_library import DealBook, not_on_request  # noqa: E402
from training.ncf_sample import generate as gen  # noqa: E402

DEFAULT_COHORTS = gen.DEFAULT_COHORTS
DEFAULT_SEED = gen.DEFAULT_SEED
DEFAULT_SAMPLES = os.path.join(SOURCE_ROOT, "frontend-react", "public", "samples")
DEFAULT_CACHE = os.path.join(REPO_ROOT, "deployment", ".ncf-cache")

#: Files whose bytes are training inputs, relative to SOURCE_ROOT.
FINGERPRINTED_FILES = (
    "training/ncf_sample/generate.py",
    "training/ncf_sample/build.py",
    "shared/ncf_ids.py",
    "shared/ncf_model.py",
)

#: The generator package, repo-relative, recorded in the manifest and the registry.
GENERATOR_PATH = "source/training/ncf_sample"

MODEL_FILE = "model.onnx"
MANIFEST_FILE = "manifest.json"
INTERACTIONS_FILE = "interactions.csv"
GATE_FILE = "gate.json"
OUTPUT_FILES = (MODEL_FILE, MANIFEST_FILE, INTERACTIONS_FILE)

TRAIN_PARAMS: Mapping[str, Any] = {
    "epochs": 40,
    "batch_size": 256,
    "learning_rate": 0.002,
    "torch_seed": 20261007,
    "mf_dim": 64,
    "mlp_layer_sizes": [256, 128, 64],
}

#: Distance every gated score must keep from the threshold it is judged against.
GATE_MARGIN = 0.15
ACTIVATE_MIN = round(ncf_ids.ACTIVATE_THRESHOLD + GATE_MARGIN, 6)   # 0.75
SUPPRESS_MAX = round(ncf_ids.SUPPRESS_THRESHOLD - GATE_MARGIN, 6)   # 0.15
KEEP_MIN = round(ncf_ids.SUPPRESS_THRESHOLD + GATE_MARGIN, 6)       # 0.45
OTHER_MAX = round(ncf_ids.ACTIVATE_THRESHOLD - GATE_MARGIN, 6)      # 0.45


class GateFailed(RuntimeError):
    """The exported model does not reproduce every scenario target."""

    def __init__(self, report: "GateReport"):
        super().__init__(f"{sum(1 for s in report.scenarios if not s.passed)} scenario(s) failed the gate")
        self.report = report


# --------------------------------------------------------------------- inputs


@dataclass
class Inputs:
    cohorts: dict
    seed: dict
    #: sample file name -> parsed sample, or None when the file is missing or unreadable.
    samples: dict
    cohorts_path: str
    seed_path: str
    samples_dir: str


def load_inputs(cohorts_path: str = DEFAULT_COHORTS, seed_path: str = DEFAULT_SEED,
                samples_dir: str = DEFAULT_SAMPLES) -> Inputs:
    """Read the spec, the seed and every sample in ``samples_dir``."""
    cohorts = gen.load_json(cohorts_path)
    seed = gen.load_json(seed_path)
    samples: dict = {}
    names = sorted(n for n in os.listdir(samples_dir) if n.endswith(".json")) if os.path.isdir(samples_dir) else []
    for name in names:
        try:
            samples[name] = gen.load_json(os.path.join(samples_dir, name))
        except (OSError, ValueError):
            samples[name] = None
    for cohort in cohorts.get("cohorts") or ():
        if isinstance(cohort, Mapping) and isinstance(cohort.get("sample"), str):
            samples.setdefault(cohort["sample"], None)
    return Inputs(cohorts, seed, samples, cohorts_path, seed_path, samples_dir)


def _user_of(sample: Optional[Mapping]) -> Optional[str]:
    if not isinstance(sample, Mapping):
        return None
    user = (sample.get("bid_request") or {}).get("user") or {}
    uid = user.get("id") if isinstance(user, Mapping) else None
    return uid if isinstance(uid, str) else None


def _offered_ids(imp: Mapping) -> list[str]:
    deals = ((imp.get("pmp") or {}).get("deals") or []) if isinstance(imp.get("pmp"), Mapping) else []
    return [d.get("id", f"deal-{i}") for i, d in enumerate(deals) if isinstance(d, Mapping)]


def gated_samples(inputs: Inputs) -> list[tuple[dict, str]]:
    """(cohort, sample name) pairs the gate checks, in cohort then sample order.

    Each cohort's named sample, plus any other sample whose user is one of the
    cohort's scenario users.
    """
    pairs: list[tuple[dict, str]] = []
    for cohort in inputs.cohorts.get("cohorts") or ():
        users = set(cohort.get("scenario_users") or ())
        names = [cohort.get("sample")]
        names += [n for n, s in sorted(inputs.samples.items())
                  if n != cohort.get("sample") and _user_of(s) in users]
        pairs.extend((cohort, n) for n in names)
    return pairs


def _torch_major_minor() -> str:
    version = importlib.metadata.version("torch")
    return ".".join(version.split("+", 1)[0].split(".")[:2])


def fingerprint(inputs: Inputs, params: Mapping[str, Any] = TRAIN_PARAMS) -> str:
    """sha256 hex over the canonical JSON of the training inputs (D5). Excludes prices."""
    deals = [
        {k: d.get(k) for k in ("id", "seller", "adtypes", "inventory")}
        for d in inputs.seed.get("deals") or ()
    ]
    samples = {}
    for _, name in gated_samples(inputs):
        sample = inputs.samples.get(name)
        if not isinstance(sample, Mapping):
            samples[name] = {"missing": True}
            continue
        imps = (sample.get("bid_request") or {}).get("imp") or []
        samples[name] = {
            "user_id": _user_of(sample),
            "offered": [[imp.get("id"), _offered_ids(imp)] for imp in imps if isinstance(imp, Mapping)],
        }
    files = {}
    for rel in FINGERPRINTED_FILES:
        with open(os.path.join(SOURCE_ROOT, rel), "rb") as handle:
            files[rel] = hashlib.sha256(handle.read()).hexdigest()
    canonical = {
        "deals": deals,
        "cohorts": inputs.cohorts,
        "samples": samples,
        "train_params": dict(params),
        "files": files,
        "torch": _torch_major_minor(),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def model_label(fp: str) -> str:
    return f"{ncf_ids.SAMPLE_VERSION_PREFIX}{fp[:12]}"


# ---------------------------------------------------------------------- train


def _is_source_root(entry: str) -> bool:
    try:
        return os.path.realpath(entry or ".") == os.path.realpath(SOURCE_ROOT)
    except (OSError, ValueError):
        return False


def import_torch_dynamo_unshadowed() -> None:
    """Import torch._dynamo once with source/ off sys.path.

    Every torch optimizer imports torch._dynamo on first use, and torch._dynamo runs
    `import triton`. With source/ on sys.path that finds source/triton/ (this repo's
    Triton tooling, a namespace package) and fails with "module 'triton' has no
    attribute 'language'". Once imported, torch._dynamo stays in sys.modules.
    """
    if "torch._dynamo" in sys.modules:
        return
    saved_path = list(sys.path)
    saved = {n: m for n, m in sys.modules.items() if n == "triton" or n.startswith("triton.")}
    try:
        sys.path[:] = [p for p in sys.path if not _is_source_root(p)]
        for name in saved:
            del sys.modules[name]
        import torch._dynamo  # noqa: F401
    finally:
        sys.path[:] = saved_path
        sys.modules.update(saved)


def train(rows: Sequence[gen.Row], params: Mapping[str, Any] = TRAIN_PARAMS):
    """Train a NeuMF on ``rows``. Returns (model in eval mode, report)."""
    import torch
    import torch.nn as nn

    from shared.ncf_model import NeuMF

    import_torch_dynamo_unshadowed()

    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    # The ATen flag behind torch.use_deterministic_algorithms(True). The public call
    # also imports torch._inductor, whose `import triton` resolves to source/triton/
    # when source/ is on sys.path and raises AttributeError; inductor settings only
    # affect torch.compile, which this trainer does not use.
    torch._C._set_deterministic_algorithms(True)
    started = time.monotonic()
    try:
        torch.manual_seed(int(params["torch_seed"]))
        model = NeuMF(mf_dim=int(params["mf_dim"]), mlp_layer_sizes=list(params["mlp_layer_sizes"]))
        users = torch.tensor([r.user_bucket for r in rows], dtype=torch.int64)
        items = torch.tensor([r.item_bucket for r in rows], dtype=torch.int64)
        labels = torch.tensor([float(r.label) for r in rows], dtype=torch.float32)
        order = torch.Generator().manual_seed(int(params["torch_seed"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
        loss_fn = nn.BCELoss()
        size = int(params["batch_size"])
        n = len(rows)
        for _ in range(int(params["epochs"])):
            model.train()
            perm = torch.randperm(n, generator=order)
            for start in range(0, n, size):
                idx = perm[start:start + size]
                optimizer.zero_grad()
                loss = loss_fn(model(users[idx], items[idx]), labels[idx])
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = model(users, items)
            final_loss = float(loss_fn(scores, labels))
            scenario = torch.tensor([bool(r.is_scenario_user) for r in rows])
            scenario_loss = float(loss_fn(scores[scenario], labels[scenario])) if bool(scenario.any()) else None
    finally:
        torch._C._set_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)
    report = {
        **dict(params),
        "device": "cpu",
        "torch_version": torch.__version__,
        "final_bce": round(final_loss, 6),
        "scenario_user_bce": None if scenario_loss is None else round(scenario_loss, 6),
        "seconds": round(time.monotonic() - started, 1),
    }
    return model, report


def export(model, path: str) -> str:
    from shared.ncf_model import export_onnx

    return export_onnx(model, path)


# ----------------------------------------------------------------------- gate


@dataclass
class DealCheck:
    deal_id: str
    imp_id: str
    role: str          # activate | suppress | keep | candidate | offered
    score: Optional[float]
    rule: str          # e.g. ">= 0.75"
    ok: bool


@dataclass
class ScenarioResult:
    cohort: str
    sample: str
    user: Optional[str]
    passed: bool
    activated: list = field(default_factory=list)
    suppressed: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    failures: list = field(default_factory=list)


@dataclass
class GateReport:
    passed: bool
    margin: float
    thresholds: dict
    scenarios: list

    def as_dict(self) -> dict:
        return {
            "result": "pass" if self.passed else "fail",
            "margin": self.margin,
            "thresholds": self.thresholds,
            "scenarios": [asdict(s) for s in self.scenarios],
        }

    def summary(self) -> dict:
        return {
            "result": "pass" if self.passed else "fail",
            "margin": self.margin,
            "thresholds": self.thresholds,
            "scenarios": [
                {"cohort": s.cohort, "sample": s.sample, "user": s.user,
                 "result": "PASS" if s.passed else "FAIL",
                 "activated": s.activated, "suppressed": s.suppressed}
                for s in self.scenarios
            ],
        }


def onnx_scorer(model_path: str) -> Callable[[str, Sequence[str]], list[float]]:
    """score(user_id, deal_ids) through onnxruntime on the exported model."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def score(user_id: str, deal_ids: Sequence[str]) -> list[float]:
        if not deal_ids:
            return []
        u = np.array([ncf_ids.bucket(user_id, ncf_ids.NB_USERS)] * len(deal_ids), dtype=np.int64)
        d = np.array([ncf_ids.bucket(x, ncf_ids.NB_ITEMS) for x in deal_ids], dtype=np.int64)
        (out,) = session.run(["relevance_scores"], {"user_ids": u, "item_ids": d})
        return [float(v) for v in np.asarray(out).reshape(-1)]

    return score


def _check_scenario(cohort: Mapping, name: str, sample: Optional[Mapping], book: DealBook,
                    score: Callable[[str, Sequence[str]], list[float]]) -> ScenarioResult:
    cid = cohort.get("id", "?")
    result = ScenarioResult(cohort=cid, sample=name, user=_user_of(sample), passed=False)
    fail = result.failures.append
    if not isinstance(sample, Mapping) or not isinstance(sample.get("bid_request"), Mapping):
        fail(f"sample {name} is missing or unreadable")
        return result
    user = result.user
    if user not in set(cohort.get("scenario_users") or ()):
        fail(f"sample user {user!r} is not a scenario user of cohort {cid}")
        return result
    request = sample["bid_request"]
    activate = list(cohort.get("activate") or ())
    suppress = list(cohort.get("suppress") or ())
    keep = list(cohort.get("keep") or ())
    library = book.deals_for(request)

    offered_any: set[str] = set()
    candidate_any: set[str] = set()
    for imp in request.get("imp") or []:
        if not isinstance(imp, Mapping):
            continue
        imp_id = str(imp.get("id", ""))
        pmp = imp.get("pmp")
        offered = _offered_ids(imp)
        offered_dicts = ((pmp or {}).get("deals") or []) if isinstance(pmp, Mapping) else []
        candidates = [d.id for d in not_on_request(library, offered_dicts)] if pmp is not None else []
        if not offered and not candidates:
            continue
        offered_any.update(offered)
        candidate_any.update(candidates)
        ids = offered + candidates
        scores = score(user, ids)
        by_id = dict(zip(ids, scores))
        to_act, to_sup = ncf_ids.classify(ids, scores, ncf_ids.ACTIVATE_THRESHOLD, ncf_ids.SUPPRESS_THRESHOLD)
        to_act, to_sup = ncf_ids.restrict(offered, candidates, to_act, to_sup)
        result.activated.extend(f"{imp_id}:{d}" for d in to_act)
        result.suppressed.extend(f"{imp_id}:{d}" for d in to_sup)

        for deal in candidates:
            s = by_id[deal]
            if deal in activate:
                ok = s >= ACTIVATE_MIN
                result.checks.append(DealCheck(deal, imp_id, "activate", round(s, 6), f">= {ACTIVATE_MIN}", ok))
                if not ok:
                    fail(f"{imp_id}: activate target {deal} scores {s:.4f}, needs >= {ACTIVATE_MIN}")
            else:
                ok = s < OTHER_MAX
                result.checks.append(DealCheck(deal, imp_id, "candidate", round(s, 6), f"< {OTHER_MAX}", ok))
                if not ok:
                    fail(f"{imp_id}: library candidate {deal} is not an activate target and scores {s:.4f}, needs < {OTHER_MAX}")
        for deal in offered:
            s = by_id[deal]
            if deal in suppress:
                ok = s < SUPPRESS_MAX
                result.checks.append(DealCheck(deal, imp_id, "suppress", round(s, 6), f"< {SUPPRESS_MAX}", ok))
                if not ok:
                    fail(f"{imp_id}: suppress target {deal} scores {s:.4f}, needs < {SUPPRESS_MAX}")
            elif deal in keep:
                ok = s >= KEEP_MIN
                result.checks.append(DealCheck(deal, imp_id, "keep", round(s, 6), f">= {KEEP_MIN}", ok))
                if not ok:
                    fail(f"{imp_id}: keep target {deal} scores {s:.4f}, needs >= {KEEP_MIN}")
            else:
                result.checks.append(DealCheck(deal, imp_id, "offered", round(s, 6), "in keep or suppress", False))
                fail(f"{imp_id}: offered deal {deal} is in neither keep nor suppress of cohort {cid}")
        expected_act = [d for d in candidates if d in activate]
        expected_sup = [d for d in offered if d in suppress]
        if sorted(to_act) != sorted(expected_act):
            fail(f"{imp_id}: the decision rule activates {to_act}, the cohort expects {expected_act}")
        if sorted(to_sup) != sorted(expected_sup):
            fail(f"{imp_id}: the decision rule suppresses {to_sup}, the cohort expects {expected_sup}")

    for deal in activate:
        if deal not in candidate_any:
            fail(f"activate target {deal} is not a library candidate on any impression with a pmp")
    for deal in suppress:
        if deal not in offered_any:
            fail(f"suppress target {deal} is not offered on the request")
    for deal in keep:
        if deal not in offered_any:
            fail(f"keep target {deal} is not offered on the request")
    result.passed = not result.failures
    return result


def gate(model_path: str, inputs: Inputs,
         scorer: Optional[Callable[[str, Sequence[str]], list[float]]] = None) -> GateReport:
    """Check every gated scenario on the exported model (D6)."""
    score = scorer or onnx_scorer(model_path)
    book = DealBook(inputs.seed.get("deals") or ())
    pairs = gated_samples(inputs)
    scenarios = [_check_scenario(c, n, inputs.samples.get(n), book, score) for c, n in pairs]
    # A sample that cannot be read cannot be ruled out as a scenario sample.
    gated = {n for _, n in pairs}
    for name, sample in sorted(inputs.samples.items()):
        if sample is None and name not in gated:
            scenarios.append(ScenarioResult(cohort="-", sample=name, user=None, passed=False,
                                            failures=[f"sample {name} is unreadable"]))
    if not scenarios:
        scenarios.append(ScenarioResult(cohort="-", sample="-", user=None, passed=False,
                                        failures=["the cohort spec gates no scenario"]))
    return GateReport(
        passed=all(s.passed for s in scenarios),
        margin=GATE_MARGIN,
        thresholds={"activate": ncf_ids.ACTIVATE_THRESHOLD, "suppress": ncf_ids.SUPPRESS_THRESHOLD},
        scenarios=scenarios,
    )


def format_table(report: GateReport) -> str:
    """The per-scenario gate table, then each failure."""
    head = ("cohort", "sample", "user", "result", "activated", "suppressed", "scores")
    lines = []
    rows = []
    for s in report.scenarios:
        scores = " ".join(f"{c.deal_id}={c.score:.3f}({c.role})" for c in s.checks if c.score is not None)
        rows.append((s.cohort, s.sample, s.user or "-", "PASS" if s.passed else "FAIL",
                     ",".join(s.activated) or "-", ",".join(s.suppressed) or "-", scores or "-"))
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head) - 1)]
    for r in [head, *rows]:
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(r[:-1], widths)) + "  " + str(r[-1]))
    failures = [(s.cohort, s.sample, f) for s in report.scenarios for f in s.failures]
    if failures:
        lines.append("")
        lines.append("Gate failures:")
        lines.extend(f"  {c} ({n}): {f}" for c, n, f in failures)
    verdict = "PASS" if report.passed else "FAIL"
    lines.append("")
    lines.append(
        f"Deal Scorer gate {verdict}: margin {report.margin} "
        f"(activate >= {ACTIVATE_MIN}, suppress < {SUPPRESS_MAX}, keep >= {KEEP_MIN}, other candidates < {OTHER_MAX})"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------- manifest


def manifest(fp: str, spec: Mapping, train_report: Mapping, generator_stats: Mapping,
             report: GateReport, trained_at: str) -> dict:
    from shared.ncf_model import INPUT_NAMES, OUTPUT_NAME

    return {
        "model_type": "ncf_deal_manager",
        "model_version": model_label(fp),
        "fingerprint": fp,
        "fingerprint12": fp[:12],
        "training_data": "sample",
        "target": "implicit-engagement-sample",
        "description": "Deal Scorer trained on sample interactions generated from the deal book, not observed outcomes",
        "producer": "training/ncf_sample/build.py",
        "generator_seed": spec["generator_seed"],
        "generator": {
            "path": GENERATOR_PATH,
            "format": spec.get("format"),
            "members_per_cohort": spec.get("members_per_cohort"),
            "noise_rate": spec.get("noise_rate"),
            **dict(generator_stats),
        },
        "nb_users": ncf_ids.NB_USERS,
        "nb_items": ncf_ids.NB_ITEMS,
        "triton_input_names": list(INPUT_NAMES),
        "triton_output_names": [OUTPUT_NAME],
        "thresholds": {"activate": ncf_ids.ACTIVATE_THRESHOLD, "suppress": ncf_ids.SUPPRESS_THRESHOLD},
        "train": dict(train_report),
        "gate": report.summary(),
        "trained_at": trained_at,
    }


# ---------------------------------------------------------------------- build


@dataclass
class BuildResult:
    fingerprint: str
    model_version: str
    cache_hit: bool
    report: GateReport
    manifest: dict
    output_dir: Optional[str]


def _write_json(path: str, value: Any) -> None:
    """Write through a temp file and a rename, so an interrupted write leaves the old file."""
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _cached_manifest(entry: str, fp: str) -> Optional[dict]:
    """The cache entry's manifest when it parses and records ``fp``, else None."""
    try:
        cached = gen.load_json(os.path.join(entry, MANIFEST_FILE))
    except (OSError, ValueError):
        return None
    return cached if isinstance(cached, dict) and cached.get("fingerprint") == fp else None


def build(output_dir: Optional[str], cohorts_path: str = DEFAULT_COHORTS, seed_path: str = DEFAULT_SEED,
          samples_dir: str = DEFAULT_SAMPLES, cache_dir: Optional[str] = DEFAULT_CACHE,
          log: Callable[[str], None] = print) -> BuildResult:
    """Fingerprint, train or reuse, gate, and write the outputs. Raises GateFailed."""
    inputs = load_inputs(cohorts_path, seed_path, samples_dir)
    spec = inputs.cohorts
    deal_ids = gen.seed_deal_ids(inputs.seed)
    gen.validate(spec, deal_ids)
    fp = fingerprint(inputs)
    label = model_label(fp)
    log(f"ncf sample build: fingerprint {fp} (model version {label}), generator seed {spec['generator_seed']}")

    entry = os.path.join(cache_dir, fp) if cache_dir else None
    cache_hit = bool(entry) and all(os.path.isfile(os.path.join(entry, f)) for f in OUTPUT_FILES)
    cached = _cached_manifest(entry, fp) if cache_hit else None
    if cache_hit and cached is None:
        log(f"ncf sample build: cache entry {entry} has an unreadable manifest or one for another build; "
            "discarding it and rebuilding")
        shutil.rmtree(entry, ignore_errors=True)
        cache_hit = False
    work = None
    if cache_hit:
        log(f"ncf sample build: cache hit {entry}; re-running the gate on the cached model")
        source = entry
        report = gate(os.path.join(entry, MODEL_FILE), inputs)
        record = {**cached, "gate": report.summary()}
    else:
        log("ncf sample build: cache miss; generating interactions and training on CPU")
        rows = gen.generate(spec, deal_ids)
        # Built beside the cache entry, so a passing build moves in with one rename.
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        work = tempfile.mkdtemp(prefix=f".build-{fp[:12]}-", dir=cache_dir or None)
        source = work
        with open(os.path.join(work, INTERACTIONS_FILE), "wb") as handle:
            handle.write(gen.to_csv_bytes(rows))
        model, train_report = train(rows)
        log(f"ncf sample build: trained {len(rows)} rows in {train_report['seconds']}s, final BCE {train_report['final_bce']}")
        export(model, os.path.join(work, MODEL_FILE))
        report = gate(os.path.join(work, MODEL_FILE), inputs)
        record = manifest(fp, spec, train_report, gen.stats(rows, spec), report,
                          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        _write_json(os.path.join(work, MANIFEST_FILE), record)
    log(format_table(report))
    try:
        if not report.passed:
            raise GateFailed(report)
        _write_json(os.path.join(source, GATE_FILE), report.as_dict())
        if cache_hit:
            _write_json(os.path.join(source, MANIFEST_FILE), record)
        elif entry:
            if os.path.isdir(entry):
                shutil.rmtree(entry)
            os.replace(work, entry)
            work = None
            source = entry
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for name in OUTPUT_FILES:
                shutil.copyfile(os.path.join(source, name), os.path.join(output_dir, name))
            log(f"ncf sample build: wrote {', '.join(OUTPUT_FILES)} to {output_dir}")
    finally:
        if work and os.path.isdir(work) and work != entry:
            shutil.rmtree(work, ignore_errors=True)
    return BuildResult(fp, label, cache_hit, report, record, output_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train, export and gate the Deal Scorer on sample interactions.")
    parser.add_argument("--output-dir", help="where to write model.onnx, manifest.json and interactions.csv")
    parser.add_argument("--fingerprint-only", action="store_true", help="print the training fingerprint and exit")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, help="build cache (default deployment/.ncf-cache)")
    parser.add_argument("--cohorts", default=DEFAULT_COHORTS, help="cohort spec (default cohorts.json)")
    parser.add_argument("--seed-file", default=DEFAULT_SEED, help="deals seed (default deployment/seed/deals_and_campaigns.json)")
    parser.add_argument("--samples-dir", default=DEFAULT_SAMPLES, help="scenario samples directory")
    args = parser.parse_args(argv)
    if args.fingerprint_only:
        print(fingerprint(load_inputs(args.cohorts, args.seed_file, args.samples_dir)))
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required unless --fingerprint-only is given")
    try:
        build(args.output_dir, args.cohorts, args.seed_file, args.samples_dir, args.cache_dir)
    except GateFailed as exc:
        print(f"ncf sample build: {exc}; no model written", file=sys.stderr)
        return 1
    except gen.SpecError as exc:
        print(f"ncf sample build: cohort spec error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
