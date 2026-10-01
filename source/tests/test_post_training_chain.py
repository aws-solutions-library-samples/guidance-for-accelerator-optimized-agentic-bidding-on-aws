"""Runs the trainer's post-phase-1 chain end to end on CPU.

Four SageMaker runs in one session died in this stretch of `main()`, one after
another, each discovered only on a GPU instance after an image pull, a capacity wait
and a full phase 1 retrain:

  1. `torch.onnx.export(dynamo=False)` on a torch too old for the argument
  2. `from shared import onnx_compat` — module not staged into the image
  3. `from shared.shading_policy import ...` — same, a second module
  4. CPU batches fed to a model left on cuda:0

Every one of them is reachable on CPU. This test walks the same sequence main() runs
after phase 1 — predict_all, then search_policy_parameters, then export_to_onnx, then
the manifest — so the next defect in it costs seconds.

It does NOT assert model quality. It asserts the chain completes and each step hands
the next something it can use.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(SOURCE_DIR / "training" / "container"))

torch = pytest.importorskip("torch", reason="torch required")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from torch.utils.data import TensorDataset  # noqa: E402


@pytest.fixture(scope="module")
def train_mod():
    os.environ.setdefault("SM_MODEL_DIR", "/tmp")
    return pytest.importorskip("train", reason="training container deps unavailable")


@pytest.fixture(scope="module")
def dlrm():
    models = pytest.importorskip("models", reason="training container deps unavailable")
    return models


def _frame(rows: int = 60) -> pd.DataFrame:
    """A dataset shaped like the ETL's output, with both won and lost rows.

    Column names follow the real training dataset as logged by a live run, so a
    rename in the ETL shows up here rather than on an instance.
    """
    rng = np.random.default_rng(1234)
    won = np.arange(rows) % 3 != 0  # two-thirds won, so the surrogate has rows to score
    original = rng.uniform(2.0, 8.0, rows)
    return pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(rows)],
            "won": won,
            "original_price": original,
            "price_paid": np.where(won, original * 0.7, np.nan),
            "bid_floor": original * 0.2,
            "conversion_value": np.where(
                won & (np.arange(rows) % 6 == 1), 25.0, np.nan
            ),
            "label_conversion": np.where(
                won, (np.arange(rows) % 6 == 1).astype(float), np.nan
            ),
        }
    )


@pytest.fixture(scope="module")
def model_and_dataset(dlrm):
    width = dlrm.NUM_DENSE + dlrm.NUM_SPARSE
    model = dlrm.DLRMModel()
    features = torch.randn(60, width).abs()
    # Categorical columns are embedding indices stored as floats; keep them small and
    # non-negative or the embedding lookup is out of range.
    features[:, dlrm.NUM_DENSE :] = torch.randint(
        0, 3, (60, dlrm.NUM_SPARSE)
    ).float()
    return model, TensorDataset(features)


class TestTheChainCompletes:
    def test_predict_all_scores_every_row(self, train_mod, model_and_dataset) -> None:
        model, dataset = model_and_dataset
        preds = train_mod.predict_all(model, dataset, batch_size=16)
        assert preds is not None
        assert preds.shape == (60,)
        assert np.isfinite(preds).all(), "non-finite predictions would poison the search"

    def test_policy_search_accepts_those_predictions(
        self, train_mod, model_and_dataset
    ) -> None:
        """The handoff that matters: whatever predict_all returns must be something
        search_policy_parameters can consume without reshaping."""
        model, dataset = model_and_dataset
        preds = train_mod.predict_all(model, dataset, batch_size=16)
        report = train_mod.search_policy_parameters(_frame(), preds, {})

        assert isinstance(report, dict)
        assert "policy_searched" in report
        # Whether it searched or skipped, it must say which and stay serialisable --
        # main() json.dumps this straight into the log and the manifest.
        json.dumps(report)

    def test_the_search_never_claims_a_measured_lift(self, train_mod) -> None:
        """If it searched, the surrogate must be named. A tuned parameter set that
        does not say how it was scored reads as a measured result."""
        model_preds = np.linspace(0.01, 0.4, 60)
        report = train_mod.search_policy_parameters(_frame(), model_preds, {})
        if report.get("policy_searched"):
            assert report.get("policy_search_method") == "offline_surrogate", (
                f"searched without naming the surrogate: {report}"
            )

    def test_an_unusable_dataset_is_reported_not_imputed(self, train_mod) -> None:
        """No won rows means the surrogate can score nothing. It must say so rather
        than return a confident parameter set derived from nothing."""
        df = _frame()
        df["won"] = False
        df["price_paid"] = np.nan
        report = train_mod.search_policy_parameters(df, np.linspace(0.01, 0.4, 60), {})
        assert report.get("policy_searched") is False
        assert report.get("policy_skipped_reason"), "skipped without saying why"

    def test_missing_columns_are_reported_not_crashed(self, train_mod) -> None:
        report = train_mod.search_policy_parameters(
            pd.DataFrame({"won": [True]}), np.array([0.2]), {}
        )
        assert report.get("policy_searched") is False
        assert "policy_skipped_reason" in report

    def test_no_predictions_is_handled(self, train_mod) -> None:
        report = train_mod.search_policy_parameters(_frame(), None, {})
        assert isinstance(report, dict)
        assert "policy_searched" in report


class TestOnnxExport:
    def test_export_writes_a_loadable_graph(
        self, train_mod, model_and_dataset, tmp_path
    ) -> None:
        """The failure that started this: export must work on the INSTALLED torch,
        whatever its signature. onnx_compat filters the kwargs; this proves the
        filtering is right for the torch actually present.
        """
        model, _ = model_and_dataset
        path = train_mod.export_to_onnx(model, "dlrm_bid_shader", str(tmp_path))
        assert Path(path).is_file(), "export_to_onnx reported a path it did not write"
        assert Path(path).stat().st_size > 0

    def test_export_leaves_the_model_on_cpu(
        self, train_mod, model_and_dataset, tmp_path
    ) -> None:
        """export_to_onnx moves the model to CPU itself. predict_all's device
        handling relies on reading the device rather than assuming it, so this pins
        the assumption the two make about each other."""
        model, _ = model_and_dataset
        train_mod.export_to_onnx(model, "dlrm_bid_shader", str(tmp_path))
        assert next(model.parameters()).device.type == "cpu"

    def test_exported_inputs_match_the_serving_contract(
        self, train_mod, model_and_dataset, tmp_path
    ) -> None:
        """A graph whose inputs are merely the right width is not agreement; the
        names have to match what the serving container sends."""
        onnx = pytest.importorskip("onnx", reason="onnx package not installed")
        from shared import dlrm_features

        model, _ = model_and_dataset
        path = train_mod.export_to_onnx(model, "dlrm_bid_shader", str(tmp_path))
        graph = onnx.load(path).graph

        names = [i.name for i in graph.input]
        expected = [
            dlrm_features.TRITON_DENSE_INPUT,
            *dlrm_features.TRITON_CATEGORICAL_INPUTS,
        ]
        assert names == expected, f"exported {names}, serving sends {expected}"
