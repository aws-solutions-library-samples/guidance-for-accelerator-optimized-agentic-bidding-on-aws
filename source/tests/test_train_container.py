"""Unit tests for the NeMo-RL training container entrypoint (train.py).

Covers real production bugs found while diagnosing live failed training
jobs (dlrm-bid-shader-1787311543-72232191, base version
nvd-artf-dlrm-bid-shader/1; dlrm-bid-shader-1787398867-e572c442) and their
predecessors:

1. load_training_data() used a non-recursive glob, but the Glue ETL job
   writes Hive-style partitioned Parquet (window_start=.../window_end=.../
   *.parquet), which SageMaker preserves under the training channel.
2. Feature/outcome column selection referenced columns that never exist in
   the real Glue ETL output ("feature_*", "win", "revenue") instead of the
   actual columns ("bid_floor", "won", "conversion_value", etc.).
3. export_to_onnx()'s DLRM dummy input was width 4, but DLRMModel.forward()
   requires width NUM_DENSE+NUM_SPARSE=7.
4. load_hyperparameters()'s JSON-file branch returned the caller's raw
   HyperParameters verbatim with no defaults applied -- a real on-demand
   "Train from load test" job (which only sends model_type/
   base_model_version/window_days/cadence_hours/triggered_by) crashed with
   KeyError: 'supervised_epochs' because that key was never sent and never
   defaulted.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "training", "container")
)

import train as train_module  # noqa: E402
from train import (  # noqa: E402
    _DLRM_DENSE_COLUMNS,
    _DLRM_SPARSE_COLUMNS,
    _HP_DEFAULTS,
    _hash_to_idx,
    build_dlrm_features,
    build_features,
    load_hyperparameters,
    load_training_data,
)


def _write_parquet(path, n_rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"feature_0": range(n_rows), "label": [0] * n_rows})
    df.to_parquet(path)
    return df


class TestLoadTrainingData:
    def test_raises_when_no_parquet_files_anywhere(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .parquet files found"):
            load_training_data(str(tmp_path), "dlrm_bid_shader")

    def test_loads_flat_parquet_files(self, tmp_path):
        """Files directly under data_dir (no partitioning) must still load."""
        _write_parquet(tmp_path / "part-00000.snappy.parquet", n_rows=2)
        _write_parquet(tmp_path / "part-00001.snappy.parquet", n_rows=3)

        df = load_training_data(str(tmp_path), "dlrm_bid_shader")

        assert len(df) == 5

    def test_loads_hive_partitioned_parquet_files(self, tmp_path):
        """Reproduces the real SageMaker layout: Glue ETL writes
        window_start=.../window_end=.../*.parquet, and SageMaker preserves
        that structure under the training channel. This is the exact shape
        that caused FileNotFoundError in production
        (dlrm-bid-shader-1787311543-72232191)."""
        _write_parquet(
            tmp_path
            / "window_start=2026-08-20T18:03:34.294592+00:00"
            / "window_end=2026-08-21T00:03:34.294592+00:00"
            / "part-00000-f223c126.snappy.parquet",
            n_rows=4,
        )
        _write_parquet(
            tmp_path
            / "window_start=2026-08-20T12:03:39.724147+00:00"
            / "window_end=2026-08-20T18:03:39.724147+00:00"
            / "part-00000-26dd8c83.snappy.parquet",
            n_rows=2,
        )

        df = load_training_data(str(tmp_path), "dlrm_bid_shader")

        assert len(df) == 6

    def test_loads_mixed_flat_and_partitioned(self, tmp_path):
        _write_parquet(tmp_path / "flat.snappy.parquet", n_rows=1)
        _write_parquet(
            tmp_path / "window_start=x" / "window_end=y" / "part-0.snappy.parquet",
            n_rows=1,
        )

        df = load_training_data(str(tmp_path), "dlrm_bid_shader")

        assert len(df) == 2


def _make_dlrm_df(n_rows=5):
    """A DataFrame shaped like real Glue ETL output for dlrm_bid_shader."""
    return pd.DataFrame({
        "bid_floor": [2.0 + i * 0.1 for i in range(n_rows)],
        "hour_of_day": [i % 24 for i in range(n_rows)],
        "shade_factor_used": [0.65] * n_rows,
        "conversion_value_estimate_used": [12.0] * n_rows,
        "device_type": ["mobile", "desktop", "tablet", "mobile", "desktop"][:n_rows],
        "site_domain": ["espn.com", "cnn.com", "espn.com", "nyt.com", "cnn.com"][:n_rows],
        "win_rate_bucket": [3, 5, 2, 3, 5][:n_rows],
        "label": [1, 0, 1, 0, 1][:n_rows],
        "won": [True, False, True, False, True][:n_rows],
        "price_paid": [1.8, None, 2.1, None, 1.9][:n_rows],
        "conversion_value": [10.0, None, None, None, 8.0][:n_rows],
    })


class TestBuildDlrmFeatures:
    def test_output_shape_is_dense_plus_sparse(self):
        """DLRMModel.forward() slices [:NUM_DENSE] and
        [NUM_DENSE:NUM_DENSE+NUM_SPARSE] from a single tensor — width must
        be exactly len(dense_columns) + len(sparse_columns) = 7."""
        df = _make_dlrm_df()

        features = build_dlrm_features(df)

        assert features.shape == (5, len(_DLRM_DENSE_COLUMNS) + len(_DLRM_SPARSE_COLUMNS))
        assert features.shape[1] == 7

    def test_no_feature_columns_are_leaked_targets(self):
        """shaded_price/shade_ratio (the model's own past decision) and roi
        (computed from post-bid outcomes) must never be selected as inputs."""
        leaked = {"shaded_price", "shade_ratio", "roi", "won", "price_paid", "conversion_value", "label"}
        assert leaked.isdisjoint(_DLRM_DENSE_COLUMNS)
        assert leaked.isdisjoint(_DLRM_SPARSE_COLUMNS)

    def test_hour_of_day_normalized_to_unit_interval(self):
        df = _make_dlrm_df(n_rows=1)
        df.loc[0, "hour_of_day"] = 12

        features = build_dlrm_features(df)

        hour_col_idx = _DLRM_DENSE_COLUMNS.index("hour_of_day")
        assert features[0, hour_col_idx].item() == pytest.approx(0.5)

    def test_sparse_columns_are_hashed_consistently(self):
        """Same categorical value must hash to the same index whether it
        appears in row 0 or row N (i.e. hashing is a pure function of the
        string, not row-dependent)."""
        df = _make_dlrm_df()

        features = build_dlrm_features(df)

        device_col_idx = len(_DLRM_DENSE_COLUMNS) + _DLRM_SPARSE_COLUMNS.index("device_type")
        # Rows 0 and 3 are both "mobile" in _make_dlrm_df.
        assert features[0, device_col_idx].item() == features[3, device_col_idx].item()
        assert features[0, device_col_idx].item() == float(_hash_to_idx("mobile"))

    def test_forward_pass_through_real_dlrm_model_succeeds(self):
        """End-to-end shape check against the actual DLRMModel used in
        training (models/__init__.py) — not just a shape assertion, but a
        real forward pass, since this is exactly where the previous width-4
        dummy input would have failed."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training", "container"))
        from models import DLRMModel

        df = _make_dlrm_df()
        features = build_dlrm_features(df)
        model = DLRMModel()

        output = model(features)

        assert output.shape == (5,)


class TestBuildFeatures:
    def test_dispatches_to_dlrm_builder(self):
        df = _make_dlrm_df()

        features = build_features(df, "dlrm_bid_shader")

        assert features.shape == (5, 7)

    def test_ncf_raises_clear_error_instead_of_silent_wrong_shape(self):
        """ncf_deal_manager has no deal_id column to build item_ids from —
        must fail loudly, not silently produce an empty/wrong-shaped tensor."""
        df = _make_dlrm_df()

        with pytest.raises(ValueError, match="ncf_deal_manager"):
            build_features(df, "ncf_deal_manager")


class TestOutcomeColumnSelection:
    """main()'s outcome-column selection for the RL phase must match the
    real dataframe columns (won/price_paid/conversion_value), not the
    previous nonexistent (win/price_paid/revenue) names."""

    def test_real_outcome_columns_present_in_glue_etl_schema(self):
        df = _make_dlrm_df()
        from train import _OUTCOME_COLUMNS

        assert all(c in df.columns for c in _OUTCOME_COLUMNS)

    def test_previous_broken_columns_absent(self):
        df = _make_dlrm_df()

        assert "win" not in df.columns
        assert "revenue" not in df.columns


class TestExportToOnnxServingSignature:
    """export_to_onnx() for DLRM must emit the SERVING signature so a retrained
    artifact is loadable by dlrm_bid_shader_stable/_canary and compilable by the
    Model Optimizer: four named inputs (dense_features FP32 [b,4];
    sparse_user/domain/device INT64 [b]) and a sigmoid'd ctr_prediction output —
    matching source/triton/export_models.py::export_dlrm. (Previously it exported
    a single width-7 ``features`` input named ``output``, which no served config
    accepts.)
    """

    def test_dlrm_export_passes_four_named_serving_inputs(self, tmp_path, monkeypatch):
        import train as train_module
        from models import DLRMModel, NUM_DENSE

        captured = {}

        def _fake_onnx_export(model, args, output_path, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            Path(output_path).write_bytes(b"fake-onnx-for-signature-test")

        monkeypatch.setattr(train_module.torch.onnx, "export", _fake_onnx_export)

        onnx_path = train_module.export_to_onnx(DLRMModel(), "dlrm_bid_shader", str(tmp_path))

        assert os.path.exists(onnx_path)
        args = captured["args"]
        # Four positional inputs, not a single combined tensor.
        assert isinstance(args, tuple) and len(args) == 4
        dense, s_user, s_domain, s_device = args
        assert tuple(dense.shape) == (1, NUM_DENSE)
        assert dense.dtype == torch.float32
        for s in (s_user, s_domain, s_device):
            assert tuple(s.shape) == (1,)
            assert s.dtype == torch.int64
        assert captured["kwargs"]["input_names"] == [
            "dense_features", "sparse_user", "sparse_domain", "sparse_device",
        ]
        assert captured["kwargs"]["output_names"] == ["ctr_prediction"]

    def test_dlrm_export_real_onnx_graph_matches_serving_contract(self, tmp_path):
        """Real ONNX export (tracing exporter, dynamo=False — no triton path
        collision) and inspect the graph: the retrained model must expose the
        exact input names/dtypes and output name the Triton config declares."""
        import onnx

        import train as train_module
        from models import DLRMModel

        onnx_path = train_module.export_to_onnx(DLRMModel(), "dlrm_bid_shader", str(tmp_path))
        graph = onnx.load(onnx_path).graph

        input_names = [i.name for i in graph.input]
        assert input_names == [
            "dense_features", "sparse_user", "sparse_domain", "sparse_device",
        ]
        assert [o.name for o in graph.output] == ["ctr_prediction"]

        # elem_type: 1 = FLOAT, 7 = INT64 (onnx.TensorProto).
        by_name = {i.name: i for i in graph.input}
        assert by_name["dense_features"].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for name in ("sparse_user", "sparse_domain", "sparse_device"):
            assert by_name[name].type.tensor_type.elem_type == onnx.TensorProto.INT64

    def test_dlrm_forward_is_seeded_deterministic(self):
        """A seeded DLRMModel produces identical outputs across builds — the
        exported engine is reproducible, not a moving target."""
        from models import DLRMModel

        x = torch.tensor([[0.5, 0.4, 0.1, 1.0, 3.0, 7.0, 2.0]], dtype=torch.float32)

        torch.manual_seed(1234)
        out_a = DLRMModel().eval()(x)
        torch.manual_seed(1234)
        out_b = DLRMModel().eval()(x)

        assert torch.equal(out_a, out_b)
        assert out_a.shape == (1,)


class TestLoadHyperparametersDefaults:
    """load_hyperparameters()'s JSON-file branch previously returned the
    caller's raw HyperParameters verbatim with no defaults merged in.
    SageMaker always writes SM_HP_FILE whenever HyperParameters is
    non-empty, and none of this repo's three real CreateTrainingJob callers
    (training_trigger.py's on-demand path, the scheduled
    RetrainingTriggerFunction Lambda, TrainingPipeline) send a complete
    hyperparameter set — reproduces the exact live failure
    (dlrm-bid-shader-1787398867-e572c442: KeyError: 'supervised_epochs')."""

    def test_partial_hyperparameters_file_gets_defaults_merged_in(self, tmp_path, monkeypatch):
        """Reproduces training_trigger.py's real payload: only 5 keys, none
        of the 4 phase-1/phase-2 keys train.py's supervised_train()/
        rl_finetune() require."""
        hp_file = tmp_path / "hyperparameters.json"
        hp_file.write_text(json.dumps({
            "model_type": "dlrm_bid_shader",
            "base_model_version": "arn:aws:sagemaker:us-east-1:960328030835:model-package/nv5-artf-dlrm-bid-shader/1",
            "window_days": "7",
            "cadence_hours": "6.0",
            "triggered_by": "governance_ui_on_demand",
        }))
        monkeypatch.setattr(train_module, "SM_HP_FILE", str(hp_file))

        hp = load_hyperparameters()

        # The exact key that crashed the real job must be present and usable.
        assert hp["supervised_epochs"] == _HP_DEFAULTS["supervised_epochs"]
        assert hp["learning_rate"] == _HP_DEFAULTS["learning_rate"]
        assert hp["rl_epochs"] == _HP_DEFAULTS["rl_epochs"]
        assert hp["reward_function"] == _HP_DEFAULTS["reward_function"]
        assert hp["rl_learning_rate"] == _HP_DEFAULTS["rl_learning_rate"]
        assert hp["batch_size"] == _HP_DEFAULTS["batch_size"]
        assert hp["validation_split"] == _HP_DEFAULTS["validation_split"]
        assert hp["use_reinforcement_learning"] == _HP_DEFAULTS["use_reinforcement_learning"]
        # Caller-supplied values are preserved, not overwritten by defaults.
        assert hp["model_type"] == "dlrm_bid_shader"
        assert hp["window_days"] == 7
        assert hp["cadence_hours"] == 6.0
        assert hp["triggered_by"] == "governance_ui_on_demand"

    def test_caller_supplied_value_overrides_default(self, tmp_path, monkeypatch):
        hp_file = tmp_path / "hyperparameters.json"
        hp_file.write_text(json.dumps({
            "model_type": "dlrm_bid_shader",
            "supervised_epochs": "20",
        }))
        monkeypatch.setattr(train_module, "SM_HP_FILE", str(hp_file))

        hp = load_hyperparameters()

        assert hp["supervised_epochs"] == 20

    def test_full_hyperparameters_file_unaffected_by_defaults(self, tmp_path, monkeypatch):
        """A caller sending every key (e.g. TrainingPipeline) must get its
        own values back unchanged, not silently overridden."""
        full_payload = {
            "model_type": "dlrm_bid_shader",
            "base_model_version": "v3",
            "validation_split": "0.2",
            "use_reinforcement_learning": "false",
            "reward_function": "ctr",
            "rl_learning_rate": "5e-5",
            "rl_epochs": "8",
            "supervised_epochs": "15",
            "batch_size": "128",
            "learning_rate": "2e-3",
            "window_days": "7",
            "cadence_hours": "6.0",
        }
        hp_file = tmp_path / "hyperparameters.json"
        hp_file.write_text(json.dumps(full_payload))
        monkeypatch.setattr(train_module, "SM_HP_FILE", str(hp_file))

        hp = load_hyperparameters()

        assert hp["validation_split"] == 0.2
        assert hp["use_reinforcement_learning"] is False
        assert hp["reward_function"] == "ctr"
        assert hp["rl_learning_rate"] == 5e-5
        assert hp["rl_epochs"] == 8
        assert hp["supervised_epochs"] == 15
        assert hp["batch_size"] == 128
        assert hp["learning_rate"] == 2e-3

    def test_env_var_fallback_path_unaffected(self, tmp_path, monkeypatch):
        """When no hyperparameters.json exists at all (non-SageMaker local
        run), the env-var fallback branch must still return every key with
        its own defaults, matching pre-fix behavior exactly."""
        missing_file = tmp_path / "does-not-exist.json"
        monkeypatch.setattr(train_module, "SM_HP_FILE", str(missing_file))
        monkeypatch.delenv("SM_HP_SUPERVISED_EPOCHS", raising=False)

        hp = load_hyperparameters()

        assert hp["supervised_epochs"] == _HP_DEFAULTS["supervised_epochs"]
        assert hp["model_type"] == _HP_DEFAULTS["model_type"]
