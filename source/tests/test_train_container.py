"""Unit tests for the NeMo-RL training container entrypoint (train.py).

Covers failure modes of live SageMaker training jobs:

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
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

# APPENDED, not inserted at position 0. The container directory now holds a
# staged `shared/` package (see source/training/stage_shared.sh) carrying only
# dlrm_features.py. Prepending would make `shared` resolve there for the rest of
# the pytest session, and every later `shared.feedback_models` import in the
# suite would fail. Appending leaves source/shared/ -- the real package, which
# the staged copy is copied from -- as the one that resolves, while `train` and
# `models` still import because they exist nowhere else.
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "training", "container")
)

from shared import dlrm_features  # noqa: E402

import train as train_module  # noqa: E402
from train import (  # noqa: E402
    _HP_DEFAULTS,
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
    """A DataFrame shaped like real Glue ETL output for dlrm_bid_shader.

    Carries the columns the feature spec reads AND several it deliberately does
    not (shade_factor_used, conversion_value_estimate_used, win_rate_bucket),
    so the leakage assertions below have something to be true about.
    """
    return pd.DataFrame({
        # Read by the spec.
        "bid_floor": [2.0 + i * 0.1 for i in range(n_rows)],
        "hour_of_day": [i % 24 for i in range(n_rows)],
        "day_of_week": [0, 5, 6, 2, 3][:n_rows],
        "has_video": [False, True, False, True, True][:n_rows],
        "site_domain": ["espn.com", "cnn.com", "espn.com", "nyt.com", "cnn.com"][:n_rows],
        "device_type": [2, 1, 5, 2, 1][:n_rows],
        "geo_country": ["USA", "CAN", "USA", "GBR", "CAN"][:n_rows],
        # Present in the ETL output, held out of the vector by design.
        "shade_factor_used": [0.65] * n_rows,
        "conversion_value_estimate_used": [12.0] * n_rows,
        "win_rate_bucket": [3, 5, 2, 3, 5][:n_rows],
        # Label and outcomes.
        "label": [1, 0, 1, 0, 1][:n_rows],
        "won": [True, False, True, False, True][:n_rows],
        "price_paid": [1.8, None, 2.1, None, 1.9][:n_rows],
        "conversion_value": [10.0, None, None, None, 8.0][:n_rows],
    })


class TestBuildDlrmFeatures:
    """The trainer's vector comes from shared/dlrm_features.py.

    These assertions are stated against the shared spec rather than against
    lists in train.py, because lists in train.py were the defect: they agreed
    with the serving container on width and disagreed on meaning, which nothing
    detected. Cross-side equality itself is asserted in
    tests/test_dlrm_feature_parity.py.
    """

    def test_output_shape_is_dense_plus_categorical(self):
        """DLRMModel.forward() slices [:NUM_DENSE] and
        [NUM_DENSE:NUM_DENSE+NUM_SPARSE] from a single tensor — width must be
        exactly the spec's FEATURE_WIDTH."""
        df = _make_dlrm_df()

        features = build_dlrm_features(df)

        assert features.shape == (5, dlrm_features.FEATURE_WIDTH)
        assert features.shape[1] == 7

    def test_no_feature_columns_are_leaked_targets(self):
        """The shader's own parameters and the bid's consequences stay out.

        shaded_price/shade_ratio/roi are outcomes of the bid; shade_factor_used
        and conversion_value_estimate_used are the policy that produced the
        row, and a prediction conditioned on them cannot be used to evaluate a
        change to that policy.
        """
        leaked = {
            "shaded_price",
            "shade_ratio",
            "roi",
            "won",
            "price_paid",
            "conversion_value",
            "label",
            "shade_factor_used",
            "conversion_value_estimate_used",
        }
        assert leaked.isdisjoint(dlrm_features.DENSE_COLUMNS)
        assert leaked.isdisjoint(dlrm_features.CATEGORICAL_COLUMNS)

    def test_hour_of_day_normalized_to_unit_interval(self):
        df = _make_dlrm_df(n_rows=1)
        df.loc[0, "hour_of_day"] = 12

        features = build_dlrm_features(df)

        hour_col_idx = dlrm_features.DENSE_COLUMNS.index("hour_norm")
        assert features[0, hour_col_idx].item() == pytest.approx(0.5)

    def test_categorical_columns_are_hashed_consistently(self):
        """Same categorical value must hash to the same index whether it
        appears in row 0 or row N (i.e. hashing is a pure function of the
        value, not row-dependent)."""
        df = _make_dlrm_df()

        features = build_dlrm_features(df)

        device_col_idx = dlrm_features.DENSE_WIDTH + dlrm_features.CATEGORICAL_COLUMNS.index(
            "device_type"
        )
        # Rows 0 and 3 are both devicetype 2 in _make_dlrm_df.
        assert features[0, device_col_idx].item() == features[3, device_col_idx].item()
        assert features[0, device_col_idx].item() == float(
            dlrm_features.hash_to_idx(2, "device_type")
        )

    def test_each_categorical_stays_inside_its_own_vocabulary(self):
        """Per-feature vocab sizes: an index past a table's end is an
        out-of-range embedding lookup at training time."""
        features = build_dlrm_features(_make_dlrm_df())

        for offset, column in enumerate(dlrm_features.CATEGORICAL_COLUMNS):
            col = features[:, dlrm_features.DENSE_WIDTH + offset]
            vocab = dlrm_features.VOCAB_SIZES[column]
            assert col.min().item() >= 0
            assert col.max().item() < vocab, (
                f"{column} index {col.max().item()} exceeds its vocabulary of {vocab}"
            )

    def test_weekend_flag_derived_from_day_of_week(self):
        """day_of_week follows datetime.weekday(): Monday 0, Sunday 6."""
        df = _make_dlrm_df()
        weekend_idx = dlrm_features.DENSE_COLUMNS.index("is_weekend")

        flags = [features.item() for features in build_dlrm_features(df)[:, weekend_idx]]

        # _make_dlrm_df's day_of_week is [0, 5, 6, 2, 3] -> Mon, Sat, Sun, Wed, Thu.
        assert flags == [0.0, 1.0, 1.0, 0.0, 0.0]

    def test_absent_categorical_encodes_to_the_reserved_index(self):
        """A row missing a categorical must not share an embedding with a real
        value. Slot 0 is reserved so "missing" is learnable."""
        df = _make_dlrm_df(n_rows=1)
        df["site_domain"] = [None]

        features = build_dlrm_features(df)

        domain_idx = dlrm_features.DENSE_WIDTH + dlrm_features.CATEGORICAL_COLUMNS.index(
            "site_domain"
        )
        assert features[0, domain_idx].item() == float(dlrm_features.UNKNOWN_INDEX)

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


def _config_input_names(config_path):
    """Input tensor names declared by a Triton config.pbtxt, in order.

    A deliberately small reader: the file's `input [ { name: "x" ... } ]` block
    is the contract a served engine must satisfy, and parsing it is what lets
    these tests compare the exported graph against the thing Triton will load
    rather than against a list restated in the test.
    """
    text = Path(config_path).read_text()
    start = text.index("input [")
    end = text.index("]", text.index("output [") - 1) if "output [" in text else len(text)
    block = text[start : text.index("output [")] if "output [" in text else text[start:end]
    return re.findall(r'name:\s*"([^"]+)"', block)


class TestExportToOnnxServingSignature:
    """export_to_onnx() for DLRM must emit the SERVING signature so a retrained
    artifact is loadable by dlrm_bid_shader_stable/_canary and compilable by the
    Model Optimizer: one dense input (FP32 [b, DENSE_WIDTH]) plus one INT64 [b]
    input per categorical feature, and a sigmoid'd ctr_prediction output —
    matching source/triton/export_models.py's export_dlrm. (Previously it
    exported a single width-7 ``features`` input named ``output``, which no
    served config accepts.)

    The names are asserted against shared/dlrm_features.py and against the
    Triton config.pbtxt files themselves, not against a list written out here.
    Restating them in each place is how the exporter, the container and the
    served config came to disagree about what an input is called.
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
        # Separate positional inputs, not a single combined tensor.
        expected_arity = 1 + dlrm_features.CATEGORICAL_WIDTH
        assert isinstance(args, tuple) and len(args) == expected_arity
        dense, *categorical = args
        assert tuple(dense.shape) == (1, NUM_DENSE)
        assert dense.dtype == torch.float32
        for s in categorical:
            assert tuple(s.shape) == (1,)
            assert s.dtype == torch.int64
        assert captured["kwargs"]["input_names"] == [
            dlrm_features.TRITON_DENSE_INPUT,
            *dlrm_features.TRITON_CATEGORICAL_INPUTS,
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

        expected = [
            dlrm_features.TRITON_DENSE_INPUT,
            *dlrm_features.TRITON_CATEGORICAL_INPUTS,
        ]
        input_names = [i.name for i in graph.input]
        assert input_names == expected
        assert [o.name for o in graph.output] == ["ctr_prediction"]

        # elem_type: 1 = FLOAT, 7 = INT64 (onnx.TensorProto).
        by_name = {i.name: i for i in graph.input}
        assert (
            by_name[dlrm_features.TRITON_DENSE_INPUT].type.tensor_type.elem_type
            == onnx.TensorProto.FLOAT
        )
        for name in dlrm_features.TRITON_CATEGORICAL_INPUTS:
            assert by_name[name].type.tensor_type.elem_type == onnx.TensorProto.INT64

    @pytest.mark.parametrize(
        "config_name",
        [
            "dlrm_bid_shader/config.pbtxt",
            "dlrm_bid_shader/config_tensorrt.pbtxt",
            "dlrm_bid_shader_stable/config.pbtxt",
            "dlrm_bid_shader_canary/config.pbtxt",
        ],
    )
    def test_exported_graph_matches_every_served_config(self, tmp_path, config_name):
        """Every config Triton can load must declare exactly what we export.

        The router config additionally declares an optional `target_variant`
        input, which is control-plane and not part of the model signature; it is
        excluded here. Everything else must match name-for-name and in order,
        because Triton binds inputs by name and an unmatched name is a load-time
        or request-time failure rather than a wrong number.
        """
        import onnx

        import train as train_module
        from models import DLRMModel

        repo = Path(__file__).resolve().parents[1] / "triton" / "model_repository"
        declared = [
            n
            for n in _config_input_names(repo / config_name)
            if n != "target_variant"
        ]

        onnx_path = train_module.export_to_onnx(
            DLRMModel(), "dlrm_bid_shader", str(tmp_path)
        )
        exported = [i.name for i in onnx.load(onnx_path).graph.input]

        assert exported == declared, (
            f"{config_name} declares {declared} but the trainer exports "
            f"{exported}. A retrained artifact would not load."
        )

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
            "base_model_version": "arn:aws:sagemaker:us-east-1:123456789012:model-package/nv5-artf-dlrm-bid-shader/1",
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
