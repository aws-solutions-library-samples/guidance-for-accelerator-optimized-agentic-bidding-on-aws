"""Unit tests for source/training/export_xgboost_genesis.py.

Verifies the genesis XGBoost booster is a REAL, deterministic constant-output
model (never fabricated data): trained via a real xgb.train() call, its
prediction matches the documented no-op convention (floor=1.0, margin=0.0)
across arbitrary feature vectors, not just the single training point. Also
verifies both artifact forms (native XGBoost JSON for FIL, ONNX for registry
bookkeeping) are written to the expected paths and are independently loadable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.export_xgboost_genesis import (
    FEATURE_VECTOR_LENGTH,
    export_genesis_model,
    main,
    train_genesis_booster,
)
from training.xgboost_pipeline import MODEL_TYPE_BY_TARGET, TARGET_FLOOR, TARGET_MARGIN

# A spread of feature vectors covering the real value ranges documented in
# source/containers/deal_yield_manager/features.py (auction-type one-hot,
# bidfloor + tier, category tier, hour/weekday norms) -- not just zeros.
_SAMPLE_FEATURE_VECTORS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 5.00, 2.0, 2.0, 0.9, 0.9],
    [1.0, 0.0, 10.00, 2.0, 1.0, 0.5, 0.3],
    [0.0, 0.0, 3.25, 1.0, 0.0, 0.25, 0.6],
]


class TestTrainGenesisBooster:
    def test_floor_target_predicts_constant_one(self):
        """floor_multiplier == 1.0 is the documented "no change" convention
        (BR-3) -- the genesis booster must predict exactly this, for every
        input, not just the training point."""
        booster = train_genesis_booster(TARGET_FLOOR)
        for vec in _SAMPLE_FEATURE_VECTORS:
            pred = booster.predict(xgb.DMatrix(np.array([vec], dtype=np.float32)))[0]
            assert abs(pred - 1.0) < 1e-4, f"expected 1.0 for {vec}, got {pred}"

    def test_margin_target_predicts_constant_zero(self):
        """margin_value == 0.0 is the documented "no change" convention
        (BR-4)."""
        booster = train_genesis_booster(TARGET_MARGIN)
        for vec in _SAMPLE_FEATURE_VECTORS:
            pred = booster.predict(xgb.DMatrix(np.array([vec], dtype=np.float32)))[0]
            assert abs(pred - 0.0) < 1e-4, f"expected 0.0 for {vec}, got {pred}"

    def test_invalid_target_raises_key_error(self):
        with pytest.raises(KeyError):
            train_genesis_booster("not-a-real-target")

    def test_booster_is_a_real_xgboost_booster_not_a_stub(self):
        """Confirms this is a genuine trained model object (has real trees),
        not a fabricated/mocked stand-in."""
        booster = train_genesis_booster(TARGET_FLOOR)
        assert isinstance(booster, xgb.Booster)
        dump = booster.get_dump()
        assert len(dump) > 0
        assert any("leaf" in tree for tree in dump)


class TestExportGenesisModel:
    def test_writes_both_artifact_forms_at_expected_paths(self, tmp_path):
        xgboost_json_path, onnx_path = export_genesis_model(TARGET_FLOOR, str(tmp_path))

        expected_dir = tmp_path / MODEL_TYPE_BY_TARGET[TARGET_FLOOR] / "1"
        assert xgboost_json_path == str(expected_dir / "xgboost.json")
        assert onnx_path == str(expected_dir / "model.onnx")
        assert os.path.isfile(xgboost_json_path)
        assert os.path.isfile(onnx_path)

    def test_xgboost_json_is_loadable_and_matches_genesis_convention(self, tmp_path):
        xgboost_json_path, _ = export_genesis_model(TARGET_MARGIN, str(tmp_path))

        loaded = xgb.Booster()
        loaded.load_model(xgboost_json_path)
        for vec in _SAMPLE_FEATURE_VECTORS:
            pred = loaded.predict(xgb.DMatrix(np.array([vec], dtype=np.float32)))[0]
            assert abs(pred - 0.0) < 1e-4

    def test_onnx_artifact_is_a_real_nonempty_file(self, tmp_path):
        """Does not assert on ONNX internals (that's onnxmltools' contract,
        not this module's) -- just that a real, non-trivial file was
        written, confirming the conversion call actually ran rather than
        silently no-op-ing."""
        _, onnx_path = export_genesis_model(TARGET_FLOOR, str(tmp_path))
        assert os.path.getsize(onnx_path) > 0

    def test_floor_and_margin_write_to_different_model_type_dirs(self, tmp_path):
        floor_json, _ = export_genesis_model(TARGET_FLOOR, str(tmp_path))
        margin_json, _ = export_genesis_model(TARGET_MARGIN, str(tmp_path))
        assert "deal_yield_manager_floor" in floor_json
        assert "deal_yield_manager_margin" in margin_json
        assert floor_json != margin_json


class TestMainCli:
    def test_main_all_exports_both_targets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["export_xgboost_genesis.py", "--output-dir", str(tmp_path)]
        )
        exit_code = main()
        assert exit_code == 0
        for target in (TARGET_FLOOR, TARGET_MARGIN):
            model_type = MODEL_TYPE_BY_TARGET[target]
            assert (tmp_path / model_type / "1" / "xgboost.json").is_file()
            assert (tmp_path / model_type / "1" / "model.onnx").is_file()

    def test_main_single_target_exports_only_that_target(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["export_xgboost_genesis.py", "--output-dir", str(tmp_path), "--target", TARGET_FLOOR],
        )
        exit_code = main()
        assert exit_code == 0
        assert (tmp_path / "deal_yield_manager_floor" / "1" / "xgboost.json").is_file()
        assert not (tmp_path / "deal_yield_manager_margin").exists()


def test_feature_vector_length_matches_container_convention():
    """FEATURE_VECTOR_LENGTH is duplicated (not imported) from
    features.py by design (see module docstring) -- this test is the
    tripwire that catches drift if features.py's shape ever changes.

    Loaded via importlib.util.spec_from_file_location (not sys.path
    manipulation + import_module) so this test never mutates the global
    sys.path for the rest of the pytest session -- a bare-name "features"
    module made globally importable could collide with anything else in
    the suite importing a same-named module.
    """
    import importlib.util

    features_path = os.path.join(
        os.path.dirname(__file__), "..", "containers", "deal_yield_manager", "features.py"
    )
    spec = importlib.util.spec_from_file_location("_deal_yield_features_check", features_path)
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)

    assert FEATURE_VECTOR_LENGTH == features.FEATURE_VECTOR_LENGTH
