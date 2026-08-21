"""Unit tests for the NeMo-RL training container entrypoint (train.py).

Covers three real production bugs found while diagnosing a live failed
training job (dlrm-bid-shader-1787311543-72232191, base version
nvd-artf-dlrm-bid-shader/1) and its predecessors:

1. load_training_data() used a non-recursive glob, but the Glue ETL job
   writes Hive-style partitioned Parquet (window_start=.../window_end=.../
   *.parquet), which SageMaker preserves under the training channel.
2. Feature/outcome column selection referenced columns that never exist in
   the real Glue ETL output ("feature_*", "win", "revenue") instead of the
   actual columns ("bid_floor", "won", "conversion_value", etc.).
3. export_to_onnx()'s DLRM dummy input was width 4, but DLRMModel.forward()
   requires width NUM_DENSE+NUM_SPARSE=7.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "training", "container")
)

from train import (  # noqa: E402
    _DLRM_DENSE_COLUMNS,
    _DLRM_SPARSE_COLUMNS,
    _hash_to_idx,
    build_dlrm_features,
    build_features,
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


class TestExportToOnnxDummyInputShape:
    """export_to_onnx()'s DLRM dummy input previously had width 4, but
    DLRMModel.forward() requires width NUM_DENSE+NUM_SPARSE=7 — this bug was
    unreachable until the parquet-glob and feature-selection bugs were fixed,
    since training always failed before reaching export.

    Asserts the dummy tensor shape passed to torch.onnx.export directly
    (via a monkeypatched export call) rather than running a real ONNX
    export, since torch's dynamo-based exporter is sensitive to whichever
    `triton` import resolves first on sys.path in this dev environment
    (this repo's own source/triton/ package vs. the PyPI compiler package)
    — a local test-environment path collision, not a training-code
    correctness question.
    """

    def test_dlrm_dummy_width_matches_model_forward_requirement(self, tmp_path, monkeypatch):
        import train as train_module
        from models import DLRMModel, NUM_DENSE, NUM_SPARSE

        captured = {}

        def _fake_onnx_export(model, dummy, output_path, **kwargs):
            captured["dummy_shape"] = tuple(dummy.shape)
            Path(output_path).write_bytes(b"fake-onnx-for-shape-test")

        monkeypatch.setattr(train_module.torch.onnx, "export", _fake_onnx_export)
        model = DLRMModel()

        onnx_path = train_module.export_to_onnx(model, "dlrm_bid_shader", str(tmp_path))

        assert os.path.exists(onnx_path)
        assert captured["dummy_shape"] == (1, NUM_DENSE + NUM_SPARSE)
        assert captured["dummy_shape"][1] == 7
