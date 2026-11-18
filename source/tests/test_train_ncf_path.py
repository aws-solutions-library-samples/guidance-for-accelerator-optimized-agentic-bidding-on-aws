"""The SageMaker trainer's NCF path (training/container/train.py, Q2 A).

train.py reads the interaction schema the sample generator writes, hashes ids with
the serving hash, trains shared.ncf_model.NeuMF with BCELoss on its sigmoid output
and exports the Triton signature. One epoch on the generated CSV is enough to check
the wiring; the deploy-time gate is what judges the model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE))
# Appended, as test_train_container.py does, so `shared` keeps resolving to source/shared.
sys.path.append(str(_SOURCE / "training" / "container"))

from shared import ncf_ids, ncf_model  # noqa: E402
from training.ncf_sample import generate as gen  # noqa: E402
from training.ncf_sample.build import import_torch_dynamo_unshadowed  # noqa: E402

import train as train_module  # noqa: E402

NCF = "ncf_deal_manager"


@pytest.fixture(scope="module")
def channel(tmp_path_factory) -> Path:
    """A training channel holding the generated interactions.csv."""
    root = tmp_path_factory.mktemp("ncf-channel")
    rows = gen.generate(gen.load_json(gen.DEFAULT_COHORTS), gen.seed_deal_ids(gen.load_json(gen.DEFAULT_SEED)))
    (root / "part").mkdir()
    (root / "part" / "interactions.csv").write_bytes(gen.to_csv_bytes(rows))
    return root


@pytest.fixture(scope="module")
def frame(channel) -> pd.DataFrame:
    return train_module.load_training_data(str(channel), NCF)


def test_the_ncf_registry_entry_is_the_shared_model():
    assert train_module.MODEL_REGISTRY[NCF] == "shared.ncf_model"
    assert not (_SOURCE / "training" / "container" / "models" / "ncf.py").exists()


def test_csv_is_read_for_ncf_only(channel, frame):
    assert len(frame) > 0 and {"user_id", "deal_id", "label"} <= set(frame.columns)
    with pytest.raises(FileNotFoundError, match="parquet"):
        train_module.load_training_data(str(channel), "dlrm_bid_shader")


def test_the_label_is_implicit_engagement():
    assert train_module.resolve_label_column({}, NCF) == ("implicit_engagement", "label")
    with pytest.raises(train_module.DatasetUnfitError, match="does not apply"):
        train_module.resolve_label_column({"objective": "cpa"}, NCF)
    # The DLRM default is unchanged.
    assert train_module.resolve_label_column({}) == ("cpa", "label_conversion")


def test_the_dataset_gate_reports_sample_interactions(frame):
    report = train_module.validate_dataset(frame, "label", "implicit_engagement", model_type=NCF)
    assert report["training_data"] == "sample"
    assert report["id_cardinality"]["deal_id"] == 25
    assert 0 < report["positive_rate"] < 1


def test_training_data_is_read_from_the_provenance_column(frame):
    assert set(frame["provenance"]) == {"sample"}
    assert train_module.ncf_training_data(frame) == "sample"
    assert train_module.ncf_training_data(frame.drop(columns=["provenance"])) == "unknown"
    observed = frame.assign(provenance="observed")
    assert train_module.ncf_training_data(observed) == "observed"
    mixed = frame.copy()
    mixed.loc[0, "provenance"] = "observed"
    assert train_module.ncf_training_data(mixed) == "mixed"
    # The generator's other columns no longer stand in for provenance.
    report = train_module.validate_dataset(frame.drop(columns=["provenance"]), "label", "implicit_engagement",
                                           model_type=NCF)
    assert report["training_data"] == "unknown"


def test_the_dataset_gate_refuses_a_missing_id_column(frame):
    with pytest.raises(train_module.DatasetUnfitError, match="deal_id"):
        train_module.validate_dataset(frame.drop(columns=["deal_id"]), "label", "implicit_engagement", model_type=NCF)


def test_features_are_the_serving_buckets(frame):
    features = train_module.build_features(frame, NCF)
    assert features.dtype == torch.int64 and tuple(features.shape) == (len(frame), 2)
    assert features[:, 0].tolist() == [ncf_ids.bucket(u, ncf_ids.NB_USERS) for u in frame["user_id"]]
    assert features[:, 1].tolist() == [ncf_ids.bucket(d, ncf_ids.NB_ITEMS) for d in frame["deal_id"]]
    assert features[:, 0].tolist() == frame["user_bucket"].tolist()


def test_a_dataset_hashed_differently_is_refused(frame):
    skewed = frame.copy()
    skewed.loc[0, "item_bucket"] = (int(skewed.loc[0, "item_bucket"]) + 1) % ncf_ids.NB_ITEMS
    with pytest.raises(ValueError, match="item_bucket"):
        train_module.build_features(skewed, NCF)


def test_one_epoch_trains_and_exports_the_triton_signature(frame, tmp_path):
    import onnx
    import onnxruntime as ort
    from torch.utils.data import DataLoader, TensorDataset

    import_torch_dynamo_unshadowed()
    torch.manual_seed(7)
    model = train_module.build_model(NCF, {})
    assert isinstance(model.net, ncf_model.NeuMF)
    features = train_module.build_features(frame, NCF)
    labels = torch.tensor(frame["label"].astype(float).values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(features, labels), batch_size=500, shuffle=False)
    metrics = train_module.supervised_train(model, loader, loader, {"supervised_epochs": 1, "learning_rate": 1e-3})
    assert 0 < metrics["supervised_train_loss"] < 1.0

    calibration = train_module.calibration_report(model, loader)
    assert calibration["calibration"] == "measured"
    assert 0.0 <= calibration["mean_predicted"] <= 1.0

    path = train_module.export_to_onnx(model, NCF, str(tmp_path))
    graph = onnx.load(path).graph
    assert [(i.name, i.type.tensor_type.elem_type) for i in graph.input] == [
        ("user_ids", onnx.TensorProto.INT64), ("item_ids", onnx.TensorProto.INT64)]
    assert [(o.name, o.type.tensor_type.elem_type) for o in graph.output] == [
        ("relevance_scores", onnx.TensorProto.FLOAT)]
    sample = features[:16]
    (scores,) = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
        ["relevance_scores"], {"user_ids": sample[:, 0].numpy(), "item_ids": sample[:, 1].numpy()})
    with torch.no_grad():
        expected = model(sample).numpy()
    assert abs(scores - expected).max() < 1e-5


def test_main_writes_the_ncf_model_manifest_and_metrics(channel, tmp_path, monkeypatch):
    import_torch_dynamo_unshadowed()
    hp_file = tmp_path / "hyperparameters.json"
    hp_file.write_text(json.dumps({"model_type": NCF, "supervised_epochs": "1", "batch_size": "512"}))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(train_module, "SM_CHANNEL_TRAINING", str(channel))
    monkeypatch.setattr(train_module, "SM_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(train_module, "SM_HP_FILE", str(hp_file))
    torch.manual_seed(11)
    train_module.main()
    assert sorted(p.name for p in model_dir.iterdir()) == ["manifest.json", "metrics.json", "model.onnx"]
    manifest = json.loads((model_dir / "manifest.json").read_text())
    assert manifest["model_type"] == NCF
    assert manifest["training_data"] == "sample"
    assert manifest["target"] == "implicit-engagement-sample"
    assert manifest["triton_input_names"] == ["user_ids", "item_ids"]
    assert manifest["thresholds"] == {"activate": 0.60, "suppress": 0.30}
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert metrics["objective"] == "implicit_engagement"
