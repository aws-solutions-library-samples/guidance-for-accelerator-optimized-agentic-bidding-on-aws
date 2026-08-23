"""NeMo-RL Training Entrypoint for ARTF Recommender Models.

Runs inside a SageMaker training job. Performs:
1. Supervised warm-up on labeled bid outcome data
2. RL fine-tuning using NeMo-RL with a custom reward function
3. ONNX export of the trained model

Environment (SageMaker convention):
    /opt/ml/input/data/training/  — Parquet training data
    /opt/ml/input/config/hyperparameters.json — Job hyperparameters
    /opt/ml/model/ — Output directory for trained ONNX model

Supported model types: dlrm_bid_shader only. ncf_deal_manager is registered
in MODEL_REGISTRY (matches the serving container split) but cannot be
trained yet: NCFModel.forward() requires user_ids/item_ids (a per-deal
identifier), and BidShadingOutcomeEvent/BidShadingOutcomeRecord
(shared/feedback_models.py) never captures a deal_id anywhere in the
schema. Adding NCF training support requires adding deal_id to that event
schema and threading it through the Feedback Collector -> Glue ETL path
first, not a change local to this file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# SageMaker paths
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_HP_FILE = "/opt/ml/input/config/hyperparameters.json"

# Model architecture registry. widedeep_segment_activator is intentionally
# absent — segment activation is rule-based, not a trainable model. See
# source/containers/widedeep_segment_activator/app.py.
MODEL_REGISTRY = {
    "dlrm_bid_shader": "models.dlrm",
    "ncf_deal_manager": "models.ncf",
}

# DLRM dense/sparse feature columns, drawn from the real Glue ETL output
# schema (glue_feature_engineering.py's engineer_features()/
# compute_win_rate_buckets()) rather than a "feature_*" naming convention
# that ETL never produces. Chosen to be known at bid time only —
# shaded_price/shade_ratio (the model's own past decision) and roi (computed
# from post-bid outcomes) are excluded as target leakage. Mirrors the
# dense-feature ordering the serving container already uses
# (source/containers/dlrm_bid_shader/app.py's _extract_features_torch:
# [bidfloor, hour_normalized, ..., ...]) so training and serving stay
# consistent in shape (NUM_DENSE=4, NUM_SPARSE=3 — models/__init__.py).
_DLRM_DENSE_COLUMNS = ["bid_floor", "hour_of_day", "shade_factor_used", "conversion_value_estimate_used"]
_DLRM_SPARSE_COLUMNS = ["device_type", "site_domain", "win_rate_bucket"]
_DLRM_VOCAB_SIZE = 1000  # Matches models/__init__.py's DLRMModel VOCAB_SIZE.

# Real outcome columns for the RL phase, in the order reward.py's
# compute_reward() expects ([win, price_paid, revenue]). The dataframe has
# "won" (not "win") and "conversion_value" (not "revenue") — see
# shared/feedback_models.py's BidShadingOutcomeEvent/Record.
_OUTCOME_COLUMNS = ["won", "price_paid", "conversion_value"]


def _hash_to_idx(value: str, vocab: int = _DLRM_VOCAB_SIZE) -> int:
    """Hash a categorical string to an embedding index.

    Same convention as source/containers/dlrm_bid_shader/app.py's
    _hash_to_idx, so a category maps to the same embedding slot whether
    computed at training time or serving time.
    """
    return int(hashlib.md5(str(value).encode(), usedforsecurity=False).hexdigest(), 16) % vocab  # nosec B324


def build_dlrm_features(df: pd.DataFrame) -> torch.Tensor:
    """Build the DLRM input tensor: [dense (4) | sparse indices (3)].

    Matches DLRMModel.forward()'s expected layout (models/__init__.py):
    columns [:NUM_DENSE] are continuous features, columns
    [NUM_DENSE:NUM_DENSE+NUM_SPARSE] are embedding indices (as floats,
    cast to long inside the model).
    """
    dense = df[_DLRM_DENSE_COLUMNS].astype(float).copy()
    # hour_of_day (0-23) normalized to [0, 1), matching the serving
    # container's dense-feature convention (app.py: hour/24.0).
    dense["hour_of_day"] = dense["hour_of_day"] / 24.0

    sparse = pd.DataFrame({
        col: df[col].apply(_hash_to_idx) for col in _DLRM_SPARSE_COLUMNS
    }).astype(float)

    combined = pd.concat([dense[_DLRM_DENSE_COLUMNS], sparse[_DLRM_SPARSE_COLUMNS]], axis=1)
    return torch.tensor(combined.values, dtype=torch.float32)


def build_features(df: pd.DataFrame, model_type: str) -> torch.Tensor:
    """Build the model-specific input feature tensor from real ETL columns."""
    if model_type == "dlrm_bid_shader":
        return build_dlrm_features(df)
    raise ValueError(
        f"No feature-building logic for model_type '{model_type}'. "
        "ncf_deal_manager training requires a deal_id column that "
        "BidShadingOutcomeEvent/Record does not currently capture — see "
        "training/container/train.py module docstring."
    )



# Defaults for every hyperparameter this script reads. None of the three
# real CreateTrainingJob callers (orchestrator/training_trigger.py's
# on-demand "Train from load test" path, governance_eventbridge_cfn.yaml's
# scheduled RetrainingTriggerFunction Lambda, and this repo's
# TrainingPipeline._build_hyperparameters) send a complete set — e.g.
# training_trigger.py only sends model_type/base_model_version/window_days/
# cadence_hours/triggered_by. SageMaker always writes SM_HP_FILE verbatim
# whenever ANY HyperParameters are passed to CreateTrainingJob, so the
# JSON-file branch below previously returned exactly what the caller sent
# with no defaults applied at all -- confirmed live: a real job
# (dlrm-bid-shader-1787398867-e572c442) crashed with
# `KeyError: 'supervised_epochs'` for exactly this reason. Every hp[...]
# lookup in this file must be able to fall back to one of these, regardless
# of which branch loaded the raw values.
_HP_DEFAULTS: dict = {
    "model_type": "dlrm_bid_shader",
    "use_reinforcement_learning": True,
    "reward_function": "roi",
    "rl_learning_rate": 1e-4,
    "rl_epochs": 5,
    "supervised_epochs": 10,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "validation_split": 0.1,
}


def load_hyperparameters() -> dict:
    """Load hyperparameters from SageMaker config or environment.

    Whichever source supplies raw values (the JSON file SageMaker writes
    when HyperParameters is non-empty, or the SM_HP_* env vars otherwise),
    the result is merged over _HP_DEFAULTS so a caller sending only a
    partial hyperparameter set (e.g. model_type/base_model_version/
    window_days/cadence_hours/triggered_by) never crashes on a missing key
    later in this script -- it just gets this file's documented defaults
    for whatever it didn't specify.
    """
    if os.path.exists(SM_HP_FILE):
        with open(SM_HP_FILE) as f:
            raw = {k: _parse_hp_value(v) for k, v in json.load(f).items()}
        return {**_HP_DEFAULTS, **raw}
    # Fallback: read from environment (SM_HP_* prefix)
    return {
        "model_type": os.environ.get("SM_HP_MODEL_TYPE", _HP_DEFAULTS["model_type"]),
        "use_reinforcement_learning": os.environ.get("SM_HP_USE_REINFORCEMENT_LEARNING", "true"),
        "reward_function": os.environ.get("SM_HP_REWARD_FUNCTION", _HP_DEFAULTS["reward_function"]),
        "rl_learning_rate": float(os.environ.get("SM_HP_RL_LEARNING_RATE", str(_HP_DEFAULTS["rl_learning_rate"]))),
        "rl_epochs": int(os.environ.get("SM_HP_RL_EPOCHS", str(_HP_DEFAULTS["rl_epochs"]))),
        "supervised_epochs": int(os.environ.get("SM_HP_SUPERVISED_EPOCHS", str(_HP_DEFAULTS["supervised_epochs"]))),
        "batch_size": int(os.environ.get("SM_HP_BATCH_SIZE", str(_HP_DEFAULTS["batch_size"]))),
        "learning_rate": float(os.environ.get("SM_HP_LEARNING_RATE", str(_HP_DEFAULTS["learning_rate"]))),
        "validation_split": float(os.environ.get("SM_HP_VALIDATION_SPLIT", str(_HP_DEFAULTS["validation_split"]))),
    }


def _parse_hp_value(v):
    """Parse SageMaker hyperparameter string to appropriate type."""
    if isinstance(v, str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
    return v


def load_training_data(data_dir: str, model_type: str) -> pd.DataFrame:
    """Load Parquet training data from the SageMaker training channel.

    The Glue ETL job (glue_etl_cfn.yaml) writes Hive-style partitioned output
    (window_start=.../window_end=.../*.parquet), and SageMaker preserves that
    directory structure when it downloads the S3 training-data prefix. A
    non-recursive glob therefore finds nothing even when the channel has
    real data, so this must search subdirectories.
    """
    data_path = Path(data_dir)
    parquet_files = list(data_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {data_dir} (searched recursively)")

    logger.info("Loading %d parquet files from %s", len(parquet_files), data_dir)
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


def build_model(model_type: str, hp: dict) -> nn.Module:
    """Instantiate the model architecture for the given model type."""
    if model_type == "dlrm_bid_shader":
        from models.dlrm import DLRMModel
        return DLRMModel()
    elif model_type == "ncf_deal_manager":
        from models.ncf import NCFModel
        return NCFModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def supervised_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hp: dict,
) -> dict[str, float]:
    """Phase 1: Supervised training on labeled bid outcomes."""
    logger.info("Phase 1: Supervised training (%d epochs)", hp["supervised_epochs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(hp["supervised_epochs"]):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch[0].to(device), batch[1].to(device)
                output = model(features)
                loss = criterion(output.squeeze(), labels.float())
                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        logger.info(
            "  Epoch %d/%d — train_loss: %.4f, val_loss: %.4f",
            epoch + 1, hp["supervised_epochs"], avg_train, avg_val
        )
        best_val_loss = min(best_val_loss, avg_val)

    return {"supervised_train_loss": avg_train, "supervised_val_loss": best_val_loss}


def rl_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    hp: dict,
) -> dict[str, float]:
    """Phase 2: Reinforcement learning fine-tuning using NeMo-RL reward shaping.

    Uses the custom bid outcome reward function to optimize model parameters
    for downstream business metrics (ROI, revenue) rather than just CTR accuracy.

    The RL objective is REINFORCE-style policy gradient where:
    - Action: the model's predicted score (affects bid price)
    - Reward: computed from bid outcome (win/loss, price paid, revenue generated)
    - Baseline: moving average of recent rewards for variance reduction
    """
    logger.info(
        "Phase 2: RL fine-tuning (%d epochs, reward=%s, lr=%s)",
        hp["rl_epochs"], hp["reward_function"], hp["rl_learning_rate"]
    )

    from reward import compute_reward

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=hp["rl_learning_rate"])
    baseline = 0.0
    baseline_decay = 0.99
    total_reward = 0.0
    n_steps = 0

    for epoch in range(hp["rl_epochs"]):
        epoch_reward = 0.0
        epoch_steps = 0

        for batch in train_loader:
            features = batch[0].to(device)
            # Outcome columns: win (bool), price_paid, revenue
            outcomes = batch[2].to(device) if len(batch) > 2 else None

            if outcomes is None:
                continue

            # Forward pass — model predicts score used for bid pricing
            scores = model(features).squeeze()

            # Compute reward from outcomes using the NeMo-RL reward function
            rewards = compute_reward(
                scores=scores.detach(),
                outcomes=outcomes,
                reward_type=hp["reward_function"],
            )

            # REINFORCE policy gradient with baseline
            advantage = rewards - baseline
            # Log-probability approximation for continuous scores
            log_prob = -0.5 * (scores - scores.mean()) ** 2
            policy_loss = -(log_prob * advantage.detach()).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update baseline (moving average)
            batch_reward = rewards.mean().item()
            baseline = baseline_decay * baseline + (1 - baseline_decay) * batch_reward
            epoch_reward += batch_reward
            epoch_steps += 1

        avg_reward = epoch_reward / max(epoch_steps, 1)
        logger.info("  RL Epoch %d/%d — avg_reward: %.4f", epoch + 1, hp["rl_epochs"], avg_reward)
        total_reward += epoch_reward
        n_steps += epoch_steps

    return {
        "rl_avg_reward": total_reward / max(n_steps, 1),
        "rl_final_baseline": baseline,
    }


def export_to_onnx(model: nn.Module, model_type: str, output_dir: str) -> str:
    """Export trained PyTorch model to ONNX format for Triton serving."""
    model.eval()
    model = model.cpu()

    output_path = os.path.join(output_dir, "model.onnx")

    # Create dummy input matching the model's expected input shape.
    # DLRMModel.forward() (models/__init__.py) slices a single combined
    # tensor into dense [:NUM_DENSE] and sparse [NUM_DENSE:NUM_DENSE+NUM_SPARSE]
    # — width 4+3=7, not 4. A width-4 dummy previously went unnoticed because
    # earlier bugs (empty parquet glob, then an empty feature tensor) meant
    # training always failed before reaching export.
    if model_type == "dlrm_bid_shader":
        from models import NUM_DENSE, NUM_SPARSE
        dummy = torch.randn(1, NUM_DENSE + NUM_SPARSE)
        input_names = ["features"]
    elif model_type == "ncf_deal_manager":
        dummy = torch.randn(1, 2)  # [user_id, item_id] as floats for export
        input_names = ["features"]
    else:
        raise ValueError(f"Unknown model_type for ONNX export: {model_type}")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes={input_names[0]: {0: "batch_size"}},
        opset_version=17,
    )

    logger.info("Exported ONNX model to %s (%.2f MB)", output_path, os.path.getsize(output_path) / 1e6)
    return output_path


def main():
    """Main training entrypoint — SageMaker compatible."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("NeMo-RL Training Job Starting")
    logger.info("=" * 60)

    hp = load_hyperparameters()
    model_type = hp["model_type"]
    logger.info("Model type: %s", model_type)
    logger.info("Hyperparameters: %s", json.dumps(hp, indent=2, default=str))

    # Load training data
    df = load_training_data(SM_CHANNEL_TRAINING, model_type)

    # Build model
    model = build_model(model_type, hp)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2f MB)", param_count, param_count * 4 / 1e6)

    # Prepare data loaders
    features = build_features(df, model_type)
    labels = torch.tensor(df["label"].values, dtype=torch.float32)

    # Optional outcome columns for RL: [won, price_paid, conversion_value],
    # matching reward.py's compute_reward() column order ([win, price_paid,
    # revenue]). Missing conversion_value (no conversion) becomes 0.0 —
    # compute_reward() already treats revenue=0 as "no revenue", not a
    # fabricated outcome.
    outcome_cols = [c for c in _OUTCOME_COLUMNS if c in df.columns]
    outcomes = (
        torch.tensor(df[outcome_cols].fillna(0).astype(float).values, dtype=torch.float32)
        if len(outcome_cols) == len(_OUTCOME_COLUMNS)
        else None
    )

    # Train/val split
    n = len(features)
    val_size = int(n * hp.get("validation_split", 0.1))
    train_size = n - val_size

    if outcomes is not None:
        train_ds = TensorDataset(features[:train_size], labels[:train_size], outcomes[:train_size])
        val_ds = TensorDataset(features[train_size:], labels[train_size:], outcomes[train_size:])
    else:
        train_ds = TensorDataset(features[:train_size], labels[:train_size])
        val_ds = TensorDataset(features[train_size:], labels[train_size:])

    batch_size = hp.get("batch_size", 256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Phase 1: Supervised training
    supervised_metrics = supervised_train(model, train_loader, val_loader, hp)
    logger.info("Supervised metrics: %s", supervised_metrics)

    # Phase 2: RL fine-tuning (optional)
    rl_metrics = {}
    if hp.get("use_reinforcement_learning", True) and outcomes is not None:
        rl_metrics = rl_finetune(model, train_loader, hp)
        logger.info("RL metrics: %s", rl_metrics)
    else:
        logger.info("Skipping RL fine-tuning (disabled or no outcome data)")

    # Export to ONNX
    onnx_path = export_to_onnx(model, model_type, SM_MODEL_DIR)

    # Write metrics file for SageMaker
    all_metrics = {**supervised_metrics, **rl_metrics}
    metrics_path = os.path.join(SM_MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training complete in %.1fs", duration)
    logger.info("Output: %s", SM_MODEL_DIR)
    logger.info("Metrics: %s", all_metrics)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
