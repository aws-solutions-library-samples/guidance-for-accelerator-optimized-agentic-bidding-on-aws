"""NeMo-RL Training Entrypoint for ARTF Recommender Models.

Runs inside a SageMaker training job. Performs:
1. Supervised warm-up on labeled bid outcome data
2. RL fine-tuning using NeMo-RL with a custom reward function
3. ONNX export of the trained model

Environment (SageMaker convention):
    /opt/ml/input/data/training/  — Parquet training data
    /opt/ml/input/config/hyperparameters.json — Job hyperparameters
    /opt/ml/model/ — Output directory for trained ONNX model
"""

from __future__ import annotations

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


def load_hyperparameters() -> dict:
    """Load hyperparameters from SageMaker config or environment."""
    if os.path.exists(SM_HP_FILE):
        with open(SM_HP_FILE) as f:
            return {k: _parse_hp_value(v) for k, v in json.load(f).items()}
    # Fallback: read from environment (SM_HP_* prefix)
    return {
        "model_type": os.environ.get("SM_HP_MODEL_TYPE", "dlrm_bid_shader"),
        "use_reinforcement_learning": os.environ.get("SM_HP_USE_REINFORCEMENT_LEARNING", "true"),
        "reward_function": os.environ.get("SM_HP_REWARD_FUNCTION", "roi"),
        "rl_learning_rate": float(os.environ.get("SM_HP_RL_LEARNING_RATE", "1e-4")),
        "rl_epochs": int(os.environ.get("SM_HP_RL_EPOCHS", "5")),
        "supervised_epochs": int(os.environ.get("SM_HP_SUPERVISED_EPOCHS", "10")),
        "batch_size": int(os.environ.get("SM_HP_BATCH_SIZE", "256")),
        "learning_rate": float(os.environ.get("SM_HP_LEARNING_RATE", "1e-3")),
        "validation_split": float(os.environ.get("SM_HP_VALIDATION_SPLIT", "0.1")),
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
    """Load Parquet training data from the SageMaker training channel."""
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

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

    # Create dummy input matching the model's expected input shape
    if model_type == "dlrm_bid_shader":
        dummy = torch.randn(1, 4)  # dense_features
        input_names = ["dense_features"]
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
    features = torch.tensor(df.filter(like="feature_").values, dtype=torch.float32)
    labels = torch.tensor(df["label"].values, dtype=torch.float32)

    # Optional outcome columns for RL
    outcome_cols = [c for c in df.columns if c in ("win", "price_paid", "revenue")]
    outcomes = torch.tensor(df[outcome_cols].values, dtype=torch.float32) if outcome_cols else None

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
