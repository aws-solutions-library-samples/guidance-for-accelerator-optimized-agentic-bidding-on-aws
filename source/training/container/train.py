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
from torch.utils.data import DataLoader, Dataset, TensorDataset

# The DLRM feature vector contract, shared with the serving container. In the
# training image this resolves to /opt/ml/code/shared/, which the Dockerfile
# COPYs from a directory staged by source/training/stage_shared.sh; the file
# itself is source/shared/dlrm_features.py, which the bid shader imports
# directly. One definition, two readers.
from shared import dlrm_features
from shared import onnx_compat

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
#
# SUPERSEDED. The vector is now defined once in shared/dlrm_features.py and
# built from there by both this trainer and the serving container. The lists
# that used to sit here were a local copy: they agreed with the container on
# width (4 continuous, 3 categorical) and disagreed on two continuous positions
# and two of three categoricals. Nothing detected it, because equal widths and
# equal dtypes load cleanly. `tests/test_dlrm_feature_parity.py` is the
# assertion that catches it now.
#
# Columns the spec deliberately does not read:
#   shade_factor_used, conversion_value_estimate_used -- the shader's own
#     parameters. A prediction conditioned on the policy that produced its own
#     training data cannot be used to evaluate a change to that policy.
#   shaded_price, roi, price_paid -- consequences of the bid.
#   user_id_hash -- little signal once hashed into a bounded table, and a
#     privacy obligation the signal does not justify.
#
# win_rate_bucket is a legitimate feature but has no serving-side source yet;
# it returns with Item 4 of the dlrm-shading-correctness task list, which
# publishes it as a lookup the container carries.

# Real outcome columns for the RL phase, in the order reward.py's
# compute_reward() expects ([win, price_paid, revenue]). The dataframe has
# "won" (not "win") and "conversion_value" (not "revenue") — see
# shared/feedback_models.py's BidShadingOutcomeEvent/Record.
_OUTCOME_COLUMNS = ["won", "price_paid", "conversion_value"]

# The supervised label, per objective. Selected by the `objective` hyperparameter so
# the training run and the manifest it writes cannot disagree about what was
# optimised.
#
# `ev = p x conversion_value` only holds when p is the probability of the response
# the advertiser PAYS FOR. So a CPA objective trains on a conversion and a CPC
# objective trains on a click.
#
# "profitable_win" is the legacy behaviour, kept selectable so a like-for-like
# comparison against a previously trained model is possible. It folds the response
# and the price paid into one bit, which is why it is not the default: a model
# trained on it predicts "will this bid be cheap enough to be worth winning", and
# multiplying that by a conversion value is not an expected value of anything.
_OBJECTIVE_LABELS: dict = {
    "cpa": "label_conversion",
    "cpc": "label_click",
    "profitable_win": "label",
}
_DEFAULT_OBJECTIVE = "cpa"


class DatasetUnfitError(RuntimeError):
    """The dataset cannot support the requested objective.

    Raised before phase 1, so a run ends with a named diagnosis instead of
    producing a model from data that cannot teach it anything. Every message says
    what was found and what would have to change.
    """


def resolve_label_column(hp: dict) -> tuple[str, str]:
    """(objective, label_column) for this run.

    An unrecognised objective is refused rather than defaulted: silently training
    on a different label than the caller asked for is the class of defect this work
    exists to remove.
    """
    objective = str(hp.get("objective", _DEFAULT_OBJECTIVE)).strip().lower()
    if objective not in _OBJECTIVE_LABELS:
        raise DatasetUnfitError(
            f"objective '{objective}' is not one of {sorted(_OBJECTIVE_LABELS)}. "
            "The objective selects which response the model is trained to predict, "
            "so there is no safe default to fall back to."
        )
    return objective, _OBJECTIVE_LABELS[objective]


def policy_parameters_of(model: nn.Module) -> list:
    """The parameters phase 2 is allowed to optimise.

    A model declares them by exposing `policy_parameters()`. Anything that does not
    is a probability estimator with no policy inside it, and phase 2 has nothing to
    do — which is reported, not silently treated as "optimise everything".

    Defaulting the other way is what the bug was: `model.parameters()` reads as a
    reasonable default and happens to mean "including the probability head".
    """
    getter = getattr(model, "policy_parameters", None)
    if getter is None:
        return []
    return list(getter())


def validate_dataset(df: pd.DataFrame, label_column: str, objective: str) -> dict:
    """Refuse a dataset that cannot teach the model anything, before phase 1.

    Every check here corresponds to a state this pipeline has actually been in, and
    in every case training ran to completion and reported a plausible loss:

    - **No label column.** The ETL had not been re-run, or the objective names a
      response the outcome table does not carry.
    - **No labelled rows.** Every label is NULL. The emit path wrote no response
      signal, so there is nothing to learn from. Indistinguishable from a healthy
      run if NULLs are read as zeros.
    - **One class only.** A positive rate of exactly 0 or 1 trains a model to
      predict the constant, at a loss that falls steadily and means nothing.
    - **A constant categorical.** A feature with one distinct value carries no
      information; its embedding cannot be trained. The load-test generator used to
      emit `site_domain="load-test"` for every row, which made two of the three
      categoricals constant.
    - **An all-unique categorical.** One row per value means every embedding row is
      seen once. That is memorisation, not a learned representation.

    Returns the measurements so they can be recorded in the manifest -- a run's
    diagnosis should be readable after the fact, not only when it fails.
    """
    if label_column not in df.columns:
        raise DatasetUnfitError(
            f"objective '{objective}' trains on '{label_column}', which is not in the "
            f"dataset. Columns present: {sorted(df.columns)}. Re-run the Glue ETL so it "
            f"emits '{label_column}', or choose an objective whose label the data carries."
        )

    total_rows = len(df)
    labels = df[label_column]
    labelled = labels.notna()
    n_labelled = int(labelled.sum())

    if n_labelled == 0:
        raise DatasetUnfitError(
            f"all {total_rows} rows have a NULL '{label_column}', so none of them is "
            "labelled. The outcome records carry no response signal for this objective — "
            "check that the emit path sets it rather than leaving it at its default."
        )

    positives = int((labels[labelled] > 0).sum())
    positive_rate = positives / n_labelled

    if positives == 0 or positives == n_labelled:
        which = "no positives" if positives == 0 else "no negatives"
        raise DatasetUnfitError(
            f"'{label_column}' has {which} across {n_labelled} labelled rows "
            f"(positive rate {positive_rate:.3f}). A single-class label trains the model to "
            "predict a constant, at a loss that falls steadily and means nothing."
        )

    categorical_cardinality: dict = {}
    for column in dlrm_features.CATEGORICAL_COLUMNS:
        if column not in df.columns:
            raise DatasetUnfitError(
                f"categorical feature '{column}' is not in the dataset. The feature spec "
                f"(version {dlrm_features.FEATURE_SPEC_VERSION}) requires "
                f"{list(dlrm_features.CATEGORICAL_COLUMNS)}."
            )
        distinct = int(df[column].nunique(dropna=False))
        categorical_cardinality[column] = distinct
        if distinct <= 1:
            raise DatasetUnfitError(
                f"categorical feature '{column}' has {distinct} distinct value(s) across "
                f"{total_rows} rows, so it carries no information and its embedding cannot "
                "be trained. Check the source of this column — a constant here usually "
                "means it was never populated."
            )
        if total_rows > 1 and distinct == total_rows:
            raise DatasetUnfitError(
                f"categorical feature '{column}' has one distinct value per row "
                f"({distinct} of {total_rows}). Every embedding row would be seen exactly "
                "once, which memorises rather than learns. This column is too "
                "high-cardinality to be used as a categorical."
            )

    report = {
        "objective": objective,
        "label_column": label_column,
        "total_rows": total_rows,
        "labelled_rows": n_labelled,
        "unlabelled_rows": total_rows - n_labelled,
        "positive_rate": round(positive_rate, 6),
        "categorical_cardinality": categorical_cardinality,
        "outcome_provenance": outcome_provenance_mix(df),
    }
    logger.info("Dataset gate passed: %s", json.dumps(report))
    return report


def outcome_provenance_mix(df: pd.DataFrame) -> dict:
    """Count the dataset's rows by where their outcomes came from.

    Reported, not enforced: a run on simulated outcomes is legitimate — it is how the
    loop is exercised without a live signal feed — but it must be labelled, because
    the resulting model's accuracy says nothing about real advertiser behaviour. The
    mix goes into the model manifest so that label survives the training run.

    A dataset written before the column existed reports ``{"unknown": n}`` rather
    than claiming its outcomes were observed.
    """
    if "outcome_provenance" not in df.columns:
        return {"unknown": len(df)}

    counts = df["outcome_provenance"].fillna("unresolved").value_counts()
    return {str(label): int(count) for label, count in counts.items()}


def calibration_report(
    model: nn.Module, val_loader: DataLoader, n_bins: int = 10
) -> dict:
    """Measure how well the model's output behaves as a probability.

    `ev = p x conversion_value` is only an expected value if p is calibrated: among
    the impressions the model scores 0.2, about a fifth should convert. A model can
    rank perfectly -- a fine AUC -- and still be badly calibrated, and the shading
    arithmetic would then be wrong by whatever the miscalibration is.

    Returns the expected calibration error, the maximum per-bin gap, and the bin
    table, over the HELD-OUT split. Measured, not enforced: this reports the number
    so a promotion decision can use it.
    """
    device = next(model.parameters()).device
    model.eval()

    probabilities: list[float] = []
    observed: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch[0].to(device), batch[1].to(device)
            # The training forward returns logits (BCEWithLogitsLoss); the served
            # graph applies the sigmoid. Apply it here so this measures the
            # probability that is actually served.
            probs = torch.sigmoid(model(features).squeeze(-1).reshape(-1))
            probabilities.extend(probs.cpu().tolist())
            observed.extend(labels.float().reshape(-1).cpu().tolist())

    if not probabilities:
        return {"calibration": "unmeasured", "reason": "the validation split was empty"}

    probs_arr = np.asarray(probabilities, dtype=np.float64)
    obs_arr = np.asarray(observed, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins = []
    weighted_gap = 0.0
    max_gap = 0.0
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        # Upper edge inclusive on the last bin so p == 1.0 is counted.
        in_bin = (probs_arr >= low) & (
            (probs_arr < high) if i < n_bins - 1 else (probs_arr <= high)
        )
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"bin": [round(low, 3), round(high, 3)], "count": 0})
            continue
        mean_predicted = float(probs_arr[in_bin].mean())
        observed_rate = float(obs_arr[in_bin].mean())
        gap = abs(mean_predicted - observed_rate)
        weighted_gap += (count / len(probs_arr)) * gap
        max_gap = max(max_gap, gap)
        bins.append(
            {
                "bin": [round(low, 3), round(high, 3)],
                "count": count,
                "mean_predicted": round(mean_predicted, 6),
                "observed_rate": round(observed_rate, 6),
                "gap": round(gap, 6),
            }
        )

    return {
        "calibration": "measured",
        "expected_calibration_error": round(weighted_gap, 6),
        "max_bin_gap": round(max_gap, 6),
        "n_bins": n_bins,
        "n_samples": len(probs_arr),
        "mean_predicted": round(float(probs_arr.mean()), 6),
        "observed_base_rate": round(float(obs_arr.mean()), 6),
        "bins": bins,
    }


def build_dlrm_features(df: pd.DataFrame) -> torch.Tensor:
    """Build the DLRM input tensor: [dense | categorical indices].

    Every value comes from `shared.dlrm_features.build_from_row`, which the
    serving container also calls. Nothing about the layout, the normalisation or
    the hash is restated here -- restating it is how the two sides came to
    compute different vectors of the same width.

    Layout matches DLRMModel.forward() (models/__init__.py): columns
    [:DENSE_WIDTH] are continuous, the rest are embedding indices carried as
    floats and cast to long inside the model.
    """
    rows = [
        dlrm_features.flatten(*dlrm_features.build_from_row(row))
        for row in df.to_dict("records")
    ]
    return torch.tensor(rows, dtype=torch.float32)


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

    # Phase 2 optimises the POLICY, never the probability head.
    #
    # It used to take `model.parameters()`, which is every parameter the probability
    # comes from. An ROI reward has no term that rewards calibration, so phase 2
    # was free to move the output away from being a probability in exchange for a
    # better reward -- and phase 1's whole purpose is to make it one. The two
    # objectives were fighting over the same weights, and the ROI objective ran
    # second.
    #
    # The DLRM has no policy parameters. The policy here is the shade factor and the
    # conversion-value estimate, which live in the DynamoDB Parameter Store and are
    # tuned by the Adaptive Bidding agent against live outcomes -- a closed loop that
    # already exists and does not go through this trainer. So there is nothing for
    # phase 2 to optimise, and the honest outcome is to skip it and say so, rather
    # than to run it on the parameters it must not touch.
    policy_parameters = [p for p in policy_parameters_of(model) if p.requires_grad]
    if not policy_parameters:
        reason = (
            "no policy parameters in the network: the DLRM is a probability estimator. "
            "The bid policy is the three coefficients in shared/shading_policy.py, "
            "which are searched separately by search_policy_parameters() rather than "
            "by gradient descent through the probability head -- optimising that head "
            "against an ROI reward would undo phase 1's calibration."
        )
        logger.info("Phase 2 gradient step skipped — %s", reason)
        return {"rl_skipped": True, "rl_skipped_reason": reason}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(policy_parameters, lr=hp["rl_learning_rate"])
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
            # Clip the parameters being optimised, not every parameter in the model —
            # clipping over model.parameters() would scale gradients by a norm that
            # includes the frozen probability head.
            torch.nn.utils.clip_grad_norm_(policy_parameters, max_norm=1.0)
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


def search_policy_parameters(
    df: pd.DataFrame,
    predictions: "np.ndarray | None",
    hp: dict,
    in_sample: "pd.Series | None" = None,
) -> dict:
    """Phase 2 for the DLRM: search the shading policy's three coefficients.

    ``in_sample`` is an optional boolean Series indexed like ``df``, True for rows that
    were in the supervised training split. It is reported, never used to filter: the
    search scores every scorable row, and the split of its evidence into in-sample and
    out-of-sample rows goes into the report so a reviewer can see what the score rests
    on. At the default 90/10 split most scored rows are in-sample, which is why this
    search's own score is not evidence of lift.

    The network has no policy parameters — it estimates a probability, and the policy
    is the parametric form in shared/shading_policy.py. So phase 2 is a bounded search
    over those three coefficients against an ROI objective, leaving the probability
    head untouched.

    **What this search can and cannot establish.** The dataset records what happened
    under the policy that produced it. Whether a *different* price would have won the
    same auction is counterfactual and not in the data. This search therefore scores a
    candidate against an explicit surrogate: a bid at or above the recorded clearing
    price is assumed to win, a bid below it to lose, and profit is the conversion value
    actually observed minus the price bid. The surrogate is named in the returned
    report and in the model manifest as ``offline_surrogate`` so nobody can mistake its
    output for a measured lift. Only a live split test measures that.

    Rows the surrogate cannot score are skipped rather than imputed:

    * A lost bid has no clearing price, so nothing bounds what would have won it.
    * A row with no reported outcome has no profit to attribute.

    If no row survives, the search reports why and returns the genesis policy. It does
    not invent a tuned parameter set from an unusable dataset.
    """
    from shared.shading_policy import (
        GENESIS_POLICY,
        ShadingPolicy,
        policy_search_space,
    )

    required = {"won", "price_paid", "bid_floor", "original_price"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {
            "policy_searched": False,
            "policy_skipped_reason": (
                f"dataset lacks {missing}, so no candidate price can be scored"
            ),
            "policy": GENESIS_POLICY.as_dict(),
        }

    scorable = df[
        (df["won"] == True)  # noqa: E712 -- pandas needs ==, and `is True` is wrong here
        & df["price_paid"].notna()
    ].copy()
    if predictions is not None and len(predictions) == len(df):
        scorable["_p"] = pd.Series(predictions.reshape(-1), index=df.index).loc[
            scorable.index
        ]
    else:
        return {
            "policy_searched": False,
            "policy_skipped_reason": (
                "no aligned model predictions, so expected value cannot be computed "
                "per row"
            ),
            "policy": GENESIS_POLICY.as_dict(),
        }

    if scorable.empty:
        return {
            "policy_searched": False,
            "policy_skipped_reason": (
                "no won rows with a recorded price_paid: the surrogate has no clearing "
                "price to compare a candidate bid against, and assuming one would "
                "manufacture the result"
            ),
            "policy": GENESIS_POLICY.as_dict(),
        }

    # How much of the search's evidence the model had already seen. Reindexed onto
    # scorable.index rather than sliced positionally: df is only reset_index'd when
    # unlabelled rows were dropped, so on a fully-labelled dataset its index is not
    # 0..n-1 and anything positional would mismatch silently.
    if in_sample is not None:
        flags = in_sample.reindex(scorable.index).fillna(False).astype(bool)
        in_sample_rows = int(flags.sum())
        out_of_sample_rows = int(len(scorable) - in_sample_rows)
    else:
        in_sample_rows = None
        out_of_sample_rows = None

    value_column = (
        "conversion_value" if "conversion_value" in scorable.columns else None
    )
    conversion_value = (
        scorable[value_column].fillna(0.0).to_numpy(dtype=float)
        if value_column
        else np.zeros(len(scorable))
    )
    clearing = scorable["price_paid"].to_numpy(dtype=float)
    floor = scorable["bid_floor"].fillna(0.0).to_numpy(dtype=float)
    original = scorable["original_price"].fillna(0.0).to_numpy(dtype=float)
    p = np.clip(scorable["_p"].fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    # The conversion-value estimate the policy prices against, same knob the serving
    # path reads from the Parameter Store.
    ev = p * float(hp.get("conversion_value_estimate", 5.0))

    space = policy_search_space()
    # Fixed seed: a search whose result cannot be reproduced cannot be reviewed.
    rng = np.random.default_rng(int(hp.get("policy_search_seed", 20260927)))
    n_candidates = int(hp.get("policy_search_candidates", 512))

    def score(candidate: ShadingPolicy) -> float:
        raw = candidate.base + candidate.slope * np.power(ev, candidate.curvature)
        price = np.clip(raw, floor, np.maximum(original, floor))
        won = price >= clearing
        return float(np.mean(np.where(won, conversion_value - price, 0.0)))

    best = GENESIS_POLICY
    best_score = score(best)
    genesis_score = best_score
    evaluated = 1

    for _ in range(n_candidates):
        try:
            candidate = ShadingPolicy(
                base=float(rng.uniform(*space["base"])),
                slope=float(rng.uniform(*space["slope"])),
                curvature=float(rng.uniform(*space["curvature"])),
            )
        except Exception:
            continue
        evaluated += 1
        value = score(candidate)
        if value > best_score:
            best, best_score = candidate, value

    # A searched policy is only returned when it beats genesis by the configured
    # margin. Strict `>`: when no candidate improves on genesis, best_score ==
    # genesis_score, and a tie is not an improvement.
    #
    # The default 0.0 is a regression guard, not a significance test — on an in-sample,
    # simulated dataset a small positive improvement sits well inside noise. That is why
    # `policy` falls back to genesis on rejection, and why acceptance is reported next to
    # the in-sample counts and the provenance mix rather than standing alone.
    min_improvement = float(hp.get("policy_search_min_improvement", 0.0))
    improvement = best_score - genesis_score
    accepted = improvement > min_improvement

    return {
        # Whether the search RAN. Distinct from policy_accepted: collapsing the two
        # would make a rejected search indistinguishable from one that never happened.
        "policy_searched": True,
        "policy_search_method": "offline_surrogate",
        "policy_search_surrogate": (
            "a candidate wins when its price >= the recorded price_paid; profit is the "
            "observed conversion_value minus the price. Not a measured lift."
        ),
        "policy_search_rows": int(len(scorable)),
        "policy_search_in_sample_rows": in_sample_rows,
        "policy_search_out_of_sample_rows": out_of_sample_rows,
        # The provenance of the SCORED rows, which is a strict subset of the dataset
        # (won, with a recorded price_paid). The manifest carries the whole-dataset mix;
        # what matters here is what the coefficients were actually tuned on.
        "policy_search_outcome_provenance": outcome_provenance_mix(scorable),
        "policy_search_candidates_evaluated": evaluated,
        "policy_genesis_score": round(genesis_score, 6),
        "policy_best_score": round(best_score, 6),
        "policy_improvement": round(improvement, 6),
        "policy_search_min_improvement": min_improvement,
        "policy_accepted": accepted,
        "policy": (best if accepted else GENESIS_POLICY).as_dict(),
    }


def predict_all(
    model: nn.Module, dataset: "Dataset", batch_size: int = 1024
) -> "np.ndarray | None":
    """Score every row of ``dataset`` and return one flat array of predictions.

    Extracted from main() because it had two defects that only a GPU run could
    reveal, and each cost a full SageMaker round trip to find:

    * **Device.** Phase 1 and rl_finetune leave the model on the GPU while the
      DataLoader yields CPU tensors, so the batch has to be moved to wherever the
      weights are. The device is read off the parameters rather than re-derived from
      `cuda.is_available()`, so this follows the model if anything above moved it.
    * **Shape.** `DLRMModel.forward` already ends in `reshape(-1)`, so its output is
      1-D. An extra `squeeze(-1)` collapsed a final batch of exactly one row from
      `[1]` to a 0-d array, which `np.concatenate` rejects. Invisible unless the row
      count leaves a remainder of one.

    Returns None for an empty dataset, which is what search_policy_parameters takes
    as "no predictions to search against".
    """
    model.eval()
    device = next(model.parameters()).device

    preds = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            preds.append(model(batch[0].to(device)).cpu().numpy())

    if not preds:
        return None
    return np.concatenate(preds)


def export_to_onnx(model: nn.Module, model_type: str, output_dir: str) -> str:
    """Export trained PyTorch model to ONNX format for Triton serving."""
    model.eval()
    model = model.cpu()

    output_path = os.path.join(output_dir, "model.onnx")

    if model_type == "dlrm_bid_shader":
        # Export the SERVING signature so the retrained artifact is loadable by
        # dlrm_bid_shader_stable/_canary and compilable by the Model Optimizer:
        # one dense input (FP32 [b, DENSE_WIDTH]) plus one INT64 [b] input per
        # categorical feature, and a sigmoid'd ctr_prediction output.
        # DLRMExportModel wraps the trained net (reusing its weights) and applies
        # the sigmoid + I/O naming; see source/triton/export_models.py's
        # export_dlrm, which this mirrors so training output == served contract.
        #
        # The names come from the feature spec, which the serving container reads
        # too, so a retrained engine cannot declare inputs the container does not
        # send.
        from models import DLRMExportModel, NUM_DENSE

        export_model = DLRMExportModel(model).eval()
        dense = torch.randn(1, NUM_DENSE)
        categorical = [
            torch.tensor([1], dtype=torch.int64)
            for _ in dlrm_features.TRITON_CATEGORICAL_INPUTS
        ]

        input_names = [
            dlrm_features.TRITON_DENSE_INPUT,
            *dlrm_features.TRITON_CATEGORICAL_INPUTS,
        ]
        torch.onnx.export(
            export_model,
            (dense, *categorical),
            output_path,
            input_names=input_names,
            output_names=["ctr_prediction"],
            dynamic_axes={
                **{name: {0: "batch"} for name in input_names},
                "ctr_prediction": {0: "batch"},
            },
            opset_version=17,
            # `dynamo` only exists from torch 2.6. The training image is older, and
            # passing it there raised TypeError after phase 1 had already completed —
            # a whole run thrown away for a keyword argument.
            **onnx_compat.onnx_export_kwargs(dynamo=False),
        )
    elif model_type == "ncf_deal_manager":
        dummy = torch.randn(1, 2)  # [user_id, item_id] as floats for export
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["features"],
            output_names=["output"],
            dynamic_axes={"features": {0: "batch_size"}},
            opset_version=17,
        )
    else:
        raise ValueError(f"Unknown model_type for ONNX export: {model_type}")

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

    # Which response this run is trained to predict, and the gate that refuses a
    # dataset which cannot teach it. Both run BEFORE any training, so an unfit
    # dataset ends the job with a named diagnosis rather than a trained artifact.
    objective, label_column = resolve_label_column(hp)
    logger.info("Objective: %s (label column '%s')", objective, label_column)
    dataset_report = validate_dataset(df, label_column, objective)

    # Unlabelled rows are dropped, not coerced to 0. A NULL label means no response
    # signal was recorded; reading it as a negative teaches the model that every
    # unobserved impression failed to convert.
    if dataset_report["unlabelled_rows"]:
        logger.info(
            "Dropping %d unlabelled row(s) of %d",
            dataset_report["unlabelled_rows"],
            dataset_report["total_rows"],
        )
        df = df[df[label_column].notna()].reset_index(drop=True)

    # Prepare data loaders
    features = build_features(df, model_type)
    labels = torch.tensor(df[label_column].astype(float).values, dtype=torch.float32)

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

    # Calibration on the held-out split, measured straight after phase 1 — before
    # anything else can move the output. `ev = p x conversion_value` is only an
    # expected value if p is calibrated, so this is the number that says whether the
    # shading arithmetic means anything.
    calibration = calibration_report(model, val_loader)
    logger.info("Calibration: %s", json.dumps(
        {k: v for k, v in calibration.items() if k != "bins"}
    ))

    # Phase 2: RL fine-tuning (optional). Optimises the policy only; see
    # rl_finetune's note on why the probability head is off-limits.
    rl_metrics = {}
    if hp.get("use_reinforcement_learning", True) and outcomes is not None:
        rl_metrics = rl_finetune(model, train_loader, hp)
        logger.info("RL metrics: %s", rl_metrics)
    else:
        logger.info("Skipping RL fine-tuning (disabled or no outcome data)")

    # Phase 2 for the DLRM proper: search the shading policy's coefficients. The
    # gradient step above has nothing to optimise for this model; the policy does.
    policy_report = {"policy_searched": False, "policy_skipped_reason": "not a shader"}
    if model_type == "dlrm_bid_shader":
        # Score EVERY row of df, not train_loader.dataset. The guard in
        # search_policy_parameters requires len(predictions) == len(df) so it can align
        # predictions onto the scorable subset by index; passing the training split
        # (225 of 249 rows at a 10% validation split) failed that check on every run, so
        # the search never executed and the shader kept serving GENESIS_POLICY.
        #
        # `features` is built from `df`, so the lengths agree structurally rather than by
        # coincidence at one particular row count.
        policy_report = search_policy_parameters(
            df,
            predict_all(model, TensorDataset(features)),
            hp,
            # Reported, not used to filter: which rows the model had already trained on.
            # Derived from train_size and len(df) so the mask cannot disagree with the
            # slice that actually built train_ds.
            in_sample=pd.Series(
                [True] * train_size + [False] * (len(df) - train_size), index=df.index
            ),
        )
        logger.info("Policy search: %s", json.dumps(policy_report))

    # Export to ONNX
    onnx_path = export_to_onnx(model, model_type, SM_MODEL_DIR)

    # Provenance beside the artifact, inside the model.tar.gz SageMaker builds from
    # SM_MODEL_DIR. The promotion path reads feature_spec_version from here and
    # refuses a model it cannot interpret; an engine whose inputs are merely the
    # right width is not evidence that both sides agree on what position two means.
    if model_type == "dlrm_bid_shader":
        manifest_path = os.path.join(SM_MODEL_DIR, dlrm_features.MANIFEST_FILENAME)
        with open(manifest_path, "w") as f:
            json.dump(
                dlrm_features.manifest(
                    model_type,
                    producer="train.py",
                    base_model_version=hp.get("base_model_version", ""),
                    objective=objective,
                    label_column=label_column,
                    reward_function=hp.get("reward_function", ""),
                    # False when phase 2 skipped itself for want of policy
                    # parameters, which is the normal case for this model — see
                    # rl_finetune. A reader must be able to tell whether an ROI
                    # objective has been applied to these weights.
                    used_reinforcement_learning=bool(rl_metrics)
                    and not rl_metrics.get("rl_skipped"),
                    probability_head_frozen_in_phase_2=True,
                    # Top level, not only inside `dataset`, so a reader or a
                    # promotion gate can tell at a glance whether these weights
                    # learned from observed outcomes or from the outcome simulator.
                    outcome_provenance=dataset_report.get("outcome_provenance", {}),
                    # The shading policy these weights were tuned alongside, and how
                    # it was arrived at. `policy_search_method` is what stops the
                    # coefficients being read as a measured improvement: the search
                    # scores candidates against a stated surrogate, because whether a
                    # different price would have won is not in the data.
                    shading_policy=policy_report.get("policy", {}),
                    policy_search=policy_report,
                    dataset=dataset_report,
                    calibration=calibration,
                    trained_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ),
                f,
                indent=2,
            )
        logger.info("Wrote %s", manifest_path)

    # Write metrics file for SageMaker. The calibration error and the dataset shape
    # go in alongside the losses: a run that trained cleanly on a dataset with a
    # 0.001 positive rate should be readable as such from its metrics, not only from
    # its logs.
    all_metrics = {
        **supervised_metrics,
        **rl_metrics,
        "objective": objective,
        "label_column": label_column,
        "labelled_rows": dataset_report["labelled_rows"],
        "positive_rate": dataset_report["positive_rate"],
        "outcome_provenance": dataset_report.get("outcome_provenance", {}),
        "expected_calibration_error": calibration.get("expected_calibration_error"),
        "calibration": calibration.get("calibration"),
    }
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
