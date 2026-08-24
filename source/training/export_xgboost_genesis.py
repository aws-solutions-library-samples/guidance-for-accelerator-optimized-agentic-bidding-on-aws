#!/usr/bin/env python3
"""Export genesis (v1, unretrained-starter) XGBoost models for the Yield
Optimizer's floor/margin sub-models.

Unlike DLRM/NCF's genesis path (a PyTorch net with seeded-random weights,
exported to ONNX -- see source/triton/export_models.py), XGBoost has no
"just export the current weights" equivalent: a booster only exists after
training on some data. This script trains each sub-model on a single
synthetic sample whose label encodes the existing no-op convention already
used by business-rules.md (deal-yield-model unit) BR-3/BR-4:

    - floor target label = 1.0  (floor_multiplier == 1.0 means "no change")
    - margin target label = 0.0 (margin_value == 0.0 means "no change")

With exactly one training sample, XGBoost cannot find a real split, so the
resulting booster is a genuine constant-output model across the ENTIRE
input space (verified: arbitrary feature vectors all predict the same
label) -- not a model overfit to one point that behaves unpredictably
elsewhere. This is a real, deterministic booster built via a real
xgb.train() call, not a fabricated file: every deal recommends "no change"
until the first real closed-loop retraining job runs.

Writes TWO artifacts per sub-model, matching the dual-format genesis
convention already established for this feature (see
aidlc-docs/construction/deal-yield-training-pipeline/functional-design/
business-logic-model.md Logic Flow 2):
    1. Native XGBoost JSON (what Triton's FIL backend actually loads):
       <output-dir>/<model_type>/1/xgboost.json
    2. ONNX (registry-bookkeeping only, mirrors the DLRM/NCF onnx-source/
       convention so register_genesis_models.py's existing
       _onnx_artifact_exists()/upload logic needs no format-specific
       branching):
       <output-dir>/<model_type>/1/model.onnx

Usage:
    python source/training/export_xgboost_genesis.py \
        --output-dir source/triton/onnx_export
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import xgboost as xgb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from training.xgboost_pipeline import MODEL_TYPE_BY_TARGET, TARGET_FLOOR, TARGET_MARGIN  # noqa: E402

# Matches features.py::FEATURE_VECTOR_LENGTH (source/containers/deal_yield_manager/
# features.py) -- kept as a literal here rather than importing that module,
# since this script must run standalone at deploy/build time without the
# full container package's runtime dependencies (tritonclient, etc.).
FEATURE_VECTOR_LENGTH = 7

# target -> genesis label, matching the existing no-op convention (BR-3/BR-4
# in deal-yield-model/functional-design/business-rules.md): a
# floor_multiplier of 1.0 means "no floor change"; a margin_value of 0.0
# means "no margin change".
_GENESIS_LABEL_BY_TARGET = {
    TARGET_FLOOR: 1.0,
    TARGET_MARGIN: 0.0,
}


def train_genesis_booster(target: str) -> xgb.Booster:
    """Trains a genesis (constant-output) booster for one target.

    A single training sample means XGBoost cannot find a real split, so
    the booster predicts the same label for every input -- verified
    (see module docstring) across arbitrary feature vectors, not just the
    training point itself.

    base_score is set explicitly to the target label rather than left to
    XGBoost's default estimation. Default base_score behavior differs
    across XGBoost versions (e.g. 1.7.x defaults to a fixed 0.5 and needs
    many boosting rounds to converge on a single-sample label; 3.x
    estimates it from the label mean and converges almost immediately) --
    pinning it directly makes the genesis (exact constant-output) property
    hold deterministically regardless of which XGBoost version trained it.
    """
    label = _GENESIS_LABEL_BY_TARGET[target]
    x_train = np.zeros((1, FEATURE_VECTOR_LENGTH), dtype=np.float32)
    y_train = np.array([label], dtype=np.float32)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        "max_depth": 2,
        "eta": 0.3,
        "objective": "reg:squarederror",
        "base_score": label,
    }
    return xgb.train(params, dtrain, num_boost_round=10)


def export_genesis_model(target: str, output_dir: str) -> tuple[str, str]:
    """Trains and writes both artifact forms for one target.

    Returns (xgboost_json_path, onnx_path).
    """
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.utils import save_model as save_onnx_model

    model_type = MODEL_TYPE_BY_TARGET[target]
    booster = train_genesis_booster(target)

    model_dir = os.path.join(output_dir, model_type, "1")
    os.makedirs(model_dir, exist_ok=True)

    xgboost_json_path = os.path.join(model_dir, "xgboost.json")
    booster.save_model(xgboost_json_path)

    onnx_model = convert_xgboost(
        booster, initial_types=[("input__0", FloatTensorType([None, FEATURE_VECTOR_LENGTH]))]
    )
    onnx_path = os.path.join(model_dir, "model.onnx")
    save_onnx_model(onnx_model, onnx_path)

    print(f"  Exported {model_type}: {xgboost_json_path}, {onnx_path}")
    return xgboost_json_path, onnx_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "triton", "onnx_export"),
        help="Base directory to write <model_type>/1/{xgboost.json,model.onnx} under "
        "(default: source/triton/onnx_export, matching export_models.py's convention).",
    )
    parser.add_argument(
        "--target",
        choices=[TARGET_FLOOR, TARGET_MARGIN, "all"],
        default="all",
    )
    args = parser.parse_args()

    targets = [TARGET_FLOOR, TARGET_MARGIN] if args.target == "all" else [args.target]
    for target in targets:
        export_genesis_model(target, args.output_dir)

    print(f"Done. Genesis XGBoost artifacts ready under {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
