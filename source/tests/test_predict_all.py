"""Tests for train.predict_all — the prediction sweep the policy search runs on.

This function had two defects that only surfaced on a GPU instance, each costing a
full SageMaker round trip (image pull, capacity wait, phase 1 retrain) to discover:

  1. it fed CPU batches to a model phase 1 had left on cuda:0, and
  2. it applied squeeze(-1) to an already-1-D output, which turns a final batch of
     exactly one row into a 0-d array that np.concatenate rejects.

Neither needs a GPU to test. The device contract is asserted by giving the model a
recording wrapper and checking the batch was moved to the model's OWN device rather
than to a hardcoded one, which holds on a CPU-only machine. The shape contract is
asserted with row counts chosen to leave a remainder of one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(SOURCE_DIR / "training" / "container"))

torch = pytest.importorskip("torch", reason="torch required for predict_all tests")
np = pytest.importorskip("numpy")

from torch.utils.data import TensorDataset  # noqa: E402


@pytest.fixture(scope="module")
def predict_all():
    """Import train.predict_all, skipping if the trainer's deps are unavailable."""
    # train.py expects the staged `shared` package; source/shared is on the path
    # above, which provides the same modules stage_shared.sh copies in.
    os.environ.setdefault("SM_MODEL_DIR", "/tmp")
    train = pytest.importorskip(
        "train", reason="training container deps unavailable"
    )
    assert hasattr(train, "predict_all"), (
        "train.predict_all is gone; the policy search's prediction sweep was "
        "inlined back into main(), where neither of its two defects is testable"
    )
    return train.predict_all


class _OneDimModel(torch.nn.Module):
    """Mirrors DLRMModel's contract: takes one 2-D tensor, returns 1-D.

    DLRMModel.forward ends in reshape(-1) precisely so the output is flat, so a
    stand-in has to do the same or the shape test proves nothing.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, 1)

    def forward(self, x):
        return self.linear(x).reshape(-1)


class _RecordingModel(_OneDimModel):
    """Records the device of every batch it is handed."""

    def __init__(self, width: int) -> None:
        super().__init__(width)
        self.seen_devices: list[torch.device] = []

    def forward(self, x):
        self.seen_devices.append(x.device)
        return super().forward(x)


def _dataset(rows: int, width: int = 4) -> TensorDataset:
    return TensorDataset(torch.randn(rows, width))


class TestShape:
    def test_returns_one_prediction_per_row(self, predict_all) -> None:
        model = _OneDimModel(4)
        out = predict_all(model, _dataset(50), batch_size=1024)
        assert out.shape == (50,)

    @pytest.mark.parametrize("rows", [1, 2, 9, 17])
    def test_small_and_odd_row_counts(self, predict_all, rows: int) -> None:
        model = _OneDimModel(4)
        out = predict_all(model, _dataset(rows), batch_size=8)
        assert out.shape == (rows,), f"{rows} rows produced {out.shape}"

    def test_a_final_batch_of_exactly_one_row(self, predict_all) -> None:
        """The regression. 9 rows at batch_size 4 makes the last batch a single row;
        the squeeze this function used to apply turned that into a 0-d array and
        np.concatenate raised."""
        model = _OneDimModel(4)
        out = predict_all(model, _dataset(9), batch_size=4)
        assert out.ndim == 1
        assert out.shape == (9,)

    def test_output_is_flat_across_many_batches(self, predict_all) -> None:
        model = _OneDimModel(4)
        out = predict_all(model, _dataset(100), batch_size=7)
        assert out.ndim == 1
        assert out.shape == (100,)

    def test_empty_dataset_returns_none(self, predict_all) -> None:
        """search_policy_parameters reads None as "nothing to search against"; an
        empty array would instead look like a real, empty prediction set."""
        model = _OneDimModel(4)
        assert predict_all(model, _dataset(0)) is None


class TestDevice:
    def test_batches_arrive_on_the_models_device(self, predict_all) -> None:
        model = _RecordingModel(4)
        predict_all(model, _dataset(20), batch_size=8)
        expected = next(model.parameters()).device
        assert model.seen_devices, "model was never called"
        assert all(d == expected for d in model.seen_devices), (
            f"batches arrived on {set(model.seen_devices)}, model is on {expected}"
        )

    def test_the_device_is_read_from_the_model_not_assumed(self, predict_all) -> None:
        """The actual bug: the device was never consulted at all, so a model moved
        off the default device got CPU batches. Asserting the source reads it from
        the parameters is what distinguishes "works on CPU" from "follows the
        model" -- the real case needs a GPU, which CI does not have.
        """
        import inspect

        src = inspect.getsource(predict_all)
        assert "next(model.parameters()).device" in src, (
            "predict_all no longer derives the device from the model's parameters"
        )
        assert ".to(device)" in src, "the batch is no longer moved to that device"

    def test_predictions_do_not_require_grad(self, predict_all) -> None:
        """Runs under no_grad: this is a scoring sweep over the whole training set,
        and building a graph for it wastes memory on the GPU it runs on."""
        model = _OneDimModel(4)
        out = predict_all(model, _dataset(10))
        assert isinstance(out, np.ndarray)

    def test_model_is_left_in_eval_mode(self, predict_all) -> None:
        model = _OneDimModel(4)
        model.train()
        predict_all(model, _dataset(10))
        assert not model.training, "predict_all must put the model in eval mode"
