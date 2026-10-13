"""Tests for the ONNX export compatibility shim.

A training job failed at `torch.onnx.export(..., dynamo=False)` AFTER phase 1 had
completed and calibration had been measured — the training image runs a torch older
than 2.6, where that keyword does not exist. The run was discarded for a keyword
argument.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from shared.onnx_compat import onnx_export_kwargs, supports_dynamo_flag  # noqa: E402


class TestKwargFiltering:
    def test_keeps_a_supported_argument(self, monkeypatch):
        def fake_export(model, args, f, *, opset_version=None, dynamo=False):
            ...

        import torch

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        assert onnx_export_kwargs(dynamo=False) == {"dynamo": False}

    def test_drops_an_unsupported_argument(self, monkeypatch):
        """The exact shape of the failure: an older torch with no `dynamo`."""

        def fake_export(model, args, f, *, opset_version=None):
            ...

        import torch

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        assert onnx_export_kwargs(dynamo=False) == {}

    def test_passes_everything_through_for_a_var_keyword_signature(self, monkeypatch):
        def fake_export(model, args, f, **kwargs):
            ...

        import torch

        monkeypatch.setattr(torch.onnx, "export", fake_export)
        assert onnx_export_kwargs(dynamo=False) == {"dynamo": False}

    def test_reports_support_on_the_installed_torch(self):
        assert isinstance(supports_dynamo_flag(), bool)


class TestCallSites:
    """Neither exporter may pass the keyword unconditionally again."""

    @pytest.mark.parametrize(
        "relative",
        ["training/container/train.py", "triton/export_models.py"],
    )
    def test_no_unconditional_dynamo_kwarg(self, relative):
        source = (_SOURCE / relative).read_text()
        assert "dynamo=False,\n" not in source.replace(
            "onnx_export_kwargs(dynamo=False),\n", ""
        )
        assert "onnx_compat.onnx_export_kwargs(dynamo=False)" in source

    def test_the_shim_still_requests_the_tracing_exporter(self):
        """Dropping the argument entirely would switch exporter on a newer torch.

        The dynamo exporter emits a different graph than the Triton config and the
        Model Optimizer expect, so `dynamo=False` must still be ASKED for — the shim
        only removes it where torch cannot accept it.
        """
        shim = (_SOURCE / "shared" / "onnx_compat.py").read_text()
        assert "dynamo" in shim
        train = (_SOURCE / "training" / "container" / "train.py").read_text()
        assert "dynamo=False" in train
