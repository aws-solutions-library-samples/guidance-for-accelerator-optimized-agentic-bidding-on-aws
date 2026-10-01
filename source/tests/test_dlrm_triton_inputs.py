"""The Triton call sends the spec's inputs, under the spec's names, in order.


`predict_ctr` used to take four positionally-named arrays — `dense_features`,
`sparse_user`, `sparse_domain`, `sparse_device` — which named a feature set the
spec does not have. It now takes the spec's `(dense, categorical)` pair and
reads the input names from `dlrm_features.TRITON_CATEGORICAL_INPUTS`, so a
tensor cannot be sent under another feature's name.

The container's own module imports `container.triton_inference`, a path that
only resolves inside the built image, so these tests import the module by file
path and stub the client.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from shared import dlrm_features as F

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "containers"
    / "dlrm_bid_shader"
    / "triton_inference.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_dlrm_triton", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def __init__(self, ctr=0.25):
        self._ctr = ctr

    def as_numpy(self, name):
        if name == "ctr_prediction":
            return np.array([[self._ctr]], dtype=np.float32)
        return np.array([b"stable"], dtype=object)


class _RecordingClient:
    """Captures what was sent instead of reaching a server."""

    def __init__(self):
        self.calls = []

    def infer(self, model_name, inputs, outputs):
        self.calls.append(
            {
                "model_name": model_name,
                "input_names": [i.name() for i in inputs],
                "output_names": [o.name() for o in outputs],
            }
        )
        return _FakeResult()


@pytest.fixture
def module_and_client(monkeypatch):
    module = _load_module()
    client = _RecordingClient()
    monkeypatch.setattr(module, "_get_client", lambda: client)
    return module, client


def _vectors():
    dense, categorical = F.build_from_bid_request(
        {
            "imp": [{"id": "i1", "bidfloor": 2.5}],
            "site": {"domain": "espn.com"},
            "device": {"devicetype": 4, "geo": {"country": "CAN"}},
        }
    )
    return (
        np.array([dense], dtype=np.float32),
        [np.array([[i]], dtype=np.int64) for i in categorical],
    )


# ---------------------------------------------------------------------------
# Input naming and ordering
# ---------------------------------------------------------------------------

def test_input_names_come_from_the_spec(module_and_client):
    module, client = module_and_client
    dense, categorical = _vectors()

    module.predict_ctr(dense=dense, categorical=categorical)

    sent = client.calls[0]["input_names"]
    expected = [F.TRITON_DENSE_INPUT, *F.TRITON_CATEGORICAL_INPUTS]
    assert sent == expected


def test_categorical_inputs_are_named_after_their_features(module_and_client):
    """The old names described a different feature set."""
    module, client = module_and_client
    dense, categorical = _vectors()

    module.predict_ctr(dense=dense, categorical=categorical)

    sent = client.calls[0]["input_names"]
    for column in F.CATEGORICAL_COLUMNS:
        assert f"sparse_{column}" in sent
    for stale in ("sparse_user", "sparse_domain", "sparse_device"):
        assert stale not in sent


def test_one_input_per_declared_feature(module_and_client):
    module, client = module_and_client
    dense, categorical = _vectors()

    module.predict_ctr(dense=dense, categorical=categorical)

    assert len(client.calls[0]["input_names"]) == 1 + F.CATEGORICAL_WIDTH


def test_a_count_mismatch_raises_rather_than_mislabelling(module_and_client):
    """Sending fewer arrays than the spec declares would put a tensor under
    another feature's input name."""
    module, _ = module_and_client
    dense, categorical = _vectors()

    with pytest.raises(ValueError, match="categorical"):
        module.predict_ctr(dense=dense, categorical=categorical[:-1])

    with pytest.raises(ValueError, match="categorical"):
        module.predict_ctr(dense=dense, categorical=categorical + categorical[:1])


# ---------------------------------------------------------------------------
# The variant override stays load-test only
# ---------------------------------------------------------------------------

def test_variant_input_is_absent_for_live_traffic(module_and_client):
    module, client = module_and_client
    dense, categorical = _vectors()

    module.predict_ctr(dense=dense, categorical=categorical, target_variant=None)

    assert "target_variant" not in client.calls[0]["input_names"]


@pytest.mark.parametrize("variant", ["stable", "canary"])
def test_variant_input_is_sent_when_a_load_test_forces_one(module_and_client, variant):
    module, client = module_and_client
    dense, categorical = _vectors()

    module.predict_ctr(dense=dense, categorical=categorical, target_variant=variant)

    assert "target_variant" in client.calls[0]["input_names"]


def test_an_unrecognised_variant_is_not_forwarded(module_and_client):
    module, client = module_and_client
    dense, categorical = _vectors()

    module.predict_ctr(dense=dense, categorical=categorical, target_variant="bogus")

    assert "target_variant" not in client.calls[0]["input_names"]


# ---------------------------------------------------------------------------
# Shapes match the spec's declared widths
# ---------------------------------------------------------------------------

def test_dense_array_width_matches_the_spec():
    dense, categorical = _vectors()
    assert dense.shape == (1, F.DENSE_WIDTH)
    assert len(categorical) == F.CATEGORICAL_WIDTH
    for array in categorical:
        assert array.shape == (1, 1)
