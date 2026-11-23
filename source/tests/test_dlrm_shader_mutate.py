"""BID_SHADE mutate(): the shaded price stays within [floor, original bid]."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

import containers.dlrm_bid_shader.app as dlrm_app  # noqa: E402
from shared.artf_types import RTBRequest  # noqa: E402


def _request(price: float, floor: float, model_params: dict | None = None) -> RTBRequest:
    body = {
        "id": "shade-test",
        "lifecycle": "LIFECYCLE_DSP_BID_RESPONSE",
        "applicable_intents": ["BID_SHADE"],
        "bid_request": {
            "id": "auction-1",
            "imp": [{"id": "imp-1", "banner": {"w": 300, "h": 250}, "bidfloor": floor}],
            "site": {"domain": "news.example"},
        },
        "bid_response": {
            "id": "auction-1",
            "seatbid": [{"seat": "seat-1", "bid": [{"id": "bid-1", "impid": "imp-1", "price": price}]}],
        },
    }
    if model_params is not None:
        body["model_params"] = model_params
    return RTBRequest(**body)


@pytest.fixture(autouse=True)
def static_defaults(monkeypatch):
    """No parameter store, so the container defaults (0.65, 12.0) apply."""
    monkeypatch.delenv("PARAMETER_STORE_TABLE", raising=False)
    monkeypatch.setattr(dlrm_app, "_parameter_cache", None)
    monkeypatch.setattr(dlrm_app, "_parameter_cache_initialized", False)


@pytest.fixture
def ctr(monkeypatch):
    """Fix the prediction so the test controls the expected value."""
    def _set(value: float):
        monkeypatch.setattr(dlrm_app, "_predict_ctr_from_request", lambda _req: (value, "", ""))
    return _set


def test_a_bid_below_the_floor_is_never_raised(ctr):
    ctr(0.9)
    resp = dlrm_app.mutate(_request(price=1.50, floor=2.00))
    assert resp.mutations == []


def test_a_bid_above_the_floor_is_shaded_down(ctr):
    ctr(0.5)  # ev = 0.5 x 12.0 = 6.0, price = 6.0 x 0.65 = 3.90
    resp = dlrm_app.mutate(_request(price=10.00, floor=0.50))
    assert len(resp.mutations) == 1
    assert resp.mutations[0].adjust_bid.price == pytest.approx(3.90)


def test_the_shaded_price_is_held_at_the_floor(ctr):
    ctr(0.01)  # ev x slope = 0.078, under the floor
    resp = dlrm_app.mutate(_request(price=5.00, floor=1.00))
    assert resp.mutations[0].adjust_bid.price == pytest.approx(1.00)


def test_a_change_within_one_cent_emits_no_mutation(ctr):
    ctr(0.5)
    resp = dlrm_app.mutate(
        _request(price=3.905, floor=0.10, model_params={"shade_factor": 0.65, "conversion_value": 12.0})
    )
    assert resp.mutations == []


def test_the_bid_pricer_image_installs_the_sdk_its_parameter_cache_imports():
    """The image is built from triton/Dockerfile.triton-artf, which installs only the
    container's own requirements.txt. Without boto3 there, every ParameterCache
    refresh failed with "No module named 'boto3'" and the Adaptive Bidding agent's
    parameter updates never reached the bid path."""
    root = os.path.join(os.path.dirname(__file__), "..")
    req = os.path.join(root, "containers", "dlrm_bid_shader", "requirements.txt")
    assert os.path.exists(req)
    assert any(line.split("==")[0].strip() == "boto3" for line in open(req))
    dockerfile = open(os.path.join(root, "triton", "Dockerfile.triton-artf")).read()
    assert "pip install --no-cache-dir -r /app/container/requirements.txt" in dockerfile
