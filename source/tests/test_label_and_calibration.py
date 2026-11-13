"""Item 2: what the model is trained to predict, and whether it stays a probability.

Three defects sat behind the shading arithmetic, and none of them made a run fail:

1. **The label was a profitable win.** `label` is 1 when the bid won AND
   `conversion_value > price_paid`, so it folds the response and the price into one
   bit. `ev = p x conversion_value` needs p to be the probability of the RESPONSE the
   advertiser pays for; multiplying a "was this cheap enough to be worth winning"
   score by a conversion value is not an expected value of anything.

2. **Phase 2 retrained the probability head.** The RL phase took
   `model.parameters()` and optimised an ROI reward, which has no term that rewards
   calibration. Phase 1's entire purpose is to make the output a probability, and
   phase 2 ran second.

3. **Nothing measured whether the output was calibrated**, and nothing refused a
   dataset that could not teach it — a single-class label or a constant categorical
   trains cleanly to a falling loss that means nothing.

These tests cover the three fixes plus the label plumbing, and the abstention that
replaced a hardcoded CTR of 0.5.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))
# Appended, not prepended — see tests/test_train_container.py for why.
sys.path.append(str(SOURCE_DIR / "training" / "container"))

from shared import dlrm_features  # noqa: E402

from train import (  # noqa: E402
    DatasetUnfitError,
    _DEFAULT_OBJECTIVE,
    _OBJECTIVE_LABELS,
    calibration_report,
    policy_parameters_of,
    resolve_label_column,
    rl_finetune,
    validate_dataset,
)


def _fit_df(n: int = 40, label_column: str = "label_conversion") -> pd.DataFrame:
    """A dataset the gate accepts: two label classes, varied categoricals."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "bid_floor": rng.uniform(0.5, 5.0, n),
            "hour_of_day": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "has_video": rng.integers(0, 2, n).astype(bool),
            "site_domain": [f"site{i % 5}.com" for i in range(n)],
            "device_type": [int(i % 3) + 1 for i in range(n)],
            "geo_country": [["USA", "CAN", "GBR"][i % 3] for i in range(n)],
            label_column: [i % 2 for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# 2.1 / 2.2 — which response the run is trained on
# ---------------------------------------------------------------------------

class TestObjectiveSelectsTheLabel:
    def test_the_default_objective_is_the_response_an_advertiser_pays_for(self) -> None:
        objective, column = resolve_label_column({})

        assert objective == _DEFAULT_OBJECTIVE == "cpa"
        assert column == "label_conversion"

    def test_cpc_trains_on_a_click(self) -> None:
        assert resolve_label_column({"objective": "cpc"}) == ("cpc", "label_click")

    def test_the_legacy_profitable_win_label_stays_selectable(self) -> None:
        """Kept so a like-for-like comparison against a previously trained model is
        possible — not because it is a good objective."""
        assert resolve_label_column({"objective": "profitable_win"}) == (
            "profitable_win",
            "label",
        )

    def test_an_unrecognised_objective_is_refused_not_defaulted(self) -> None:
        """Silently training on a different label than the caller asked for is the
        class of defect this work removes."""
        with pytest.raises(DatasetUnfitError, match="not one of"):
            resolve_label_column({"objective": "whatever"})

    def test_case_and_whitespace_are_tolerated(self) -> None:
        assert resolve_label_column({"objective": "  CPA "})[0] == "cpa"

    def test_every_objective_maps_to_a_distinct_column(self) -> None:
        assert len(set(_OBJECTIVE_LABELS.values())) == len(_OBJECTIVE_LABELS)


class TestEtlEmitsTheResponseLabels:
    """The ETL is PySpark, so these assert the source rather than run it.

    Running it needs a SparkSession, which is not a dependency of this test suite.
    What matters here is the decision the code encodes: NULL for an unlabelled row,
    a real 0 for a lost impression, and `label` untouched.
    """

    def test_both_response_labels_are_added(self) -> None:
        text = (SOURCE_DIR / "etl" / "glue_feature_engineering.py").read_text()
        # Asserted as the (column, signal) argument pair rather than a whole call
        # expression: _add_response_label also takes an attribution deadline, and the
        # conversion call is wrapped across lines, so pinning the full call text made
        # this fail on formatting alone.
        assert '"label_conversion", "conversion"' in text
        assert '"label_click", "click"' in text
        assert text.count("_add_response_label(") >= 3  # 1 def + 2 call sites

    def test_an_unresolved_signal_is_null_not_zero(self) -> None:
        """0 would make an unlabelled dataset look like a dataset of negatives, which
        trains the model to predict zero at a healthy-looking loss.

        "Unresolved" is the precise claim, and it is narrower than "absent". Absence is
        only unlabelled while the row's attribution window is still open; see
        test_an_absent_signal_past_the_deadline_is_a_real_zero for the other half. What
        must never become a 0 is a row nothing has reported on at all.
        """
        text = (SOURCE_DIR / "etl" / "glue_feature_engineering.py").read_text()
        # The absent branch routes through `absent`, which is NULL unless a deadline
        # was supplied...
        assert ".when(F.col(signal).isNull(), absent)" in text
        # ...and is NULL unconditionally when it was not, so a caller that has not
        # opted into deadline labelling keeps the original behaviour exactly.
        assert "absent = F.lit(None).cast(IntegerType())" in text

    def test_an_absent_signal_past_the_deadline_is_a_real_zero(self) -> None:
        """Downstream signals only ever report events that HAPPENED -- there is no "no
        click" message from a pixel or from the outcome simulator, because a non-event
        cannot be observed. So without this rule every labelled row is a 1, the label
        has a single class, and the trainer's dataset gate refuses the run.

        The 0 is guarded twice: the bid must have been WON (an unresolved row has no
        impression for a response to have followed) and its attribution window must
        have closed.
        """
        text = (SOURCE_DIR / "etl" / "glue_feature_engineering.py").read_text()
        assert 'F.col("won") == True' in text
        assert 'F.col("timestamp") <= F.lit(attribution_deadline_epoch)' in text

    def test_a_lost_impression_is_a_real_zero(self) -> None:
        """A lost impression cannot have produced a response, so this is known, not
        missing."""
        text = (SOURCE_DIR / "etl" / "glue_feature_engineering.py").read_text()
        assert 'F.when(F.col("won") == False, F.lit(0))' in text

    def test_an_absent_signal_column_still_emits_the_label(self) -> None:
        """All-NULL, so the trainer's gate reports "no labelled rows" rather than the
        trainer dying on a missing column and leaving the cause to be guessed."""
        text = (SOURCE_DIR / "etl" / "glue_feature_engineering.py").read_text()
        assert "if signal not in df.columns:" in text

    def test_the_legacy_label_is_unchanged(self) -> None:
        text = (SOURCE_DIR / "etl" / "glue_feature_engineering.py").read_text()
        assert 'F.col("conversion_value") > F.col("price_paid")' in text


# ---------------------------------------------------------------------------
# 2.4 — the dataset gate
# ---------------------------------------------------------------------------

class TestDatasetGate:
    def test_accepts_a_fit_dataset_and_reports_what_it_measured(self) -> None:
        report = validate_dataset(_fit_df(), "label_conversion", "cpa")

        assert report["labelled_rows"] == 40
        assert report["positive_rate"] == 0.5
        assert report["categorical_cardinality"] == {
            "site_domain": 5,
            "device_type": 3,
            "geo_country": 3,
        }

    def test_refuses_a_missing_label_column(self) -> None:
        df = _fit_df().drop(columns=["label_conversion"])

        with pytest.raises(DatasetUnfitError) as exc:
            validate_dataset(df, "label_conversion", "cpa")

        assert "Re-run the Glue ETL" in str(exc.value)

    def test_refuses_an_entirely_unlabelled_dataset(self) -> None:
        """Every label NULL: the emit path recorded no response signal."""
        df = _fit_df()
        df["label_conversion"] = None

        with pytest.raises(DatasetUnfitError, match="none of them is"):
            validate_dataset(df, "label_conversion", "cpa")

    def test_refuses_an_all_negative_label(self) -> None:
        """The state the live pipeline is actually in — `emit_bid_outcome` writes
        conversion=False for every event."""
        df = _fit_df()
        df["label_conversion"] = 0

        with pytest.raises(DatasetUnfitError) as exc:
            validate_dataset(df, "label_conversion", "cpa")

        assert "no positives" in str(exc.value)
        assert "means nothing" in str(exc.value)

    def test_refuses_an_all_positive_label(self) -> None:
        df = _fit_df()
        df["label_conversion"] = 1

        with pytest.raises(DatasetUnfitError, match="no negatives"):
            validate_dataset(df, "label_conversion", "cpa")

    def test_counts_unlabelled_rows_without_refusing_when_some_are_labelled(self) -> None:
        df = _fit_df()
        df.loc[0:9, "label_conversion"] = None

        report = validate_dataset(df, "label_conversion", "cpa")

        assert report["unlabelled_rows"] == 10
        assert report["labelled_rows"] == 30

    def test_refuses_a_constant_categorical(self) -> None:
        """The load-test generator emitted site_domain="load-test" for every row,
        which made two of the three categoricals constant."""
        df = _fit_df()
        df["site_domain"] = "load-test"

        with pytest.raises(DatasetUnfitError) as exc:
            validate_dataset(df, "label_conversion", "cpa")

        assert "carries no information" in str(exc.value)

    def test_refuses_an_all_unique_categorical(self) -> None:
        """One row per embedding value memorises rather than learns."""
        df = _fit_df()
        df["site_domain"] = [f"unique{i}.com" for i in range(len(df))]

        with pytest.raises(DatasetUnfitError, match="one distinct value per row"):
            validate_dataset(df, "label_conversion", "cpa")

    def test_refuses_a_missing_categorical_feature(self) -> None:
        df = _fit_df().drop(columns=["geo_country"])

        with pytest.raises(DatasetUnfitError) as exc:
            validate_dataset(df, "label_conversion", "cpa")

        assert "geo_country" in str(exc.value)
        assert str(dlrm_features.FEATURE_SPEC_VERSION) in str(exc.value)

    def test_the_gate_names_the_feature_spec_columns_it_requires(self) -> None:
        """So a failure is actionable without reading the spec module."""
        df = _fit_df().drop(columns=["device_type"])

        with pytest.raises(DatasetUnfitError) as exc:
            validate_dataset(df, "label_conversion", "cpa")

        for column in dlrm_features.CATEGORICAL_COLUMNS:
            assert column in str(exc.value)


# ---------------------------------------------------------------------------
# 2.3 — the probability head is off-limits in phase 2
# ---------------------------------------------------------------------------

class _ProbabilityOnly(nn.Module):
    """Shaped like the DLRM: every parameter contributes to the probability."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 1)

    def forward(self, x):  # noqa: D102
        return self.head(x).reshape(-1)


class _WithPolicy(_ProbabilityOnly):
    """A model that declares policy parameters, to show the gate is not a blanket
    refusal to ever run phase 2.

    `policy` scales the score phase 2 optimises, so it receives a gradient — the
    shape a real policy head would have. `head` still holds the probability, and the
    test asserts phase 2 moves the first and not the second.
    """

    def __init__(self) -> None:
        super().__init__()
        self.policy = nn.Parameter(torch.ones(1))

    def forward(self, x):  # noqa: D102
        return super().forward(x) * self.policy

    def policy_parameters(self):  # noqa: D102
        return [self.policy]


class TestPolicyParameters:
    def test_a_model_with_no_declaration_has_no_policy_parameters(self) -> None:
        """Defaulting to model.parameters() is what the bug was: it reads as a
        reasonable default and happens to mean "including the probability head"."""
        assert policy_parameters_of(_ProbabilityOnly()) == []

    def test_a_declared_policy_is_returned(self) -> None:
        model = _WithPolicy()

        assert [id(p) for p in policy_parameters_of(model)] == [id(model.policy)]

    def test_the_dlrm_declares_none(self) -> None:
        from models import DLRMModel

        assert policy_parameters_of(DLRMModel()) == []


@pytest.fixture
def unshadow_triton(monkeypatch: pytest.MonkeyPatch):
    """Drop `source/` from sys.path so `import triton` finds NVIDIA's package.

    `source/triton/` is this repo's Triton model-repository tooling, and with
    `source/` on sys.path it shadows the `triton` PyPI package that PyTorch imports.
    Constructing a `torch.optim.Adam` pulls in `torch._dynamo`, which imports
    `triton.language` and dies with `module 'triton' has no attribute 'language'` —
    a failure that names neither this repo nor the shadowing.

    Everything these tests need from `source/` (`shared`, `train`, `models`) is
    already imported by the time a test body runs, so removing the entry is safe.
    Phase 1 and phase 2 both construct an optimiser, so any test that exercises them
    needs this.

    Entries are compared by RESOLVED path, not by string: other test modules add
    `source/` under forms like `<dir>/tests/..`, which is the same directory and a
    different string. Comparing strings made this pass in isolation and fail in the
    full suite.
    """
    kept = []
    for entry in sys.path:
        try:
            same = Path(entry or ".").resolve() == SOURCE_DIR
        except (OSError, ValueError):
            same = False
        if not same:
            kept.append(entry)
    monkeypatch.setattr(sys, "path", kept, raising=False)
    for name in [n for n in sys.modules if n == "triton" or n.startswith("triton.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return None


class TestPhase2LeavesTheProbabilityHeadAlone:
    def _loader(self) -> DataLoader:
        features = torch.randn(8, 4)
        labels = torch.randint(0, 2, (8,)).float()
        outcomes = torch.tensor([[1.0, 2.0, 5.0]] * 8)
        return DataLoader(TensorDataset(features, labels, outcomes), batch_size=4)

    def _hp(self) -> dict:
        return {"rl_epochs": 2, "reward_function": "roi", "rl_learning_rate": 1e-3}

    def test_weights_are_bit_identical_after_phase_2(self) -> None:
        """The assertion that matters. Before this change, phase 2 optimised
        model.parameters() against an ROI reward, so the calibrated probability from
        phase 1 was moved by an objective with no calibration term."""
        model = _ProbabilityOnly()
        before = {k: v.clone() for k, v in model.state_dict().items()}

        rl_finetune(model, self._loader(), self._hp())

        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), f"phase 2 changed {key}"

    def test_the_skip_is_reported_with_a_reason(self) -> None:
        """Silently doing nothing and doing nothing for a stated reason are different
        facts, and the manifest records which."""
        metrics = rl_finetune(_ProbabilityOnly(), self._loader(), self._hp())

        assert metrics["rl_skipped"] is True
        reason = metrics["rl_skipped_reason"]
        # Asserted on substance, not on a fixed phrase. The reason has to say what the
        # policy is and why gradient descent must not reach it; an earlier version
        # pinned the words "Parameter Store", which went stale when the reason was
        # rewritten to name the module and search function that actually set the
        # coefficients.
        assert "polic" in reason.lower()
        assert "calibration" in reason.lower()
        assert "shading_policy" in reason
        assert "calibration" in metrics["rl_skipped_reason"]

    def test_a_model_with_a_policy_does_run_phase_2(self, unshadow_triton) -> None:
        """Not a blanket disable: a declared policy parameter is optimised, and the
        probability head still is not."""
        model = _WithPolicy()
        head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
        policy_before = model.policy.clone()

        metrics = rl_finetune(model, self._loader(), self._hp())

        assert "rl_skipped" not in metrics
        for key, value in model.head.state_dict().items():
            assert torch.equal(value, head_before[key]), f"phase 2 changed head.{key}"
        assert not torch.equal(model.policy, policy_before), "the policy was not trained"


# ---------------------------------------------------------------------------
# 2.5 — calibration is measured
# ---------------------------------------------------------------------------

class _FixedLogits(nn.Module):
    """Returns a logit read from the input, so a test can pin the probability."""

    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # noqa: D102
        return x[:, 0]


def _calibration_loader(logits: list[float], labels: list[float]) -> DataLoader:
    features = torch.tensor([[v] for v in logits], dtype=torch.float32)
    return DataLoader(
        TensorDataset(features, torch.tensor(labels, dtype=torch.float32)),
        batch_size=4,
    )


class TestCalibrationReport:
    def test_a_perfectly_calibrated_model_scores_near_zero(self) -> None:
        """Half the rows at p=0.5, half positive: predicted and observed agree."""
        logit = 0.0  # sigmoid(0) == 0.5
        loader = _calibration_loader([logit] * 100, [1.0] * 50 + [0.0] * 50)

        report = calibration_report(_FixedLogits(), loader)

        assert report["calibration"] == "measured"
        assert report["expected_calibration_error"] < 0.01
        assert report["mean_predicted"] == pytest.approx(0.5, abs=1e-4)
        assert report["observed_base_rate"] == pytest.approx(0.5, abs=1e-4)

    def test_a_miscalibrated_model_is_reported_as_such(self) -> None:
        """Confidently wrong: p=0.5 on rows that never convert. A model can rank
        perfectly and still be this badly calibrated, which is why ranking metrics
        do not answer the question."""
        loader = _calibration_loader([0.0] * 100, [0.0] * 100)

        report = calibration_report(_FixedLogits(), loader)

        assert report["expected_calibration_error"] == pytest.approx(0.5, abs=1e-3)
        assert report["max_bin_gap"] == pytest.approx(0.5, abs=1e-3)

    def test_it_measures_the_probability_that_is_served(self) -> None:
        """The training forward returns logits; the served graph applies the sigmoid.
        Measuring the logits would report a calibration error for a quantity nobody
        serves."""
        loader = _calibration_loader([2.0] * 40, [1.0] * 40)
        expected = float(torch.sigmoid(torch.tensor(2.0)))

        report = calibration_report(_FixedLogits(), loader)

        assert report["mean_predicted"] == pytest.approx(expected, abs=1e-4)
        assert 0.0 <= report["mean_predicted"] <= 1.0

    def test_empty_bins_are_reported_as_empty_not_as_agreement(self) -> None:
        """A bin with no samples has no gap to report; counting it as 0 would
        flatter the error."""
        loader = _calibration_loader([0.0] * 20, [1.0] * 10 + [0.0] * 10)

        report = calibration_report(_FixedLogits(), loader, n_bins=10)

        populated = [b for b in report["bins"] if b["count"]]
        assert len(populated) == 1
        assert all("gap" not in b for b in report["bins"] if not b["count"])

    def test_an_empty_split_is_unmeasured_not_zero(self) -> None:
        """Reporting a calibration error of 0 for a model nothing was measured on
        would be a fabricated verification."""
        loader = DataLoader(
            TensorDataset(torch.empty(0, 1), torch.empty(0)), batch_size=4
        )

        report = calibration_report(_FixedLogits(), loader)

        assert report["calibration"] == "unmeasured"
        assert "expected_calibration_error" not in report

    def test_p_equal_to_one_is_counted(self) -> None:
        """The last bin's upper edge is inclusive, so a saturated probability is not
        silently dropped from the measurement."""
        loader = _calibration_loader([40.0] * 20, [1.0] * 20)

        report = calibration_report(_FixedLogits(), loader)

        assert report["n_samples"] == 20
        assert sum(b["count"] for b in report["bins"]) == 20
