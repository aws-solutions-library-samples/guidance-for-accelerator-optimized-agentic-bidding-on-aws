"""The policy search must score every row of the dataframe it is given.

`search_policy_parameters` aligns per-row predictions onto its `scorable` subset by
index, so it requires `len(predictions) == len(df)`. The call site scored
`train_loader.dataset` instead — 225 of 249 rows at a 10% validation split — so the
guard refused on every run, the search never executed, and the shader kept serving
GENESIS_POLICY however often it was retrained. The only symptom was a field in the
training log:

    {"policy_searched": false,
     "policy_skipped_reason": "no aligned model predictions, so expected value cannot
                               be computed per row"}

Every test here uses a row count that does NOT divide evenly by the validation split,
because that is the shape the defect needed: 249 rows at 10% gives val=24, train=225.
A row count that divides evenly would still mismatch, but reproducing the original
arithmetic keeps the regression honest.

These run on CPU with no SageMaker round trip. The defect previously cost ~25 minutes
per observation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import TensorDataset  # noqa: E402

TRAINING_CONTAINER = (
    Path(__file__).resolve().parents[1] / "training" / "container"
)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(SOURCE_ROOT), str(TRAINING_CONTAINER)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

train = pytest.importorskip("train")

# The row count that produced the defect.
N_ROWS = 249
VALIDATION_SPLIT = 0.1


def _split_sizes(n: int = N_ROWS, split: float = VALIDATION_SPLIT) -> tuple[int, int]:
    """Mirror train.py's split arithmetic exactly."""
    val_size = int(n * split)
    return n - val_size, val_size


def _scorable_df(n: int = N_ROWS, provenance: str | None = "simulated") -> pd.DataFrame:
    """A dataframe every row of which the surrogate can score.

    Every row is won with a recorded price_paid, so `scorable` == the whole frame and
    the in-sample/out-of-sample counts are predictable from the split alone.
    """
    rng = np.random.default_rng(1234)
    data = {
        "won": [True] * n,
        "price_paid": rng.uniform(1.0, 5.0, n),
        "bid_floor": rng.uniform(0.5, 2.0, n),
        "original_price": rng.uniform(5.0, 9.0, n),
        "conversion_value": rng.uniform(0.0, 20.0, n),
    }
    if provenance is not None:
        data["outcome_provenance"] = [provenance] * n
    return pd.DataFrame(data)


def _in_sample_mask(df: pd.DataFrame) -> pd.Series:
    train_size, _ = _split_sizes(len(df))
    return pd.Series(
        [True] * train_size + [False] * (len(df) - train_size), index=df.index
    )


class _ConstantModel(torch.nn.Module):
    """Returns a fixed probability per row, shaped like DLRMModel.forward's output
    (1-D after its reshape(-1)), so predict_all's real batching path is exercised."""

    def __init__(self, value: float = 0.4) -> None:
        super().__init__()
        self.value = value
        # predict_all reads the device off the parameters, so there must be one.
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.value) + self.weight * 0


class TestPredictionsCoverTheWholeDataframe:
    def test_scoring_the_whole_frame_yields_one_prediction_per_row(self):
        df = _scorable_df()
        features = torch.randn(len(df), 4)
        preds = train.predict_all(_ConstantModel(), TensorDataset(features))
        assert preds is not None
        assert len(preds) == len(df) == N_ROWS

    def test_scoring_the_training_split_does_not(self):
        """The defect's arithmetic, pinned. If this ever equals len(df), the split
        changed and the guard below stops being the thing under test."""
        df = _scorable_df()
        train_size, val_size = _split_sizes(len(df))
        assert (train_size, val_size) == (225, 24)

        features = torch.randn(len(df), 4)
        train_ds = TensorDataset(features[:train_size])
        preds = train.predict_all(_ConstantModel(), train_ds)
        assert len(preds) == train_size
        assert len(preds) != len(df)

    def test_a_final_batch_of_one_row_still_concatenates(self):
        """DLRMModel.forward already ends in reshape(-1), so its output is 1-D. An
        extra squeeze(-1) collapsed a final batch of exactly one row to 0-d, which
        np.concatenate rejects — invisible unless the row count leaves a remainder
        of one."""
        features = torch.randn(1025, 4)
        preds = train.predict_all(
            _ConstantModel(), TensorDataset(features), batch_size=1024
        )
        assert preds is not None
        assert len(preds) == 1025


class TestTheSearchRunsWhenAligned:
    def test_aligned_predictions_are_not_refused(self):
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {}, in_sample=_in_sample_mask(df))

        assert report["policy_searched"] is True
        assert "policy_skipped_reason" not in report
        assert report["policy_search_method"] == "offline_surrogate"
        assert set(report["policy"]) >= {"base", "slope", "curvature"}

    def test_misaligned_predictions_are_still_refused(self):
        """The guard itself must keep working — this is what fails against the old
        call site, and what would fail again if someone reverted it."""
        df = _scorable_df()
        train_size, _ = _split_sizes(len(df))
        preds = np.full(train_size, 0.4)

        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_searched"] is False
        assert "no aligned model predictions" in report["policy_skipped_reason"]

    def test_no_predictions_at_all_is_refused(self):
        df = _scorable_df()
        report = train.search_policy_parameters(df, None, {})
        assert report["policy_searched"] is False
        assert "no aligned model predictions" in report["policy_skipped_reason"]


class TestInSampleAccounting:
    def test_the_two_counts_partition_the_scored_rows(self):
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {}, in_sample=_in_sample_mask(df))

        assert (
            report["policy_search_in_sample_rows"]
            + report["policy_search_out_of_sample_rows"]
            == report["policy_search_rows"]
        )

    def test_the_counts_match_the_split(self):
        df = _scorable_df()
        train_size, val_size = _split_sizes(len(df))
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {}, in_sample=_in_sample_mask(df))

        # Every row here is scorable, so the counts are exactly the split.
        assert report["policy_search_in_sample_rows"] == train_size
        assert report["policy_search_out_of_sample_rows"] == val_size

    def test_only_scorable_rows_are_counted(self):
        """The mask covers df; the counts must describe the scored subset.

        Rows are made unscorable on BOTH sides of the split boundary so each count has
        to drop by its own amount — a version that only removed rows from one side would
        pass even if the implementation ignored the mask for the other.
        """
        df = _scorable_df()
        train_size, val_size = _split_sizes(N_ROWS)  # 225, 24
        # 10 in-sample rows (index 0-9) and 9 out-of-sample rows (the last 9, index
        # 240-248) become unscorable.
        df.loc[df.index[:10], "won"] = False
        df.loc[df.index[-9:], "won"] = False

        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {}, in_sample=_in_sample_mask(df))

        assert report["policy_search_in_sample_rows"] == train_size - 10 == 215
        assert report["policy_search_out_of_sample_rows"] == val_size - 9 == 15
        assert report["policy_search_rows"] == N_ROWS - 19 == 230

    def test_a_non_default_index_is_handled(self):
        """df is only reset_index'd when unlabelled rows were dropped, so a
        fully-labelled dataset arrives with its original index. Positional slicing
        would mismatch silently here."""
        df = _scorable_df()
        df.index = pd.RangeIndex(start=1000, stop=1000 + len(df))
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {}, in_sample=_in_sample_mask(df))

        train_size, val_size = _split_sizes(len(df))
        assert report["policy_search_in_sample_rows"] == train_size
        assert report["policy_search_out_of_sample_rows"] == val_size

    def test_the_counts_are_none_when_no_mask_is_given(self):
        """Absent information is reported as absent, not as zero."""
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_search_in_sample_rows"] is None
        assert report["policy_search_out_of_sample_rows"] is None


class TestAcceptanceGate:
    def test_an_unbeatable_genesis_is_not_accepted(self):
        """A margin the search cannot clear must leave genesis in place."""
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(
            df, preds, {"policy_search_min_improvement": 1e9}
        )

        assert report["policy_searched"] is True
        assert report["policy_accepted"] is False

    def test_a_rejected_search_returns_the_genesis_coefficients(self):
        from shared.shading_policy import GENESIS_POLICY

        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(
            df, preds, {"policy_search_min_improvement": 1e9}
        )
        assert report["policy"] == GENESIS_POLICY.as_dict()

    def test_searched_and_accepted_are_separate_facts(self):
        """A rejected search must remain distinguishable from one that never ran."""
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        rejected = train.search_policy_parameters(
            df, preds, {"policy_search_min_improvement": 1e9}
        )
        never_ran = train.search_policy_parameters(df, None, {})

        assert rejected["policy_searched"] is True
        assert rejected["policy_accepted"] is False
        assert never_ran["policy_searched"] is False
        assert "policy_accepted" not in never_ran

    def test_improvement_is_reported_and_consistent(self):
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_improvement"] == pytest.approx(
            report["policy_best_score"] - report["policy_genesis_score"], abs=1e-6
        )
        assert report["policy_improvement"] >= 0.0

    def test_the_margin_is_echoed_so_a_reviewer_sees_what_was_applied(self):
        df = _scorable_df()
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(
            df, preds, {"policy_search_min_improvement": 0.25}
        )
        assert report["policy_search_min_improvement"] == 0.25


class TestProvenanceOfScoredRows:
    def test_the_provenance_of_the_scored_subset_is_reported(self):
        df = _scorable_df(provenance="simulated")
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_search_outcome_provenance"] == {"simulated": N_ROWS}

    def test_it_describes_the_scored_rows_not_the_whole_dataset(self):
        """The manifest already carries the whole-dataset mix. This field has to be
        the subset the coefficients were tuned on, or it says nothing new."""
        df = _scorable_df(provenance="simulated")
        # 60 observed rows, of which 40 are unscorable.
        df.loc[df.index[:60], "outcome_provenance"] = "observed"
        df.loc[df.index[:40], "won"] = False

        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        mix = report["policy_search_outcome_provenance"]
        assert mix.get("observed") == 20
        assert mix.get("simulated") == N_ROWS - 60
        assert sum(mix.values()) == report["policy_search_rows"]

    def test_a_missing_provenance_column_reports_unknown_not_an_assumption(self):
        df = _scorable_df(provenance=None)
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_search_outcome_provenance"] == {"unknown": N_ROWS}


class TestTheCallSitePassesAlignedPredictions:
    """The tests above prove the function behaves correctly when handed aligned
    predictions. None of them would fail if `main()` went back to handing it the
    training split — which is precisely the defect.

    Running `main()` needs SageMaker paths and a real training cycle, so the call site
    is pinned structurally instead: parse train.py and inspect the argument actually
    passed. Structural, not behavioural, and it says so — but it is the check that
    would have caught this, and it costs nothing.
    """

    @staticmethod
    def _policy_search_call() -> "ast.Call":
        import ast

        source = (TRAINING_CONTAINER / "train.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "search_policy_parameters"
        ]
        assert len(calls) == 1, (
            f"expected exactly one call to search_policy_parameters in train.py, "
            f"found {len(calls)} — this guard inspects a single known call site"
        )
        return calls[0]

    def test_the_predictions_argument_is_not_the_training_split(self):
        import ast

        call = self._policy_search_call()
        # Second positional arg is `predictions`.
        assert len(call.args) >= 2, "search_policy_parameters called without predictions"
        rendered = ast.unparse(call.args[1])

        assert "train_loader.dataset" not in rendered, (
            "the policy search is being handed the TRAINING SPLIT again. "
            "search_policy_parameters requires len(predictions) == len(df) and will "
            f"refuse, so the search silently never runs. Got: {rendered}"
        )
        assert "val_loader" not in rendered, (
            f"the policy search is being handed the validation split. Got: {rendered}"
        )

    def test_the_predictions_argument_covers_the_whole_feature_tensor(self):
        import ast

        call = self._policy_search_call()
        rendered = ast.unparse(call.args[1])

        # `features` is built from `df`, so scoring TensorDataset(features) is what makes
        # the lengths agree structurally rather than at one particular row count.
        assert "TensorDataset(features)" in rendered, (
            "expected the policy search to score TensorDataset(features), which is "
            f"aligned with df by construction. Got: {rendered}"
        )
        assert "[" not in rendered, (
            f"the feature tensor appears to be sliced before scoring. Got: {rendered}"
        )

    def test_the_in_sample_mask_is_supplied(self):
        import ast

        call = self._policy_search_call()
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}

        assert "in_sample" in kwargs, (
            "the call site no longer supplies in_sample, so the report cannot say how "
            "much of the search's evidence the model had already trained on"
        )
        assert "train_size" in kwargs["in_sample"], (
            "the in_sample mask should be derived from train_size, the value that "
            f"actually built train_ds. Got: {kwargs['in_sample']}"
        )


class TestExistingRefusalsStillHold:
    def test_a_dataset_missing_required_columns_is_refused(self):
        df = _scorable_df().drop(columns=["price_paid"])
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_searched"] is False
        assert "price_paid" in report["policy_skipped_reason"]

    def test_no_won_rows_is_refused(self):
        df = _scorable_df()
        df["won"] = False
        preds = np.full(len(df), 0.4)
        report = train.search_policy_parameters(df, preds, {})

        assert report["policy_searched"] is False
        assert "no won rows" in report["policy_skipped_reason"]
