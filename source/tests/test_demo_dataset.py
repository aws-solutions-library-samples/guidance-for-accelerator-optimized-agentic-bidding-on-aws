"""Unit tests for orchestrator.demo_dataset (FR1, NFR3).

Covers sizing, bounds, seeded determinism, the pre-write check (and its parity
with the trainer's validate_dataset), the row shapes against the Glue jobs, and
the replace-on-generate S3 write against a stubbed client.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import demo_dataset as dd  # noqa: E402

_SOURCE = Path(__file__).resolve().parent.parent
_REPO = _SOURCE.parent


def _defaults(model_type: str) -> dict:
    return dd.validate_params(model_type, None)


# ---------------------------------------------------------------------------
# Sizing and bounds (FR1.2, FR1.3, FR1.4)
# ---------------------------------------------------------------------------


class TestSizing:
    def test_default_dlrm_sizing_targets_50_conversions(self):
        sizing = dd.size_dataset("dlrm_bid_shader", _defaults("dlrm_bid_shader"))
        assert sizing.rows == 55_556
        assert round(sizing.expected_positives) == 50
        assert sizing.positive_kind == "conversions"
        assert sizing.capped is False

    def test_low_rates_are_capped_at_100000_rows(self):
        params = dd.validate_params(
            "dlrm_bid_shader", {"win_rate": 0.1, "click_rate": 0.005, "conversion_rate": 0.01}
        )
        sizing = dd.size_dataset("dlrm_bid_shader", params)
        assert sizing.rows == 100_000
        assert sizing.capped is True
        assert sizing.expected_positives == pytest.approx(0.5)

    @pytest.mark.parametrize("model_type", ["deal_yield_manager_floor", "deal_yield_manager_margin"])
    def test_sixty_percent_clear_share_gives_834_deals(self, model_type):
        sizing = dd.size_dataset(model_type, _defaults(model_type))
        assert sizing.rows == 834
        assert round(sizing.expected_positives) == 500
        assert sizing.positive_kind == "labelled_deals"

    def test_out_of_bound_rate_names_the_field(self):
        with pytest.raises(dd.DemoParamsError) as exc_info:
            dd.validate_params("dlrm_bid_shader", {"click_rate": 0.9})
        assert exc_info.value.field == "click_rate"
        assert "0.005" in str(exc_info.value) and "0.5" in str(exc_info.value)

    def test_unknown_parameter_and_non_number_are_refused(self):
        with pytest.raises(dd.DemoParamsError) as unknown:
            dd.validate_params("dlrm_bid_shader", {"ctr": 0.1})
        assert unknown.value.field == "ctr"
        with pytest.raises(dd.DemoParamsError) as not_number:
            dd.validate_params("deal_yield_manager_floor", {"clear_share": "high"})
        assert not_number.value.field == "clear_share"

    def test_untrainable_model_type_has_no_demo_dataset(self):
        with pytest.raises(dd.UnknownDemoModelTypeError):
            dd.validate_params("ncf_deal_manager", None)


# ---------------------------------------------------------------------------
# Determinism and the pre-write check (FR1.8, FR1.9)
# ---------------------------------------------------------------------------


class TestGeneration:
    def test_same_seed_gives_byte_identical_parquet(self):
        params = _defaults("dlrm_bid_shader")
        first = dd.to_parquet_bytes("dlrm_bid_shader", dd.generate_dlrm_rows(params, 3000, seed=7))
        second = dd.to_parquet_bytes("dlrm_bid_shader", dd.generate_dlrm_rows(params, 3000, seed=7))
        other = dd.to_parquet_bytes("dlrm_bid_shader", dd.generate_dlrm_rows(params, 3000, seed=8))
        assert first == second
        assert first != other

    def test_default_dlrm_dataset_passes_check_within_time_budget(self):
        started = time.monotonic()
        dataset = dd.generate_demo_dataset("dlrm_bid_shader", None)
        elapsed = time.monotonic() - started
        assert elapsed < 60  # NFR3
        meta = dataset.metadata
        assert meta["rows_written"] == 55_556
        assert meta["actual_positives"] > 0
        assert meta["demonstration_only"] is True
        assert meta["source"] == "demo_dataset"
        assert meta["data_key"] == "demo-training-data/part-00000.parquet"
        assert dataset.parquet[:4] == b"PAR1"

    def test_rows_with_no_conversions_are_refused(self):
        columns = dd.generate_dlrm_rows(_defaults("dlrm_bid_shader"), 200, seed=1)
        assert sum(columns["label_conversion"]) == 0
        with pytest.raises(dd.DemoDatasetUnfitError, match="no positives"):
            dd.check_dlrm_rows(columns)

    def test_constant_categorical_is_refused(self):
        columns = dd.generate_dlrm_rows(_defaults("dlrm_bid_shader"), 60_000, seed=3)
        columns["device_type"] = ["mobile"] * len(columns["device_type"])
        with pytest.raises(dd.DemoDatasetUnfitError, match="device_type"):
            dd.check_dlrm_rows(columns)

    def test_check_matches_trainer_validate_dataset(self):
        pd = pytest.importorskip("pandas", reason="trainer parity needs pandas")
        pytest.importorskip("torch", reason="trainer parity needs torch")
        container = str(_SOURCE / "training" / "container")
        if container not in sys.path:
            sys.path.append(container)
        train = pytest.importorskip("train", reason="trainer module did not import")

        good = dd.generate_dlrm_rows(_defaults("dlrm_bid_shader"), 60_000, seed=dd.DEFAULT_SEED)
        ours = dd.check_dlrm_rows(good)
        theirs = train.validate_dataset(pd.DataFrame(good), dd.DLRM_LABEL_COLUMN, dd.DLRM_OBJECTIVE)
        assert ours["labelled_rows"] == theirs["labelled_rows"]
        assert ours["positive_rate"] == theirs["positive_rate"]
        assert ours["categorical_cardinality"] == theirs["categorical_cardinality"]
        assert theirs["outcome_provenance"] == {"simulated": 60_000}

        empty = dd.generate_dlrm_rows(_defaults("dlrm_bid_shader"), 200, seed=1)
        with pytest.raises(dd.DemoDatasetUnfitError):
            dd.check_dlrm_rows(empty)
        with pytest.raises(train.DatasetUnfitError):
            train.validate_dataset(pd.DataFrame(empty), dd.DLRM_LABEL_COLUMN, dd.DLRM_OBJECTIVE)

    @pytest.mark.parametrize(
        "model_type,low,high",
        [("deal_yield_manager_floor", 0.5, 1.5), ("deal_yield_manager_margin", -0.5, 0.5)],
    )
    def test_deal_yield_rows_shape_and_label_bounds(self, model_type, low, high):
        dataset = dd.generate_demo_dataset(model_type, None)
        import pyarrow.parquet as pq

        table = pq.read_table(io.BytesIO(dataset.parquet))
        assert table.column_names == ["label", *dd.DEAL_YIELD_FEATURE_COLUMNS]
        labels = table.column("label").to_pylist()
        assert labels and all(low <= v <= high for v in labels)
        assert dataset.metadata["sizing"]["rows"] == 834
        assert dataset.metadata["rows_written"] == len(labels)

    def test_deal_yield_label_outside_bounds_is_refused(self):
        columns = dd.generate_deal_yield_rows(
            "deal_yield_manager_margin", _defaults("deal_yield_manager_margin"), 50, seed=2
        )
        columns["label"][0] = 0.9
        with pytest.raises(dd.DemoDatasetUnfitError, match="outside"):
            dd.check_deal_yield_rows("deal_yield_manager_margin", columns)


# ---------------------------------------------------------------------------
# Row shape against the Glue jobs (FR1.7)
# ---------------------------------------------------------------------------


_GLUE_TYPES = {"string": "string", "double": "double", "boolean": "bool", "int": "int32"}


def _raw_bid_outcome_columns() -> list[tuple[str, str]]:
    text = (_REPO / "deployment" / "feedback_pipeline_cfn.yaml").read_text(encoding="utf-8")
    block = text.split("Name: raw_bid_outcomes", 1)[1].split("Location:", 1)[0]
    pairs = re.findall(r"- Name: (\w+)\s+Type: (\w+)", block)
    return [(name, _GLUE_TYPES[kind]) for name, kind in pairs]


def _glue_derived_columns() -> set[str]:
    text = (_SOURCE / "etl" / "glue_feature_engineering.py").read_text(encoding="utf-8")
    with_column = set(re.findall(r'withColumn\(\s*"([a-z]\w*)"', text))
    response = set(re.findall(r'_add_response_label\(\s*df,\s*"(\w+)"', text))
    return with_column | response


class TestRowShape:
    def test_dlrm_columns_equal_glue_output(self):
        raw = _raw_bid_outcome_columns()
        expected = {name for name, _ in raw} | _glue_derived_columns()
        ours = {name for name, _ in dd.DLRM_COLUMNS}
        assert ours == expected
        types = dict(dd.DLRM_COLUMNS)
        for name, kind in raw:
            assert types[name] == kind, name

    def test_dlrm_rows_carry_demo_source_and_simulated_provenance(self):
        columns = dd.generate_dlrm_rows(_defaults("dlrm_bid_shader"), 50, seed=4)
        assert set(columns["source"]) == {"demo_dataset"}
        assert set(columns["outcome_provenance"]) == {"simulated"}

    def test_deal_yield_feature_columns_match_glue(self):
        text = (_SOURCE / "etl" / "glue_deal_yield_feature_engineering.py").read_text(encoding="utf-8")
        block = text.split("FEATURE_COLUMNS = [", 1)[1].split("]", 1)[0]
        assert tuple(re.findall(r'"(\w+)"', block)) == dd.DEAL_YIELD_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# S3 (FR1.10, FR1.11)
# ---------------------------------------------------------------------------


def _small_dataset() -> dd.GeneratedDemoDataset:
    return dd.generate_demo_dataset("deal_yield_manager_floor", None)


class TestStorage:
    def test_write_deletes_old_objects_before_writing(self):
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "demo-training-data-deal-yield-floor/part-00000.parquet"},
                {"Key": "demo-training-data-deal-yield-floor/_demo_metadata.json"},
            ],
            "IsTruncated": False,
        }
        client.delete_objects.return_value = {}
        dd.write_demo_dataset("bucket", _small_dataset(), client=client)

        names = [c[0] for c in client.method_calls]
        assert names == ["list_objects_v2", "delete_objects", "put_object", "put_object"]
        assert client.list_objects_v2.call_args.kwargs["Prefix"] == "demo-training-data-deal-yield-floor/"
        keys = [c.kwargs["Key"] for c in client.put_object.call_args_list]
        assert keys == [
            "demo-training-data-deal-yield-floor/part-00000.parquet",
            "demo-training-data-deal-yield-floor/_demo_metadata.json",
        ]
        meta = json.loads(client.put_object.call_args_list[1].kwargs["Body"])
        assert meta["demonstration_only"] is True

    def test_delete_errors_stop_the_write(self):
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [{"Key": "demo-training-data-deal-yield-floor/part-00000.parquet"}],
            "IsTruncated": False,
        }
        client.delete_objects.return_value = {
            "Errors": [{"Key": "demo-training-data-deal-yield-floor/part-00000.parquet",
                        "Code": "AccessDenied", "Message": "Access Denied"}]
        }
        with pytest.raises(dd.DemoDatasetStorageError, match="AccessDenied"):
            dd.write_demo_dataset("bucket", _small_dataset(), client=client)
        client.put_object.assert_not_called()

    def test_read_metadata_returns_none_when_absent(self):
        client = MagicMock()
        client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
        )
        assert dd.read_demo_metadata("bucket", "dlrm_bid_shader", client=client) is None

    def test_read_metadata_raises_on_other_errors(self):
        client = MagicMock()
        client.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
        )
        with pytest.raises(dd.DemoDatasetStorageError):
            dd.read_demo_metadata("bucket", "dlrm_bid_shader", client=client)
