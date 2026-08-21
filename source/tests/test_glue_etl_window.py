"""Tests for glue_feature_engineering._resolve_window() — no PySpark dependency.

Kept in a separate file from test_glue_etl.py because that file's
pytest.importorskip("pyspark") skips its ENTIRE module on collection when
PySpark isn't installed, which would also skip these PySpark-independent
tests. _resolve_window() itself has no PySpark dependency, but the
containing module (etl.glue_feature_engineering) imports pyspark.sql at
module level, so we stub those imports out here when the real PySpark isn't
available — this project's environment doesn't always have PySpark
installed (see test_glue_etl.py's own documented constraint).

Regression coverage for the scheduled trigger bug: glue_etl_cfn.yaml's
FeatureEngineeringSchedule previously passed --start_timestamp/--end_timestamp
with literal unresolvable placeholder text ("6h_ago_epoch_millis",
"now_epoch_millis") — CloudFormation cannot compute a relative "now" or "N
hours ago" timestamp at template-render time, so every scheduled run would
have failed with a GlueArgumentError (confirmed live). The fix moves window
computation into the script itself via --window_hours (a static duration
CFN CAN render), self-computing [now - window_hours, now) at execution time.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pyspark  # noqa: F401

    from etl.glue_feature_engineering import _resolve_window
except ImportError:
    import types as _types

    _stub_names = ("pyspark", "pyspark.sql", "pyspark.sql.types")
    _fake_sql = _types.ModuleType("pyspark.sql")
    _fake_sql.DataFrame = object
    _fake_sql.Window = object
    _fake_sql.functions = _types.ModuleType("pyspark.sql.functions")
    _fake_types = _types.ModuleType("pyspark.sql.types")
    _fake_types.IntegerType = object
    _fake_types.DoubleType = object
    sys.modules["pyspark"] = _types.ModuleType("pyspark")
    sys.modules["pyspark.sql"] = _fake_sql
    sys.modules["pyspark.sql.types"] = _fake_types

    from etl.glue_feature_engineering import _resolve_window

    # Remove the stubs so other test modules' own real-pyspark detection
    # (e.g. test_glue_etl.py's pytest.importorskip("pyspark")) isn't fooled
    # by this stub still being cached in sys.modules.
    for _name in _stub_names:
        del sys.modules[_name]
    del sys.modules["etl.glue_feature_engineering"]


class TestResolveWindow:
    def test_explicit_window_used_as_is(self):
        start, end = _resolve_window(
            "2026-08-20T14:00:00+00:00", "2026-08-20T15:00:00+00:00", None
        )
        assert start.isoformat() == "2026-08-20T14:00:00+00:00"
        assert end.isoformat() == "2026-08-20T15:00:00+00:00"

    def test_self_computed_rolling_window(self):
        before = datetime.now(timezone.utc)
        start, end = _resolve_window(None, None, "6")
        after = datetime.now(timezone.utc)
        assert before <= end <= after
        assert abs((end - start) - timedelta(hours=6)) < timedelta(seconds=5)

    def test_default_window_hours_is_six(self):
        start, end = _resolve_window(None, None, None)
        assert abs((end - start) - timedelta(hours=6)) < timedelta(seconds=5)

    def test_partial_window_raises(self):
        with pytest.raises(ValueError, match="must be provided together"):
            _resolve_window("2026-08-20T14:00:00+00:00", None, None)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="must be after"):
            _resolve_window("2026-08-20T15:00:00+00:00", "2026-08-20T14:00:00+00:00", None)
