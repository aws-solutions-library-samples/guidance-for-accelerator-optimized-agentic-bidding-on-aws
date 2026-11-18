"""Tests for the outcome simulator and the tri-state outcome signals it feeds.

Covers four things that were built together:

1. The simulator itself — deterministic, off by default, funnel-consistent, and
   labelling everything it produces as simulated.
2. Tri-state outcome signals — None means "not reported yet", and must never be read
   as a confirmed negative.
3. Provenance propagation — a simulated signal stays labelled through the associator,
   the enriched event and the trainer's dataset report.
4. The ETL's conversion-lag window, which decides whether a late signal can be joined
   to its bid at all.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1]
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from orchestrator.outcome_simulator import (  # noqa: E402
    ENABLE_ENV_VAR,
    OutcomeSimulator,
    OutcomeSimulatorConfig,
    is_enabled,
    signals_for,
    simulate_outcome,
)
from shared.feedback_models import (  # noqa: E402
    BidShadingOutcomeEvent,
    SignalEvent,
    validate_bid_outcome,
)
from shared.signal_associator import (  # noqa: E402
    DownstreamSignal,
    SignalAssociator,
    SignalType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCollector:
    """Stands in for FeedbackCollector, keeping what was emitted."""

    def __init__(self) -> None:
        self.emitted: list = []

    async def emit(self, event) -> None:
        self.emitted.append(event)


def _bid_event(request_id: str | None = None) -> BidShadingOutcomeEvent:
    """A bid-time event: priced, outcome not yet reported."""
    return BidShadingOutcomeEvent(
        request_id=request_id or str(uuid.uuid4()),
        timestamp=1_700_000_000.0,
        model_version="test-v1",
        source="live",
        original_price=5.0,
        shaded_price=3.0,
        bid_floor=1.0,
        won=None,
        price_paid=None,
        impression=None,
        click=None,
        conversion=None,
        outcome_provenance="unresolved",
        user_id_hash="a1b2c3d4e5f60718",
        site_domain="example.com",
        device_type="2",
        hour_of_day=12,
        shade_factor_used=0.6,
        conversion_value_estimate_used=5.0,
    )


# ---------------------------------------------------------------------------
# 1. The simulator
# ---------------------------------------------------------------------------


class TestSimulatorIsOffByDefault:
    def test_disabled_when_env_var_absent(self, monkeypatch):
        monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
        assert is_enabled() is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
    def test_disabled_for_non_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(ENABLE_ENV_VAR, value)
        assert is_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
    def test_enabled_only_for_explicit_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(ENABLE_ENV_VAR, value)
        assert is_enabled() is True


class TestSimulatorIsDeterministic:
    def test_same_request_id_gives_same_outcome(self):
        rid = str(uuid.uuid4())
        assert simulate_outcome(rid) == simulate_outcome(rid)

    def test_repeated_calls_never_drift(self):
        rid = str(uuid.uuid4())
        first = simulate_outcome(rid)
        assert all(simulate_outcome(rid) == first for _ in range(50))

    def test_different_request_ids_are_not_all_identical(self):
        outcomes = {simulate_outcome(str(uuid.uuid4())).won for _ in range(200)}
        assert outcomes == {True, False}

    def test_does_not_use_the_random_module(self):
        """A seeded `random` would still be reproducible only within one process."""
        source = (
            _SOURCE / "orchestrator" / "outcome_simulator.py"
        ).read_text()
        assert "import random" not in source
        assert "random." not in source


class TestSimulatorFunnelConsistency:
    """A click cannot happen without an impression, nor an impression without a win."""

    def test_monotonic_chain_holds_for_every_request(self):
        for _ in range(500):
            o = simulate_outcome(str(uuid.uuid4()))
            if o.conversion:
                assert o.click
            if o.click:
                assert o.impression
            if o.impression:
                assert o.won

    def test_conversion_value_present_only_on_conversion(self):
        for _ in range(500):
            o = simulate_outcome(str(uuid.uuid4()))
            if o.conversion:
                assert o.conversion_value is not None
            else:
                assert o.conversion_value is None

    def test_zero_win_rate_produces_no_outcomes(self):
        cfg = OutcomeSimulatorConfig(win_rate=0.0)
        for _ in range(100):
            o = simulate_outcome(str(uuid.uuid4()), cfg)
            assert (o.won, o.impression, o.click, o.conversion) == (
                False,
                False,
                False,
                False,
            )

    def test_certain_funnel_converts_every_bid(self):
        cfg = OutcomeSimulatorConfig(
            win_rate=1.0, impression_rate=1.0, click_rate=1.0, conversion_rate=1.0
        )
        for _ in range(100):
            o = simulate_outcome(str(uuid.uuid4()), cfg)
            assert (o.won, o.impression, o.click, o.conversion) == (
                True,
                True,
                True,
                True,
            )

    def test_measured_rates_track_the_configured_ones(self):
        """Not an exact equality — a hash-derived draw is a sample, not a quota."""
        cfg = OutcomeSimulatorConfig(win_rate=0.4, impression_rate=0.95)
        ids = [str(uuid.uuid4()) for _ in range(4000)]
        outcomes = [simulate_outcome(i, cfg) for i in ids]
        win_rate = sum(o.won for o in outcomes) / len(outcomes)
        assert 0.36 < win_rate < 0.44


class TestSimulatorConfigValidation:
    @pytest.mark.parametrize(
        "field", ["win_rate", "impression_rate", "click_rate", "conversion_rate"]
    )
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rates_must_be_probabilities(self, field, bad):
        with pytest.raises(ValueError, match=field):
            OutcomeSimulatorConfig(**{field: bad})

    def test_conversion_value_cannot_be_negative(self):
        with pytest.raises(ValueError, match="conversion_value"):
            OutcomeSimulatorConfig(conversion_value=-1.0)

    def test_from_env_reads_overrides(self, monkeypatch):
        monkeypatch.setenv("OUTCOME_SIMULATOR_WIN_RATE", "0.9")
        monkeypatch.setenv("OUTCOME_SIMULATOR_CONVERSION_VALUE", "42.5")
        cfg = OutcomeSimulatorConfig.from_env()
        assert cfg.win_rate == 0.9
        assert cfg.conversion_value == 42.5

    def test_from_env_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("OUTCOME_SIMULATOR_WIN_RATE", "not-a-number")
        assert OutcomeSimulatorConfig.from_env().win_rate == OutcomeSimulatorConfig().win_rate


class TestSimulatorSignals:
    def test_lost_bid_produces_no_signals(self):
        cfg = OutcomeSimulatorConfig(win_rate=0.0)
        assert signals_for(str(uuid.uuid4()), 1.0, cfg) == []

    def test_signals_are_in_funnel_order(self):
        cfg = OutcomeSimulatorConfig(
            win_rate=1.0, impression_rate=1.0, click_rate=1.0, conversion_rate=1.0
        )
        types = [s.signal_type for s in signals_for(str(uuid.uuid4()), 1.0, cfg)]
        assert types == [
            SignalType.IMPRESSION,
            SignalType.CLICK,
            SignalType.CONVERSION,
        ]

    def test_every_signal_is_labelled_simulated(self):
        cfg = OutcomeSimulatorConfig(
            win_rate=1.0, impression_rate=1.0, click_rate=1.0, conversion_rate=1.0
        )
        for signal in signals_for(str(uuid.uuid4()), 1.0, cfg):
            assert signal.provenance == "simulated"

    def test_impression_only_when_no_click(self):
        cfg = OutcomeSimulatorConfig(
            win_rate=1.0, impression_rate=1.0, click_rate=0.0
        )
        types = [s.signal_type for s in signals_for(str(uuid.uuid4()), 1.0, cfg)]
        assert types == [SignalType.IMPRESSION]

    def test_conversion_signal_carries_the_value(self):
        cfg = OutcomeSimulatorConfig(
            win_rate=1.0,
            impression_rate=1.0,
            click_rate=1.0,
            conversion_rate=1.0,
            conversion_value=17.5,
        )
        conv = [
            s
            for s in signals_for(str(uuid.uuid4()), 1.0, cfg)
            if s.signal_type == SignalType.CONVERSION
        ]
        assert conv[0].conversion_value == 17.5


class TestSimulatorApply:
    @pytest.mark.asyncio
    async def test_applies_signals_through_the_associator(self):
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        event = _bid_event()
        associator.register_bid(event)

        cfg = OutcomeSimulatorConfig(
            win_rate=1.0, impression_rate=1.0, click_rate=1.0, conversion_rate=1.0
        )
        applied = await OutcomeSimulator(associator, cfg).apply(event)

        assert applied == 3
        final = collector.emitted[-1]
        assert (final.won, final.impression, final.click, final.conversion) == (
            True,
            True,
            True,
            True,
        )

    @pytest.mark.asyncio
    async def test_simulated_loss_applies_nothing(self):
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        event = _bid_event()
        associator.register_bid(event)

        cfg = OutcomeSimulatorConfig(win_rate=0.0)
        assert await OutcomeSimulator(associator, cfg).apply(event) == 0
        assert collector.emitted == []

    @pytest.mark.asyncio
    async def test_never_raises_when_the_associator_fails(self):
        class _Broken:
            async def handle_signal(self, signal):
                raise RuntimeError("kinesis is down")

        cfg = OutcomeSimulatorConfig(win_rate=1.0, impression_rate=1.0)
        assert await OutcomeSimulator(_Broken(), cfg).apply(_bid_event()) == 0

    @pytest.mark.asyncio
    async def test_unregistered_bid_applies_nothing(self):
        """A dropped signal is reported as zero applied, not as a success."""
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        cfg = OutcomeSimulatorConfig(win_rate=1.0, impression_rate=1.0)
        assert await OutcomeSimulator(associator, cfg).apply(_bid_event()) == 0


class TestSimulatorWiring:
    """The orchestrator's bid path must not construct a simulator unless asked."""

    @staticmethod
    def _module():
        import orchestrator.feedback_integration as fi

        return fi

    def test_not_constructed_when_disabled(self, monkeypatch):
        fi = self._module()
        monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
        monkeypatch.setattr(
            fi, "_signal_associator", SignalAssociator(_RecordingCollector())
        )
        fi.reset_outcome_simulator()
        assert fi._get_outcome_simulator() is None

    def test_constructed_when_enabled(self, monkeypatch):
        fi = self._module()
        monkeypatch.setenv(ENABLE_ENV_VAR, "true")
        monkeypatch.setattr(
            fi, "_signal_associator", SignalAssociator(_RecordingCollector())
        )
        fi.reset_outcome_simulator()
        try:
            assert isinstance(fi._get_outcome_simulator(), OutcomeSimulator)
        finally:
            fi.reset_outcome_simulator()

    def test_not_constructed_without_an_associator(self, monkeypatch):
        """Nothing to write signals into means no simulator, not a crash."""
        fi = self._module()
        monkeypatch.setenv(ENABLE_ENV_VAR, "true")
        monkeypatch.setattr(fi, "_signal_associator", None)
        fi.reset_outcome_simulator()
        try:
            assert fi._get_outcome_simulator() is None
        finally:
            fi.reset_outcome_simulator()

    def test_register_bid_context_is_called_on_the_bid_path(self, monkeypatch):
        """The call whose absence made every arriving signal miss its bid."""
        fi = self._module()
        associator = SignalAssociator(_RecordingCollector())
        monkeypatch.setattr(fi, "_signal_associator", associator)
        event = _bid_event()
        fi.register_bid_context(event)
        assert associator.cache_size == 1

    def test_register_bid_context_is_a_noop_without_an_associator(self, monkeypatch):
        fi = self._module()
        monkeypatch.setattr(fi, "_signal_associator", None)
        fi.register_bid_context(_bid_event())  # must not raise

    def test_signal_route_references_only_defined_names(self):
        """Regression: the route named a module global that did not exist.

        `app.py`'s `receive_signal` passed `_feedback_collector`, which was never
        defined or imported in that module, so every request to POST /v1/signals
        raised NameError and returned 500 and no SignalEvent ever reached Kinesis.
        Nothing caught it: the signal_receiver tests pass a collector in explicitly,
        so none of them exercised the route's own wiring. Checked statically because
        importing app.py constructs live AWS clients.
        """
        import ast

        tree = ast.parse((_SOURCE / "orchestrator" / "app.py").read_text())

        module_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    module_names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.Assign):
                module_names.update(
                    t.id for t in node.targets if isinstance(t, ast.Name)
                )
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                module_names.add(node.target.id)
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                module_names.add(node.name)

        route = next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "receive_signal"
        )
        local_names = {a.arg for a in route.args.args}
        for node in ast.walk(route):
            if isinstance(node, ast.Assign):
                local_names.update(
                    t.id for t in node.targets if isinstance(t, ast.Name)
                )

        undefined = sorted(
            node.id
            for node in ast.walk(route)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id.startswith("_")
            and node.id not in module_names
            and node.id not in local_names
        )
        assert undefined == [], f"receive_signal references undefined names: {undefined}"

    def test_signal_path_reuses_the_bid_path_collector(self):
        """One stream, or the ETL can never join a signal to its bid."""
        source = (_SOURCE / "orchestrator" / "app.py").read_text()
        assert "collector = _get_feedback_collector()" in source

    def test_load_test_outcomes_are_labelled_simulated(self):
        """A load test generates its outcomes; it does not observe them."""
        source = (
            _SOURCE / "orchestrator" / "feedback_integration.py"
        ).read_text()
        assert 'outcome_provenance="simulated"' in source


# ---------------------------------------------------------------------------
# 2. Tri-state outcome signals
# ---------------------------------------------------------------------------


def _outcome_errors(
    *,
    won,
    price_paid=None,
    impression,
    click,
    conversion,
    conversion_value=None,
) -> list[str]:
    """validate_bid_outcome with the price fields held valid and constant."""
    return validate_bid_outcome(
        request_id=str(uuid.uuid4()),
        original_price=5.0,
        shaded_price=3.0,
        bid_floor=1.0,
        won=won,
        price_paid=price_paid,
        impression=impression,
        click=click,
        conversion=conversion,
        conversion_value=conversion_value,
    )


class TestTriStateOutcomes:
    def test_bid_time_event_leaves_outcomes_unreported(self):
        event = _bid_event()
        assert event.won is None
        assert event.impression is None
        assert event.click is None
        assert event.conversion is None
        assert event.outcome_provenance == "unresolved"

    def test_unreported_outcomes_are_valid(self):
        """None is a legal state: the auction has not resolved yet."""
        assert (
            _outcome_errors(won=None, impression=None, click=None, conversion=None)
            == []
        )

    def test_a_lost_bid_is_a_real_negative(self):
        assert (
            _outcome_errors(
                won=False, impression=False, click=False, conversion=False
            )
            == []
        )

    def test_impression_without_a_win_is_rejected(self):
        assert _outcome_errors(
            won=False, impression=True, click=None, conversion=None
        ) != []

    def test_click_without_an_impression_is_rejected(self):
        assert _outcome_errors(
            won=True, price_paid=1.0, impression=False, click=True, conversion=None
        ) != []

    def test_conversion_without_a_click_is_rejected(self):
        assert _outcome_errors(
            won=True, price_paid=1.0, impression=True, click=False, conversion=True
        ) != []

    def test_unreported_click_does_not_block_a_reported_impression(self):
        """An impression says nothing about a click; None must not read as False."""
        assert (
            _outcome_errors(
                won=True, price_paid=2.0, impression=True, click=None, conversion=None
            )
            == []
        )

    def test_unreported_win_does_not_forbid_a_price_paid(self):
        """Rule 4 keys on a confirmed loss, not on an unreported one."""
        assert (
            _outcome_errors(
                won=None, price_paid=None, impression=None, click=None, conversion=None
            )
            == []
        )

    def test_confirmed_loss_still_forbids_a_price_paid(self):
        assert _outcome_errors(
            won=False, price_paid=2.0, impression=False, click=False, conversion=False
        ) != []


# ---------------------------------------------------------------------------
# 3. Provenance propagation
# ---------------------------------------------------------------------------


class TestProvenancePropagation:
    @pytest.mark.asyncio
    async def test_simulated_signal_labels_the_enriched_event(self):
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        event = _bid_event()
        associator.register_bid(event)

        await associator.handle_signal(
            DownstreamSignal(
                request_id=event.request_id,
                signal_type=SignalType.IMPRESSION,
                timestamp=event.timestamp,
                provenance="simulated",
            )
        )
        assert collector.emitted[-1].outcome_provenance == "simulated"

    @pytest.mark.asyncio
    async def test_observed_signal_labels_the_enriched_event(self):
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        event = _bid_event()
        associator.register_bid(event)

        await associator.handle_signal(
            DownstreamSignal(
                request_id=event.request_id,
                signal_type=SignalType.IMPRESSION,
                timestamp=event.timestamp,
            )
        )
        assert collector.emitted[-1].outcome_provenance == "observed"

    @pytest.mark.asyncio
    async def test_impression_does_not_invent_a_click(self):
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        event = _bid_event()
        associator.register_bid(event)

        await associator.handle_signal(
            DownstreamSignal(
                request_id=event.request_id,
                signal_type=SignalType.IMPRESSION,
                timestamp=event.timestamp,
            )
        )
        enriched = collector.emitted[-1]
        assert enriched.impression is True
        assert enriched.won is True
        assert enriched.click is None
        assert enriched.conversion is None

    @pytest.mark.asyncio
    async def test_enrichment_preserves_every_non_outcome_field(self):
        """Regression: enrichment enumerated fields and silently dropped three.

        `_enrich_event` listed the fields to copy, and the list predated
        `day_of_week`, `geo_country` and `has_video`. Every enriched event reverted
        them to their defaults — confirmed in S3, where 9 enriched rows carried
        `day_of_week=0` on a Sunday and an empty `geo_country` while their originals
        carried 6 and a real country. Two of the DLRM's three categoricals were being
        destroyed on precisely the rows that have labels.
        """
        collector = _RecordingCollector()
        associator = SignalAssociator(feedback_collector=collector)
        event = BidShadingOutcomeEvent(
            request_id=str(uuid.uuid4()),
            timestamp=1_700_000_000.0,
            model_version="ctx-v9",
            model_type="dlrm_bid_shader",
            source="live",
            original_price=9.0,
            shaded_price=6.0,
            bid_floor=2.0,
            won=None,
            price_paid=None,
            impression=None,
            click=None,
            conversion=None,
            outcome_provenance="unresolved",
            user_id_hash="0123456789abcdef",
            site_domain="nytimes.com",
            device_type="4",
            hour_of_day=17,
            day_of_week=6,
            geo_country="CAN",
            has_video=True,
            shade_factor_used=0.55,
            conversion_value_estimate_used=31.0,
        )
        associator.register_bid(event)

        await associator.handle_signal(
            DownstreamSignal(
                request_id=event.request_id,
                signal_type=SignalType.IMPRESSION,
                timestamp=event.timestamp,
                provenance="simulated",
            )
        )
        enriched = collector.emitted[-1]

        outcome_fields = {
            "won",
            "impression",
            "click",
            "conversion",
            "conversion_value",
            "outcome_provenance",
        }
        for name in type(event).model_fields:
            if name in outcome_fields:
                continue
            assert getattr(enriched, name) == getattr(event, name), (
                f"enrichment changed {name}: "
                f"{getattr(event, name)!r} -> {getattr(enriched, name)!r}"
            )

    def test_enrichment_does_not_enumerate_fields(self):
        """The enumeration is the defect; a field list will drift again."""
        source = (_SOURCE / "shared" / "signal_associator.py").read_text()
        assert "original.model_dump()" in source
        assert "site_domain=original.site_domain" not in source

    def test_signal_event_carries_provenance(self):
        signal = SignalEvent(
            request_id=str(uuid.uuid4()),
            signal_type="impression",
            timestamp=1_700_000_000.0,
            provenance="simulated",
        )
        assert signal.provenance == "simulated"

    def test_signal_event_defaults_to_observed(self):
        signal = SignalEvent(
            request_id=str(uuid.uuid4()),
            signal_type="click",
            timestamp=1_700_000_000.0,
        )
        assert signal.provenance == "observed"


class TestProvenanceMixReporting:
    """The trainer must report the mix so the manifest can carry it."""

    def test_counts_each_provenance(self):
        pd = pytest.importorskip("pandas")
        from training.container.train import outcome_provenance_mix

        df = pd.DataFrame(
            {"outcome_provenance": ["simulated"] * 3 + ["observed"] * 2 + ["unresolved"]}
        )
        assert outcome_provenance_mix(df) == {
            "simulated": 3,
            "observed": 2,
            "unresolved": 1,
        }

    def test_missing_column_reports_unknown_not_observed(self):
        pd = pytest.importorskip("pandas")
        from training.container.train import outcome_provenance_mix

        df = pd.DataFrame({"label_conversion": [0, 1, 1]})
        assert outcome_provenance_mix(df) == {"unknown": 3}

    def test_nulls_count_as_unresolved(self):
        pd = pytest.importorskip("pandas")
        from training.container.train import outcome_provenance_mix

        df = pd.DataFrame({"outcome_provenance": ["observed", None, None]})
        assert outcome_provenance_mix(df) == {"observed": 1, "unresolved": 2}


# ---------------------------------------------------------------------------
# 4. The ETL conversion-lag window
# ---------------------------------------------------------------------------


def _load_resolve_window():
    """Import _resolve_window, stubbing pyspark when it is not installed.

    The window logic is pure datetime arithmetic and needs no Spark, but its module
    imports pyspark at the top. Skipping these tests on a machine without pyspark
    would leave the lag calculation unverified everywhere it actually runs, so stub
    instead of skip -- and tear the stub down so other modules' real-pyspark
    detection is not fooled by it. Mirrors test_glue_etl_window.py.
    """
    try:
        import pyspark  # noqa: F401

        from etl.glue_feature_engineering import _resolve_window

        return _resolve_window
    except ImportError:
        pass

    import types as _types

    fake_sql = _types.ModuleType("pyspark.sql")
    fake_sql.DataFrame = object
    fake_sql.Window = object
    fake_sql.functions = _types.ModuleType("pyspark.sql.functions")
    fake_types = _types.ModuleType("pyspark.sql.types")
    fake_types.IntegerType = object
    fake_types.DoubleType = object
    fake_types.StringType = object
    sys.modules["pyspark"] = _types.ModuleType("pyspark")
    sys.modules["pyspark.sql"] = fake_sql
    sys.modules["pyspark.sql.types"] = fake_types
    try:
        from etl.glue_feature_engineering import _resolve_window

        return _resolve_window
    finally:
        for name in ("pyspark", "pyspark.sql", "pyspark.sql.types"):
            sys.modules.pop(name, None)
        sys.modules.pop("etl.glue_feature_engineering", None)


class TestConversionLagWindow:
    @staticmethod
    def _resolve():
        return _load_resolve_window()

    def test_default_lag_is_zero_and_preserves_the_old_window(self):
        resolve = self._resolve()
        start, end = resolve(None, None, "6")
        assert abs((end - start) - timedelta(hours=6)) < timedelta(seconds=5)

    def test_lag_shifts_the_window_into_the_past(self):
        resolve = self._resolve()
        _, end_no_lag = resolve(None, None, "6", "0")
        _, end_lagged = resolve(None, None, "6", "24")
        gap = end_no_lag - end_lagged
        assert abs(gap - timedelta(hours=24)) < timedelta(seconds=5)

    def test_lag_does_not_change_the_window_duration(self):
        resolve = self._resolve()
        start, end = resolve(None, None, "6", "12")
        assert abs((end - start) - timedelta(hours=6)) < timedelta(seconds=5)

    def test_negative_lag_is_rejected(self):
        resolve = self._resolve()
        with pytest.raises(ValueError, match="conversion_lag_hours"):
            resolve(None, None, "6", "-1")

    def test_lag_with_an_explicit_window_is_rejected(self):
        """A backfill names its own window; shifting it silently would lie."""
        resolve = self._resolve()
        with pytest.raises(ValueError, match="cannot be combined"):
            resolve(
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T06:00:00+00:00",
                None,
                "24",
            )

    def test_explicit_window_still_honoured_without_lag(self):
        resolve = self._resolve()
        start, end = resolve(
            "2026-01-01T00:00:00+00:00", "2026-01-01T06:00:00+00:00", None
        )
        assert (end - start) == timedelta(hours=6)

    def test_zero_lag_warns_about_dropped_signals(self, caplog):
        resolve = self._resolve()
        with caplog.at_level("WARNING"):
            resolve(None, None, "6", "0")
        assert any(
            "conversion_lag_hours=0" in r.message for r in caplog.records
        )
