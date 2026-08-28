"""Unit + property tests for orchestrator.loadtest_instrumentation.

Validates:
- generate_outcome_sample() is deterministic (same run_id/index -> same
  sample) and produces varied samples across different indices.
- aggregate_run_model_version() (pure function) always returns either ""
  (empty input) or one of the observed input values — property-tested per
  NFR-5's Partial PBT scope.
- emit_load_test_outcome() constructs a BidShadingOutcomeEvent with source="load_test"
  and never raises (matches emit_bid_outcome()'s error-swallowing contract).

Maps to: FR-1, FR-2 (Story 1, load-test-outcome-capture unit).
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.loadtest_instrumentation import (
    aggregate_run_model_version,
    emit_load_test_outcome,
    generate_outcome_sample,
)
from shared.feedback_models import BidShadingOutcomeEvent


class TestGenerateOutcomeSample:
    def test_deterministic_same_inputs(self):
        """Same (seed, request_index, shaded_price) always produces the same sample."""
        s1 = generate_outcome_sample(seed=42, request_index=5, shaded_price=3.5)
        s2 = generate_outcome_sample(seed=42, request_index=5, shaded_price=3.5)
        assert s1 == s2

    def test_different_indices_can_differ(self):
        """Across many indices at one seed, samples are not all identical
        (real variation, not a constant stuck value)."""
        samples = [
            generate_outcome_sample(seed=42, request_index=i, shaded_price=3.5)
            for i in range(50)
        ]
        won_values = {s.won for s in samples}
        assert len(won_values) > 1  # both True and False appear

    def test_outcome_responds_to_the_model_price(self):
        """The whole point of the market model: a higher bid wins more often.

        Before this, won/price_paid were drawn from a fixed pool indexed by
        hash((run_id, request_index)) and the container's price never entered, so
        a stable-vs-canary A/B could not reflect the model at all.
        """
        def win_rate(price: float) -> float:
            wins = sum(
                generate_outcome_sample(
                    seed=42, request_index=i, shaded_price=price
                ).won
                for i in range(500)
            )
            return wins / 500

        assert win_rate(2.0) < win_rate(3.5) < win_rate(5.0)

    def test_same_seed_gives_both_variants_identical_market(self):
        """Two runs replayed at the same seed must face the same market, so any
        difference in outcomes is attributable to the model's pricing."""
        low = [
            generate_outcome_sample(seed=7, request_index=i, shaded_price=2.5)
            for i in range(200)
        ]
        high = [
            generate_outcome_sample(seed=7, request_index=i, shaded_price=4.5)
            for i in range(200)
        ]
        # Identical market draw => every request the low bid won, the high bid
        # also won (the clearing price it had to beat was the same).
        for lo, hi in zip(low, high):
            if lo.won:
                assert hi.won

    def test_run_id_cannot_influence_the_outcome(self):
        """Regression guard: generate_outcome_sample must not accept or depend on
        a run id. It previously keyed on hash((run_id, request_index)), which made
        two runs differ even at an identical seed -- and made the difference an
        artifact of the uuid4 run id rather than the model."""
        import inspect

        params = inspect.signature(generate_outcome_sample).parameters
        assert "run_id" not in params
        assert set(params) == {"seed", "request_index", "shaded_price"}

    def test_impression_value_is_fixed_by_the_seed_not_the_model(self):
        """The advertiser's valuation must be a property of the request, so a
        model cannot inflate its own score by bidding higher."""
        from orchestrator.loadtest_instrumentation import impression_value

        assert impression_value(42, 3) == impression_value(42, 3)
        assert impression_value(42, 3) != impression_value(99, 3)

    def test_monotonic_signals_respected(self):
        """Generated samples respect the same monotonic rule BidShadingOutcomeEvent
        enforces: conversion -> click -> impression -> won."""
        for i in range(50):
            s = generate_outcome_sample(seed=42, request_index=i, shaded_price=3.5)
            if s.conversion:
                assert s.click
            if s.click:
                assert s.impression
            if s.impression:
                assert s.won
            if not s.won:
                assert s.price_paid is None
            if not s.conversion:
                assert s.conversion_value is None


class TestAggregateRunModelVersion:
    def test_empty_input_returns_empty_string(self):
        assert aggregate_run_model_version([]) == ""

    def test_single_value(self):
        assert aggregate_run_model_version(["v1"]) == "v1"

    def test_majority_wins(self):
        result = aggregate_run_model_version(["v1", "v2", "v1", "v1"])
        assert result == "v1"

    def test_tie_broken_by_first_occurrence(self):
        result = aggregate_run_model_version(["v2", "v1", "v2", "v1"])
        assert result == "v2"

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=50))
    def test_result_is_always_one_of_the_inputs(self, versions):
        """Property: for non-empty input, the result is always a value that
        actually appeared in the input — never fabricated."""
        result = aggregate_run_model_version(versions)
        assert result in versions

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=50))
    def test_result_is_empty_or_in_input(self, versions):
        """Property: for any input (including empty), the result is either
        "" or one of the observed values."""
        result = aggregate_run_model_version(versions)
        assert result == "" or result in versions


class TestEmitLoadTestOutcome:
    def test_emits_event_with_source_load_test(self):
        """The emitted BidShadingOutcomeEvent has source='load_test'."""
        from orchestrator import feedback_integration

        mock_collector = MagicMock()
        mock_collector.emit = AsyncMock()

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = mock_collector

            async def _run():
                emit_load_test_outcome(
                    "run-1", 0, "dlrm_bid_shader_stable:v1", "dlrm_bid_shader",
                    seed=42, shaded_price=3.5,
                )
                await asyncio.sleep(0.01)

            asyncio.run(_run())
            mock_collector.emit.assert_called_once()
            event = mock_collector.emit.call_args[0][0]
            assert isinstance(event, BidShadingOutcomeEvent)
            assert event.source == "load_test"
            assert event.model_version == "dlrm_bid_shader_stable:v1"
            assert event.model_type == "dlrm_bid_shader"
        finally:
            feedback_integration._feedback_collector = original

    def test_noop_when_collector_disabled(self):
        """When no collector is configured, this never raises."""
        from orchestrator import feedback_integration

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = None
            emit_load_test_outcome(
                "run-1", 0, "v1", "dlrm_bid_shader", seed=42, shaded_price=3.5
            )  # must not raise
        finally:
            feedback_integration._feedback_collector = original

    def test_never_raises_on_collector_error(self):
        """Errors from the collector are swallowed, matching emit_bid_outcome()."""
        from orchestrator import feedback_integration

        mock_collector = MagicMock()
        mock_collector.emit = AsyncMock(side_effect=RuntimeError("Kinesis down"))

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = mock_collector

            async def _run():
                emit_load_test_outcome(
                "run-1", 0, "v1", "dlrm_bid_shader", seed=42, shaded_price=3.5
            )  # must not raise
                await asyncio.sleep(0.01)

            asyncio.run(_run())
        finally:
            feedback_integration._feedback_collector = original


class TestSurplusMetric:
    """The value emit_load_test_outcome() returns is what feeds the governance
    A/B comparison, so it has to reward the behavior bid shading exists for:
    win often, and win cheaply.

    This replaced `price_paid if won else 0.0`. Once outcomes became a function of
    the model's price, that older metric scored a LESS aggressive shader higher —
    it won more and paid more — which is backwards.
    """

    @staticmethod
    def _mean_surplus(price: float, n: int = 1500, seed: int = 42) -> float:
        from orchestrator import feedback_integration

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = None  # emission off; metric still returned
            values = [
                emit_load_test_outcome(
                    "run-surplus", i, "v1", "dlrm_bid_shader",
                    seed=seed, shaded_price=price,
                )
                for i in range(n)
            ]
        finally:
            feedback_integration._feedback_collector = original
        return sum(values) / len(values)

    def test_underbidding_is_penalised(self):
        """Bidding far below the market forfeits winnable impressions."""
        assert self._mean_surplus(1.5) < self._mean_surplus(4.0)

    def test_overbidding_is_penalised(self):
        """Bidding far above the impression's worth overpays for what it wins.

        This is the property the old price_paid metric got wrong: there, bidding
        higher always scored higher.
        """
        assert self._mean_surplus(8.0) < self._mean_surplus(4.0)

    def test_optimum_is_interior(self):
        """A real objective has a best bid somewhere in the middle, rather than
        being maximized at an extreme."""
        prices = [1.5, 2.5, 3.5, 4.0, 4.5, 6.0, 8.0]
        scores = {p: self._mean_surplus(p) for p in prices}
        best = max(scores, key=scores.get)
        assert best not in (prices[0], prices[-1]), scores

    def test_losing_scores_exactly_zero(self):
        """No impression served and nothing spent."""
        from orchestrator import feedback_integration
        from orchestrator.loadtest_instrumentation import market_clearing_price

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = None
            # Bid below the clearing price for this exact request => guaranteed loss.
            clearing = market_clearing_price(42, 11)
            value = emit_load_test_outcome(
                "run-lose", 11, "v1", "dlrm_bid_shader",
                seed=42, shaded_price=round(clearing - 0.5, 4),
            )
        finally:
            feedback_integration._feedback_collector = original
        assert value == 0.0

    def test_no_price_returns_none_not_zero(self):
        """A request the container never priced is not a zero-surplus sample —
        there is no model decision to score, so it must not enter the population.
        """
        from orchestrator import feedback_integration

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = None
            assert emit_load_test_outcome(
                "run-noprice", 0, "v1", "dlrm_bid_shader", seed=42, shaded_price=None
            ) is None
        finally:
            feedback_integration._feedback_collector = original
