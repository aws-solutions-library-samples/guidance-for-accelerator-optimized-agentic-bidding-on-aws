"""Unit + property tests for orchestrator.loadtest_instrumentation.

Validates:
- generate_outcome_sample() is deterministic (same run_id/index -> same
  sample) and produces varied samples across different indices.
- aggregate_run_model_version() (pure function) always returns either ""
  (empty input) or one of the observed input values — property-tested per
  NFR-5's Partial PBT scope.
- emit_load_test_outcome() constructs a BidOutcomeEvent with source="load_test"
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
from shared.feedback_models import BidOutcomeEvent


class TestGenerateOutcomeSample:
    def test_deterministic_same_inputs(self):
        """Same (run_id, request_index) always produces the same sample."""
        s1 = generate_outcome_sample("run-abc", 5)
        s2 = generate_outcome_sample("run-abc", 5)
        assert s1 == s2

    def test_different_indices_can_differ(self):
        """Across many indices within one run, samples are not all identical
        (real variation, not a constant stuck value)."""
        samples = [generate_outcome_sample("run-abc", i) for i in range(50)]
        won_values = {s.won for s in samples}
        assert len(won_values) > 1  # both True and False appear

    def test_different_run_ids_can_differ(self):
        """Different run_ids at the same index are not required to match."""
        s1 = generate_outcome_sample("run-a", 0)
        s2 = generate_outcome_sample("run-b", 0)
        # Not asserting inequality (could coincide) — just that both are
        # valid, well-formed samples.
        for s in (s1, s2):
            assert isinstance(s.won, bool)
            assert s.shaded_price >= 0.0

    def test_monotonic_signals_respected(self):
        """Generated samples respect the same monotonic rule BidOutcomeEvent
        enforces: conversion -> click -> impression -> won."""
        for i in range(50):
            s = generate_outcome_sample("run-monotonic", i)
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
        """The emitted BidOutcomeEvent has source='load_test'."""
        from orchestrator import feedback_integration

        mock_collector = MagicMock()
        mock_collector.emit = AsyncMock()

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = mock_collector

            async def _run():
                emit_load_test_outcome("run-1", 0, "dlrm_bid_shader_stable:v1")
                await asyncio.sleep(0.01)

            asyncio.run(_run())
            mock_collector.emit.assert_called_once()
            event = mock_collector.emit.call_args[0][0]
            assert isinstance(event, BidOutcomeEvent)
            assert event.source == "load_test"
            assert event.model_version == "dlrm_bid_shader_stable:v1"
        finally:
            feedback_integration._feedback_collector = original

    def test_noop_when_collector_disabled(self):
        """When no collector is configured, this never raises."""
        from orchestrator import feedback_integration

        original = feedback_integration._feedback_collector
        try:
            feedback_integration._feedback_collector = None
            emit_load_test_outcome("run-1", 0, "v1")  # must not raise
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
                emit_load_test_outcome("run-1", 0, "v1")  # must not raise
                await asyncio.sleep(0.01)

            asyncio.run(_run())
        finally:
            feedback_integration._feedback_collector = original
