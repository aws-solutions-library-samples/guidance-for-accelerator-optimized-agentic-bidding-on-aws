"""Tests for shared.load_test_context -- the per-request signals that scope
load-test-only behavior (target-variant override, and the plain
is-load-test flag) away from real bid-serving traffic.

The is-load-test flag (get_is_load_test()/load_test_scope()) is what
the yield containers' bounded exploration gates on -- see
that module and CLOSED_LOOP.md's "Yield Optimizer" section for why this
distinction matters: without it, enabling exploration by default would
perturb real auction bids, not just load-test traffic used to bootstrap
training data.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.load_test_context import (
    HEADER_NAME,
    IS_LOAD_TEST_HEADER_NAME,
    get_is_load_test,
    get_target_variant,
    load_test_scope,
    target_variant_scope,
)


class TestIsLoadTestDefault:
    def test_defaults_to_false_outside_any_scope(self):
        """Real bid-serving traffic never enters load_test_scope() at all --
        this is what must default to False for that path."""
        assert get_is_load_test() is False


class TestLoadTestScope:
    def test_scope_true_sets_flag_for_duration(self):
        assert get_is_load_test() is False
        with load_test_scope(True):
            assert get_is_load_test() is True
        assert get_is_load_test() is False

    def test_scope_false_is_a_no_op(self):
        with load_test_scope(False):
            assert get_is_load_test() is False

    def test_resets_even_if_body_raises(self):
        """Must never leak True into a later, unrelated request handled by
        the same worker thread/task if the scoped call raised."""
        try:
            with load_test_scope(True):
                assert get_is_load_test() is True
                raise ValueError("boom")
        except ValueError:
            pass
        assert get_is_load_test() is False

    def test_nested_scopes_restore_outer_value(self):
        with load_test_scope(True):
            with load_test_scope(False):
                assert get_is_load_test() is False
            assert get_is_load_test() is True


class TestIsLoadTestIndependentOfTargetVariant:
    """get_target_variant() alone cannot serve as an is-load-test signal --
    it's also None when a load test targets 'current' (no override header
    at all). These two context vars must be independently settable."""

    def test_load_test_true_with_no_target_variant_override(self):
        with load_test_scope(True):
            assert get_is_load_test() is True
            assert get_target_variant() is None

    def test_target_variant_set_without_load_test_flag(self):
        """Exercises that the two scopes are genuinely independent context
        vars, not accidentally coupled."""
        with target_variant_scope("canary"):
            assert get_target_variant() == "canary"
            assert get_is_load_test() is False


class TestHeaderNames:
    def test_header_names_are_distinct(self):
        assert HEADER_NAME != IS_LOAD_TEST_HEADER_NAME
