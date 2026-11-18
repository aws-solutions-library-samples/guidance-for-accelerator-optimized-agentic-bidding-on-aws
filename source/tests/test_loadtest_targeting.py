"""Unit tests for orchestrator.loadtest_targeting — ChallengerTargetOverride.

Validates:
- canary_supported() correctly distinguishes Triton-backed model types from
  rule-based ones.
- build_override_headers() only ever produces the out-of-band header for
  "challenger", never for "current" — never something a live-traffic call
  path could accidentally construct.
- validate_challenger_target() raises the correct, distinct error for
  "not supported" vs. "not staged" (BR-5), and never raises when a canary
  really is staged.
- is_canary_staged() never fabricates "staged" on error (real "unknown").

Requirements: FR-3 (Story 2, load-test-outcome-capture unit).
"""

from __future__ import annotations

import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.loadtest_targeting import (
    CANARY_SUPPORTED_MODEL_TYPES,
    CanaryNotStagedError,
    CanaryNotSupportedError,
    build_override_headers,
    canary_supported,
    is_canary_staged,
    validate_challenger_target,
)
from shared.load_test_context import HEADER_NAME


class TestCanarySupported:
    def test_dlrm_bid_shader_supported(self):
        assert canary_supported("dlrm_bid_shader") is True

    def test_ncf_deal_manager_supported(self):
        assert canary_supported("ncf_deal_manager") is True

    def test_widedeep_segment_activator_not_supported(self):
        assert canary_supported("widedeep_segment_activator") is False

    def test_metrics_enricher_not_supported(self):
        assert canary_supported("metrics_enricher") is False

    def test_unknown_model_type_not_supported(self):
        assert canary_supported("some_future_model") is False

    @given(st.sampled_from(sorted(CANARY_SUPPORTED_MODEL_TYPES)))
    def test_all_declared_supported_types_return_true(self, model_type):
        """Property: every model type in the declared supported set returns True."""
        assert canary_supported(model_type) is True


class TestBuildOverrideHeaders:
    def test_challenger_produces_header(self):
        headers = build_override_headers("challenger")
        assert headers == {HEADER_NAME: "canary"}

    def test_current_produces_no_header(self):
        """Never construct the override header for 'current' — only load-test
        code targeting the challenger should ever emit this header."""
        headers = build_override_headers("current")
        assert headers == {}

    @given(st.sampled_from(["current", "challenger"]))
    def test_headers_only_ever_contain_the_expected_key(self, target_variant):
        """Property: the returned dict never has any key other than HEADER_NAME."""
        headers = build_override_headers(target_variant)
        assert set(headers.keys()) <= {HEADER_NAME}


class TestIsCanaryStaged:
    @pytest.mark.asyncio
    async def test_unsupported_model_type_returns_false_without_probing(self):
        """A rule-based model type is never "staged" — no HTTP probe needed."""
        result = await is_canary_staged("metrics_enricher")
        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error_returns_false_not_true(self, monkeypatch):
        """On any probe failure, report False (unknown) — never fabricate staged=True."""
        monkeypatch.setenv("TRITON_URL", "triton-that-does-not-exist.invalid:8000")
        result = await is_canary_staged("dlrm_bid_shader")
        assert result is False


class TestValidateChallengerTarget:
    @pytest.mark.asyncio
    async def test_unsupported_model_type_raises_not_supported(self):
        with pytest.raises(CanaryNotSupportedError) as exc_info:
            await validate_challenger_target("widedeep_segment_activator")
        assert exc_info.value.model_type == "widedeep_segment_activator"

    @pytest.mark.asyncio
    async def test_supported_but_unstaged_raises_not_staged(self, monkeypatch):
        monkeypatch.setenv("TRITON_URL", "triton-that-does-not-exist.invalid:8000")
        with pytest.raises(CanaryNotStagedError) as exc_info:
            await validate_challenger_target("dlrm_bid_shader")
        assert exc_info.value.model_type == "dlrm_bid_shader"

    @pytest.mark.asyncio
    async def test_supported_and_staged_does_not_raise(self, monkeypatch):
        """When the canary probe reports ready, validation passes silently."""
        import orchestrator.loadtest_targeting as targeting_module

        async def _fake_is_canary_staged(model_type: str) -> bool:
            return True

        monkeypatch.setattr(targeting_module, "is_canary_staged", _fake_is_canary_staged)
        # Should not raise
        await validate_challenger_target("dlrm_bid_shader")
