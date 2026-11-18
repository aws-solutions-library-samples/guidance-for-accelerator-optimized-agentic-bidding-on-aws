"""Unit tests for orchestrator.promotion_service — PromotionService.

Validates:
- promote() raises PromotionNotRecommendedError for any recommendation
  other than "promote", BEFORE touching Triton/registry/audit (a reject/
  inconclusive result must never trigger the real side effects).
- A confirmed promote() call performs the same real sequence the automated
  pipeline uses: promote the canary engine to stable, wait for readiness,
  zero the router split + update version-ARN params, remove the canary,
  update the registry, write an audit record — in that order.
- If the new stable version never becomes ready, promotion aborts BEFORE
  the registry update or audit write (no partial-success side effects).
- The audit record's shape matches agents/governance/reasoning.py's own
  timestamp_version format.

Maps to: FR-9, FR-10 (Story 6, governance-comparison-promotion unit).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.promotion_service import (
    PromotionNotRecommendedError,
    PromotionResult,
    promote,
)


def _make_triton_loader(*, ready: bool = True):
    loader = MagicMock()
    loader.stable_name = MagicMock(side_effect=lambda m: f"{m}_stable")
    loader.canary_name = MagicMock(side_effect=lambda m: f"{m}_canary")
    loader.canary_engine_uri = MagicMock(
        side_effect=lambda m: f"s3://bucket/triton-models/{m}_canary/1/model.plan"
    )
    loader.promote_engine = AsyncMock(return_value=2)
    loader.wait_until_ready = AsyncMock(return_value=ready)
    loader.set_router_split = AsyncMock()
    loader.remove_canary = AsyncMock()
    return loader


def _make_sagemaker_client():
    client = MagicMock()
    client.update_model_package = MagicMock(return_value={})
    return client


def _make_audit_table():
    table = MagicMock()
    table.put_item = MagicMock(return_value={})
    return table


_VERSION_ARN = "arn:aws:sagemaker:us-east-1:123456789012:model-package/artf-dlrm-bid-shader/2"


class TestPromotionRecommendationGate:
    @pytest.mark.asyncio
    async def test_reject_recommendation_raises_before_any_side_effect(self):
        triton_loader = _make_triton_loader()
        sagemaker_client = _make_sagemaker_client()
        audit_table = _make_audit_table()

        with pytest.raises(PromotionNotRecommendedError) as exc_info:
            await promote(
                model_type="dlrm_bid_shader",
                version_arn=_VERSION_ARN,
                recommendation="reject",
                reason="challenger underperformed",
                triton_loader=triton_loader,
                sagemaker_client=sagemaker_client,
                audit_table=audit_table,
            )
        assert exc_info.value.recommendation == "reject"
        triton_loader.promote_engine.assert_not_called()
        sagemaker_client.update_model_package.assert_not_called()
        audit_table.put_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_extend_recommendation_raises_before_any_side_effect(self):
        triton_loader = _make_triton_loader()
        sagemaker_client = _make_sagemaker_client()
        audit_table = _make_audit_table()

        with pytest.raises(PromotionNotRecommendedError):
            await promote(
                model_type="dlrm_bid_shader",
                version_arn=_VERSION_ARN,
                recommendation="extend",
                reason="inconclusive",
                triton_loader=triton_loader,
                sagemaker_client=sagemaker_client,
                audit_table=audit_table,
            )
        triton_loader.promote_engine.assert_not_called()
        sagemaker_client.update_model_package.assert_not_called()
        audit_table.put_item.assert_not_called()


class TestPromotionSuccess:
    @pytest.mark.asyncio
    async def test_promote_performs_real_sequence(self):
        triton_loader = _make_triton_loader()
        sagemaker_client = _make_sagemaker_client()
        audit_table = _make_audit_table()

        result = await promote(
            model_type="dlrm_bid_shader",
            version_arn=_VERSION_ARN,
            recommendation="promote",
            reason="challenger outperformed at p=0.02",
            triton_loader=triton_loader,
            sagemaker_client=sagemaker_client,
            audit_table=audit_table,
        )

        assert isinstance(result, PromotionResult)
        assert result.model_type == "dlrm_bid_shader"
        assert result.version_arn == _VERSION_ARN

        # Triton: promote engine, wait ready, zero split + version-ARN params, remove canary
        triton_loader.promote_engine.assert_called_once_with(
            "dlrm_bid_shader", "s3://bucket/triton-models/dlrm_bid_shader_canary/1/model.plan"
        )
        triton_loader.wait_until_ready.assert_called_once_with("dlrm_bid_shader_stable")
        triton_loader.set_router_split.assert_called_once_with(
            "dlrm_bid_shader", 0.0, stable_version_arn=_VERSION_ARN, canary_version_arn=""
        )
        triton_loader.remove_canary.assert_called_once_with("dlrm_bid_shader")

        # Registry updated to Approved
        sagemaker_client.update_model_package.assert_called_once()
        call_kwargs = sagemaker_client.update_model_package.call_args[1]
        assert call_kwargs["ModelPackageArn"] == _VERSION_ARN
        assert call_kwargs["ModelApprovalStatus"] == "Approved"

        # Audit record written with the expected shape
        audit_table.put_item.assert_called_once()
        item = audit_table.put_item.call_args[1]["Item"]
        assert item["model_type"] == "dlrm_bid_shader"
        assert item["decision"] == "promote"
        assert item["version_arn"] == _VERSION_ARN
        assert "#" in item["timestamp_version"]
        assert item["timestamp_version"].endswith("#2")  # trailing segment of the version ARN
        assert item["audit_record_id"] == result.audit_record_id

    @pytest.mark.asyncio
    async def test_promotion_aborts_if_new_stable_never_ready(self):
        """If the new stable version doesn't become ready, the registry
        update and audit write must never happen — no partial success."""
        triton_loader = _make_triton_loader(ready=False)
        sagemaker_client = _make_sagemaker_client()
        audit_table = _make_audit_table()

        with pytest.raises(RuntimeError):
            await promote(
                model_type="dlrm_bid_shader",
                version_arn=_VERSION_ARN,
                recommendation="promote",
                reason="challenger outperformed",
                triton_loader=triton_loader,
                sagemaker_client=sagemaker_client,
                audit_table=audit_table,
            )

        triton_loader.set_router_split.assert_not_called()
        triton_loader.remove_canary.assert_not_called()
        sagemaker_client.update_model_package.assert_not_called()
        audit_table.put_item.assert_not_called()

    @pytest.mark.asyncio
    async def test_source_label_recorded_for_attribution(self):
        """FR-10: every promotion via this path is attributed distinctly
        from the automated pipeline's own promotions."""
        triton_loader = _make_triton_loader()
        sagemaker_client = _make_sagemaker_client()
        audit_table = _make_audit_table()

        await promote(
            model_type="dlrm_bid_shader",
            version_arn=_VERSION_ARN,
            recommendation="promote",
            reason="challenger outperformed",
            triton_loader=triton_loader,
            sagemaker_client=sagemaker_client,
            audit_table=audit_table,
        )

        item = audit_table.put_item.call_args[1]["Item"]
        assert item["source"] == "load_test_comparison"
