"""Independently-callable Promote action (PromotionService).

Performs the SAME real side effects the automated governance pipeline
performs on promotion — CanaryDeployer.promote(), a SageMaker Model
Registry ModelApprovalStatus update, and a real audit-trail DynamoDB
write — reusing those exact primitives rather than the agent's own
on_new_model_version() orchestration function (which is the terminal
branch of its own live-A/B-monitoring state machine, not independently
callable — see the resolved Application Design Follow-up Question B).

Maps to: FR-9, FR-10 (Story 6, governance-comparison-promotion unit).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


class PromotionNotRecommendedError(Exception):
    """Raised when promote() is called for a comparison that did not
    recommend promotion — a reject/inconclusive result must never trigger
    the real promotion side effects (BR from Story 6's acceptance criteria)."""

    def __init__(self, recommendation: str):
        super().__init__(
            f"Cannot promote: comparison recommendation was '{recommendation}', not 'promote'."
        )
        self.recommendation = recommendation


@dataclass(frozen=True)
class PromotionResult:
    """Result of a successful promotion."""

    model_type: str
    version_arn: str
    audit_record_id: str


def _make_timestamp_version(timestamp: float, version_arn: str) -> str:
    """Build the audit table's sort key, matching
    agents/governance/reasoning.py::make_timestamp_version()'s exact format
    (ISO timestamp + '#' + the version ARN's trailing segment) so records
    written by this independent path are indistinguishable in shape from
    the automated pipeline's own audit records."""
    iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    version_suffix = version_arn.rsplit("/", 1)[-1] if version_arn else "unknown"
    return f"{iso}#{version_suffix}"


async def promote(
    *,
    model_type: str,
    version_arn: str,
    recommendation: str,
    reason: str,
    triton_loader: Any,
    sagemaker_client: Any,
    audit_table: Any,
    source_label: str = "load_test_comparison",
) -> PromotionResult:
    """Promote a challenger, performing the real side effects.

    Args:
        model_type: Logical model name (e.g. "dlrm_bid_shader").
        version_arn: The challenger's SageMaker Model Package ARN being promoted.
        recommendation: The ComparisonService result's recommendation
            ("promote" | "reject" | "extend"). Raises
            PromotionNotRecommendedError unless this is exactly "promote".
        reason: Human-readable reason recorded in the audit trail.
        triton_loader: A real TritonModelLoader instance.
        sagemaker_client: A real boto3 SageMaker client.
        audit_table: A real boto3 DynamoDB Table resource (the audit-trail table).
        source_label: Recorded on the audit record so it's distinguishable
            from the automated pipeline's own promotions (FR-10) — plain
            factual labeling, not integrity-asserting language.

    Raises:
        PromotionNotRecommendedError: If recommendation != "promote".

    Note on architecture: this calls TritonModelLoader's primitives directly
    rather than going through CanaryDeployer.promote(), because
    CanaryDeployer tracks the active canary's engine URI in its own
    in-process DeploymentState — state that only exists in whichever
    process called deploy_canary() (the governance agent runtime), not in
    the orchestrator process this function runs in. The canary engine's S3
    location is otherwise fully deterministic (TritonModelLoader's own
    documented repo layout: "<repo_prefix>/<m>_canary/1/model.plan"), so
    this reconstructs it directly rather than depending on cross-process
    state that doesn't exist here.
    """
    if recommendation != "promote":
        raise PromotionNotRecommendedError(recommendation)

    # Reconstruct the canary engine's real S3 URI via TritonModelLoader's own
    # public, deterministic-layout helper — the same path stage_canary()
    # itself writes to, not a guess.
    stable_model = triton_loader.stable_name(model_type)
    engine_uri = triton_loader.canary_engine_uri(model_type)

    # Step 1a: publish the canary engine as a NEW stable version.
    new_version = await triton_loader.promote_engine(model_type, engine_uri)
    if not await triton_loader.wait_until_ready(stable_model):
        raise RuntimeError(
            f"New stable version {new_version} for '{model_type}' did not become "
            "ready — promotion aborted before touching the registry or router split."
        )

    # Step 1b: route 100% back to stable and update the router's version-ARN
    # parameters (same control-plane calls CanaryDeployer.promote() makes).
    await triton_loader.set_router_split(
        model_type, 0.0, stable_version_arn=version_arn, canary_version_arn=""
    )
    await triton_loader.remove_canary(model_type)

    # Step 2: update the SageMaker Model Registry's ModelApprovalStatus.
    sagemaker_client.update_model_package(
        ModelPackageArn=version_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=reason,
    )

    # Step 3: write a real audit-trail record, in the same table/schema shape
    # the automated pipeline's records use (PK=model_type, SK=timestamp_version).
    timestamp = time.time()
    audit_record_id = str(uuid.uuid4())
    item = {
        "model_type": model_type,
        "timestamp_version": _make_timestamp_version(timestamp, version_arn),
        "timestamp": timestamp,
        "actor": "governance_ui_load_test_comparison",
        "version_arn": version_arn,
        "decision": "promote",
        "reason": reason,
        "source": source_label,
        "audit_record_id": audit_record_id,
    }
    audit_table.put_item(Item=item)

    return PromotionResult(
        model_type=model_type,
        version_arn=version_arn,
        audit_record_id=audit_record_id,
    )
