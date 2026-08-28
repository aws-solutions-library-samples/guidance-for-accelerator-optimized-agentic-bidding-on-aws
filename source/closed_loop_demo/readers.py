"""Read helpers for real Part 2 state (for visualization).

Everything here reads **real** persisted state:

- Current bidding parameters and their history come from the real DynamoDB
  parameter store and its audit table.
- Model versions and their approval status come from the real SageMaker Model
  Registry — this is the durable record of governance promotion/rejection
  decisions.

Live canary traffic state is held in-memory inside the running Governance Agent
runtime (it is not persisted), so it is not readable cross-process from the
orchestrator. The model-registry approval status is the durable, verifiable
signal of what governance decided, and is what the deployment view surfaces.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def read_parameters(parameter_store: Any, model_type: str) -> list[dict]:
    """Read current parameter state for a model type from the real store."""
    params = await parameter_store.read_all_parameters(model_type)
    out = []
    for name, p in params.items():
        out.append(
            {
                "parameter_name": name,
                "current_value": p.current_value,
                "previous_value": p.previous_value,
                "version": p.version,
                "updated_by": p.updated_by,
                "reason": p.reason,
                "confidence": p.confidence,
                "updated_at": p.updated_at,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "max_delta_per_update": p.max_delta_per_update,
            }
        )
    out.sort(key=lambda d: d["parameter_name"])
    return out


def read_audit_trail(
    dynamodb_resource: Any,
    audit_table_name: str,
    model_type: str,
    limit: int = 50,
) -> list[dict]:
    """Read the most recent audit records for a model type.

    The audit table is keyed by ``model_type`` (partition) and
    ``timestamp_version`` (sort, ``"<timestamp>#<version>"``). Records are
    returned newest-first.
    """
    from boto3.dynamodb.conditions import Key

    table = dynamodb_resource.Table(audit_table_name)
    resp = table.query(
        KeyConditionExpression=Key("model_type").eq(model_type),
        ScanIndexForward=False,  # newest first
        Limit=limit,
    )

    records = []
    for item in resp.get("Items", []):
        ts_version = str(item.get("timestamp_version", ""))
        ts_str = ts_version.split("#", 1)[0] if "#" in ts_version else ts_version
        try:
            ts = float(ts_str)
        except (ValueError, TypeError):
            ts = None
        records.append(
            {
                "model_type": item.get("model_type"),
                "parameter_name": item.get("parameter_name"),
                "old_value": _to_float(item.get("old_value")),
                "new_value": _to_float(item.get("new_value")),
                "updated_by": item.get("updated_by"),
                "reason": item.get("reason"),
                "confidence": _to_float(item.get("confidence")),
                "version": _to_int(item.get("version")),
                "timestamp": ts,
            }
        )
    return records


def read_model_versions(
    sagemaker_client: Any,
    model_package_group_name: str,
    limit: int = 20,
) -> list[dict]:
    """List model package versions and their approval status from the registry.

    ``ModelApprovalStatus`` reflects the governance decision:
    ``Approved`` (promoted), ``Rejected``, or ``PendingManualApproval``.
    """
    resp = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=limit,
    )

    versions = []
    for pkg in resp.get("ModelPackageSummaryList", []):
        created = pkg.get("CreationTime")
        arn = pkg.get("ModelPackageArn")
        versions.append(
            {
                "model_package_arn": arn,
                "version": pkg.get("ModelPackageVersion"),
                "approval_status": pkg.get("ModelApprovalStatus"),
                "status": pkg.get("ModelPackageStatus"),
                "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
                "description": pkg.get("ModelPackageDescription"),
                "approval_description": _approval_description(sagemaker_client, arn),
            }
        )
    return versions


def _approval_description(sagemaker_client, model_package_arn: Optional[str]) -> Optional[str]:
    """The governance agent's stated reason for a version's approval status.

    Requires a per-version DescribeModelPackage call: ModelPackageSummaryList
    carries ModelApprovalStatus but NOT ApprovalDescription, so the reason is
    simply absent from the list response.

    The agent writes this field whenever it sets a status (see
    agents/governance/governance_agent.py's _update_registry, which passes
    ApprovalDescription=reason), so a rejection normally has a reason. It is not
    guaranteed though: a version registered but never evaluated has none, and a
    status set by hand outside the agent may have none. Returns None in that case
    so callers can render "no reason recorded" rather than invent one.

    Best-effort by design — a failed describe must not take down the whole
    registry listing, which is the caller's primary content.
    """
    if not model_package_arn:
        return None
    try:
        detail = sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )
    except Exception:  # noqa: BLE001 — listing must survive a per-version failure
        return None
    return detail.get("ApprovalDescription") or None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
