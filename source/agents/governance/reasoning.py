"""Reasoning layer for the Model Promotion Governance Agent (Strands + Bedrock).

The promote/reject/inconclusive decision is produced by the deterministic statistical
gate (Welch's t-test + SPRT + guardrails) in ``governance_agent.py`` and is
authoritative. This module adds the *reasoning* on top of that real result:

  1. ``generate_decision_rationale`` — turn a real GovernanceDecision + its real metrics
     into a natural-language rationale for the audit trail and the HTTP response. It
     never invents metrics; if Bedrock is unavailable it returns ("", "unavailable").
  2. ``answer_governance_query`` — answer an operator's question strictly from the real
     audit records passed in (fetched from DynamoDB). If there are no records, the
     caller returns an honest "no record" without invoking the model.
  3. ``CollectingAuditStore`` — captures the audit record the deterministic pipeline
     produces so the handler can enrich it (rationale, rationale_source,
     signals_considered) and write a single real record.
  4. ``read_audit_records`` — read real audit records from DynamoDB.

Requirements: 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.4.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("agentcore.governance.reasoning")


# ---------------------------------------------------------------------------
# Audit capture + read (real data only)
# ---------------------------------------------------------------------------


class CollectingAuditStore:
    """Audit store that captures records in memory instead of writing them.

    Passed to the deterministic ``ModelPromotionGovernanceAgent`` so the handler can
    enrich the single record it produces (with the reasoning rationale) and then write
    one real record to DynamoDB. Matches the ``put_record`` async interface the pipeline
    expects.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def put_record(self, record: dict[str, Any]) -> None:
        self.records.append(dict(record))


def default_gate_reason(decision: str, metrics: dict[str, Any]) -> str:
    """Build a short factual reason string from real metrics when the caller
    (e.g. the closed-loop demo UI, which computes the decision itself via the
    real ``ABEvaluator``) does not supply one. Derived only from the provided
    values — never invents a number that isn't in ``metrics``.
    """
    lift = metrics.get("relative_lift")
    p_value = metrics.get("p_value")
    lift_str = f"{lift:.4f}" if isinstance(lift, (int, float)) else "n/a"
    p_str = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "n/a"
    if decision == "promote":
        return f"Treatment outperforms control (lift={lift_str}, p={p_str})"
    if decision == "reject":
        violations = metrics.get("guardrail_violations")
        if violations:
            return f"Guardrail violation: {', '.join(violations)}"
        return f"Treatment underperforms control (lift={lift_str}, p={p_str})"
    return f"Not yet statistically significant (lift={lift_str}, p={p_str})"


def signals_considered(metrics: dict[str, Any]) -> list[str]:
    """Return the names of the real signals that were present in the decision metrics.

    This is derived deterministically from the actual metrics dict (not invented by the
    model), so ``signals_considered`` in the audit reflects what genuinely informed the
    decision/recommendation.
    """
    if not metrics:
        return []
    signals: list[str] = []
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        signals.append(key)
    return sorted(signals)


def read_audit_records(
    table: Any,
    *,
    model_type: str | None = None,
    version_arn: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Read real governance audit records from the DynamoDB audit table.

    Queries by ``model_type`` (partition key) when provided, most-recent first; otherwise
    scans a bounded number of items. Filters by ``version_arn`` client-side when given.
    Returns [] when nothing matches (honest empty — the caller reports "no record").
    """
    from boto3.dynamodb.conditions import Key

    items: list[dict[str, Any]] = []
    try:
        if model_type:
            resp = table.query(
                KeyConditionExpression=Key("model_type").eq(model_type),
                ScanIndexForward=False,  # most recent first (SK is timestamp-prefixed)
                Limit=max(1, min(limit, 100)),
            )
            items = resp.get("Items", [])
        else:
            resp = table.scan(Limit=max(1, min(limit, 100)))
            items = resp.get("Items", [])
            items.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
    except Exception as exc:  # noqa: BLE001 - surface as empty; handler decides messaging
        logger.error("Failed to read audit records: %s", exc)
        return []

    if version_arn:
        items = [r for r in items if r.get("version_arn") == version_arn]

    return items[:limit]


# ---------------------------------------------------------------------------
# Strands + Bedrock reasoning
# ---------------------------------------------------------------------------


_RATIONALE_SYSTEM_PROMPT = """\
You are the reasoning layer of a Model Promotion Governance Agent for a real-time
bidding platform. A deterministic statistical gate (Welch's t-test + SPRT + guardrail
checks) has ALREADY decided whether to promote, reject, or mark a challenger model
version inconclusive. That decision is final and authoritative.

Your job is to explain the decision in clear, honest language for the audit trail:
- Summarize what the real metrics show and why they support the decision that was made.
- Use ONLY the numbers provided. Never invent p-values, lifts, sample counts, or
  statuses. If a value is missing, say it is not available.
- When the decision is "inconclusive", you MAY note whether the available secondary
  signals lean toward extending the test or rejecting, but be explicit that the gate's
  decision stands.
Keep it to a short paragraph.
"""

_QUERY_SYSTEM_PROMPT = """\
You are the Model Promotion Governance Agent's review assistant. Answer the operator's
question using ONLY the governance audit records provided to you as JSON. These records
are the real, authoritative history of promotion/rejection decisions.

Rules:
- Base every statement on the provided records. Do not invent decisions, metrics, or
  versions that are not present.
- If the records do not contain the version or information asked about, say clearly that
  there is no matching governance record.
- Stay within governance scope (model promotion decisions, their metrics and reasons).
  Do not discuss credentials, infrastructure internals, or anything outside the records.
Be concise.
"""


def _make_agent(model_id: str, region: str, system_prompt: str):
    """Build a tool-less Strands agent backed by a Bedrock model."""
    from strands import Agent
    from strands.models import BedrockModel

    model = BedrockModel(model_id=model_id, region_name=region)
    return Agent(model=model, system_prompt=system_prompt, callback_handler=None)


def generate_decision_rationale(
    *,
    model_id: str,
    region: str,
    decision: str,
    reason: str,
    model_type: str,
    version_arn: str,
    metrics: dict[str, Any],
) -> tuple[str, str]:
    """Generate a natural-language rationale from the real decision + metrics.

    Returns (rationale, source) where source is "bedrock" on success or "unavailable" if
    the model could not be reached. On unavailability the deterministic decision still
    stands (the caller does not block on this).
    """
    payload = {
        "decision": decision,
        "gate_reason": reason,
        "model_type": model_type,
        "version_arn": version_arn,
        "metrics": metrics,
    }
    prompt = (
        "The deterministic gate produced this governance result. Write the audit "
        "rationale explaining it, using only these values:\n\n"
        + json.dumps(payload, default=str, indent=2)
    )
    try:
        agent = _make_agent(model_id, region, _RATIONALE_SYSTEM_PROMPT)
        result = agent(prompt)
        text = str(result).strip()
        if not text:
            return ("", "unavailable")
        return (text, "bedrock")
    except Exception as exc:  # noqa: BLE001 - reasoning is best-effort, decision stands
        logger.warning("Rationale generation unavailable (%s); decision stands.", exc)
        return ("", "unavailable")


def answer_governance_query(
    *,
    model_id: str,
    region: str,
    query: str,
    records: list[dict[str, Any]],
) -> str:
    """Answer an operator question strictly from the provided real audit records.

    The caller must pass the real records (already fetched from DynamoDB) and must handle
    the empty-records case before calling this (so we never fabricate a record). Raises
    on model failure so the handler can return an honest error.
    """
    prompt = (
        f"Operator question: {query}\n\n"
        "Governance audit records (JSON, most recent first):\n"
        + json.dumps(records, default=str, indent=2)
    )
    agent = _make_agent(model_id, region, _QUERY_SYSTEM_PROMPT)
    return str(agent(prompt)).strip()


def make_timestamp_version(record: dict[str, Any]) -> str:
    """Build the audit table sort key (timestamp_version) from a record.

    The audit table's schema is PK=model_type, SK=timestamp_version. The deterministic
    pipeline's record carries an epoch ``timestamp`` and a ``version_arn`` but not the
    composite SK; construct a time-sortable, unique SK here.
    """
    ts = record.get("timestamp")
    try:
        iso = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        iso = datetime.now(tz=timezone.utc).isoformat()
    version_arn = str(record.get("version_arn", ""))
    version_suffix = version_arn.rsplit("/", 1)[-1] if version_arn else "unknown"
    return f"{iso}#{version_suffix}"
