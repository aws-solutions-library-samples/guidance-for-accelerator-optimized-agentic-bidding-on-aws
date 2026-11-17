"""Deployment wiring for the demo datasets and the Triton scaling cap (FR2.3, FR4, NFR2).

Reads the templates and deploy.sh as text, renders them the way deploy.sh does,
and checks the result. No AWS calls.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
DEPLOY_SH = REPO / "deployment" / "deploy.sh"
TRITON_HPA = REPO / "deployment" / "eks" / "triton-hpa.yaml"
ORCHESTRATOR_MANIFEST = REPO / "deployment" / "eks" / "orchestrator-deployment.yaml"
CLOSED_LOOP_CFN = REPO / "deployment" / "closed_loop_cfn.yaml"
GOVERNANCE_CFN = REPO / "deployment" / "governance_eventbridge_cfn.yaml"

DEMO_PREFIXES = (
    "demo-training-data/",
    "demo-training-data-deal-yield-floor/",
    "demo-training-data-deal-yield-margin/",
)
# The IAM quota for a customer managed policy document, whitespace excluded.
MANAGED_POLICY_MAX_CHARS = 6144


def _deploy() -> str:
    return DEPLOY_SH.read_text(encoding="utf-8")


def _hpa(text: str, name: str) -> dict:
    for doc in yaml.safe_load_all(text):
        if doc and doc.get("kind") == "HorizontalPodAutoscaler" and doc["metadata"]["name"] == name:
            return doc
    raise AssertionError(f"no HorizontalPodAutoscaler named {name}")


# ---------------------------------------------------------------------------
# FR4.1 / FR4.2
# ---------------------------------------------------------------------------


def test_triton_hpa_max_replicas_is_the_max_gpus_placeholder():
    text = TRITON_HPA.read_text(encoding="utf-8")
    assert re.search(r"^\s*maxReplicas: __MAX_GPUS__\s*$", text, re.M)
    rendered = _hpa(text.replace("__MAX_GPUS__", "3"), "triton-hpa")
    assert rendered["spec"]["maxReplicas"] == 3
    assert rendered["spec"]["minReplicas"] == 1


def test_deploy_substitutes_and_guards_max_gpus_in_the_step_8b_loop():
    deploy = _deploy()
    loop = deploy[deploy.index("_APPLIED_MANIFESTS=("):]
    loop = loop[: loop.index('log "  Applied ${#_APPLIED_MANIFESTS[@]} manifests"')]
    assert "triton-hpa.yaml" in loop.split("\n", 1)[0]
    assert '-e "s|__MAX_GPUS__|${MAX_GPUS}|g"' in loop
    assert "grep -q '__MAX_GPUS__' \"${PROCESSED}\"" in loop
    assert 'fail 3 "${manifest} still has an unsubstituted __MAX_GPUS__"' in loop
    # A re-run restores --maxGPUs from .deploy-state.json before Step 8b.
    assert deploy.index("_reuse MAX_GPUS") < deploy.index("_APPLIED_MANIFESTS=(")


def test_orchestrator_hpa_is_unchanged():
    hpa = _hpa(ORCHESTRATOR_MANIFEST.read_text(encoding="utf-8"), "orchestrator-hpa")
    spec = hpa["spec"]
    assert (spec["minReplicas"], spec["maxReplicas"]) == (2, 10)
    cpu = [m for m in spec["metrics"] if m["resource"]["name"] == "cpu"]
    assert cpu[0]["resource"]["target"]["averageUtilization"] == 70
    assert "behavior" not in spec


# ---------------------------------------------------------------------------
# NFR2: the orchestrator's closed-loop policy
# ---------------------------------------------------------------------------

_SUBSTITUTIONS = {
    "AWS_REGION": "ap-southeast-2",
    "ACCOUNT_ID": "123456789012",
    "MODEL_BUCKET": "my-long-stack-prefix-model-bucket-123456789012-ap-southeast-2",
    "TRAINING_DATA_BUCKET": "my-long-stack-prefix-training-data-123456789012-ap-southeast-2",
    "SAGEMAKER_TRAINING_ROLE_NAME": "my-long-stack-prefix-sagemaker-training-execution-role",
}


def _closed_loop_policy() -> dict:
    deploy = _deploy()
    start = deploy.index('CLOSED_LOOP_POLICY_DOC="') + len('CLOSED_LOOP_POLICY_DOC="')
    end = deploy.index(']}"\n', start) + 2
    raw = deploy[start:end].replace("\\\n", "").replace('\\"', '"')

    def substitute(match):
        name = match.group(1)
        assert name in _SUBSTITUTIONS, f"policy references ${{{name}}}; add it to the test"
        return _SUBSTITUTIONS[name]

    rendered = re.sub(r"\$\{(\w+)\}", substitute, raw)
    return rendered, json.loads(rendered)


def _statement(policy: dict, sid: str) -> dict:
    return next(s for s in policy["Statement"] if s.get("Sid") == sid)


def test_closed_loop_policy_grants_write_only_on_demo_prefixes():
    _, policy = _closed_loop_policy()
    bucket = _SUBSTITUTIONS["TRAINING_DATA_BUCKET"]
    write = _statement(policy, "DemoTrainingDataReadWrite")
    assert set(write["Action"]) == {"s3:GetObject", "s3:PutObject", "s3:DeleteObject"}
    assert write["Resource"] == [
        *(f"arn:aws:s3:::{bucket}/{p}*" for p in DEMO_PREFIXES),
        f"arn:aws:s3:::{bucket}/training-manifests/*",
    ]
    # No other statement grants anything on the training-data bucket objects.
    for statement in policy["Statement"]:
        if statement is write:
            continue
        resources = statement["Resource"] if isinstance(statement["Resource"], list) else [statement["Resource"]]
        assert not any(r.startswith(f"arn:aws:s3:::{bucket}/") for r in resources), statement.get("Sid")


def test_closed_loop_policy_scopes_listing_and_kms():
    _, policy = _closed_loop_policy()
    listing = _statement(policy, "TrainingDataListForDemoManifest")
    assert listing["Action"] == ["s3:ListBucket"]
    prefixes = listing["Condition"]["StringLike"]["s3:prefix"]
    assert set(prefixes) >= {f"{p}*" for p in DEMO_PREFIXES}
    assert {"training-data/*", "training-data-deal-yield-floor/*", "training-data-deal-yield-margin/*"} <= set(prefixes)
    kms = _statement(policy, "KmsForTrainingDataBucket")
    assert kms["Condition"]["StringEquals"]["kms:ViaService"] == ["s3.ap-southeast-2.amazonaws.com"]
    assert set(kms["Action"]) == {"kms:Decrypt", "kms:GenerateDataKey"}


def test_closed_loop_policy_stays_under_the_managed_policy_size_limit():
    rendered, _ = _closed_loop_policy()
    size = len(re.sub(r"\s", "", rendered))
    assert size < MANAGED_POLICY_MAX_CHARS, size


# ---------------------------------------------------------------------------
# NFR2: SageMaker training role, and FR2.3: registration Lambda
# ---------------------------------------------------------------------------


def test_training_role_can_read_demo_prefixes_and_manifests():
    text = CLOSED_LOOP_CFN.read_text(encoding="utf-8")
    block = text[text.index("PolicyName: TrainingDataReadAccess"):text.index("PolicyName: TrainingDataKMSAccess")]
    for prefix in (*DEMO_PREFIXES, "training-manifests/"):
        assert f"${{TrainingDataBucketName}}/{prefix}*" in block, prefix


def test_registration_lambda_copies_demo_provenance():
    text = GOVERNANCE_CFN.read_text(encoding="utf-8")
    for key in ("includes_demo_dataset", "demo_dataset_generated_at"):
        assert re.search(rf"'{key}':\s*_provenance\('{key}'\)", text), key
