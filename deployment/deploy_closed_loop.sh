#!/usr/bin/env bash
# =============================================================================
# deploy_closed_loop.sh — Deploy the Closed-Loop Learning System infrastructure
#
# Deploys the full closed-loop stack in dependency order:
#   1. Feedback Pipeline (Kinesis/Firehose/S3/KMS)
#   2. Glue ETL (Feature engineering job + training data bucket)
#   3. Closed-Loop Core (DynamoDB/DAX/SageMaker Model Registry/SNS)
#   4. AgentCore Security (IAM roles, EventBridge rule + scheduler)
#   5. AgentCore Runtimes (agentcore deploy for both agents)
#
# Prerequisites:
#   - AWS CLI v2 with valid credentials
#   - Docker (for AgentCore image build)
#   - Python 3.11+ with boto3
#   - Existing EKS cluster (deploy.sh must have run first)
#   - VPC and subnets for DAX cluster
#
# Usage:
#   ./deploy_closed_loop.sh
#   ./deploy_closed_loop.sh --prefix prod
#   ./deploy_closed_loop.sh --skip-agentcore        # skip AgentCore deploy step
#   ./deploy_closed_loop.sh --stack-only            # deploy CFN stacks only, no agents
#   AWS_REGION=us-west-2 ./deploy_closed_loop.sh
#
# Required environment or parameters:
#   VPC_ID         — VPC for the DAX cluster
#   SUBNET_IDS     — Comma-separated subnet IDs for DAX (at least 2 in different AZs)
#   EKS_NODE_ROLE  — ARN of the EKS node IAM role (for FeedbackCollector access)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_PREFIX="${STACK_PREFIX:-}"
SKIP_AGENTCORE=0
STACK_ONLY=0
START_AT=1

for arg in "$@"; do
  case "${arg}" in
    --skip-agentcore)  SKIP_AGENTCORE=1 ;;
    --stack-only)      STACK_ONLY=1; SKIP_AGENTCORE=1 ;;
    --start-at=*)      START_AT="${arg#--start-at=}" ;;
    --start-at)        ;;
    --prefix=*)        STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)          ;;
    -h|--help)         sed -n '2,30p' "$0"; exit 0 ;;
    *)
      if [[ "${_PREV_ARG:-}" == "--prefix" ]]; then
        STACK_PREFIX="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--start-at" ]]; then
        START_AT="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

log()  { printf '\033[0;32m[closed-loop]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[closed-loop][warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[closed-loop][fail]\033[0m %s\n' "$*" >&2; exit 1; }

# =========================================================================
# Preflight
# =========================================================================
log "Preflight checks"
for bin in aws jq; do
  command -v "${bin}" >/dev/null 2>&1 || fail "missing: ${bin}"
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "${ACCOUNT_ID}" ]] || fail "cannot resolve AWS account"

# Validate required env vars (only EKS_NODE_ROLE needed for step 1)
if [[ "${START_AT}" -le 1 ]]; then
  [[ -n "${EKS_NODE_ROLE:-}" ]] || fail "EKS_NODE_ROLE is required for step 1 (ARN of the EKS node IAM role)"
fi

# Stack naming
FEEDBACK_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}feedback-pipeline"
GLUE_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}glue-etl"
CLOSED_LOOP_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}closed-loop-core"
AGENTCORE_SECURITY_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}agentcore-security"

log "Account=${ACCOUNT_ID}  Region=${AWS_REGION}  Prefix=${STACK_PREFIX:-<none>}"
log "Stacks: ${FEEDBACK_STACK} → ${GLUE_STACK} → ${CLOSED_LOOP_STACK} → ${AGENTCORE_SECURITY_STACK}"

# =========================================================================
# Helper: deploy a CloudFormation stack (create or update, wait for completion)
# =========================================================================
deploy_cfn_stack() {
  local stack_name="$1"
  local template_file="$2"
  shift 2
  local params=("$@")

  log "  Deploying stack: ${stack_name}"

  local action="create-stack"
  local existing_status=""
  existing_status="$(aws cloudformation describe-stacks --stack-name "${stack_name}" --region "${AWS_REGION}" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"

  if [[ "${existing_status}" == "ROLLBACK_COMPLETE" || "${existing_status}" == "DELETE_FAILED" ]]; then
    log "  Stack ${stack_name} is in ${existing_status} - deleting before recreate..."
    aws cloudformation delete-stack --stack-name "${stack_name}" --region "${AWS_REGION}"
    aws cloudformation wait stack-delete-complete --stack-name "${stack_name}" --region "${AWS_REGION}"
    log "  Deleted. Recreating..."
  elif [[ -n "${existing_status}" && "${existing_status}" != "None" ]]; then
    action="update-stack"
  fi

  local cmd=(aws cloudformation "${action}"
    --stack-name "${stack_name}"
    --template-body "file://${template_file}"
    --capabilities CAPABILITY_NAMED_IAM
    --region "${AWS_REGION}")

  if [[ ${#params[@]} -gt 0 ]]; then
    cmd+=(--parameters "${params[@]}")
  fi

  if "${cmd[@]}" 2>&1 | tee /tmp/cfn-deploy-output.txt | grep -q "StackId"; then
    local wait_action="stack-create-complete"
    [[ "${action}" == "update-stack" ]] && wait_action="stack-update-complete"
    log "  Waiting for ${stack_name} (${wait_action})..."
    aws cloudformation wait "${wait_action}" \
      --stack-name "${stack_name}" \
      --region "${AWS_REGION}"
    log "  Stack ${stack_name}: COMPLETE"
  else
    local cfn_error
    cfn_error="$(cat /tmp/cfn-deploy-output.txt 2>/dev/null || echo '')"
    # Check if it's a "no updates" situation
    if echo "${cfn_error}" | grep -q "No updates are to be performed"; then
      log "  Stack ${stack_name}: no updates needed"
      return 0
    fi
    local status
    status="$(aws cloudformation describe-stacks --stack-name "${stack_name}" --region "${AWS_REGION}" \
      --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo 'UNKNOWN')"
    if [[ "${status}" =~ COMPLETE$ ]]; then
      log "  Stack ${stack_name}: no updates needed (${status})"
    else
      warn "CFN error: ${cfn_error}"
      fail "Stack ${stack_name} deploy failed (status: ${status})"
    fi
  fi
}

# Helper: get a stack output value
get_stack_output() {
  local stack_name="$1"
  local output_key="$2"
  aws cloudformation describe-stacks \
    --stack-name "${stack_name}" \
    --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='${output_key}'].OutputValue" \
    --output text 2>/dev/null
}

# =========================================================================
# Resolve outputs from already-deployed stacks (for --start-at resumption)
# =========================================================================
if [[ "${START_AT}" -gt 1 ]]; then
  log "Resuming at step ${START_AT} — resolving existing stack outputs..."
  KMS_KEY_ARN="$(get_stack_output "${FEEDBACK_STACK}" "KMSKeyArn" 2>/dev/null || echo '')"
  [[ -n "${KMS_KEY_ARN}" ]] && log "  KMS_KEY_ARN=${KMS_KEY_ARN}"
fi
if [[ "${START_AT}" -gt 3 ]]; then
  PARAM_TABLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "ParameterStoreTableArn" 2>/dev/null || echo '')"
  AUDIT_TABLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "AuditTrailTableArn" 2>/dev/null || echo '')"
  [[ -n "${PARAM_TABLE_ARN}" ]] && log "  PARAM_TABLE_ARN=${PARAM_TABLE_ARN}"
fi

# =========================================================================
# Step 1: Feedback Pipeline (Kinesis/Firehose/S3/KMS)
# =========================================================================
if [[ "${START_AT}" -le 1 ]]; then
log "Step 1: Deploying Feedback Pipeline"
deploy_cfn_stack "${FEEDBACK_STACK}" "${SCRIPT_DIR}/feedback_pipeline_cfn.yaml" \
  "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
  "ParameterKey=EKSNodeRoleArn,ParameterValue=${EKS_NODE_ROLE}"

KMS_KEY_ARN="$(get_stack_output "${FEEDBACK_STACK}" "KMSKeyArn")"
[[ -n "${KMS_KEY_ARN}" ]] || fail "Could not retrieve KMS Key ARN from feedback pipeline stack"
log "  KMS Key: ${KMS_KEY_ARN}"

# =========================================================================
# Step 2: Glue ETL (Feature engineering + training data bucket)
# =========================================================================
fi # step 1

if [[ "${START_AT}" -le 2 ]]; then
log "Step 2: Deploying Glue ETL"

# Glue script bucket — user must pre-upload the script. Use a placeholder for
# initial deploy; the actual script path should be set via GLUE_SCRIPT_S3_PATH env var.
GLUE_SCRIPT_S3_PATH="${GLUE_SCRIPT_S3_PATH:-${STACK_PREFIX:+${STACK_PREFIX}-}artf-scripts-${ACCOUNT_ID}/etl/glue_feature_engineering.py}"
TRAINING_DATA_BUCKET="${STACK_PREFIX:+${STACK_PREFIX}-}training-data-${ACCOUNT_ID}-${AWS_REGION}"

deploy_cfn_stack "${GLUE_STACK}" "${SCRIPT_DIR}/glue_etl_cfn.yaml" \
  "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
  "ParameterKey=GlueScriptS3Path,ParameterValue=${GLUE_SCRIPT_S3_PATH}" \
  "ParameterKey=TrainingDataBucketName,ParameterValue=${TRAINING_DATA_BUCKET}"

# =========================================================================
# Step 3: Closed-Loop Core (DynamoDB/DAX/SageMaker Model Registry/SNS)
# =========================================================================
fi # step 2

if [[ "${START_AT}" -le 3 ]]; then
log "Step 3: Deploying Closed-Loop Core"
deploy_cfn_stack "${CLOSED_LOOP_STACK}" "${SCRIPT_DIR}/closed_loop_cfn.yaml" \
  "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
  "ParameterKey=KMSKeyArn,ParameterValue=${KMS_KEY_ARN}"

PARAM_TABLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "ParameterStoreTableArn")"
AUDIT_TABLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "AuditTrailTableArn")"
DAX_ENDPOINT="$(get_stack_output "${CLOSED_LOOP_STACK}" "DAXClusterEndpoint")"
SNS_TOPIC_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "TrainingAlertsTopicArn")"

log "  Parameter Store Table: ${PARAM_TABLE_ARN}"
log "  Audit Trail Table: ${AUDIT_TABLE_ARN}"
log "  DAX Endpoint: ${DAX_ENDPOINT}"
log "  SNS Topic: ${SNS_TOPIC_ARN}"

# =========================================================================
# Step 4: AgentCore Security (IAM roles, EventBridge rule + scheduler)
# =========================================================================
fi # step 3

if [[ "${START_AT}" -le 4 ]]; then
log "Step 4: Deploying AgentCore Security"

# The AgentCore runtime ARNs are needed. If agents are already deployed,
# read from env vars; otherwise use placeholder ARNs that will be updated
# after Step 5.
BID_SHADING_RUNTIME_ARN="${BID_SHADING_RUNTIME_ARN:-arn:aws:bedrock-agentcore:${AWS_REGION}:${ACCOUNT_ID}:runtime/bid-shading-agent-placeholder}"
GOVERNANCE_RUNTIME_ARN="${GOVERNANCE_RUNTIME_ARN:-arn:aws:bedrock-agentcore:${AWS_REGION}:${ACCOUNT_ID}:runtime/governance-agent-placeholder}"

deploy_cfn_stack "${AGENTCORE_SECURITY_STACK}" "${SCRIPT_DIR}/agentcore_security_cfn.yaml" \
  "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
  "ParameterKey=BidShadingAgentRuntimeArn,ParameterValue=${BID_SHADING_RUNTIME_ARN}" \
  "ParameterKey=GovernanceAgentRuntimeArn,ParameterValue=${GOVERNANCE_RUNTIME_ARN}" \
  "ParameterKey=ParameterStoreTableArn,ParameterValue=${PARAM_TABLE_ARN}" \
  "ParameterKey=AuditTrailTableArn,ParameterValue=${AUDIT_TABLE_ARN}" \
  "ParameterKey=KMSKeyArn,ParameterValue=${KMS_KEY_ARN}"

# =========================================================================
# Step 4b: Build and push NeMo-RL training container
# =========================================================================
TRAINING_REPO="artf-nemo-rl-training"
fi # step 4

log "Step 4b: Building NeMo-RL training container"

# Create ECR repo if needed
aws ecr describe-repositories --repository-names "${TRAINING_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${TRAINING_REPO}" --region "${AWS_REGION}" --image-scanning-configuration scanOnPush=true >/dev/null

TRAINING_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${TRAINING_REGISTRY}" 2>/dev/null

docker build -t "${TRAINING_REPO}:latest" "${SCRIPT_DIR}/../source/training/container/"
for tag in dlrm ncf widedeep; do
  docker tag "${TRAINING_REPO}:latest" "${TRAINING_REGISTRY}/${TRAINING_REPO}:${tag}"
  docker push "${TRAINING_REGISTRY}/${TRAINING_REPO}:${tag}"
done
log "  NeMo-RL training container pushed: ${TRAINING_REGISTRY}/${TRAINING_REPO}"

# =========================================================================
# Step 5: Deploy AgentCore Runtimes (Bid Shading + Governance agents)
# =========================================================================
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
  log "Step 5: Deploying AgentCore Runtimes"

  # Resolve the execution role (same role used by the MCP runtime)
  AC_ROLE_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}nvidia-artf-recommenders-agentcore-role"
  # Find the actual role (may have a suffix from deploy.sh)
  AC_ROLE_ARN="$(aws iam list-roles --query "Roles[?contains(RoleName,'${AC_ROLE_NAME}')].Arn | [0]" --output text 2>/dev/null || echo '')"
  if [[ -z "${AC_ROLE_ARN}" || "${AC_ROLE_ARN}" == "None" ]]; then
    warn "Could not find AgentCore execution role matching '${AC_ROLE_NAME}'"
    warn "Skipping agent runtime deployment — deploy the role first via the main deploy.sh"
  else
    # Build the dedicated bid shading agent container (ARM64 for AgentCore Firecracker)
    STACK_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}nvidia-artf-recommenders"
    REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    IMAGE_TAG="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo latest)"
    BID_SHADING_REPO="${STACK_NAME}-bid-shading-agent"
    BID_SHADING_IMAGE="${REGISTRY}/${BID_SHADING_REPO}:${IMAGE_TAG}"

    # Create ECR repo if needed
    aws ecr describe-repositories --repository-names "${BID_SHADING_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
      aws ecr create-repository --repository-name "${BID_SHADING_REPO}" --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true >/dev/null

    # Authenticate to ECR
    aws ecr get-login-password --region "${AWS_REGION}" | \
      docker login --username AWS --password-stdin "${REGISTRY}" 2>/dev/null

    # Build ARM64 image for AgentCore (Graviton-based microVMs)
    log "  Building bid shading agent image (arm64)..."
    docker buildx build \
      --platform linux/arm64 \
      -f "${SCRIPT_DIR}/../source/Dockerfile.bid-shading-agent" \
      -t "${BID_SHADING_IMAGE}" --load "${SCRIPT_DIR}/../source"
    docker push "${BID_SHADING_IMAGE}"
    log "  Pushed: ${BID_SHADING_IMAGE}"

    AC_IMAGE="${BID_SHADING_IMAGE}"

    # Deploy Bid Shading Strategy Agent (HTTP protocol — invoked by EventBridge + orchestrator)
    BID_SHADING_RUNTIME_NAME="BidShadingStrategyAgent"
    log "  Deploying ${BID_SHADING_RUNTIME_NAME}..."
    EXISTING_BID_SHADING="$(aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
      --query "agentRuntimes[?contains(agentRuntimeName,'${BID_SHADING_RUNTIME_NAME}')].agentRuntimeId | [0]" \
      --output text 2>/dev/null || echo 'None')"

    if [[ "${EXISTING_BID_SHADING}" == "None" || -z "${EXISTING_BID_SHADING}" ]]; then
      BID_SHADING_RESP="$(aws bedrock-agentcore-control create-agent-runtime \
        --agent-runtime-name "${BID_SHADING_RUNTIME_NAME}" \
        --role-arn "${AC_ROLE_ARN}" \
        --network-configuration '{"networkMode": "PUBLIC"}' \
        --protocol-configuration '{"serverProtocol": "HTTP"}' \
        --agent-runtime-artifact "{\"containerConfiguration\": {\"containerUri\": \"${AC_IMAGE}\"}}" \
        --region "${AWS_REGION}" \
        --output json 2>&1)"
      BID_SHADING_RUNTIME_ARN="$(echo "${BID_SHADING_RESP}" | jq -r '.agentRuntimeArn // empty')"
      log "  Created: ${BID_SHADING_RUNTIME_ARN}"
    else
      log "  Updating existing runtime: ${EXISTING_BID_SHADING}"
      aws bedrock-agentcore-control update-agent-runtime \
        --agent-runtime-id "${EXISTING_BID_SHADING}" \
        --role-arn "${AC_ROLE_ARN}" \
        --network-configuration '{"networkMode": "PUBLIC"}' \
        --protocol-configuration '{"serverProtocol": "HTTP"}' \
        --agent-runtime-artifact "{\"containerConfiguration\": {\"containerUri\": \"${AC_IMAGE}\"}}" \
        --region "${AWS_REGION}" >/dev/null 2>&1 || warn "Update failed for ${EXISTING_BID_SHADING}"
      BID_SHADING_RUNTIME_ARN="arn:aws:bedrock-agentcore:${AWS_REGION}:${ACCOUNT_ID}:runtime/${EXISTING_BID_SHADING}"
    fi

    # Wait for READY
    log "  Waiting for Bid Shading runtime to reach READY..."
    for i in $(seq 1 30); do
      STATUS="$(aws bedrock-agentcore-control get-agent-runtime \
        --agent-runtime-id "$(echo "${BID_SHADING_RUNTIME_ARN}" | awk -F/ '{print $NF}')" \
        --region "${AWS_REGION}" --query 'status' --output text 2>/dev/null || echo 'UNKNOWN')"
      if [[ "${STATUS}" == "READY" ]]; then
        log "  Bid Shading runtime READY: ${BID_SHADING_RUNTIME_ARN}"
        break
      fi
      sleep 10
    done

    # Export for use by the orchestrator env var injection
    export BID_SHADING_RUNTIME_ARN

    log "  AgentCore runtimes deployed."
    log "  BID_SHADING_RUNTIME_ARN=${BID_SHADING_RUNTIME_ARN}"
  fi
else
  warn "Skipping AgentCore deploy (--skip-agentcore)"
fi

# =========================================================================
# Summary
# =========================================================================
log ""
log "=========================================="
log " Closed-Loop Learning System — Deployed"
log "=========================================="
log ""
log "Infrastructure Stacks:"
log "  Feedback Pipeline:    ${FEEDBACK_STACK}"
log "  Glue ETL:            ${GLUE_STACK}"
log "  Closed-Loop Core:    ${CLOSED_LOOP_STACK}"
log "  AgentCore Security:  ${AGENTCORE_SECURITY_STACK}"
log ""
log "Key Resources:"
log "  KMS Key:             ${KMS_KEY_ARN}"
log "  Parameter Store:     ${PARAM_TABLE_ARN}"
log "  Audit Trail:         ${AUDIT_TABLE_ARN}"
log "  DAX Endpoint:        ${DAX_ENDPOINT}"
log "  SNS Alerts Topic:    ${SNS_TOPIC_ARN}"
log ""
log "Model Package Groups:"
log "  DLRM Bid Shader:     $(get_stack_output "${CLOSED_LOOP_STACK}" "DLRMModelPackageGroupName")"
log "  NCF Deal Manager:    $(get_stack_output "${CLOSED_LOOP_STACK}" "NCFModelPackageGroupName")"
log "  Wide&Deep Segment:   $(get_stack_output "${CLOSED_LOOP_STACK}" "WideDeepModelPackageGroupName")"
log ""
log "Next Steps:"
log "  1. Upload Glue ETL script to s3://${GLUE_SCRIPT_S3_PATH}"
log "  2. If AgentCore runtimes were just deployed, update the security stack"
log "     with the real runtime ARNs (re-run this script with env vars set)"
log "  3. Existing EKS/Triton manifests serve canary side-by-side — no changes needed"
log "  4. Verify with: aws cloudformation describe-stacks --stack-name ${CLOSED_LOOP_STACK}"
log ""
