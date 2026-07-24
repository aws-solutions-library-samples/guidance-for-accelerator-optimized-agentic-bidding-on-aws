#!/usr/bin/env bash
# =============================================================================
# deploy_closed_loop.sh — Deploy the Closed-Loop Learning System infrastructure
#
# Deploys the full closed-loop stack in dependency order:
#   1. Feedback Pipeline (Kinesis/Firehose/S3/KMS)
#   2. Glue ETL (Feature engineering job + training data bucket)
#   3. Closed-Loop Core (DynamoDB/DAX/SageMaker Model Registry/SNS,
#      SageMaker training execution role, genesis model registration)
#   4. AgentCore Security (IAM roles, EventBridge rule + scheduler)
#   5. AgentCore Runtimes (agentcore deploy for both agents)
#   6. Invocation paths (Scheduler->Adaptive, Rule->Governance,
#      Scheduler->CreateTrainingJob, Rule->CreateModelPackage on completion)
#
# Prerequisites:
#   - AWS CLI v2 with valid credentials
#   - Docker (only for local image builds with --local-build)
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
LOCAL_BUILD=0
NGC_SECRET="${NGC_SECRET:-}"
NGC_KEY="${NGC_KEY:-}"
# Bedrock model id for the reasoning agents (Adaptive Bidding + Governance).
# Required for Step 5. Override per-agent with ADAPTIVE_BIDDING_MODEL_ID / GOVERNANCE_MODEL_ID.
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-}"
ADAPTIVE_BIDDING_MODEL_ID="${ADAPTIVE_BIDDING_MODEL_ID:-${BEDROCK_MODEL_ID}}"
GOVERNANCE_MODEL_ID="${GOVERNANCE_MODEL_ID:-${BEDROCK_MODEL_ID}}"
# Passed by deploy.sh Step 11 (or set directly for standalone runs). Needed so the
# Model Promotion Governance runtime can reach the in-cluster Model Optimizer + Triton
# (VPC mode) and read/write the Triton model repo bucket.
MODEL_BUCKET="${MODEL_BUCKET:-}"
CLUSTER_NAME="${CLUSTER_NAME:-}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
START_AT=1

for arg in "$@"; do
  case "${arg}" in
    --skip-agentcore)  SKIP_AGENTCORE=1 ;;
    --stack-only)      STACK_ONLY=1; SKIP_AGENTCORE=1 ;;
    --local-build)     LOCAL_BUILD=1 ;;
    --remote-build)    LOCAL_BUILD=0 ;;
    --ngc-secret=*)    NGC_SECRET="${arg#--ngc-secret=}" ;;
    --ngc-secret)      ;;
    --ngc-key=*)       NGC_KEY="${arg#--ngc-key=}" ;;
    --ngc-key)         ;;
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
      elif [[ "${_PREV_ARG:-}" == "--ngc-secret" ]]; then
        NGC_SECRET="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--ngc-key" ]]; then
        NGC_KEY="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

log()  { printf '\033[0;32m[closed-loop]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[closed-loop][warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[closed-loop][fail]\033[0m %s\n' "$*" >&2; exit 1; }

# AgentCore VPC-supported Availability Zone IDs per region. Subnets outside these
# AZs fail AgentCore runtime creation. Extend as AgentCore adds regions.
_agentcore_supported_azs() {
  case "${AWS_REGION}" in
    us-east-1) echo "use1-az1 use1-az2 use1-az4" ;;
    us-east-2) echo "use2-az1 use2-az2 use2-az3" ;;
    us-west-2) echo "usw2-az1 usw2-az2 usw2-az3" ;;
    eu-west-1) echo "euw1-az1 euw1-az2 euw1-az3" ;;
    eu-central-1) echo "euc1-az1 euc1-az2 euc1-az3" ;;
    ap-south-1) echo "aps1-az1 aps1-az2 aps1-az3" ;;
    ap-southeast-1) echo "apse1-az1 apse1-az2 apse1-az3" ;;
    ap-southeast-2) echo "apse2-az1 apse2-az2 apse2-az3" ;;
    ap-northeast-1) echo "apne1-az1 apne1-az2 apne1-az4" ;;
    *) echo "" ;;
  esac
}

# Pick up to 2 EKS PRIVATE subnets in AgentCore-supported AZs for the governance
# runtime's VPC network mode (so it can reach the internal NLBs + egress via NAT).
# Prefers subnets tagged internal-elb (private); falls back to the passed SUBNET_IDS.
resolve_governance_subnets() {
  local vpc="$1"
  local supported; supported="$(_agentcore_supported_azs)"
  [[ -z "${supported}" ]] && { echo ""; return; }
  local candidates=""
  if [[ -n "${vpc}" ]]; then
    candidates="$(aws ec2 describe-subnets --region "${AWS_REGION}" \
      --filters "Name=vpc-id,Values=${vpc}" "Name=tag:kubernetes.io/role/internal-elb,Values=1" \
      --query 'Subnets[].SubnetId' --output text 2>/dev/null || echo '')"
  fi
  [[ -z "${candidates}" || "${candidates}" == "None" ]] && candidates="${SUBNET_IDS//,/ }"
  local chosen="" count=0
  for sn in ${candidates}; do
    [[ ${count} -ge 2 ]] && break
    local azid
    azid="$(aws ec2 describe-subnets --subnet-ids "${sn}" --region "${AWS_REGION}" \
      --query 'Subnets[0].AvailabilityZoneId' --output text 2>/dev/null || echo '')"
    for s in ${supported}; do
      if [[ "${azid}" == "${s}" ]]; then
        chosen="${chosen:+${chosen},}${sn}"; count=$((count+1)); break
      fi
    done
  done
  echo "${chosen}"
}

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
VPC_PROXY_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}vpc-proxy"
GOVERNANCE_EVENTBRIDGE_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}governance-eventbridge"

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
# TRAINING_DATA_BUCKET was resolved in Step 2. MODEL_BUCKET is passed in from
# deploy.sh Step 11 (the Triton model-repository bucket, already holding the
# genesis ONNX artifacts uploaded at deploy.sh Step 3a). Both are needed to
# scope the SageMaker training execution role (Condition: HasTrainingBuckets
# in closed_loop_cfn.yaml) - if either is empty the role is skipped and the
# stack still deploys (training/genesis-registration steps are then no-ops).
deploy_cfn_stack "${CLOSED_LOOP_STACK}" "${SCRIPT_DIR}/closed_loop_cfn.yaml" \
  "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
  "ParameterKey=KMSKeyArn,ParameterValue=${KMS_KEY_ARN}" \
  "ParameterKey=TrainingDataBucketName,ParameterValue=${TRAINING_DATA_BUCKET:-}" \
  "ParameterKey=ModelBucketName,ParameterValue=${MODEL_BUCKET:-}"

PARAM_TABLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "ParameterStoreTableArn")"
AUDIT_TABLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "AuditTrailTableArn")"
DAX_ENDPOINT="$(get_stack_output "${CLOSED_LOOP_STACK}" "DAXClusterEndpoint")"
SNS_TOPIC_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "TrainingAlertsTopicArn")"
SAGEMAKER_TRAINING_ROLE_ARN="$(get_stack_output "${CLOSED_LOOP_STACK}" "SageMakerTrainingExecutionRoleArn")"
DLRM_PACKAGE_GROUP="$(get_stack_output "${CLOSED_LOOP_STACK}" "DLRMModelPackageGroupName")"
NCF_PACKAGE_GROUP="$(get_stack_output "${CLOSED_LOOP_STACK}" "NCFModelPackageGroupName")"

log "  Parameter Store Table: ${PARAM_TABLE_ARN}"
log "  Audit Trail Table: ${AUDIT_TABLE_ARN}"
log "  SageMaker Training Execution Role: ${SAGEMAKER_TRAINING_ROLE_ARN:-<not created - TrainingDataBucketName/ModelBucketName empty>}"

# Idempotently register the genesis (v1, unretrained-starter) model version in
# each Model Package Group, from the ONNX artifact deploy.sh Step 3a already
# uploaded to s3://${MODEL_BUCKET}/onnx-source/<model>/model.onnx. Without this
# the registry starts empty and the first retraining job has nothing real to
# fine-tune from. Skips honestly (non-fatal) if MODEL_BUCKET is not set - that
# only happens when deploy_closed_loop.sh is run standalone without deploy.sh.
if [[ -n "${MODEL_BUCKET:-}" ]]; then
  log "  Registering genesis models (idempotent) from s3://${MODEL_BUCKET}/onnx-source/..."
  python3 "${SCRIPT_DIR}/scripts/register_genesis_models.py" \
    --model-bucket "${MODEL_BUCKET}" \
    --region "${AWS_REGION}" \
    --dlrm-package-group "${DLRM_PACKAGE_GROUP}" \
    --ncf-package-group "${NCF_PACKAGE_GROUP}" \
    --training-image-repository "artf-nemo-rl-training" \
    || warn "  Genesis model registration failed - Model Registry may be empty until it is re-run."
else
  warn "  MODEL_BUCKET not set - skipping genesis model registration (Model Registry will start empty)."
fi
log "  DAX Endpoint: ${DAX_ENDPOINT}"
log "  SNS Topic: ${SNS_TOPIC_ARN}"

# =========================================================================
# Step 4: AgentCore Security (IAM roles, EventBridge rule + scheduler)
# =========================================================================
fi # step 3

if [[ "${START_AT}" -le 4 ]]; then
log "Step 4: Deploying AgentCore execution roles (roles-only; deployed BEFORE the runtimes)"

# These roles use runtime-name-prefixed trust (ArnLike), so they do NOT depend on a
# runtime ARN and can be created before the runtimes. No placeholder ARNs.
# Resolve inputs so --start-at=4 works standalone.
PARAM_TABLE_ARN="${PARAM_TABLE_ARN:-$(get_stack_output "${CLOSED_LOOP_STACK}" "ParameterStoreTableArn")}"
AUDIT_TABLE_ARN="${AUDIT_TABLE_ARN:-$(get_stack_output "${CLOSED_LOOP_STACK}" "AuditTrailTableArn")}"
KMS_KEY_ARN="${KMS_KEY_ARN:-$(get_stack_output "${FEEDBACK_STACK}" "KMSKeyArn")}"
[[ -n "${PARAM_TABLE_ARN}" ]] || fail "Could not resolve ParameterStoreTableArn from ${CLOSED_LOOP_STACK}"
[[ -n "${AUDIT_TABLE_ARN}" ]] || fail "Could not resolve AuditTrailTableArn from ${CLOSED_LOOP_STACK}"
[[ -n "${KMS_KEY_ARN}" ]] || fail "Could not resolve KMSKeyArn from ${FEEDBACK_STACK}"

# Derive the DynamoDB table NAMES from their ARNs (arn:...:table/<name>). These are
# passed to the AgentCore runtimes as PARAMETER_STORE_TABLE / AUDIT_TABLE. Without
# this the agent gets an empty table name and falls back to the unprefixed default
# "parameter-store", reading an empty table when a stack prefix is in use (e.g. dv2).
PARAM_TABLE_NAME="${PARAM_TABLE_ARN##*/}"
AUDIT_TABLE_NAME="${AUDIT_TABLE_ARN##*/}"
[[ -n "${PARAM_TABLE_NAME}" ]] || fail "Could not derive parameter-store table name from ${PARAM_TABLE_ARN}"
[[ -n "${AUDIT_TABLE_NAME}" ]] || fail "Could not derive audit-trail table name from ${AUDIT_TABLE_ARN}"
log "  PARAM_TABLE_NAME=${PARAM_TABLE_NAME}  AUDIT_TABLE_NAME=${AUDIT_TABLE_NAME}"

# Idempotently seed the bidding parameters so the Adaptive Bidding agent has values
# to read on the scheduled EventBridge path (which never calls the orchestrator's
# /generate seeder). Safe to re-run — existing parameters are left untouched.
log "  Seeding parameter store (idempotent) so the agent works on the EventBridge path..."
python3 "${SCRIPT_DIR}/scripts/init_parameter_store.py" \
  --table-name "${PARAM_TABLE_NAME}" \
  --region "${AWS_REGION}" \
  --model-types "dlrm_bid_shader" \
  || warn "  Parameter store seed failed — the agent will read empty until a UI scenario seeds it."

deploy_cfn_stack "${AGENTCORE_SECURITY_STACK}" "${SCRIPT_DIR}/agentcore_security_cfn.yaml" \
  "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
  "ParameterKey=ParameterStoreTableArn,ParameterValue=${PARAM_TABLE_ARN}" \
  "ParameterKey=AuditTrailTableArn,ParameterValue=${AUDIT_TABLE_ARN}" \
  "ParameterKey=KMSKeyArn,ParameterValue=${KMS_KEY_ARN}" \
  "ParameterKey=ModelBucketName,ParameterValue=${MODEL_BUCKET:-}"

ADAPTIVE_BIDDING_ROLE_ARN="$(get_stack_output "${AGENTCORE_SECURITY_STACK}" "AdaptiveBiddingAgentRoleArn")"
GOVERNANCE_ROLE_ARN="$(get_stack_output "${AGENTCORE_SECURITY_STACK}" "ModelPromotionGovernanceAgentRoleArn")"
[[ -n "${ADAPTIVE_BIDDING_ROLE_ARN}" ]] || fail "Could not resolve AdaptiveBiddingAgentRoleArn from ${AGENTCORE_SECURITY_STACK}"
[[ -n "${GOVERNANCE_ROLE_ARN}" ]] || fail "Could not resolve ModelPromotionGovernanceAgentRoleArn from ${AGENTCORE_SECURITY_STACK}"
log "  Adaptive Bidding execution role: ${ADAPTIVE_BIDDING_ROLE_ARN}"
log "  Governance execution role: ${GOVERNANCE_ROLE_ARN}"

# =========================================================================
# Step 4b: Build and push NeMo-RL training container
# =========================================================================
TRAINING_REPO="artf-nemo-rl-training"
fi # step 4

log "Step 4b: Building NeMo-RL training container"

# Create ECR repo if needed
aws ecr describe-repositories --repository-names "${TRAINING_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${TRAINING_REPO}" --region "${AWS_REGION}" --image-scanning-configuration scanOnPush=true >/dev/null

# Check if NeMo images were already built
NEMO_OUTPUTS="${SCRIPT_DIR}/.nemo-outputs.json"
NEMO_BUILD_ID=""

if [[ -f "${NEMO_OUTPUTS}" ]]; then
  PREV_NEMO_REPO="$(jq -r '.Repository // empty' "${NEMO_OUTPUTS}" 2>/dev/null || echo '')"
  if [[ "${PREV_NEMO_REPO}" == "${TRAINING_REPO}" ]]; then
    if aws ecr describe-images --repository-name "${TRAINING_REPO}" \
        --image-ids imageTag="dlrm" --region "${AWS_REGION}" >/dev/null 2>&1; then
      log "  NeMo-RL images already built. Skipping. (Delete ${NEMO_OUTPUTS} to force rebuild)"
    else
      rm -f "${NEMO_OUTPUTS}"
    fi
  fi
fi

# Also check ECR directly (build may have finished on a previous interrupted run)
if [[ ! -f "${NEMO_OUTPUTS}" ]]; then
  if aws ecr describe-images --repository-name "${TRAINING_REPO}" \
      --image-ids imageTag="dlrm" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "  NeMo-RL images found in ECR. Skipping."
    TRAINING_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    cat > "${NEMO_OUTPUTS}" <<EOF
{
  "Repository": "${TRAINING_REPO}",
  "Registry": "${TRAINING_REGISTRY}",
  "Tags": ["dlrm", "ncf"],
  "BuiltAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
  fi
fi

# Build if needed
if [[ ! -f "${NEMO_OUTPUTS}" ]]; then
  TRAINING_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

  if [[ "${LOCAL_BUILD}" -eq 1 ]]; then
    # Local build (synchronous — requires ~25 GB free disk)
    aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${TRAINING_REGISTRY}" 2>/dev/null
    docker build --platform linux/amd64 -t "${TRAINING_REPO}:latest" "${SCRIPT_DIR}/../source/training/container/"
    for tag in dlrm ncf; do
      docker tag "${TRAINING_REPO}:latest" "${TRAINING_REGISTRY}/${TRAINING_REPO}:${tag}"
      docker push "${TRAINING_REGISTRY}/${TRAINING_REPO}:${tag}"
    done
    cat > "${NEMO_OUTPUTS}" <<EOF
{
  "Repository": "${TRAINING_REPO}",
  "Registry": "${TRAINING_REGISTRY}",
  "Tags": ["dlrm", "ncf"],
  "BuiltAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    log "  NeMo-RL training container pushed"
  else
    # Remote build via CodeBuild — fire async, continue deploying
    log "  Starting NeMo-RL build on CodeBuild (async — takes 15-50 min)"
    log "  Deployment will continue. The NeMo image is only needed for"
    log "  SageMaker training jobs, not for the runtime to function."
    STACK_NAME_CL="${STACK_PREFIX:+${STACK_PREFIX}-}nvidia-artf-recommenders"
    NGC_FLAG=""
    if [[ -n "${NGC_KEY}" ]]; then
      NGC_FLAG="--ngc-key ${NGC_KEY}"
    elif [[ -n "${NGC_SECRET}" ]]; then
      NGC_FLAG="--ngc-secret ${NGC_SECRET}"
    fi
    NEMO_BUILD_ID=$("${SCRIPT_DIR}/codebuild/remote_build.sh" \
      --stack-name "${STACK_NAME_CL}" \
      --target nemo \
      --tag latest \
      --region "${AWS_REGION}" \
      --no-wait \
      ${NGC_FLAG})
    log "  NeMo build started: ${NEMO_BUILD_ID}"
    log "  Continuing with remaining deployment steps..."
  fi
fi

# =========================================================================
# Step 5: Deploy AgentCore Runtimes (Adaptive Bidding + Model Promotion Governance)
#
# Both are deployed as HTTP-protocol AgentCore runtimes via
# scripts/deploy_to_agentcore.py (create-or-update, env vars, wait for READY).
# The Part-1 MCP runtime (nvidia_artf_recommenders_mcp) is separate and is NOT
# touched here.
# =========================================================================
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
  log "Step 5: Deploying AgentCore Runtimes"

  # Resolve the dedicated per-agent execution roles created in Step 4 (fallback to the
  # security stack outputs so --start-at=5 works). The runtimes use these least-privilege
  # roles — NOT the shared MCP runtime role.
  ADAPTIVE_BIDDING_ROLE_ARN="${ADAPTIVE_BIDDING_ROLE_ARN:-$(get_stack_output "${AGENTCORE_SECURITY_STACK}" "AdaptiveBiddingAgentRoleArn")}"
  GOVERNANCE_ROLE_ARN="${GOVERNANCE_ROLE_ARN:-$(get_stack_output "${AGENTCORE_SECURITY_STACK}" "ModelPromotionGovernanceAgentRoleArn")}"
  [[ -n "${ADAPTIVE_BIDDING_ROLE_ARN}" && "${ADAPTIVE_BIDDING_ROLE_ARN}" != "None" ]] || fail "Adaptive Bidding execution role not found — deploy Step 4 (${AGENTCORE_SECURITY_STACK}) first."
  [[ -n "${GOVERNANCE_ROLE_ARN}" && "${GOVERNANCE_ROLE_ARN}" != "None" ]] || fail "Governance execution role not found — deploy Step 4 (${AGENTCORE_SECURITY_STACK}) first."

  # Resolve the DynamoDB table names the agents need (from the closed-loop core stack).
  # Fetch here (not only in Step 3) so --start-at=5 still works.
  PARAM_TABLE_NAME="$(get_stack_output "${CLOSED_LOOP_STACK}" "ParameterStoreTableName")"
  AUDIT_TABLE_NAME="$(get_stack_output "${CLOSED_LOOP_STACK}" "AuditTrailTableName")"
  [[ -n "${PARAM_TABLE_NAME}" ]] || fail "Could not resolve ParameterStoreTableName from ${CLOSED_LOOP_STACK}"
  [[ -n "${AUDIT_TABLE_NAME}" ]] || fail "Could not resolve AuditTrailTableName from ${CLOSED_LOOP_STACK}"

  # Both agents are Bedrock reasoning agents — a model id is required. Fail fast rather
  # than shipping a possibly-unavailable default (no faking).
  [[ -n "${ADAPTIVE_BIDDING_MODEL_ID}" ]] || fail "BEDROCK_MODEL_ID (or ADAPTIVE_BIDDING_MODEL_ID) is required — set it to a Bedrock model/inference-profile id enabled in this account/region."
  [[ -n "${GOVERNANCE_MODEL_ID}" ]] || fail "BEDROCK_MODEL_ID (or GOVERNANCE_MODEL_ID) is required — set it to a Bedrock model/inference-profile id enabled in this account/region."

  STACK_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}nvidia-artf-recommenders"
  REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  IMAGE_TAG="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo latest)"

  # Authenticate to ECR once for all agent image pushes
  aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${REGISTRY}" 2>/dev/null

  # -----------------------------------------------------------------------
  # 5a. Adaptive Bidding Strategy Agent (HTTP, Bedrock reasoning agent)
  # -----------------------------------------------------------------------
  ADAPTIVE_BIDDING_REPO="${STACK_NAME}-adaptive-bidding-agent"
  ADAPTIVE_BIDDING_IMAGE="${REGISTRY}/${ADAPTIVE_BIDDING_REPO}:${IMAGE_TAG}"

  aws ecr describe-repositories --repository-names "${ADAPTIVE_BIDDING_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${ADAPTIVE_BIDDING_REPO}" --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true >/dev/null

  log "  Building Adaptive Bidding agent image (arm64)..."
  docker buildx build \
    --platform linux/arm64 \
    -f "${SCRIPT_DIR}/../source/Dockerfile.adaptive-bidding-agent" \
    -t "${ADAPTIVE_BIDDING_IMAGE}" --load "${SCRIPT_DIR}/../source"
  docker push "${ADAPTIVE_BIDDING_IMAGE}"
  log "  Pushed: ${ADAPTIVE_BIDDING_IMAGE}"

  log "  Deploying AdaptiveBiddingStrategyAgent (HTTP)..."
  ADAPTIVE_BIDDING_RUNTIME_ARN="$(python3 "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
    --action deploy \
    --runtime-name "AdaptiveBiddingStrategyAgent" \
    --role-arn "${ADAPTIVE_BIDDING_ROLE_ARN}" \
    --container-uri "${ADAPTIVE_BIDDING_IMAGE}" \
    --protocol HTTP \
    --environment "PARAMETER_STORE_TABLE=${PARAM_TABLE_NAME}" \
    --environment "AWS_REGION=${AWS_REGION}" \
    --environment "ADAPTIVE_BIDDING_MODEL_ID=${ADAPTIVE_BIDDING_MODEL_ID}" \
    --description "Adaptive Bidding Strategy Agent — Bedrock reasoning agent for bid parameter tuning" \
    --region "${AWS_REGION}" \
    --print-arn)"
  [[ -n "${ADAPTIVE_BIDDING_RUNTIME_ARN}" ]] || fail "AdaptiveBiddingStrategyAgent deploy did not return a runtime ARN"
  export ADAPTIVE_BIDDING_RUNTIME_ARN
  log "  ADAPTIVE_BIDDING_RUNTIME_ARN=${ADAPTIVE_BIDDING_RUNTIME_ARN}"

  # -----------------------------------------------------------------------
  # 5b. Model Promotion Governance Agent (HTTP, reasoning + deterministic gate)
  # -----------------------------------------------------------------------
  GOVERNANCE_REPO="${STACK_NAME}-model-promotion-governance-agent"
  GOVERNANCE_IMAGE="${REGISTRY}/${GOVERNANCE_REPO}:${IMAGE_TAG}"

  aws ecr describe-repositories --repository-names "${GOVERNANCE_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${GOVERNANCE_REPO}" --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true >/dev/null

  log "  Building Model Promotion Governance agent image (arm64)..."
  docker buildx build \
    --platform linux/arm64 \
    -f "${SCRIPT_DIR}/../source/Dockerfile.governance" \
    -t "${GOVERNANCE_IMAGE}" --load "${SCRIPT_DIR}/../source"
  docker push "${GOVERNANCE_IMAGE}"
  log "  Pushed: ${GOVERNANCE_IMAGE}"

  # ---- Resolve VPC networking + in-cluster endpoints for the governance runtime ----
  # The governance agent reaches the Model Optimizer + Triton over their INTERNAL
  # NLBs, which are only routable from inside the cluster VPC — hence VPC network mode.
  if [[ -n "${CLUSTER_NAME}" ]]; then
    aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || true
  fi
  # The Model Optimizer is now ON-DEMAND: the VPC proxy Lambda intercepts
  # POST .../v1/optimize and launches a one-shot K8s Job (no always-on optimizer
  # Service/NLB). OPTIMIZER_ENDPOINT is a sentinel the Lambda matches by path — it
  # never connects to this host — so it just needs to be a stable non-empty value.
  OPTIMIZER_ENDPOINT="http://model-optimizer.on-demand:8080"
  TRITON_HOST="$(kubectl get svc triton-internal -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo '')"
  GOV_TRITON_URL="${TRITON_HOST:+http://${TRITON_HOST}:8000}"
  [[ -n "${GOV_TRITON_URL}" ]] || warn "  triton-internal NLB not resolved yet — TRITON_URL empty (governance canary step fails honestly until it is)"

  # ---- VPC HTTP proxy Lambda: the governance runtime stays PUBLIC and reaches the
  #      cluster-internal optimizer/Triton NLBs by invoking this VPC-attached Lambda
  #      (agent -> lambda:InvokeFunction -> internal NLB). A Lambda has no
  #      AgentCore-supported-AZ restriction, so any private subnet works.
  GOV_VPC_ID="${VPC_ID}"
  if [[ -z "${GOV_VPC_ID}" && -n "${CLUSTER_NAME}" ]]; then
    GOV_VPC_ID="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --query 'cluster.resourcesVpcConfig.vpcId' --output text 2>/dev/null || echo '')"
  fi
  GOV_SG=""
  if [[ -n "${CLUSTER_NAME}" ]]; then
    GOV_SG="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text 2>/dev/null || echo '')"
  fi
  # Private subnets (any AZ) for the proxy Lambda ENIs — prefer internal-elb-tagged
  # (private) subnets, fall back to the passed SUBNET_IDS.
  PROXY_SUBNETS=""
  if [[ -n "${GOV_VPC_ID}" ]]; then
    PROXY_SUBNETS="$(aws ec2 describe-subnets --region "${AWS_REGION}" \
      --filters "Name=vpc-id,Values=${GOV_VPC_ID}" "Name=tag:kubernetes.io/role/internal-elb,Values=1" \
      --query 'Subnets[].SubnetId' --output text 2>/dev/null | tr '\t' ',' || echo '')"
  fi
  [[ -z "${PROXY_SUBNETS}" || "${PROXY_SUBNETS}" == "None" ]] && PROXY_SUBNETS="${SUBNET_IDS}"

  VPC_PROXY_LAMBDA_ARN=""
  if [[ -n "${PROXY_SUBNETS}" && -n "${GOV_SG}" && "${GOV_SG}" != "None" ]]; then
    # CloudFormation List<> params require commas escaped in the CLI shorthand.
    PROXY_SUBNETS_ESC="${PROXY_SUBNETS//,/\\,}"
    OPTIMIZER_IMAGE="${REGISTRY}/${STACK_NAME}-model-optimizer:${IMAGE_TAG}"
    JOB_NAMESPACE="default"
    deploy_cfn_stack "${VPC_PROXY_STACK}" "${SCRIPT_DIR}/vpc_proxy_cfn.yaml" \
      "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
      "ParameterKey=SubnetIds,ParameterValue=${PROXY_SUBNETS_ESC}" \
      "ParameterKey=SecurityGroupIds,ParameterValue=${GOV_SG}" \
      "ParameterKey=ClusterName,ParameterValue=${CLUSTER_NAME}" \
      "ParameterKey=OptimizerImage,ParameterValue=${OPTIMIZER_IMAGE}" \
      "ParameterKey=OptimizerServiceAccount,ParameterValue=model-optimizer-sa" \
      "ParameterKey=JobNamespace,ParameterValue=${JOB_NAMESPACE}"
    VPC_PROXY_LAMBDA_ARN="$(get_stack_output "${VPC_PROXY_STACK}" "ProxyFunctionArn")"
    VPC_PROXY_ROLE_ARN="$(get_stack_output "${VPC_PROXY_STACK}" "ProxyFunctionRoleArn")"
    log "  VPC proxy Lambda: ${VPC_PROXY_LAMBDA_ARN:-<unresolved>}"

    # Grant the proxy Lambda's IAM role K8s access to launch on-demand optimize Jobs:
    #   (1) map the role -> K8s group 'optimizer-job-launcher' (idempotent), and
    #   (2) apply the namespaced Role/RoleBinding for that group.
    # Without this the Lambda's K8s API calls are unauthorized and promotions fail
    # honestly (the optimize Job is never created).
    if [[ -n "${VPC_PROXY_ROLE_ARN}" && "${VPC_PROXY_ROLE_ARN}" != "None" && -n "${CLUSTER_NAME}" ]]; then
      eksctl create iamidentitymapping \
        --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" \
        --arn "${VPC_PROXY_ROLE_ARN}" \
        --group optimizer-job-launcher --username optimizer-lambda \
        --no-duplicate-arns >/dev/null 2>&1 \
        || warn "  Could not map the proxy Lambda role into aws-auth (need eksctl + cluster admin); on-demand optimize stays unauthorized until mapped."
      sed "s|__JOB_NAMESPACE__|${JOB_NAMESPACE}|g" \
        "${SCRIPT_DIR}/eks/optimizer-job-launcher-rbac.yaml" | kubectl apply -f - >/dev/null 2>&1 \
        || warn "  Could not apply optimizer-job-launcher RBAC; on-demand optimize stays unauthorized until applied."
      log "  On-demand optimize RBAC configured (group optimizer-job-launcher, ns ${JOB_NAMESPACE})."
    fi
  else
    warn "  No VPC subnets/security group resolved — skipping the VPC proxy Lambda."
    warn "  The governance runtime deploys PUBLIC and its optimize/canary step will"
    warn "  fail honestly until a cluster VPC is available (deploy via deploy.sh --with-retraining)."
  fi

  log "  Deploying ModelPromotionGovernanceAgent (HTTP)..."
  GOVERNANCE_RUNTIME_ARN="$(python3 "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
    --action deploy \
    --runtime-name "ModelPromotionGovernanceAgent" \
    --role-arn "${GOVERNANCE_ROLE_ARN}" \
    --container-uri "${GOVERNANCE_IMAGE}" \
    --protocol HTTP \
    --environment "AUDIT_TABLE=${AUDIT_TABLE_NAME}" \
    --environment "AWS_REGION=${AWS_REGION}" \
    --environment "GOVERNANCE_MODEL_ID=${GOVERNANCE_MODEL_ID}" \
    --environment "MODEL_BUCKET=${MODEL_BUCKET:-}" \
    --environment "OPTIMIZER_ENDPOINT=${OPTIMIZER_ENDPOINT:-}" \
    --environment "TRITON_URL=${GOV_TRITON_URL:-}" \
    --environment "VPC_PROXY_LAMBDA_ARN=${VPC_PROXY_LAMBDA_ARN:-}" \
    --description "Model Promotion Governance Agent — deterministic A/B gate + Bedrock reasoning" \
    --region "${AWS_REGION}" \
    --print-arn)"
  [[ -n "${GOVERNANCE_RUNTIME_ARN}" ]] || fail "ModelPromotionGovernanceAgent deploy did not return a runtime ARN"
  export GOVERNANCE_RUNTIME_ARN
  log "  GOVERNANCE_RUNTIME_ARN=${GOVERNANCE_RUNTIME_ARN}"

  log "  AgentCore runtime deploys complete."

  # -----------------------------------------------------------------------
  # Step 6: Invocation paths (Scheduler->Lambda->Adaptive, Rule->Shim->Governance,
  # Scheduler->Lambda->CreateTrainingJob, Rule->Lambda->CreateModelPackage).
  # Deployed AFTER the runtimes so the real runtime ARNs are available (no placeholders).
  #
  # The scheduled-retraining resources (HasRetrainingConfig in
  # governance_eventbridge_cfn.yaml) only get created if ALL of these resolve to
  # non-empty: the training role + both buckets (from Step 3) and both Model
  # Package Group names (also from Step 3). Falls back to stack-output lookups so
  # --start-at=5/6 works standalone.
  # -----------------------------------------------------------------------
  log "Step 6: Deploying invocation stack (${GOVERNANCE_EVENTBRIDGE_STACK})"
  SAGEMAKER_TRAINING_ROLE_ARN="${SAGEMAKER_TRAINING_ROLE_ARN:-$(get_stack_output "${CLOSED_LOOP_STACK}" "SageMakerTrainingExecutionRoleArn")}"
  DLRM_PACKAGE_GROUP="${DLRM_PACKAGE_GROUP:-$(get_stack_output "${CLOSED_LOOP_STACK}" "DLRMModelPackageGroupName")}"
  NCF_PACKAGE_GROUP="${NCF_PACKAGE_GROUP:-$(get_stack_output "${CLOSED_LOOP_STACK}" "NCFModelPackageGroupName")}"
  TRAINING_DATA_BUCKET="${TRAINING_DATA_BUCKET:-$(get_stack_output "${GLUE_STACK}" "TrainingDataBucketName")}"
  TRAINING_IMAGE_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

  if [[ -z "${SAGEMAKER_TRAINING_ROLE_ARN}" || "${SAGEMAKER_TRAINING_ROLE_ARN}" == "None" ]]; then
    warn "  SageMakerTrainingExecutionRoleArn not resolved - scheduled retraining will be skipped (Step 3 needs TrainingDataBucketName/ModelBucketName set)."
  fi

  deploy_cfn_stack "${GOVERNANCE_EVENTBRIDGE_STACK}" "${SCRIPT_DIR}/governance_eventbridge_cfn.yaml" \
    "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX:-artf}" \
    "ParameterKey=AdaptiveBiddingAgentRuntimeArn,ParameterValue=${ADAPTIVE_BIDDING_RUNTIME_ARN}" \
    "ParameterKey=ModelPromotionGovernanceAgentRuntimeArn,ParameterValue=${GOVERNANCE_RUNTIME_ARN}" \
    "ParameterKey=SageMakerTrainingExecutionRoleArn,ParameterValue=${SAGEMAKER_TRAINING_ROLE_ARN:-}" \
    "ParameterKey=ModelBucketName,ParameterValue=${MODEL_BUCKET:-}" \
    "ParameterKey=TrainingDataBucketName,ParameterValue=${TRAINING_DATA_BUCKET:-}" \
    "ParameterKey=TrainingImageRegistry,ParameterValue=${TRAINING_IMAGE_REGISTRY}" \
    "ParameterKey=DLRMModelPackageGroupName,ParameterValue=${DLRM_PACKAGE_GROUP:-}" \
    "ParameterKey=NCFModelPackageGroupName,ParameterValue=${NCF_PACKAGE_GROUP:-}"
  log "  Invocation stack deployed."

  RETRAINING_SCHEDULE_ARN="$(get_stack_output "${GOVERNANCE_EVENTBRIDGE_STACK}" "RetrainingScheduleArn")"
  if [[ -n "${RETRAINING_SCHEDULE_ARN}" && "${RETRAINING_SCHEDULE_ARN}" != "None" ]]; then
    log "  Scheduled retraining ENABLED: ${RETRAINING_SCHEDULE_ARN}"
  else
    warn "  Scheduled retraining NOT enabled (missing training role/buckets/package groups)."
  fi

  # -----------------------------------------------------------------------
  # Step 7: Rewire the UI with the now-known agent runtime ARNs.
  #
  # The React UI bakes the agent runtime ARNs in at BUILD time (VITE_* env). Those
  # ARNs only exist after Step 5, so any earlier UI build (deploy.sh Step 9) shipped
  # empty ARNs and the UI honestly shows "agent not deployed". Now that both runtimes
  # are live, grant the Cognito Identity-Pool auth role scoped InvokeAgentRuntime and
  # rebuild+redeploy the UI so the browser can call the agents directly via SigV4 (FR-6).
  #
  # Guard: a frontend re-deploy REPLACES the CloudFront origins. deploy_frontend.py
  # only keeps the /api (ALB) origin when it is given a real orchestrator URL — a
  # localhost fallback would DROP that origin and break the live UI. So if the
  # orchestrator NLB can't be resolved we skip the rebuild and say so, rather than
  # ship a broken distribution.
  # -----------------------------------------------------------------------
  if [[ -n "${ADAPTIVE_BIDDING_RUNTIME_ARN}" || -n "${GOVERNANCE_RUNTIME_ARN}" ]]; then
    log "Step 7: Rewiring the UI with the closed-loop agent runtime ARNs"

    # Grant the Identity Pool authenticated role least-privilege InvokeAgentRuntime,
    # scoped to exactly these two runtimes (+ their DEFAULT endpoint sub-resources).
    log "  Granting scoped InvokeAgentRuntime to the Cognito Identity Pool auth role..."
    python3 "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
      --action grant-agent-invoke \
      --stack-name "${STACK_NAME}" \
      --region "${AWS_REGION}" \
      --adaptive-runtime-arn "${ADAPTIVE_BIDDING_RUNTIME_ARN}" \
      --governance-runtime-arn "${GOVERNANCE_RUNTIME_ARN}" \
      || warn "  Could not attach scoped InvokeAgentRuntime policy to the auth role"

    # Resolve the orchestrator NLB so the CloudFront /api origin (ALB) is preserved
    # on re-deploy. Requires cluster access (CLUSTER_NAME is passed by deploy.sh).
    if [[ -n "${CLUSTER_NAME}" ]]; then
      aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || true
    fi
    UI_NLB_DNS=""
    if command -v kubectl >/dev/null 2>&1; then
      UI_NLB_DNS="$(kubectl get svc orchestrator -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo '')"
    fi

    if [[ -z "${UI_NLB_DNS}" ]]; then
      warn "  Could not resolve the orchestrator NLB endpoint (cluster unreachable?)."
      warn "  Skipping the UI rebuild: re-deploying with a placeholder URL would drop the"
      warn "  CloudFront /api origin and break the live UI. Re-run with CLUSTER_NAME set,"
      warn "  or run  ./deploy.sh --ui-only  to rebuild the UI with the current agent ARNs."
    else
      # Read the existing Cognito auth config so the rebuilt bundle keeps working logins.
      COGNITO_OUTPUTS="${SCRIPT_DIR}/.cognito-outputs.json"
      COGNITO_USER_POOL_ID="$(jq -r '.UserPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
      COGNITO_CLIENT_ID="$(jq -r '.ClientId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
      IDENTITY_POOL_ID="$(jq -r '.IdentityPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
      [[ -n "${COGNITO_USER_POOL_ID}" ]] || warn "  Cognito outputs not found (${COGNITO_OUTPUTS}) — UI auth vars will be empty."

      REACT_ENV="${SCRIPT_DIR}/../source/frontend-react/.env.production"
      cat > "${REACT_ENV}" <<EOF
VITE_COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
VITE_COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
VITE_COGNITO_REGION=${AWS_REGION}
VITE_IDENTITY_POOL_ID=${IDENTITY_POOL_ID}
VITE_ADAPTIVE_BIDDING_RUNTIME_ARN=${ADAPTIVE_BIDDING_RUNTIME_ARN}
VITE_GOVERNANCE_RUNTIME_ARN=${GOVERNANCE_RUNTIME_ARN}
EOF
      log "  UI env: adaptive=${ADAPTIVE_BIDDING_RUNTIME_ARN:-<none>}  governance=${GOVERNANCE_RUNTIME_ARN:-<none>}"

      log "  Rebuilding + redeploying the React UI..."
      python3 "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
        --action deploy \
        --stack-name "${STACK_NAME}" \
        --region "${AWS_REGION}" \
        --orchestrator-url "http://${UI_NLB_DNS}" \
        || warn "  Frontend re-deploy failed — UI still shows the previous (empty-ARN) build."
      log "  UI rebuilt with agent runtime ARNs."
    fi
  fi
else
  warn "Skipping AgentCore deploy (--skip-agentcore) — runtimes and invocation stack not deployed"
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
log "  Feedback Pipeline:      ${FEEDBACK_STACK}"
log "  Glue ETL:               ${GLUE_STACK}"
log "  Closed-Loop Core:       ${CLOSED_LOOP_STACK}"
log "  AgentCore Roles:        ${AGENTCORE_SECURITY_STACK}"
log "  Invocation (EventBridge): ${GOVERNANCE_EVENTBRIDGE_STACK}"
log ""
log "AgentCore Runtimes:"
log "  Adaptive Bidding:       ${ADAPTIVE_BIDDING_RUNTIME_ARN:-<not deployed>}"
log "  Model Promotion Gov.:   ${GOVERNANCE_RUNTIME_ARN:-<not deployed>}"
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
log "  (Wide & Deep Segment Activator is rule-based — no Model Package Group)"
log ""
log "Training Pipeline:"
log "  Training Execution Role: $(get_stack_output "${CLOSED_LOOP_STACK}" "SageMakerTrainingExecutionRoleArn" 2>/dev/null || echo '<not created>')"
log "  Scheduled Retraining:    $(get_stack_output "${GOVERNANCE_EVENTBRIDGE_STACK}" "RetrainingScheduleArn" 2>/dev/null || echo '<not enabled>')"
log ""
log "Next Steps:"
log "  1. Upload Glue ETL script to s3://${GLUE_SCRIPT_S3_PATH}"
log "  2. Roles are created before runtimes and the invocation stack uses the real"
log "     runtime ARNs — no placeholder reconciliation needed."
log "  3. Existing EKS/Triton manifests serve canary side-by-side — no changes needed"
log "  4. Verify with: aws cloudformation describe-stacks --stack-name ${CLOSED_LOOP_STACK}"
log "  5. Genesis models are registered automatically in Step 3 if MODEL_BUCKET is set."
log "  6. Scheduled retraining fires every 6h once the training container (Step 4b)"
log "     has finished building - check with ./check_builds.sh --prefix ${STACK_PREFIX:-<prefix>}"
log ""

# =========================================================================
# Final check: NeMo build status
# =========================================================================
if [[ -n "${NEMO_BUILD_ID}" && ! -f "${NEMO_OUTPUTS}" ]]; then
  # NeMo was fired async — check if it finished while we deployed everything else
  if aws ecr describe-images --repository-name "${TRAINING_REPO}" \
      --image-ids imageTag="dlrm" --region "${AWS_REGION}" >/dev/null 2>&1; then
    TRAINING_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    cat > "${NEMO_OUTPUTS}" <<EOF
{
  "Repository": "${TRAINING_REPO}",
  "Registry": "${TRAINING_REGISTRY}",
  "Tags": ["dlrm", "ncf"],
  "BuiltAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    log "NeMo-RL build completed while deploying. All done."
  else
    echo ""
    warn "═══════════════════════════════════════════════════════════════════"
    warn "  The NeMo-RL training container is still building on CodeBuild."
    warn "  Everything else is deployed and functional."
    warn ""
    warn "  The NeMo image is only needed for SageMaker retraining jobs —"
    warn "  the inference pipeline, UI, and agents all work without it."
    warn ""
    warn "  Check build status:"
    warn "    ./check_builds.sh --prefix ${STACK_PREFIX:-<prefix>}"
    warn ""
    warn "  Or watch until ready:"
    warn "    ./check_builds.sh --prefix ${STACK_PREFIX:-<prefix>} --watch"
    warn "═══════════════════════════════════════════════════════════════════"
    echo ""
  fi
fi
