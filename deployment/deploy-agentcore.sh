#!/usr/bin/env bash
# =============================================================================
# deploy-agentcore.sh — Deploy Accelerator-optimized Agentic Bidding to Bedrock AgentCore
#
# This script:
#   1. Checks prerequisites (aws, docker, agentcore CLI)
#   2. Creates an ECR repository for the AgentCore container
#   3. Builds the ARM64 container image
#   4. Pushes to ECR
#   5. Creates an IAM execution role for the AgentCore runtime
#   6. Creates or updates the AgentCore runtime (MCP protocol)
#   7. Waits for READY status
#   8. Prints the runtime ARN and invoke instructions
#
# Usage:
#   AWS_REGION=us-east-1 ./deploy-agentcore.sh
#   AWS_REGION=us-east-1 ./deploy-agentcore.sh --destroy   # tear down
#
# Prerequisites:
#   - AWS CLI v2 with valid credentials
#   - Docker (with buildx for ARM64 cross-compilation)
#   - Python 3.11+ with boto3 installed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_PREFIX="${STACK_PREFIX:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

DESTROY=0
STACK_PREFIX="${STACK_PREFIX:-}"
for arg in "$@"; do
  case "${arg}" in
    --destroy)  DESTROY=1 ;;
    --prefix=*) STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)   ;;
    -h|--help)  head -20 "$0"; exit 0 ;;
    *)
      if [[ "${_PREV_ARG:-}" == "--prefix" ]]; then
        STACK_PREFIX="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

# Resource names derive from STACK_PREFIX, which may have been set by --prefix above.
RUNTIME_NAME="$(echo "${STACK_PREFIX:+${STACK_PREFIX}_}nvidia_artf_recommenders_mcp" | tr '-' '_')"
ECR_REPO_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}nvidia-artf-recommenders-agentcore"

log()  { printf '\033[0;32m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[deploy][warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[deploy][fail]\033[0m %s\n' "$*" >&2; exit 1; }

# -------------------------------------------------------------------------
# 1. Preflight
# -------------------------------------------------------------------------
log "Preflight checks"
for bin in aws docker python3; do
  command -v "${bin}" >/dev/null 2>&1 || fail "missing: ${bin}"
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "${ACCOUNT_ID}" ]] || fail "cannot resolve AWS account"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO_NAME}:${IMAGE_TAG}"
# ROLE_NAME depends on ACCOUNT_ID (resolved just above) and RUNTIME_NAME.
ROLE_NAME="NvidiaArtfAgentCoreRole-$(python3 -c "import hashlib; print(hashlib.sha256('${RUNTIME_NAME}:${ACCOUNT_ID}:${AWS_REGION}'.encode()).hexdigest()[:8])")"

log "Account=${ACCOUNT_ID}  Region=${AWS_REGION}  Image=${IMAGE_URI}"

# -------------------------------------------------------------------------
# Destroy path
# -------------------------------------------------------------------------
if [[ "${DESTROY}" -eq 1 ]]; then
  warn "Destroying AgentCore runtime ${RUNTIME_NAME}"
  python3 "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
    --action destroy \
    --runtime-name "${RUNTIME_NAME}" \
    --region "${AWS_REGION}"
  log "Destroyed. ECR repo and IAM role are retained — delete manually if needed."
  exit 0
fi

# -------------------------------------------------------------------------
# 2. Create ECR repository (idempotent)
# -------------------------------------------------------------------------
log "Ensuring ECR repository ${ECR_REPO_NAME}"
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository \
    --repository-name "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE >/dev/null

# -------------------------------------------------------------------------
# 3. Build ARM64 container
# -------------------------------------------------------------------------
log "Building ARM64 container image"
docker buildx build \
  --platform linux/arm64 \
  -f "${SCRIPT_DIR}/../source/Dockerfile.agentcore" \
  -t "${IMAGE_URI}" \
  --load \
  "${SCRIPT_DIR}/../source"

# -------------------------------------------------------------------------
# 4. Push to ECR
# -------------------------------------------------------------------------
log "Pushing to ECR"
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"
docker push "${IMAGE_URI}"

# -------------------------------------------------------------------------
# 5. Create IAM role (idempotent)
# -------------------------------------------------------------------------
log "Ensuring IAM execution role ${ROLE_NAME}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  log "Creating IAM role ${ROLE_NAME}"
  TRUST_POLICY='{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock-agentcore.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }]
  }'
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" \
    --description "Execution role for Accelerator-optimized Agentic Bidding AgentCore runtime" >/dev/null

  # Attach policies for ECR pull, CloudWatch logs, X-Ray
  aws iam attach-role-policy --role-name "${ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  aws iam attach-role-policy --role-name "${ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
  aws iam attach-role-policy --role-name "${ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess"

  log "Waiting 10s for IAM propagation..."
  sleep 10
else
  log "IAM role ${ROLE_NAME} already exists"
fi

# -------------------------------------------------------------------------
# 6–8. Create/update AgentCore runtime and wait for READY
# -------------------------------------------------------------------------
log "Deploying to AgentCore"
python3 "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
  --action deploy \
  --runtime-name "${RUNTIME_NAME}" \
  --role-arn "${ROLE_ARN}" \
  --container-uri "${IMAGE_URI}" \
  --region "${AWS_REGION}"

log "Deploy complete."
