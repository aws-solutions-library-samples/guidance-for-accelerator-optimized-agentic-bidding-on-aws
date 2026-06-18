#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy the Accelerator-optimized Agentic Bidding solution to AWS
#
# Deploys:
#   1. ECR repositories for all container images
#   2. Export PyTorch models to ONNX for NVIDIA Triton
#   3. Upload ONNX model repository to S3
#   4. Build and push container images (tritonclient-backed + orchestrator)
#   5. Create or update EKS cluster with GPU node group (g5.xlarge / A10G)
#   6. Install NVIDIA Kubernetes Device Plugin
#   7. Configure IRSA for Triton S3 model access
#   8. Deploy Kubernetes manifests (Triton server, ARTF containers, orchestrator)
#   9. Deploy frontend to S3 + CloudFront (React UI)
#  10. Deploy Bedrock AgentCore MCP runtime
#
# Usage:
#   ./deploy.sh                                # full deploy
#   ./deploy.sh --prefix v1                    # resources named v1-nvidia-artf-*
#   ./deploy.sh --prefix prod --skip-agentcore # combine flags
#   ./deploy.sh --ui-only                      # redeploy frontend only (fast)
#   ./deploy.sh --skip-cluster                 # reuse existing EKS cluster
#   ./deploy.sh --export-only                  # just export ONNX models, no deploy
#   ./deploy.sh --maxGPUs 5                     # cap GPU node group max size at 5 (default 3)
#   ./deploy.sh --destroy                      # tear down the entire stack
#   AWS_REGION=us-west-2 ./deploy.sh           # different region
#
# Prerequisites:
#   - AWS CLI v2 with credentials
#   - Docker with buildx
#   - Python 3.11+ with boto3, torch, onnx, onnxscript
#   - jq, eksctl, kubectl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_PREFIX="${STACK_PREFIX:-}"
STACK_NAME="${STACK_NAME:-nvidia-artf-recommenders}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo latest)}"

DESTROY=0
SKIP_AGENTCORE=0
SKIP_IMAGES=0
UI_ONLY=0
SKIP_CLUSTER=0
EXPORT_ONLY=0
MAX_GPUS="${MAX_GPUS:-3}"
STACK_PREFIX="${STACK_PREFIX:-}"
for arg in "$@"; do
  case "${arg}" in
    --destroy)          DESTROY=1 ;;
    --skip-agentcore)   SKIP_AGENTCORE=1 ;;
    --skip-images)      SKIP_IMAGES=1 ;;
    --ui-only)          UI_ONLY=1 ;;
    --skip-cluster)     SKIP_CLUSTER=1 ;;
    --export-only)      EXPORT_ONLY=1 ;;
    --prefix=*)         STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)           ;; # value comes in next arg, handled below
    --maxGPUs=*)        MAX_GPUS="${arg#--maxGPUs=}" ;;
    --maxGPUs)          ;; # value comes in next arg, handled below
    -h|--help)          sed -n '2,24p' "$0"; exit 0 ;;
    *)
      if [[ "${_PREV_ARG:-}" == "--prefix" ]]; then
        STACK_PREFIX="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--maxGPUs" ]]; then
        MAX_GPUS="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

# Validate --maxGPUs: must be a positive integer (it caps the GPU node group's maxSize)
if ! [[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  printf '\033[0;31m[fail]\033[0m %s\n' "--maxGPUs must be a positive integer (got '${MAX_GPUS}')" >&2
  exit 1
fi

# Apply prefix to stack name AFTER arg parsing
STACK_NAME="${STACK_NAME:-nvidia-artf-recommenders}"
if [[ -n "${STACK_PREFIX}" ]]; then
  STACK_NAME="${STACK_PREFIX}-${STACK_NAME}"
fi

log()  { printf '\033[0;32m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# =========================================================================
# Preflight
# =========================================================================
log "Preflight checks"
for bin in aws docker python3 jq eksctl kubectl; do
  command -v "${bin}" >/dev/null 2>&1 || fail "missing: ${bin}"
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "${ACCOUNT_ID}" ]] || fail "cannot resolve AWS account"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Deterministic UID for resource naming
STACK_UID="$(python3 -c "import hashlib; print(hashlib.sha256('${STACK_NAME}:${ACCOUNT_ID}:${AWS_REGION}'.encode()).hexdigest()[:8])")"

CLUSTER_NAME="${STACK_NAME}-triton"
MODEL_BUCKET="${STACK_NAME}-triton-models-${STACK_UID}"
LOADTEST_TABLE="${STACK_NAME}-loadtest-history"

log "Account=${ACCOUNT_ID}  Region=${AWS_REGION}  Stack=${STACK_NAME}  Tag=${IMAGE_TAG}"
log "EKS Cluster=${CLUSTER_NAME}  Model Bucket=${MODEL_BUCKET}"

# =========================================================================
# Destroy
# =========================================================================
if [[ "${DESTROY}" -eq 1 ]]; then
  warn "=== DESTROY ==="
  warn "This will delete the EKS cluster, Triton models, CloudFront,"
  warn "AgentCore runtime, Cognito, DynamoDB table, IAM policies/roles,"
  warn "and all Kubernetes resources."
  warn "ECR repos are RETAINED (delete manually if needed)."
  read -r -p "Type 'destroy' to confirm: " CONFIRM
  [[ "${CONFIRM}" == "destroy" ]] || fail "aborted"

  log "Deleting Kubernetes resources..."
  kubectl delete -f "${SCRIPT_DIR}/eks/" --ignore-not-found 2>/dev/null || true
  kubectl delete namespace artf --ignore-not-found 2>/dev/null || true

  log "Deleting EKS cluster ${CLUSTER_NAME}..."
  # Disable termination protection on eksctl-managed stacks first, in case
  # the cluster was deployed before Step 5.5 existed (eksctl enables it by
  # default). Requires cloudformation:UpdateTerminationProtection.
  for stack in $(aws cloudformation list-stacks \
      --region "${AWS_REGION}" \
      --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE UPDATE_ROLLBACK_COMPLETE \
      --query "StackSummaries[?starts_with(StackName, 'eksctl-${CLUSTER_NAME}-')].StackName" \
      --output text 2>/dev/null); do
    aws cloudformation update-termination-protection \
      --stack-name "${stack}" --no-enable-termination-protection \
      --region "${AWS_REGION}" >/dev/null 2>&1 \
      || warn "  Could not disable termination protection on ${stack} (need cloudformation:UpdateTerminationProtection)"
  done
  eksctl delete cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --wait 2>/dev/null || true

  log "Deleting Triton model bucket..."
  aws s3 rb "s3://${MODEL_BUCKET}" --force 2>/dev/null || true

  log "Deleting DynamoDB table ${LOADTEST_TABLE}..."
  aws dynamodb delete-table --table-name "${LOADTEST_TABLE}" --region "${AWS_REGION}" 2>/dev/null || true

  log "Deleting CloudFront + S3 frontend..."
  python3 "${SCRIPT_DIR}/scripts/deploy_frontend.py" --action destroy --stack-name "${STACK_NAME}" --region "${AWS_REGION}" || true

  log "Deleting AgentCore runtime..."
  AC_RUNTIME_NAME="$(echo "${STACK_NAME}_mcp" | tr '-' '_')"
  python3 "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" --action destroy --runtime-name "${AC_RUNTIME_NAME}" --region "${AWS_REGION}" || true

  log "Deleting Cognito User Pool..."
  python3 "${SCRIPT_DIR}/scripts/deploy_cognito.py" --action destroy --stack-name "${STACK_NAME}" --region "${AWS_REGION}" || true

  log "Deleting IAM policies..."
  TRITON_POLICY_NAME="${STACK_NAME}-triton-s3-policy-${STACK_UID}"
  TRITON_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${TRITON_POLICY_NAME}"
  DYNAMO_POLICY_NAME="${STACK_NAME}-dynamo-loadtest-${STACK_UID}"
  DYNAMO_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DYNAMO_POLICY_NAME}"
  EKS_SCALE_POLICY_NAME="${STACK_NAME}-eks-gpu-scale-${STACK_UID}"
  EKS_SCALE_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${EKS_SCALE_POLICY_NAME}"

  for POLICY_ARN in "${TRITON_POLICY_ARN}" "${DYNAMO_POLICY_ARN}" "${EKS_SCALE_POLICY_ARN}"; do
    # Detach from all entities before deletion
    for ENTITY in $(aws iam list-entities-for-policy --policy-arn "${POLICY_ARN}" --query 'PolicyRoles[].RoleName' --output text 2>/dev/null); do
      aws iam detach-role-policy --role-name "${ENTITY}" --policy-arn "${POLICY_ARN}" 2>/dev/null || true
    done
    # Delete non-default policy versions
    for VER in $(aws iam list-policy-versions --policy-arn "${POLICY_ARN}" --query 'Versions[?!IsDefaultVersion].VersionId' --output text 2>/dev/null); do
      aws iam delete-policy-version --policy-arn "${POLICY_ARN}" --version-id "${VER}" 2>/dev/null || true
    done
    aws iam delete-policy --policy-arn "${POLICY_ARN}" 2>/dev/null || true
  done

  log "Deleting AgentCore IAM role..."
  ROLE_NAME="${STACK_NAME}-agentcore-role-${STACK_UID}"
  for ATTACHED in $(aws iam list-attached-role-policies --role-name "${ROLE_NAME}" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do
    aws iam detach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${ATTACHED}" 2>/dev/null || true
  done
  aws iam delete-role --role-name "${ROLE_NAME}" 2>/dev/null || true

  log "Deleting IRSA service account IAM role..."
  eksctl delete iamserviceaccount --name triton-sa --namespace default --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" 2>/dev/null || true

  log "Destroy complete."
  log ""
  log "Resources RETAINED (manual cleanup if desired):"
  log "  ECR repos: ${STACK_NAME}-* (contain pushed images)"
  log "  Delete with: for r in \$(aws ecr describe-repositories --query 'repositories[?starts_with(repositoryName,\`${STACK_NAME}\`)].repositoryName' --output text --region ${AWS_REGION}); do aws ecr delete-repository --repository-name \$r --force --region ${AWS_REGION}; done"
  exit 0
fi

# =========================================================================
# UI-only deploy — re-upload both frontends to S3 + invalidate CloudFront
# =========================================================================
if [[ "${UI_ONLY}" -eq 1 ]]; then
  log "UI-only deploy"

  # Get the NLB endpoint from the EKS cluster
  aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}" 2>/dev/null || true
  NLB_DNS="$(kubectl get svc orchestrator -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo '')"
  if [[ -z "${NLB_DNS}" ]]; then
    warn "Could not read NLB endpoint from EKS. Using placeholder."
    NLB_DNS="localhost"
  fi

  # Primary distribution: React UI
  python3 "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
    --action deploy \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --orchestrator-url "http://${NLB_DNS}"

  PRIMARY_OUTPUTS="${SCRIPT_DIR}/.frontend-outputs.json"
  CF_DOMAIN="$(jq -r '.CloudFrontDomain // empty' "${PRIMARY_OUTPUTS}" 2>/dev/null || echo '')"

  log "React UI: https://${CF_DOMAIN:-'(pending)'}"
  exit 0
fi

# =========================================================================
# Step 1: ECR repositories
# =========================================================================
REPOS=(
  ${STACK_NAME}-dlrm-bid-shader
  ${STACK_NAME}-widedeep-segment-activator
  ${STACK_NAME}-ncf-deal-manager
  ${STACK_NAME}-metrics-enricher
  ${STACK_NAME}-orchestrator
  ${STACK_NAME}-agentcore
)

log "Step 1: Ensuring ECR repositories"
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

for repo in "${REPOS[@]}"; do
  aws ecr describe-repositories --repository-names "${repo}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${repo}" --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE >/dev/null
done
log "ECR repositories ready"

# =========================================================================
# Step 1.5: DynamoDB table for load test history
# =========================================================================
log "Step 1.5: Ensuring DynamoDB table for load test history"
if ! aws dynamodb describe-table --table-name "${LOADTEST_TABLE}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws dynamodb create-table \
    --table-name "${LOADTEST_TABLE}" \
    --attribute-definitions AttributeName=id,AttributeType=S \
    --key-schema AttributeName=id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "${AWS_REGION}" >/dev/null
  log "  Created DynamoDB table: ${LOADTEST_TABLE}"
else
  log "  DynamoDB table exists: ${LOADTEST_TABLE}"
fi

# =========================================================================
# Step 2: Export PyTorch models to ONNX
# =========================================================================
log "Step 2: Exporting PyTorch models to ONNX for Triton"
python3 "${SCRIPT_DIR}/../source/triton/export_models.py" \
  --output-dir "${SCRIPT_DIR}/../source/triton/model_repository"

if [[ "${EXPORT_ONLY}" -eq 1 ]]; then
  log "Export complete (--export-only). Models at triton/model_repository/"
  exit 0
fi

# =========================================================================
# Step 3: Upload model repository to S3
# =========================================================================
log "Step 3: Ensuring S3 model bucket ${MODEL_BUCKET}"
if ! aws s3api head-bucket --bucket "${MODEL_BUCKET}" 2>/dev/null; then
  aws s3 mb "s3://${MODEL_BUCKET}" --region "${AWS_REGION}"
fi
aws s3 sync "${SCRIPT_DIR}/../source/triton/model_repository/" \
  "s3://${MODEL_BUCKET}/triton-models/" \
  --delete --region "${AWS_REGION}"
log "  Models uploaded to s3://${MODEL_BUCKET}/triton-models/"

# =========================================================================
# Step 4: Build and push container images
# =========================================================================
if [[ "${SKIP_IMAGES}" -eq 0 ]]; then
  log "Step 4: Building and pushing container images"

  # Triton-backed ARTF containers (tritonclient, no PyTorch — lighter)
  TRITON_CONTAINERS=(
    "containers/dlrm_bid_shader:${STACK_NAME}-dlrm-bid-shader"
    "containers/widedeep_segment_activator:${STACK_NAME}-widedeep-segment-activator"
    "containers/ncf_deal_manager:${STACK_NAME}-ncf-deal-manager"
  )
  for entry in "${TRITON_CONTAINERS[@]}"; do
    CONTAINER_PATH="${entry%%:*}"
    REPO_NAME="${entry##*:}"
    IMAGE="${REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"
    log "  Building ${REPO_NAME} (amd64, tritonclient)"
    docker buildx build \
      --platform linux/amd64 \
      --build-arg CONTAINER="${CONTAINER_PATH}" \
      -f "${SCRIPT_DIR}/../source/triton/Dockerfile.triton-artf" \
      -t "${IMAGE}" --load "${SCRIPT_DIR}/../source"
    docker push "${IMAGE}"
  done

  # Metrics enricher (rule-based) + orchestrator (standard Dockerfile, no Triton needed)
  for entry in \
    "containers/metrics_enricher:${STACK_NAME}-metrics-enricher:metrics-enricher" \
    "orchestrator:${STACK_NAME}-orchestrator:orchestrator"; do
    CONTAINER_PATH="${entry%%:*}"
    REMAINDER="${entry#*:}"
    REPO_NAME="${REMAINDER%%:*}"
    AGENT_NAME="${REMAINDER##*:}"
    IMAGE="${REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"
    log "  Building ${REPO_NAME} (amd64)"
    docker buildx build \
      --platform linux/amd64 \
      --build-arg CONTAINER="${CONTAINER_PATH}" \
      --build-arg AGENT_NAME="${AGENT_NAME}" \
      -f "${SCRIPT_DIR}/../source/Dockerfile" \
      -t "${IMAGE}" --load "${SCRIPT_DIR}/../source"
    docker push "${IMAGE}"
  done

  # AgentCore container (ARM64)
  if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
    AC_IMAGE="${REGISTRY}/${STACK_NAME}-agentcore:${IMAGE_TAG}"
    log "  Building AgentCore image (arm64)"
    docker buildx build \
      --platform linux/arm64 \
      -f "${SCRIPT_DIR}/../source/Dockerfile.agentcore" \
      -t "${AC_IMAGE}" --load "${SCRIPT_DIR}/../source"
    docker push "${AC_IMAGE}"
  fi

  log "All images pushed"
else
  warn "Skipping image build (--skip-images)"
fi

# =========================================================================
# Step 5: Create or reuse EKS cluster
# =========================================================================
if [[ "${SKIP_CLUSTER}" -eq 0 ]]; then
  if eksctl get cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Step 5: EKS cluster ${CLUSTER_NAME} already exists"
  else
    log "Step 5: Creating EKS cluster ${CLUSTER_NAME} (15-20 min)"
    log "  GPU node group: desired 1, max ${MAX_GPUS} (set via --maxGPUs)"
    CLUSTER_CONFIG="/tmp/${CLUSTER_NAME}-config.yaml"
    sed -e "s/__STACK_NAME__/${STACK_NAME}/g" \
        -e "s/__REGION__/${AWS_REGION}/g" \
        -e "s/__MAX_GPUS__/${MAX_GPUS}/g" \
        "${SCRIPT_DIR}/eks/cluster-config.yaml" > "${CLUSTER_CONFIG}"
    eksctl create cluster -f "${CLUSTER_CONFIG}"
  fi
else
  warn "Skipping EKS cluster creation (--skip-cluster)"
fi

# =========================================================================
# Step 5.5: Disable termination protection on eksctl-managed stacks
# =========================================================================
# eksctl enables CloudFormation termination protection on the cluster,
# nodegroup, and addon stacks it creates, and the ClusterConfig schema
# (deployment/eks/cluster-config.yaml) exposes no option to opt out. We
# disable it here so a later `./deploy.sh --destroy` (or `eksctl delete
# cluster`) is not blocked at teardown. This requires the deploying
# principal to hold cloudformation:UpdateTerminationProtection; if that
# action is denied (e.g. by a restrictive session policy), teardown will
# need a session that allows it.
log "Step 5.5: Disabling termination protection on eksctl-managed stacks"
EKSCTL_STACKS="$(aws cloudformation list-stacks \
  --region "${AWS_REGION}" \
  --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE UPDATE_ROLLBACK_COMPLETE \
  --query "StackSummaries[?starts_with(StackName, 'eksctl-${CLUSTER_NAME}-')].StackName" \
  --output text 2>/dev/null || echo '')"
if [[ -z "${EKSCTL_STACKS}" ]]; then
  warn "  No eksctl-managed stacks found for ${CLUSTER_NAME} (skipping)"
else
  for stack in ${EKSCTL_STACKS}; do
    if aws cloudformation update-termination-protection \
         --stack-name "${stack}" \
         --no-enable-termination-protection \
         --region "${AWS_REGION}" >/dev/null 2>&1; then
      log "  Termination protection disabled: ${stack}"
    else
      warn "  Could not disable termination protection on ${stack}"
      warn "    (missing cloudformation:UpdateTerminationProtection? teardown will need a session that allows it)"
    fi
  done
fi

aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}"

# =========================================================================
# Step 6: Install NVIDIA Kubernetes Device Plugin + Prometheus Operator CRDs
# =========================================================================
log "Step 6: Ensuring NVIDIA Kubernetes Device Plugin"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml 2>/dev/null || true

log "  Ensuring Prometheus Operator CRDs (for Triton metrics)"
if ! kubectl get crd podmonitors.monitoring.coreos.com >/dev/null 2>&1; then
  kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.75.0/example/prometheus-operator-crd/monitoring.coreos.com_podmonitors.yaml 2>/dev/null || \
    warn "Could not install PodMonitor CRD"
fi
if ! kubectl get crd prometheusrules.monitoring.coreos.com >/dev/null 2>&1; then
  kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.75.0/example/prometheus-operator-crd/monitoring.coreos.com_prometheusrules.yaml 2>/dev/null || \
    warn "Could not install PrometheusRule CRD"
fi

# =========================================================================
# Step 7: IRSA — IAM role for Triton S3 model access
# =========================================================================
log "Step 7: Ensuring IRSA for Triton S3 access"
TRITON_POLICY_NAME="${STACK_NAME}-triton-s3-policy-${STACK_UID}"
TRITON_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${TRITON_POLICY_NAME}"

POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${MODEL_BUCKET}\",\"arn:aws:s3:::${MODEL_BUCKET}/*\"]}]}"
if aws iam get-policy --policy-arn "${TRITON_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${TRITON_POLICY_ARN}" \
    --policy-document "${POLICY_DOC}" \
    --set-as-default 2>/dev/null || true
else
  aws iam create-policy \
    --policy-name "${TRITON_POLICY_NAME}" \
    --policy-document "${POLICY_DOC}" >/dev/null
fi

eksctl create iamserviceaccount \
  --name triton-sa \
  --namespace default \
  --cluster "${CLUSTER_NAME}" \
  --region "${AWS_REGION}" \
  --attach-policy-arn "${TRITON_POLICY_ARN}" \
  --approve \
  --override-existing-serviceaccounts 2>/dev/null || true

TRITON_ROLE_ARN="$(kubectl get sa triton-sa -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo '')"
log "  Triton IRSA role: ${TRITON_ROLE_ARN}"

# =========================================================================
# Step 7.5: DynamoDB permissions for orchestrator pods
# =========================================================================
log "Step 7.5: Ensuring DynamoDB access for orchestrator"
DYNAMO_POLICY_NAME="${STACK_NAME}-dynamo-loadtest-${STACK_UID}"
DYNAMO_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DYNAMO_POLICY_NAME}"
DYNAMO_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"dynamodb:PutItem\",\"dynamodb:GetItem\",\"dynamodb:Query\",\"dynamodb:Scan\"],\"Resource\":\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${LOADTEST_TABLE}\"}]}"

if aws iam get-policy --policy-arn "${DYNAMO_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${DYNAMO_POLICY_ARN}" \
    --policy-document "${DYNAMO_POLICY_DOC}" \
    --set-as-default 2>/dev/null || true
else
  aws iam create-policy \
    --policy-name "${DYNAMO_POLICY_NAME}" \
    --policy-document "${DYNAMO_POLICY_DOC}" >/dev/null
fi

# Attach to the services node group role (orchestrator runs there)
SERVICES_NG_ROLE="$(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" --nodegroup-name "cpu-services" --region "${AWS_REGION}" --query 'nodegroup.nodeRole' --output text 2>/dev/null | awk -F/ '{print $NF}')"
if [[ -n "${SERVICES_NG_ROLE}" && "${SERVICES_NG_ROLE}" != "None" ]]; then
  aws iam attach-role-policy --role-name "${SERVICES_NG_ROLE}" --policy-arn "${DYNAMO_POLICY_ARN}" 2>/dev/null || true
  log "  Attached DynamoDB policy to node role: ${SERVICES_NG_ROLE}"
fi

# EKS nodegroup scaling policy — allows orchestrator to start/stop GPU nodes
log "Step 7.6: Ensuring EKS nodegroup scaling access for orchestrator"
EKS_SCALE_POLICY_NAME="${STACK_NAME}-eks-gpu-scale-${STACK_UID}"
EKS_SCALE_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${EKS_SCALE_POLICY_NAME}"
EKS_SCALE_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"eks:UpdateNodegroupConfig\",\"eks:DescribeNodegroup\"],\"Resource\":\"arn:aws:eks:${AWS_REGION}:${ACCOUNT_ID}:nodegroup/${CLUSTER_NAME}/*\"}]}"

if aws iam get-policy --policy-arn "${EKS_SCALE_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${EKS_SCALE_POLICY_ARN}" \
    --policy-document "${EKS_SCALE_POLICY_DOC}" \
    --set-as-default 2>/dev/null || true
else
  aws iam create-policy \
    --policy-name "${EKS_SCALE_POLICY_NAME}" \
    --policy-document "${EKS_SCALE_POLICY_DOC}" >/dev/null
fi

if [[ -n "${SERVICES_NG_ROLE}" && "${SERVICES_NG_ROLE}" != "None" ]]; then
  aws iam attach-role-policy --role-name "${SERVICES_NG_ROLE}" --policy-arn "${EKS_SCALE_POLICY_ARN}" 2>/dev/null || true
  log "  Attached EKS scaling policy to node role: ${SERVICES_NG_ROLE}"
fi

# =========================================================================
# Step 8: Apply Kubernetes manifests (idempotent — kubectl apply)
# =========================================================================
log "Step 8: Applying Kubernetes manifests"

# --- Provision Cognito BEFORE applying manifests so the orchestrator gets the real pool ID ---
log "  Provisioning Cognito User Pool (needed for orchestrator auth)..."
python3 "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
  --action deploy \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --cloudfront-domain "${CF_DOMAIN:-localhost}"

COGNITO_OUTPUTS="${SCRIPT_DIR}/.cognito-outputs.json"
COGNITO_USER_POOL_ID="$(jq -r '.UserPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
COGNITO_CLIENT_ID="$(jq -r '.ClientId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
log "  Cognito Pool: ${COGNITO_USER_POOL_ID}  Client: ${COGNITO_CLIENT_ID}"

if [[ -z "${COGNITO_USER_POOL_ID}" ]]; then
  warn "Cognito pool ID is empty — orchestrator auth will be DISABLED until patched!"
fi

# All workloads (Triton, agent containers, orchestrator) deploy into the
# `default` namespace, matching the triton-sa IRSA service account. Keeping
# them co-located lets the orchestrator reach backends by bare service name.

# Wait for GPU node to be available before applying Triton deployment
log "  Checking for GPU node availability..."
GPU_WAIT_TIMEOUT=300
GPU_WAIT_INTERVAL=15
GPU_ELAPSED=0
while true; do
  GPU_NODES="$(kubectl get nodes -l nvidia.com/gpu=present --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${GPU_NODES}" -gt 0 ]]; then
    log "  GPU node(s) available (${GPU_NODES} found)"
    break
  fi
  if [[ "${GPU_ELAPSED}" -ge "${GPU_WAIT_TIMEOUT}" ]]; then
    warn "No GPU nodes after ${GPU_WAIT_TIMEOUT}s — Triton may stay pending."
    warn "Check node group: eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${AWS_REGION}"
    break
  fi
  log "  Waiting for GPU node to scale up... (${GPU_ELAPSED}s / ${GPU_WAIT_TIMEOUT}s)"
  sleep "${GPU_WAIT_INTERVAL}"
  GPU_ELAPSED=$((GPU_ELAPSED + GPU_WAIT_INTERVAL))
done

for manifest in triton-deployment.yaml artf-containers-deployment.yaml orchestrator-deployment.yaml triton-hpa.yaml; do
  PROCESSED="/tmp/${CLUSTER_NAME}-${manifest}"
  sed -e "s|__STACK_NAME__|${STACK_NAME}|g" \
      -e "s|__REGION__|${AWS_REGION}|g" \
      -e "s|__REGISTRY__|${REGISTRY}|g" \
      -e "s|__IMAGE_TAG__|${IMAGE_TAG}|g" \
      -e "s|__MODEL_BUCKET__|${MODEL_BUCKET}|g" \
      -e "s|__TRITON_ROLE_ARN__|${TRITON_ROLE_ARN}|g" \
      -e "s|__COGNITO_USER_POOL_ID__|${COGNITO_USER_POOL_ID:-}|g" \
      "${SCRIPT_DIR}/eks/${manifest}" > "${PROCESSED}"
  kubectl apply -f "${PROCESSED}"
done

log "  Waiting for Triton Inference Server..."
kubectl rollout status deployment/triton-inference-server --timeout=300s || \
  warn "Triton not ready yet — check GPU node availability with: kubectl get nodes -l nvidia.com/gpu=present"

log "  Waiting for orchestrator..."
kubectl rollout status deployment/orchestrator --timeout=120s || true

# Wait for the LoadBalancer to get an external hostname
log "  Waiting for orchestrator LoadBalancer endpoint..."
LB_WAIT_TIMEOUT=120
LB_WAIT_INTERVAL=10
LB_ELAPSED=0
NLB_DNS=""
while [[ -z "${NLB_DNS}" || "${NLB_DNS}" == "pending" ]]; do
  NLB_DNS="$(kubectl get svc orchestrator -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo '')"
  if [[ -n "${NLB_DNS}" && "${NLB_DNS}" != "pending" ]]; then
    log "  LoadBalancer ready: ${NLB_DNS}"
    break
  fi
  if [[ "${LB_ELAPSED}" -ge "${LB_WAIT_TIMEOUT}" ]]; then
    warn "LoadBalancer not ready after ${LB_WAIT_TIMEOUT}s — frontend will use placeholder URL."
    NLB_DNS="localhost"
    break
  fi
  log "  Waiting for LoadBalancer... (${LB_ELAPSED}s / ${LB_WAIT_TIMEOUT}s)"
  sleep "${LB_WAIT_INTERVAL}"
  LB_ELAPSED=$((LB_ELAPSED + LB_WAIT_INTERVAL))
done

# =========================================================================
# Step 8.5: Schedule GPU node group shutdown at 8pm ET daily
# =========================================================================
log "Step 8.5: Configuring GPU scheduled shutdown (8pm ET daily)"

# Find the ASG backing the GPU node group
GPU_ASG_NAME="$(aws eks describe-nodegroup \
  --cluster-name "${CLUSTER_NAME}" \
  --nodegroup-name "gpu-inference" \
  --region "${AWS_REGION}" \
  --query 'nodegroup.resources.autoScalingGroups[0].name' \
  --output text 2>/dev/null || echo '')"

if [[ -n "${GPU_ASG_NAME}" && "${GPU_ASG_NAME}" != "None" ]]; then
  # Create/update scheduled action to scale GPU to 0 at 8pm ET (00:00 UTC next day for ET, but use America/New_York)
  aws autoscaling put-scheduled-update-group-action \
    --auto-scaling-group-name "${GPU_ASG_NAME}" \
    --scheduled-action-name "${STACK_NAME}-gpu-nightly-shutdown" \
    --recurrence "0 20 * * *" \
    --time-zone "America/New_York" \
    --desired-capacity 0 \
    --min-size 0 \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Note: Cron "0 20 * * *" = 8:00 PM in the specified timezone
  # The --time-zone flag handles DST automatically
  log "  GPU ASG: ${GPU_ASG_NAME}"
  log "  Scheduled shutdown: daily at 8:00 PM America/New_York"
  log "  Users can restart GPUs via the UI 'Start GPU' button"
else
  warn "Could not find GPU ASG — skipping scheduled shutdown setup"
fi

# =========================================================================
# Step 9: Deploy frontend to S3 + CloudFront (React UI)
# =========================================================================
log "Step 9: Deploying frontend"

# Cognito was already provisioned in Step 8 (before manifest apply).
# Re-read outputs in case they're needed for frontend build.
COGNITO_OUTPUTS="${SCRIPT_DIR}/.cognito-outputs.json"
COGNITO_USER_POOL_ID="$(jq -r '.UserPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
COGNITO_CLIENT_ID="$(jq -r '.ClientId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"

# --- Step 9c: Write Cognito config for React build ---
REACT_ENV="${SCRIPT_DIR}/../source/frontend-react/.env.production"
cat > "${REACT_ENV}" <<EOF
VITE_COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
VITE_COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
VITE_COGNITO_REGION=${AWS_REGION}
EOF

# React UI distribution
python3 "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
  --action deploy \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --orchestrator-url "http://${NLB_DNS}"

PRIMARY_OUTPUTS="${SCRIPT_DIR}/.frontend-outputs.json"
CF_DOMAIN="$(jq -r '.CloudFrontDomain // empty' "${PRIMARY_OUTPUTS}" 2>/dev/null || echo '')"

# Update Cognito callback URLs now that we know the CF domain
if [[ -n "${CF_DOMAIN}" && -n "${COGNITO_USER_POOL_ID}" ]]; then
  python3 "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
    --action deploy \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --cloudfront-domain "${CF_DOMAIN}"
fi

# --- Step 9d: Ensure a default admin user exists ---
# The pool uses email as the username (UsernameAttributes=["email"] in
# deploy_cognito.py), so the admin login is an email address whose local-part
# is "admin". Override with DEMO_USER_EMAIL if you want a different login.
DEMO_USER_EMAIL="${DEMO_USER_EMAIL:-admin@example.com}"
DEMO_LOGIN_STATUS="no-auth"
DEMO_USER_TEMP_PASSWORD=""
if [[ -n "${COGNITO_USER_POOL_ID}" ]]; then
  if aws cognito-idp admin-get-user --user-pool-id "${COGNITO_USER_POOL_ID}" --username "${DEMO_USER_EMAIL}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    DEMO_LOGIN_STATUS="existing"
    log "  Admin user already exists: ${DEMO_USER_EMAIL}"
  else
    # AdminCreateUser never returns a Cognito-generated password (it is only
    # delivered via the suppressed invitation message), so generate a strong,
    # policy-compliant temporary password locally and surface it in the final
    # summary. The operator may instead supply one via DEMO_USER_PASSWORD.
    TEMP_PASS="${DEMO_USER_PASSWORD:-}"
    if [[ -z "${TEMP_PASS}" ]]; then
      TEMP_PASS="$(python3 - <<'PY'
import secrets
import string

alphabet = string.ascii_letters + string.digits
while True:
    candidate = "".join(secrets.choice(alphabet) for _ in range(16))
    if (any(c.islower() for c in candidate)
            and any(c.isupper() for c in candidate)
            and any(c.isdigit() for c in candidate)):
        break
print(candidate)
PY
)"
    fi
    aws cognito-idp admin-create-user \
      --user-pool-id "${COGNITO_USER_POOL_ID}" \
      --username "${DEMO_USER_EMAIL}" \
      --temporary-password "${TEMP_PASS}" \
      --user-attributes Name=email,Value="${DEMO_USER_EMAIL}" Name=email_verified,Value=true \
      --message-action SUPPRESS \
      --region "${AWS_REGION}" >/dev/null
    DEMO_LOGIN_STATUS="created"
    DEMO_USER_TEMP_PASSWORD="${TEMP_PASS}"
    log "  Created admin user: ${DEMO_USER_EMAIL} (credentials shown in summary below)"
  fi
fi

# =========================================================================
# Step 10: Deploy AgentCore MCP runtime
# =========================================================================
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
  log "Step 10: Deploying AgentCore MCP runtime"

  ROLE_NAME="${STACK_NAME}-agentcore-role-${STACK_UID}"
  ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
  if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
    log "  Creating IAM role ${ROLE_NAME}"
    aws iam create-role --role-name "${ROLE_NAME}" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
      --description "Accelerator-optimized Agentic Bidding AgentCore execution role" >/dev/null
    aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
    aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
    aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess"
    sleep 10
  fi

  AC_RUNTIME_NAME="$(echo "${STACK_NAME}_mcp" | tr '-' '_')"
  AC_IMAGE="${REGISTRY}/${STACK_NAME}-agentcore:${IMAGE_TAG}"
  python3 "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
    --action deploy \
    --runtime-name "${AC_RUNTIME_NAME}" \
    --role-arn "${ROLE_ARN}" \
    --container-uri "${AC_IMAGE}" \
    --region "${AWS_REGION}"
else
  warn "Skipping AgentCore deployment (--skip-agentcore)"
fi

# =========================================================================
# Summary
# =========================================================================
log ""
log "========================================================="
log "  Accelerator-optimized Agentic Bidding — Deployed (EKS + Triton)"
log "========================================================="
log ""
log "  Frontend:    https://${CF_DOMAIN:-'(pending)'}"
log "  Endpoint:    http://${NLB_DNS:-'(pending)'}/v1/mutations"
log "  Health:      http://${NLB_DNS:-'(pending)'}/health/ready"
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
log "  AgentCore:   See runtime ARN above"
fi
log ""
log "  Demo Login (Cognito):"
case "${DEMO_LOGIN_STATUS:-no-auth}" in
  created)
    log "    Username:  ${DEMO_USER_EMAIL}"
    log "    Password:  ${DEMO_USER_TEMP_PASSWORD}"
    log "    Note:      Temporary password, shown only here — you'll set a permanent one on first login."
    ;;
  existing)
    log "    Username:  ${DEMO_USER_EMAIL}"
    log "    Password:  (existing user — password unchanged, not displayed)"
    log "    Reset:     aws cognito-idp admin-set-user-password \\"
    log "                 --user-pool-id ${COGNITO_USER_POOL_ID} --username ${DEMO_USER_EMAIL} \\"
    log "                 --password '<new-password>' --permanent --region ${AWS_REGION}"
    ;;
  *)
    log "    (Cognito auth not configured — orchestrator auth is disabled)"
    ;;
esac
log ""
log "  NVIDIA Triton Inference Server:"
log "    EKS Cluster:   ${CLUSTER_NAME}"
log "    Triton Image:  nvcr.io/nvidia/tritonserver:24.08-py3"
log "    GPU:           NVIDIA A10G (g5.xlarge)"
log "    Models:        s3://${MODEL_BUCKET}/triton-models/"
log "    Backend:       ONNX Runtime + CUDA Execution Provider"
log ""
log "  Models served by Triton:"
log "    dlrm_bid_shader            — DLRM (NVIDIA DeepLearningExamples)"
log "    widedeep_segment_activator — Wide & Deep (NVIDIA Merlin)"
log "    ncf_deal_manager           — NeuMF (NVIDIA DeepLearningExamples)"
log ""
log "  Containers (via orchestrator):"
log "    DLRM Bid Shader        — BID_SHADE"
log "    Wide&Deep Segments      — ACTIVATE_SEGMENTS"
log "    NCF Deal Manager        — ACTIVATE_DEALS / SUPPRESS_DEALS"
log "    Metrics Enricher        — ADD_METRICS"
log ""
log "  Monitoring:"
log "    kubectl port-forward svc/triton-inference-server 8002:8002"
log "    curl localhost:8002/metrics  # Prometheus metrics"
log ""
