#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy the Accelerator-optimized Agentic Bidding solution to AWS
#
# Deploys in 5 phases:
#   Phase 1/5 — Preparing models:      ECR repos, ONNX export, S3 upload
#   Phase 2/5 — Building & provisioning: container images (CodeBuild/local) +
#                                       EKS cluster (GPU + CPU node groups),
#                                       built concurrently; NVIDIA device
#                                       plugin; IRSA/IAM
#   Phase 3/5 — Deploying workloads:   Kubernetes manifests (Triton, ARTF
#                                       containers, orchestrator); TensorRT
#                                       base-engine bootstrap (non-blocking —
#                                       see "Model-optimizer bootstrap" below)
#   Phase 4/5 — Setting up access:     frontend (S3 + CloudFront), Cognito
#   Phase 5/5 — Registering agents:    Bedrock AgentCore MCP runtime, and
#                                       (default) the Part 2 closed-loop stack
#
# Default output shows only phase headers + one-line checkmarked summaries.
# Pass --verbose for the full detailed log stream. On failure, the phase and
# step that failed are reported with the real error plus a remediation hint.
#
# Model-optimizer bootstrap is fire-and-forget: it runs in the background and
# does not block Phase 3+. Triton auto-loads the compiled TensorRT engines
# from its S3 model repository (poll mode) whenever the bootstrap finishes —
# check status any time with: cat deployment/.bootstrap-status.json
#
# BREAKING CHANGE: --start-at now takes a phase number 1-5 instead of the
# old ad-hoc step numbers. See RENAME_MAP.md for the old-step -> new-phase
# mapping if you have scripts referencing the previous numbering.
#
# Usage:
#   ./deploy.sh                                # full deploy
#   ./deploy.sh --prefix v1                    # resources named v1-nvidia-artf-*
#   ./deploy.sh --prefix prod --skip-agentcore # combine flags
#   ./deploy.sh                                # FULL stack incl. Part 2 closed-loop (default)
#   ./deploy.sh --no-retraining                # Part 1 only (skip NeMo-RL, Model Registry, Glue ETL, agents)
#   ./deploy.sh --model-id global.anthropic.claude-opus-4-8  # override the agents' Bedrock model
#   ./deploy.sh --local-build                  # build images locally with Docker (requires ~30 GB free disk)
#   ./deploy.sh --ngc-key KEY                  # NGC key stored automatically (for NeMo builds)
#   ./deploy.sh --ui-only                      # redeploy frontend only (fast)
#   ./deploy.sh --skip-cluster                 # reuse existing EKS cluster
#   ./deploy.sh --export-only                  # just export ONNX models, no deploy
#   ./deploy.sh --maxGPUs 5                     # cap GPU node group max size at 5 (default 3)
#   ./deploy.sh --artf-node-role inference      # co-locate ARTF model containers on the GPU node
#                                               # (default: services / CPU nodes; --artf-on-gpu = shorthand)
#   ./deploy.sh --start-at 3                   # resume from Phase 3 (skip model prep + images/cluster)
#   ./deploy.sh --verbose                      # print full detailed logs alongside phase output
#   ./deploy.sh --destroy                      # tear down the entire stack
#   AWS_REGION=us-west-2 ./deploy.sh           # different region
#
# Prerequisites:
#   - AWS CLI v2 with credentials
#   - Python 3.11+ with boto3, torch, onnx, onnxscript
#   - jq, eksctl, kubectl
#   - Docker with buildx (only if using --local-build)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Job-oriented display names for the 4 ARTF containers, used ONLY for the
# ECR repository name / image tag and any printed summary text. The
# underlying "key" (dlrm-bid-shader, widedeep-segment-activator, etc.) is
# unchanged everywhere else (source directory resolution, content hashing,
# Dockerfile selection, and the --only key sent to remote_build.sh/
# buildspec.yml) — see RENAME_MAP.md for the full rationale.
display_name() {
  case "$1" in
    dlrm-bid-shader)             echo "bid-pricer" ;;
    widedeep-segment-activator)  echo "audience-activator" ;;
    ncf-deal-manager)            echo "deal-scorer" ;;
    metrics-enricher)            echo "signals-enricher" ;;
    deal-yield-manager)          echo "yield-optimizer" ;;
    *)                           echo "$1" ;;
  esac
}

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
# --verbose: print the full detailed log stream (every log/warn line) in
# addition to the default phase headers + checkmarked summaries. Default (0)
# shows only the phase-level output.
VERBOSE=0
# Part 2 closed-loop (NeMo-RL training, Model Registry, Glue ETL, the Adaptive
# Bidding + Governance agents) is ON by default. Disable with --no-retraining.
WITH_RETRAINING=1
LOCAL_BUILD=0
NGC_SECRET="${NGC_SECRET:-}"
NGC_KEY="${NGC_KEY:-}"
# Bedrock model id for the Part 2 reasoning agents (Adaptive Bidding + Governance).
# Model access is automatic, so this defaults to the Claude Opus 4.8 GLOBAL
# cross-region inference profile (set below). Override with BEDROCK_MODEL_ID /
# --model-id, or per-agent with ADAPTIVE_BIDDING_MODEL_ID / GOVERNANCE_MODEL_ID.
# Forwarded to deploy_closed_loop.sh in Step 11.
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-}"
START_AT=1
MAX_GPUS="${MAX_GPUS:-3}"
# Node group the three NVIDIA ARTF model containers schedule onto. They call Triton
# over the network and hold no GPU, so they default to the CPU node group
# (role=services) — keeping load-test scale-out off the scarce GPU nodes. Use
# --artf-node-role=inference (or --artf-on-gpu) to co-locate them on the GPU node.
ARTF_NODE_ROLE="${ARTF_NODE_ROLE:-services}"
STACK_PREFIX="${STACK_PREFIX:-}"
for arg in "$@"; do
  case "${arg}" in
    --destroy)          DESTROY=1 ;;
    --skip-agentcore)   SKIP_AGENTCORE=1 ;;
    --skip-images)      SKIP_IMAGES=1 ;;
    --verbose)          VERBOSE=1 ;;
    --start-at=*)       START_AT="${arg#--start-at=}" ;;
    --ui-only)          UI_ONLY=1 ;;
    --skip-cluster)     SKIP_CLUSTER=1 ;;
    --export-only)      EXPORT_ONLY=1 ;;
    --with-retraining)  WITH_RETRAINING=1 ;;   # default; kept for back-compat
    --no-retraining|--skip-retraining) WITH_RETRAINING=0 ;;
    --remote-build)     LOCAL_BUILD=0 ;;
    --local-build)      LOCAL_BUILD=1 ;;
    --ngc-secret=*)     NGC_SECRET="${arg#--ngc-secret=}" ;;
    --ngc-secret)       ;; # value comes in next arg, handled below
    --ngc-key=*)        NGC_KEY="${arg#--ngc-key=}" ;;
    --ngc-key)          ;; # value comes in next arg, handled below
    --prefix=*)         STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)           ;; # value comes in next arg, handled below
    --maxGPUs=*)        MAX_GPUS="${arg#--maxGPUs=}" ;;
    --maxGPUs)          ;; # value comes in next arg, handled below
    --model-id=*)       BEDROCK_MODEL_ID="${arg#--model-id=}" ;;
    --model-id)         ;; # value comes in next arg, handled below
    --artf-node-role=*) ARTF_NODE_ROLE="${arg#--artf-node-role=}" ;;
    --artf-node-role)   ;; # value comes in next arg, handled below
    --artf-on-gpu)      ARTF_NODE_ROLE="inference" ;;
    --start-at)         ;; # value comes in next arg, handled below
    -h|--help)          sed -n '2,57p' "$0"; exit 0 ;;
    *)
      if [[ "${_PREV_ARG:-}" == "--prefix" ]]; then
        STACK_PREFIX="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--maxGPUs" ]]; then
        MAX_GPUS="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--start-at" ]]; then
        START_AT="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--model-id" ]]; then
        BEDROCK_MODEL_ID="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--artf-node-role" ]]; then
        ARTF_NODE_ROLE="${arg}"
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

# --start-at: skip earlier PHASES by setting SKIP flags. BREAKING CHANGE —
# this now takes a phase number 1-5 (see the header comment), not the old
# ad-hoc step numbers. Phase 1 = models/ECR, Phase 2 = images + cluster +
# NVIDIA device plugin + IAM, Phase 3 = manifests, Phase 4 = frontend,
# Phase 5 = agentcore + closed-loop.
if ! [[ "${START_AT}" =~ ^[1-5]$ ]]; then
  printf '\033[0;31m[fail]\033[0m %s\n' "--start-at must be 1-5 (got '${START_AT}'). See RENAME_MAP.md for the old-step -> new-phase mapping." >&2
  exit 1
fi
# Phase 2 = "images + cluster + NVIDIA device plugin + IAM" per the header
# comment above, and the Phase 2 code block itself is gated on
# START_AT -le 2 (see "phase 2" below). SKIP_IMAGES/SKIP_CLUSTER must only
# take effect once Phase 2 is actually being skipped (START_AT -gt 2) —
# a -gt 1 threshold on SKIP_IMAGES was off by one phase, causing
# --start-at 2 to skip image building even though Phase 2 is exactly the
# phase that's supposed to build images (confirmed live: a resumed deploy
# targeting Phase 2 printed "Skipping image build (--skip-images)" despite
# the user never passing --skip-images).
if [[ "${START_AT}" -gt 2 ]]; then SKIP_IMAGES=1; fi
if [[ "${START_AT}" -gt 2 ]]; then SKIP_CLUSTER=1; fi

# Validate --maxGPUs: must be a positive integer (it caps the GPU node group's maxSize)
if ! [[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  printf '\033[0;31m[fail]\033[0m %s\n' "--maxGPUs must be a positive integer (got '${MAX_GPUS}')" >&2
  exit 1
fi

# Validate --artf-node-role: 'services' (CPU nodes, default) or 'inference' (GPU node)
if [[ "${ARTF_NODE_ROLE}" != "services" && "${ARTF_NODE_ROLE}" != "inference" ]]; then
  printf '\033[0;31m[fail]\033[0m %s\n' "--artf-node-role must be 'services' or 'inference' (got '${ARTF_NODE_ROLE}')" >&2
  exit 1
fi

# Apply prefix to stack name AFTER arg parsing
STACK_NAME="${STACK_NAME:-nvidia-artf-recommenders}"
if [[ -n "${STACK_PREFIX}" ]]; then
  STACK_NAME="${STACK_PREFIX}-${STACK_NAME}"
fi

# Resolve the Bedrock model id for the Part 2 reasoning agents. Model access is
# automatic, so default to the Claude Opus 4.8 GLOBAL cross-region inference profile,
# which routes to all commercial regions and works from any source region.
# Overridable via BEDROCK_MODEL_ID / --model-id.
if [[ -z "${BEDROCK_MODEL_ID}" ]]; then
  BEDROCK_MODEL_ID="global.anthropic.claude-opus-4-8"
fi
# Per-agent overrides fall back to the shared model id (mirrors deploy_closed_loop.sh).
ADAPTIVE_BIDDING_MODEL_ID="${ADAPTIVE_BIDDING_MODEL_ID:-${BEDROCK_MODEL_ID}}"
GOVERNANCE_MODEL_ID="${GOVERNANCE_MODEL_ID:-${BEDROCK_MODEL_ID}}"

# say(): always-visible output, regardless of --verbose. Used for the
# --destroy narration (FR-12: its UX stays exactly as before) and the final
# deployment summary (FR-6: "print ONLY what the user needs to get started").
say() { printf '%s\n' "$*"; }

# log(): detailed step-by-step narration. Printed only under --verbose so the
# default output stays to the 5 phase headers + checkmarked summaries below
# (FR-10). warn()/fail() always print — a warning or a hard failure is never
# hidden regardless of verbosity.
log()  { [[ "${VERBOSE}" -eq 1 ]] && printf '\033[0;32m[deploy]\033[0m %s\n' "$*"; return 0; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n' "$*"; }

# phase(): always-visible section header. N is 1-5 (see the phase table in
# design.md section 2 / RENAME_MAP.md's breaking-change note).
phase() { printf '\n\033[1;36mPhase %s/5: %s\033[0m\n' "$1" "$2"; }

# ok(): always-visible one-line checkmarked completion summary, printed after
# the real underlying command has actually succeeded (never before — no
# fabricated "done" markers).
ok() { printf '  \033[0;32m[OK]\033[0m %s\n' "$*"; }

# phase_hint(): a short remediation hint per phase, printed by fail() as
# additional context above the real captured error — never a replacement
# for it (FR-11). A case statement (not an associative array) for bash 3.2
# compatibility (macOS default) — see _source_hash()/display_name() above
# for the same pattern already used elsewhere in this script.
phase_hint() {
  case "$1" in
    1) echo "Check AWS credentials (aws sts get-caller-identity) and that boto3/torch/onnx/onnxscript are installed." ;;
    2) echo "Check NVIDIA NGC credentials (--ngc-key/--ngc-secret) for CodeBuild, and your EC2 g5 GPU service quota for cluster creation." ;;
    3) echo "Check GPU node availability: kubectl get nodes -l nvidia.com/gpu=present ; kubectl get pods" ;;
    4) echo "Check the CloudFront/S3 frontend deploy output above and Cognito user pool creation." ;;
    5) echo "Check Bedrock model access (--model-id) and the AgentCore runtime IAM roles." ;;
    *) echo "" ;;
  esac
}

# fail(): always prints the real error, then exits non-zero. When called with
# a phase number as the first argument (fail "<phase>" "<message>"), also
# prints that phase's remediation hint as additional context above it.
fail() {
  if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then
    local _phase="$1"; shift
    printf '\033[0;31m[fail]\033[0m Phase %s/5: %s\n' "${_phase}" "$*" >&2
    local _hint
    _hint="$(phase_hint "${_phase}")"
    if [[ -n "${_hint}" ]]; then
      printf '\033[0;33m[hint]\033[0m %s\n' "${_hint}" >&2
    fi
  else
    printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2
  fi
  exit 1
}

# =========================================================================
# Preflight
# =========================================================================
log "Preflight checks"
for bin in aws jq eksctl kubectl; do
  command -v "${bin}" >/dev/null 2>&1 || fail "missing: ${bin}"
done
if [[ "${LOCAL_BUILD}" -eq 1 ]]; then
  command -v docker >/dev/null 2>&1 || fail "missing: docker (required for --local-build)"
fi

# Resolve Python 3 binary (prefer python3, fall back to python if it's 3.x)
if command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1 && python -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
  PYTHON="python"
else
  fail "missing: python3 (>= 3.11)"
fi
log "Using Python: $(${PYTHON} --version) ($(command -v ${PYTHON}))"

# Ensure required Python packages are available (install if missing)
REQUIRED_PY_PACKAGES="torch onnx onnxscript boto3"
MISSING_PY_PACKAGES=""
for pkg in ${REQUIRED_PY_PACKAGES}; do
  if ! ${PYTHON} -c "import ${pkg}" 2>/dev/null; then
    MISSING_PY_PACKAGES="${MISSING_PY_PACKAGES} ${pkg}"
  fi
done
if [[ -n "${MISSING_PY_PACKAGES}" ]]; then
  log "Installing missing Python packages:${MISSING_PY_PACKAGES}"
  ${PYTHON} -m pip install --quiet ${MISSING_PY_PACKAGES} || fail "pip install failed for:${MISSING_PY_PACKAGES}"
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "${ACCOUNT_ID}" ]] || fail "cannot resolve AWS account"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Deterministic UID for resource naming
STACK_UID="$(${PYTHON} -c "import hashlib; print(hashlib.sha256('${STACK_NAME}:${ACCOUNT_ID}:${AWS_REGION}'.encode()).hexdigest()[:8])")"

CLUSTER_NAME="${STACK_NAME}-triton"
MODEL_BUCKET="${STACK_NAME}-triton-models-${STACK_UID}"
# Deterministic names matching deploy_closed_loop.sh's own conventions
# (TRAINING_DATA_BUCKET mirrors its glue_etl_cfn.yaml TrainingDataBucketName
# naming; these only resolve to real resources if --with-retraining was
# used, but the orchestrator's governance_api.py already reports a clear
# 503 rather than fabricating success if the underlying bucket/role
# doesn't exist).
TRAINING_DATA_BUCKET="${STACK_PREFIX:+${STACK_PREFIX}-}training-data-${ACCOUNT_ID}-${AWS_REGION}"
LOADTEST_TABLE="${STACK_NAME}-loadtest-history"
# Deterministic name matching feedback_pipeline_cfn.yaml's BidOutcomeStream
# naming (HasStackPrefix condition). Only resolves to a real stream once
# deploy_closed_loop.sh Step 1 creates it (--with-retraining, default on) —
# set unconditionally here so the orchestrator picks it up once it exists,
# without a separate redeploy/restart step.
FEEDBACK_STREAM_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}bid-outcome-stream"
# Same pattern, for DealYieldOutcomeStream (deal floor/margin outcomes —
# see source/orchestrator/deal_yield_feedback.py).
DEAL_YIELD_FEEDBACK_STREAM_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}deal-yield-outcome-stream"

log "Account=${ACCOUNT_ID}  Region=${AWS_REGION}  Stack=${STACK_NAME}  Tag=${IMAGE_TAG}"
log "EKS Cluster=${CLUSTER_NAME}  Model Bucket=${MODEL_BUCKET}"

# =========================================================================
# Destroy
# =========================================================================
if [[ "${DESTROY}" -eq 1 ]]; then
  warn "=== DESTROY ==="
  warn "This will delete EVERYTHING deploy.sh + deploy_closed_loop.sh create for"
  warn "this stack (prefix: ${STACK_PREFIX:-<none>}): EKS cluster, Triton models,"
  warn "CloudFront, AgentCore runtimes, Cognito, DynamoDB tables, S3 buckets"
  warn "(including the ones marked DeletionPolicy: Retain in the CFN templates),"
  warn "SageMaker Model Registry, IAM policies/roles, ECR repos, and all Kubernetes"
  warn "resources. NOTHING is retained — this is not reversible."
  read -r -p "Type 'destroy' to confirm: " CONFIRM
  [[ "${CONFIRM}" == "destroy" ]] || fail "aborted"

  # AgentCore runtime names only allow [a-zA-Z0-9_] (no hyphens) and must start
  # with a letter, so a hyphenated STACK_PREFIX is translated to underscores —
  # same convention deploy_closed_loop.sh uses when creating these runtimes.
  RUNTIME_NAME_PREFIX=""
  [[ -n "${STACK_PREFIX}" ]] && RUNTIME_NAME_PREFIX="$(echo "${STACK_PREFIX}_" | tr '-' '_')"

  # Part 2 closed-loop stack names (mirrors deploy_closed_loop.sh's naming).
  VPC_PROXY_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}vpc-proxy"
  GOVERNANCE_EVENTBRIDGE_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}governance-eventbridge"
  CLOSED_LOOP_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}closed-loop-core"
  AGENTCORE_SECURITY_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}agentcore-security"
  GLUE_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}glue-etl"
  FEEDBACK_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}feedback-pipeline"
  CODEBUILD_STACK="${STACK_NAME}-codebuild"

  # FR-12: --destroy's UX stays exactly as it was — its progress narration
  # uses say() (always-visible), not log() (verbose-gated), so teardown
  # doesn't go silent by default and look hung.
  #
  # CRITICAL: kubectl has no implicit cluster scoping — it always operates on
  # whatever context is "current" in the shared ~/.kube/config. Running this
  # script concurrently with ANY other deploy.sh/kubectl invocation (including
  # against a completely different stack) races on that shared file. If another
  # process's `aws eks update-kubeconfig` wins the race, these deletes silently
  # wipe THAT cluster's live workloads instead of this one's — confirmed live:
  # a concurrent deploy left another environment's Triton/orchestrator/ARTF pods
  # deleted while its CloudFormation stacks stayed intact. Always pin --context
  # explicitly derived from THIS run's CLUSTER_NAME; never rely on ambient state.
  KUBE_CONTEXT="arn:aws:eks:${AWS_REGION}:${ACCOUNT_ID}:cluster/${CLUSTER_NAME}"
  if kubectl config get-contexts "${KUBE_CONTEXT}" >/dev/null 2>&1; then
    say "Deleting Kubernetes resources (context: ${KUBE_CONTEXT})..."
    kubectl --context "${KUBE_CONTEXT}" delete -f "${SCRIPT_DIR}/eks/" --ignore-not-found 2>/dev/null || true
    kubectl --context "${KUBE_CONTEXT}" delete namespace artf --ignore-not-found 2>/dev/null || true
  else
    warn "  No kubeconfig context for ${CLUSTER_NAME} — skipping in-cluster resource deletion (cluster likely already gone or never had kubeconfig fetched; eksctl will still tear down the cluster itself below)."
  fi

  # --- VPC proxy stack FIRST: its Lambda's ENIs live in the EKS cluster's
  # private subnets. If the cluster is deleted first, those ENIs are left
  # behind and block subnet deletion (observed live: eksctl-*-cluster stack
  # gets stuck DELETE_FAILED on "subnet has dependencies and cannot be
  # deleted"). Wait for completion so the ENIs are gone before eksctl runs.
  if aws cloudformation describe-stacks --stack-name "${VPC_PROXY_STACK}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    say "Deleting VPC proxy Lambda stack (${VPC_PROXY_STACK}) before the EKS cluster..."
    aws cloudformation delete-stack --stack-name "${VPC_PROXY_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
    aws cloudformation wait stack-delete-complete --stack-name "${VPC_PROXY_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  fi

  say "Deleting EKS cluster ${CLUSTER_NAME}..."
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
  # IRSA service accounts before the cluster is gone (eksctl needs the cluster/OIDC
  # provider to resolve the CFN stack it created for each). Both triton-sa (Step 7)
  # and model-optimizer-sa (Step 7.1) are created during deploy; deleting only
  # triton-sa here was a gap.
  eksctl delete iamserviceaccount --name triton-sa --namespace default --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" 2>/dev/null || true
  eksctl delete iamserviceaccount --name model-optimizer-sa --namespace default --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" 2>/dev/null || true
  eksctl delete cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --wait 2>/dev/null || true

  say "Deleting Triton model bucket..."
  aws s3 rb "s3://${MODEL_BUCKET}" --force 2>/dev/null || true

  say "Deleting DynamoDB table ${LOADTEST_TABLE}..."
  aws dynamodb delete-table --table-name "${LOADTEST_TABLE}" --region "${AWS_REGION}" 2>/dev/null || true

  say "Deleting CloudFront + S3 frontend..."
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_frontend.py" --action destroy --stack-name "${STACK_NAME}" --region "${AWS_REGION}" || true
  # deploy_frontend.py's destroy only disables the CloudFront distribution (AWS
  # requires Enabled=false to propagate before a distribution can be deleted) —
  # it does not delete the distribution or its S3 bucket. Nothing is retained
  # here, so remove both once the disable has propagated.
  FRONTEND_UID="$(${PYTHON} -c "import hashlib; print(hashlib.sha256('${STACK_NAME}:${ACCOUNT_ID}:${AWS_REGION}'.encode()).hexdigest()[:8])")"
  FRONTEND_BUCKET="${STACK_NAME}-frontend-${FRONTEND_UID}"
  say "  Emptying and deleting frontend bucket ${FRONTEND_BUCKET}..."
  aws s3 rb "s3://${FRONTEND_BUCKET}" --force 2>/dev/null || true
  FRONTEND_DIST_ID="$(aws cloudfront list-distributions --query "DistributionList.Items[?Comment=='${STACK_NAME}'].Id | [0]" --output text 2>/dev/null || echo '')"
  if [[ -n "${FRONTEND_DIST_ID}" && "${FRONTEND_DIST_ID}" != "None" ]]; then
    say "  Waiting for CloudFront distribution ${FRONTEND_DIST_ID} to finish disabling..."
    aws cloudfront wait distribution-deployed --id "${FRONTEND_DIST_ID}" 2>/dev/null || true
    FRONTEND_ETAG="$(aws cloudfront get-distribution-config --id "${FRONTEND_DIST_ID}" --query 'ETag' --output text 2>/dev/null || echo '')"
    if [[ -n "${FRONTEND_ETAG}" ]]; then
      aws cloudfront delete-distribution --id "${FRONTEND_DIST_ID}" --if-match "${FRONTEND_ETAG}" 2>/dev/null \
        || warn "  Could not delete CloudFront distribution ${FRONTEND_DIST_ID} yet (may still be propagating disable) — retry: aws cloudfront delete-distribution --id ${FRONTEND_DIST_ID} --if-match \$(aws cloudfront get-distribution-config --id ${FRONTEND_DIST_ID} --query ETag --output text)"
    fi
  fi

  say "Deleting AgentCore MCP runtime..."
  AC_RUNTIME_NAME="$(echo "${STACK_NAME}_mcp" | tr '-' '_')"
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" --action destroy --runtime-name "${AC_RUNTIME_NAME}" --region "${AWS_REGION}" || true

  say "Deleting Cognito User Pool..."
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_cognito.py" --action destroy --stack-name "${STACK_NAME}" --region "${AWS_REGION}" || true

  say "Deleting IAM policies..."
  TRITON_POLICY_NAME="${STACK_NAME}-triton-s3-policy-${STACK_UID}"
  TRITON_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${TRITON_POLICY_NAME}"
  OPTIMIZER_POLICY_NAME="${STACK_NAME}-model-optimizer-s3-${STACK_UID}"
  OPTIMIZER_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${OPTIMIZER_POLICY_NAME}"
  DYNAMO_POLICY_NAME="${STACK_NAME}-dynamo-loadtest-${STACK_UID}"
  DYNAMO_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DYNAMO_POLICY_NAME}"
  EKS_SCALE_POLICY_NAME="${STACK_NAME}-eks-gpu-scale-${STACK_UID}"
  EKS_SCALE_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${EKS_SCALE_POLICY_NAME}"
  CLOSED_LOOP_POLICY_NAME="${STACK_NAME}-closed-loop-${STACK_UID}"
  CLOSED_LOOP_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${CLOSED_LOOP_POLICY_NAME}"

  # model-optimizer-s3 (Step 7.1's IRSA policy) was missing from this cleanup —
  # every other IRSA/orchestrator policy Step 7/7.5/7.6/9 creates was already here.
  for POLICY_ARN in "${TRITON_POLICY_ARN}" "${OPTIMIZER_POLICY_ARN}" "${DYNAMO_POLICY_ARN}" "${EKS_SCALE_POLICY_ARN}" "${CLOSED_LOOP_POLICY_ARN}"; do
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

  say "Deleting AgentCore IAM role..."
  ROLE_NAME="${STACK_NAME}-agentcore-role-${STACK_UID}"
  for ATTACHED in $(aws iam list-attached-role-policies --role-name "${ROLE_NAME}" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do
    aws iam detach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${ATTACHED}" 2>/dev/null || true
  done
  aws iam delete-role --role-name "${ROLE_NAME}" 2>/dev/null || true

  # --- Part 2 closed-loop resources (if deployed) ---
  # NOTE: `--query '...|[0]' --output text` renders as TWO lines ("None" then
  # the real value, or "None" alone with no match) for this pipe-into-index
  # JMESPath shape — the same quirk documented in deploy.sh's frontend .env
  # generation above. Use --output json + jq -r for a clean single scalar.
  say "Deleting closed-loop AgentCore runtimes..."
  for RUNTIME_NAME in "${RUNTIME_NAME_PREFIX}AdaptiveBiddingStrategyAgent" "${RUNTIME_NAME_PREFIX}ModelPromotionGovernanceAgent"; do
    RID="$(aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
      --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME}'].agentRuntimeId | [0]" \
      --output json 2>/dev/null | jq -r '. // empty')"
    if [[ -n "${RID}" ]]; then
      aws bedrock-agentcore-control delete-agent-runtime --agent-runtime-id "${RID}" --region "${AWS_REGION}" 2>/dev/null || true
      say "  Deleted runtime ${RID} (${RUNTIME_NAME})"
    else
      say "  Runtime ${RUNTIME_NAME} not found — nothing to destroy"
    fi
  done

  say "Deleting invocation stack (${GOVERNANCE_EVENTBRIDGE_STACK})..."
  aws cloudformation delete-stack --stack-name "${GOVERNANCE_EVENTBRIDGE_STACK}" --region "${AWS_REGION}" 2>/dev/null || true

  # --- SageMaker Model Registry: package groups must be emptied before
  # closed-loop-core's stack delete, or it fails DELETE_FAILED with "Model
  # Package Group ... cannot be deleted because it still contains Model
  # Packages" (observed live). Names mirror closed_loop_cfn.yaml exactly.
  say "Emptying SageMaker Model Package Groups (so closed-loop-core can delete)..."
  for GROUP in "${STACK_PREFIX:+${STACK_PREFIX}-}artf-dlrm-bid-shader" "${STACK_PREFIX:+${STACK_PREFIX}-}artf-ncf-deal-manager" "${STACK_PREFIX:+${STACK_PREFIX}-}artf-deal-yield-manager"; do
    if aws sagemaker describe-model-package-group --model-package-group-name "${GROUP}" --region "${AWS_REGION}" >/dev/null 2>&1; then
      for PKG_ARN in $(aws sagemaker list-model-packages --model-package-group-name "${GROUP}" --region "${AWS_REGION}" --query 'ModelPackageSummaryList[].ModelPackageArn' --output text 2>/dev/null); do
        aws sagemaker delete-model-package --model-package-name "${PKG_ARN}" --region "${AWS_REGION}" 2>/dev/null || true
      done
      say "  Emptied ${GROUP}"
    fi
  done

  say "Deleting closed-loop-core stack (${CLOSED_LOOP_STACK})..."
  aws cloudformation delete-stack --stack-name "${CLOSED_LOOP_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  aws cloudformation wait stack-delete-complete --stack-name "${CLOSED_LOOP_STACK}" --region "${AWS_REGION}" 2>/dev/null || true

  say "Deleting agentcore-security stack (${AGENTCORE_SECURITY_STACK})..."
  aws cloudformation delete-stack --stack-name "${AGENTCORE_SECURITY_STACK}" --region "${AWS_REGION}" 2>/dev/null || true

  # glue-etl imports FeedbackPipelineKMSKeyArn/RawOutcomesBucketArn from
  # feedback-pipeline (Fn::ImportValue) — feedback-pipeline's delete fails with
  # "Export ... is in use" while glue-etl still exists, so glue-etl must finish
  # deleting first.
  say "Deleting glue-etl stack (${GLUE_STACK})..."
  aws cloudformation delete-stack --stack-name "${GLUE_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  aws cloudformation wait stack-delete-complete --stack-name "${GLUE_STACK}" --region "${AWS_REGION}" 2>/dev/null || true

  say "Deleting feedback-pipeline stack (${FEEDBACK_STACK})..."
  aws cloudformation delete-stack --stack-name "${FEEDBACK_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  aws cloudformation wait stack-delete-complete --stack-name "${FEEDBACK_STACK}" --region "${AWS_REGION}" 2>/dev/null || true

  # --- Resources with DeletionPolicy: Retain in the CFN templates. Stack
  # deletion above leaves these behind by design; force-delete them explicitly
  # since nothing is meant to survive --destroy.
  say "Force-deleting retained DynamoDB tables..."
  for TABLE in "${STACK_PREFIX:+${STACK_PREFIX}-}parameter-store" "${STACK_PREFIX:+${STACK_PREFIX}-}audit-trail" "${STACK_PREFIX:+${STACK_PREFIX}-}user-features"; do
    aws dynamodb delete-table --table-name "${TABLE}" --region "${AWS_REGION}" 2>/dev/null || true
  done

  say "Force-deleting retained S3 buckets..."
  RAW_OUTCOMES_BUCKET="${STACK_PREFIX:+${STACK_PREFIX}-}raw-outcomes-${ACCOUNT_ID}-${AWS_REGION}"
  aws s3 rb "s3://${RAW_OUTCOMES_BUCKET}" --force 2>/dev/null || true
  aws s3 rb "s3://${TRAINING_DATA_BUCKET}" --force 2>/dev/null || true

  # --- CodeBuild project stack + its (non-CFN) source bucket. Not created by
  # deploy.sh's own destroy path at all previously — it's a separate stack
  # remote_build.sh deploys on demand.
  if aws cloudformation describe-stacks --stack-name "${CODEBUILD_STACK}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    say "Deleting CodeBuild project stack (${CODEBUILD_STACK})..."
    aws cloudformation delete-stack --stack-name "${CODEBUILD_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  fi
  CODEBUILD_SOURCE_BUCKET="${STACK_NAME}-codebuild-source-${ACCOUNT_ID}"
  aws s3 rb "s3://${CODEBUILD_SOURCE_BUCKET}" --force 2>/dev/null || true

  say "Deleting Cognito (user pool, identity pool, authenticated role)..."
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_cognito.py" --action destroy \
    --stack-name "${STACK_NAME}" --region "${AWS_REGION}" 2>/dev/null || true

  # --- ECR repositories: previously retained. Every repo deploy.sh/remote_build.sh
  # creates is prefixed with STACK_NAME (see the REPOS array + ADAPTIVE_BIDDING_REPO/
  # GOVERNANCE_REPO in deploy_closed_loop.sh), so this glob is exhaustive and safe —
  # it does NOT touch the shared, unprefixed artf-nemo-rl-training repo used by other
  # environments' training pipelines.
  say "Deleting ECR repositories (${STACK_NAME}-*)..."
  for REPO in $(aws ecr describe-repositories --region "${AWS_REGION}" --query "repositories[?starts_with(repositoryName,\`${STACK_NAME}\`)].repositoryName" --output text 2>/dev/null); do
    aws ecr delete-repository --repository-name "${REPO}" --force --region "${AWS_REGION}" 2>/dev/null || true
    say "  Deleted ${REPO}"
  done

  say "Destroy complete. No resources retained."
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

  # Regenerate the frontend build config so a standalone UI redeploy bakes in the
  # CURRENT Cognito + agent runtime ARNs. The UI must be (re)built AFTER the agents
  # exist for the VITE_* env vars to be embedded in the bundle at build time.
  COGNITO_OUTPUTS="${SCRIPT_DIR}/.cognito-outputs.json"
  COGNITO_USER_POOL_ID="$(jq -r '.UserPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
  COGNITO_CLIENT_ID="$(jq -r '.ClientId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
  IDENTITY_POOL_ID="$(jq -r '.IdentityPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
  # NOTE: `--query '...|[0]' --output text` on the AWS CLI renders this specific
  # pipe-into-index JMESPath shape as TWO lines ("None" then the real value) —
  # not the single scalar you'd expect — because the underlying result is a
  # single-element list, and the text formatter prints one line per list
  # element with "None" standing in for the (nonexistent) filter match at the
  # top level. Capturing that into a shell variable via $(...) previously
  # produced a literal "None\n<arn>" string, which got baked into the IAM
  # policy Resource list as leading garbage before failing to match any real
  # ARN, and also broken the VITE_* env values consumed by Vite (which reads
  # only the first line as the value up to the first newline, if you're lucky,
  # otherwise corrupts .env.production's later lines). Use --output json + jq
  # -r instead, which returns exactly the scalar (or empty string if no match).
  # Match the prefixed runtime name deploy_closed_loop.sh actually creates
  # (RUNTIME_NAME_PREFIX there == same STACK_PREFIX, hyphens->underscores).
  RUNTIME_NAME_PREFIX=""
  [[ -n "${STACK_PREFIX}" ]] && RUNTIME_NAME_PREFIX="$(echo "${STACK_PREFIX}_" | tr '-' '_')"
  ADAPTIVE_BIDDING_RUNTIME_ARN="$(aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
    --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME_PREFIX}AdaptiveBiddingStrategyAgent'].agentRuntimeArn | [0]" \
    --output json 2>/dev/null | jq -r '. // empty')"
  GOVERNANCE_RUNTIME_ARN="$(aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
    --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME_PREFIX}ModelPromotionGovernanceAgent'].agentRuntimeArn | [0]" \
    --output json 2>/dev/null | jq -r '. // empty')"
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

  # Primary distribution: React UI (fresh build embeds the env above)
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
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
# Preflight: GPU capacity for Triton + (optional) Model Optimizer
# =========================================================================
# Triton and the Model Optimizer each require their own A10G GPU node (the
# gpu-inference group uses the g5 family: g5.xlarge/2xlarge/4xlarge = 4/8/16 vCPU,
# 1 A10G each). With Part 2 closed-loop enabled (default) BOTH must run, so the
# deploy needs room for 2 GPU nodes. Fail fast with a clear message instead of
# letting the Model Optimizer sit Pending for hours (which leaves Triton unable to
# load its tensorrt_plan engines and the GPU containers stuck "gpu offline").
if [[ "${EXPORT_ONLY}" -eq 0 ]]; then
  # Steady state needs ONE GPU node (Triton only). The Model Optimizer no longer
  # runs as an always-on GPU Deployment — base engines are built by a one-shot
  # bootstrap Job (which exits and frees the GPU before Triton claims it) and
  # promotions by on-demand Jobs, both using a GPU only transiently. So 1 is the
  # hard minimum; a 2nd GPU is only needed transiently (bursts within --maxGPUs).
  REQUIRED_GPUS=1

  if [[ "${MAX_GPUS}" -lt "${REQUIRED_GPUS}" ]]; then
    fail "GPU capacity too low: --maxGPUs=${MAX_GPUS} but at least ${REQUIRED_GPUS} GPU node is needed for Triton (g5-family A10G). Re-run with --maxGPUs 1 (or 2+ to leave headroom for on-demand optimize / NeMo-RL retraining bursts)."
  fi

  # Retraining (NeMo-RL) spins up a GPU training job on top of the 2 steady-state
  # nodes; recommend headroom so it doesn't fight Triton/optimizer for a node.
  if [[ "${WITH_RETRAINING}" -eq 1 && "${MAX_GPUS}" -lt 3 ]]; then
    warn "  --with-retraining is on (default) but --maxGPUs=${MAX_GPUS}. NeMo-RL retraining jobs may wait for a GPU. Consider --maxGPUs 3 (or --no-retraining)."
  fi

  # Soft check: the account's On-Demand G/VT vCPU service quota must fit the GPU
  # nodes. This is a FLOOR based on the smallest g5 size (g5.xlarge = 4 vCPU); if
  # EKS falls back to a larger A10G size (g5.2xlarge=8, g5.4xlarge=16 vCPU) it
  # consumes more quota. This is the usual cause of a node that never launches.
  # Warn (don't hard-fail) if the quota can't be read.
  NEEDED_VCPUS=$(( REQUIRED_GPUS * 4 ))
  GVT_QUOTA="$(aws service-quotas get-service-quota \
    --service-code ec2 --quota-code L-DB2E81BA \
    --region "${AWS_REGION}" --query 'Quota.Value' --output text 2>/dev/null || echo '')"
  if [[ -n "${GVT_QUOTA}" && "${GVT_QUOTA}" != "None" ]]; then
    GVT_VCPUS="${GVT_QUOTA%.*}"   # strip any decimal
    if [[ "${GVT_VCPUS}" -lt "${NEEDED_VCPUS}" ]]; then
      fail "AWS quota too low for GPU nodes: 'Running On-Demand G and VT instances' (L-DB2E81BA) is ${GVT_VCPUS} vCPUs in ${AWS_REGION}, but at least ${NEEDED_VCPUS} are needed for ${REQUIRED_GPUS}x g5-family A10G nodes. Request an increase (Service Quotas console -> EC2 -> L-DB2E81BA) or use --no-retraining."
    fi
    log "  GPU quota OK: ${GVT_VCPUS} G/VT vCPUs available, need >= ${NEEDED_VCPUS} (${REQUIRED_GPUS}x g5-family A10G, min 4 vCPU each)"
  else
    warn "  Could not read the G/VT vCPU service quota (needs service-quotas:GetServiceQuota)."
    warn "  Ensure 'Running On-Demand G and VT instances' >= ${NEEDED_VCPUS} vCPUs in ${AWS_REGION}, or the GPU node(s) may never launch."
  fi
fi

# =========================================================================
# Step 1: ECR repositories
# =========================================================================
REPOS=(
  ${STACK_NAME}-$(display_name dlrm-bid-shader)
  ${STACK_NAME}-$(display_name widedeep-segment-activator)
  ${STACK_NAME}-$(display_name ncf-deal-manager)
  ${STACK_NAME}-$(display_name metrics-enricher)
  ${STACK_NAME}-$(display_name deal-yield-manager)
  ${STACK_NAME}-orchestrator
  ${STACK_NAME}-agentcore
)

if [[ "${START_AT}" -le 1 ]]; then
phase 1 "Preparing models"
log "Step 1: Ensuring ECR repositories"
if [[ "${LOCAL_BUILD}" -eq 1 ]]; then
  aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${REGISTRY}"
fi

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
log "Step 2: Exporting PyTorch models to ONNX (source for the Model Optimizer)"
# Part 2: Triton serves TensorRT engines (tensorrt_plan). The exported ONNX is the
# SOURCE the Model Optimizer compiles into engines — it is uploaded to onnx-source/
# (Step 3a), NOT served directly. Export to a staging dir so it does not collide
# with the router/engine configs committed under triton/model_repository/.
ONNX_STAGING="${SCRIPT_DIR}/../source/triton/onnx_export"
rm -rf "${ONNX_STAGING}"
${PYTHON} "${SCRIPT_DIR}/../source/triton/export_models.py" \
  --output-dir "${ONNX_STAGING}"

# Step 2.5: Genesis XGBoost export for the Yield Optimizer (floor/margin).
# Best-effort, non-blocking (per explicit user instruction: train the
# genesis models, but never make deployment depend on it completing).
# xgboost/onnxmltools are intentionally NOT added to REQUIRED_PY_PACKAGES
# above (that check uses fail(), which would block the deploy) -- checked
# and installed independently here, with a warn()-only fallback. If genesis
# export doesn't succeed,
# register_genesis_models.py (Step 3, deploy_closed_loop.sh) skips that
# model type honestly (its own existing "missing artifact" path -- see
# aidlc-docs/construction/deal-yield-training-pipeline/tasks.md Group 8) and
# scheduled retraining/genesis registration can be re-run later once the
# packages are available, using the exact same known bucket path/env vars
# (MODEL_BUCKET, AWS_REGION) the rest of this script already uses -- no new
# environment variables introduced for this step.
#
# xgboost is pinned to 1.7.6 (not "latest") because save_model()'s JSON
# output format changed in a later minor version to add a "cats"
# (categorical-feature) field under gradient_booster.model -- present even
# for models with zero categorical features. Triton 24.08's bundled FIL/
# treelite backend predates that field and hard-fails to load the model
# ("Error: key \"cats\" is not recognized!"), taking deal_yield_manager_floor/
# margin down even though the exported model is otherwise valid. 1.7.6
# matches the SageMaker training-side pin (see xgboost_pipeline.py's
# image_uris.retrieve(..., version="1.7-1")) and the FIL backend's
# documented support matrix (Triton 24.03-24.09 -> XGBoost JSON 1.7+,
# no "cats" field). If a system already has a newer xgboost installed,
# force-reinstall the pinned version rather than trusting the existing one.
XGBOOST_PIN="xgboost==1.7.6"
XGBOOST_OK=0
if ${PYTHON} -c "
import sys
try:
    import onnxmltools, xgboost
except ImportError:
    sys.exit(1)
sys.exit(0 if xgboost.__version__ == '1.7.6' else 1)
" 2>/dev/null; then
  XGBOOST_OK=1
fi

log "Step 2.5: Exporting genesis XGBoost models (Yield Optimizer floor/margin)"
if [[ "${XGBOOST_OK}" -ne 1 ]]; then
  log "  Installing ${XGBOOST_PIN}/onnxmltools (best-effort, non-blocking)..."
  ${PYTHON} -m pip install --quiet "${XGBOOST_PIN}" onnxmltools 2>/dev/null || true
  if ${PYTHON} -c "
import sys
try:
    import onnxmltools, xgboost
except ImportError:
    sys.exit(1)
sys.exit(0 if xgboost.__version__ == '1.7.6' else 1)
" 2>/dev/null; then
    XGBOOST_OK=1
  fi
fi

if [[ "${XGBOOST_OK}" -eq 1 ]]; then
  ${PYTHON} "${SCRIPT_DIR}/../source/training/export_xgboost_genesis.py" \
    --output-dir "${ONNX_STAGING}" \
    && log "  Genesis XGBoost artifacts exported to ${ONNX_STAGING}/" \
    || warn "  Genesis XGBoost export failed - deal_yield_manager_floor/margin genesis registration will be skipped honestly until this is re-run (see Step 3's register_genesis_models.py)."
else
  warn "  Could not install ${XGBOOST_PIN}/onnxmltools - skipping genesis XGBoost export (deal_yield_manager_floor/margin genesis registration will be skipped honestly). Install manually with: ${PYTHON} -m pip install ${XGBOOST_PIN} onnxmltools"
fi

if [[ "${EXPORT_ONLY}" -eq 1 ]]; then
  log "Export complete (--export-only). ONNX at ${ONNX_STAGING}/"
  exit 0
fi

# =========================================================================
# Step 3: Upload model repository to S3
# =========================================================================
log "Step 3: Ensuring S3 model bucket ${MODEL_BUCKET}"
if ! aws s3api head-bucket --bucket "${MODEL_BUCKET}" 2>/dev/null; then
  aws s3 mb "s3://${MODEL_BUCKET}" --region "${AWS_REGION}"
fi

# widedeep_segment_activator is intentionally absent — segment activation is
# rule-based, not a Triton/TensorRT model (see
# source/containers/widedeep_segment_activator/app.py).
RECOMMENDER_MODELS=(dlrm_bid_shader ncf_deal_manager)

# Yield Optimizer sub-models (XGBoost via Triton FIL) -- genesis artifacts
# come from Step 2.5, which is best-effort/non-blocking. Only uploaded if
# Step 2.5 actually produced them (checked per-model below), so a missing
# xgboost/onnxmltools install never fails Step 3 for the other models.
YIELD_MODELS=(deal_yield_manager_floor deal_yield_manager_margin)

# 3a. Upload exported ONNX to onnx-source/ — the Model Optimizer reads these to
#     build TensorRT engines. NOT served directly by Triton.
for m in "${RECOMMENDER_MODELS[@]}"; do
  aws s3 cp "${ONNX_STAGING}/${m}/1/model.onnx" \
    "s3://${MODEL_BUCKET}/onnx-source/${m}/model.onnx" --region "${AWS_REGION}"
done
log "  ONNX uploaded to s3://${MODEL_BUCKET}/onnx-source/"

# 3a2. Yield Optimizer genesis: ONNX form for registry bookkeeping
# (onnx-source/, matches the DLRM/NCF convention exactly so
# register_genesis_models.py needs no format-specific branching) AND the
# native XGBoost JSON form (triton-models/<model>/1/xgboost.json) --
# Triton's FIL backend does NOT read ONNX, only the native format (see
# aidlc-docs/construction/deal-yield-training-pipeline/functional-design/
# business-logic-model.md Logic Flow 2). Skipped honestly per-model if Step
# 2.5 did not produce that model's artifacts.
for m in "${YIELD_MODELS[@]}"; do
  if [[ -f "${ONNX_STAGING}/${m}/1/model.onnx" && -f "${ONNX_STAGING}/${m}/1/xgboost.json" ]]; then
    aws s3 cp "${ONNX_STAGING}/${m}/1/model.onnx" \
      "s3://${MODEL_BUCKET}/onnx-source/${m}/model.onnx" --region "${AWS_REGION}"
    aws s3 cp "${ONNX_STAGING}/${m}/1/xgboost.json" \
      "s3://${MODEL_BUCKET}/triton-models/${m}/1/xgboost.json" --region "${AWS_REGION}"
    log "  ${m}: genesis ONNX + native XGBoost JSON uploaded"
  else
    warn "  ${m}: no genesis artifact from Step 2.5 - skipping upload (Model Registry genesis and Triton FIL load for this model will start empty until Step 2.5 succeeds)."
  fi
done

# 3b. Assemble and upload the served Triton repo: router configs + stable/canary
#     engine configs (committed under triton/model_repository/) plus the canary
#     router model.py injected into each router model's version dir. Engines
#     (model.plan) are built by the Model Optimizer at Step 8b, so they are NOT
#     uploaded here (and NO --delete, which would wipe engines on re-deploy).
SERVED_STAGING="$(mktemp -d)"
cp -R "${SCRIPT_DIR}/../source/triton/model_repository/." "${SERVED_STAGING}/"
for m in "${RECOMMENDER_MODELS[@]}"; do
  mkdir -p "${SERVED_STAGING}/${m}/1"
  cp "${SCRIPT_DIR}/../source/triton/router/model.py" "${SERVED_STAGING}/${m}/1/model.py"
done
aws s3 sync "${SERVED_STAGING}/" "s3://${MODEL_BUCKET}/triton-models/" \
  --exclude "*.onnx" --region "${AWS_REGION}"
rm -rf "${SERVED_STAGING}"
log "  Triton served repo (routers + engine configs) uploaded; engines built at Step 8b"

# 3c. Upload the Model Optimizer bootstrap spec (base-engine build instructions).
BOOTSTRAP_TMP="$(mktemp)"
sed "s|__MODEL_BUCKET__|${MODEL_BUCKET}|g" \
  "${SCRIPT_DIR}/optimizer-bootstrap.json" > "${BOOTSTRAP_TMP}"
aws s3 cp "${BOOTSTRAP_TMP}" \
  "s3://${MODEL_BUCKET}/optimizer-bootstrap/spec.json" --region "${AWS_REGION}"
rm -f "${BOOTSTRAP_TMP}"
log "  Model Optimizer bootstrap spec uploaded"
ok "Models exported and uploaded"
fi # START_AT <= 1 (Phase 1)

# =========================================================================
# Step 4: Build and push container images
# =========================================================================
if [[ "${START_AT}" -le 2 ]]; then
phase 2 "Building containers & provisioning infrastructure"
log "  This typically takes 15-20 minutes (bounded by EKS cluster creation)."
fi
_PHASE2_START=$(date +%s)
IMAGE_OUTPUTS="${SCRIPT_DIR}/.image-outputs.json"

# Always read the outputs file if it exists (needed for --start-at to use the correct tag)
if [[ -f "${IMAGE_OUTPUTS}" ]]; then
  PREV_TAG="$(jq -r '.ImageTag // empty' "${IMAGE_OUTPUTS}" 2>/dev/null || echo '')"
  PREV_REGISTRY="$(jq -r '.Registry // empty' "${IMAGE_OUTPUTS}" 2>/dev/null || echo '')"
  if [[ -n "${PREV_TAG}" && "${PREV_REGISTRY}" == "${REGISTRY}" ]]; then
    IMAGE_TAG="${PREV_TAG}"
  fi
fi

# =========================================================================
# Content-hash helpers — detect source drift under a reused IMAGE_TAG
# =========================================================================
# IMAGE_TAG defaults to the git short SHA, so re-running deploy.sh at the SAME
# commit (e.g. after fixing a live issue without committing, or re-running
# --start-at=4) previously skipped a build purely because that tag already
# existed in ECR — even if the image content underneath it was stale (e.g.
# built from source that predates a bugfix, then never rebuilt because the SHA
# didn't change). This computes a content hash over each image's actual
# Dockerfile + COPY'd source, and treats that hash — not just IMAGE_TAG — as
# the source of truth for "is a rebuild needed".

# Pick whichever sha256 tool is on PATH (sha256sum on Linux/CodeBuild, shasum
# on macOS).
_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    shasum -a 256
  fi
}

# Compute a short content hash for the given image key over exactly the files
# that Dockerfile actually COPYs (kept in sync with build_image_local's cases
# and the Dockerfiles under source/). Returns empty on an unrecognized key so
# callers can fail safe (rebuild) instead of silently trusting a stale image.
_source_hash() {
  local key="$1"
  local src="${SCRIPT_DIR}/../source"
  local paths=()
  case "${key}" in
    dlrm-bid-shader|ncf-deal-manager|deal-yield-manager)
      paths=("${src}/triton/Dockerfile.triton-artf" "${src}/shared" "${src}/containers/${key//-/_}") ;;
    widedeep-segment-activator|metrics-enricher)
      paths=("${src}/Dockerfile" "${src}/shared" "${src}/containers/${key//-/_}") ;;
    orchestrator)
      paths=("${src}/Dockerfile.orchestrator" "${src}/shared" "${src}/agents" "${src}/closed_loop_demo" "${src}/orchestrator") ;;
    agentcore)
      paths=("${src}/Dockerfile.agentcore" "${src}/shared" "${src}/containers" "${src}/agentcore") ;;
    model-optimizer)
      paths=("${src}/Dockerfile.optimizer" "${src}/optimizer") ;;
    *)
      return 1 ;;
  esac
  local f rel manifest=""
  for p in "${paths[@]}"; do
    if [[ -f "${p}" ]]; then
      rel="${p#${src}/}"
      manifest+="${rel}:$(_sha256 < "${p}" | awk '{print $1}')"$'\n'
    elif [[ -d "${p}" ]]; then
      while IFS= read -r f; do
        rel="${f#${src}/}"
        manifest+="${rel}:$(_sha256 < "${f}" | awk '{print $1}')"$'\n'
      done < <(find "${p}" -type f | LC_ALL=C sort)
    fi
    # A path that is neither a file nor a directory (e.g. deleted) is silently
    # skipped — this only affects the resulting hash value, not correctness.
  done
  printf '%s' "${manifest}" | _sha256 | awk '{print $1}' | cut -c1-16
}

# Make ${dest_tag} point at the same image as ${src_tag} in ${repo}, without
# re-uploading any layers (fetches the manifest and re-registers it under the
# new tag). Used to (a) attach a content-hash tag right after a fresh build,
# and (b) reuse an already-built image under a new IMAGE_TAG when its content
# hash shows nothing actually changed. Best-effort: returns non-zero on any
# failure so the caller can fall back to a full rebuild instead of trusting a
# tag that may not have been created.
_ensure_tag_alias() {
  local repo="$1" src_tag="$2" dest_tag="$3"
  [[ "${src_tag}" == "${dest_tag}" ]] && return 0

  local src_digest dest_digest
  src_digest="$(aws ecr describe-images --repository-name "${repo}" --image-ids imageTag="${src_tag}" \
    --region "${AWS_REGION}" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || echo '')"
  [[ -z "${src_digest}" || "${src_digest}" == "None" ]] && return 1

  dest_digest="$(aws ecr describe-images --repository-name "${repo}" --image-ids imageTag="${dest_tag}" \
    --region "${AWS_REGION}" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || echo '')"
  [[ "${dest_digest}" == "${src_digest}" ]] && return 0

  local manifest media_type
  manifest="$(aws ecr batch-get-image --repository-name "${repo}" --image-ids imageTag="${src_tag}" \
    --region "${AWS_REGION}" --query 'images[0].imageManifest' --output text 2>/dev/null || echo '')"
  [[ -z "${manifest}" || "${manifest}" == "None" ]] && return 1
  media_type="$(aws ecr batch-get-image --repository-name "${repo}" --image-ids imageTag="${src_tag}" \
    --region "${AWS_REGION}" --query 'images[0].imageManifestMediaType' --output text 2>/dev/null || echo '')"

  if [[ -n "${media_type}" && "${media_type}" != "None" ]]; then
    aws ecr put-image --repository-name "${repo}" --image-tag "${dest_tag}" \
      --image-manifest "${manifest}" --image-manifest-media-type "${media_type}" \
      --region "${AWS_REGION}" >/dev/null 2>&1
  else
    aws ecr put-image --repository-name "${repo}" --image-tag "${dest_tag}" \
      --image-manifest "${manifest}" --region "${AWS_REGION}" >/dev/null 2>&1
  fi
}

# Build one Part-1/optimizer image locally by key (repo suffix). The ECR repo
# name uses the job-oriented display name (see display_name()); "key" itself
# stays the old model-architecture name for directory/Dockerfile resolution.
build_image_local() {
  local key="$1"
  local repo="${STACK_NAME}-$(display_name "${key}")"
  local image="${REGISTRY}/${repo}:${IMAGE_TAG}"
  local src="${SCRIPT_DIR}/../source"
  aws ecr describe-repositories --repository-names "${repo}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${repo}" --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true >/dev/null
  case "${key}" in
    dlrm-bid-shader|ncf-deal-manager|deal-yield-manager)
      log "  Building ${repo} (amd64, tritonclient)"
      docker buildx build --platform linux/amd64 --build-arg CONTAINER="containers/${key//-/_}" \
        -f "${src}/triton/Dockerfile.triton-artf" -t "${image}" --load "${src}"
      docker push "${image}" ;;
    widedeep-segment-activator|metrics-enricher)
      # Rule-based container — no Triton dependency (segment activation was
      # switched from the Wide & Deep Triton model to deterministic rules; see
      # source/containers/widedeep_segment_activator/app.py). Uses the plain
      # ARTF Dockerfile like metrics-enricher.
      log "  Building ${repo} (amd64)"
      docker buildx build --platform linux/amd64 \
        --build-arg CONTAINER="containers/${key//-/_}" --build-arg AGENT_NAME="${key}" \
        -f "${src}/Dockerfile" -t "${image}" --load "${src}"
      docker push "${image}" ;;
    orchestrator)
      log "  Building ${repo} (amd64)"
      docker buildx build --platform linux/amd64 --build-arg AGENT_NAME="artf-orchestrator" \
        -f "${src}/Dockerfile.orchestrator" -t "${image}" --load "${src}"
      docker push "${image}" ;;
    agentcore)
      log "  Building ${repo} (arm64)"
      docker buildx build --platform linux/arm64 \
        -f "${src}/Dockerfile.agentcore" -t "${image}" --load "${src}"
      docker push "${image}" ;;
    model-optimizer)
      log "  Building ${repo} (amd64)"
      docker buildx build --platform linux/amd64 \
        -f "${src}/Dockerfile.optimizer" -t "${image}" --load "${src}"
      docker push "${image}" ;;
    *)
      warn "  Unknown image key '${key}' — skipping" ;;
  esac
}

# Per-image build: (re)build only images whose CONTENT has changed, instead
# of an all-or-nothing rebuild. deploy.sh (Step 4) owns the Part 1 images +
# the Model Optimizer; the Part 2 agents and NeMo are built later by
# deploy_closed_loop.sh.
#
# IMAGE_TAG alone (default: git short SHA) is NOT sufficient to decide
# "skip this build" — re-running deploy.sh at the same commit (e.g. after
# editing source without committing, or via --start-at=2) would previously
# skip rebuilding as long as ANY image existed at that tag, even one built
# from stale source. Each image also gets a content-hash tag
# (src-<16 hex chars>, hashed over exactly the files its Dockerfile COPYs).
# A build is skipped only when that hash tag already exists in ECR — i.e.
# this exact source content has already been built — not merely because
# IMAGE_TAG exists. To force a rebuild regardless, delete both tags (or
# change the source, which changes the hash automatically).
#
# Factored into a function (was previously inline) so it can run concurrently
# with ensure_eks_cluster() below (FR-8) — building images has no dependency
# on the cluster existing.
build_images() {
  STEP4_KEYS=(dlrm-bid-shader widedeep-segment-activator ncf-deal-manager metrics-enricher deal-yield-manager orchestrator model-optimizer)
  if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then STEP4_KEYS+=(agentcore); fi

  MISSING_KEYS=()
  HASH_KEYS=()   # parallel array to MISSING_KEYS (bash 3.2 has no assoc arrays)
  HASH_VALS=()
  for key in "${STEP4_KEYS[@]}"; do
    repo="${STACK_NAME}-$(display_name "${key}")"
    src_hash="$(_source_hash "${key}" || echo '')"
    if [[ -z "${src_hash}" ]]; then
      warn "Step 4: could not hash source for '${key}' (unrecognized key) — will rebuild"
      MISSING_KEYS+=("${key}")
      continue
    fi
    src_tag="src-${src_hash}"
    if aws ecr describe-images --repository-name "${repo}" --image-ids imageTag="${src_tag}" \
         --region "${AWS_REGION}" >/dev/null 2>&1; then
      log "Step 4: ${key} content unchanged (${src_tag}) — reusing existing image, no rebuild"
      if _ensure_tag_alias "${repo}" "${src_tag}" "${IMAGE_TAG}"; then
        log "  ${repo}:${IMAGE_TAG} -> ${src_tag}"
      else
        warn "  Could not alias ${repo}:${IMAGE_TAG} to ${src_tag} — rebuilding to be safe"
        MISSING_KEYS+=("${key}")
        HASH_KEYS+=("${key}"); HASH_VALS+=("${src_tag}")
      fi
    else
      MISSING_KEYS+=("${key}")
      HASH_KEYS+=("${key}"); HASH_VALS+=("${src_tag}")
    fi
  done

  if [[ ${#MISSING_KEYS[@]} -eq 0 ]]; then
    log "Step 4: All required images content-matched. Nothing to build."
  elif [[ "${LOCAL_BUILD}" -eq 0 ]]; then
    log "Step 4: Building changed images via CodeBuild (tag=${IMAGE_TAG}): ${MISSING_KEYS[*]}"
    NGC_FLAG=()
    if [[ -n "${NGC_KEY}" ]]; then NGC_FLAG=(--ngc-key "${NGC_KEY}")
    elif [[ -n "${NGC_SECRET}" ]]; then NGC_FLAG=(--ngc-secret "${NGC_SECRET}"); fi
    "${SCRIPT_DIR}/codebuild/remote_build.sh" \
      --stack-name "${STACK_NAME}" \
      --only "${MISSING_KEYS[*]}" \
      --tag "${IMAGE_TAG}" \
      --region "${AWS_REGION}" \
      "${NGC_FLAG[@]}"
  else
    log "Step 4: Building changed images locally (tag=${IMAGE_TAG}): ${MISSING_KEYS[*]}"
    aws ecr get-login-password --region "${AWS_REGION}" | \
      docker login --username AWS --password-stdin "${REGISTRY}"
    for key in "${MISSING_KEYS[@]}"; do
      build_image_local "${key}"
    done
    log "  Changed images built and pushed"
  fi

  # Record a content-hash tag alias for every image we just (re)built, so the
  # NEXT run recognizes this exact source content without rebuilding — works
  # for both the CodeBuild and local build paths since it only reads/writes
  # ECR tag metadata (no re-upload of layers).
  for i in "${!HASH_KEYS[@]}"; do
    key="${HASH_KEYS[$i]}"; src_tag="${HASH_VALS[$i]}"
    repo="${STACK_NAME}-$(display_name "${key}")"
    if _ensure_tag_alias "${repo}" "${IMAGE_TAG}" "${src_tag}"; then
      log "  Tagged ${repo}:${IMAGE_TAG} as ${src_tag} (content-hash cache)"
    else
      warn "  Could not record content-hash tag ${src_tag} for ${repo} — next run will rebuild it even if unchanged"
    fi
  done

  # Record the tag/registry so --start-at re-runs reference the same images.
  cat > "${IMAGE_OUTPUTS}" <<EOF
{
  "Registry": "${REGISTRY}",
  "ImageTag": "${IMAGE_TAG}",
  "StackName": "${STACK_NAME}",
  "BuiltAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
  ok "Images built and pushed"
}

# =========================================================================
# Step 5: Create or reuse EKS cluster
# =========================================================================
# Render the eksctl ClusterConfig and run `eksctl create cluster`.
create_eks_cluster() {
  log "  GPU node group: desired 1, max ${MAX_GPUS} (set via --maxGPUs)"
  local CLUSTER_CONFIG="/tmp/${CLUSTER_NAME}-config.yaml"
  sed -e "s/__STACK_NAME__/${STACK_NAME}/g" \
      -e "s/__REGION__/${AWS_REGION}/g" \
      -e "s/__MAX_GPUS__/${MAX_GPUS}/g" \
      "${SCRIPT_DIR}/eks/cluster-config.yaml" > "${CLUSTER_CONFIG}"
  eksctl create cluster -f "${CLUSTER_CONFIG}"
}

# Factored into a function (was previously inline) so it can run concurrently
# with build_images() above (FR-8) — cluster creation has no dependency on the
# images existing. Any internal `fail` call here runs in the backgrounded
# subshell (see the orchestration block below) and only exits that subshell;
# the caller checks the real exit code via `wait`.
ensure_eks_cluster() {
  if eksctl get cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Step 5: EKS cluster ${CLUSTER_NAME} already exists"
    return 0
  fi

  # `eksctl get cluster` found no active cluster, but a previous
  # `eksctl create cluster` may have left an orphaned CloudFormation stack
  # (e.g. ROLLBACK_COMPLETE / CREATE_FAILED after a mid-create failure). In
  # that state a fresh create fails with:
  #   AlreadyExistsException: Stack [eksctl-<cluster>-cluster] already exists
  # eksctl base cluster stacks cannot be updated in place, so recovery is:
  # reuse it if it's healthy, otherwise delete the failed stack and recreate.
  CLUSTER_STACK="eksctl-${CLUSTER_NAME}-cluster"
  STACK_STATUS="$(aws cloudformation describe-stacks \
    --stack-name "${CLUSTER_STACK}" \
    --region "${AWS_REGION}" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"

  if [[ -z "${STACK_STATUS}" ]]; then
    log "Step 5: Creating EKS cluster ${CLUSTER_NAME} (15-20 min)"
    create_eks_cluster
  else
    warn "Step 5: Found existing eksctl stack ${CLUSTER_STACK} (status: ${STACK_STATUS})"
    case "${STACK_STATUS}" in
      CREATE_COMPLETE|UPDATE_COMPLETE|UPDATE_ROLLBACK_COMPLETE)
        # Stack is healthy but `eksctl get cluster` didn't list it — reuse it
        # (kubeconfig is refreshed below in Step 5.5). No create needed.
        log "  Stack is healthy; reusing existing cluster"
        ;;
      *_IN_PROGRESS)
        fail "  Stack ${CLUSTER_STACK} is ${STACK_STATUS}; wait for it to settle, then re-run"
        ;;
      ROLLBACK_COMPLETE|ROLLBACK_FAILED|CREATE_FAILED|DELETE_FAILED|UPDATE_ROLLBACK_FAILED)
        warn "  Stack is in a failed state; deleting the orphaned cluster before recreating"
        # Prefer eksctl (cleans up all associated stacks); fall back to a raw
        # CFN delete if eksctl can't (the cluster resource may never have existed).
        eksctl delete cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --wait 2>/dev/null \
          || aws cloudformation delete-stack --stack-name "${CLUSTER_STACK}" --region "${AWS_REGION}"
        aws cloudformation wait stack-delete-complete \
          --stack-name "${CLUSTER_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
        log "  Orphaned stack deleted; creating fresh cluster ${CLUSTER_NAME} (15-20 min)"
        create_eks_cluster
        ;;
      *)
        fail "  Stack ${CLUSTER_STACK} in unexpected state ${STACK_STATUS}; resolve manually then re-run"
        ;;
    esac
  fi
  ok "EKS cluster ready"
}

# -------------------------------------------------------------------------
# Orchestration (FR-8): when both images and cluster need work, build images
# in the foreground while the cluster is created in a backgrounded subshell,
# then wait and propagate the subshell's real exit code. `set -e` does not
# apply inside `$(...)`/background subshells the way it does inline, so the
# explicit `wait` exit-code check below is what actually catches a cluster
# creation failure — without it, a failed background `eksctl create cluster`
# would be silently swallowed and the script would carry on into Step 6+
# against a cluster that doesn't exist.
# -------------------------------------------------------------------------
if [[ "${SKIP_IMAGES}" -eq 0 && "${SKIP_CLUSTER}" -eq 0 ]]; then
  ( ensure_eks_cluster ) &
  _cluster_pid=$!
  build_images
  if ! wait "${_cluster_pid}"; then
    fail 2 "EKS cluster creation failed (see output above). Check: eksctl get cluster --name ${CLUSTER_NAME} --region ${AWS_REGION}"
  fi
elif [[ "${SKIP_IMAGES}" -eq 0 ]]; then
  build_images
  warn "Skipping EKS cluster creation (--skip-cluster)"
elif [[ "${SKIP_CLUSTER}" -eq 0 ]]; then
  ensure_eks_cluster
  warn "Skipping image build (--skip-images)"
else
  warn "Skipping image build (--skip-images)"
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

# eksctl returns before the ServiceAccount's IRSA annotation is guaranteed to
# be readable back from the API server. Reading it immediately after create
# was racing that propagation and sometimes returning empty, which then got
# baked into triton-deployment.yaml via sed -- leaving triton-sa with
# eks.amazonaws.com/role-arn: "" and Triton unable to reach its S3 model repo.
# Poll instead of reading once.
TRITON_ROLE_ARN=""
for attempt in $(seq 1 10); do
  TRITON_ROLE_ARN="$(kubectl get sa triton-sa -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo '')"
  [[ -n "${TRITON_ROLE_ARN}" ]] && break
  sleep 3
done
if [[ -z "${TRITON_ROLE_ARN}" ]]; then
  # FATAL, not a warning: an empty value here gets sed'd into
  # triton-deployment.yaml's ServiceAccount manifest in Step 8 and applied —
  # silently shipping a triton-sa with eks.amazonaws.com/role-arn: "" that can
  # never reach its S3 model repo. This has happened live when triton-sa's
  # eksctl CFN stack survives (e.g. after a partial destroy or manual
  # `kubectl delete`) but the ServiceAccount object itself doesn't: eksctl
  # sees the CFN stack and treats the ServiceAccount as already provisioned,
  # skipping recreation, so it never appears for this poll to find. Stop here
  # instead of deploying a broken Triton.
  fail "Triton IRSA role annotation did not populate after 30s. If triton-sa's ServiceAccount is missing from the cluster but its CFN stack (eksctl-${CLUSTER_NAME}-addon-iamserviceaccount-default-triton-sa) still exists, eksctl will not recreate it automatically. Fix by deleting that CFN stack first, or manually: kubectl annotate sa triton-sa eks.amazonaws.com/role-arn=<role-arn> --overwrite (find the role ARN via: aws cloudformation describe-stacks --stack-name eksctl-${CLUSTER_NAME}-addon-iamserviceaccount-default-triton-sa --query 'Stacks[0].Outputs')"
fi
log "  Triton IRSA role: ${TRITON_ROLE_ARN}"

# =========================================================================
# Step 7.1: IRSA for the Model Optimizer (S3 read/write on the model repo)
# The optimizer reads ONNX from onnx-source/ and writes TensorRT engines to
# triton-models/ (base engines + canary engines). It needs read+write+list.
# =========================================================================
log "Step 7.1: Ensuring IRSA for the Model Optimizer S3 access"
OPTIMIZER_POLICY_NAME="${STACK_NAME}-model-optimizer-s3-${STACK_UID}"
OPTIMIZER_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${OPTIMIZER_POLICY_NAME}"
OPTIMIZER_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":\"arn:aws:s3:::${MODEL_BUCKET}/*\"},{\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::${MODEL_BUCKET}\"}]}"
if aws iam get-policy --policy-arn "${OPTIMIZER_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${OPTIMIZER_POLICY_ARN}" \
    --policy-document "${OPTIMIZER_POLICY_DOC}" \
    --set-as-default 2>/dev/null || true
else
  aws iam create-policy \
    --policy-name "${OPTIMIZER_POLICY_NAME}" \
    --policy-document "${OPTIMIZER_POLICY_DOC}" >/dev/null
fi

eksctl create iamserviceaccount \
  --name model-optimizer-sa \
  --namespace default \
  --cluster "${CLUSTER_NAME}" \
  --region "${AWS_REGION}" \
  --attach-policy-arn "${OPTIMIZER_POLICY_ARN}" \
  --approve \
  --override-existing-serviceaccounts 2>/dev/null || true

# Same eksctl/kubectl propagation race as triton-sa above -- poll instead of
# reading the annotation once.
OPTIMIZER_ROLE_ARN=""
for attempt in $(seq 1 10); do
  OPTIMIZER_ROLE_ARN="$(kubectl get sa model-optimizer-sa -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo '')"
  [[ -n "${OPTIMIZER_ROLE_ARN}" ]] && break
  sleep 3
done
if [[ -z "${OPTIMIZER_ROLE_ARN}" ]]; then
  # FATAL, not a warning — same reasoning as triton-sa above: an empty value
  # here gets baked into applied manifests (model-optimizer-bootstrap-job.yaml,
  # the on-demand optimize Jobs) and silently ships a ServiceAccount that can
  # never reach S3.
  fail "Model Optimizer IRSA role annotation did not populate after 30s. If model-optimizer-sa's ServiceAccount is missing from the cluster but its CFN stack (eksctl-${CLUSTER_NAME}-addon-iamserviceaccount-default-model-optimizer-sa) still exists, eksctl will not recreate it automatically. Fix by deleting that CFN stack first, or manually: kubectl annotate sa model-optimizer-sa eks.amazonaws.com/role-arn=<role-arn> --overwrite (find the role ARN via: aws cloudformation describe-stacks --stack-name eksctl-${CLUSTER_NAME}-addon-iamserviceaccount-default-model-optimizer-sa --query 'Stacks[0].Outputs')"
fi
log "  Model Optimizer IRSA role: ${OPTIMIZER_ROLE_ARN}"

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
# Step 7.7: Closed-loop (Part 2) permissions for the orchestrator
# =========================================================================
# The orchestrator exposes the Part 2 closed-loop demo API, which:
#   - emits synthetic input metrics and reads market metrics (CloudWatch)
#   - reads/writes the DynamoDB Parameter Store + Audit Trail (KMS-encrypted)
#   - lists/describes SageMaker model package versions
# These stores are created by deploy_closed_loop.sh; the policy is scoped by
# name/ARN patterns so it is valid whether or not a stack prefix is used.
log "Step 7.7: Ensuring closed-loop (Part 2) access for orchestrator"
# Matches closed_loop_cfn.yaml's SageMakerTrainingExecutionRole naming
# (prefixed if STACK_PREFIX is set, unprefixed otherwise) — the role the
# governance UI's on-demand training trigger must be allowed to pass to
# SageMaker's CreateTrainingJob RoleArn.
if [[ -n "${STACK_PREFIX}" ]]; then
  SAGEMAKER_TRAINING_ROLE_NAME="${STACK_PREFIX}-sagemaker-training-execution-role"
else
  SAGEMAKER_TRAINING_ROLE_NAME="sagemaker-training-execution-role"
fi
CLOSED_LOOP_POLICY_NAME="${STACK_NAME}-closed-loop-${STACK_UID}"
CLOSED_LOOP_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${CLOSED_LOOP_POLICY_NAME}"
CLOSED_LOOP_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[\
{\"Sid\":\"EmitBidOutcomeMetrics\",\"Effect\":\"Allow\",\"Action\":[\"cloudwatch:PutMetricData\"],\"Resource\":\"*\",\"Condition\":{\"StringLike\":{\"cloudwatch:namespace\":[\"ARTF/*\"]}}},\
{\"Sid\":\"ReadMetrics\",\"Effect\":\"Allow\",\"Action\":[\"cloudwatch:GetMetricData\",\"cloudwatch:GetMetricStatistics\"],\"Resource\":\"*\"},\
{\"Sid\":\"ParameterStoreAndAudit\",\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:Query\",\"dynamodb:PutItem\",\"dynamodb:BatchGetItem\",\"dynamodb:DescribeTable\"],\"Resource\":[\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/parameter-store\",\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/audit-trail\",\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/*-parameter-store\",\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/*-audit-trail\"]},\
{\"Sid\":\"SchedulerToggle\",\"Effect\":\"Allow\",\"Action\":[\"scheduler:GetSchedule\",\"scheduler:UpdateSchedule\"],\"Resource\":[\"arn:aws:scheduler:${AWS_REGION}:${ACCOUNT_ID}:schedule/default/*\"]},\
{\"Sid\":\"ModelRegistryRead\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:ListModelPackages\",\"sagemaker:DescribeModelPackage\"],\"Resource\":[\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:model-package-group/*artf-*\",\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:model-package/*artf-*/*\"]},\
{\"Sid\":\"TrainingTriggerFromGovernanceUI\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:CreateTrainingJob\",\"sagemaker:DescribeTrainingJob\"],\"Resource\":[\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:training-job/dlrm-bid-shader-*\",\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:training-job/ncf-deal-manager-*\"]},\
{\"Sid\":\"ListTrainingJobsFromGovernanceUI\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:ListTrainingJobs\"],\"Resource\":\"*\"},\
{\"Sid\":\"PassSageMakerTrainingRole\",\"Effect\":\"Allow\",\"Action\":[\"iam:PassRole\"],\"Resource\":\"arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_TRAINING_ROLE_NAME}\",\"Condition\":{\"StringEquals\":{\"iam:PassedToService\":\"sagemaker.amazonaws.com\"}}},\
{\"Sid\":\"PromoteFromGovernanceUI\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:UpdateModelPackage\"],\"Resource\":[\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:model-package/*artf-*/*\"]},\
{\"Sid\":\"TritonModelRepoReadWrite\",\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${MODEL_BUCKET}\",\"arn:aws:s3:::${MODEL_BUCKET}/triton-models/*\"]},\
{\"Sid\":\"KmsForDynamoDb\",\"Effect\":\"Allow\",\"Action\":[\"kms:Decrypt\",\"kms:GenerateDataKey\",\"kms:DescribeKey\"],\"Resource\":\"*\",\"Condition\":{\"StringEquals\":{\"kms:ViaService\":[\"dynamodb.${AWS_REGION}.amazonaws.com\"]}}}\
]}"

if aws iam get-policy --policy-arn "${CLOSED_LOOP_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${CLOSED_LOOP_POLICY_ARN}" \
    --policy-document "${CLOSED_LOOP_POLICY_DOC}" \
    --set-as-default 2>/dev/null || true
else
  aws iam create-policy \
    --policy-name "${CLOSED_LOOP_POLICY_NAME}" \
    --policy-document "${CLOSED_LOOP_POLICY_DOC}" >/dev/null
fi

if [[ -n "${SERVICES_NG_ROLE}" && "${SERVICES_NG_ROLE}" != "None" ]]; then
  aws iam attach-role-policy --role-name "${SERVICES_NG_ROLE}" --policy-arn "${CLOSED_LOOP_POLICY_ARN}" 2>/dev/null || true
  log "  Attached closed-loop policy to node role: ${SERVICES_NG_ROLE}"
fi

if [[ "${START_AT}" -le 2 ]]; then
  _phase2_elapsed=$(( $(date +%s) - _PHASE2_START ))
  ok "Images and EKS cluster ready ($(( _phase2_elapsed / 60 ))m $(( _phase2_elapsed % 60 ))s)"
fi

# =========================================================================
# Step 8: Apply Kubernetes manifests (idempotent — kubectl apply)
# =========================================================================
if [[ "${START_AT}" -le 3 ]]; then
phase 3 "Deploying workloads"
log "Step 8: Applying Kubernetes manifests"

# --- Provision Cognito BEFORE applying manifests so the orchestrator gets the real pool ID ---
log "  Provisioning Cognito User Pool (needed for orchestrator auth)..."
${PYTHON} "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
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

# --- Cognito Identity Pool + authenticated role (provisioned by deploy_cognito.py) ---
# deploy_cognito.py creates the Identity Pool and the authenticated IAM role. The
# least-privilege InvokeAgentRuntime grant (scoped to the specific runtime ARNs) is
# applied in Step 11 via `deploy_cognito.py --action grant-agent-invoke`, once the
# ARNs exist. The orchestrator is deliberately NOT granted agent-invoke — the browser
# invokes the closed-loop agents directly via SigV4 (see FR-6 / agentcore-verification-findings.md).
IDENTITY_POOL_ID="$(jq -r '.IdentityPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
ID_POOL_AUTH_ROLE_NAME="$(jq -r '.AuthRoleName // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
log "  Identity Pool: ${IDENTITY_POOL_ID:-<none>}  Auth role: ${ID_POOL_AUTH_ROLE_NAME:-<none>}"

# NOTE: The closed-loop agent runtime ARNs are resolved and wired into the frontend
# build in Step 11 (after the agents are deployed by deploy_closed_loop.sh). The
# orchestrator is NOT wired to any agent runtime — the browser invokes them directly
# via SigV4 (FR-6). See Step 11 for the ARN resolution + Identity-Pool invoke grant.

# All workloads (Triton, agent containers, orchestrator) deploy into the
# `default` namespace, matching the triton-sa IRSA service account. Keeping
# them co-located lets the orchestrator reach backends by bare service name.

# Steady state needs ONE GPU node (Triton only). The Model Optimizer no longer runs
# as an always-on GPU Deployment — base engines come from a one-shot bootstrap Job
# and promotions from on-demand Jobs. The node group can still burst to --maxGPUs
# for those transient Jobs / NeMo retraining.
GPU_DESIRED=1

# The bootstrap Job (Step 8a) needs its OWN GPU while it runs. On a first-time
# deploy Triton isn't up yet, so 1 GPU node would be enough — but on a REDEPLOY,
# Triton is typically already running and occupying the sole GPU node, which
# leaves the bootstrap Job stuck Pending (0/N nodes available: Insufficient
# nvidia.com/gpu) until its 15-minute wait below times out. Scale to 2 GPU nodes
# BEFORE running the bootstrap Job so it always has room alongside any
# already-running Triton, then scale back down to GPU_DESIRED (1) once the Job
# finishes, since steady state only needs Triton's GPU.
BOOTSTRAP_GPU_DESIRED=$(( GPU_DESIRED + 1 ))
if [[ "${BOOTSTRAP_GPU_DESIRED}" -gt "${MAX_GPUS}" ]]; then
  BOOTSTRAP_GPU_DESIRED="${MAX_GPUS}"
  if [[ "${BOOTSTRAP_GPU_DESIRED}" -le "${GPU_DESIRED}" ]]; then
    warn "  --maxGPUs=${MAX_GPUS} leaves no headroom for the bootstrap Job to run alongside an already-running Triton; it may have to wait for a GPU to free up."
  fi
fi

log "  Scaling GPU node group 'gpu-inference' to desired ${BOOTSTRAP_GPU_DESIRED} (headroom for the one-shot bootstrap Job)"
aws eks update-nodegroup-config --cluster-name "${CLUSTER_NAME}" --nodegroup-name gpu-inference \
  --scaling-config "minSize=1,maxSize=${MAX_GPUS},desiredSize=${BOOTSTRAP_GPU_DESIRED}" \
  --region "${AWS_REGION}" >/dev/null 2>&1 || warn "  Could not scale gpu-inference node group (continuing)"

# Wait for GPU node(s) to be available before applying Triton + optimizer deployments
log "  Checking for GPU node availability..."
GPU_WAIT_TIMEOUT=420
GPU_WAIT_INTERVAL=15
GPU_ELAPSED=0
while true; do
  GPU_NODES="$(kubectl get nodes -l nvidia.com/gpu=present --no-headers 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${GPU_NODES}" -ge "${BOOTSTRAP_GPU_DESIRED}" ]]; then
    log "  GPU node(s) available (${GPU_NODES} found, need ${BOOTSTRAP_GPU_DESIRED})"
    break
  fi
  if [[ "${GPU_ELAPSED}" -ge "${GPU_WAIT_TIMEOUT}" ]]; then
    warn "Only ${GPU_NODES}/${BOOTSTRAP_GPU_DESIRED} GPU nodes after ${GPU_WAIT_TIMEOUT}s — the bootstrap Job and/or Triton may stay pending."
    warn "Check node group: eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${AWS_REGION}"
    break
  fi
  log "  Waiting for GPU node(s) to scale up... (${GPU_NODES}/${BOOTSTRAP_GPU_DESIRED}, ${GPU_ELAPSED}s / ${GPU_WAIT_TIMEOUT}s)"
  sleep "${GPU_WAIT_INTERVAL}"
  GPU_ELAPSED=$((GPU_ELAPSED + GPU_WAIT_INTERVAL))
done

# -------------------------------------------------------------------------
# Images are guaranteed ready (Step 4 is synchronous). Apply manifests.
# -------------------------------------------------------------------------

# Step 8a: build base TensorRT engines via a one-shot Job. Runs on its own GPU
# node (see the scale-up above), independent of whether Triton is already
# running on another node, and EXITS when done — no always-on optimizer pod.
log "  Step 8a: starting base TensorRT engine build (one-shot optimizer bootstrap Job)..."
BOOTSTRAP_MANIFEST="/tmp/${CLUSTER_NAME}-model-optimizer-bootstrap-job.yaml"
sed -e "s|__STACK_NAME__|${STACK_NAME}|g" \
    -e "s|__REGION__|${AWS_REGION}|g" \
    -e "s|__REGISTRY__|${REGISTRY}|g" \
    -e "s|__IMAGE_TAG__|${IMAGE_TAG}|g" \
    -e "s|__MODEL_BUCKET__|${MODEL_BUCKET}|g" \
    -e "s|__OPTIMIZER_ROLE_ARN__|${OPTIMIZER_ROLE_ARN}|g" \
    "${SCRIPT_DIR}/eks/model-optimizer-bootstrap-job.yaml" > "${BOOTSTRAP_MANIFEST}"
# Jobs are immutable — delete any prior run before re-applying (idempotent redeploy).
kubectl delete job model-optimizer-bootstrap --ignore-not-found >/dev/null 2>&1 || true
kubectl apply -f "${BOOTSTRAP_MANIFEST}"

# FR-9: do NOT block deploy.sh on the bootstrap Job's completion. Nothing
# downstream (frontend, Cognito, AgentCore, closed-loop) depends on the
# compiled engines existing yet — Triton runs in --model-control-mode=poll
# against a fixed S3 path (deployment/eks/triton-deployment.yaml) and will
# auto-load the engines whenever the Job finishes, independent of deploy.sh's
# own process lifetime. A detached background watcher performs the real
# `kubectl wait` and, only on genuine completion, triggers the GPU
# scale-down that used to happen synchronously right here (R-2) — writing
# its result to a fixed, checkable path.
BOOTSTRAP_STATUS_FILE="${SCRIPT_DIR}/.bootstrap-status.json"
rm -f "${BOOTSTRAP_STATUS_FILE}"
(
  if kubectl wait --for=condition=complete job/model-optimizer-bootstrap --timeout=900s >/tmp/"${CLUSTER_NAME}"-bootstrap-wait.log 2>&1; then
    if [[ "${BOOTSTRAP_GPU_DESIRED}" -ne "${GPU_DESIRED}" ]]; then
      aws eks update-nodegroup-config --cluster-name "${CLUSTER_NAME}" --nodegroup-name gpu-inference \
        --scaling-config "minSize=1,maxSize=${MAX_GPUS},desiredSize=${GPU_DESIRED}" \
        --region "${AWS_REGION}" >/dev/null 2>&1
    fi
    printf '{"status":"complete","finishedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${BOOTSTRAP_STATUS_FILE}"
  else
    printf '{"status":"timeout_or_failed","checkedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${BOOTSTRAP_STATUS_FILE}"
  fi
) &
disown
ok "Model-optimizer bootstrap Job started (compiling TensorRT engines in background, up to 15 min)"
log "  Triton auto-loads engines from s3://${MODEL_BUCKET}/triton-models/ once ready — no restart needed."
log "  Check status any time: kubectl get job model-optimizer-bootstrap  |  cat ${BOOTSTRAP_STATUS_FILE}"

# NOTE: the GPU node group is intentionally left at BOOTSTRAP_GPU_DESIRED here
# (the scale-down to GPU_DESIRED now happens in the background watcher above,
# once the bootstrap Job genuinely completes — FR-9/R-2). This costs one extra
# GPU node for a bit longer than before, in exchange for not blocking the rest
# of the deploy on the bootstrap Job.

# Step 8b: apply Triton + ARTF containers + orchestrator. The Model Optimizer is
# NOT here anymore (on-demand Jobs only). ARTF model containers default to the CPU
# node group (role=services) since they call Triton over the network and hold no
# GPU; --artf-node-role=inference co-locates them on the GPU node instead.
for manifest in triton-deployment.yaml triton-internal-nlb.yaml artf-containers-deployment.yaml orchestrator-deployment.yaml triton-hpa.yaml; do
  PROCESSED="/tmp/${CLUSTER_NAME}-${manifest}"
  sed -e "s|__STACK_NAME__|${STACK_NAME}|g" \
      -e "s|__REGION__|${AWS_REGION}|g" \
      -e "s|__REGISTRY__|${REGISTRY}|g" \
      -e "s|__IMAGE_TAG__|${IMAGE_TAG}|g" \
      -e "s|__MODEL_BUCKET__|${MODEL_BUCKET}|g" \
      -e "s|__TRITON_ROLE_ARN__|${TRITON_ROLE_ARN}|g" \
      -e "s|__OPTIMIZER_ROLE_ARN__|${OPTIMIZER_ROLE_ARN}|g" \
      -e "s|__ARTF_NODE_ROLE__|${ARTF_NODE_ROLE}|g" \
      -e "s|__COGNITO_USER_POOL_ID__|${COGNITO_USER_POOL_ID:-}|g" \
      -e "s|__PARAMETER_STORE_TABLE__|${STACK_PREFIX:+${STACK_PREFIX}-}parameter-store|g" \
      -e "s|__AUDIT_TRAIL_TABLE__|${STACK_PREFIX:+${STACK_PREFIX}-}audit-trail|g" \
      -e "s|__DLRM_MODEL_GROUP__|${STACK_PREFIX:+${STACK_PREFIX}-}artf-dlrm-bid-shader|g" \
      -e "s|__NCF_MODEL_GROUP__|${STACK_PREFIX:+${STACK_PREFIX}-}artf-ncf-deal-manager|g" \
      -e "s|__SAGEMAKER_TRAINING_ROLE_ARN__|arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_TRAINING_ROLE_NAME}|g" \
      -e "s|__TRAINING_DATA_BUCKET__|${TRAINING_DATA_BUCKET}|g" \
      -e "s|__TRAINING_IMAGE_REGISTRY__|${REGISTRY}|g" \
      -e "s|__FEEDBACK_STREAM_NAME__|${FEEDBACK_STREAM_NAME}|g" \
      -e "s|__DEAL_YIELD_FEEDBACK_STREAM_NAME__|${DEAL_YIELD_FEEDBACK_STREAM_NAME}|g" \
      "${SCRIPT_DIR}/eks/${manifest}" > "${PROCESSED}"
  kubectl apply -f "${PROCESSED}"
done

log "  Waiting for Triton Inference Server..."
kubectl rollout status deployment/triton-inference-server --timeout=300s || \
  warn "Triton not ready yet — check GPU node availability with: kubectl get nodes -l nvidia.com/gpu=present"

log "  Waiting for orchestrator..."
kubectl rollout status deployment/orchestrator --timeout=120s || true

# NOTE: the orchestrator is intentionally NOT patched with any agent runtime ARN.
# It must never invoke the closed-loop agents (FR-6) — the browser invokes them
# directly via SigV4. Agent ARNs are wired into the frontend build in Step 11.

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

ok "Kubernetes workloads deployed"
fi # START_AT <= 3 (Phase 3)

# =========================================================================
# Step 9: Deploy frontend to S3 + CloudFront (React UI)
# =========================================================================
if [[ "${START_AT}" -le 4 ]]; then
phase 4 "Setting up access"
log "Step 9: Deploying frontend"

# Cognito was already provisioned in Step 8 (before manifest apply).
# Re-read outputs in case they're needed for frontend build.
COGNITO_OUTPUTS="${SCRIPT_DIR}/.cognito-outputs.json"
COGNITO_USER_POOL_ID="$(jq -r '.UserPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
COGNITO_CLIENT_ID="$(jq -r '.ClientId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"

# --- Step 9c: Write frontend build config (auth). The agent runtime ARNs are not
# known until the agents deploy in Step 11 (--with-retraining), which re-writes this
# file with the resolved ARNs and rebuilds. Empty ARNs here → the UI honestly shows
# "agent not deployed" until Step 11 completes. This file is generated (git-ignored),
# never committed with real values. ---
REACT_ENV="${SCRIPT_DIR}/../source/frontend-react/.env.production"
cat > "${REACT_ENV}" <<EOF
VITE_COGNITO_USER_POOL_ID=${COGNITO_USER_POOL_ID}
VITE_COGNITO_CLIENT_ID=${COGNITO_CLIENT_ID}
VITE_COGNITO_REGION=${AWS_REGION}
VITE_IDENTITY_POOL_ID=${IDENTITY_POOL_ID}
VITE_ADAPTIVE_BIDDING_RUNTIME_ARN=
VITE_GOVERNANCE_RUNTIME_ARN=
EOF

# React UI distribution
${PYTHON} "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
  --action deploy \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --orchestrator-url "http://${NLB_DNS}"

PRIMARY_OUTPUTS="${SCRIPT_DIR}/.frontend-outputs.json"
CF_DOMAIN="$(jq -r '.CloudFrontDomain // empty' "${PRIMARY_OUTPUTS}" 2>/dev/null || echo '')"

# Update Cognito callback URLs now that we know the CF domain
if [[ -n "${CF_DOMAIN}" && -n "${COGNITO_USER_POOL_ID}" ]]; then
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
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
      TEMP_PASS="$(${PYTHON} - <<'PY'
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

ok "Frontend and Cognito ready"
fi # START_AT <= 4 (Phase 4)

# =========================================================================
# Step 10: Deploy AgentCore MCP runtime
# =========================================================================
phase 5 "Registering agents"
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
    # Bid Shading Agent needs CloudWatch metrics + DynamoDB for parameter store
    aws iam put-role-policy --role-name "${ROLE_NAME}" \
      --policy-name BidShadingAgentPermissions \
      --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"CloudWatch\",\"Effect\":\"Allow\",\"Action\":[\"cloudwatch:GetMetricData\",\"cloudwatch:GetMetricStatistics\",\"cloudwatch:PutMetricData\"],\"Resource\":\"*\"},{\"Sid\":\"DynamoDB\",\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:PutItem\",\"dynamodb:Query\",\"dynamodb:UpdateItem\"],\"Resource\":[\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/parameter-store\",\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/audit-trail\",\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/*-parameter-store\",\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/*-audit-trail\"]}]}"
    sleep 10
  fi

  AC_RUNTIME_NAME="$(echo "${STACK_NAME}_mcp" | tr '-' '_')"
  AC_IMAGE="${REGISTRY}/${STACK_NAME}-agentcore:${IMAGE_TAG}"
  ${PYTHON} "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
    --action deploy \
    --runtime-name "${AC_RUNTIME_NAME}" \
    --role-arn "${ROLE_ARN}" \
    --container-uri "${AC_IMAGE}" \
    --region "${AWS_REGION}"
else
  warn "Skipping AgentCore deployment (--skip-agentcore)"
fi

# =========================================================================
# Step 11 (optional): Deploy closed-loop retraining infrastructure
# =========================================================================
if [[ "${WITH_RETRAINING}" -eq 1 ]]; then
  log ""
  log "Step 11: Deploying closed-loop retraining infrastructure (--with-retraining)"
  log ""

  # Resolve VPC, subnets, and node role from the EKS cluster for the closed-loop stack
  CL_VPC_ID="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.vpcId' --output text 2>/dev/null || echo '')"
  CL_SUBNET_IDS="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.subnetIds' --output text 2>/dev/null | tr '\t' ',' || echo '')"
  # EKS_NODE_ROLE must be the full ARN (used in KMS key policies)
  CL_NODE_ROLE_ARN="$(aws eks describe-nodegroup --cluster-name "${CLUSTER_NAME}" --nodegroup-name "cpu-services" --region "${AWS_REGION}" \
    --query 'nodegroup.nodeRole' --output text 2>/dev/null || echo '')"

  # Pass through the prefix and skip flags
  CLOSED_LOOP_ARGS=""
  if [[ -n "${STACK_PREFIX}" ]]; then
    CLOSED_LOOP_ARGS="--prefix ${STACK_PREFIX}"
  fi
  if [[ "${SKIP_AGENTCORE}" -eq 1 ]]; then
    CLOSED_LOOP_ARGS="${CLOSED_LOOP_ARGS} --skip-agentcore"
  fi
  if [[ "${LOCAL_BUILD}" -eq 1 ]]; then
    CLOSED_LOOP_ARGS="${CLOSED_LOOP_ARGS} --local-build"
  fi
  if [[ -n "${NGC_SECRET}" ]]; then
    CLOSED_LOOP_ARGS="${CLOSED_LOOP_ARGS} --ngc-secret ${NGC_SECRET}"
  fi
  if [[ -n "${NGC_KEY}" ]]; then
    CLOSED_LOOP_ARGS="${CLOSED_LOOP_ARGS} --ngc-key ${NGC_KEY}"
  fi

  VPC_ID="${CL_VPC_ID}" \
  SUBNET_IDS="${CL_SUBNET_IDS}" \
  EKS_NODE_ROLE="${CL_NODE_ROLE_ARN}" \
  MODEL_BUCKET="${MODEL_BUCKET}" \
  CLUSTER_NAME="${CLUSTER_NAME}" \
  BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID}" \
  ADAPTIVE_BIDDING_MODEL_ID="${ADAPTIVE_BIDDING_MODEL_ID}" \
  GOVERNANCE_MODEL_ID="${GOVERNANCE_MODEL_ID}" \
  "${SCRIPT_DIR}/deploy_closed_loop.sh" ${CLOSED_LOOP_ARGS} || {
    warn "Closed-loop deployment returned non-zero. Check output above for errors."
    warn "The core EKS deployment succeeded — retraining infra may need manual intervention."
  }

  # deploy_closed_loop.sh (Step 7) grants the Identity-Pool auth role scoped
  # InvokeAgentRuntime and rebuilds+redeploys the UI with the now-known agent runtime
  # ARNs baked in — it owns that step because it is where the ARNs first exist. We do
  # NOT repeat it here (a second frontend build would be wasteful and could race the
  # first). The orchestrator is never wired to the agents — the browser invokes them
  # directly via SigV4 (FR-6).

  log "  Closed-loop retraining infrastructure deployed."
else
  log ""
  log "  Skipping closed-loop retraining (pass --with-retraining to enable)."
  log "  This includes: NeMo-RL training container, SageMaker Model Registry,"
  log "  Glue ETL, EventBridge scheduled retraining, and AgentCore agents."
fi
ok "Agents registered"

# =========================================================================
# Summary — always printed regardless of --verbose (FR-6: "print ONLY what
# the user needs to get started"), so this block uses say() (always-visible)
# rather than log() (verbose-gated).
# =========================================================================
say ""
say "========================================================="
say "  Accelerator-optimized Agentic Bidding — Deployed (EKS + Triton)"
say "========================================================="
say ""
say "  Frontend:    https://${CF_DOMAIN:-'(pending)'}"
say "  Endpoint:    http://${NLB_DNS:-'(pending)'}/v1/mutations"
say "  Health:      http://${NLB_DNS:-'(pending)'}/health/ready"
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
say "  AgentCore:   See runtime ARN above"
fi
say ""
say "  Demo Login (Cognito):"
case "${DEMO_LOGIN_STATUS:-no-auth}" in
  created)
    say "    Username:  ${DEMO_USER_EMAIL}"
    say "    Password:  ${DEMO_USER_TEMP_PASSWORD}"
    say "    Note:      Temporary password, shown only here — you'll set a permanent one on first login."
    ;;
  existing)
    say "    Username:  ${DEMO_USER_EMAIL}"
    say "    Password:  (existing user — password unchanged, not displayed)"
    say "    Reset:     aws cognito-idp admin-set-user-password \\"
    say "                 --user-pool-id ${COGNITO_USER_POOL_ID} --username ${DEMO_USER_EMAIL} \\"
    say "                 --password '<new-password>' --permanent --region ${AWS_REGION}"
    ;;
  *)
    say "    (Cognito auth not configured — orchestrator auth is disabled)"
    ;;
esac
say ""
say "  NVIDIA Triton Inference Server:"
say "    EKS Cluster:   ${CLUSTER_NAME}"
say "    Triton Image:  nvcr.io/nvidia/tritonserver:24.08-py3"
say "    GPU:           NVIDIA A10G (g5.xlarge)"
say "    Models:        s3://${MODEL_BUCKET}/triton-models/"
say "    Backend:       ONNX Runtime + CUDA Execution Provider"
say ""
say "  Models served by Triton:"
say "    dlrm_bid_shader            — DLRM (NVIDIA DeepLearningExamples)"
say "    ncf_deal_manager           — NeuMF (NVIDIA DeepLearningExamples)"
say ""
say "  Containers (via orchestrator):"
say "    Bid Pricer              — BID_SHADE"
say "    Audience Activator      — ACTIVATE_SEGMENTS (rule-based, no Triton)"
say "    Deal Scorer             — ACTIVATE_DEALS / SUPPRESS_DEALS"
say "    Signals Enricher        — ADD_METRICS"
say ""
# FR-9/NFR-2: honest status — do not claim the models are ready to serve
# unless the background bootstrap watcher actually observed completion.
# Check the fixed path directly (not just the in-run variable) so a
# --start-at run that skips Phase 3 still reports a real prior status
# instead of a misleading "pending".
BOOTSTRAP_STATUS_FILE="${BOOTSTRAP_STATUS_FILE:-${SCRIPT_DIR}/.bootstrap-status.json}"
say "  Model-optimizer bootstrap (TensorRT engine build):"
if [[ -f "${BOOTSTRAP_STATUS_FILE}" ]]; then
  _bootstrap_status="$(jq -r '.status // "unknown"' "${BOOTSTRAP_STATUS_FILE}" 2>/dev/null || echo unknown)"
else
  _bootstrap_status="pending"
fi
case "${_bootstrap_status}" in
  complete)
    say "    Status:    complete — engines are in S3, Triton has loaded them (or will on its next poll)."
    ;;
  timeout_or_failed)
    say "    Status:    did not complete within 15 min — Triton cannot serve tensorrt_plan models yet."
    say "    Inspect:   kubectl logs job/model-optimizer-bootstrap ; kubectl describe job/model-optimizer-bootstrap"
    ;;
  *)
    say "    Status:    still running in the background (may take up to 15 min from when Phase 3 started)."
    say "    Check:     kubectl get job model-optimizer-bootstrap  |  cat ${BOOTSTRAP_STATUS_FILE}"
    ;;
esac
say ""
say "  Monitoring:"
say "    kubectl port-forward svc/triton-inference-server 8002:8002"
say "    curl localhost:8002/metrics  # Prometheus metrics"
say ""
