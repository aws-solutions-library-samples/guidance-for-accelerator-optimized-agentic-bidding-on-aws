#!/usr/bin/env bash
# =============================================================================
# remote_build.sh — Build container images remotely using AWS CodeBuild
#
# Uploads the source/ directory to CodeBuild and runs the buildspec to build
# and push all container images to ECR. This avoids the need for local Docker
# disk space (especially for the ~20 GB NeMo-RL image).
#
# Usage:
#   ./remote_build.sh --stack-name stg-nvidia-artf-recommenders
#   ./remote_build.sh --stack-name stg-nvidia-artf-recommenders --target nemo
#   ./remote_build.sh --stack-name stg-nvidia-artf-recommenders --target part1 --tag abc1234
#
# Options:
#   --stack-name NAME   Resource stack name (required)
#   --target TARGET     Build target: all|part1|nemo|agents|optimizer (default: all)
#   --tag TAG           Docker image tag (default: git short SHA)
#   --nemo-src-tag TAG  Extra content-hash tag for the NeMo image (src-<hash>)
#   --ngc-secret NAME   Secrets Manager secret name for NGC API key
#   --no-wait           Start build and exit without waiting for completion
#   --region REGION     AWS region (default: $AWS_REGION or us-east-1)
#
# Prerequisites:
#   - The CodeBuild project stack must be deployed first:
#       aws cloudformation deploy --stack-name <name>-codebuild \
#         --template-file deployment/codebuild/codebuild_cfn.yaml \
#         --capabilities CAPABILITY_NAMED_IAM \
#         --parameter-overrides StackName=<name>
#   - For NeMo builds: store your NGC API key in Secrets Manager:
#       aws secretsmanager create-secret --name artf-ngc-api-key \
#         --secret-string "YOUR_NGC_API_KEY"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Job-oriented display names for the 4 ARTF containers — used ONLY for the ECR
# repository name. The "key" tokens passed via --only (dlrm-bid-shader, etc.)
# stay unchanged, matching buildspec.yml's derivation of its own KEY from the
# (also-translated) repo name — see RENAME_MAP.md.
display_name() {
  case "$1" in
    dlrm-bid-shader)             echo "bid-pricer" ;;
    widedeep-segment-activator)  echo "audience-activator" ;;
    ncf-deal-manager)            echo "deal-scorer" ;;
    metrics-enricher)            echo "signals-enricher" ;;
    # The two Yield Optimizer containers were named for their job when the
    # combined deal-yield-manager was split, so their keys already ARE their
    # display names (kept in step with deploy.sh's display_name()).
    yield-optimizer-floor)       echo "yield-optimizer-floor" ;;
    yield-optimizer-margin)      echo "yield-optimizer-margin" ;;
    *)                           echo "$1" ;;
  esac
}

AWS_REGION="${AWS_REGION:-us-east-1}"

STACK_NAME=""
BUILD_TARGET="all"
BUILD_ONLY=""
IMAGE_TAG=""
# Content-hash tag (src-<hash>) for the NeMo training image, supplied by
# deploy_closed_loop.sh. Pushed alongside dlrm/ncf so a later run can tell
# whether the image matches source/training/container/ as it stands now.
NEMO_SRC_TAG=""
NGC_SECRET=""
NGC_KEY=""
NO_WAIT=0

for arg in "$@"; do
  case "${arg}" in
    --stack-name=*) STACK_NAME="${arg#--stack-name=}" ;;
    --stack-name)   ;; # value in next arg
    --target=*)     BUILD_TARGET="${arg#--target=}" ;;
    --target)       ;; # value in next arg
    --only=*)       BUILD_ONLY="${arg#--only=}" ;;
    --only)         ;; # value in next arg (space-separated image keys)
    --tag=*)        IMAGE_TAG="${arg#--tag=}" ;;
    --tag)          ;; # value in next arg
    --nemo-src-tag=*) NEMO_SRC_TAG="${arg#--nemo-src-tag=}" ;;
    --nemo-src-tag) ;; # value in next arg
    --ngc-secret=*) NGC_SECRET="${arg#--ngc-secret=}" ;;
    --ngc-secret)   ;; # value in next arg
    --ngc-key=*)    NGC_KEY="${arg#--ngc-key=}" ;;
    --ngc-key)      ;; # value in next arg
    --no-wait)      NO_WAIT=1 ;;
    --region=*)     AWS_REGION="${arg#--region=}" ;;
    --region)       ;; # value in next arg
    *)
      if [[ "${_PREV_ARG:-}" == "--stack-name" ]]; then STACK_NAME="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--target" ]]; then BUILD_TARGET="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--only" ]]; then BUILD_ONLY="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--tag" ]]; then IMAGE_TAG="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--nemo-src-tag" ]]; then NEMO_SRC_TAG="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--ngc-secret" ]]; then NGC_SECRET="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--ngc-key" ]]; then NGC_KEY="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--region" ]]; then AWS_REGION="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

log()  { printf '\033[0;32m[remote-build]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# Validate inputs
[[ -n "${STACK_NAME}" ]] || fail "--stack-name is required"
[[ "${BUILD_TARGET}" =~ ^(all|part1|nemo|agents|optimizer)$ ]] || fail "--target must be one of: all, part1, nemo, agents, optimizer"

if [[ -z "${IMAGE_TAG}" ]]; then
  IMAGE_TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo latest)"
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "${ACCOUNT_ID}" ]] || fail "Cannot resolve AWS account ID"

# =========================================================================
# If --ngc-key is provided, create or update the Secrets Manager secret
# =========================================================================
if [[ -n "${NGC_KEY}" ]]; then
  NGC_SECRET_NAME="${STACK_NAME}-ngc-api-key"
  log "Storing NGC API key in Secrets Manager: ${NGC_SECRET_NAME}"
  if aws secretsmanager describe-secret --secret-id "${NGC_SECRET_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    aws secretsmanager put-secret-value \
      --secret-id "${NGC_SECRET_NAME}" \
      --secret-string "${NGC_KEY}" \
      --region "${AWS_REGION}" >/dev/null
    log "  Updated existing secret: ${NGC_SECRET_NAME}"
  else
    aws secretsmanager create-secret \
      --name "${NGC_SECRET_NAME}" \
      --secret-string "${NGC_KEY}" \
      --description "NVIDIA NGC API key for pulling NeMo container images" \
      --region "${AWS_REGION}" >/dev/null
    log "  Created secret: ${NGC_SECRET_NAME}"
  fi
  NGC_SECRET="${NGC_SECRET_NAME}"
fi

# Does this build include the NeMo container (the only image that hard-requires NGC)?
_BUILDS_NEMO=0
if [[ -n "${BUILD_ONLY}" ]]; then
  case " ${BUILD_ONLY} " in *" nemo "*|*" nemo-rl-training "*) _BUILDS_NEMO=1 ;; esac
elif [[ "${BUILD_TARGET}" == "all" || "${BUILD_TARGET}" == "nemo" ]]; then
  _BUILDS_NEMO=1
fi
# If neither --ngc-key nor --ngc-secret was passed on THIS invocation, check
# whether a secret already exists from a prior run before prompting — a
# re-run (e.g. deploy.sh --start-at) should not re-prompt for credentials
# that are already stored in Secrets Manager for this stack.
if [[ -z "${NGC_SECRET}" && "${_BUILDS_NEMO}" -eq 1 ]]; then
  _EXISTING_NGC_SECRET="${STACK_NAME}-ngc-api-key"
  if aws secretsmanager describe-secret --secret-id "${_EXISTING_NGC_SECRET}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Found existing NGC secret for this stack: ${_EXISTING_NGC_SECRET} (reusing, no prompt)"
    NGC_SECRET="${_EXISTING_NGC_SECRET}"
  fi
fi
if [[ -z "${NGC_SECRET}" && "${_BUILDS_NEMO}" -eq 1 ]]; then
  # NOTE: the 'optimizer' target builds on the PUBLIC nvcr.io/nvidia/tensorrt image,
  # which needs no NGC auth — so it is intentionally NOT in this prompt condition
  # (prompting would block non-interactive deploys). NGC login still happens in the
  # buildspec if a secret is present, which is harmless.
  warn "No NGC credentials provided. The NeMo build requires access to nvcr.io."
  warn "  Pass --ngc-key YOUR_API_KEY  (creates secret automatically)"
  warn "  Or   --ngc-secret SECRET_NAME (if you already stored it in Secrets Manager)"
  warn ""
  printf '\033[0;33m[warn]\033[0m Continue without NGC credentials? The build will fail if nvcr.io requires auth. [y/N]: ' >&2
  read -r CONTINUE </dev/tty
  if [[ "${CONTINUE}" != "y" && "${CONTINUE}" != "Y" ]]; then
    fail "Aborted. Get your NGC API key from https://ngc.nvidia.com/ (Profile > Generate API Key)"
  fi
fi

if [[ -n "${BUILD_ONLY}" ]]; then
  log "Remote build: only=[${BUILD_ONLY}] stack=${STACK_NAME} tag=${IMAGE_TAG} region=${AWS_REGION}"
else
  log "Remote build: target=${BUILD_TARGET} stack=${STACK_NAME} tag=${IMAGE_TAG} region=${AWS_REGION}"
fi

# =========================================================================
# Step 1: Ensure CodeBuild project exists (deploy/update CFN stack)
# =========================================================================
CB_STACK_NAME="${STACK_NAME}-codebuild"
CB_PROJECT_X86="${STACK_NAME}-image-builder"
CB_PROJECT_ARM="${STACK_NAME}-arm-image-builder"

# Always run deploy (--no-fail-on-empty-changeset handles the no-op case).
# This ensures that if NGC_SECRET is added after the initial deploy, the
# IAM policy gets updated to include Secrets Manager access.
log "Ensuring CodeBuild project stack: ${CB_STACK_NAME}"
aws cloudformation deploy \
  --stack-name "${CB_STACK_NAME}" \
  --template-file "${SCRIPT_DIR}/codebuild_cfn.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    StackName="${STACK_NAME}" \
    NgcApiKeySecret="${NGC_SECRET}" \
  --region "${AWS_REGION}" \
  --no-fail-on-empty-changeset || \
  fail "Failed to deploy CodeBuild stack. Check CloudFormation events for ${CB_STACK_NAME}"
log "CodeBuild stack ready"

# =========================================================================
# Step 2: Ensure ECR repositories exist (build will push to them)
# =========================================================================
log "Ensuring ECR repositories exist..."
REPOS=(
  "${STACK_NAME}-$(display_name dlrm-bid-shader)"
  "${STACK_NAME}-$(display_name widedeep-segment-activator)"
  "${STACK_NAME}-$(display_name ncf-deal-manager)"
  "${STACK_NAME}-$(display_name metrics-enricher)"
  "${STACK_NAME}-$(display_name yield-optimizer-floor)"
  "${STACK_NAME}-$(display_name yield-optimizer-margin)"
  "${STACK_NAME}-orchestrator"
  "${STACK_NAME}-agentcore"
)
for repo in "${REPOS[@]}"; do
  aws ecr describe-repositories --repository-names "${repo}" --region "${AWS_REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${repo}" --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE >/dev/null 2>&1 || true
done

# =========================================================================
# Step 3: Package source directory into a zip for CodeBuild
# =========================================================================
log "Packaging source directory for upload..."
SOURCE_ZIP="/tmp/${STACK_NAME}-source-$(date +%s).zip"

# Create a zip of the source directory (build context for all images)
(cd "${REPO_ROOT}" && zip -qr "${SOURCE_ZIP}" source/ \
  -x "source/frontend-react/node_modules/*" \
  -x "source/.pytest_cache/*" \
  -x "source/**/__pycache__/*" \
  -x "source/**/.DS_Store" \
  -x "source/tests/*")

# Also include the buildspec
zip -qj "${SOURCE_ZIP}" "${SCRIPT_DIR}/buildspec.yml"

SOURCE_SIZE=$(du -h "${SOURCE_ZIP}" | cut -f1)
log "Source package: ${SOURCE_ZIP} (${SOURCE_SIZE})"

# =========================================================================
# Step 4: Upload source to S3 (CodeBuild needs it in a bucket)
# =========================================================================
SOURCE_BUCKET="${STACK_NAME}-codebuild-source-${ACCOUNT_ID}"
if ! aws s3api head-bucket --bucket "${SOURCE_BUCKET}" 2>/dev/null; then
  log "Creating source bucket: ${SOURCE_BUCKET}"
  if [[ "${AWS_REGION}" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "${SOURCE_BUCKET}" --region "${AWS_REGION}" >/dev/null
  else
    aws s3api create-bucket --bucket "${SOURCE_BUCKET}" --region "${AWS_REGION}" \
      --create-bucket-configuration LocationConstraint="${AWS_REGION}" >/dev/null
  fi
fi

S3_KEY="builds/${IMAGE_TAG}/source.zip"
log "Uploading source to s3://${SOURCE_BUCKET}/${S3_KEY}..."
aws s3 cp "${SOURCE_ZIP}" "s3://${SOURCE_BUCKET}/${S3_KEY}" --region "${AWS_REGION}" || \
  fail "Failed to upload source to S3. Check that your IAM user has s3:PutObject on ${SOURCE_BUCKET}"
rm -f "${SOURCE_ZIP}"

# =========================================================================
# Step 5: Determine which CodeBuild project(s) to trigger
# =========================================================================
# For ARM-only builds (agents with --target agents), use the ARM project
# For everything else, use x86 project (it does cross-compile via buildx for ARM when needed)
# Strategy: use x86 project for all/part1/nemo; use ARM for native ARM builds
#
# Actually for simplicity and reliability, we use the x86 project for all builds.
# The x86 project uses buildx to cross-compile ARM images (--push instead of --load).
# This avoids needing to coordinate two builds.

CB_PROJECT="${CB_PROJECT_X86}"

# Build environment overrides
ENV_OVERRIDES="[
  {\"name\":\"BUILD_TARGET\",\"value\":\"${BUILD_TARGET}\",\"type\":\"PLAINTEXT\"},
  {\"name\":\"BUILD_ONLY\",\"value\":\"${BUILD_ONLY}\",\"type\":\"PLAINTEXT\"},
  {\"name\":\"IMAGE_TAG\",\"value\":\"${IMAGE_TAG}\",\"type\":\"PLAINTEXT\"},
  {\"name\":\"STACK_NAME\",\"value\":\"${STACK_NAME}\",\"type\":\"PLAINTEXT\"},
  {\"name\":\"AWS_ACCOUNT_ID\",\"value\":\"${ACCOUNT_ID}\",\"type\":\"PLAINTEXT\"},
  {\"name\":\"AWS_DEFAULT_REGION\",\"value\":\"${AWS_REGION}\",\"type\":\"PLAINTEXT\"},
  {\"name\":\"NEMO_SRC_TAG\",\"value\":\"${NEMO_SRC_TAG}\",\"type\":\"PLAINTEXT\"}
]"

if [[ -n "${NGC_SECRET}" ]]; then
  ENV_OVERRIDES="$(echo "${ENV_OVERRIDES}" | sed 's/]$//' ),{\"name\":\"NGC_API_KEY_SECRET\",\"value\":\"${NGC_SECRET}\",\"type\":\"PLAINTEXT\"}]"
fi

# =========================================================================
# Step 6: Check for in-progress builds (only for async/--no-wait calls)
# =========================================================================
# For synchronous calls (Part 1 builds), CodeBuild handles concurrency fine
# and we just start a new build. For async calls (NeMo), the user might
# accidentally re-trigger a long build, so we prompt.
if [[ "${NO_WAIT}" -eq 1 ]]; then
EXISTING_BUILD=$(aws codebuild list-builds-for-project \
  --project-name "${CB_PROJECT}" \
  --region "${AWS_REGION}" \
  --query 'ids[0]' --output text 2>/dev/null || echo "None")

if [[ "${EXISTING_BUILD}" != "None" && -n "${EXISTING_BUILD}" ]]; then
  EXISTING_STATUS=$(aws codebuild batch-get-builds \
    --ids "${EXISTING_BUILD}" \
    --region "${AWS_REGION}" \
    --query 'builds[0].buildStatus' --output text 2>/dev/null || echo "UNKNOWN")

  if [[ "${EXISTING_STATUS}" == "IN_PROGRESS" ]]; then
    EXISTING_PHASE=$(aws codebuild batch-get-builds \
      --ids "${EXISTING_BUILD}" \
      --region "${AWS_REGION}" \
      --query 'builds[0].currentPhase' --output text 2>/dev/null || echo "UNKNOWN")
    EXISTING_START=$(aws codebuild batch-get-builds \
      --ids "${EXISTING_BUILD}" \
      --region "${AWS_REGION}" \
      --query 'builds[0].startTime' --output text 2>/dev/null || echo "")

    warn "An existing build is already IN_PROGRESS for this project:"
    warn "  Build ID: ${EXISTING_BUILD}"
    warn "  Phase:    ${EXISTING_PHASE}"
    warn "  Started:  ${EXISTING_START}"
    warn "  Console:  https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${EXISTING_BUILD}?region=${AWS_REGION}"
    warn ""
    printf '\033[0;33m[warn]\033[0m Do you want to (w)ait for the existing build, (n)ew build, or (q)uit? [w/n/q]: ' >&2
    read -r CHOICE </dev/tty
    case "${CHOICE}" in
      w|W)
        log "Waiting for existing build: ${EXISTING_BUILD}"
        BUILD_ID="${EXISTING_BUILD}"
        # Skip starting a new build — jump straight to the wait loop
        SKIP_START=1
        ;;
      n|N)
        log "Starting a new build (existing build will continue in the background)"
        SKIP_START=0
        ;;
      *)
        log "Exiting. The existing build continues remotely."
        log "  Monitor at: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${EXISTING_BUILD}?region=${AWS_REGION}"
        exit 0
        ;;
    esac
  fi
fi
fi  # NO_WAIT in-progress check

# =========================================================================
# Step 7: Start the CodeBuild build (unless waiting for existing)
# =========================================================================
if [[ "${SKIP_START:-0}" -ne 1 ]]; then
  log "Starting CodeBuild build: project=${CB_PROJECT} target=${BUILD_TARGET}"

  BUILD_ID=$(aws codebuild start-build \
    --project-name "${CB_PROJECT}" \
    --source-type-override S3 \
    --source-location-override "${SOURCE_BUCKET}/${S3_KEY}" \
    --buildspec-override "buildspec.yml" \
    --environment-variables-override "${ENV_OVERRIDES}" \
    --region "${AWS_REGION}" \
    --query 'build.id' --output text)

  log "Build started: ${BUILD_ID}"
  log "Console: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${BUILD_ID}?region=${AWS_REGION}"
fi

if [[ "${NO_WAIT}" -eq 1 ]]; then
  log "Exiting (--no-wait). Monitor at the console URL above."
  echo "${BUILD_ID}"
  exit 0
fi

# =========================================================================
# Step 8: Wait for build completion (stream status)
# =========================================================================
log "Waiting for build to complete (Ctrl+C to detach; build continues remotely)..."

POLL_INTERVAL=15
POLL_START=$(date +%s)
while true; do
  BUILD_STATUS=$(aws codebuild batch-get-builds \
    --ids "${BUILD_ID}" \
    --region "${AWS_REGION}" \
    --query 'builds[0].buildStatus' --output text)

  PHASE=$(aws codebuild batch-get-builds \
    --ids "${BUILD_ID}" \
    --region "${AWS_REGION}" \
    --query 'builds[0].currentPhase' --output text 2>/dev/null || echo "UNKNOWN")

  ELAPSED=$(( $(date +%s) - POLL_START ))

  case "${BUILD_STATUS}" in
    SUCCEEDED)
      log "Build SUCCEEDED"
      break
      ;;
    FAILED|FAULT|TIMED_OUT|STOPPED)
      fail "Build ${BUILD_STATUS}. Check logs: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${BUILD_ID}?region=${AWS_REGION}"
      ;;
    IN_PROGRESS)
      printf '\r\033[0;32m[remote-build]\033[0m Status: IN_PROGRESS  Phase: %-20s  Elapsed: %dm%02ds\n' "${PHASE}" "$(( ELAPSED / 60 ))" "$(( ELAPSED % 60 ))"
      sleep "${POLL_INTERVAL}"
      ;;
    *)
      printf '\r\033[0;32m[remote-build]\033[0m Status: %-15s Phase: %-20s  Elapsed: %dm%02ds\n' "${BUILD_STATUS}" "${PHASE}" "$(( ELAPSED / 60 ))" "$(( ELAPSED % 60 ))"
      sleep "${POLL_INTERVAL}"
      ;;
  esac
done

# Print final build duration. The `COMPLETED` phase is a terminal sentinel
# CodeBuild appends to mark "build is done" — it never carries a
# durationInSeconds value (always null), so querying it directly always
# yielded the literal text "None" here. Compute the real duration instead
# from startTime/endTime (both are populated once a build finishes).
BUILD_TIMES=$(aws codebuild batch-get-builds \
  --ids "${BUILD_ID}" \
  --region "${AWS_REGION}" \
  --query 'builds[0].[startTime,endTime]' --output text 2>/dev/null || echo "")
START_TS=$(echo "${BUILD_TIMES}" | awk '{print $1}')
END_TS=$(echo "${BUILD_TIMES}" | awk '{print $2}')
if [[ -n "${START_TS}" && -n "${END_TS}" && "${START_TS}" != "None" && "${END_TS}" != "None" ]]; then
  DURATION=$(awk -v s="${START_TS}" -v e="${END_TS}" 'BEGIN{printf "%d", e-s}')
else
  DURATION="unknown"
fi
log "Build duration: ${DURATION}s"

# Cleanup source from S3 (optional, keep for debugging)
# aws s3 rm "s3://${SOURCE_BUCKET}/${S3_KEY}" --region "${AWS_REGION}"
