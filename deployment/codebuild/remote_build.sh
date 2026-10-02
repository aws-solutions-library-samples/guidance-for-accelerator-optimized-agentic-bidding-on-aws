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
#   --profile NAME      AWS CLI profile (default: $AWS_PROFILE, else "default");
#                       exported as AWS_PROFILE so every aws call here uses it
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
    # The template container was named for its job from the start, so its key
    # already IS its display name (kept in step with deploy.sh's display_name()).
    artf-template)               echo "artf-template" ;;
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
    --profile=*)    DEPLOY_PROFILE="${arg#--profile=}" ;;
    --profile)      ;; # value in next arg
    *)
      if [[ "${_PREV_ARG:-}" == "--stack-name" ]]; then STACK_NAME="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--target" ]]; then BUILD_TARGET="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--only" ]]; then BUILD_ONLY="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--tag" ]]; then IMAGE_TAG="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--nemo-src-tag" ]]; then NEMO_SRC_TAG="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--ngc-secret" ]]; then NGC_SECRET="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--ngc-key" ]]; then NGC_KEY="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--region" ]]; then AWS_REGION="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--profile" ]]; then DEPLOY_PROFILE="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

log()  { printf '\033[0;32m[remote-build]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# _can_prompt(): true only when this process can ACTUALLY read the terminal.
#
# `[[ -t 0 ]]` is not that test. For `nohup … &` stdin is still the terminal, so
# `-t 0` passes — but a BACKGROUND process that reads the terminal is sent SIGTTIN
# and stops. `-t 0` therefore turns a clean failure into a silently stopped
# deployment, which is worse than the `</dev/tty` bug it replaced.
#
# The real question is whether our process group is the terminal's FOREGROUND
# process group; `ps -o tpgid=` reports that group for the controlling terminal.
# If ps cannot answer, assume we cannot prompt — a wrong default costs one
# decision, a block costs the whole run. Kept in step with deploy.sh's copy.
_can_prompt() {
  [[ -t 0 ]] || return 1
  local _pgid _tpgid
  _pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
  _tpgid="$(ps -o tpgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
  [[ -n "${_pgid}" && -n "${_tpgid}" && "${_pgid}" == "${_tpgid}" ]]
}

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
  # Gated on stdin being a TERMINAL, not on /dev/tty existing.
  #
  # This prompt previously read `</dev/tty` unconditionally. With no controlling
  # terminal -- a detached run, nohup, setsid, CI -- the REDIRECT ITSELF fails, so
  # `read` returns non-zero and `set -euo pipefail` (top of this file) exits the
  # script. deploy_closed_loop.sh calls this in a command substitution under its own
  # `set -e`, so it died here at its Step 4b: after the feedback-pipeline, glue-etl,
  # closed-loop-core and agentcore-security stacks were created, and BEFORE the
  # vpc-proxy and governance-eventbridge stacks. deploy.sh then turned that into two
  # warnings and printed its success banner. Reported from a real walkthrough.
  #
  # Non-interactively we fail with the remedy instead. The interactive default here
  # is already N -> fail, so this matches the overwhelmingly likely interactive
  # outcome rather than inventing a new behaviour -- and it fails BEFORE any build
  # starts, rather than 20 minutes later at the nvcr.io pull.
  if ! _can_prompt; then
    fail "NGC credentials are required for the NeMo build and none were found for this stack. Pass --ngc-key YOUR_API_KEY (stored in Secrets Manager automatically and reused on later runs) or --ngc-secret SECRET_NAME. Get a key from https://ngc.nvidia.com/ (Profile > Generate API Key). This is not interactive, so there is nothing to confirm."
  fi
  printf '\033[0;33m[warn]\033[0m Continue without NGC credentials? The build will fail if nvcr.io requires auth. [y/N]: ' >&2
  read -r CONTINUE || CONTINUE=""
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
  "${STACK_NAME}-$(display_name artf-template)"
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
# Pick the builder by the architecture of what is being built.
#
# Three images here are arm64 (they back AgentCore runtimes, which are arm64-only):
# agentcore, adaptive-bidding-agent, model-promotion-governance-agent. Everything
# else is amd64.
#
# A build containing ONLY arm64 images goes to the ARM project, which runs natively
# on Graviton (ARM_CONTAINER / amazonlinux-aarch64-standard) and needs no emulation.
# Anything else -- mixed, or amd64-only -- goes to the x86 project, which
# cross-compiles any arm64 image with QEMU + buildx.
#
# This used to send every build to the x86 project unconditionally, leaving
# CB_PROJECT_ARM defined and never used. That made `--target agents` cross-compile
# on x86 for no reason, and put it behind a Docker Hub anonymous pull of
# multiarch/qemu-user-static that CodeBuild regularly loses. The buildspec's own
# rate-limit guidance tells the reader to run `--target agents` *because* it builds
# natively on Graviton; until now that was not true of this script.
_ARM_ONLY=0
if [[ -n "${BUILD_ONLY}" ]]; then
  _ARM_ONLY=1
  for _img in ${BUILD_ONLY}; do
    case "${_img}" in
      agentcore|adaptive-bidding-agent|model-promotion-governance-agent) ;;
      *) _ARM_ONLY=0 ;;
    esac
  done
elif [[ "${BUILD_TARGET}" == "agents" ]]; then
  _ARM_ONLY=1
fi

if [[ "${_ARM_ONLY}" -eq 1 ]]; then
  CB_PROJECT="${CB_PROJECT_ARM}"
  log "Build is arm64-only — using the native Graviton project (no QEMU needed)"
else
  CB_PROJECT="${CB_PROJECT_X86}"
fi

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
    # Same tty guard as the NGC prompt above, and the same reason: `</dev/tty`
    # fails outright with no controlling terminal, taking the script down under
    # `set -e`. This prompt is reachable on any re-run while a prior NeMo build is
    # still IN_PROGRESS, which makes it the second way a detached deploy died.
    #
    # Non-interactively we WAIT. Waiting joins the existing build; starting a new
    # one would duplicate a 15-50 minute GPU-adjacent build, double the spend, and
    # race the first build on the same ECR tags. Quitting would abandon a build the
    # caller is about to depend on.
    if ! _can_prompt; then
      log "Not running in the foreground — waiting for the in-progress build instead of starting a second one."
      CHOICE="w"
    else
      printf '\033[0;33m[warn]\033[0m Do you want to (w)ait for the existing build, (n)ew build, or (q)uit? [w/n/q]: ' >&2
      read -r CHOICE || CHOICE="w"
    fi
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

# Progress line. The previous form was `printf '\r...%s\n'` -- a carriage return
# AND a newline -- so every 15-second poll left its line behind: 60-80 lines for a
# 20-minute build, and the single largest contributor to this script's output.
#
# On a terminal the line now rewrites in place. Off a terminal (redirected to a
# log, or CI) in-place rewriting produces one unreadable line, so there we print
# only when the build's PHASE actually changes -- a handful of lines that are worth
# having in a log file, instead of one per poll.
_LAST_REPORTED_PHASE=""
_PROGRESS_LINE_OPEN=0
report_progress() {
  local status="$1" phase="$2" elapsed="$3"
  local text
  text="$(printf '\033[0;32m[remote-build]\033[0m Status: %-12s Phase: %-20s Elapsed: %dm%02ds' \
    "${status}" "${phase}" "$(( elapsed / 60 ))" "$(( elapsed % 60 ))")"
  if [[ -t 2 ]]; then
    printf '\r%s\033[K' "${text}" >&2
    _PROGRESS_LINE_OPEN=1
  elif [[ "${phase}" != "${_LAST_REPORTED_PHASE}" ]]; then
    printf '%s\n' "${text}" >&2
  fi
  _LAST_REPORTED_PHASE="${phase}"
}

# Close the in-place line so whatever prints next starts on its own row.
end_progress() {
  if [[ "${_PROGRESS_LINE_OPEN}" -eq 1 ]]; then
    printf '\n' >&2
    _PROGRESS_LINE_OPEN=0
  fi
}

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
      end_progress
      log "Build SUCCEEDED"
      break
      ;;
    FAILED|FAULT|TIMED_OUT|STOPPED)
      end_progress
      fail "Build ${BUILD_STATUS}. Check logs: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${BUILD_ID}?region=${AWS_REGION}"
      ;;
    IN_PROGRESS|*)
      report_progress "${BUILD_STATUS}" "${PHASE}" "${ELAPSED}"
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
