#!/usr/bin/env bash
# =============================================================================
# check_builds.sh — Check if the NeMo-RL container image build is complete
#
# The NeMo-RL training container (~20 GB) is the only image that builds
# asynchronously. This script checks its status and writes .nemo-outputs.json
# when ready, so deploy.sh can resume without re-checking.
#
# Usage:
#   ./check_builds.sh --prefix dv2
#   ./check_builds.sh --prefix dv2 --watch   # poll every 30s until ready
#
# Exit codes:
#   0 — NeMo image is ready (also writes .nemo-outputs.json)
#   1 — NeMo image is not ready yet
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_PREFIX=""
WATCH=0

for arg in "$@"; do
  case "${arg}" in
    --prefix=*)  STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)    ;;
    --watch)     WATCH=1 ;;
    -h|--help)   sed -n '2,15p' "$0"; exit 0 ;;
    *)
      if [[ "${_PREV_ARG:-}" == "--prefix" ]]; then
        STACK_PREFIX="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

STACK_NAME="nvidia-artf-recommenders"
if [[ -n "${STACK_PREFIX}" ]]; then
  STACK_NAME="${STACK_PREFIX}-${STACK_NAME}"
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo latest)}"
NEMO_OUTPUTS="${SCRIPT_DIR}/.nemo-outputs.json"
TRAINING_REPO="artf-nemo-rl-training"

# =========================================================================
# Check functions
# =========================================================================
check_nemo_ecr() {
  aws ecr describe-images --repository-name "${TRAINING_REPO}" \
    --image-ids imageTag="dlrm" --region "${AWS_REGION}" >/dev/null 2>&1
}

check_codebuild_status() {
  local project="${STACK_NAME}-image-builder"
  local latest_build
  latest_build=$(aws codebuild list-builds-for-project \
    --project-name "${project}" \
    --region "${AWS_REGION}" \
    --query 'ids[0]' --output text 2>/dev/null || echo "None")

  if [[ "${latest_build}" == "None" || -z "${latest_build}" ]]; then
    echo "NO_BUILDS"
    return
  fi

  local status
  status=$(aws codebuild batch-get-builds \
    --ids "${latest_build}" \
    --region "${AWS_REGION}" \
    --query 'builds[0].buildStatus' --output text 2>/dev/null || echo "UNKNOWN")
  echo "${status}"
}

write_nemo_outputs() {
  cat > "${NEMO_OUTPUTS}" <<EOF
{
  "Repository": "${TRAINING_REPO}",
  "Registry": "${REGISTRY}",
  "Tags": ["dlrm", "ncf", "widedeep"],
  "BuiltAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# =========================================================================
# Display
# =========================================================================
display_status() {
  local cb_status
  cb_status=$(check_codebuild_status)

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  Container Image Build Status"
  echo "  Stack: ${STACK_NAME}  Tag: ${IMAGE_TAG}  Region: ${AWS_REGION}"
  echo "══════════════════════════════════════════════════════════════"
  echo ""

  # Part 1 images (synchronous — should already be done). Repo names are the
  # job-oriented display names (see RENAME_MAP.md) — matching deploy.sh's
  # display_name() lookup.
  echo "  Part 1 Images (needed for: kubectl apply, AgentCore):"
  echo "  ─────────────────────────────────────────────────────"
  local part1_ok=true
  for repo in \
    "${STACK_NAME}-bid-pricer" \
    "${STACK_NAME}-audience-activator" \
    "${STACK_NAME}-deal-scorer" \
    "${STACK_NAME}-signals-enricher" \
    "${STACK_NAME}-orchestrator" \
    "${STACK_NAME}-agentcore"; do
    if aws ecr describe-images --repository-name "${repo}" \
        --image-ids imageTag="${IMAGE_TAG}" --region "${AWS_REGION}" >/dev/null 2>&1; then
      printf '    ✓  %s:%s\n' "${repo}" "${IMAGE_TAG}"
    else
      printf '    ✗  %s:%s  (missing)\n' "${repo}" "${IMAGE_TAG}"
      part1_ok=false
    fi
  done

  echo ""
  echo "  NeMo-RL Training Image (needed for: SageMaker retraining):"
  echo "  ───────────────────────────────────────────────────────────"

  case "${cb_status}" in
    SUCCEEDED)   printf '    CodeBuild:  ✓ SUCCEEDED\n' ;;
    IN_PROGRESS) printf '    CodeBuild:  ⧗ IN PROGRESS\n' ;;
    FAILED)      printf '    CodeBuild:  ✗ FAILED\n' ;;
    NO_BUILDS)   printf '    CodeBuild:  — No builds started\n' ;;
    *)           printf '    CodeBuild:  ? %s\n' "${cb_status}" ;;
  esac

  if check_nemo_ecr; then
    printf '    ✓  %s:dlrm (+ ncf, widedeep)\n' "${TRAINING_REPO}"
    # Write outputs as side effect
    if [[ ! -f "${NEMO_OUTPUTS}" ]]; then
      write_nemo_outputs
    fi
  else
    printf '    ✗  %s:dlrm  (missing)\n' "${TRAINING_REPO}"
  fi

  echo ""
  echo "  ─────────────────────────────────────────────────────"

  # Determine overall status
  if [[ "${part1_ok}" == "true" ]] && check_nemo_ecr; then
    echo "  ✓ ALL IMAGES READY"
    echo ""
    echo "  The NeMo-RL training container is in ECR. SageMaker retraining"
    echo "  jobs will use it automatically when triggered by EventBridge."
    echo "  No further action needed."
    echo ""
    return 0
  elif [[ "${part1_ok}" == "true" ]] && ! check_nemo_ecr; then
    echo ""
    echo "  Part 1 is ready. NeMo build is still in progress."
    echo ""
    if [[ "${cb_status}" == "IN_PROGRESS" ]]; then
      echo "  Re-check with:"
      echo "    ./check_builds.sh --prefix ${STACK_PREFIX:-<prefix>}"
    elif [[ "${cb_status}" == "FAILED" ]]; then
      echo "  Build failed. Check CodeBuild console, fix, and re-run:"
      echo "    ./deploy.sh --prefix ${STACK_PREFIX:-<prefix>} --with-retraining --start-at=11"
    fi
    echo ""
    return 1
  else
    echo ""
    echo "  Part 1 images are missing. Run the full deployment:"
    echo "    ./deploy.sh --prefix ${STACK_PREFIX:-<prefix>}"
    echo ""
    return 1
  fi
}

# =========================================================================
# Main
# =========================================================================
if [[ "${WATCH}" -eq 1 ]]; then
  while true; do
    clear
    if display_status; then
      exit 0
    fi
    echo "  Refreshing in 30s... (Ctrl+C to stop)"
    sleep 30
  done
else
  display_status
fi
