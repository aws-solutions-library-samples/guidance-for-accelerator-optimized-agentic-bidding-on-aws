#!/usr/bin/env bash
# Requires real bash, not bash in POSIX mode. See the same guard in deploy.sh.
if [ -z "${BASH_VERSION:-}" ] || { command -v shopt >/dev/null 2>&1 && shopt -qo posix; }; then
  printf 'deploy_prebid.sh must be run with bash, not sh.\n\n  bash %s %s\n\n' "$0" "$*" >&2
  exit 1
fi
# =============================================================================
# deploy_prebid.sh - Deploy Prebid Server as the sell-side ARTF host
#
# Invoked by `./deploy.sh --with-prebid`, mirroring the
# `--with-retraining` -> deploy_closed_loop.sh pattern. Runs standalone too.
#
# WHAT THIS DEPLOYS, AND WHERE IT DIFFERS FROM UPSTREAM
#
# The source is the AWS guidance "Prebid Server Deployment on AWS", consumed as a
# PINNED, UNFORKED release. The TOPOLOGY is ours: Prebid Server runs as pods on the
# EKS cluster this repository already creates, not on the upstream guidance's ECS
# Fargate service.
#
# That is a deliberate divergence. ARTF places agent containers inside the host
# platform's own infrastructure; a host outside the cluster would reinstate the
# network hop ARTF exists to remove, and would need VPC peering or an RTB Fabric
# link to talk to the orchestrator at all. The cost of the divergence is that
# availability and scaling of the Prebid pods are OURS, not the upstream stack's.
#
# Consequences worth stating plainly:
#   - The upstream CDK app is NOT deployed. No ECS service, no ALB, no CloudFront,
#     no EFS, no DataSync, and no RTB Fabric link exist in this topology. FR-35's
#     "delete any RTB Fabric link before the stacks" therefore has nothing to
#     delete here, which is checked rather than assumed at teardown.
#   - The upstream guidance's published cost figure describes ITS deployment, not
#     this one, and is not reused. See the disclosure below.
#
# Steps:
#   1. Preflight        report EVERY missing prerequisite at once
#   2. Cost             disclose idle cost BEFORE anything is provisioned
#   3. Acquire          fetch and verify the pinned release
#   4. Place            add ARTF hook module + artfhouse adapter, additively
#   5. Build            CodeBuild -> ECR (never a local Docker build)
#   6. Deploy           prebid_cfn.yaml, then the Kubernetes manifest
#   7. Wire             substitute endpoints, upload prebid-config.yaml to S3
#   8. Record           pinned version, image digest, stack name, endpoints
#
# Usage:
#   ./deploy_prebid.sh --prefix nv5
#   ./deploy_prebid.sh --prefix nv5 --yes            # no cost confirmation prompt
#   ./deploy_prebid.sh --prefix nv5 --start-at=5     # resume at the build step
#   ./deploy_prebid.sh --prefix nv5 --destroy
#
# Options:
#   --prefix NAME        Stack prefix (or STACK_PREFIX env)
#   --cluster NAME       EKS cluster name (default: <prefix->nvidia-artf-recommenders-triton)
#   --user-pool-id ID    Cognito user pool to extend (default: discovered)
#   --namespace NAME     Kubernetes namespace (default: default)
#   --region REGION      AWS region (default: $AWS_REGION or us-east-1)
#   --tag TAG            Image tag (default: pinned version + source hash)
#   --start-at N         Resume from step N
#   --skip-build         Reuse the image already in ECR
#   --yes                Accept the cost disclosure without prompting
#   --destroy            Tear down (Kubernetes first, then the CFN stack)
#
# Prerequisites (all checked in step 1, all reported together):
#   - deploy.sh has run: EKS cluster, Cognito user pool, CodeBuild project
#   - AWS CLI v2 with credentials, kubectl, python3
#   - Free CPU and memory on the cluster for the Prebid pod
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_PREFIX="${STACK_PREFIX:-}"
BASE_NAME="${BASE_NAME:-nvidia-artf-recommenders}"
CLUSTER_NAME="${CLUSTER_NAME:-}"
USER_POOL_ID="${USER_POOL_ID:-}"
NAMESPACE="${NAMESPACE:-default}"
IMAGE_TAG="${IMAGE_TAG:-}"
START_AT=1
SKIP_BUILD=0
ASSUME_YES=0
DESTROY=0

# A second bidder seat, so the auction is CONTESTED rather than a single seat
# bidding against nothing. On by default: an auction with one seat has no winner
# to speak of, and the release already carries both halves -- the AMT adapter and
# the bidder simulator it calls.
#
# The value of the simulator's endpoint is NOT knowable at build time: the image is
# built in step 5 and the stack that publishes the URL is created in step 6. Only
# the DECISION is needed early, which is what this flag carries. The URL reaches
# the pod as an environment variable, and step 6 fails if the stack did not
# publish it -- because amt.yaml interpolates
# ${AMT_BIDDING_SERVER_SIMULATOR_ENDPOINT} with no default, so a registered amt
# bidder with no endpoint is a pod that will not start.
WITH_SIMULATOR=1

# The cost of any node capacity the Prebid pods force. Passed to the disclosure
# helper, which REQUIRES it: on a cluster without headroom it is the dominant term.
# Left empty means "not yet known", and the disclosure then says so rather than
# printing a total that looks authoritative. Step 1 measures headroom and sets it.
ADDITIONAL_NODE_MONTHLY_USD=""

for arg in "$@"; do
  case "${arg}" in
    --prefix=*)        STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)          ;; # value in next arg
    --cluster=*)       CLUSTER_NAME="${arg#--cluster=}" ;;
    --cluster)         ;;
    --user-pool-id=*)  USER_POOL_ID="${arg#--user-pool-id=}" ;;
    --user-pool-id)    ;;
    --namespace=*)     NAMESPACE="${arg#--namespace=}" ;;
    --namespace)       ;;
    --region=*)        AWS_REGION="${arg#--region=}" ;;
    --region)          ;;
    --tag=*)           IMAGE_TAG="${arg#--tag=}" ;;
    --tag)             ;;
    --start-at=*)      START_AT="${arg#--start-at=}" ;;
    --start-at)        ;;
    --skip-build)      SKIP_BUILD=1 ;;
    --no-simulator)    WITH_SIMULATOR=0 ;;
    --yes|--non-interactive) ASSUME_YES=1 ;;
    --destroy)         DESTROY=1 ;;
    -h|--help)         sed -n '7,67p' "$0"; exit 0 ;;
    *)
      if   [[ "${_PREV_ARG:-}" == "--prefix" ]];       then STACK_PREFIX="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--cluster" ]];      then CLUSTER_NAME="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--user-pool-id" ]]; then USER_POOL_ID="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--namespace" ]];    then NAMESPACE="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--region" ]];       then AWS_REGION="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--tag" ]];          then IMAGE_TAG="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--start-at" ]];     then START_AT="${arg}"
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
unset _PREV_ARG

log()  { printf '\033[0;32m[prebid]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[prebid][warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[prebid][fail]\033[0m %s\n' "$*" >&2; exit 1; }

# Exported because the inline python below reads it to find prebid_release.py.
# Exported HERE rather than at first use, so no step can run before it is set.
export SCRIPT_DIR

# ---------------------------------------------------------------- names
# ${STACK_PREFIX:+${STACK_PREFIX}-} yields "" when unset and "prefix-" when set, so
# an unprefixed deployment does not end up with a leading hyphen.
PFX="${STACK_PREFIX:+${STACK_PREFIX}-}"
STACK_NAME="${PFX}${BASE_NAME}"
CLUSTER_NAME="${CLUSTER_NAME:-${STACK_NAME}-triton}"
PREBID_STACK="${PFX}prebid-artf"
CB_PROJECT="${STACK_NAME}-image-builder"
ECR_REPO="${PFX}prebid-server"
# The two prefixes the image entrypoint reads, in this order. `current/` is copied
# over `default/` into one directory, so a file in both is REPLACED, not merged.
DEFAULT_CONFIG_PREFIX="prebid-server/default/"
CONFIG_KEY="prebid-server/current/prebid-config.yaml"
# The orchestrator's in-cluster address. Its Cognito JWT middleware is enforced in the
# application, so reaching it over service DNS does not bypass authentication.
ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://orchestrator.${NAMESPACE}.svc.cluster.local/v1/mutations}"
K8S_MANIFEST="${SCRIPT_DIR}/eks/prebid-server-deployment.yaml"
CFN_TEMPLATE="${SCRIPT_DIR}/prebid_cfn.yaml"
CONFIG_TEMPLATE="${SCRIPT_DIR}/scripts/prebid_config_template.yaml"
RECORD_FILE="${REPO_ROOT}/.prebid-deployment.json"

# The pinned release, read from the helper so the version lives in exactly one place
# and is unit-tested there rather than duplicated in shell.
PINNED_VERSION="$(python3 -c 'import sys; sys.path.insert(0, "'"${SCRIPT_DIR}"'/scripts"); import prebid_release as p; print(p.PINNED_VERSION)')"
UPSTREAM_REPO="aws-solutions-library-samples/prebid-server-deployment-on-aws"
UPSTREAM_TARBALL="https://github.com/${UPSTREAM_REPO}/archive/refs/tags/${PINNED_VERSION}.tar.gz"
WORK_DIR="${WORK_DIR:-/tmp/prebid-artf-${PINNED_VERSION}}"

# =============================================================================
# Helper: reconcile a CloudFormation stack
#
# Classification is delegated to prebid_release.classify_stack_status rather than
# matched inline, because the distinction that matters -- a stack that is merely
# IN_PROGRESS versus one in terminal failure -- is the difference between waiting
# and deleting, and getting it wrong destroys a live stack.
# =============================================================================
classify_stack() {
  local status="$1"
  python3 -c 'import sys; sys.path.insert(0, "'"${SCRIPT_DIR}"'/scripts"); import prebid_release as p; print(p.classify_stack_status(sys.argv[1] if len(sys.argv) > 1 else None))' "${status}"
}

deploy_cfn_stack() {
  local stack_name="$1"; local template_file="$2"; shift 2
  local params=("$@")

  local existing_status
  existing_status="$(aws cloudformation describe-stacks --stack-name "${stack_name}" \
    --region "${AWS_REGION}" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"
  [[ "${existing_status}" == "None" ]] && existing_status=""

  local kind
  kind="$(classify_stack "${existing_status}")"
  log "  Stack ${stack_name}: ${kind}${existing_status:+ (${existing_status})}"

  local action="create-stack"
  case "${kind}" in
    absent)   action="create-stack" ;;
    healthy)  action="update-stack" ;;
    failed_terminal)
      log "  Deleting ${existing_status} remnant before recreating..."
      aws cloudformation delete-stack --stack-name "${stack_name}" --region "${AWS_REGION}"
      aws cloudformation wait stack-delete-complete --stack-name "${stack_name}" --region "${AWS_REGION}"
      action="create-stack"
      ;;
    in_progress)
      fail "Stack ${stack_name} is ${existing_status}. Wait for it to settle; deleting a stack mid-operation is destructive."
      ;;
    *)
      fail "Stack ${stack_name} is in an unrecognised status (${existing_status}). Refusing to act on it."
      ;;
  esac

  local cmd=(aws cloudformation "${action}"
    --stack-name "${stack_name}"
    --template-body "file://${template_file}"
    --capabilities CAPABILITY_NAMED_IAM
    --region "${AWS_REGION}")
  [[ ${#params[@]} -gt 0 ]] && cmd+=(--parameters "${params[@]}")

  local out="/tmp/${stack_name}-cfn-output.txt"
  if "${cmd[@]}" >"${out}" 2>&1 && grep -q "StackId" "${out}"; then
    local wait_action="stack-create-complete"
    [[ "${action}" == "update-stack" ]] && wait_action="stack-update-complete"
    log "  Waiting for ${stack_name} (${wait_action})..."
    aws cloudformation wait "${wait_action}" --stack-name "${stack_name}" --region "${AWS_REGION}" \
      || fail "Stack ${stack_name} did not reach ${wait_action}. See the CloudFormation events."
    log "  Stack ${stack_name}: COMPLETE"
  else
    local cfn_error; cfn_error="$(cat "${out}" 2>/dev/null || echo '')"
    # A reconciliation that finds nothing to do is a SUCCESS. Delegated to the
    # helper so the exact upstream wording lives in one tested place.
    if python3 -c 'import sys; sys.path.insert(0, "'"${SCRIPT_DIR}"'/scripts"); import prebid_release as p; sys.exit(0 if p.is_no_op_error(sys.stdin.read()) else 1)' <"${out}"; then
      log "  Stack ${stack_name}: already current, nothing to change"
      return 0
    fi
    warn "CFN error: ${cfn_error}"
    fail "Stack ${stack_name} deploy failed"
  fi
}

stack_output() {
  aws cloudformation describe-stacks --stack-name "$1" --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text 2>/dev/null || echo ''
}

# =============================================================================
# kubectl, PINNED to this run's cluster
#
# kubectl has no implicit cluster scoping: it acts on whatever context is current in
# the shared ~/.kube/config. deploy.sh carries a first-hand account of what that costs
# (deploy.sh:527) -- a concurrent run won the race on that file and a teardown deleted
# a DIFFERENT environment's live workloads while its CloudFormation stacks stayed put.
#
# Everything here therefore goes through KUBECTL, which pins --context to a name
# derived from THIS run's CLUSTER_NAME whenever such a context exists. When it does
# not, the ambient context is used and the cluster it resolves to is PRINTED, so an
# apply against the wrong cluster is visible rather than silent.
# =============================================================================
KUBECTL=(kubectl)
resolve_kube_context() {
  local arn_context="arn:${AWS_PARTITION:-aws}:eks:${AWS_REGION}:${ACCOUNT_ID}:cluster/${CLUSTER_NAME}"
  if kubectl config get-contexts "${arn_context}" >/dev/null 2>&1; then
    KUBECTL=(kubectl --context "${arn_context}")
    log "  kubectl pinned to context ${arn_context}"
    return 0
  fi

  local current cluster_entry
  current="$(kubectl config current-context 2>/dev/null || echo '')"
  if [[ -z "${current}" ]]; then
    warn "No kubectl context at all."
    return 1
  fi
  cluster_entry="$(kubectl config view --minify -o jsonpath='{.clusters[0].name}' 2>/dev/null || echo '')"
  KUBECTL=(kubectl --context "${current}")

  if [[ "${cluster_entry}" == *"cluster/${CLUSTER_NAME}" || "${cluster_entry}" == "${CLUSTER_NAME}" ]]; then
    log "  kubectl pinned to context '${current}' (cluster ${cluster_entry})"
    return 0
  fi

  warn "No kubectl context named for ${CLUSTER_NAME}."
  warn "Falling back to the current context '${current}', whose cluster entry is"
  warn "  ${cluster_entry:-<unknown>}"
  warn "If that is not ${CLUSTER_NAME}, STOP and run:"
  warn "  aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${AWS_REGION}"
  return 0
}

# =============================================================================
# Helper: publish the release's own default configuration to S3
#
# The image entrypoint requires FOUR files under prebid-server/default/ --
# entrypoint.sh, prebid-config.yaml, prebid-logging.xml and
# prebid-analytics-logging.xml -- and exits 1 if any is missing. They ship in the
# release at deployment/ecr/prebid-server/default-config/, so they are copied from
# there rather than authored here: hand-written substitutes would drift from the
# entrypoint that consumes them.
#
# The default prefix is refreshed on every deploy. That is safe because operator
# edits belong in the `current/` prefix, which overrides it and which this script
# never overwrites once present.
# =============================================================================
upload_default_config() {
  local src="${BUILD_CONTEXT}/default-config"
  local bucket="$1"

  [[ -d "${src}" ]] \
    || fail "No default-config directory at ${src}. The pod cannot start without it; the release layout has changed."

  local required=(entrypoint.sh prebid-config.yaml prebid-logging.xml prebid-analytics-logging.xml)
  local missing=()
  local f
  for f in "${required[@]}"; do
    [[ -f "${src}/${f}" ]] || missing+=("${f}")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    warn "The release's default-config is missing files the image entrypoint requires:"
    for f in "${missing[@]}"; do warn "    ${f}"; done
    fail "The pod would exit 1 on start. Refusing to deploy."
  fi

  log "  Publishing the release default config to s3://${bucket}/prebid-server/default/"
  aws s3 cp "${src}/" "s3://${bucket}/prebid-server/default/" --recursive \
    --exclude 'README.md' --region "${AWS_REGION}" >/dev/null \
    || fail "Could not publish the default configuration to s3://${bucket}/prebid-server/default/"
  log "  Published ${#required[@]} required files"
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# deploy_amt_simulator -- the second auction seat's demand, in-cluster.
#
# Sets AMT_SIMULATOR_ENDPOINT on success. A single seat bidding against nothing
# has a winner only in the trivial sense, so this is what makes the auction
# contested.
#
# WHY IN-CLUSTER AND NOT API GATEWAY + LAMBDA
#
# The caller is Prebid's `amt` adapter, used UNMODIFIED, and it builds its request
# with BidderUtil.defaultRequest(...) -- default headers only, no bearer token, no
# API key, no signature. So a public endpoint for it could not authenticate its
# caller and would have to accept unauthenticated bid requests from anyone. A
# ClusterIP Service has no public surface to authenticate, and Prebid is already
# in this cluster.
#
# WHY A ConfigMap AND A STOCK IMAGE
#
# The release ships the simulator as a Lambda handler that reads only `body` from
# its event and imports nothing outside the standard library. Mounting it beside a
# thin wrapper of ours on a stock Python image avoids an ECR repository, a
# CodeBuild project and a second image build on every deploy, for something that
# is not on the ARTF request path.
#
# The release's handler is NEVER copied into this repository: it is Apache-2.0 and
# this repo is MIT-0. It is read from the fetched release at deploy time.
# ---------------------------------------------------------------------------
deploy_amt_simulator() {
  AMT_SIMULATOR_ENDPOINT=""

  if [[ "${WITH_SIMULATOR}" -ne 1 ]]; then
    log "  --no-simulator: no second seat is deployed."
    return 0
  fi

  local sim_src="${UPSTREAM_DIR}/source/loadtest/bidder_simulator/lambdas/loadtest_bidder"
  local wrapper="${REPO_ROOT}/source/prebid/amt-simulator/server.py"

  for f in "${sim_src}/handler.py" "${sim_src}/bid_response.json" "${wrapper}"; do
    [[ -f "${f}" ]] || {
      warn "Cannot deploy the bid simulator: ${f} is missing."
      warn "The auction will have one seat and no competing bid."
      return 0
    }
  done

  if [[ ${#KUBECTL[@]} -eq 1 ]]; then
    resolve_kube_context || {
      warn "No usable kubectl context; the bid simulator was not deployed."
      return 0
    }
  fi

  # --dry-run=client | apply is the idempotent create-or-update for a ConfigMap:
  # `create` alone fails once it exists, and this must be re-runnable.
  "${KUBECTL[@]}" create configmap amt-simulator-code -n "${NAMESPACE}" \
    --from-file="handler.py=${sim_src}/handler.py" \
    --from-file="bid_response.json=${sim_src}/bid_response.json" \
    --from-file="server.py=${wrapper}" \
    --dry-run=client -o yaml \
    | "${KUBECTL[@]}" apply -f - >/dev/null \
    || { warn "Could not write the amt-simulator-code ConfigMap; no second seat."; return 0; }

  local processed="/tmp/${CLUSTER_NAME}-amt-simulator.yaml"
  sed -e "s|__NAMESPACE__|${NAMESPACE}|g" \
      "${SCRIPT_DIR}/eks/amt-simulator-deployment.yaml" >"${processed}"

  if grep -q '__[A-Z_]*__' "${processed}"; then
    grep -o '__[A-Z_]*__' "${processed}" | sort -u >&2
    fail "Refusing to apply the simulator manifest with unsubstituted placeholders"
  fi

  "${KUBECTL[@]}" apply -f "${processed}" >/dev/null \
    || { warn "kubectl apply failed for the bid simulator; no second seat."; return 0; }

  # The ConfigMap content changes without the Deployment's spec changing, so
  # `apply` alone would leave the old pod serving the previous code -- the same
  # trap the Prebid rollout documents.
  "${KUBECTL[@]}" rollout restart deployment/amt-simulator -n "${NAMESPACE}" >/dev/null 2>&1 || true

  if "${KUBECTL[@]}" rollout status deployment/amt-simulator -n "${NAMESPACE}" --timeout=180s >/dev/null 2>&1; then
    AMT_SIMULATOR_ENDPOINT="http://amt-simulator.${NAMESPACE}.svc.cluster.local/amt-exchange"
    log "  Bid simulator ready at ${AMT_SIMULATOR_ENDPOINT} (cluster-internal only)"
  else
    # Left EMPTY on purpose. An endpoint reported for a pod that never became
    # ready would enable the seat and turn every auction into a per-bidder
    # timeout, which reads as a flaky bidder rather than a failed deployment.
    warn "The bid simulator did not become ready within 180s, so no endpoint is"
    warn "reported and the amt seat stays disabled. Investigate with:"
    warn "  kubectl logs deployment/amt-simulator -n ${NAMESPACE}"
  fi
}

# ---------------------------------------------------------------------------
# publish_config_overlay -- render and upload current/prebid-config.yaml.
#
# MUST run BEFORE the Kubernetes manifest is applied. The pod's entrypoint copies
# default/ and then current/ over it, and the release's default prebid-config.yaml
# contains Spring placeholders this topology never sets -- ${LOG_ANALYTICS_ENABLED}
# among them. Reaching Spring with the default file alone is fatal:
#
#   PlaceholderResolutionException: Could not resolve placeholder
#   'LOG_ANALYTICS_ENABLED' in value "${LOG_ANALYTICS_ENABLED}"
#   ... Failed to execute entrypoint.sh
#
# The overlay is what removes those placeholders, so uploading it after the rollout
# wait -- as this script first did -- gave the pod no config it could boot from and
# the wait could never succeed.
#
# Editing this object is the documented way to reconfigure (FR-38), so a blind
# overwrite on every deploy would discard the operator's work. But uploading only
# when absent -- what this did originally -- meant a change to CONFIG_TEMPLATE
# reached a fresh account and silently never reached an already-deployed cluster.
#
# So the object carries a `rendered-sha` metadata stamp of the render that produced
# it. Stamp still matches the bytes: nobody edited it, replace it. Stamp missing or
# no longer matching: an operator owns it, leave it and say plainly what is
# therefore not applied. Either way this is safe to call twice, which is why step 7
# still calls it and simply reports.
# ---------------------------------------------------------------------------
publish_config_overlay() {
  local bucket="$1"
  local endpoint="$2"

  # Rendered FIRST, unconditionally, because the render is what an existing object
  # has to be compared against.
  # With the simulator disabled the amt seat is not in the image, so the block is
  # rendered disabled and pointed at an address nothing resolves -- rather than left
  # claiming a seat that cannot answer. A disabled adapter is never called, so the
  # address is inert by construction and not a route anything can take.
  local amt_enabled="false"
  local amt_endpoint="http://amt-simulator.not-deployed.invalid/"
  if [[ "${WITH_SIMULATOR}" -eq 1 && -n "${AMT_SIMULATOR_ENDPOINT:-}" ]]; then
    amt_enabled="true"
    amt_endpoint="${AMT_SIMULATOR_ENDPOINT}"
  elif [[ "${WITH_SIMULATOR}" -eq 1 ]]; then
    # The seat is wanted but nothing has published an endpoint for it. Rendered
    # DISABLED rather than enabled-and-pointed-nowhere: a bidder configured with an
    # address that does not resolve produces per-auction timeouts attributed to the
    # seat, which reads as a flaky bidder rather than as a missing deployment.
    #
    # The pod still boots either way, because the manifest always supplies
    # AMT_BIDDING_SERVER_SIMULATOR_ENDPOINT and Spring resolves that placeholder
    # before anything consults `enabled`.
    warn "The amt seat was requested but no simulator endpoint is available, so it is"
    warn "rendered DISABLED. The auction will have one seat (artfhouse) and no"
    warn "competing bid. This is the expected state until the simulator is deployed."
  fi

  local rendered="${WORK_DIR}/prebid-config.yaml"
  sed -e "s|__DEMAND_ENDPOINT__|${endpoint}|g" \
      -e "s|__CONFIG_BUCKET__|${bucket}|g" \
      -e "s|__AWS_REGION__|${AWS_REGION}|g" \
      -e "s|__AMT_ENABLED__|${amt_enabled}|g" \
      -e "s|__AMT_SIMULATOR_ENDPOINT__|${amt_endpoint}|g" \
      "${CONFIG_TEMPLATE}" >"${rendered}"

  if grep -q '__[A-Z_]*__' "${rendered}"; then
    warn "Unsubstituted placeholders remain in the rendered configuration:"
    grep -o '__[A-Z_]*__' "${rendered}" | sort -u >&2
    fail "Refusing to upload a configuration with unsubstituted placeholders"
  fi

  local render_sha
  render_sha="$(shasum -a 256 "${rendered}" | cut -d' ' -f1)"

  if aws s3api head-object --bucket "${bucket}" --key "${CONFIG_KEY}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    # An object exists. "Upload only if absent" -- what this did before -- meant a
    # change to CONFIG_TEMPLATE reached a fresh account and silently never reached an
    # already-deployed cluster, so a newly registered bidder would simply not appear.
    # But blind overwriting would discard the operator edits that FR-38 makes the
    # documented way to reconfigure.
    #
    # Distinguish the two by stamping our own render on the object as user metadata.
    # If the stamp still matches the object's bytes, nobody has edited it since we
    # wrote it, and replacing it loses nothing.
    local stamped live_sha
    stamped="$(aws s3api head-object --bucket "${bucket}" --key "${CONFIG_KEY}" \
      --region "${AWS_REGION}" --query 'Metadata."rendered-sha"' --output text 2>/dev/null || echo '')"
    aws s3 cp "s3://${bucket}/${CONFIG_KEY}" "${WORK_DIR}/live-config.yaml" \
      --region "${AWS_REGION}" --quiet 2>/dev/null \
      || fail "Could not read the live configuration overlay to compare against"
    live_sha="$(shasum -a 256 "${WORK_DIR}/live-config.yaml" | cut -d' ' -f1)"

    if [[ -n "${stamped}" && "${stamped}" != "None" && "${stamped}" == "${live_sha}" ]]; then
      if [[ "${live_sha}" == "${render_sha}" ]]; then
        log "  s3://${bucket}/${CONFIG_KEY} already matches this render - nothing to change."
        return 0
      fi
      aws s3 cp "${rendered}" "s3://${bucket}/${CONFIG_KEY}" --region "${AWS_REGION}" \
        --metadata "rendered-sha=${render_sha}" >/dev/null \
        || fail "Could not update the configuration overlay"
      log "  Configuration overlay UPDATED from the template (it was unmodified since the last deploy)."
      return 0
    fi

    # Objects uploaded before stamping existed carry no stamp, so every deployment
    # made by the earlier code would warn forever. When such an object's bytes are
    # already identical to the render, it IS our render and only the stamp is
    # missing: adopt it in place. That leaves the content untouched and confines the
    # warning below to files somebody has genuinely edited.
    if [[ "${live_sha}" == "${render_sha}" ]]; then
      aws s3 cp "${rendered}" "s3://${bucket}/${CONFIG_KEY}" --region "${AWS_REGION}" \
        --metadata "rendered-sha=${render_sha}" >/dev/null \
        || fail "Could not stamp the existing configuration overlay"
      log "  s3://${bucket}/${CONFIG_KEY} already matched this render; recorded the stamp so"
      log "  future template changes can be applied automatically. Content unchanged."
      return 0
    fi

    # Either the object predates this stamping AND differs from the render, or its
    # bytes no longer match its stamp. Both mean someone owns this file.
    warn "s3://${bucket}/${CONFIG_KEY} has been modified outside this script (or predates"
    warn "stamping), so it was LEFT UNTOUCHED and any template change is NOT applied."
    warn "Anything the template adds - a newly registered bidder among them - is absent"
    warn "from this deployment until the render is adopted. To see the difference and"
    warn "then adopt it:"
    warn "  aws s3 cp s3://${bucket}/${CONFIG_KEY} /tmp/live-prebid-config.yaml --region ${AWS_REGION}"
    warn "  diff /tmp/live-prebid-config.yaml ${rendered}"
    warn "  aws s3 cp ${rendered} s3://${bucket}/${CONFIG_KEY} --region ${AWS_REGION} --metadata rendered-sha=${render_sha}"
    return 0
  fi

  aws s3 cp "${rendered}" "s3://${bucket}/${CONFIG_KEY}" --region "${AWS_REGION}" \
    --metadata "rendered-sha=${render_sha}" >/dev/null \
    || fail "Could not upload the configuration overlay"
  log "  Uploaded the configuration OVERLAY to s3://${bucket}/${CONFIG_KEY}"
  warn "That overlay REPLACES the release default file - the entrypoint copies"
  warn "current/ over default/ into one directory and does not merge them."
  warn "It registers NO hooks and sets none of the four auction settings, so Prebid"
  warn "will serve auctions and apply no ARTF mutations until those are filled in."
}

# =============================================================================
# TEARDOWN
#
# Kubernetes first, then the stack. Order matters: the pod authenticates with the
# secret and assumes the IRSA role, and removing those first leaves a pod crash-
# looping on errors that describe a missing credential rather than a teardown.
#
# The shared Cognito USER POOL is never deleted -- the frontend and the
# orchestrator authenticate against it. Only the resources this feature ADDED to
# it (domain, resource servers, M2M client) go, and they go with the stack that
# created them.
#
# Every step treats "already absent" as success, so a partial deployment can be
# torn down without hand-editing.
# =============================================================================
if [[ "${DESTROY}" -eq 1 ]]; then
  log "Tearing down the Prebid ARTF stack (prefix=${STACK_PREFIX:-<none>})"

  # ACCOUNT_ID is needed to build the pinned context name. Resolved before any delete,
  # because a delete against an unverified context is the destructive case.
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo '')"
  if [[ -n "${ACCOUNT_ID}" ]] && resolve_kube_context; then
    log "Removing Kubernetes objects..."
    "${KUBECTL[@]}" delete deployment prebid-server -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    "${KUBECTL[@]}" delete service    prebid-server -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    "${KUBECTL[@]}" delete serviceaccount prebid-artf-host-sa -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    # Created by this script, not by the stack, so the stack's deletion does not remove it.
    # A stale credential Secret left in the namespace would outlive the app client it names.
    "${KUBECTL[@]}" delete secret prebid-artf-credential -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

    # The second seat's demand. Created by this script rather than by the stack,
    # so the stack's deletion does not take it, and a simulator left running
    # after teardown would outlive the Prebid deployment that was its only caller.
    "${KUBECTL[@]}" delete deployment amt-simulator -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    "${KUBECTL[@]}" delete service    amt-simulator -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
    "${KUBECTL[@]}" delete configmap  amt-simulator-code -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true

    # Return the orchestrator to its pre-Prebid authorization behaviour. Leaving the
    # requirement in place would keep an existing component altered by a feature that
    # is no longer installed -- and its only machine caller has just been deleted.
    if "${KUBECTL[@]}" get deployment orchestrator -n "${NAMESPACE}" >/dev/null 2>&1; then
      log "Clearing the orchestrator's machine-scope requirement..."
      "${KUBECTL[@]}" set env deployment/orchestrator -n "${NAMESPACE}" \
        ARTF_MUTATIONS_REQUIRED_SCOPE- >/dev/null 2>&1 \
        || "${KUBECTL[@]}" set env deployment/orchestrator -n "${NAMESPACE}" \
             "ARTF_MUTATIONS_REQUIRED_SCOPE=" >/dev/null 2>&1 \
        || warn "Could not clear ARTF_MUTATIONS_REQUIRED_SCOPE on the orchestrator. Clear it by hand, or the mutations routes keep requiring a scope no deployed client holds."
    fi
  else
    warn "No usable kubectl context; skipping the Kubernetes half. Re-run with a context configured if pods remain."
  fi

  # No RTB Fabric link exists in this topology (Prebid runs in-cluster and reaches
  # the orchestrator over service DNS). FR-35 asks for the link to be deleted before
  # the stacks; checked rather than assumed, so a future topology change surfaces here.
  if aws ssm get-parameter --name "/${STACK_NAME}/fabric-link/link-id" --region "${AWS_REGION}" >/dev/null 2>&1; then
    warn "An RTB Fabric link id is recorded at /${STACK_NAME}/fabric-link/link-id."
    warn "This deployment does not create one. Delete it manually before the stacks if it is live."
  else
    log "No RTB Fabric link recorded - none is created by this topology."
  fi

  if aws cloudformation describe-stacks --stack-name "${PREBID_STACK}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Deleting stack ${PREBID_STACK}..."
    aws cloudformation delete-stack --stack-name "${PREBID_STACK}" --region "${AWS_REGION}"
    aws cloudformation wait stack-delete-complete --stack-name "${PREBID_STACK}" --region "${AWS_REGION}" \
      || warn "Stack ${PREBID_STACK} did not finish deleting. Check its events."
    log "Stack ${PREBID_STACK}: deleted"
  else
    log "Stack ${PREBID_STACK}: already absent"
  fi

  # The ECR repository is deleted HERE and not by the stack, because this script
  # creates it in step 5 -- the image has to be pushable before the stack exists.
  # --force is required: a repository holding images will not delete without it, and
  # every image in it is rebuildable from the pinned release.
  if aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Deleting ECR repository ${ECR_REPO} and its images..."
    aws ecr delete-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" --force >/dev/null \
      && log "ECR repository ${ECR_REPO}: deleted" \
      || warn "Could not delete ECR repository ${ECR_REPO}; remove it by hand to stop its storage charge."
  else
    log "ECR repository ${ECR_REPO}: already absent"
  fi

  log "The shared Cognito user pool was NOT deleted - the frontend and orchestrator use it."
  rm -f "${RECORD_FILE}"
  log "Teardown complete"
  exit 0
fi

# =============================================================================
# STEP 1 - PREFLIGHT
#
# Every missing prerequisite is collected and reported TOGETHER. Failing on the
# first one turns a five-minute fix into five sequential five-minute fixes.
# =============================================================================
if [[ "${START_AT}" -le 1 ]]; then
  log "Step 1/8: Preflight"
  MISSING=()

  for tool in aws kubectl python3 curl tar; do
    command -v "${tool}" >/dev/null 2>&1 || MISSING+=("${tool} is not on PATH")
  done

  ACCOUNT_ID=""
  if ! ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"; then
    MISSING+=("AWS credentials are not valid (aws sts get-caller-identity failed)")
  fi

  if [[ -n "${ACCOUNT_ID}" ]]; then
    aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 \
      || MISSING+=("EKS cluster '${CLUSTER_NAME}' not found - run ./deploy.sh first, or pass --cluster")

    aws codebuild batch-get-projects --names "${CB_PROJECT}" --region "${AWS_REGION}" \
      --query 'projects[0].name' --output text 2>/dev/null | grep -q "${CB_PROJECT}" \
      || MISSING+=("CodeBuild project '${CB_PROJECT}' not found - it is created by ./deploy.sh. The build always runs on CodeBuild; there is no local-build fallback.")

    if [[ -z "${USER_POOL_ID}" ]]; then
      USER_POOL_ID="$(aws cognito-idp list-user-pools --max-results 60 --region "${AWS_REGION}" \
        --query "UserPools[?Name=='${STACK_NAME}-users'].Id | [0]" --output text 2>/dev/null || echo '')"
      [[ "${USER_POOL_ID}" == "None" ]] && USER_POOL_ID=""
    fi
    [[ -n "${USER_POOL_ID}" ]] \
      || MISSING+=("Cognito user pool '${STACK_NAME}-users' not found - pass --user-pool-id, or run ./deploy.sh first")
  fi

  kubectl config current-context >/dev/null 2>&1 \
    || MISSING+=("no kubectl context - run: aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${AWS_REGION}")

  # CDK bootstrap: CHECKED, never created. Creating it silently provisions a bucket,
  # an ECR repo and IAM roles in the operator's account as a side effect of a flag
  # they set for something else.
  if [[ -n "${ACCOUNT_ID}" ]]; then
    if aws cloudformation describe-stacks --stack-name CDKToolkit --region "${AWS_REGION}" >/dev/null 2>&1; then
      log "  CDK bootstrap: present (not required by this deployment)"
    else
      log "  CDK bootstrap: absent. Not created - this deployment uses CloudFormation directly and does not need it."
    fi
  fi

  for required_file in "${CFN_TEMPLATE}" "${K8S_MANIFEST}" "${CONFIG_TEMPLATE}"; do
    [[ -f "${required_file}" ]] || MISSING+=("missing repository file: ${required_file}")
  done

  # ---------------------------------------------------------------- capacity
  # Measured, not assumed. A cluster with no room schedules nothing, and the resulting
  # Pending pod reports "insufficient cpu" long after the operator has been told the
  # deployment succeeded. The requests themselves and the parsing live in
  # prebid_release.py, so this check and the manifest cannot drift apart unnoticed.
  CAPACITY_KNOWN=0
  if kubectl config current-context >/dev/null 2>&1; then
    NODES_JSON="${WORK_DIR}/nodes.json"
    mkdir -p "${WORK_DIR}"
    # Pinned before reading nodes, so capacity is measured on the cluster this run
    # targets rather than on whichever one the shared kubeconfig happens to point at.
    # NOT silenced: if this falls back to an unrelated context, that is precisely the
    # thing the operator needs to see, and it also explains a capacity verdict that
    # would otherwise look like it came from a cluster that does not exist.
    resolve_kube_context || true
    if "${KUBECTL[@]}" get nodes -o json >"${NODES_JSON}" 2>/dev/null; then
      FITS="$(python3 -c '
import json, sys
sys.path.insert(0, sys.argv[1])
import prebid_release as p
with open(sys.argv[2]) as fh:
    nodes = json.load(fh).get("items", [])
verdict = p.node_fits(nodes)
print("yes" if verdict is True else "no" if verdict is False else "unknown")
print(p.POD_CPU_REQUEST_M, p.POD_MEMORY_REQUEST_MI)
' "${SCRIPT_DIR}/scripts" "${NODES_JSON}" 2>/dev/null)" || FITS=$'unknown\n? ?'
      NEEDED="$(printf '%s' "${FITS}" | sed -n '2p')"
      case "$(printf '%s' "${FITS}" | sed -n '1p')" in
        yes)
          CAPACITY_KNOWN=1
          log "  Capacity: at least one node is large enough for the Prebid pod (needs ${NEEDED% *}m CPU, ${NEEDED#* }Mi memory)."
          log "            That is node ALLOCATABLE, not free capacity - it does not subtract what other pods"
          log "            already request. If the pod stays Pending, scale the CPU nodegroup."
          ;;
        no)
          MISSING+=("no single node is large enough for the Prebid pod (needs ${NEEDED% *}m CPU and ${NEEDED#* }Mi memory on one node)")
          ;;
        *)
          warn "Could not measure cluster capacity. If the pod stays Pending after deploy, that is why."
          ;;
      esac
    else
      warn "Could not read cluster nodes. Capacity is unverified."
    fi
  fi

  # The node-cost term for the disclosure. Set to 0.00 only when a node was
  # measured as large enough, so the figure is never asserted without evidence.
  if [[ "${CAPACITY_KNOWN}" -eq 1 ]]; then
    ADDITIONAL_NODE_MONTHLY_USD="0.0"
  fi

  if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Preflight found ${#MISSING[@]} problem(s). All of them, so they can be fixed in one pass:"
    for m in "${MISSING[@]}"; do printf '  - %s\n' "${m}" >&2; done
    fail "Preflight failed"
  fi
  log "  Preflight passed"
  log "  Account=${ACCOUNT_ID} Region=${AWS_REGION} Cluster=${CLUSTER_NAME} UserPool=${USER_POOL_ID}"
else
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
  if [[ -z "${USER_POOL_ID}" ]]; then
    USER_POOL_ID="$(aws cognito-idp list-user-pools --max-results 60 --region "${AWS_REGION}" \
      --query "UserPools[?Name=='${STACK_NAME}-users'].Id | [0]" --output text 2>/dev/null || echo '')"
    [[ "${USER_POOL_ID}" == "None" ]] && USER_POOL_ID=""
  fi
fi

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# =============================================================================
# STEP 2 - COST DISCLOSURE, BEFORE ACQUISITION
#
# Before anything is downloaded, built or provisioned. A disclosure printed after
# the stack exists is not a disclosure.
# =============================================================================
if [[ "${START_AT}" -le 2 ]]; then
  log "Step 2/8: Idle cost"
  echo
  python3 - "${ADDITIONAL_NODE_MONTHLY_USD}" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["SCRIPT_DIR"], "scripts"))
import prebid_release as p

raw = sys.argv[1] if len(sys.argv) > 1 else ""
node = float(raw) if raw not in ("", "None") else None
print(p.format_cost_disclosure(p.build_cost_disclosure(node)))
PY
  echo

  if [[ "${ASSUME_YES}" -ne 1 ]]; then
    # Gated on stdin being a TERMINAL, not on /dev/tty existing.
    #
    # `-e /dev/tty` is true even when stdin is a pipe -- /dev/tty is the controlling
    # terminal of the process, which a piped script still has. Reading from it then
    # blocks forever with nothing attached to answer, so a piped or CI invocation HUNG
    # here instead of failing. Observed while testing this script. A deployment that
    # hangs is worse than one that stops, because nothing says why.
    if [[ ! -t 0 ]]; then
      fail "Cost confirmation needs an interactive terminal (stdin is not a tty). Re-run with --yes to accept the disclosure non-interactively."
    fi
    printf '\033[0;33m[prebid]\033[0m Proceed and provision these resources? [y/N]: '
    read -r REPLY || REPLY="n"
    case "${REPLY}" in
      y|Y|yes|YES) log "  Confirmed" ;;
      *) log "  Declined. Nothing has been provisioned."; exit 0 ;;
    esac
  else
    log "  --yes given; disclosure accepted without prompting"
  fi
fi

# =============================================================================
# STEP 3 - ACQUIRE THE PINNED RELEASE
#
# Pinned and unforked. The archive root is DETECTED rather than assumed: a
# versioned GitHub tarball nests everything under a directory named for the tag,
# and a wrong guess here fails later, during placement or build, with an error
# that points at the wrong thing.
# =============================================================================
if [[ "${START_AT}" -le 3 ]]; then
  log "Step 3/8: Acquire upstream release ${PINNED_VERSION}"
  rm -rf "${WORK_DIR}"
  mkdir -p "${WORK_DIR}"
  TARBALL="${WORK_DIR}/upstream.tar.gz"

  log "  Fetching ${UPSTREAM_TARBALL}"
  curl -fsSL --retry 3 -o "${TARBALL}" "${UPSTREAM_TARBALL}" \
    || fail "Could not download the pinned release. Check network access to github.com, and that tag ${PINNED_VERSION} still exists."
  [[ -s "${TARBALL}" ]] || fail "Downloaded archive is empty"

  ENTRIES="${WORK_DIR}/entries.txt"
  tar -tzf "${TARBALL}" >"${ENTRIES}" || fail "Downloaded file is not a valid gzip tar archive"

  ARCHIVE_ROOT="$(python3 - "${ENTRIES}" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["SCRIPT_DIR"], "scripts"))
import prebid_release as p
with open(sys.argv[1]) as fh:
    print(p.normalise_archive_root(line for line in fh))
PY
)"
  log "  Archive root: ${ARCHIVE_ROOT}"

  tar -xzf "${TARBALL}" -C "${WORK_DIR}"
  if [[ "${ARCHIVE_ROOT}" == "." ]]; then
    UPSTREAM_DIR="${WORK_DIR}"
  else
    UPSTREAM_DIR="${WORK_DIR}/${ARCHIVE_ROOT}"
  fi
  [[ -d "${UPSTREAM_DIR}" ]] || fail "Expected checkout at ${UPSTREAM_DIR}, which does not exist"

  # Recorded as data. The release assets carry no pre-built container image, which is
  # why the from-source build below is mandatory rather than an optimisation.
  log "  Release carries a pre-built image: $(python3 -c 'import sys; sys.path.insert(0, "'"${SCRIPT_DIR}"'/scripts"); import prebid_release as p; print("yes" if p.RELEASE_CARRIES_PREBUILT_IMAGE else "no")')"
  echo "${UPSTREAM_DIR}" >"${WORK_DIR}/upstream_dir.txt"
  log "  Acquired at ${UPSTREAM_DIR}"
fi

UPSTREAM_DIR="$(cat "${WORK_DIR}/upstream_dir.txt" 2>/dev/null || echo '')"
[[ -n "${UPSTREAM_DIR}" && -d "${UPSTREAM_DIR}" ]] \
  || fail "No acquired release found at ${WORK_DIR}. Re-run without --start-at, or with --start-at=3."

BUILD_CONTEXT="${UPSTREAM_DIR}/deployment/ecr/prebid-server"

# =============================================================================
# STEP 4 - PLACE OUR SOURCE, ADDITIVELY
#
# Files are ADDED under the build context's extension path. No upstream file is
# edited, which is what makes "no fork" a checkable property rather than a claim:
# `git diff` against the release would show additions only.
#
# The destination is VERIFIED to exist before anything is copied. Placing source
# into a path the upstream build does not read produces a container that starts,
# serves auctions, and contains none of our code -- the exact failure that looks
# like success.
# =============================================================================
if [[ "${START_AT}" -le 4 ]]; then
  log "Step 4/8: Place the ARTF hook module and artfhouse adapter"

  [[ -d "${BUILD_CONTEXT}" ]] \
    || fail "Upstream build context not found at ${BUILD_CONTEXT}. The release layout has changed; re-check it before deploying."
  [[ -f "${BUILD_CONTEXT}/Dockerfile" ]] \
    || fail "No Dockerfile in ${BUILD_CONTEXT}. The release layout has changed."

  # -------------------------------------------------------------------------------
  # WHERE OUR SOURCE GOES, AND WHY IT IS NOT extra-modules/
  #
  # extra-modules/ is real, but placing a directory there does NOT get it compiled.
  # `COPY extra-modules extra-modules/` puts it in the build context and that is all:
  # the upstream Dockerfile then names, by hand, three paths under
  # extra-modules/log-module-reporter/ and copies those into the Prebid source tree.
  # It does not iterate the directory. A module added alongside would be carried into
  # the build context and never reach Maven -- the image builds, starts, serves
  # auctions, and contains none of our code.
  #
  # Nor is Prebid's own module system in play. prebid-server-java does have one
  # (extra/modules with an all-modules aggregator, extra/bundle producing a fat jar),
  # but this Dockerfile runs `mvn clean package` in the ROOT pom and ships
  # target/prebid-server.jar -- PBS-Core only, with no modules at all.
  #
  # The one extension point the Dockerfile offers is a script WE supply:
  #
  #   ARG INCLUDE_AMT_BIDDER=false
  #   if [ "$INCLUDE_AMT_BIDDER" = "true" ]; then
  #       cp -r ../amt-bidder . && chmod +x ../amt-bidder/copy-bidder-files.sh && \
  #       ../amt-bidder/copy-bidder-files.sh ; fi && mvn clean package ...
  #
  # amt-bidder/ ships with only a .gitkeep; its contents are the deployer's, exactly
  # as the guidance's own --deploy-bidding-simulator uses it. The script runs with the
  # working directory at the prebid-server-java checkout and BEFORE the Maven build,
  # so it can place both the artfhouse adapter and the ARTF hook module into
  # src/main/java/... . No upstream file is edited, so U1-NFR-16 holds.
  #
  # Establishing that route was the NFR-6 verification gate's job; see
  # -------------------------------------------------------------------------------
  INJECT_DIR="${BUILD_CONTEXT}/amt-bidder"
  PREBID_SRC="${REPO_ROOT}/source/prebid"

  mkdir -p "${INJECT_DIR}"

  if [[ -d "${PREBID_SRC}" ]]; then
    # Copied wholesale. The internal layout, and what copy-bidder-files.sh does with
    # it, is the contract of the units that own the Java source -- not this script's
    # to interpret.
    cp -R "${PREBID_SRC}/." "${INJECT_DIR}/"
    log "  Placed ${PREBID_SRC} into the build context's amt-bidder/ injection slot"
  else
    warn "No ${PREBID_SRC} directory - nothing to inject."
  fi

  # --------------------------------------------------- the release's AMT bidder
  # The second seat. Its sources come from the release we already fetched, and are
  # NEVER copied into source/prebid/: that directory is ours and MIT-0, the release
  # is Apache-2.0, and vendoring it here would create a NOTICE obligation this repo
  # does not carry. They live only in the throwaway build context.
  #
  # Placed in a SUBDIRECTORY rather than beside our files, because upstream's own
  # copy-bidder-files.sh has exactly the same name as ours and a flat copy would
  # have one silently overwrite the other -- and whichever won, the build would
  # still succeed while injecting only half of what was intended.
  #
  # Our script places these files itself, per upstream's destination map. We do not
  # invoke theirs: it resolves sources as ./amt-bidder/<file> relative to the
  # checkout, which is a path that only exists because the Dockerfile copies the
  # slot in wholesale, and our slot has a different internal shape.
  AMT_SRC="${UPSTREAM_DIR}/source/loadtest/amt-bidder"
  if [[ "${WITH_SIMULATOR}" -eq 1 ]]; then
    if [[ -d "${AMT_SRC}" ]]; then
      mkdir -p "${INJECT_DIR}/upstream-amt-bidder"
      # Only the files the MAIN build needs. Test sources and upstream's append to
      # src/test/resources/.../test-application.properties are deliberately left
      # out: MVN_CLI_OPTIONS carries -Dmaven.test.skip so no test source is ever
      # compiled, and that append would MODIFY an upstream file, which is the one
      # thing the additions-only property forbids.
      for f in amt.json amt.yaml AmtBidder.java AmtConfiguration.java ExtImpAmt.java; do
        [[ -f "${AMT_SRC}/${f}" ]] \
          || fail "The release is missing source/loadtest/amt-bidder/${f}, so the amt seat cannot be injected. Re-run with --no-simulator to deploy a single-seat auction."
        cp "${AMT_SRC}/${f}" "${INJECT_DIR}/upstream-amt-bidder/${f}"
      done
      log "  Placed the release's AMT bidder (5 main sources) into upstream-amt-bidder/"
    else
      fail "No ${AMT_SRC} in the fetched release, so there is no second seat to inject. Re-run with --no-simulator for a single-seat auction."
    fi
  else
    log "  --no-simulator: the amt seat is NOT injected. The auction will have one"
    log "  seat (artfhouse) and therefore no competing bid."
  fi

  # The build argument is only switched on when this script is present, so the
  # difference between an ARTF image and a stock one is a single observable fact
  # rather than an assumption.
  if [[ -f "${INJECT_DIR}/copy-bidder-files.sh" ]]; then
    log "  copy-bidder-files.sh present: the build WILL inject ARTF sources"
    rm -f "${WORK_DIR}/no_artf_code.txt"
  else
    warn "-----------------------------------------------------------------------"
    warn "NO ARTF CODE WILL BE IN THIS IMAGE."
    warn ""
    warn "amt-bidder/copy-bidder-files.sh is absent, so the build runs with"
    warn "INCLUDE_AMT_BIDDER=false and injects nothing. The image will be stock"
    warn "upstream Prebid Server: it starts, serves auctions, and applies no ARTF"
    warn "mutations."
    warn ""
    warn "That script is produced by the units that own the Java source. Continuing,"
    warn "because the stack, the credential path and the demand endpoint are worth"
    warn "deploying and verifying on their own."
    warn "-----------------------------------------------------------------------"
    echo "amt-bidder/copy-bidder-files.sh absent" >"${WORK_DIR}/no_artf_code.txt"
  fi

  # No upstream file is edited: additions only, which is what makes "no fork" a
  # checkable property rather than a claim.
  log "  Upstream files modified by this step: 0 (additions only)"
fi

# =============================================================================
# STEP 5 - BUILD ON CODEBUILD
#
# Always CodeBuild, never local Docker.
#
# NOTE ON THE JDK. The design recorded that the CodeBuild image supplies the JDK via
# runtime-versions, and that is true of the image -- but it is not what compiles this
# project. The upstream Dockerfile installs amazon-corretto-21 and maven INSIDE its own
# build stage, clones prebid-server-java at the tag pinned in
# docker-build-config.json, and runs Maven there. So the only toolchain CodeBuild needs
# is Docker, which the project already has (PrivilegedMode: true on
# BUILD_GENERAL1_2XLARGE). No runtime-versions block is set below, because declaring a
# Java runtime that nothing uses would imply this build depends on it.
#
# What the build DOES need from the environment is nothing beyond ECR credentials.
# =============================================================================
if [[ "${START_AT}" -le 5 && "${SKIP_BUILD}" -ne 1 ]]; then
  log "Step 5/8: Build the image on CodeBuild"

  if [[ -z "${IMAGE_TAG}" ]]; then
    SRC_HASH="$(find "${BUILD_CONTEXT}" -type f -print0 2>/dev/null | sort -z | xargs -0 shasum -a 256 2>/dev/null | shasum -a 256 | cut -c1-12)"
    IMAGE_TAG="${PINNED_VERSION}-${SRC_HASH:-manual}"
  fi
  log "  Image tag: ${IMAGE_TAG}"

  # ---------------------------------------------------------------------------
  # THIS SCRIPT OWNS THE ECR REPOSITORY, and it has to.
  #
  # The image is built and pushed here in step 5; the CloudFormation stack is not
  # created until step 6. A repository declared in prebid_cfn.yaml would therefore
  # not exist when the push needs it. It was declared in both places at first, and
  # CloudFormation rejected the whole stack before creating anything:
  #
  #   The following hook(s)/validation failed:
  #   [AWS::EarlyValidation::ResourceExistenceCheck]
  #
  # The lifecycle policy came across with it rather than being dropped, so image
  # retention still behaves as designed. --destroy deletes the repository, because
  # the stack no longer does.
  # ---------------------------------------------------------------------------
  aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
         --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE >/dev/null

  # Keep the 5 most recent images. Older ones are rebuildable from the pinned release,
  # so retaining them indefinitely only accrues per-GB-month storage.
  aws ecr put-lifecycle-policy --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
    --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Keep the 5 most recent images; older ones are rebuildable from the pinned release.","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' \
    >/dev/null 2>&1 || warn "Could not set the ECR lifecycle policy on ${ECR_REPO}; images will accumulate"

  # ---------------------------------------------------------------------------
  # LET THE CODEBUILD ROLE PUSH TO THIS REPOSITORY.
  #
  # The role created by codebuild_cfn.yaml scopes its ECR grant to
  # repository/${StackName}-* -- i.e. "<prefix>-nvidia-artf-recommenders-*". This
  # repository is "<prefix>-prebid-server", which does not match, so the build
  # produced a correct image and then failed on `docker push`:
  #
  #   denied: ... not authorized to perform: ecr:InitiateLayerUpload on resource:
  #   .../repository/<prefix>-prebid-server
  #
  # Fixed with a REPOSITORY policy rather than by widening the shared role, for two
  # reasons. First, the existing CodeBuild role is left byte-identical, so a
  # deployment that never opts into Prebid is unaffected -- widening a shared role
  # would hand every such deployment a grant for a repository it does not have.
  # Second, the permission is created with the repository and dies with it, instead
  # of outliving what it was for.
  #
  # Sufficient on its own: ECR evaluates identity and repository policies together
  # and an action allowed by EITHER is allowed
  # (docs.aws.amazon.com/AmazonECR/latest/userguide/repository-policies.html).
  # ecr:GetAuthorizationToken is the documented exception -- it must come from an
  # identity policy -- and the CodeBuild role already holds it on "*".
  #
  # Scoped to one principal, one repository, and only the actions a push makes.
  # ---------------------------------------------------------------------------
  CODEBUILD_ROLE_ARN="$(aws codebuild batch-get-projects --names "${CB_PROJECT}" \
      --region "${AWS_REGION}" --query 'projects[0].serviceRole' --output text 2>/dev/null)"

  if [[ -n "${CODEBUILD_ROLE_ARN}" && "${CODEBUILD_ROLE_ARN}" != "None" ]]; then
    cat >"${WORK_DIR}/ecr-repo-policy.json" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCodeBuildPush",
      "Effect": "Allow",
      "Principal": { "AWS": "${CODEBUILD_ROLE_ARN}" },
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ]
    }
  ]
}
POLICY
    aws ecr set-repository-policy --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
      --policy-text "file://${WORK_DIR}/ecr-repo-policy.json" >/dev/null \
      && log "  ECR repository policy set: ${CODEBUILD_ROLE_ARN##*/} may push to ${ECR_REPO}" \
      || warn "Could not set the ECR repository policy; the build will fail on docker push"
  else
    # Named plainly instead of continuing into a build whose push cannot succeed.
    warn "Could not resolve the CodeBuild service role for ${CB_PROJECT}."
    warn "Without it this repository has no push grant and the build will fail at docker push."
  fi

  BUILDSPEC="${WORK_DIR}/buildspec.yml"
  cat >"${BUILDSPEC}" <<'SPEC'
version: 0.2
# The upstream Dockerfile is a multi-stage build that installs its own JDK 21 and
# Maven, clones prebid-server-java at the tag in docker-build-config.json, and
# compiles there. This buildspec therefore only needs Docker and an ECR login.
phases:
  pre_build:
    commands:
      - echo "Logging in to ECR ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
      - aws ecr get-login-password --region "${AWS_DEFAULT_REGION}" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
      - test -f Dockerfile || { echo "No Dockerfile in the build context"; exit 1; }
      - test -f docker-build-config.json || { echo "No docker-build-config.json; this is not the upstream build context"; exit 1; }
      # Recorded in the build log so the image can be traced to a Prebid version
      # without unpacking it.
      - echo "prebid-server-java tag:" $(jq -r .GIT_TAG_VERSION docker-build-config.json 2>/dev/null || echo unknown)
      - ls -la extra-modules 2>/dev/null || echo "no extra-modules directory"
  build:
    commands:
      - echo "Building ${ECR_REPO}:${IMAGE_TAG} from the pinned upstream build context"
      # INCLUDE_AMT_BIDDER is the upstream Dockerfile's ONLY hook for injecting our own
      # sources without editing one of its files: when true it runs
      # amt-bidder/copy-bidder-files.sh, which we supply, with the working directory at
      # the prebid-server-java checkout and BEFORE `mvn clean package`.
      #
      # Gated on the script actually existing. The directory ships with only a
      # .gitkeep, so turning this on without the script fails the build -- and turning
      # it on automatically once the script is there is what stops a silent
      # ARTF-free image.
      - |
        if [ -f amt-bidder/copy-bidder-files.sh ]; then
          echo "copy-bidder-files.sh present: building WITH source injection"
          docker build --build-arg INCLUDE_AMT_BIDDER=true -t "${ECR_REPO}:${IMAGE_TAG}" .
        else
          echo "WARNING: no amt-bidder/copy-bidder-files.sh."
          echo "WARNING: building stock upstream Prebid Server - NO ARTF code in this image."
          docker build -t "${ECR_REPO}:${IMAGE_TAG}" .
        fi
      - docker tag "${ECR_REPO}:${IMAGE_TAG}" "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
  post_build:
    commands:
      - docker push "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
      - echo "Pushed ${ECR_REPO}:${IMAGE_TAG}"
SPEC

  SOURCE_ZIP="${WORK_DIR}/prebid-build-context.zip"
  rm -f "${SOURCE_ZIP}"
  (cd "${BUILD_CONTEXT}" && zip -qr "${SOURCE_ZIP}" . -x '*/.git/*' -x '*.DS_Store')
  zip -qj "${SOURCE_ZIP}" "${BUILDSPEC}"

  SOURCE_BUCKET="${STACK_NAME}-codebuild-source-${ACCOUNT_ID}"
  if ! aws s3api head-bucket --bucket "${SOURCE_BUCKET}" 2>/dev/null; then
    log "  Creating source bucket ${SOURCE_BUCKET}"
    if [[ "${AWS_REGION}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "${SOURCE_BUCKET}" --region "${AWS_REGION}" >/dev/null
    else
      aws s3api create-bucket --bucket "${SOURCE_BUCKET}" --region "${AWS_REGION}" \
        --create-bucket-configuration LocationConstraint="${AWS_REGION}" >/dev/null
    fi
  fi

  S3_KEY="prebid-builds/${IMAGE_TAG}/source.zip"
  aws s3 cp "${SOURCE_ZIP}" "s3://${SOURCE_BUCKET}/${S3_KEY}" --region "${AWS_REGION}" >/dev/null \
    || fail "Could not upload the build context to s3://${SOURCE_BUCKET}/${S3_KEY}"
  log "  Uploaded build context to s3://${SOURCE_BUCKET}/${S3_KEY}"

  ENV_OVERRIDES="[
    {\"name\":\"ECR_REPO\",\"value\":\"${ECR_REPO}\",\"type\":\"PLAINTEXT\"},
    {\"name\":\"IMAGE_TAG\",\"value\":\"${IMAGE_TAG}\",\"type\":\"PLAINTEXT\"},
    {\"name\":\"AWS_ACCOUNT_ID\",\"value\":\"${ACCOUNT_ID}\",\"type\":\"PLAINTEXT\"},
    {\"name\":\"AWS_DEFAULT_REGION\",\"value\":\"${AWS_REGION}\",\"type\":\"PLAINTEXT\"}
  ]"

  BUILD_ID="$(aws codebuild start-build \
    --project-name "${CB_PROJECT}" \
    --source-type-override S3 \
    --source-location-override "${SOURCE_BUCKET}/${S3_KEY}" \
    --buildspec-override "buildspec.yml" \
    --environment-variables-override "${ENV_OVERRIDES}" \
    --region "${AWS_REGION}" \
    --query 'build.id' --output text)"
  log "  Build started: ${BUILD_ID}"
  log "  Console: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${BUILD_ID}?region=${AWS_REGION}"

  POLL_START="$(date +%s)"
  while true; do
    BUILD_STATUS="$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${AWS_REGION}" \
      --query 'builds[0].buildStatus' --output text)"
    PHASE="$(aws codebuild batch-get-builds --ids "${BUILD_ID}" --region "${AWS_REGION}" \
      --query 'builds[0].currentPhase' --output text 2>/dev/null || echo UNKNOWN)"
    ELAPSED=$(( $(date +%s) - POLL_START ))
    case "${BUILD_STATUS}" in
      SUCCEEDED) log "  Build SUCCEEDED after $((ELAPSED / 60))m$((ELAPSED % 60))s"; break ;;
      FAILED|FAULT|TIMED_OUT|STOPPED)
        fail "Build ${BUILD_STATUS}. Logs: https://${AWS_REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${CB_PROJECT}/build/${BUILD_ID}?region=${AWS_REGION}" ;;
      *) printf '\033[0;32m[prebid]\033[0m   %s  phase=%-18s elapsed=%dm%02ds\n' "${BUILD_STATUS}" "${PHASE}" "$((ELAPSED / 60))" "$((ELAPSED % 60))"; sleep 15 ;;
    esac
  done
  echo "${IMAGE_TAG}" >"${WORK_DIR}/image_tag.txt"
elif [[ "${SKIP_BUILD}" -eq 1 ]]; then
  log "Step 5/8: --skip-build; reusing the image already in ECR"
fi

if [[ -z "${IMAGE_TAG}" ]]; then
  IMAGE_TAG="$(cat "${WORK_DIR}/image_tag.txt" 2>/dev/null || echo '')"
fi
[[ -n "${IMAGE_TAG}" ]] || fail "No image tag known. Pass --tag, or run without --skip-build."

# =============================================================================
# STEP 6 - DEPLOY
#
# CloudFormation first (it creates the IRSA role and the secret the pod needs),
# then the Kubernetes manifest, then an EXPLICIT rollout restart.
#
# The restart is not redundant. `kubectl apply` only rolls pods when the manifest
# text changes; a rebuilt image under an unchanged tag leaves the old pods running,
# which is a defect this repository has hit before.
# =============================================================================
if [[ "${START_AT}" -le 6 ]]; then
  log "Step 6/8: Deploy"

  OIDC_ISSUER="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.identity.oidc.issuer' --output text)"
  OIDC_HOST="${OIDC_ISSUER#https://}"
  OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/${OIDC_HOST}"
  aws iam get-open-id-connect-provider --open-id-connect-provider-arn "${OIDC_ARN}" >/dev/null 2>&1 \
    || fail "No IAM OIDC provider for this cluster (${OIDC_ARN}). IRSA cannot work without it. Create it with: eksctl utils associate-iam-oidc-provider --cluster ${CLUSTER_NAME} --approve"

  # The Cognito hosted domain prefix must be globally unique and DNS-safe.
  DOMAIN_PREFIX="$(printf '%s' "${PFX}artf-prebid-${ACCOUNT_ID}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | cut -c1-63)"

  # Package the demand endpoint handler if its source is present. Absent, the stack
  # deploys a placeholder that returns 501 - an honest "not deployed", never a
  # synthetic bid response.
  DEMAND_BUCKET=""
  DEMAND_KEY=""
  if [[ -d "${REPO_ROOT}/source/demand" ]]; then
    DEMAND_ZIP="${WORK_DIR}/artfhouse-demand.zip"
    rm -f "${DEMAND_ZIP}"
    (cd "${REPO_ROOT}/source" && zip -qr "${DEMAND_ZIP}" demand -x '*__pycache__*' -x '*.pyc')
    # The key must be a function of the demand SOURCE, not of IMAGE_TAG. IMAGE_TAG
    # hashes the Prebid Java build context, so a Python-only change to source/demand
    # left the key identical, CloudFormation saw no change to the function's Code
    # property, and the Lambda kept running the previous package. Hashed over paths
    # relative to source/ so the key does not move with the checkout directory.
    DEMAND_HASH="$(cd "${REPO_ROOT}/source" \
      && find demand -type f ! -name '*.pyc' ! -path '*__pycache__*' -print0 \
      | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-12)"
    [[ -n "${DEMAND_HASH}" ]] \
      || fail "Could not hash source/demand. Refusing to upload under a fixed key, which would silently pin the Lambda to a stale package."
    DEMAND_BUCKET="${STACK_NAME}-codebuild-source-${ACCOUNT_ID}"
    DEMAND_KEY="prebid-lambda/demand-${DEMAND_HASH}/artfhouse-demand.zip"
    aws s3 cp "${DEMAND_ZIP}" "s3://${DEMAND_BUCKET}/${DEMAND_KEY}" --region "${AWS_REGION}" >/dev/null \
      || fail "Could not upload the demand endpoint package"
    log "  Demand endpoint packaged to s3://${DEMAND_BUCKET}/${DEMAND_KEY}"
  else
    warn "No source/demand directory - the stack will deploy the 501 placeholder handler."
  fi

  CONFIG_BUCKET="${STACK_NAME}-codebuild-source-${ACCOUNT_ID}"

  deploy_cfn_stack "${PREBID_STACK}" "${CFN_TEMPLATE}" \
    "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
    "ParameterKey=UserPoolId,ParameterValue=${USER_POOL_ID}" \
    "ParameterKey=DomainPrefix,ParameterValue=${DOMAIN_PREFIX}" \
    "ParameterKey=ClusterOidcProviderArn,ParameterValue=${OIDC_ARN}" \
    "ParameterKey=ClusterOidcIssuerHost,ParameterValue=${OIDC_HOST}" \
    "ParameterKey=Namespace,ParameterValue=${NAMESPACE}" \
    "ParameterKey=ServiceAccountName,ParameterValue=prebid-artf-host-sa" \
    "ParameterKey=ConfigBucket,ParameterValue=${CONFIG_BUCKET}" \
    "ParameterKey=DemandLambdaS3Bucket,ParameterValue=${DEMAND_BUCKET}" \
    "ParameterKey=DemandLambdaS3Key,ParameterValue=${DEMAND_KEY}"

  PREBID_ROLE_ARN="$(stack_output "${PREBID_STACK}" PrebidHostRoleArn)"
  CREDENTIAL_SECRET="$(stack_output "${PREBID_STACK}" CredentialSecretArn)"
  TOKEN_ENDPOINT="$(stack_output "${PREBID_STACK}" TokenEndpoint)"
  DEMAND_ENDPOINT="$(stack_output "${PREBID_STACK}" DemandEndpointUrl)"
  ORCHESTRATOR_SCOPE="$(stack_output "${PREBID_STACK}" OrchestratorScope)"
  DEMAND_SCOPE="$(stack_output "${PREBID_STACK}" DemandScope)"
  # The pod requests BOTH scopes in one client_credentials grant, space-delimited as
  # OAuth2 specifies, because the hook and the adapter share a single TokenCache but
  # call two different authorities.
  ARTF_TOKEN_SCOPES="${ORCHESTRATOR_SCOPE} ${DEMAND_SCOPE}"
  # An empty value here would be substituted into the manifest as an empty string and
  # the pod would start with, say, no scope at all -- which reads as "no scope
  # required" rather than as a deployment fault.
  for pair in "PrebidHostRoleArn:${PREBID_ROLE_ARN}" "CredentialSecretArn:${CREDENTIAL_SECRET}" \
              "TokenEndpoint:${TOKEN_ENDPOINT}" "DemandEndpointUrl:${DEMAND_ENDPOINT}" \
              "OrchestratorScope:${ORCHESTRATOR_SCOPE}" "DemandScope:${DEMAND_SCOPE}"; do
    value="${pair#*:}"
    [[ -n "${value}" && "${value}" != "None" ]] \
      || fail "Stack output ${pair%%:*} is empty. The manifest cannot be wired without it."
  done

  # ---------------------------------------------------------- config, before the pod
  # The image entrypoint fetches prebid-server/default/ and EXITS 1 if any of its four
  # required files is missing. So the default prefix is populated from the release's
  # own default-config/ BEFORE the deployment is applied -- otherwise the first thing
  # the pod does is fail, with an error about a missing file rather than about ordering.
  upload_default_config "${CONFIG_BUCKET}"

  # ------------------------------------------------- the second seat's demand
  # Deployed BEFORE the manifest is rendered and before the config overlay is
  # published, because both need its address. The address is deterministic
  # (a ClusterIP Service name), so nothing has to be read back to learn it.
  deploy_amt_simulator

  PROCESSED="/tmp/${CLUSTER_NAME}-prebid-server-deployment.yaml"
  sed -e "s|__PREBID_ROLE_ARN__|${PREBID_ROLE_ARN}|g" \
      -e "s|__IMAGE__|${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}|g" \
      -e "s|__CONFIG_BUCKET__|${CONFIG_BUCKET}|g" \
      -e "s|__AWS_REGION__|${AWS_REGION}|g" \
      -e "s|__TOKEN_ENDPOINT__|${TOKEN_ENDPOINT}|g" \
      -e "s|__CREDENTIAL_SECRET__|${CREDENTIAL_SECRET}|g" \
      -e "s|__DEMAND_ENDPOINT__|${DEMAND_ENDPOINT}|g" \
      -e "s|__ORCHESTRATOR_URL__|${ORCHESTRATOR_URL}|g" \
      -e "s|__ORCHESTRATOR_SCOPE__|${ORCHESTRATOR_SCOPE}|g" \
      -e "s|__ARTF_TOKEN_SCOPES__|${ARTF_TOKEN_SCOPES}|g" \
      -e "s|__AMT_SIMULATOR_ENDPOINT__|${AMT_SIMULATOR_ENDPOINT:-http://amt-simulator.not-deployed.invalid/}|g" \
      "${K8S_MANIFEST}" >"${PROCESSED}"

  if grep -q '__[A-Z_]*__' "${PROCESSED}"; then
    warn "Unsubstituted placeholders remain in ${PROCESSED}:"
    grep -o '__[A-Z_]*__' "${PROCESSED}" | sort -u >&2
    fail "Refusing to apply a manifest with unsubstituted placeholders"
  fi

  # ---------------------------------------------------------------------------
  # DOES THE DEPLOYED ORCHESTRATOR UNDERSTAND THE SCOPE FLAG?
  #
  # Later in this step the orchestrator's ARTF_MUTATIONS_REQUIRED_SCOPE is set. An
  # orchestrator image built before this feature has no code that reads it, so the
  # flag is a SILENT NO-OP: the variable is present, the deployment rolls, and every
  # machine token is still authorized on scope alone. a token with
  # the wrong scope reached the handler instead of being refused with 403.
  #
  # Checked rather than assumed, and warned about rather than hidden, because a
  # security control that is set but not enforced is worse than one that is absent:
  # it reads as protection.
  # ---------------------------------------------------------------------------
  if [[ ${#KUBECTL[@]} -eq 1 ]]; then resolve_kube_context || true; fi
  ORCH_POD="$("${KUBECTL[@]}" get pods -n "${NAMESPACE}" -l component=orchestrator \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo '')"
  if [[ -n "${ORCH_POD}" ]]; then
    if "${KUBECTL[@]}" exec -n "${NAMESPACE}" "${ORCH_POD}" -- \
         python -c "from orchestrator import auth; raise SystemExit(0 if hasattr(auth,'_ROUTE_SCOPES') else 1)" \
         >/dev/null 2>&1; then
      log "  Orchestrator image supports scope authorization"
    else
      warn "The deployed orchestrator image does NOT contain the scope-authorization code."
      warn "Setting ARTF_MUTATIONS_REQUIRED_SCOPE on it changes nothing: machine tokens"
      warn "will still be authorized without the scope. Rebuild and redeploy the"
      warn "orchestrator image (./deploy.sh) so the flag is actually enforced."
    fi
  fi

  # ---------------------------------------------------------------------------
  # POPULATE THE CREDENTIAL SECRET.
  #
  # prebid_cfn.yaml creates the Secrets Manager secret but cannot fill it: the value
  # is the Cognito app client's id and generated secret, and CloudFormation has no
  # way to read ClientSecret off AWS::Cognito::UserPoolClient and write it into
  # AWS::SecretsManager::Secret. Declared with a Name and no SecretString, the secret
  # exists with NO VERSION, and reading it fails:
  #
  #   ResourceNotFoundException when calling GetSecretValue
  #
  # Which the hook reported honestly rather than crashing -- the live auction came
  # back with outcome "transport_failure", reason "no valid credential; no token
  # held", action no_action. Correct behaviour, and a useless deployment.
  #
  # Written here, after the stack exists and BEFORE the pods start, so the first
  # token fetch has something to read. Idempotent: put-secret-value adds a new
  # version each run, which is what rotating the app client would also do.
  # ---------------------------------------------------------------------------
  PREBID_CLIENT_ID="$(stack_output "${PREBID_STACK}" PrebidHostClientId)"
  CREDENTIAL_SECRET_ARN="$(stack_output "${PREBID_STACK}" CredentialSecretArn)"

  if [[ -n "${PREBID_CLIENT_ID}" && -n "${CREDENTIAL_SECRET_ARN}" ]]; then
    PREBID_CLIENT_SECRET="$(aws cognito-idp describe-user-pool-client \
      --user-pool-id "${USER_POOL_ID}" --client-id "${PREBID_CLIENT_ID}" \
      --region "${AWS_REGION}" --query 'UserPoolClient.ClientSecret' --output text 2>/dev/null)"

    if [[ -n "${PREBID_CLIENT_SECRET}" && "${PREBID_CLIENT_SECRET}" != "None" ]]; then
      # Written via a file so the secret never appears in a process argument list,
      # where `ps` would show it to any other process on the host.
      SECRET_FILE="${WORK_DIR}/credential-secret.json"
      umask 077
      printf '{"client_id":"%s","client_secret":"%s"}\n' \
        "${PREBID_CLIENT_ID}" "${PREBID_CLIENT_SECRET}" >"${SECRET_FILE}"
      aws secretsmanager put-secret-value --secret-id "${CREDENTIAL_SECRET_ARN}" \
        --secret-string "file://${SECRET_FILE}" --region "${AWS_REGION}" >/dev/null \
        && log "  Credential secret populated for client ${PREBID_CLIENT_ID}" \
        || warn "Could not write the credential secret; the hook will report 'no token held'"
      rm -f "${SECRET_FILE}"

      # -------------------------------------------------------------------------
      # DELIVER THE CREDENTIAL TO THE POD AS A KUBERNETES SECRET.
      #
      # The pod cannot read Secrets Manager: the Prebid image is built from the pinned
      # upstream release whose pom declares only the AWS SDK's `s3` module, so the jar
      # has neither `secretsmanager` nor `sts`. Without `sts` the SDK cannot use the
      # pod's IRSA web identity at all, which the container reports on startup:
      #
      #   To use web identity tokens, the 'sts' service module must be on the class path.
      #
      # Adding either dependency means editing the upstream pom -- a fork, forbidden by
      # U1-NFR-16. This script has credentials that CAN read the secret, so it does the
      # read and hands the pod the result. Secrets Manager remains the source of record.
      #
      # Written BEFORE `kubectl apply`, because the Deployment references this Secret by
      # name via secretKeyRef; a missing Secret leaves the pod stuck in
      # CreateContainerConfigError.
      #
      # `--dry-run=client -o yaml | kubectl apply -f -` is the idempotent create-or-update
      # form; `kubectl create secret` alone fails on the second run.
      # -------------------------------------------------------------------------
      if [[ ${#KUBECTL[@]} -eq 1 ]]; then resolve_kube_context || true; fi
      if "${KUBECTL[@]}" create secret generic prebid-artf-credential \
            -n "${NAMESPACE}" \
            --from-literal=client_id="${PREBID_CLIENT_ID}" \
            --from-literal=client_secret="${PREBID_CLIENT_SECRET}" \
            --dry-run=client -o yaml 2>/dev/null \
          | "${KUBECTL[@]}" apply -f - >/dev/null 2>&1; then
        log "  Kubernetes Secret prebid-artf-credential written to namespace ${NAMESPACE}"
      else
        warn "Could not write the Kubernetes Secret prebid-artf-credential."
        warn "The pod will not start: its Deployment references it via secretKeyRef."
      fi
    else
      warn "Cognito app client ${PREBID_CLIENT_ID} returned no ClientSecret."
      warn "The hook cannot authenticate and will report 'no valid credential'."
    fi
  else
    warn "Stack outputs PrebidHostClientId / CredentialSecretArn missing; secret not populated."
  fi

  # The configuration overlay goes up BEFORE the manifest, not in step 7. The pod
  # reads its config at start, and the release default alone contains placeholders
  # this topology never sets (${LOG_ANALYTICS_ENABLED}), which is fatal to Spring.
  # Uploading afterwards left the pod in CrashLoopBackOff and made the rollout wait
  # below unsatisfiable. See publish_config_overlay for the failure it produced.
  publish_config_overlay "${CONFIG_BUCKET}" \
    "${DEMAND_ENDPOINT:-$(stack_output "${PREBID_STACK}" DemandEndpointUrl)}"

  # Re-resolved only when preflight was skipped via --start-at; otherwise KUBECTL is
  # already pinned and this is a no-op that reprints the same line.
  if [[ ${#KUBECTL[@]} -eq 1 ]]; then
    resolve_kube_context || fail "No usable kubectl context; cannot apply the Prebid manifest."
  fi
  "${KUBECTL[@]}" apply -f "${PROCESSED}" || fail "kubectl apply failed for the Prebid manifest"

  log "  Forcing a pod rollout (apply alone does not restart pods when only the image CONTENT changed)"
  "${KUBECTL[@]}" rollout restart deployment/prebid-server -n "${NAMESPACE}" || true
  "${KUBECTL[@]}" rollout status deployment/prebid-server -n "${NAMESPACE}" --timeout=300s \
    || warn "The Prebid deployment did not become ready within 300s. Check: kubectl describe pod -l app=prebid-server -n ${NAMESPACE}"

  # ---------------------------------------------------------------------------
  # Turn ON the orchestrator's machine-scope requirement.
  #
  # The orchestrator requires no scope until something is asked to. That default is
  # deliberate: a stack deployed without Prebid has no machine caller, so it must not
  # carry an authorization rule it has no client for -- its behaviour stays exactly as
  # it was before this feature existed.
  #
  # Set HERE rather than only in deploy.sh because this script also runs standalone,
  # against a cluster whose orchestrator was deployed without --with-prebid.
  # `kubectl set env` triggers its own rollout, so the change takes effect.
  #
  # This does not affect the frontend: a Cognito USER access token is authorized by
  # the user_session mechanism, which does not consult scopes.
  # ---------------------------------------------------------------------------
  if [[ -n "${ORCHESTRATOR_SCOPE}" ]]; then
    if "${KUBECTL[@]}" get deployment orchestrator -n "${NAMESPACE}" >/dev/null 2>&1; then
      log "  Requiring scope '${ORCHESTRATOR_SCOPE}' of machine callers on the orchestrator"
      "${KUBECTL[@]}" set env deployment/orchestrator -n "${NAMESPACE}" \
        "ARTF_MUTATIONS_REQUIRED_SCOPE=${ORCHESTRATOR_SCOPE}" >/dev/null \
        || warn "Could not set ARTF_MUTATIONS_REQUIRED_SCOPE on the orchestrator. The hook will still be authenticated, but its scope will NOT be enforced -- SECURITY-06 would be unmet."
    else
      warn "No orchestrator deployment in namespace ${NAMESPACE}; scope enforcement not enabled."
    fi
  fi
fi

# =============================================================================
# STEP 7 - WIRE THE CONFIGURATION
#
# The config object is uploaded ONLY if absent. Overwriting on every deploy would
# silently discard the operator's edits, and FR-38's whole point is that editing
# this object is the way to reconfigure.
# =============================================================================
if [[ "${START_AT}" -le 7 ]]; then
  log "Step 7/8: Configuration overlay"
  CONFIG_BUCKET="${STACK_NAME}-codebuild-source-${ACCOUNT_ID}"
  DEMAND_ENDPOINT="${DEMAND_ENDPOINT:-$(stack_output "${PREBID_STACK}" DemandEndpointUrl)}"

  # Step 6 already published this, because the pod cannot boot without it. Kept here
  # so `--start-at=7` remains a usable entry point, and idempotent so the normal path
  # simply reports what exists.
  publish_config_overlay "${CONFIG_BUCKET}" "${DEMAND_ENDPOINT}"

  log "  To reconfigure: edit s3://${CONFIG_BUCKET}/${CONFIG_KEY}, then"
  log "    kubectl rollout restart deployment/prebid-server -n ${NAMESPACE}"
  log "  No rebuild is required."
fi

# =============================================================================
# STEP 8 - RECORD WHAT WAS DEPLOYED
#
# Written so a later run, or a reader, can tell exactly which upstream release and
# which image are live without inferring it from tags.
# =============================================================================
if [[ "${START_AT}" -le 8 ]]; then
  log "Step 8/8: Record"
  IMAGE_DIGEST="$(aws ecr describe-images --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
    --image-ids "imageTag=${IMAGE_TAG}" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || echo '')"
  [[ "${IMAGE_DIGEST}" == "None" ]] && IMAGE_DIGEST=""

  # These are normally resolved in step 6. Entering at step 7 or 8 skips that, and
  # recording them as null would overwrite a correct record with a WORSE one -- the
  # file then reports the deployment has no token endpoint and no required scope,
  # which reads as a misconfiguration rather than as an artifact of where the run
  # started. Backfill from the stack so the record is the same wherever we entered.
  for _out in TOKEN_ENDPOINT:TokenEndpoint DEMAND_ENDPOINT:DemandEndpointUrl \
              ORCHESTRATOR_SCOPE:OrchestratorScope; do
    _var="${_out%%:*}"
    if [[ -z "${!_var:-}" ]]; then
      _val="$(stack_output "${PREBID_STACK}" "${_out#*:}" 2>/dev/null || echo '')"
      [[ "${_val}" == "None" ]] && _val=""
      printf -v "${_var}" '%s' "${_val}"
    fi
  done

  python3 - <<PY >"${RECORD_FILE}"
import json
record = {
    "upstream_repo": "${UPSTREAM_REPO}",
    "pinned_version": "${PINNED_VERSION}",
    "unforked": True,
    "topology": "pods on the existing EKS cluster (NOT the upstream ECS Fargate CDK app)",
    "stack_name": "${PREBID_STACK}",
    "cluster": "${CLUSTER_NAME}",
    "namespace": "${NAMESPACE}",
    "region": "${AWS_REGION}",
    "ecr_repository": "${ECR_REPO}",
    "image_tag": "${IMAGE_TAG}",
    "image_digest": "${IMAGE_DIGEST}" or None,
    "config_default_prefix": "s3://${STACK_NAME}-codebuild-source-${ACCOUNT_ID}/${DEFAULT_CONFIG_PREFIX}",
    "config_overlay_object": "s3://${STACK_NAME}-codebuild-source-${ACCOUNT_ID}/${CONFIG_KEY}",
    "artf_code_in_image": $(if [[ -s "${WORK_DIR}/no_artf_code.txt" ]]; then echo 'False'; else echo 'True'; fi),
    "artf_code_note": "$(if [[ -s "${WORK_DIR}/no_artf_code.txt" ]]; then cat "${WORK_DIR}/no_artf_code.txt"; else echo "built with INCLUDE_AMT_BIDDER=true via amt-bidder/copy-bidder-files.sh"; fi)",
    "token_endpoint": "${TOKEN_ENDPOINT:-}" or None,
    "demand_endpoint": "${DEMAND_ENDPOINT:-}" or None,
    "orchestrator_scope": "${ORCHESTRATOR_SCOPE:-}" or None,
}
print(json.dumps(record, indent=2, sort_keys=True))
PY
  log "  Recorded to ${RECORD_FILE}"
  cat "${RECORD_FILE}"
fi

echo
log "Prebid ARTF host deployed."
log "  Stack:    ${PREBID_STACK}"
log "  Image:    ${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"
log "  In-cluster: http://prebid-server.${NAMESPACE}.svc.cluster.local"
log "  Teardown: ${0} --prefix ${STACK_PREFIX:-''} --destroy"
