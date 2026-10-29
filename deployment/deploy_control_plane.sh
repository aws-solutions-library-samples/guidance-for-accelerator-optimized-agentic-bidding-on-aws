#!/usr/bin/env bash
# Requires real bash, not bash in POSIX mode. See the same guard in deploy.sh.
if [ -z "${BASH_VERSION:-}" ]; then echo "deploy_control_plane.sh needs bash" >&2; exit 1; fi
# =============================================================================
# deploy_control_plane.sh - the campaign/deals MCP control plane
#
# Invoked by deploy_prebid.sh as its Step 9 whenever Prebid Server is deployed
# (deploy.sh --with-prebid, or --learner-mode on the branch that has it), unless
# --no-control-plane. Runs standalone too.
#
# WHAT THIS DEPLOYS
#
#   - an ECR repository and an arm64 image built from source/Dockerfile.control-plane
#     (built HERE with Docker buildx: see the Docker note below)
#   - deployment/control_plane_cfn.yaml: one Cognito resource server (four control
#     scopes), one client_credentials app client, and the AgentCore execution role,
#     all attached to the SHARED user pool and the hosted domain prebid_cfn.yaml made
#   - one Bedrock AgentCore Runtime (serverProtocol=MCP) with an inbound JWT
#     authorizer that accepts ONLY that client, and the Authorization header
#     allowlisted so the container can verify the token a second time
#
# What it writes to: the stack's existing deals table (<stack>-deals). The runtime
# is the first thing in this deployment that can change a record; everything else
# reads. The seeder is told about it by deploy.sh (--campaign-status paused
# --skip-managed), not by this script.
#
# DOCKER. The rest of the Prebid path never builds locally so that --with-prebid
# works on a machine without Docker. This step does: the control-plane image is
# arm64 (AgentCore requires it) and small, and the Phase 5 AgentCore image is built
# the same way. When Docker with buildx is absent, preflight reports it and this
# script exits 2; deploy_prebid.sh treats that as a warned skip of Step 9 and the
# rest of the Prebid host is unaffected. --skip-build reuses the image already in
# ECR, and --image URI uses an image built elsewhere.
#
# Steps:
#   1. Preflight   report EVERY missing prerequisite at once
#   2. Build       docker buildx (arm64) -> ECR, skipped when the source hash matches
#   3. Stack       control_plane_cfn.yaml
#   4. Runtime     create or update the AgentCore runtime (authorizer + header allowlist)
#   5. Record      runtime ARN, client id, scopes, token endpoint; NEVER the secret
#
# Usage:
#   ./deploy_control_plane.sh --prefix nv5
#   ./deploy_control_plane.sh --prefix nv5 --control-advertisers acme.example,pub.example
#   ./deploy_control_plane.sh --prefix nv5 --skip-build
#   ./deploy_control_plane.sh --prefix nv5 --destroy
#
# Options:
#   --prefix NAME               Stack prefix (or STACK_PREFIX env)
#   --region REGION             AWS region (default: $AWS_REGION, else the region deploy.sh
#                               remembered for --prefix, else us-east-1)
#   --profile NAME              AWS CLI profile (default: $AWS_PROFILE, else the profile
#                               deploy.sh remembered for --prefix, else "default")
#   --user-pool-id ID           Cognito user pool (default: read from the Prebid stack)
#   --control-advertisers LIST  Comma-separated advertiser/seller domains the deploy's
#                               client may act for. Default '*' (every record).
#   --tag TAG                   Image tag (default: src-<hash of control_plane sources>)
#   --image URI                 Use this image instead of building (implies --skip-build)
#   --skip-build                Reuse the image already in ECR under --tag
#   --start-at N                Resume at step N (1-5)
#   --yes                       Accept the cost note without prompting (standalone only)
#   --verbose                   Stream narration to the terminal under deploy.sh
#   --destroy                   Remove the runtime, the stack and the ECR repository
#   --connection-info           Deploy nothing. Read what an agent needs to connect from
#                               AWS (not from a local file) and print it: the MCP endpoint
#                               URL, runtime ARN, token endpoint, client id, scopes, issuer
#                               and discovery URL, plus an MCP client config snippet.
#     --json                    ...as one JSON object instead
#     --token                   ...and mint a fresh access token (about 60 min) for the
#                               config snippet. Reads the client secret from Cognito to do
#                               so; prints the token, never the secret.
#
# Exit codes: 0 done; 1 failed; 2 prerequisites missing (deploy_prebid.sh warns and
# continues on 2, fails on 1).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AWS_REGION="${AWS_REGION:-}"
STACK_PREFIX="${STACK_PREFIX:-}"
BASE_NAME="${BASE_NAME:-nvidia-artf-recommenders}"
USER_POOL_ID="${USER_POOL_ID:-}"
CONTROL_ADVERTISERS="${CONTROL_ADVERTISERS:-*}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_URI=""
START_AT=1
SKIP_BUILD=0
ASSUME_YES=0
DESTROY=0
CONNECTION_INFO=0
INFO_JSON=0
INFO_TOKEN=0
VERBOSE="${VERBOSE:-0}"
DEPLOY_PROFILE="${AWS_PROFILE:-}"
#: Set by deploy_prebid.sh (which inherits it from deploy.sh). Narration goes there.
CONTROL_PLANE_DETAIL_LOG="${CONTROL_PLANE_DETAIL_LOG:-${PREBID_DETAIL_LOG:-}}"

_PREV_ARG=""
for arg in "$@"; do
  case "${arg}" in
    --prefix=*)              STACK_PREFIX="${arg#--prefix=}" ;;
    --region=*)              AWS_REGION="${arg#--region=}" ;;
    --profile=*)             DEPLOY_PROFILE="${arg#--profile=}" ;;
    --user-pool-id=*)        USER_POOL_ID="${arg#--user-pool-id=}" ;;
    --control-advertisers=*) CONTROL_ADVERTISERS="${arg#--control-advertisers=}" ;;
    --tag=*)                 IMAGE_TAG="${arg#--tag=}" ;;
    --image=*)               IMAGE_URI="${arg#--image=}"; SKIP_BUILD=1 ;;
    --start-at=*)            START_AT="${arg#--start-at=}" ;;
    --skip-build)            SKIP_BUILD=1 ;;
    --yes)                   ASSUME_YES=1 ;;
    --verbose)               VERBOSE=1 ;;
    --destroy)               DESTROY=1 ;;
    --connection-info)       CONNECTION_INFO=1 ;;
    --json)                  INFO_JSON=1 ;;
    --token)                 INFO_TOKEN=1 ;;
    --prefix|--region|--profile|--user-pool-id|--control-advertisers|--tag|--image|--start-at) ;;
    # deploy.sh's flags, given here by mistake. Named, because "unknown argument" does
    # not say that this script only adds the control plane to a stack that has Prebid.
    --with-prebid|--no-prebid|--learner-mode|--no-learner-mode|--with-control-plane|--no-control-plane)
      printf "'%s' is a deploy.sh flag. This script installs only the control plane, onto a stack\n" "${arg}" >&2
      printf "that already has Prebid. Either run:\n  ./deploy_control_plane.sh --prefix <prefix> [--profile NAME]\n" >&2
      printf "or deploy (or update) everything, the control plane included:\n  ./deploy.sh --prefix <prefix> %s\n" "${arg}" >&2
      exit 1 ;;
    -h|--help) awk '/^# ={10,}/{n++; next} n==1' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)
      case "${_PREV_ARG}" in
        --prefix)              STACK_PREFIX="${arg}" ;;
        --region)              AWS_REGION="${arg}" ;;
        --profile)             DEPLOY_PROFILE="${arg}" ;;
        --user-pool-id)        USER_POOL_ID="${arg}" ;;
        --control-advertisers) CONTROL_ADVERTISERS="${arg}" ;;
        --tag)                 IMAGE_TAG="${arg}" ;;
        --image)               IMAGE_URI="${arg}"; SKIP_BUILD=1 ;;
        --start-at)            START_AT="${arg}" ;;
        *) printf "unknown argument '%s'. See --help.\n" "${arg}" >&2; exit 1 ;;
      esac ;;
  esac
  _PREV_ARG="${arg}"
done
# Profile and region: an explicit flag, then the environment, then what deploy.sh
# remembered for this prefix in .deploy-state.json, then the defaults. The same
# order deploy.sh uses, so a prefix deployed with --profile X is not silently
# looked for in another account.
_remembered() {  # _remembered <key>
  [[ -n "${STACK_PREFIX}" && -f "${SCRIPT_DIR}/.deploy-state.json" ]] || return 0
  python3 -c 'import json, sys
try:
    d = json.load(open(sys.argv[1]))["deployments"][sys.argv[2]]["remembered"]
    v = d.get(sys.argv[3])
    print(v if v else "")
except Exception:
    pass' "${SCRIPT_DIR}/.deploy-state.json" "${STACK_PREFIX}" "$1" 2>/dev/null || true
}
_PROFILE_FROM=""; _REGION_FROM=""
_REMEMBERED_PROFILE="$(_remembered profile)"
if [[ -n "${DEPLOY_PROFILE}" && -n "${_REMEMBERED_PROFILE}" && "${DEPLOY_PROFILE}" != "${_REMEMBERED_PROFILE}" ]]; then
  printf '\033[0;33m[control-plane][warn]\033[0m profile %s (from --profile or AWS_PROFILE) differs from %s, the profile deploy.sh remembered for %s. If the stack is in the other account, preflight will not find it; pass --profile %s.\n' \
    "${DEPLOY_PROFILE}" "${_REMEMBERED_PROFILE}" "${STACK_PREFIX}" "${_REMEMBERED_PROFILE}" >&2
fi
if [[ -z "${DEPLOY_PROFILE}" ]]; then
  DEPLOY_PROFILE="$(_remembered profile)"; [[ -n "${DEPLOY_PROFILE}" ]] && _PROFILE_FROM=" (remembered for ${STACK_PREFIX})"
fi
if [[ -z "${AWS_REGION}" ]]; then
  AWS_REGION="$(_remembered region)"; [[ -n "${AWS_REGION}" ]] && _REGION_FROM=" (remembered for ${STACK_PREFIX})"
fi
AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION
export AWS_PROFILE="${DEPLOY_PROFILE:-default}"

if ! [[ "${START_AT}" =~ ^[1-5]$ ]]; then echo "--start-at must be 1 to 5" >&2; exit 1; fi

# ------------------------------------------------------------- output helpers
# Same quiet contract as deploy_prebid.sh: step lines always reach the terminal;
# narration goes to the detail log when one is set and --verbose is not.
_quiet() { [[ -n "${CONTROL_PLANE_DETAIL_LOG}" && "${VERBOSE}" -eq 0 ]]; }
log() {
  if _quiet; then printf '[control-plane] %s\n' "$*" >> "${CONTROL_PLANE_DETAIL_LOG}" 2>/dev/null || true
  else printf '\033[0;32m[control-plane]\033[0m %s\n' "$*"; fi
}
step() {
  printf '  \033[0;90m->\033[0m %s\n' "$*"
  if _quiet; then printf '[control-plane] %s\n' "$*" >> "${CONTROL_PLANE_DETAIL_LOG}" 2>/dev/null || true; fi
}
warn() { printf '\033[0;33m[control-plane][warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m[control-plane][fail]\033[0m %s\n' "$*" >&2; exit 1; }
_run() { if _quiet; then "$@" >> "${CONTROL_PLANE_DETAIL_LOG}" 2>&1; else "$@"; fi; }

# --------------------------------------------------------------------- names
PFX="${STACK_PREFIX:+${STACK_PREFIX}-}"
STACK_NAME="${PFX}${BASE_NAME}"
PREBID_STACK="${PFX}prebid-artf"
CP_STACK="${PFX}artf-control-plane"
DEALS_TABLE="${STACK_NAME}-deals"
ECR_REPO="${STACK_NAME}-control-plane"
# AgentCore runtime names allow [a-zA-Z0-9_] only and must start with a letter.
RUNTIME_NAME="$(printf '%s' "${STACK_NAME}_control" | tr '-' '_')"
[[ "${RUNTIME_NAME}" =~ ^[a-zA-Z] ]] || RUNTIME_NAME="cp_${RUNTIME_NAME}"
SCOPE_PREFIX="artf-control"
RECORD_FILE="${REPO_ROOT}/.control-plane-deployment.json"
PYTHON="${PYTHON:-python3}"
# The interpreter the printed verify commands name. The deploy venv has mcp and
# certifi; a bare python3 may have neither.
if [[ -x "${SCRIPT_DIR}/.deploy-venv/bin/python" ]]; then
  CONNECT_PYTHON="${SCRIPT_DIR}/.deploy-venv/bin/python"
else
  CONNECT_PYTHON="${PYTHON}"
fi

stack_output() {
  aws cloudformation describe-stacks --stack-name "$1" --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text 2>/dev/null || echo ''
}
runtime_field() {  # runtime_field <name> <field>
  aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
    --query "agentRuntimes[?agentRuntimeName=='$1'].$2 | [0]" --output text 2>/dev/null \
    | sed 's/^None$//'
}
# The AgentCore streamable-HTTP URL for a runtime: the ARN percent-encoded into the
# PATH (colons and the slash). Same construction as scripts/control_plane_connect.py.
mcp_url_for() {  # mcp_url_for <runtime-arn> <region>
  printf 'https://bedrock-agentcore.%s.amazonaws.com/runtimes/%s/invocations?qualifier=DEFAULT' \
    "$2" "$(printf '%s' "$1" | sed -e 's/:/%3A/g' -e 's#/#%2F#g')"
}

# Content hash of everything that goes into the image, so an unchanged source is
# not rebuilt and a changed one is never mistaken for the previous image.
source_hash() {
  ( cd "${REPO_ROOT}/source" && \
    find control_plane -type f -name '*.py' -print0 | sort -z | xargs -0 cat; \
    cat Dockerfile.control-plane requirements-control-plane.txt ) | shasum -a 256 | cut -c1-16
}

# ============================================================ CONNECTION INFO
# Everything read live from AWS, so it works from any machine with credentials for
# the account, not only the one that ran the deploy. Deploys nothing.
if [[ "${CONNECTION_INFO}" -eq 1 ]]; then
  _v() { local x; x="$(stack_output "$1" "$2")"; [[ "${x}" == "None" ]] && x=""; printf '%s' "${x}"; }
  CI_RUNTIME_ARN="$(runtime_field "${RUNTIME_NAME}" agentRuntimeArn)"
  CI_CLIENT_ID="$(_v "${CP_STACK}" ControlClientId)"
  CI_SCOPES="$(_v "${CP_STACK}" ControlScopes)"
  CI_ISSUER="$(_v "${CP_STACK}" TokenIssuer)"
  CI_DISCOVERY="$(_v "${CP_STACK}" DiscoveryUrl)"
  CI_TOKEN_ENDPOINT="$(_v "${PREBID_STACK}" TokenEndpoint)"
  CI_POOL="${USER_POOL_ID:-${CI_ISSUER##*/}}"
  if [[ -z "${CI_RUNTIME_ARN}" || -z "${CI_CLIENT_ID}" ]]; then
    printf 'The control plane is not deployed for prefix %s in %s (no runtime %s or no stack %s).\n' \
      "${STACK_PREFIX:-<none>}" "${AWS_REGION}" "${RUNTIME_NAME}" "${CP_STACK}" >&2
    printf 'Deploy it: ./deploy.sh --prefix %s --with-prebid   (or ./deploy_control_plane.sh --prefix %s)\n' \
      "${STACK_PREFIX}" "${STACK_PREFIX}" >&2
    exit 1
  fi
  CI_MCP_URL="$(mcp_url_for "${CI_RUNTIME_ARN}" "${AWS_REGION}")"
  CI_TOKEN=""; CI_TOKEN_EXPIRES=""
  if [[ "${INFO_TOKEN}" -eq 1 ]]; then
    # The secret is read into this process only, handed to python on stdin, and never
    # printed; only the resulting short-lived access token is shown.
    _secret="$(aws cognito-idp describe-user-pool-client --user-pool-id "${CI_POOL}" --client-id "${CI_CLIENT_ID}" \
      --region "${AWS_REGION}" --query 'UserPoolClient.ClientSecret' --output text 2>/dev/null || true)"
    [[ -n "${_secret}" && "${_secret}" != "None" ]] || { echo "could not read the client secret (needs cognito-idp:DescribeUserPoolClient)" >&2; exit 1; }
    _tok="$(printf '%s' "${_secret}" | "${PYTHON}" -c '
import base64, json, sys, urllib.parse, urllib.request
endpoint, client_id, scopes, scripts = sys.argv[1:5]
sys.path.insert(0, scripts)
from control_plane_connect import tls_context
secret = sys.stdin.read()
basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
req = urllib.request.Request(endpoint, method="POST",
    data=urllib.parse.urlencode({"grant_type": "client_credentials", "scope": scopes}).encode(),
    headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})
with urllib.request.urlopen(req, timeout=15, context=tls_context()) as r:
    body = json.loads(r.read())
print(body["access_token"], body.get("expires_in", ""))
' "${CI_TOKEN_ENDPOINT}" "${CI_CLIENT_ID}" "${CI_SCOPES}" "${SCRIPT_DIR}/scripts")" || { echo "token request to ${CI_TOKEN_ENDPOINT} failed" >&2; exit 1; }
    unset _secret
    CI_TOKEN="${_tok%% *}"; CI_TOKEN_EXPIRES="${_tok##* }"
  fi
  "${PYTHON}" - "${INFO_JSON}" "${CI_MCP_URL}" "${CI_RUNTIME_ARN}" "${AWS_REGION}" "${CI_TOKEN_ENDPOINT}" \
      "${CI_CLIENT_ID}" "${CI_SCOPES}" "${CI_ISSUER}" "${CI_DISCOVERY}" "${CI_POOL}" "${CI_TOKEN}" "${CI_TOKEN_EXPIRES}" "${CONNECT_PYTHON}" "${SCRIPT_DIR}" <<'PY'
import json, sys
(as_json, url, arn, region, token_ep, client_id, scopes, issuer, discovery, pool, token, expires,
 connect_python, script_dir) = sys.argv[1:15]
secret_cmd = (f"aws cognito-idp describe-user-pool-client --user-pool-id {pool} --client-id {client_id} "
              f"--region {region} --query UserPoolClient.ClientSecret --output text")
client_cfg = {"mcpServers": {"artf-control-plane": {
    "url": url,
    "headers": {"Authorization": f"Bearer {token or '<access token>'}"},
}}}
info = {
    "mcp_url": url, "transport": "streamable-http", "runtime_arn": arn, "region": region,
    "oauth": {"grant_type": "client_credentials", "token_endpoint": token_ep, "client_id": client_id,
              "client_auth": "HTTP Basic (client_id:client_secret)", "scopes": scopes.split(),
              "issuer": issuer, "discovery_url": discovery, "client_secret_command": secret_cmd},
    "mcp_client_config": client_cfg,
}
if token:
    info["access_token"] = token
    info["access_token_expires_in"] = int(expires) if expires.isdigit() else expires
if as_json == "1":
    print(json.dumps(info, indent=2))
    sys.exit(0)
print(f"""
Campaign/deals MCP control plane: connection details ({region})

  MCP endpoint (streamable HTTP)
    {url}
  Runtime ARN      {arn}

  OAuth 2.0 client_credentials
    Token endpoint {token_ep}
    Client id      {client_id}
    Client auth    HTTP Basic with client_id:client_secret
    Scopes         {scopes}
    Issuer         {issuer}
    Discovery URL  {discovery}
    Client secret  not printed; read it with:
                   {secret_cmd}

  Every MCP request carries:  Authorization: Bearer <access token>
  Tokens last about 60 minutes; fetch a new one from the token endpoint when it expires.
""")
if token:
    print(f"  Access token (expires in {expires}s):\n    {token}\n")
print("  MCP client config (clients that take a URL and headers):")
print("    " + json.dumps(client_cfg, indent=2).replace("\n", "\n    "))
if not token:
    print("\n  Add --token to fill in a freshly minted access token.")
print("\n  Verify the whole path (token, authorizer, container, tools):")
print(f"    {connect_python} {script_dir}/scripts/control_plane_connect.py --runtime-arn {arn} \\")
print(f"      --token-endpoint {token_ep} --client-id {client_id} --client-secret \"$({secret_cmd})\" \\")
print("      --call-tool list_campaigns")
PY
  exit 0
fi

# ===================================================================== DESTROY
if [[ "${DESTROY}" -eq 1 ]]; then
  log "Removing the campaign/deals control plane (prefix=${STACK_PREFIX:-<none>})"
  # 1. The runtime, by exact name, resolved to an id first. Deleted BEFORE the stack
  #    because the stack owns the role the runtime runs as.
  RID="$(runtime_field "${RUNTIME_NAME}" agentRuntimeId)"
  if [[ -n "${RID}" ]]; then
    _run "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" --action destroy \
      --runtime-name "${RUNTIME_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}" \
      && log "  deleted AgentCore runtime ${RUNTIME_NAME} (${RID})" \
      || warn "could not delete AgentCore runtime ${RID}; delete it by hand"
  else
    log "  no AgentCore runtime named ${RUNTIME_NAME}"
  fi
  # 2. The stack: resource server, client, role.
  if aws cloudformation describe-stacks --stack-name "${CP_STACK}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    aws cloudformation delete-stack --stack-name "${CP_STACK}" --region "${AWS_REGION}"
    aws cloudformation wait stack-delete-complete --stack-name "${CP_STACK}" --region "${AWS_REGION}" \
      && log "  deleted stack ${CP_STACK}" \
      || warn "stack ${CP_STACK} did not finish deleting; check its events"
  else
    log "  no stack ${CP_STACK}"
  fi
  # 3. The image repository.
  aws ecr delete-repository --repository-name "${ECR_REPO}" --force --region "${AWS_REGION}" >/dev/null 2>&1 \
    && log "  deleted ECR repository ${ECR_REPO}" || log "  no ECR repository ${ECR_REPO}"
  rm -f "${RECORD_FILE}"
  log "Control plane removed. The deals table and its records are untouched (deploy.sh --destroy removes them)."
  exit 0
fi

# =================================================================== STEP 1
step "Step 1/5: Preflight (control plane)"
MISSING=()
command -v aws >/dev/null 2>&1 || MISSING+=("aws CLI")
command -v "${PYTHON}" >/dev/null 2>&1 || MISSING+=("python3")
if ! command -v docker >/dev/null 2>&1; then
  MISSING+=("docker (the control-plane image is built locally, arm64; pass --skip-build or --image to reuse one)")
elif [[ "${SKIP_BUILD}" -eq 0 ]] && ! docker buildx version >/dev/null 2>&1; then
  MISSING+=("docker buildx (needed to build the arm64 image on this machine)")
fi
ACCOUNT_ID=""
if ! ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"; then
  MISSING+=("AWS credentials for profile '${AWS_PROFILE}' in ${AWS_REGION}")
fi
if [[ -n "${ACCOUNT_ID}" ]]; then
  aws dynamodb describe-table --table-name "${DEALS_TABLE}" --region "${AWS_REGION}" >/dev/null 2>&1 \
    || MISSING+=("deals table ${DEALS_TABLE} (deploy.sh Phase 1 creates it)")
  TOKEN_ENDPOINT="$(stack_output "${PREBID_STACK}" TokenEndpoint)"
  [[ "${TOKEN_ENDPOINT}" == "None" ]] && TOKEN_ENDPOINT=""
  [[ -n "${TOKEN_ENDPOINT}" ]] || MISSING+=("Prebid stack ${PREBID_STACK} with a TokenEndpoint output (the hosted domain the client_credentials grant needs; deploy_prebid.sh Step 6 creates it)")
  if [[ -z "${USER_POOL_ID}" ]]; then
    USER_POOL_ID="$(aws cloudformation describe-stacks --stack-name "${PREBID_STACK}" --region "${AWS_REGION}" \
      --query "Stacks[0].Parameters[?ParameterKey=='UserPoolId'].ParameterValue" --output text 2>/dev/null || echo '')"
    [[ "${USER_POOL_ID}" == "None" ]] && USER_POOL_ID=""
  fi
  [[ -n "${USER_POOL_ID}" ]] || MISSING+=("Cognito user pool id (pass --user-pool-id; could not read it from ${PREBID_STACK})")
fi
if [[ ${#MISSING[@]} -gt 0 ]]; then
  for m in "${MISSING[@]}"; do warn "missing: ${m}"; done
  warn "The control plane was not deployed. Prebid Server itself is unaffected."
  exit 2
fi
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
SRC_HASH="$(source_hash)"
IMAGE_TAG="${IMAGE_TAG:-src-${SRC_HASH}}"
IMAGE_URI="${IMAGE_URI:-${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}}"
log "  account ${ACCOUNT_ID}, profile ${AWS_PROFILE}${_PROFILE_FROM}, region ${AWS_REGION}${_REGION_FROM}, pool ${USER_POOL_ID}"
log "  runtime ${RUNTIME_NAME}, stack ${CP_STACK}, image ${IMAGE_URI}"

# Standalone, the cost note is shown and confirmed here. Under deploy.sh the Prebid
# Step 2 disclosure already carried this term and the decision was --with-prebid.
if [[ "${ASSUME_YES}" -ne 1 && -z "${CONTROL_PLANE_DETAIL_LOG}" ]]; then
  echo
  echo "  Cost: one AgentCore runtime, consumption-billed per second only while a tool call's"
  echo "  session runs (a few cents per month at demonstration volume, nothing when idle), plus"
  echo "  one small ECR repository. The Cognito resource server, app client and IAM role have"
  echo "  no standing charge."
  echo
  if [[ ! -t 0 ]]; then
    fail "Confirmation needs an interactive terminal (stdin is not a tty). Re-run with --yes."
  fi
  printf '\033[0;33m[control-plane]\033[0m Proceed? [y/N]: '
  read -r REPLY || REPLY="n"
  case "${REPLY}" in y|Y|yes|YES) ;; *) log "Declined. Nothing has been provisioned."; exit 0 ;; esac
fi

# =================================================================== STEP 2
if [[ "${START_AT}" -le 2 ]]; then
  if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    step "Step 2/5: --skip-build; using ${IMAGE_URI}"
  else
    step "Step 2/5: Build the control-plane image (arm64)"
    aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
      || _run aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
           --image-scanning-configuration scanOnPush=true
    if aws ecr describe-images --repository-name "${ECR_REPO}" --image-ids imageTag="${IMAGE_TAG}" \
         --region "${AWS_REGION}" >/dev/null 2>&1; then
      log "  ${ECR_REPO}:${IMAGE_TAG} already in ECR (source unchanged); not rebuilding"
    else
      _run bash -c "aws ecr get-login-password --region '${AWS_REGION}' | docker login --username AWS --password-stdin '${REGISTRY}'"
      _run docker buildx build --platform linux/arm64 \
        -f "${REPO_ROOT}/source/Dockerfile.control-plane" -t "${IMAGE_URI}" --load "${REPO_ROOT}/source" \
        || fail "docker buildx build failed for the control-plane image"
      _run docker push "${IMAGE_URI}" || fail "docker push ${IMAGE_URI} failed"
      log "  pushed ${IMAGE_URI}"
    fi
  fi
fi
aws ecr describe-images --repository-name "${ECR_REPO}" --image-ids imageTag="${IMAGE_TAG}" \
  --region "${AWS_REGION}" >/dev/null 2>&1 \
  || fail "the control-plane image ${IMAGE_URI} is not in ECR. Run without --skip-build, or pass --image."

# =================================================================== STEP 3
if [[ "${START_AT}" -le 3 ]]; then
  step "Step 3/5: Deploy ${CP_STACK} (resource server, client, execution role)"
  existing="$(aws cloudformation describe-stacks --stack-name "${CP_STACK}" --region "${AWS_REGION}" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"
  [[ "${existing}" == "None" ]] && existing=""
  action="create-stack"
  case "${existing}" in
    "") action="create-stack" ;;
    ROLLBACK_COMPLETE|*FAILED)
      # A failed first create leaves a stack that can only be deleted.
      log "  deleting ${existing} remnant before recreating"
      aws cloudformation delete-stack --stack-name "${CP_STACK}" --region "${AWS_REGION}"
      aws cloudformation wait stack-delete-complete --stack-name "${CP_STACK}" --region "${AWS_REGION}"
      action="create-stack" ;;
    *COMPLETE) action="update-stack" ;;
    *IN_PROGRESS) fail "Stack ${CP_STACK} is ${existing}. Wait for it to settle." ;;
    *) fail "Stack ${CP_STACK} is in an unrecognised status (${existing}). Refusing to act on it." ;;
  esac
  out="/tmp/${CP_STACK}-cfn-output.txt"
  if aws cloudformation "${action}" --stack-name "${CP_STACK}" \
       --template-body "file://${SCRIPT_DIR}/control_plane_cfn.yaml" \
       --capabilities CAPABILITY_NAMED_IAM --region "${AWS_REGION}" \
       --parameters \
         "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
         "ParameterKey=UserPoolId,ParameterValue=${USER_POOL_ID}" \
         "ParameterKey=DealsTableName,ParameterValue=${DEALS_TABLE}" \
         "ParameterKey=EcrRepositoryName,ParameterValue=${ECR_REPO}" \
         "ParameterKey=RuntimeName,ParameterValue=${RUNTIME_NAME}" \
         "ParameterKey=ScopePrefix,ParameterValue=${SCOPE_PREFIX}" \
       >"${out}" 2>&1 && grep -q StackId "${out}"; then
    wait_for="stack-create-complete"; [[ "${action}" == "update-stack" ]] && wait_for="stack-update-complete"
    aws cloudformation wait "${wait_for}" --stack-name "${CP_STACK}" --region "${AWS_REGION}" \
      || fail "Stack ${CP_STACK} did not reach ${wait_for}. See: aws cloudformation describe-stack-events --stack-name ${CP_STACK}"
    log "  stack ${CP_STACK}: COMPLETE"
  elif grep -q "No updates are to be performed" "${out}"; then
    log "  stack ${CP_STACK}: already current"
  else
    warn "CFN error: $(cat "${out}")"
    fail "Stack ${CP_STACK} deploy failed"
  fi
fi
ROLE_ARN="$(stack_output "${CP_STACK}" ControlPlaneRoleArn)"
CLIENT_ID="$(stack_output "${CP_STACK}" ControlClientId)"
TOKEN_ISSUER="$(stack_output "${CP_STACK}" TokenIssuer)"
DISCOVERY_URL="$(stack_output "${CP_STACK}" DiscoveryUrl)"
CONTROL_SCOPES="$(stack_output "${CP_STACK}" ControlScopes)"
TOKEN_ENDPOINT="${TOKEN_ENDPOINT:-$(stack_output "${PREBID_STACK}" TokenEndpoint)}"
for v in ROLE_ARN CLIENT_ID TOKEN_ISSUER DISCOVERY_URL CONTROL_SCOPES; do
  [[ -n "${!v}" && "${!v}" != "None" ]] || fail "${CP_STACK} has no ${v} output; the stack is not the one this script expects"
done

# =================================================================== STEP 4
if [[ "${START_AT}" -le 4 ]]; then
  step "Step 4/5: Register AgentCore runtime ${RUNTIME_NAME}"
  # Which principals the deploy's own client may act for. A client_credentials token
  # carries no user and no custom:advertisers claim, so the grant lives on the
  # runtime where get-agent-runtime shows it and a redeploy changes it.
  ENTITLEMENTS="${CLIENT_ID}=${CONTROL_ADVERTISERS}"
  ORIGIN="$("${PYTHON}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from control_plane import records; print(records.render_origin(sys.argv[2]))' "${REPO_ROOT}/source" "${STACK_PREFIX}")"
  # allowedScopes: the four control scopes, as a JSON list.
  SCOPES_JSON="$("${PYTHON}" -c 'import json, sys; print(json.dumps(sys.argv[1].split()))' "${CONTROL_SCOPES}")"
  AUTHORIZER="{\"customJWTAuthorizer\":{\"discoveryUrl\":\"${DISCOVERY_URL}\",\"allowedClients\":[\"${CLIENT_ID}\"],\"allowedScopes\":${SCOPES_JSON}}}"
  log "  inbound auth: OAuth client_credentials (customJWTAuthorizer, client ${CLIENT_ID})"
  log "  entitlement : ${ENTITLEMENTS}"
  # IAM propagation: the role was created seconds ago and CreateAgentRuntime validates
  # it can be assumed. Retry a few times before calling it a failure.
  attempt=0
  until _run "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" --action deploy \
      --runtime-name "${RUNTIME_NAME}" --role-arn "${ROLE_ARN}" --container-uri "${IMAGE_URI}" \
      --region "${AWS_REGION}" --profile "${AWS_PROFILE}" --protocol MCP \
      --description "ARTF campaign/deals control plane (MCP, OAuth client_credentials)" \
      --authorizer-json "${AUTHORIZER}" --request-header-allowlist Authorization \
      --environment "ARTF_CONTROL_TABLE=${DEALS_TABLE}" \
      --environment "ARTF_CONTROL_TOKEN_ISSUER=${TOKEN_ISSUER}" \
      --environment "ARTF_CONTROL_DEAL_ORIGIN=${ORIGIN}" \
      --environment "ARTF_CONTROL_SCOPE_PREFIX=${SCOPE_PREFIX}" \
      --environment "ARTF_CONTROL_CLIENT_ENTITLEMENTS=${ENTITLEMENTS}" \
      --environment "AWS_REGION=${AWS_REGION}"; do
    attempt=$((attempt + 1))
    [[ ${attempt} -lt 4 ]] || fail "AgentCore runtime registration failed after ${attempt} attempts. Diagnose: aws bedrock-agentcore-control list-agent-runtimes --region ${AWS_REGION}"
    log "  registration refused (attempt ${attempt}); waiting 15s for IAM to propagate"
    sleep 15
  done
fi
RUNTIME_ARN="$(runtime_field "${RUNTIME_NAME}" agentRuntimeArn)"
[[ -n "${RUNTIME_ARN}" ]] || fail "runtime ${RUNTIME_NAME} is not listed after registration"

# =================================================================== STEP 5
step "Step 5/5: Record"
"${PYTHON}" - "${RECORD_FILE}" <<PY
import json, sys
record = {
    "stack_name": "${CP_STACK}",
    "region": "${AWS_REGION}",
    "runtime_name": "${RUNTIME_NAME}",
    "runtime_arn": "${RUNTIME_ARN}",
    "mcp_url": "$(mcp_url_for "${RUNTIME_ARN}" "${AWS_REGION}")",
    "ecr_repository": "${ECR_REPO}",
    "image": "${IMAGE_URI}",
    "deals_table": "${DEALS_TABLE}",
    "user_pool_id": "${USER_POOL_ID}",
    "token_endpoint": "${TOKEN_ENDPOINT}" or None,
    "token_issuer": "${TOKEN_ISSUER}",
    "control_client_id": "${CLIENT_ID}",
    "control_scopes": "${CONTROL_SCOPES}".split(),
    "client_entitlements": "${ENTITLEMENTS:-}" or None,
    "secret": "on the Cognito app client only; never written here",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
log "  recorded to ${RECORD_FILE}"

cat <<EOF

  Campaign/deals control plane deployed.
    Runtime      ${RUNTIME_NAME}
    Runtime ARN  ${RUNTIME_ARN}
    MCP URL      $(mcp_url_for "${RUNTIME_ARN}" "${AWS_REGION}")
    Client id    ${CLIENT_ID}   (scopes: ${CONTROL_SCOPES})
    Token URL    ${TOKEN_ENDPOINT}

  Full connection details, from any machine with access to the account:
    ${SCRIPT_DIR}/deploy_control_plane.sh${STACK_PREFIX:+ --prefix ${STACK_PREFIX}} --region ${AWS_REGION} --connection-info [--token] [--json]

  Seeded campaigns are PAUSED until activated through the control plane. Verify the
  path and activate one (the secret is read from the app client at run time):

    export CLIENT_SECRET=\$(aws cognito-idp describe-user-pool-client --user-pool-id ${USER_POOL_ID} \\
      --client-id ${CLIENT_ID} --region ${AWS_REGION} --query UserPoolClient.ClientSecret --output text)
    ${CONNECT_PYTHON} ${SCRIPT_DIR}/scripts/control_plane_connect.py --runtime-arn ${RUNTIME_ARN} \\
      --token-endpoint ${TOKEN_ENDPOINT} --client-id ${CLIENT_ID} --client-secret "\$CLIENT_SECRET" \\
      --call-tool list_campaigns
    ${CONNECT_PYTHON} ${SCRIPT_DIR}/scripts/control_plane_connect.py --runtime-arn ${RUNTIME_ARN} \\
      --token-endpoint ${TOKEN_ENDPOINT} --client-id ${CLIENT_ID} --client-secret "\$CLIENT_SECRET" \\
      --call-tool activate_campaign --arg campaign_id=camp-cedar

EOF
