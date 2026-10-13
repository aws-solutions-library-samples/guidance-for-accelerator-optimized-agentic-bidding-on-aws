#!/usr/bin/env bash

# This script requires real bash, not bash in POSIX mode. `sh deploy.sh` used to
# fail with an opaque "syntax error near unexpected token `<'" pointing at a line
# a thousand lines below anything the reader had done. That construct is gone, but
# POSIX mode disables other bash behaviour too, so fail here with something
# actionable rather than somewhere later with something puzzling.
if [ -z "${BASH_VERSION:-}" ] || { command -v shopt >/dev/null 2>&1 && shopt -qo posix; }; then
  printf 'deploy.sh must be run with bash, not sh.\n\n  bash %s %s\n\n' "$0" "$*" >&2
  exit 1
fi
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
# Default output shows phase headers, the step being entered, and one line per
# finished step marked with a check or a cross. A running step animates in place so
# a long wait reads as a wait, not a hang; set DEPLOY_NO_SPINNER=1 to hold it still
# (it is skipped automatically when output is redirected).
#
# The terminal shows phases and steps only. Everything a step runs -- the ONNX
# export, pip, S3 copies, eksctl, every kubectl apply, CodeBuild polling, the
# AgentCore SDK's INFO logging, deploy_prebid.sh and deploy_closed_loop.sh -- writes
# to ONE file per run, deployment/.deploy-<prefix>.log (the previous run is kept as
# .deploy-<prefix>.prev.log). Its path is printed at the start of the run and again
# by every failure. A FAILURE is never quiet: the last 15 lines of the log are
# printed inline and the file is named.
#
# Pass --verbose to stream it all to the terminal instead. On failure, the phase and
# step that failed are reported with the real error plus a remediation hint.
#
# A flag the script does not recognise STOPS the run before anything is touched --
# including a flag typed with an em dash (—with-prebid), which a chat client or word
# processor substitutes for two hyphens.
#
# NETWORK LAYOUT. Both EKS node groups sit in PRIVATE subnets behind one NAT gateway
# per AZ (eks/cluster-config.yaml: privateNetworking: true, nat.gateway:
# HighlyAvailable). Nothing in the deployment reaches a node by its IP: Prebid Server
# and the ARTF containers talk to the orchestrator through in-cluster Service DNS,
# the Triton load balancer is internal, and both Lambdas (vpc-proxy, ui-api-proxy)
# are attached to the VPC. NOTHING IN THE VPC IS INTERNET-FACING. The orchestrator has
# no public address; the React UI reaches it by invoking the <prefix>-ui-api-proxy
# Lambda (deployment/ui_api_proxy_cfn.yaml) with SigV4 credentials from the Cognito
# Identity Pool, and that Lambda forwards to the orchestrator's INTERNAL NLB
# (eks/orchestrator-internal-nlb.yaml). Not to orchestrator.<ns>.svc.cluster.local:
# that name lives only in CoreDNS, and a ClusterIP has no route from a Lambda ENI.
# This is what lets the stack deploy unchanged in an account running VPC Block
# Public Access in block-ingress mode: a VPC Lambda ENI is not an internet path,
# whereas an internet-facing load balancer (and a CloudFront VPC origin) is dropped
# at the internet gateway there.
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
# Usage (--prefix is REQUIRED on every run: exactly 3 characters, a letter then
# letters/digits, e.g. dv1, bt3, nv5 -- it keys the stack names, the EKS cluster,
# the AgentCore runtime names and the local .deploy-state.json record):
#   ./deploy.sh --prefix dv1                   # full deploy; resources named dv1-nvidia-artf-*
#   ./deploy.sh --prefix dv1 --profile myprof  # use a named AWS CLI profile (see below)
#   ./deploy.sh --prefix dv1 --skip-agentcore  # combine flags
#   ./deploy.sh --prefix dv1                   # FULL stack incl. Part 2 closed-loop (default)
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
#   ./deploy.sh --status                       # what IS deployed, read live from AWS; deploys nothing
#   ./deploy.sh --start-at 3                   # force a starting phase, overriding the AWS check
#   ./deploy.sh --verbose                      # print full detailed logs alongside phase output
#   ./deploy.sh --destroy                      # tear down the entire stack
#   AWS_REGION=us-west-2 ./deploy.sh           # different region
#   (every example above also needs --prefix <3-char>; omitted here for width)
#
# AWS PROFILE. Which credentials every aws/eksctl/kubectl/boto3 call uses:
#
#     ./deploy.sh --prefix dv1 --profile my-admin-profile
#
# Resolution order: --profile, then the AWS_PROFILE environment variable, then the
# profile remembered in .deploy-state.json for this prefix, then the literal
# "default". The result is exported as AWS_PROFILE and passed as an explicit
# --profile argument to every child this script branches into (remote_build.sh,
# deploy_closed_loop.sh, deploy_prebid.sh, the scripts/*.py helpers) and to
# `aws eks update-kubeconfig`, so the kubeconfig it writes pins the same profile.
#
# Once a prefix has been deployed with a profile, that profile is REMEMBERED for
# the prefix. A later run that names a different one (flag or env) STOPS before
# touching AWS and says so -- the same prefix in two accounts is the mistake this
# prevents. To switch on purpose, delete the prefix's record from
# deployment/.deploy-state.json first. A named profile that does not exist in
# ~/.aws/config or ~/.aws/credentials also stops the run before any phase.
#
# WHERE DID I GET TO? Ask AWS, not this script:
#
#     ./deploy.sh --prefix v1 --status
#
# Progress is derived from the account -- CloudFormation stack statuses, CodeBuild
# build states, EKS nodegroups AND the nodes actually registered in the cluster,
# ECR images, Kubernetes readiness, CloudFront, Cognito, AgentCore runtimes. That
# works from any machine, after a reboot, for a colleague, and while a deploy is
# running, because it only reads.
#
# Consequences worth knowing:
#   - Re-running skips whatever already exists. No flags, no prompts, no resume.
#   - If AWS is mid-operation (a stack creating, a build running), this WAITS for it
#     rather than starting a competing one. Two terminals are safe.
#   - A phase whose state cannot be verified is RUN, not skipped. Every phase is
#     idempotent, so re-running one costs time; skipping one ships a deployment that
#     claims to be complete and is not.
#   - --start-at N still forces a starting phase, for when you know something the
#     probe cannot (e.g. rebuild my container even though an image exists).
#
# deployment/.deploy-state.json holds only the flags a previous run was GIVEN -- the
# NGC secret name above all -- so you need not repeat them. It records no progress;
# that was tried, and a file describing what the script did disagreed with what the
# account actually contained. Gitignored, safe to delete, removed by --destroy.
#
# Prerequisites:
#   - AWS CLI v2 with credentials
#   - Python 3.11+ with boto3, torch, onnx, onnxscript (and sagemaker for the
#     default Part 2 closed-loop stack — deploy.sh installs it automatically)
#   - jq, eksctl, kubectl
#   - Docker with buildx. Needed by Phase 5 for the two arm64 AgentCore agent
#     images (built locally by design), and by everything if --local-build. It does
#     NOT need to be running first: if the daemon is down when it is needed, this
#     starts it and waits.
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
    # The two Yield Optimizer containers were named for their job when the
    # combined deal-yield-manager was split, so their keys already ARE their
    # display names -- listed explicitly rather than left to the identity
    # fallback below so the mapping is obvious next to the other four.
    yield-optimizer-floor)       echo "yield-optimizer-floor" ;;
    yield-optimizer-margin)      echo "yield-optimizer-margin" ;;
    # The template container was named for its job from the start, so its key
    # already IS its display name. Listed explicitly for the same reason as the
    # two above.
    artf-template)               echo "artf-template" ;;
    *)                           echo "$1" ;;
  esac
}

# =========================================================================
# "Was this explicitly given?" markers, for remembered inputs
# =========================================================================
# A previous run's values are reused for flags this run omitted (see the state
# block after ACCOUNT_ID resolution below). That requires distinguishing "omitted"
# from "given the default value", and the `"${VAR:-default}"` expressions in this
# section destroy that distinction -- after they run, an unset AWS_REGION and an
# explicit AWS_REGION=us-east-1 look identical.
#
# So the env-var-supplied cases are captured HERE, before the defaults are
# applied. The flag-supplied cases are captured in the arg loop below.
#
# `if` blocks rather than `[[ ... ]] && x=1`: under `set -e` a failing `[[ ]]` as
# the last command on a line exits the script.
_GIVEN_REGION=0
_GIVEN_MAXGPUS=0
_GIVEN_ARTF_NODE_ROLE=0
_GIVEN_MODEL_ID=0
_GIVEN_PROFILE=0
# The profile this run was GIVEN -- by --profile (arg loop below) or by the
# AWS_PROFILE environment variable. Empty means neither, and the remembered /
# "default" fallbacks apply once the state file is readable (see "AWS profile"
# below). Captured before the loop so an explicit value can never be confused
# with a fallback.
DEPLOY_PROFILE="${AWS_PROFILE:-}"
DEPLOY_PROFILE_SOURCE="AWS_PROFILE environment variable"
if [[ -n "${AWS_REGION:-}" ]];      then _GIVEN_REGION=1; fi
if [[ -n "${AWS_PROFILE:-}" ]];     then _GIVEN_PROFILE=1; fi
if [[ -n "${MAX_GPUS:-}" ]];        then _GIVEN_MAXGPUS=1; fi
if [[ -n "${ARTF_NODE_ROLE:-}" ]];  then _GIVEN_ARTF_NODE_ROLE=1; fi
if [[ -n "${BEDROCK_MODEL_ID:-}" ]];then _GIVEN_MODEL_ID=1; fi
# Flag-only markers; set in the arg loop.
_GIVEN_LOCAL_BUILD=0
_GIVEN_RETRAINING=0
_GIVEN_PREBID=0
_GIVEN_SKIP_AGENTCORE=0
_GIVEN_START_AT=0
# Recorded (redacted) in the state file for diagnostics.
_DEPLOY_ARGV=("$@")

AWS_REGION="${AWS_REGION:-us-east-1}"
# Auction Theater caption model. The `global.` inference profile is available in
# every region, so this needs no per-region variation. Haiku 4.5 supports
# INFERENCE_PROFILE only -- the bare model id is not a usable alternative.
# Override to a us./eu./au./jp. profile for in-geography routing.
CAPTION_INFERENCE_PROFILE_ID="${CAPTION_INFERENCE_PROFILE_ID:-global.anthropic.claude-haiku-4-5-20251001-v1:0}"
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
# Prebid Server as a second, sell-side ARTF host is OFF by default. Unlike
# retraining it is opt-IN: it deploys a Cognito domain, an M2M client, a
# Secrets Manager secret and an auction host pod, none of which the core
# deployment needs. Without --with-prebid nothing below changes.
WITH_PREBID=0
# Scope the orchestrator requires of a MACHINE credential on POST /v1/mutations.
# EMPTY unless --with-prebid, and empty means the orchestrator's authorization
# behaves exactly as it did before this feature existed. Resolved after arg parsing.
ARTF_MUTATIONS_REQUIRED_SCOPE=""
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
# --resume: start from the phase the last recorded run did not finish. Resolved
# against the state file after identity is known.
RESUME=0
# --status: print what AWS says is deployed, then exit. Read-only, needs no
# terminal, safe to run from any number of shells while a deploy is running.
STATUS_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --destroy)          DESTROY=1 ;;
    --skip-agentcore)   SKIP_AGENTCORE=1; _GIVEN_SKIP_AGENTCORE=1 ;;
    --skip-images)      SKIP_IMAGES=1 ;;
    --verbose)          VERBOSE=1 ;;
    --resume)           RESUME=1 ;;
    --status)           STATUS_ONLY=1 ;;
    --start-at=*)       START_AT="${arg#--start-at=}"; _GIVEN_START_AT=1 ;;
    --ui-only)          UI_ONLY=1 ;;
    --skip-cluster)     SKIP_CLUSTER=1 ;;
    --export-only)      EXPORT_ONLY=1 ;;
    --with-retraining)  WITH_RETRAINING=1; _GIVEN_RETRAINING=1 ;;   # default; kept for back-compat
    --no-retraining|--skip-retraining) WITH_RETRAINING=0; _GIVEN_RETRAINING=1 ;;
    --with-prebid)      WITH_PREBID=1; _GIVEN_PREBID=1 ;;       # opt-in; see Step 12
    --no-prebid)        WITH_PREBID=0; _GIVEN_PREBID=1 ;;       # default; explicit for symmetry
    --remote-build)     LOCAL_BUILD=0; _GIVEN_LOCAL_BUILD=1 ;;
    --local-build)      LOCAL_BUILD=1; _GIVEN_LOCAL_BUILD=1 ;;
    --ngc-secret=*)     NGC_SECRET="${arg#--ngc-secret=}" ;;
    --ngc-secret)       ;; # value comes in next arg, handled below
    --ngc-key=*)        NGC_KEY="${arg#--ngc-key=}" ;;
    --ngc-key)          ;; # value comes in next arg, handled below
    --prefix=*)         STACK_PREFIX="${arg#--prefix=}" ;;
    --prefix)           ;; # value comes in next arg, handled below
    --maxGPUs=*)        MAX_GPUS="${arg#--maxGPUs=}"; _GIVEN_MAXGPUS=2 ;;
    --maxGPUs)          ;; # value comes in next arg, handled below
    --model-id=*)       BEDROCK_MODEL_ID="${arg#--model-id=}"; _GIVEN_MODEL_ID=1 ;;
    --model-id)         ;; # value comes in next arg, handled below
    --artf-node-role=*) ARTF_NODE_ROLE="${arg#--artf-node-role=}"; _GIVEN_ARTF_NODE_ROLE=1 ;;
    --artf-node-role)   ;; # value comes in next arg, handled below
    --artf-on-gpu)      ARTF_NODE_ROLE="inference"; _GIVEN_ARTF_NODE_ROLE=1 ;;
    --profile=*)        DEPLOY_PROFILE="${arg#--profile=}"; DEPLOY_PROFILE_SOURCE="--profile"; _GIVEN_PROFILE=1 ;;
    --profile)          ;; # value comes in next arg, handled below
    --start-at)         _GIVEN_START_AT=1 ;; # value comes in next arg, handled below
    # Print the documentation block -- everything between the first and second
    # `# =====` banner. Derived, not a hardcoded line range: the previous
    # `sed -n '2,96p'` silently truncated --help mid-sentence the moment anyone
    # added a paragraph above line 96, which is exactly what happened.
    -h|--help)          awk '/^# ={10,}/{n++; next} n==1' "$0"; exit 0 ;;
    *)
      if [[ "${_PREV_ARG:-}" == "--prefix" ]]; then
        STACK_PREFIX="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--maxGPUs" ]]; then
        MAX_GPUS="${arg}"; _GIVEN_MAXGPUS=1
      elif [[ "${_PREV_ARG:-}" == "--start-at" ]]; then
        START_AT="${arg}"; _GIVEN_START_AT=1
      elif [[ "${_PREV_ARG:-}" == "--model-id" ]]; then
        BEDROCK_MODEL_ID="${arg}"; _GIVEN_MODEL_ID=1
      elif [[ "${_PREV_ARG:-}" == "--artf-node-role" ]]; then
        ARTF_NODE_ROLE="${arg}"; _GIVEN_ARTF_NODE_ROLE=1
      elif [[ "${_PREV_ARG:-}" == "--ngc-secret" ]]; then
        NGC_SECRET="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--ngc-key" ]]; then
        NGC_KEY="${arg}"
      elif [[ "${_PREV_ARG:-}" == "--profile" ]]; then
        DEPLOY_PROFILE="${arg}"; DEPLOY_PROFILE_SOURCE="--profile"; _GIVEN_PROFILE=1
      else
        # Anything else stops the run: a token that fell through every branch above
        # would otherwise deploy a different stack than the one asked for ("—with-prebid"
        # pasted with an em dash deploys with Prebid OFF). The em/en dash case gets its
        # own hint because the two glyphs are indistinguishable from "--" in most
        # terminal fonts.
        _dash_hint=""
        case "${arg}" in
          $'\xe2\x80\x94'*|$'\xe2\x80\x93'*) _dash_hint=" (that first character is an em/en dash, not two hyphens)" ;;
        esac
        printf '\033[0;31m[fail]\033[0m %s\n' "unknown argument '${arg}'${_dash_hint} -- see ./deploy.sh --help" >&2
        exit 1
      fi
      ;;
  esac
  _PREV_ARG="${arg}"
done
# A value-taking flag as the LAST argument used to be swallowed silently: the loop
# records it in _PREV_ARG, no next arg ever arrives, and the run proceeds with the
# default -- so `--profile` on its own deployed with whatever the shell had.
case "${_PREV_ARG:-}" in
  --prefix|--maxGPUs|--start-at|--model-id|--artf-node-role|--ngc-secret|--ngc-key|--profile)
    printf '\033[0;31m[fail]\033[0m %s\n' "${_PREV_ARG} needs a value (it was the last argument)" >&2
    exit 1 ;;
esac
unset _PREV_ARG

# --prefix is required, and its shape is fixed: exactly three characters, a letter
# followed by letters or digits. It is spliced into S3 bucket names (lowercase
# only), AgentCore runtime names (must start with a letter) and the local state
# record's key, and it is the ONLY thing separating two deployments in one account.
# An optional prefix meant the no-prefix case was a real, un-namespaced deployment
# that --destroy could reach by accident.
if ! [[ "${STACK_PREFIX}" =~ ^[a-z][a-z0-9]{2}$ ]]; then
  if [[ -z "${STACK_PREFIX}" ]]; then
    printf '\033[0;31m[fail]\033[0m %s\n' "--prefix is required: exactly 3 characters, a letter then letters/digits (e.g. --prefix dv1)" >&2
  else
    printf '\033[0;31m[fail]\033[0m %s\n' "--prefix must be exactly 3 characters, a letter then letters/digits (got '${STACK_PREFIX}')" >&2
  fi
  exit 1
fi

# An EXPLICIT profile takes effect immediately, before anything else runs, so the
# first process this script starts already sees it. The remembered/"default"
# fallbacks need the state file and are resolved after the libraries load (see
# "AWS profile" below); nothing between here and there talks to AWS.
if [[ -n "${DEPLOY_PROFILE}" ]]; then
  export AWS_PROFILE="${DEPLOY_PROFILE}"
fi

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
# The orchestrator's machine-scope requirement is set ONLY when the Prebid ARTF host
# is being deployed, because that host is its only machine caller. Without the flag
# this stays empty, the orchestrator registers no protected routes, and its
# authorization path is byte-identical to what it was before the Prebid feature.
if [[ "${WITH_PREBID}" -eq 1 ]]; then
  ARTF_MUTATIONS_REQUIRED_SCOPE="artf-orchestrator/mutations:write"
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

# _emit(): one place that formats a line of output. Kept when the journalling it
# used to feed was removed -- a single formatter is still worth having, and the
# five emitters below are one line each because of it.
_emit() {
  local _fmt="$1"; shift
  # shellcheck disable=SC2059  # _fmt is a literal from the call sites below
  printf "${_fmt}\n" "$@"
  return 0
}

# say(): always-visible output, regardless of --verbose. Used for the
# --destroy narration (FR-12: its UX stays exactly as before) and the final
# deployment summary (FR-6: "print ONLY what the user needs to get started").
say() { _emit '%s' "$*"; }

# log(): detailed step-by-step narration. Printed only under --verbose so the
# default output stays to the 5 phase headers + checkmarked summaries below
# (FR-10). warn()/fail() always print — a warning or a hard failure is never
# hidden regardless of verbosity.
log()  { [[ "${VERBOSE}" -eq 1 ]] && _emit '\033[0;32m[deploy]\033[0m %s' "$*"; return 0; }
warn() { _emit '\033[0;33m[warn]\033[0m %s' "$*"; }

# phase(): always-visible section header. N is 1-5 (see the phase table in
# design.md section 2 / RENAME_MAP.md's breaking-change note).
phase() { _emit '\n\033[1;36mPhase %s/5: %s\033[0m' "$1" "$2"; }

# ok(): always-visible one-line checkmarked completion summary, printed after
# the real underlying command has actually succeeded (never before — no
# fabricated "done" markers).
ok() { _emit '  \033[0;32m[OK]\033[0m %s' "$*"; }

# ---- The detail log --------------------------------------------------------
#
# One file per run: deployment/.deploy-<prefix>.log. Everything that is not a phase
# header, a step, a completion mark, a warning, a failure or the summary goes there
# -- model export, every S3 copy, image builds, kubectl apply/rollout, the Cognito
# and frontend helpers, the AgentCore SDK, the closed-loop and Prebid children. The
# terminal reads as the sequence of steps; the log reads as what each step did.
# Under --verbose nothing is redirected and the log receives only the headers.
#
# Set by _deploy_log_init() once the prefix is validated; empty until then, and the
# helpers below fall back to the terminal while it is empty so nothing is lost.
DEPLOY_LOG=""

_deploy_log_init() {
  DEPLOY_LOG="${SCRIPT_DIR}/.deploy-${STACK_PREFIX}.log"
  # Keep exactly one previous run. A log that only ever appended would carry every
  # run since the prefix was created, which makes the one you want hard to find.
  if [[ -s "${DEPLOY_LOG}" ]]; then
    mv -f "${DEPLOY_LOG}" "${DEPLOY_LOG%.log}.prev.log" 2>/dev/null || true
  fi
  if ! : > "${DEPLOY_LOG}" 2>/dev/null; then
    DEPLOY_LOG="/tmp/deploy-${STACK_PREFIX}-$$.log"
    : > "${DEPLOY_LOG}"
  fi
  {
    printf '===== deploy.sh =====\n'
    printf '===== %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'argv: %s\n' "${_DEPLOY_ARGV[*]+"${_DEPLOY_ARGV[*]}"}"
  } >> "${DEPLOY_LOG}" 2>/dev/null || true
  return 0
}

# _log_section <label>: a dated header in the detail log, so a reader can find the
# step that produced the lines after it.
_log_section() {
  [[ -n "${DEPLOY_LOG}" ]] || return 0
  {
    printf '\n===== %s =====\n' "$1"
    printf '===== %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >> "${DEPLOY_LOG}" 2>/dev/null || true
  return 0
}

# quiet <label> <command...>: run a command with its output in the detail log.
#
# For the short, chatty commands whose only interesting output is a failure: an S3
# copy, a kubectl apply, an IAM policy version. Under --verbose, or before the log
# exists, the command runs unchanged. On failure the last lines of the log are
# printed, because the failing command's own message is still the real error and
# this is where it went. The return code is the command's, so `|| warn` and
# `|| true` at the call sites keep their meaning.
quiet() {
  local label="$1"; shift
  if [[ "${VERBOSE}" -eq 1 || -z "${DEPLOY_LOG}" ]]; then
    "$@"
    return $?
  fi
  _log_section "${label}"
  local rc=0
  local _ee=0; case "$-" in *e*) _ee=1 ;; esac
  set +e
  "$@" >> "${DEPLOY_LOG}" 2>&1
  rc=$?
  if [[ "${_ee}" -eq 1 ]]; then set -e; fi
  if [[ "${rc}" -ne 0 ]]; then
    printf '\033[0;31m[fail]\033[0m %s (exit %s) -- last 15 lines of %s:\n' "${label}" "${rc}" "${DEPLOY_LOG}" >&2
    tail -15 "${DEPLOY_LOG}" 2>/dev/null | sed 's/^/      /' >&2
  fi
  return "${rc}"
}

# run_step <label> <command...>: quiet(), plus a visible wait.
#
# For the commands a human waits on -- model export, a CodeBuild run, a rollout, the
# frontend build. The label is shown while the command runs (a spinner on a
# terminal, one static line otherwise) and marked done when it finishes; the output
# goes to the detail log. Under --verbose the label is printed once and the command
# streams. Wraps lib/deploy_progress.sh's run_logged, which is sourced later than
# this definition but earlier than any call.
run_step() {
  local label="$1"; shift
  if [[ "${VERBOSE}" -eq 1 || -z "${DEPLOY_LOG}" ]]; then
    step "${label}"
    "$@"
    return $?
  fi
  run_logged "${DEPLOY_LOG}" "${label}" "$@"
}

# _ecr_docker_login: a pipeline, so it is a function (quiet/run_step take a command,
# not a shell string). Used by Phase 1 (--local-build daemon check) and Phase 2.
_ecr_docker_login() {
  aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${REGISTRY}"
}

# ---- UI API proxy Lambda (the orchestrator's only way in) ---------------------
#
# The orchestrator has no public address; the React UI reaches it by invoking the
# <prefix>-ui-api-proxy Lambda (deployment/ui_api_proxy_cfn.yaml), which runs on the
# cluster's private subnets with the cluster security group and forwards each call
# to the orchestrator's internal NLB (eks/orchestrator-internal-nlb.yaml). A VPC
# Lambda ENI is not an internet path and an internal NLB has no public address, so
# this works in an account running VPC Block Public Access in block-ingress mode,
# where an internet-facing load balancer would be dropped at the internet gateway.
# Earlier releases gated on that mode and stopped; nothing here reads it any more,
# because nothing here needs public ingress.
#
# Why an NLB and not the ClusterIP Service: the Lambda resolves names through the
# VPC's Route 53 Resolver, which has no cluster.local zone, and a ClusterIP is a
# kube-proxy rewrite rule on the nodes with no route from an ENI. The first live
# use of the cluster.local URL failed every call in 2 ms with
# "[Errno 16] Device or resource busy" (a failed getaddrinfo), returned as
# 502 orchestrator_unreachable.
#
# The stack is (re)deployed on every run that applies the manifests: it is cheap,
# idempotent (update-stack / "No updates"), and the only way the Lambda's subnets
# and target URL follow a recreated cluster. The function ARN is then granted to
# the Identity Pool authenticated role (deploy_cognito.py grant-ui-api-invoke) and
# baked into the UI build as VITE_UI_API_PROXY_ARN.

UI_API_PROXY_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}ui-api-proxy"
ORCHESTRATOR_INTERNAL_SVC="orchestrator-internal"
ORCHESTRATOR_INTERNAL_URL=""

# resolve_orchestrator_internal_url <phase>: wait for the in-tree cloud controller
# to provision the internal NLB behind the orchestrator-internal Service and
# export ORCHESTRATOR_INTERNAL_URL=http://<hostname>. Provisioning normally takes
# one to three minutes after the Service is applied; five is the ceiling before this
# is a failure, not a wait.
resolve_orchestrator_internal_url() {
  local _phase="$1" deadline host
  deadline=$(( $(date +%s) + 300 ))  # nosemgrep: unquoted-command-substitution-in-command arithmetic context, not subject to word splitting
  while :; do
    host="$(kubectl get svc "${ORCHESTRATOR_INTERNAL_SVC}" \
      -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo '')"
    if [[ -n "${host}" ]]; then
      ORCHESTRATOR_INTERNAL_URL="http://${host}"
      ok "Orchestrator internal NLB: ${host}"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      fail "${_phase}" "Service ${ORCHESTRATOR_INTERNAL_SVC} has no load balancer hostname after 5 minutes. The UI API proxy has nothing to forward to. Check: kubectl describe svc ${ORCHESTRATOR_INTERNAL_SVC} (Events show why the NLB was not created; the node role needs elasticloadbalancing:* and the private subnets the kubernetes.io/role/internal-elb=1 tag)."
    fi
    sleep 10
  done
}

# _ui_api_proxy_probe <function-arn>: invoke the proxy once with GET /api/health/ready
# through boto3 and print the proxy's reply (or the SDK error) on stdout. Exit 0 only
# when the proxy reports status 200 from the orchestrator. boto3 rather than
# `aws lambda invoke`: the binary-payload flag that call needs exists only in
# AWS CLI v2, and the first live run of this check hit a shell whose `aws` was v1,
# so every invoke failed locally and the deploy blamed the network. boto3 is already
# a hard requirement of this script and behaves the same everywhere.
_ui_api_proxy_probe() {
  "${PYTHON}" - "$1" "${AWS_REGION}" <<'PY'
import json, sys
import boto3
fn, region = sys.argv[1], sys.argv[2]
event = {"method": "GET", "path": "/api/health/ready", "query": {}, "headers": {}, "body": None}
try:
    resp = boto3.client("lambda", region_name=region).invoke(
        FunctionName=fn, Payload=json.dumps(event).encode("utf-8"))
    raw = resp["Payload"].read().decode("utf-8", "replace")
except Exception as exc:  # the SDK call itself failed; say so, do not fake a reply
    print(f"invoke failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
print(raw[:300])
if resp.get("FunctionError"):
    sys.exit(1)
try:
    status = json.loads(raw).get("status")
except ValueError:
    sys.exit(1)
sys.exit(0 if status == 200 else 1)
PY
}

# verify_ui_api_proxy <phase>: drive one request through the proxy and keep trying
# for up to three minutes, because a freshly created NLB registers its targets
# after it has a hostname. The deploy does not report a UI path it has not driven
# end to end. A persistent failure is a Phase failure, with the last reply (or SDK
# error) in the message.
verify_ui_api_proxy() {
  local _phase="$1" deadline last
  deadline=$(( $(date +%s) + 180 ))  # nosemgrep: unquoted-command-substitution-in-command arithmetic context, not subject to word splitting
  while :; do
    if last="$(_ui_api_proxy_probe "${UI_API_PROXY_ARN}" 2>&1)"; then
      ok "UI API proxy reaches the orchestrator (GET /api/health/ready -> 200)"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      fail "${_phase}" "The UI API proxy could not reach the orchestrator through ${ORCHESTRATOR_INTERNAL_URL:-<unset>} after 3 minutes. Last reply: ${last:-<none>}. Check: kubectl get svc ${ORCHESTRATOR_INTERNAL_SVC}; aws logs tail /aws/lambda/${UI_API_PROXY_STACK}."
    fi
    sleep 10
  done
}

# _cfn_output <stack> <output-key>: one output value, or empty.
_cfn_output() {
  aws cloudformation describe-stacks --stack-name "$1" --region "${AWS_REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text 2>/dev/null \
    | grep -v '^None$' | head -n 1
}

# deploy_ui_api_proxy <phase> <nlb-url>: discover the cluster's private
# subnets and security group, create or update the proxy stack pointed at the
# orchestrator's internal NLB, and export UI_API_PROXY_ARN.
deploy_ui_api_proxy() {
  local _phase="$1" orch_url="${2:-}" vpc_id subnets sg status action cfn_log
  [[ "${orch_url}" == http://*.amazonaws.com* ]] \
    || fail "${_phase}" "deploy_ui_api_proxy needs the orchestrator's internal NLB URL (got '${orch_url:-<empty>}'). A cluster.local name cannot be resolved from a Lambda ENI."
  vpc_id="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.vpcId' --output text 2>/dev/null || echo '')"
  sg="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text 2>/dev/null || echo '')"
  [[ -n "${vpc_id}" && "${vpc_id}" != "None" && -n "${sg}" && "${sg}" != "None" ]] \
    || fail "${_phase}" "Could not resolve the VPC and cluster security group of ${CLUSTER_NAME}; the UI API proxy Lambda needs both."
  # The same private subnets eksctl tags for internal load balancers; they route
  # through the NAT gateways, never the internet gateway.
  subnets="$(aws ec2 describe-subnets --region "${AWS_REGION}" \
    --filters "Name=vpc-id,Values=${vpc_id}" "Name=tag:kubernetes.io/role/internal-elb,Values=1" \
    --query 'Subnets[].SubnetId' --output text 2>/dev/null | tr '\t' ',' | tr ' ' ',')"
  [[ -n "${subnets}" && "${subnets}" != "None" ]] \
    || fail "${_phase}" "No private subnets (tag kubernetes.io/role/internal-elb=1) found in ${vpc_id} for the UI API proxy Lambda."

  status="$(aws cloudformation describe-stacks --stack-name "${UI_API_PROXY_STACK}" --region "${AWS_REGION}" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"
  if [[ "${status}" == "ROLLBACK_COMPLETE" || "${status}" == "DELETE_FAILED" ]]; then
    log "  ${UI_API_PROXY_STACK} is ${status}; deleting before recreate"
    aws cloudformation delete-stack --stack-name "${UI_API_PROXY_STACK}" --region "${AWS_REGION}"
    aws cloudformation wait stack-delete-complete --stack-name "${UI_API_PROXY_STACK}" --region "${AWS_REGION}"
    status=""
  fi
  action="create-stack"
  [[ -n "${status}" && "${status}" != "None" ]] && action="update-stack"
  cfn_log="/tmp/cfn-deploy-${UI_API_PROXY_STACK}-$(date +%s).log"
  log "  ${action} ${UI_API_PROXY_STACK} (subnets ${subnets}, sg ${sg})"
  if aws cloudformation "${action}" --stack-name "${UI_API_PROXY_STACK}" \
      --template-body "file://${SCRIPT_DIR}/ui_api_proxy_cfn.yaml" \
      --capabilities CAPABILITY_NAMED_IAM --region "${AWS_REGION}" \
      --parameters "ParameterKey=StackPrefix,ParameterValue=${STACK_PREFIX}" \
                   "ParameterKey=SubnetIds,ParameterValue=\"${subnets}\"" \
                   "ParameterKey=SecurityGroupIds,ParameterValue=${sg}" \
                   "ParameterKey=OrchestratorBaseUrl,ParameterValue=${orch_url}" \
      >"${cfn_log}" 2>&1; then
    local wait_action="stack-create-complete"
    [[ "${action}" == "update-stack" ]] && wait_action="stack-update-complete"
    aws cloudformation wait "${wait_action}" --stack-name "${UI_API_PROXY_STACK}" --region "${AWS_REGION}" \
      || fail "${_phase}" "UI API proxy stack ${UI_API_PROXY_STACK} did not reach ${wait_action}; see: aws cloudformation describe-stack-events --stack-name ${UI_API_PROXY_STACK}"
  elif ! grep -q "No updates are to be performed" "${cfn_log}"; then
    fail "${_phase}" "UI API proxy stack ${UI_API_PROXY_STACK} ${action} failed: $(cat "${cfn_log}")"
  fi
  UI_API_PROXY_ARN="$(_cfn_output "${UI_API_PROXY_STACK}" ProxyFunctionArn)"
  [[ -n "${UI_API_PROXY_ARN}" ]] || fail "${_phase}" "${UI_API_PROXY_STACK} has no ProxyFunctionArn output."
  ok "UI API proxy: ${UI_API_PROXY_ARN}"
}

# phase_hint(): a short remediation hint per phase, printed by fail() as
# additional context above the real captured error — never a replacement
# for it (FR-11). A case statement (not an associative array) for bash 3.2
# compatibility (macOS default) — see _source_hash()/display_name() above
# for the same pattern already used elsewhere in this script.
phase_hint() {
  case "$1" in
    1) echo "Check AWS credentials (aws sts get-caller-identity) and that boto3/torch/onnx/onnxscript are installed." ;;
    2) echo "Check NVIDIA NGC credentials (--ngc-key/--ngc-secret) for CodeBuild and your EC2 g5 GPU service quota for cluster creation. Nodes 'failed to join' usually means they cannot reach the EKS endpoint or ECR from their private subnets (NAT gateway / route table)." ;;
    3) echo "Check GPU node availability: kubectl get nodes -l nvidia.com/gpu=present ; kubectl get pods ; and the UI API proxy stack: aws cloudformation describe-stack-events --stack-name ${UI_API_PROXY_STACK:-<prefix>-ui-api-proxy}" ;;
    4) echo "Check the CloudFront/S3 frontend deploy output above and Cognito user pool creation." ;;
    5) echo "Check Bedrock model access (--model-id) and the AgentCore runtime IAM roles." ;;
    *) echo "" ;;
  esac
}

# _can_prompt(): true only when this process can ACTUALLY read the terminal.
#
# `[[ -t 0 ]]` alone is not that test, and the difference is a hang. For
# `nohup ./deploy.sh … &` stdin is STILL the terminal, so `-t 0` passes — but a
# background process that reads the terminal is sent SIGTTIN and STOPS. Observed
# live: a backgrounded deploy sat in state T ("stopped") at the resume prompt with
# the EKS cluster already built and nothing progressing, which is precisely the
# failure mode this feature exists to remove.
#
# The real question is whether our process group is the terminal's FOREGROUND
# process group. `ps -o tpgid=` reports the foreground group of the controlling
# terminal; comparing it with our own pgid answers it.
#
# If ps cannot answer, the safe assumption is "cannot prompt". Skipping a question
# costs a default; blocking on one costs the deployment.
_can_prompt() {
  [[ -t 0 ]] || return 1
  local _pgid _tpgid
  _pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
  _tpgid="$(ps -o tpgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
  [[ -n "${_pgid}" && -n "${_tpgid}" && "${_pgid}" == "${_tpgid}" ]]
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
    printf '\033[0;33m[hint]\033[0m Where this got to, live from AWS: ./deploy.sh --prefix %s --status\n' \
      "${STACK_PREFIX:-}" >&2
  else
    printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2
  fi
  if [[ -n "${DEPLOY_LOG:-}" ]]; then
    printf '\033[0;33m[hint]\033[0m Full output of every step this run: %s\n' "${DEPLOY_LOG}" >&2
  fi
  exit 1
}

# report_closed_loop_stacks(): name which of the closed-loop stacks actually
# exist, and which do not.
#
# Replaces a generic "retraining infra may need manual intervention" with the real
# list. The failure this exists for lands mid-sequence -- the first four stacks get
# created and the last two do not -- so "it failed" is much less useful than "these
# four are there, these two are not". Names mirror deploy_closed_loop.sh's own
# conventions (see its stack-naming block).
report_closed_loop_stacks() {
  local stack status
  warn "Closed-loop stack status:"
  for stack in \
    "${STACK_PREFIX:+${STACK_PREFIX}-}feedback-pipeline" \
    "${STACK_PREFIX:+${STACK_PREFIX}-}glue-etl" \
    "${STACK_PREFIX:+${STACK_PREFIX}-}closed-loop-core" \
    "${STACK_PREFIX:+${STACK_PREFIX}-}agentcore-security" \
    "${STACK_PREFIX:+${STACK_PREFIX}-}vpc-proxy" \
    "${STACK_PREFIX:+${STACK_PREFIX}-}governance-eventbridge"
  do
    status="$(aws cloudformation describe-stacks --stack-name "${stack}" --region "${AWS_REGION}" \
      --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"
    if [[ -n "${status}" && "${status}" != "None" ]]; then
      printf '    %-28s present (%s)\n' "${stack}" "${status}" >&2
    else
      printf '    %-28s ABSENT\n' "${stack}" >&2
    fi
  done
  return 0
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

# Resolve a base Python 3 interpreter (prefer python3, fall back to python if 3.x).
if command -v python3 >/dev/null 2>&1; then
  _BASE_PYTHON="python3"
elif command -v python >/dev/null 2>&1 && python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
  _BASE_PYTHON="python"
else
  fail "missing: python3 (>= 3.11)"
fi

# Run all Python through a virtualenv so this script never MODIFIES the caller's
# system Python. Installing our deps into a shared system interpreter is what
# caused the earlier failures: a system sagemaker v3 (plus sagemaker-mlops/
# -train/-serve) shadowed the v2 `image_uris` this deploy needs, and pinning it
# there produced sagemaker-core dependency conflicts. The venv is created with
# --system-site-packages so it REUSES already-installed packages (torch, boto3,
# the agentcore toolkit, ...) instead of refetching them, while anything we pip
# install (e.g. sagemaker v2) lands in the venv and SHADOWS the system copy --
# so the system site-packages are never changed. If a venv is already active,
# it's respected.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON="${_BASE_PYTHON}"
  log "Using the active virtualenv Python: $("${PYTHON}" --version 2>&1) ($(command -v "${PYTHON}"))"
elif "${_BASE_PYTHON}" -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null; then
  # The resolved interpreter is itself a venv (not the system one) -- reuse it.
  PYTHON="${_BASE_PYTHON}"
  log "Using virtualenv Python: $("${PYTHON}" --version 2>&1) ($(command -v "${PYTHON}"))"
else
  DEPLOY_VENV="${SCRIPT_DIR}/.deploy-venv"
  if [[ ! -x "${DEPLOY_VENV}/bin/python" ]]; then
    log "Creating a deploy virtualenv at ${DEPLOY_VENV} (so this script installs its Python deps here, not into your system Python)"
    "${_BASE_PYTHON}" -m venv --system-site-packages "${DEPLOY_VENV}" \
      || fail "could not create a virtualenv at ${DEPLOY_VENV} (need the stdlib 'venv' module for ${_BASE_PYTHON})"
  fi
  PYTHON="${DEPLOY_VENV}/bin/python"
  "${PYTHON}" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
  log "Using deploy virtualenv: $("${PYTHON}" --version 2>&1) (${PYTHON})"  # nosemgrep: unquoted-variable-expansion-in-command the remaining expansion is inside the double-quoted message; semgrep bash parse artifact
fi

# Directory holding the resolved interpreter, prepended to PATH when invoking
# the child deploy_closed_loop.sh (which calls bare `python3`) so it uses the
# same venv -- otherwise Phase 5 (register_genesis_models.py, its own XGBoost
# image_uris resolution) would fall back to the system Python this venv exists
# to avoid.
PYTHON_BIN_DIR="$(cd "$(dirname "$(command -v "${PYTHON}" 2>/dev/null || echo "${PYTHON}")")" 2>/dev/null && pwd || true)"

# Ensure required Python packages are available (install if missing).
# onnxruntime: Step 2's Deal Scorer gate scores the exported ONNX with it, and the
# export fails without it. When missing it is installed at ONNXRUNTIME_PIN, the
# version the gate was verified with.
REQUIRED_PY_PACKAGES="torch onnx onnxscript onnxruntime boto3"
ONNXRUNTIME_PIN="onnxruntime==1.27.0"
MISSING_PY_PACKAGES=""
for pkg in ${REQUIRED_PY_PACKAGES}; do
  if ! "${PYTHON}" -c "import ${pkg}" 2>/dev/null; then
    case "${pkg}" in
      onnxruntime) MISSING_PY_PACKAGES="${MISSING_PY_PACKAGES} ${ONNXRUNTIME_PIN}" ;;
      *) MISSING_PY_PACKAGES="${MISSING_PY_PACKAGES} ${pkg}" ;;
    esac
  fi
done
if [[ -n "${MISSING_PY_PACKAGES}" ]]; then
  log "Installing missing Python packages:${MISSING_PY_PACKAGES}"
  "${PYTHON}" -m pip install --quiet ${MISSING_PY_PACKAGES} || fail "pip install failed for:${MISSING_PY_PACKAGES}"  # nosemgrep: unquoted-variable-expansion-in-command intentional word splitting of a space separated package list
fi

# sagemaker SDK: needed only for the Part 2 closed-loop path — resolving the
# built-in XGBoost training image URI (below) and registering genesis model
# versions (Step 5, register_genesis_models.py). It is NOT required for a
# Part 1-only deploy, so it is installed here conditionally rather than added
# to REQUIRED_PY_PACKAGES above. Without it, the XGBoost image URI resolves
# empty and on-demand Yield Optimizer (floor/margin) training reports a 503 in
# the UI ("XGBOOST_TRAINING_IMAGE_URI unset").
#
# The check imports `image_uris` specifically, NOT just `sagemaker`: the SDK's
# v3 major version dropped the top-level `image_uris` module this project's
# resolution relies on (`image_uris.retrieve`, also used by
# register_genesis_models.py), so a python that has sagemaker v3 installed
# would pass a plain `import sagemaker` yet still fail to resolve the URI. Pin
# to the v2 line (>=2,<3), which is what the whole closed-loop path targets.
if [[ "${WITH_RETRAINING}" -eq 1 ]] && ! "${PYTHON}" -c "from sagemaker import image_uris" 2>/dev/null; then
  log "Installing/ensuring sagemaker SDK (v2, with image_uris) for closed-loop retraining"
  "${PYTHON}" -m pip install --quiet 'sagemaker>=2,<3' || fail "pip install failed for: sagemaker>=2,<3 (needed for --with-retraining; re-run with --no-retraining to skip Part 2)"
fi

# step()/done_ok()/run_logged() and the status marks. Sourced before the gate so
# `--status` output uses the same glyphs as the deploy itself.
# shellcheck source=lib/deploy_progress.sh
source "${SCRIPT_DIR}/lib/deploy_progress.sh"

STATE_PYTHON="${PYTHON}"
# shellcheck source=lib/deploy_state.sh
source "${SCRIPT_DIR}/lib/deploy_state.sh"
# shellcheck source=lib/deploy_gate.sh
source "${SCRIPT_DIR}/lib/deploy_gate.sh"
# shellcheck source=lib/k8s_rollout.sh
source "${SCRIPT_DIR}/lib/k8s_rollout.sh"

# =========================================================================
# AWS profile -- resolved here, before the first AWS call in any mode
# =========================================================================
# Order: --profile, AWS_PROFILE env (both captured as DEPLOY_PROFILE above), the
# profile remembered for this prefix in .deploy-state.json, then "default".
#
# This sits BEFORE --status and before ACCOUNT_ID on purpose. --status is the one
# mode that skips the remembered-values block further down, and ACCOUNT_ID is the
# first AWS call -- a profile resolved after it would have authenticated with the
# wrong one. Every mode (deploy, --status, --destroy, --ui-only) passes through here.
_REMEMBERED_PROFILE="$(state_read "${STACK_PREFIX}" remembered.profile)"
if [[ -n "${DEPLOY_PROFILE}" ]]; then
  # A profile remembered for this prefix is the one that created its resources. A
  # different one now is either a typo or a second account, and both are stopped
  # here rather than discovered as a CloudFormation name collision 20 minutes in.
  # Deleting the prefix's record is the deliberate way to switch.
  if [[ -n "${_REMEMBERED_PROFILE}" && "${_REMEMBERED_PROFILE}" != "${DEPLOY_PROFILE}" ]]; then
    fail "AWS profile mismatch for prefix '${STACK_PREFIX}': this run resolved '${DEPLOY_PROFILE}' (from ${DEPLOY_PROFILE_SOURCE}) but the prefix was deployed with '${_REMEMBERED_PROFILE}' (remembered in ${DEPLOY_STATE_FILE}). Re-run with --profile ${_REMEMBERED_PROFILE}, or -- to move this prefix to another account on purpose -- remove its record from ${DEPLOY_STATE_FILE} first."
  fi
elif [[ -n "${_REMEMBERED_PROFILE}" ]]; then
  DEPLOY_PROFILE="${_REMEMBERED_PROFILE}"
  DEPLOY_PROFILE_SOURCE="remembered for prefix '${STACK_PREFIX}' in ${DEPLOY_STATE_FILE}"
else
  DEPLOY_PROFILE="default"
  DEPLOY_PROFILE_SOURCE="fallback; no --profile, no AWS_PROFILE, nothing remembered"
fi
export AWS_PROFILE="${DEPLOY_PROFILE}"

# A NAMED profile that is not configured locally fails on the first SDK call with a
# message that names the step, not the cause. Check it once here instead. "default"
# is exempt: botocore tolerates its absence (environment-variable credentials still
# work), so refusing it would break a perfectly valid setup.
#
# Checked with botocore rather than `aws configure list-profiles`: that subcommand
# is CLI v2 only, and a v1 binary earlier on PATH (observed on a developer Mac) makes
# it print usage text and exit 0 -- which would have passed every profile.
if [[ "${AWS_PROFILE}" != "default" ]]; then
  if ! env -u AWS_PROFILE "${PYTHON}" -c '
import sys, botocore.session
sys.exit(0 if sys.argv[1] in botocore.session.Session().available_profiles else 1)
' "${AWS_PROFILE}" 2>/dev/null; then
    fail "AWS profile '${AWS_PROFILE}' (from ${DEPLOY_PROFILE_SOURCE}) is not configured in ~/.aws/config or ~/.aws/credentials. Available: $(env -u AWS_PROFILE "${PYTHON}" -c 'import botocore.session; print(" ".join(sorted(botocore.session.Session().available_profiles)))' 2>/dev/null || echo '(could not list)')"
  fi
fi
say "  AWS profile: ${AWS_PROFILE} (${DEPLOY_PROFILE_SOURCE})"

# =========================================================================
# What this checkout would deploy, for the phase probe (read-only)
# =========================================================================
# The probe can see the Phase 1 marker in the model bucket and the Deal Scorer
# engine's source-fingerprint, but not what THIS checkout would put there. These
# two values are that; lib/deploy_gate.sh passes them to every probe, which reads
# Phase 1 (marker) and Phase 3 (engine) as stale when they differ. Computed here,
# before --status and the first gate_refresh, and without any AWS call.
DEPLOY_EXPECT_DEALS_SEED_SHA=""
DEPLOY_EXPECT_NCF_FINGERPRINT=""

# Phases _run_phase runs even when the probe reads them as complete, each with
# the reason it prints. Exporting FORCE_PHASE_<n>=1 before the run forces one by
# hand; force_phase sets them from the read-only checks below.
FORCE_PHASE_1="${FORCE_PHASE_1:-0}"; FORCE_PHASE_1_WHY="FORCE_PHASE_1=1 is set"
FORCE_PHASE_3="${FORCE_PHASE_3:-0}"; FORCE_PHASE_3_WHY="FORCE_PHASE_3=1 is set"
FORCE_PHASE_5="${FORCE_PHASE_5:-0}"; FORCE_PHASE_5_WHY="FORCE_PHASE_5=1 is set"

# force_phase <1|3|5> <why>: make _run_phase run phase n, and say why.
force_phase() {
  local n="$1" why="$2" prev
  case "${n}" in
    1) prev=""; if [[ "${FORCE_PHASE_1}" == 1 ]]; then prev="${FORCE_PHASE_1_WHY}"; fi
       FORCE_PHASE_1=1; FORCE_PHASE_1_WHY="${prev:+${prev}; }${why}" ;;
    3) prev=""; if [[ "${FORCE_PHASE_3}" == 1 ]]; then prev="${FORCE_PHASE_3_WHY}"; fi
       FORCE_PHASE_3=1; FORCE_PHASE_3_WHY="${prev:+${prev}; }${why}" ;;
    5) prev=""; if [[ "${FORCE_PHASE_5}" == 1 ]]; then prev="${FORCE_PHASE_5_WHY}"; fi
       FORCE_PHASE_5=1; FORCE_PHASE_5_WHY="${prev:+${prev}; }${why}" ;;
  esac
  return 0
}

# file_sha256 <path>: hex sha256 of the file's bytes (same digest as shasum -a 256).
file_sha256() {
  "${PYTHON}" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
}

# compute_phase_expectations: set DEPLOY_EXPECT_DEALS_SEED_SHA (sha256 of the
# deals seed Step 1.5 writes) and DEPLOY_EXPECT_NCF_FINGERPRINT (the training
# fingerprint Step 2 builds, from build.py --fingerprint-only). A value that
# cannot be computed forces Phase 1, whose own step then reports the cause.
compute_phase_expectations() {
  local seed="${SCRIPT_DIR}/seed/deals_and_campaigns.json" out
  out="$(file_sha256 "${seed}" 2>/dev/null || true)"
  if [[ "${out}" =~ ^[0-9a-f]{64}$ ]]; then
    DEPLOY_EXPECT_DEALS_SEED_SHA="${out}"
  else
    warn "Could not hash ${seed}; running Phase 1 so Step 1.5 reports why."
    force_phase 1 "the local deals seed could not be hashed"
  fi
  out="$("${PYTHON}" "${SCRIPT_DIR}/../source/training/ncf_sample/build.py" --fingerprint-only 2>/dev/null || true)"
  out="${out##*$'\n'}"
  if [[ "${out}" =~ ^[0-9a-f]{64}$ ]]; then
    DEPLOY_EXPECT_NCF_FINGERPRINT="${out}"
  else
    warn "Could not compute the local Deal Scorer fingerprint (source/training/ncf_sample/build.py --fingerprint-only); running Phase 1 so Step 2 reports why."
    force_phase 1 "the local Deal Scorer fingerprint could not be computed"
  fi
  return 0
}

if [[ "${STATUS_ONLY}" -eq 1 ]] || [[ "${DESTROY}" -eq 0 && "${UI_ONLY}" -eq 0 && "${EXPORT_ONLY}" -eq 0 ]]; then
  compute_phase_expectations
fi

# --status: answer from AWS and stop. Deliberately before every other check --
# it needs no credentials beyond read access, no kubeconfig, no Docker, and no
# knowledge of whether anything is running.
if [[ "${STATUS_ONLY}" -eq 1 ]]; then
  gate_print "${STACK_PREFIX}"
  # The probe's own code, so `--status` is usable in a script: 0 = fully deployed.
  exit "${GATE_PRINT_RC}"
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
[[ -n "${ACCOUNT_ID}" ]] || fail "cannot resolve AWS account (profile '${AWS_PROFILE}')"
# Always visible, next to the profile line above: a run against the wrong account is
# readable at the top of the output rather than from a stack-name collision later.
say "  AWS account: ${ACCOUNT_ID}  region: ${AWS_REGION}  prefix: ${STACK_PREFIX}"

# The detail log exists from here on. Named now, before the first thing worth
# logging, and named in every fail() after this point.
_deploy_log_init
if [[ "${VERBOSE}" -eq 0 ]]; then
  say "  Detail log:  ${DEPLOY_LOG}   (tail -f to watch a step; --verbose streams it here instead)"
fi

# =========================================================================
# Local deployment state — remembered inputs and the resume decision
# =========================================================================
# Sits HERE, between account resolution and the first thing derived from
# AWS_REGION, because a remembered region has to be in place before REGISTRY,
# STACK_UID, MODEL_BUCKET and friends are computed from it. Moving this later
# would compute half the resource names from the default region and the other
# half from the remembered one.
#
# Skipped entirely for --destroy / --ui-only / --export-only: none of them run
# phases, so there is no progress to resume and nothing to remember. (--destroy
# still clears the record; that happens inside its own block.)
# State helpers and the follow decision are already loaded above (they had to run
# before ACCOUNT_ID). This block continues with the parts that need the resolved
# account and region.
if [[ "${DESTROY}" -eq 0 && "${UI_ONLY}" -eq 0 && "${EXPORT_ONLY}" -eq 0 ]]; then
  _STATE_CLUSTER_NAME="${STACK_NAME}-triton"   # mirrors CLUSTER_NAME below
  _STATE_CONFLICTS="$(state_start_run "${STACK_PREFIX}" "${ACCOUNT_ID}" "${AWS_REGION}" \
    "${STACK_NAME}" "${_STATE_CLUSTER_NAME}" "${_DEPLOY_ARGV[@]+"${_DEPLOY_ARGV[@]}"}")"

  # A conflict means the record describes a DIFFERENT environment under the same
  # prefix key -- a different account, or a STACK_NAME env override. Reusing its
  # values would silently point this run at the wrong place, so reuse is skipped
  # wholesale. start-run already printed the detail to stderr.
  _STATE_REUSE=1
  if [[ -n "${_STATE_CONFLICTS}" ]]; then
    _STATE_REUSE=0
    warn "Not reusing remembered values for prefix '${STACK_PREFIX:-<none>}' (${_STATE_CONFLICTS} differ from the recorded run)."
  fi

  # --- Apply remembered values for flags this run did NOT give (FR-10/FR-11) ---
  # Explicit always wins: each branch is gated on its _GIVEN_ marker. Every value
  # actually applied is collected and printed, so reuse is never silent (FR-12).
  _REUSED_LINES=()
  _reuse() {  # _reuse <var-name> <remembered-key> <given-marker> [label]
    local var="$1" key="$2" given="$3" label="${4:-$2}" stored
    [[ "${_STATE_REUSE}" -eq 1 && "${given}" -eq 0 ]] || return 0
    stored="$(state_read "${STACK_PREFIX}" "remembered.${key}")"
    [[ -n "${stored}" ]] || return 0
    eval "${var}=\"\${stored}\""
    _REUSED_LINES+=("    ${label} = ${stored}")
    return 0
  }
  _reuse AWS_REGION       region        "${_GIVEN_REGION}"           "AWS_REGION"
  _reuse MAX_GPUS         maxGPUs       "${_GIVEN_MAXGPUS}"          "--maxGPUs"
  _reuse ARTF_NODE_ROLE   artfNodeRole  "${_GIVEN_ARTF_NODE_ROLE}"   "--artf-node-role"
  _reuse BEDROCK_MODEL_ID modelId       "${_GIVEN_MODEL_ID}"         "--model-id"
  _reuse LOCAL_BUILD      localBuild    "${_GIVEN_LOCAL_BUILD}"      "--local-build"
  _reuse WITH_RETRAINING  withRetraining "${_GIVEN_RETRAINING}"      "--with-retraining"
  _reuse WITH_PREBID      withPrebid    "${_GIVEN_PREBID}"           "--with-prebid"
  _reuse SKIP_AGENTCORE   skipAgentcore "${_GIVEN_SKIP_AGENTCORE}"   "--skip-agentcore"
  # The NGC secret name is the case that prompted all of this: a re-run that
  # omits --ngc-key should not re-prompt for a key already stored for this stack.
  # Only applied when neither NGC flag was given on this invocation.
  if [[ "${_STATE_REUSE}" -eq 1 && -z "${NGC_KEY}" && -z "${NGC_SECRET}" ]]; then
    _REMEMBERED_NGC="$(state_read "${STACK_PREFIX}" remembered.ngcSecretName)"
    if [[ -n "${_REMEMBERED_NGC}" ]]; then
      NGC_SECRET="${_REMEMBERED_NGC}"
      _REUSED_LINES+=("    --ngc-secret = ${_REMEMBERED_NGC}")
    fi
  fi

  if [[ ${#_REUSED_LINES[@]} -gt 0 ]]; then
    say ""
    say "  Reusing values from the last deploy of prefix '${STACK_PREFIX:-<none>}':"
    for _line in "${_REUSED_LINES[@]}"; do say "${_line}"; done
    say "    (pass the flag explicitly to override, or delete deployment/.deploy-state.json)"
  fi

  # --- Re-validate anything a remembered value could have changed ---
  # The original validation ran before this block, against the defaults. A value
  # restored from a hand-edited or older state file has not been checked yet.
  if ! [[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    fail "--maxGPUs must be a positive integer (got '${MAX_GPUS}', restored from deployment/.deploy-state.json). Pass --maxGPUs explicitly to override."
  fi
  if [[ "${ARTF_NODE_ROLE}" != "services" && "${ARTF_NODE_ROLE}" != "inference" ]]; then
    fail "--artf-node-role must be 'services' or 'inference' (got '${ARTF_NODE_ROLE}', restored from deployment/.deploy-state.json). Pass --artf-node-role explicitly to override."
  fi

  # --- What is already deployed, according to AWS ---
  #
  # This replaced a resume PROMPT, and the prompt's removal is the point rather
  # than a side effect. The prompt existed only because a local progress file
  # could not be trusted, so the script had to ask a human whether to believe it.
  # With AWS answering, there is nothing to ask: each phase is skipped if the
  # resources it creates already exist, waited on if AWS is mid-operation, and run
  # otherwise. Convergence, not interrogation.
  #
  # It also removes an entire failure mode. That prompt blocked on `read`, which
  # meant a deploy could sit for an hour holding no resources and doing no work,
  # invisible to anyone not looking at that exact terminal.
  gate_refresh "${STACK_PREFIX}"
  if [[ "${GATE_AVAILABLE}" -eq 1 ]]; then
    say ""
    gate_print "${STACK_PREFIX}"
    say ""
    if [[ "${DEPLOY_OVERALL}" == "ok" && "${_GIVEN_START_AT}" -eq 0 ]]; then
      say "  Everything above is already deployed. Re-running is safe: each phase"
      say "  checks AWS first, so this will confirm rather than rebuild."
    fi
  else
    warn "Could not read deployment status from AWS; every phase will run."
    warn "That is safe -- each phase is idempotent -- but slower than necessary."
  fi

  # --resume is kept as a no-op alias so existing scripts and docs do not break.
  # It used to mean "trust the local file and skip ahead"; skipping ahead is now
  # the default behaviour for anything AWS reports as already present.
  if [[ "${RESUME}" -eq 1 ]]; then
    say "  --resume is no longer needed: completed phases are detected from AWS."
  fi

  # Record this run's inputs so the NEXT run can reuse them. Written now rather
  # than at the end, so a run that dies mid-way still leaves them behind.
  # `profile` is what the mismatch check in the "AWS profile" block above compares
  # against on the next run. It is the resolved value, so a run that fell back to
  # "default" remembers "default" -- and a later --profile other-thing is stopped.
  state_set "${STACK_PREFIX}" remembered \
    "profile=${AWS_PROFILE}" \
    "region=${AWS_REGION}" \
    "maxGPUs=${MAX_GPUS}" \
    "artfNodeRole=${ARTF_NODE_ROLE}" \
    "modelId=${BEDROCK_MODEL_ID}" \
    "localBuild=${LOCAL_BUILD}" \
    "withRetraining=${WITH_RETRAINING}" \
    "withPrebid=${WITH_PREBID}" \
    "skipAgentcore=${SKIP_AGENTCORE}"
  # The secret NAME, never the key. remote_build.sh derives this same name from
  # STACK_NAME, so recording it needs no round trip through the child script.
  if [[ -n "${NGC_KEY}" ]]; then
    state_set "${STACK_PREFIX}" remembered "ngcSecretName=${STACK_NAME}-ngc-api-key"
  elif [[ -n "${NGC_SECRET}" ]]; then
    state_set "${STACK_PREFIX}" remembered "ngcSecretName=${NGC_SECRET}"
  fi
fi

# _run_phase <n> — should phase n execute?
#
# Three inputs, in order:
#   1. --start-at N, which still wins. It is the explicit override for someone who
#      knows something the probe cannot (e.g. "rebuild my container even though an
#      image exists").
#   2. AWS mid-operation: wait for it instead of starting a second one. This is why
#      a second terminal is now useful rather than destructive.
#   3. AWS already satisfied: skip, and say so.
#
# `unknown` runs the phase. Every phase is idempotent, so re-running one we could
# not verify costs minutes; skipping one we could not verify ships a deployment
# that claims to be complete and is not.
#
# One exception to "AWS already satisfied: skip". The probe reads what is RUNNING,
# and a Deployment whose pods run an image that Step 4 has just rebuilt under the
# same tag is "ok" by every check the probe has, so Phase 3 (which carries the
# rollout restart) would be skipped and the new image would never ship. That is
# exactly what happened once: nine images rebuilt, every phase skipped, the old
# orchestrator kept serving. So build_images records that it rebuilt something,
# and Phase 3 runs whenever it did; Phase 5 likewise when the AgentCore image is
# among the rebuilt ones, since the runtime only picks a new image up through
# update-agent-runtime.
#
# The same gap exists without a rebuild in this run, which the probe cannot see
# either: pods still on an older digest than ECR holds for the tag, or an AgentCore
# runtime registered with an older AgentCore image. The read-only checks before
# Phase 1 find those and force_phase 3 or 5 (FORCE_PHASE_<n>, with the reason).
_IMAGES_REBUILT=0
_AGENTCORE_IMAGE_REBUILT=0
_run_phase() {
  local n="$1" st why=""
  [[ "${START_AT}" -le "${n}" ]] || return 1
  gate_wait "${n}" "${STACK_PREFIX}" || true
  if gate_should_run "${n}"; then
    return 0
  fi
  # --start-at N names the phase the operator wants run; that is the documented
  # override of the probe (header: "for when you know something the probe cannot").
  # Only the named phase is forced; the ones after it still consult AWS. Without
  # this, `--start-at 3` on a healthy cluster did nothing, and the header was wrong.
  if [[ "${_GIVEN_START_AT}" -eq 1 && "${n}" -eq "${START_AT}" ]]; then
    say "  Phase ${n}/5 reads as complete in AWS; running it anyway because --start-at ${n} names it."
    return 0
  fi
  if [[ "${n}" -eq 3 && "${_IMAGES_REBUILT}" -eq 1 ]]; then
    say "  Phase 3/5 reads as complete in AWS, but Step 4 rebuilt images under the same tag; running it so the new images roll out."
    return 0
  fi
  if [[ "${n}" -eq 5 && "${_AGENTCORE_IMAGE_REBUILT}" -eq 1 ]]; then
    say "  Phase 5/5 reads as complete in AWS, but Step 4 rebuilt the AgentCore image; running it so the runtime is updated."
    return 0
  fi
  if [[ "${n}" -eq 1 && "${FORCE_PHASE_1:-0}" == 1 ]]; then
    why="${FORCE_PHASE_1_WHY:-FORCE_PHASE_1=1 is set}"
  elif [[ "${n}" -eq 3 && "${FORCE_PHASE_3:-0}" == 1 ]]; then
    why="${FORCE_PHASE_3_WHY:-FORCE_PHASE_3=1 is set}"
  elif [[ "${n}" -eq 5 && "${FORCE_PHASE_5:-0}" == 1 ]]; then
    why="${FORCE_PHASE_5_WHY:-FORCE_PHASE_5=1 is set}"
  fi
  if [[ -n "${why}" ]]; then
    say "  Phase ${n}/5 reads as complete in AWS, but ${why}; running it."
    return 0
  fi
  st="$(gate_status "${n}")"
  ok "Phase ${n}/5 already complete in AWS (${st}) — skipping"
  return 1
}

# Tracks whether any subsystem failed without aborting the run. A non-zero value
# suppresses the success banner and makes the script exit non-zero (FR-25/FR-26).
_DEPLOY_DEGRADED=0
_DEGRADED_DETAIL=""

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Deterministic UID for resource naming
STACK_UID="$("${PYTHON}" -c "import hashlib; print(hashlib.sha256('${STACK_NAME}:${ACCOUNT_ID}:${AWS_REGION}'.encode()).hexdigest()[:8])")"

CLUSTER_NAME="${STACK_NAME}-triton"
MODEL_BUCKET="${STACK_NAME}-triton-models-${STACK_UID}"
# Deterministic names matching deploy_closed_loop.sh's own conventions
# (TRAINING_DATA_BUCKET mirrors its glue_etl_cfn.yaml TrainingDataBucketName
# naming; these only resolve to real resources if --with-retraining was
# used, but the orchestrator's governance_api.py already reports a clear
# 503 rather than fabricating success if the underlying bucket/role
# doesn't exist).
TRAINING_DATA_BUCKET="${STACK_PREFIX:+${STACK_PREFIX}-}training-data-${ACCOUNT_ID}-${AWS_REGION}"
# Glue job names, matching glue_etl_cfn.yaml's naming convention exactly
# (FeatureEngineeringJob/DealYieldFeatureEngineeringJob) -- used by the
# Governance panel's "Train from load test" run picker to only list runs
# whose data has actually been swept into training-data/ by a completed
# Glue job run (see orchestrator/governance_api.py's trainable_runs_handler).
GLUE_JOB_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}feature-engineering-etl"
DEAL_YIELD_GLUE_JOB_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}deal-yield-feature-engineering-etl"
# Deterministic Model Package Group names for the Yield Optimizer's two
# independently-trained sub-models, matching closed_loop_cfn.yaml's naming
# exactly (same convention as DLRM_MODEL_GROUP/NCF_MODEL_GROUP below --
# resolved deterministically here rather than via a stack-output lookup,
# since this substitution happens in Phase 3, before deploy_closed_loop.sh
# (Phase 5) creates/queries any closed-loop stack outputs).
YIELD_FLOOR_MODEL_GROUP="${STACK_PREFIX:+${STACK_PREFIX}-}artf-deal-yield-manager-floor"
YIELD_MARGIN_MODEL_GROUP="${STACK_PREFIX:+${STACK_PREFIX}-}artf-deal-yield-manager-margin"
# SageMaker's built-in XGBoost algorithm image lives in an AWS-owned
# account that differs per region -- resolved via the sagemaker SDK, never
# hardcoded/guessed (same resolution deploy_closed_loop.sh already uses
# for the scheduled retraining Lambda). Required only for the on-demand
# "Train from load test" trigger's deal_yield_manager_floor/margin path
# (orchestrator/training_trigger.py's xgboost-shaped branch) -- left empty
# if the sagemaker SDK isn't installed, and training_trigger.py reports a
# clear 503 for that model type rather than fabricating a URI.
# NOTE the stdout redirect below. Importing sagemaker emits
# "sagemaker.config INFO - Not applying SDK defaults from location: ..." lines
# on STDOUT (one per config path it checks), so a bare `print(uri)` here
# captured THREE lines, not one. That broke the Phase-3 manifest substitution
# with "sed: 1: unescaped newline inside substitute pattern" and
# would have passed a multi-line value as a CloudFormation ParameterValue in
# deploy_closed_loop.sh. `2>/dev/null` cannot help -- the noise is on stdout.
# Redirecting stdout for the import+retrieve sends that logging to stderr
# (where it is discarded) and leaves stdout carrying only the URI.
# Import noise ("sagemaker.config INFO ..." on stdout) is swallowed into a
# throwaway buffer so only the URI reaches real stdout; SAGEMAKER_SUPPRESS_V2_WARNING
# keeps the SDK's v2 deprecation notice off stderr. Any REAL exception is written
# to stderr and captured below, so an empty result is explained rather than
# silently swallowed (the old `except: pass` hid, e.g., "No module named
# 'sagemaker'" when deploy ran with an interpreter that lacked the SDK).
_XGB_ERR_FILE="$(mktemp)"
XGBOOST_TRAINING_IMAGE_URI="$(SAGEMAKER_SUPPRESS_V2_WARNING=1 "${PYTHON}" -c "
import contextlib, io, sys
try:
    with contextlib.redirect_stdout(io.StringIO()):
        from sagemaker import image_uris
        _uri = image_uris.retrieve(framework='xgboost', region='${AWS_REGION}', version='1.7-1')
    sys.stdout.write(_uri)
except Exception as e:
    sys.stderr.write('%s: %s' % (type(e).__name__, e))
" 2>"${_XGB_ERR_FILE}" || true)"
_XGB_ERR="$(tr '\n' ' ' < "${_XGB_ERR_FILE}" 2>/dev/null | sed 's/  */ /g' | cut -c1-300)"
rm -f "${_XGB_ERR_FILE}"
# Second line of defense: only accept something that actually looks like an ECR
# image URI on a single line. Anything else (a future SDK logging change, a
# partial write) becomes empty, so training_trigger.py reports its honest 503
# instead of a fabricated or corrupted image URI reaching SageMaker.
_XGB_URI_RE='^[0-9]+\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(\.cn)?/[^[:space:]]+$'
if [[ -n "${XGBOOST_TRAINING_IMAGE_URI}" && ! "${XGBOOST_TRAINING_IMAGE_URI}" =~ ${_XGB_URI_RE} ]]; then
  warn "Ignoring unexpected XGBoost training image URI from the sagemaker SDK (not a single-line ECR URI); on-demand yield retraining will report 503 until this resolves cleanly."
  XGBOOST_TRAINING_IMAGE_URI=""
fi
# Surface an empty resolution rather than letting it pass silently: with the
# sagemaker SDK now installed above (when retraining is on), an empty value
# here means the retrieve itself failed (e.g. region/version), which the UI
# would otherwise only reveal as a 503 at "Train from load test" time.
if [[ "${WITH_RETRAINING}" -eq 1 && -z "${XGBOOST_TRAINING_IMAGE_URI}" ]]; then
  warn "Could not resolve the SageMaker built-in XGBoost training image URI${_XGB_ERR:+ (${_XGB_ERR})}; on-demand Yield Optimizer (floor/margin) training will report a 503 in the UI until this resolves. This does NOT block DLRM/NCF training or model registration."
  if [[ "${_XGB_ERR}" == *"No module named"* || "${_XGB_ERR}" == *"sagemaker"* ]]; then
    warn "  The sagemaker SDK is not importable for ${PYTHON} ($(command -v "${PYTHON}")). Install it there (e.g. activate the venv used at deploy time, or '${PYTHON} -m pip install sagemaker') and re-run."  # nosemgrep: unquoted-variable-expansion-in-command the remaining expansion is inside the double-quoted message; semgrep bash parse artifact
  fi
fi
LOADTEST_TABLE="${STACK_NAME}-loadtest-history"
# Container registry table — holds store-defined ARTF containers (name, display
# name, description, intents, endpoint, active flag) so a container can be
# described and switched on or off from the UI without rebuilding the
# orchestrator image or redeploying the stack. Deliberately a BASE-stack table
# created here rather than a partition of the closed-loop stack's
# parameter-store: that stack is optional, and coupling the template container
# to it would leave a base-stack user with no template at all.
CONTAINER_REGISTRY_TABLE="${STACK_NAME}-container-registry"
# Deals table: IAB Tech Lab Deals API v1.0 deals and the demand endpoint's
# campaigns, in its own CloudFormation stack (deployment/deals_table_cfn.yaml),
# seeded from deployment/seed/deals_and_campaigns.json in Step 1.5. Read by the
# in-cluster demand service (deploy_prebid.sh, IRSA), the Deal Scorer pod (Step 7.2 IRSA) and the
# orchestrator (Step 7.5 DealsBook statement, read-only).
DEALS_TABLE="${STACK_NAME}-deals"
DEALS_STACK="${STACK_NAME}-deals"
# Deterministic name matching feedback_pipeline_cfn.yaml's BidOutcomeStream
# naming (HasStackPrefix condition). Only resolves to a real stream once
# deploy_closed_loop.sh Step 1 creates it (--with-retraining, default on) —
# set unconditionally here so the orchestrator picks it up once it exists,
# without a separate redeploy/restart step.
FEEDBACK_STREAM_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}bid-outcome-stream"
# Same pattern, for DealYieldOutcomeStream (deal floor/margin outcomes —
# see source/orchestrator/deal_yield_feedback.py).
DEAL_YIELD_FEEDBACK_STREAM_NAME="${STACK_PREFIX:+${STACK_PREFIX}-}deal-yield-outcome-stream"
# Outcome simulator (source/orchestrator/outcome_simulator.py) — SYNTHETIC
# win/impression/click/conversion outcomes. Off unless the caller sets
# OUTCOME_SIMULATOR_ENABLED=true in the environment before running this script.
# Everything it emits is labelled provenance="simulated" all the way into the model
# manifest, so an enabled deployment stays honest about what its models learned from.
OUTCOME_SIMULATOR_ENABLED="${OUTCOME_SIMULATOR_ENABLED:-false}"
# Funnel rates for that simulator. Empty is meaningful: outcome_simulator._env_float
# reads an empty value as "use the code default" (win 0.40, impression 0.95, click
# 0.02, conversion 0.05, value 25.0), so leaving these unset reproduces the code's
# own behaviour rather than pinning a second copy of the defaults here.
OUTCOME_SIMULATOR_WIN_RATE="${OUTCOME_SIMULATOR_WIN_RATE:-}"
OUTCOME_SIMULATOR_IMPRESSION_RATE="${OUTCOME_SIMULATOR_IMPRESSION_RATE:-}"
OUTCOME_SIMULATOR_CLICK_RATE="${OUTCOME_SIMULATOR_CLICK_RATE:-}"
OUTCOME_SIMULATOR_CONVERSION_RATE="${OUTCOME_SIMULATOR_CONVERSION_RATE:-}"
OUTCOME_SIMULATOR_CONVERSION_VALUE="${OUTCOME_SIMULATOR_CONVERSION_VALUE:-}"

log "Account=${ACCOUNT_ID}  Region=${AWS_REGION}  Stack=${STACK_NAME}  Tag=${IMAGE_TAG}"
log "EKS Cluster=${CLUSTER_NAME}  Model Bucket=${MODEL_BUCKET}"

# =========================================================================
# Destroy
# =========================================================================
if [[ "${DESTROY}" -eq 1 ]]; then
  # Defined before the warning below, which names it. Referenced-then-assigned would
  # print an empty stack name in the one message whose job is to be exact.
  PREBID_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}prebid-artf"
  warn "=== DESTROY ==="
  warn "This will delete EVERYTHING deploy.sh + deploy_closed_loop.sh +"
  warn "deploy_prebid.sh create for this stack (prefix: ${STACK_PREFIX:-<none>}):"
  warn "EKS cluster, Triton models, CloudFront, AgentCore runtimes, Cognito,"
  warn "DynamoDB tables, S3 buckets (including the ones marked DeletionPolicy:"
  warn "Retain in the CFN templates), SageMaker Model Registry, IAM policies/roles,"
  warn "ECR repos, the Prebid ARTF host stack (${PREBID_STACK}) with its Cognito"
  warn "domain, resource servers, M2M client and credential secret, and all"
  warn "Kubernetes resources. NOTHING is retained — this is not reversible."
  warn "Also deleted: the UI API proxy Lambda stack (${UI_API_PROXY_STACK})."
  # Teardown genuinely needs a human, so unlike the resume prompt this does not
  # fall back to a default -- it fails. But it must fail with a reason: an
  # unguarded read here returns empty on EOF (or stops the process with SIGTTIN
  # when backgrounded), and the operator then sees a bare "aborted" with no clue
  # that the problem was the absence of a terminal.
  if ! _can_prompt; then
    fail "Teardown needs an interactive terminal (this process cannot read one -- backgrounded, piped, or no tty). Re-run ./deploy.sh --destroy in the foreground."
  fi
  read -r -p "Type 'destroy' to confirm: " CONFIRM || CONFIRM=""
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
  # PREBID_STACK is set at the top of this block, because the DESTROY warning names it.
  # It is the optional Prebid ARTF host stack from deploy_prebid.sh, and its delete
  # below is guarded on the stack existing -- so --destroy stays true to its "NOTHING
  # is retained" claim whether or not --with-prebid was ever used.

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
  # behind and block subnet deletion (the eksctl-*-cluster stack goes
  # DELETE_FAILED on "subnet has dependencies and cannot be deleted"). Wait
  # for completion so the ENIs are gone before eksctl runs.
  # The UI API proxy Lambda (Part 1) has the same VPC-attached ENIs, so it goes
  # at the same point for the same reason.
  for _vpc_lambda_stack in "${UI_API_PROXY_STACK}" "${VPC_PROXY_STACK}"; do
    if aws cloudformation describe-stacks --stack-name "${_vpc_lambda_stack}" --region "${AWS_REGION}" >/dev/null 2>&1; then
      say "Deleting VPC-attached Lambda stack (${_vpc_lambda_stack}) before the EKS cluster..."
      aws cloudformation delete-stack --stack-name "${_vpc_lambda_stack}" --region "${AWS_REGION}" 2>/dev/null || true
      aws cloudformation wait stack-delete-complete --stack-name "${_vpc_lambda_stack}" --region "${AWS_REGION}" 2>/dev/null || true
    fi
  done

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
  eksctl delete iamserviceaccount --name deal-scorer-sa --namespace default --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" 2>/dev/null || true
  eksctl delete cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --wait 2>/dev/null || true

  say "Deleting Triton model bucket..."
  aws s3 rb "s3://${MODEL_BUCKET}" --force 2>/dev/null || true

  say "Deleting DynamoDB table ${LOADTEST_TABLE}..."
  aws dynamodb delete-table --table-name "${LOADTEST_TABLE}" --region "${AWS_REGION}" 2>/dev/null || true

  say "Deleting CloudFront + S3 frontend..."
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_frontend.py" --action destroy --stack-name "${STACK_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}" || true
  # deploy_frontend.py's destroy only disables the CloudFront distribution (AWS
  # requires Enabled=false to propagate before a distribution can be deleted) —
  # it does not delete the distribution or its S3 bucket. Nothing is retained
  # here, so remove both once the disable has propagated.
  FRONTEND_UID="$("${PYTHON}" -c "import hashlib; print(hashlib.sha256('${STACK_NAME}:${ACCOUNT_ID}:${AWS_REGION}'.encode()).hexdigest()[:8])")"
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
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" --action destroy --runtime-name "${AC_RUNTIME_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}" || true

  say "Deleting Cognito User Pool..."
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_cognito.py" --action destroy --stack-name "${STACK_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}" || true

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
  DEAL_SCORER_POLICY_NAME="${STACK_NAME}-deal-scorer-deals-read-${STACK_UID}"
  DEAL_SCORER_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DEAL_SCORER_POLICY_NAME}"

  # model-optimizer-s3 (Step 7.1's IRSA policy) was missing from this cleanup —
  # every other IRSA/orchestrator policy Step 7/7.5/7.6/9 creates was already here.
  for POLICY_ARN in "${TRITON_POLICY_ARN}" "${OPTIMIZER_POLICY_ARN}" "${DYNAMO_POLICY_ARN}" "${EKS_SCALE_POLICY_ARN}" "${CLOSED_LOOP_POLICY_ARN}" "${DEAL_SCORER_POLICY_ARN}"; do
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
  # iam:DeleteRole fails with DeleteConflict while the role still has INLINE
  # policies, and only attached MANAGED policies were detached above. Step 9
  # puts the agent's permissions on this role inline (BidShadingAgentPermissions),
  # so every prefixed teardown leaked the role: the failure was swallowed by
  # `|| true` and the next deploy then hit EntityAlreadyExists on create-role.
  for INLINE in $(aws iam list-role-policies --role-name "${ROLE_NAME}" --query 'PolicyNames[]' --output text 2>/dev/null); do
    aws iam delete-role-policy --role-name "${ROLE_NAME}" --policy-name "${INLINE}" 2>/dev/null || true
  done
  aws iam delete-role --role-name "${ROLE_NAME}" 2>/dev/null || true

  # --- Part 2 closed-loop resources (if deployed) ---
  # NOTE: `--query '...|[0]' --output text` renders as TWO lines ("None" then
  # the real value, or "None" alone with no match) for this pipe-into-index
  # JMESPath shape — the same quirk documented in deploy.sh's frontend .env
  # generation above. Use --output json + jq -r for a clean single scalar.
  say "Deleting closed-loop AgentCore runtimes..."
  # Runtime-name prefixing (RUNTIME_NAME_PREFIX) was added AFTER the first
  # prefixed stacks were deployed, so a stack can legitimately own a runtime
  # under either the prefixed name ("ads_AdaptiveBiddingStrategyAgent") or the
  # legacy UNPREFIXED one ("AdaptiveBiddingStrategyAgent"). Looking only at the
  # prefixed name printed "not found — nothing to destroy" and silently left
  # those legacy runtimes running after teardown (confirmed live on prefix=ads).
  #
  # The unprefixed name is global, so it CANNOT be deleted on name alone — an
  # unprefixed stack in the same account owns that exact name. Ownership is
  # proven from the runtime's container image, which always carries this
  # stack's (already-prefixed) ECR repo — see deploy_closed_loop.sh's
  # ADAPTIVE_BIDDING_REPO / GOVERNANCE_REPO.
  for RUNTIME_SPEC in \
      "AdaptiveBiddingStrategyAgent:${STACK_NAME}-adaptive-bidding-agent" \
      "ModelPromotionGovernanceAgent:${STACK_NAME}-model-promotion-governance-agent"; do
    BASE_NAME="${RUNTIME_SPEC%%:*}"
    OWNER_REPO="${RUNTIME_SPEC##*:}"
    RUNTIME_NAME="${RUNTIME_NAME_PREFIX}${BASE_NAME}"

    RID="$(aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
      --query "agentRuntimes[?agentRuntimeName=='${RUNTIME_NAME}'].agentRuntimeId | [0]" \
      --output json 2>/dev/null | jq -r '. // empty')"

    # Legacy fallback: only when this run IS prefixed (otherwise RUNTIME_NAME
    # already equals BASE_NAME and there is nothing else to look for).
    if [[ -z "${RID}" && -n "${RUNTIME_NAME_PREFIX}" ]]; then
      LEGACY_RID="$(aws bedrock-agentcore-control list-agent-runtimes --region "${AWS_REGION}" \
        --query "agentRuntimes[?agentRuntimeName=='${BASE_NAME}'].agentRuntimeId | [0]" \
        --output json 2>/dev/null | jq -r '. // empty')"
      if [[ -n "${LEGACY_RID}" ]]; then
        LEGACY_IMAGE="$(aws bedrock-agentcore-control get-agent-runtime \
          --agent-runtime-id "${LEGACY_RID}" --region "${AWS_REGION}" --output json 2>/dev/null \
          | jq -r '.agentRuntimeArtifact.containerConfiguration.containerUri // empty')"
        if [[ "${LEGACY_IMAGE}" == *"/${OWNER_REPO}:"* ]]; then
          RID="${LEGACY_RID}"
          RUNTIME_NAME="${BASE_NAME} (legacy unprefixed, owned by ${OWNER_REPO})"
        else
          warn "  Runtime ${BASE_NAME} exists but its image (${LEGACY_IMAGE:-unknown}) is not from ${OWNER_REPO} — belongs to another stack, leaving it alone"
        fi
      fi
    fi

    if [[ -n "${RID}" ]]; then
      aws bedrock-agentcore-control delete-agent-runtime --agent-runtime-id "${RID}" --region "${AWS_REGION}" 2>/dev/null || true
      say "  Deleted runtime ${RID} (${RUNTIME_NAME})"
    else
      say "  Runtime ${RUNTIME_NAME} not found — nothing to destroy"
    fi
  done

  say "Deleting invocation stack (${GOVERNANCE_EVENTBRIDGE_STACK})..."
  aws cloudformation delete-stack --stack-name "${GOVERNANCE_EVENTBRIDGE_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  # --- Prebid ARTF host stack. Deleted here, BEFORE the Cognito user pool is torn
  # down further below: this stack owns a user pool domain, two resource servers and
  # an app client that all hang off that pool, and deleting the pool first leaves the
  # stack unable to delete its own children.
  #
  # The Kubernetes half needs nothing here — the wholesale
  # `kubectl delete -f "${SCRIPT_DIR}/eks/"` above already covers
  # eks/prebid-server-deployment.yaml. Verified rather than assumed: a client dry-run
  # delete against that file resolves all three objects (ServiceAccount
  # prebid-artf-host-sa, Deployment prebid-server, Service prebid-server) because
  # their names are literal, not placeholder-substituted.
  if aws cloudformation describe-stacks --stack-name "${PREBID_STACK}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    say "Deleting Prebid ARTF host stack (${PREBID_STACK})..."
    aws cloudformation delete-stack --stack-name "${PREBID_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
    aws cloudformation wait stack-delete-complete --stack-name "${PREBID_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  fi

  # --- SageMaker Model Registry: package groups must be emptied before
  # closed-loop-core's stack delete, or it fails DELETE_FAILED with "Model
  # Package Group ... cannot be deleted because it still contains Model
  # Packages". Names mirror closed_loop_cfn.yaml exactly.
  say "Emptying SageMaker Model Package Groups (so closed-loop-core can delete)..."
  # NOTE: the Yield Optimizer has TWO groups (floor, margin), not one. A prior
  # version of this loop listed a single "artf-deal-yield-manager", a name that
  # has never existed -- Triton's FIL backend cannot serve multi-output
  # regression, so the yield model was split into two single-target models with
  # two groups from the start (see closed_loop_cfn.yaml's
  # DealYieldManagerFloorModelPackageGroup/...MarginModelPackageGroup and
  # deploy.sh's own YIELD_FLOOR_MODEL_GROUP/YIELD_MARGIN_MODEL_GROUP). The
  # effect was that neither real group was ever emptied, so closed-loop-core's
  # delete hit the exact DELETE_FAILED this loop exists to prevent.
  for GROUP in "${STACK_PREFIX:+${STACK_PREFIX}-}artf-dlrm-bid-shader" "${STACK_PREFIX:+${STACK_PREFIX}-}artf-ncf-deal-manager" "${STACK_PREFIX:+${STACK_PREFIX}-}artf-deal-yield-manager-floor" "${STACK_PREFIX:+${STACK_PREFIX}-}artf-deal-yield-manager-margin"; do
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
  say "Deleting deals table stack (${DEALS_STACK})..."
  aws cloudformation delete-stack --stack-name "${DEALS_STACK}" --region "${AWS_REGION}" 2>/dev/null || true
  aws cloudformation wait stack-delete-complete --stack-name "${DEALS_STACK}" --region "${AWS_REGION}" 2>/dev/null || true

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
  # CONTAINER_REGISTRY_TABLE (defined above, ${STACK_NAME}-container-registry) was
  # missing from this list, so it outlived a teardown that reports "No resources
  # retained". Its DeletionPolicy is the same as the others' -- the omission was an
  # oversight, not a decision.
  for TABLE in "${STACK_PREFIX:+${STACK_PREFIX}-}parameter-store" "${STACK_PREFIX:+${STACK_PREFIX}-}audit-trail" "${STACK_PREFIX:+${STACK_PREFIX}-}user-features" "${CONTAINER_REGISTRY_TABLE}"; do
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
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_cognito.py" --action destroy \
    --stack-name "${STACK_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}" 2>/dev/null || true

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

  # The local record for this prefix. After a teardown the next run legitimately
  # starts from nothing, so keeping progress or resolved values would only invite
  # a resume onto resources that no longer exist.
  state_clear "${STACK_PREFIX}"
  say "  Cleared local deployment state for prefix '${STACK_PREFIX:-<none>}'"

  say "Destroy complete. No resources retained."
  exit 0
fi

# =========================================================================
# UI-only deploy — re-upload both frontends to S3 + invalidate CloudFront
# =========================================================================
if [[ "${UI_ONLY}" -eq 1 ]]; then
  log "UI-only deploy"

  # The UI reaches the orchestrator through the ui-api-proxy Lambda; its ARN is a
  # build-time input (VITE_UI_API_PROXY_ARN). Read it from the proxy stack. A UI
  # built without it falls back to plain fetch against /api, which nothing serves
  # from CloudFront, so a missing stack is a hard stop here.
  UI_API_PROXY_ARN="$(_cfn_output "${UI_API_PROXY_STACK}" ProxyFunctionArn)"
  [[ -n "${UI_API_PROXY_ARN}" ]] || fail "UI API proxy stack ${UI_API_PROXY_STACK} not found. Run the full deploy (it is created in Phase 3) before --ui-only."

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
VITE_BEDROCK_REGION=${AWS_REGION}
VITE_CAPTION_INFERENCE_PROFILE_ID=${CAPTION_INFERENCE_PROFILE_ID}
VITE_ADAPTIVE_BIDDING_RUNTIME_ARN=${ADAPTIVE_BIDDING_RUNTIME_ARN}
VITE_GOVERNANCE_RUNTIME_ARN=${GOVERNANCE_RUNTIME_ARN}
VITE_UI_API_PROXY_ARN=${UI_API_PROXY_ARN}
EOF
  log "  UI env: adaptive=${ADAPTIVE_BIDDING_RUNTIME_ARN:-<none>}  governance=${GOVERNANCE_RUNTIME_ARN:-<none>}  ui-api-proxy=${UI_API_PROXY_ARN}"

  # Primary distribution: React UI (fresh build embeds the env above)
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
    --action deploy \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}"

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
  ${STACK_NAME}-$(display_name yield-optimizer-floor)
  ${STACK_NAME}-$(display_name yield-optimizer-margin)
  ${STACK_NAME}-$(display_name artf-template)
  ${STACK_NAME}-orchestrator
  ${STACK_NAME}-agentcore
)

# =========================================================================
# Deal Scorer build identity (ncf_deal_manager)
# =========================================================================
# Step 2 trains the Deal Scorer on sample interactions generated from the deal
# book (source/training/ncf_sample/build.py) and writes manifest.json beside its
# ONNX. The manifest's 64-hex `fingerprint` travels with the build as S3 metadata
# source-fingerprint on onnx-source/ncf_deal_manager/model.onnx and as
# ncf_fingerprint in the Phase 1 marker; its `model_version`
# (ncf-neumf-sample-<12 hex>) becomes NCF_MODEL_LABEL on the deal-scorer
# Deployment, which refuses to start on Triton without one. Each build is also
# kept under NCF_BUILDS_PREFIX/<12 hex>/, the key its Model Registry version
# names, because the keys directly under NCF_SOURCE_PREFIX are overwritten by
# every build.
NCF_LABEL_RE='^ncf-neumf-sample-[0-9a-f]{12}$'
NCF_SOURCE_PREFIX="onnx-source/ncf_deal_manager"
NCF_MANIFEST_KEY="${NCF_SOURCE_PREFIX}/manifest.json"
NCF_BUILDS_PREFIX="${NCF_SOURCE_PREFIX}/builds"
PHASE1_MARKER_KEY="deploy-markers/phase1.json"
NCF_FINGERPRINT=""
NCF_MODEL_LABEL=""

# ncf_manifest_field <manifest.json> <field>: print a non-empty string field, or
# exit non-zero with the reason on stderr.
ncf_manifest_field() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import json
import sys

path, field = sys.argv[1], sys.argv[2]
try:
    with open(path) as handle:
        value = json.load(handle).get(field)
except (OSError, ValueError, AttributeError) as exc:
    sys.exit(f"{path}: {exc}")
if not isinstance(value, str) or not value:
    sys.exit(f"{path} has no string field {field!r}")
print(value)
PY
}

# upload_ncf_source <dir>: upload the Deal Scorer build in <dir> (model.onnx,
# manifest.json, interactions.csv) to onnx-source/ncf_deal_manager/builds/<12 hex>/
# and then to onnx-source/ncf_deal_manager/, with source-fingerprint metadata on
# both copies of model.onnx, then set NCF_FINGERPRINT and NCF_MODEL_LABEL from its
# manifest. The Model Optimizer reads the keys under onnx-source/ncf_deal_manager/;
# register_genesis_models.py points the build's registry version at its builds/
# copy, which no later build overwrites.
upload_ncf_source() {
  local dir="$1" fp label f build_prefix
  for f in model.onnx manifest.json interactions.csv; do
    [[ -f "${dir}/${f}" ]] \
      || fail 1 "Step 2 left no ${f} in ${dir}; the Deal Scorer build is incomplete. Re-run Step 2 (source/triton/export_models.py --model ncf)."
  done
  fp="$(ncf_manifest_field "${dir}/manifest.json" fingerprint)" \
    || fail 1 "The Deal Scorer manifest ${dir}/manifest.json has no fingerprint."
  [[ "${fp}" =~ ^[0-9a-f]{64}$ ]] \
    || fail 1 "The Deal Scorer manifest fingerprint '${fp}' is not 64 hex characters."
  label="$(ncf_manifest_field "${dir}/manifest.json" model_version)" \
    || fail 1 "The Deal Scorer manifest ${dir}/manifest.json has no model_version."
  [[ "${label}" =~ ${NCF_LABEL_RE} ]] \
    || fail 1 "The Deal Scorer manifest model_version '${label}' does not match ${NCF_LABEL_RE}."
  build_prefix="${NCF_BUILDS_PREFIX}/${fp:0:12}"
  quiet "s3 cp ncf_deal_manager/model.onnx -> ${build_prefix}/" \
    aws s3 cp "${dir}/model.onnx" "s3://${MODEL_BUCKET}/${build_prefix}/model.onnx" \
      --metadata "source-fingerprint=${fp}" --region "${AWS_REGION}" \
    || fail 1 "Could not upload the Deal Scorer ONNX to s3://${MODEL_BUCKET}/${build_prefix}/"
  for f in manifest.json interactions.csv; do
    quiet "s3 cp ncf_deal_manager/${f} -> ${build_prefix}/" \
      aws s3 cp "${dir}/${f}" "s3://${MODEL_BUCKET}/${build_prefix}/${f}" --region "${AWS_REGION}" \
      || fail 1 "Could not upload the Deal Scorer ${f} to s3://${MODEL_BUCKET}/${build_prefix}/"
  done
  quiet "s3 cp ncf_deal_manager/model.onnx -> onnx-source/ (source-fingerprint ${fp:0:12})" \
    aws s3 cp "${dir}/model.onnx" "s3://${MODEL_BUCKET}/${NCF_SOURCE_PREFIX}/model.onnx" \
      --metadata "source-fingerprint=${fp}" --region "${AWS_REGION}" \
    || fail 1 "Could not upload the Deal Scorer ONNX to s3://${MODEL_BUCKET}/${NCF_SOURCE_PREFIX}/"
  quiet "s3 cp ncf_deal_manager/manifest.json -> onnx-source/" \
    aws s3 cp "${dir}/manifest.json" "s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY}" --region "${AWS_REGION}" \
    || fail 1 "Could not upload the Deal Scorer manifest to s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY}"
  quiet "s3 cp ncf_deal_manager/interactions.csv -> onnx-source/" \
    aws s3 cp "${dir}/interactions.csv" "s3://${MODEL_BUCKET}/${NCF_SOURCE_PREFIX}/interactions.csv" --region "${AWS_REGION}" \
    || fail 1 "Could not upload the Deal Scorer interactions to s3://${MODEL_BUCKET}/${NCF_SOURCE_PREFIX}/"
  NCF_FINGERPRINT="${fp}"
  NCF_MODEL_LABEL="${label}"
  log "  Deal Scorer build ${NCF_MODEL_LABEL} uploaded to s3://${MODEL_BUCKET}/${NCF_SOURCE_PREFIX}/ (kept at ${build_prefix}/)"
}

# write_phase1_marker <deals_seed_sha256> <ncf_fingerprint>: record what Phase 1
# seeded and built, as deploy-markers/phase1.json in the model bucket.
write_phase1_marker() {
  local seed_sha="$1" ncf_fp="$2" tmp
  [[ "${seed_sha}" =~ ^[0-9a-f]{64}$ ]] || fail 1 "Deals seed sha256 '${seed_sha}' is not 64 hex characters."
  [[ "${ncf_fp}" =~ ^[0-9a-f]{64}$ ]] || fail 1 "Deal Scorer fingerprint '${ncf_fp}' is not 64 hex characters."
  tmp="$(mktemp)"
  "${PYTHON}" - "${seed_sha}" "${ncf_fp}" > "${tmp}" <<'PY'
import datetime
import json
import sys

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
marker = {"deals_seed_sha256": sys.argv[1], "ncf_fingerprint": sys.argv[2], "written_at": now}
print(json.dumps(marker, sort_keys=True))
PY
  if ! quiet "s3 cp ${PHASE1_MARKER_KEY}" \
      aws s3 cp "${tmp}" "s3://${MODEL_BUCKET}/${PHASE1_MARKER_KEY}" \
        --content-type application/json --region "${AWS_REGION}"; then
    rm -f "${tmp}"
    fail 1 "Could not write s3://${MODEL_BUCKET}/${PHASE1_MARKER_KEY}"
  fi
  rm -f "${tmp}"
  log "  Phase 1 marker: s3://${MODEL_BUCKET}/${PHASE1_MARKER_KEY} (deals seed ${seed_sha:0:12}, Deal Scorer ${ncf_fp:0:12})"
}

# register_ncf_build: warn-only. Registers the uploaded Deal Scorer build in its
# Model Registry group when retraining is on and the group already exists; on a
# first deploy the group is created, and the build registered, in Phase 5.
register_ncf_build() {
  local group="${STACK_PREFIX:+${STACK_PREFIX}-}artf-ncf-deal-manager"
  [[ "${WITH_RETRAINING}" -eq 1 ]] || return 0
  if ! aws sagemaker describe-model-package-group --model-package-group-name "${group}" \
      --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "  Model Registry group ${group} does not exist yet; Phase 5 registers the Deal Scorer build"
    return 0
  fi
  # nosemgrep: unquoted-variable-expansion-in-command intentional word splitting; the AWS_PROFILE:+ expansion below yields zero or two arguments
  quiet "register_genesis_models.py --only ncf_deal_manager" \
    "${PYTHON}" "${SCRIPT_DIR}/scripts/register_genesis_models.py" \
      --model-bucket "${MODEL_BUCKET}" \
      --region "${AWS_REGION}" \
      ${AWS_PROFILE:+--profile "${AWS_PROFILE}"} \
      --ncf-package-group "${group}" \
      --only ncf_deal_manager \
      --training-image-repository artf-nemo-rl-training \
    || warn "  Deal Scorer Model Registry update failed (non-fatal); Phase 5 runs the registration again."
}

# resolve_ncf_model_label: NCF_MODEL_LABEL as Phase 1 of this run set it, else
# from onnx-source/ncf_deal_manager/manifest.json in the model bucket (a run that
# skips Phase 1). Fails when neither yields a valid label.
resolve_ncf_model_label() {
  local tmp label
  if [[ -n "${NCF_MODEL_LABEL}" ]]; then
    [[ "${NCF_MODEL_LABEL}" =~ ${NCF_LABEL_RE} ]] \
      || fail 3 "NCF_MODEL_LABEL '${NCF_MODEL_LABEL}' does not match ${NCF_LABEL_RE}."
    return 0
  fi
  tmp="$(mktemp)"
  if ! aws s3 cp "s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY}" "${tmp}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    rm -f "${tmp}"
    fail 3 "No Deal Scorer label: Phase 1 did not run in this deploy and s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY} could not be read. Run ./deploy.sh --start-at 1 so Step 2 builds the Deal Scorer."
  fi
  if ! label="$(ncf_manifest_field "${tmp}" model_version)"; then
    rm -f "${tmp}"
    fail 3 "s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY} has no model_version."
  fi
  rm -f "${tmp}"
  [[ "${label}" =~ ${NCF_LABEL_RE} ]] \
    || fail 3 "s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY} model_version '${label}' does not match ${NCF_LABEL_RE}."
  NCF_MODEL_LABEL="${label}"
  log "  Deal Scorer label from s3://${MODEL_BUCKET}/${NCF_MANIFEST_KEY}: ${NCF_MODEL_LABEL}"
}

# =========================================================================
# Staleness the phase probe cannot see (read-only)
# =========================================================================
# The probe reads what EXISTS. It cannot tell whether a running pod's image is
# the one ECR now holds for the deploy's tag (a rebuild under the same tag leaves
# the pod template unchanged), which AgentCore image the MCP runtime was last
# registered with (AgentCore resolves the tag when the runtime is created or
# updated), or whether the Prebid host was deployed from the current sources.
# These functions answer with read-only calls (kubectl get, aws ecr
# describe-images) and the values .deploy-state.json recorded after the last
# successful Step 10 and Step 12. A digest or record that cannot be read counts
# as stale.

# The Deployments rolled out after Step 8b's apply. Each runs one image,
# ${STACK_NAME}-<name> in ECR.
MANAGED_DEPLOYMENTS=(bid-pricer audience-activator deal-scorer signals-enricher
                     yield-optimizer-floor yield-optimizer-margin artf-template orchestrator)
STALE_DEPLOYMENTS=()
STALE_DEPLOYMENT_LINES=()
AGENTCORE_STALE_WHY=""
PREBID_FINGERPRINT=""
PREBID_STEP_WHY=""

# recorded_image_tag: the tag Phase 2 deploys under. .image-outputs.json pins it
# to the last build's tag when that build was for this registry; otherwise the
# tag this run was given.
recorded_image_tag() {
  local f="${SCRIPT_DIR}/.image-outputs.json" tag="" reg=""
  if [[ -f "${f}" ]]; then
    tag="$(jq -r '.ImageTag // empty' "${f}" 2>/dev/null || echo '')"
    reg="$(jq -r '.Registry // empty' "${f}" 2>/dev/null || echo '')"
  fi
  if [[ -n "${tag}" && "${reg}" == "${REGISTRY}" ]]; then
    printf '%s\n' "${tag}"
  else
    printf '%s\n' "${IMAGE_TAG}"
  fi
}

# ecr_image_digest <repo> <tag>: the manifest digest (sha256:<64 hex>) ECR holds
# for <repo>:<tag>, or nothing when there is none or it cannot be read.
ecr_image_digest() {
  local digest
  digest="$(aws ecr describe-images --repository-name "$1" --image-ids "imageTag=$2" \
    --region "${AWS_REGION}" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)"
  if [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    printf '%s\n' "${digest}"
  fi
  return 0
}

# _stale_verdicts <kubectl-json> <stack> <name=digest>...: one line per name,
# "<verdict><TAB><name><TAB><detail>", where verdict is fresh (every live pod of
# the Deployment runs <digest>), stale (one runs another digest), unknown (a digest
# cannot be read on either side) or absent (no such Deployment). Pods are matched
# to Deployments through their ReplicaSet's ownerReferences; terminating, Succeeded
# and Failed pods are not compared. imageID is the repo digest the runtime pulled
# (<registry>/<repo>@sha256:<hex>, with a docker-pullable:// prefix on Docker).
_stale_verdicts() {
  "${PYTHON}" - "$@" <<'PY'
import json
import re
import sys

path, stack, pairs = sys.argv[1], sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def out(verdict, name, detail):
    print(f"{verdict}\t{name}\t{detail}")


def repo_of(image):
    last = (image or "").split("@", 1)[0].rsplit("/", 1)[-1]
    return last.split(":", 1)[0]


def digest_of(image_id):
    tail = (image_id or "").rsplit("@", 1)[-1] if "@" in (image_id or "") else ""
    return tail if DIGEST.fullmatch(tail) else ""


try:
    with open(path) as handle:
        items = json.load(handle).get("items")
    if not isinstance(items, list):
        raise ValueError("no items")
except (OSError, ValueError, AttributeError):
    for name, _ in pairs:
        out("unknown", name, "the cluster's Deployments and pods could not be read")
    sys.exit(0)

def meta(item):
    return item.get("metadata") or {}

deployments = {meta(i).get("name"): i for i in items if i.get("kind") == "Deployment"}
rs_owner = {}
for i in items:
    if i.get("kind") != "ReplicaSet":
        continue
    for ref in meta(i).get("ownerReferences") or []:
        if ref.get("kind") == "Deployment":
            rs_owner[meta(i).get("name")] = ref.get("name")
pods = {}
for i in items:
    if i.get("kind") != "Pod" or meta(i).get("deletionTimestamp"):
        continue
    if (i.get("status") or {}).get("phase") in ("Succeeded", "Failed"):
        continue
    for ref in meta(i).get("ownerReferences") or []:
        if ref.get("kind") == "ReplicaSet" and ref.get("name") in rs_owner:
            pods.setdefault(rs_owner[ref["name"]], []).append(i)

for name, want in pairs:
    repo = f"{stack}-{name}"
    deployment = deployments.get(name)
    if deployment is None:
        out("absent", name, "no such Deployment")
        continue
    if not want:
        out("unknown", name, f"ECR has no readable digest for {repo} at this tag")
        continue
    template = (((deployment.get("spec") or {}).get("template") or {}).get("spec") or {})
    containers = {c.get("name") for c in template.get("containers") or []
                  if repo_of(c.get("image")) == repo}
    live = pods.get(name, [])
    if not containers:
        out("unknown", name, f"the Deployment runs no {repo} container")
        continue
    if not live:
        out("unknown", name, "no live pods to compare")
        continue
    stale, unknown = [], []
    for pod in live:
        statuses = [cs for cs in (pod.get("status") or {}).get("containerStatuses") or []
                    if cs.get("name") in containers]
        if not statuses:
            unknown.append(f"pod {meta(pod).get('name')} reports no container status yet")
        for cs in statuses:
            running = digest_of(cs.get("imageID"))
            if not running:
                unknown.append(f"pod {meta(pod).get('name')} reports no image digest yet")
            elif running != want:
                stale.append(f"pod {meta(pod).get('name')} runs {running[7:19]}")
    if stale:
        out("stale", name, f"ECR holds {want[7:19]}; " + ", ".join(stale))
    elif unknown:
        out("unknown", name, "; ".join(unknown))
    else:
        out("fresh", name, f"{len(live)} pod(s) on {want[7:19]}")
PY
}

# stale_deployments <tag> [kube-context]: set STALE_DEPLOYMENTS to the managed
# Deployments that exist and are stale or unknown by _stale_verdicts against
# ECR's digest for ${STACK_NAME}-<name>:<tag>, and STALE_DEPLOYMENT_LINES to one
# reason per name. Without a context kubectl uses its current one. Sets globals,
# so call it bare, never inside $(...).
stale_deployments() {
  local tag="$1" ctx="${2:-}" tmp name verdicts="" verdict detail
  local kargs=() pairs=()
  STALE_DEPLOYMENTS=()
  STALE_DEPLOYMENT_LINES=()
  if [[ -n "${ctx}" ]]; then kargs=(--context "${ctx}"); fi
  tmp="$(mktemp)"
  kubectl ${kargs[@]+"${kargs[@]}"} get deployments,replicasets,pods -o json >"${tmp}" 2>/dev/null \
    || : >"${tmp}"
  for name in "${MANAGED_DEPLOYMENTS[@]}"; do
    pairs+=("${name}=$(ecr_image_digest "${STACK_NAME}-${name}" "${tag}")")
  done
  verdicts="$(_stale_verdicts "${tmp}" "${STACK_NAME}" "${pairs[@]}" 2>/dev/null)" || verdicts=""
  rm -f "${tmp}"
  if [[ -z "${verdicts}" ]]; then
    for name in "${MANAGED_DEPLOYMENTS[@]}"; do
      verdicts+="unknown"$'\t'"${name}"$'\t'"the digest comparison did not run"$'\n'
    done
  fi
  while IFS=$'\t' read -r verdict name detail; do
    [ -n "${name}" ] || continue
    case "${verdict}" in
      stale|unknown)
        STALE_DEPLOYMENTS+=("${name}")
        STALE_DEPLOYMENT_LINES+=("${name}: ${verdict}, ${detail}") ;;
    esac
  done <<VERDICTS
${verdicts}
VERDICTS
  return 0
}

# restart_stale_deployments <tag>: Step 8b's rollout. Restarts only the managed
# Deployments whose pods do not run ECR's image for <tag> (or whose digest cannot
# be read), against kubectl's current context like the apply before it.
restart_stale_deployments() {
  local tag="$1" line name restarted=0
  stale_deployments "${tag}"
  for line in ${STALE_DEPLOYMENT_LINES[@]+"${STALE_DEPLOYMENT_LINES[@]}"}; do
    log "    ${line}"
  done
  if [[ ${#STALE_DEPLOYMENTS[@]} -eq 0 ]]; then
    log "  Every managed Deployment runs ECR's image for tag ${tag}; none restarted"
    return 0
  fi
  for name in "${STALE_DEPLOYMENTS[@]}"; do
    if kubectl get "deployment/${name}" >/dev/null 2>&1; then
      if kubectl rollout restart "deployment/${name}" >/dev/null 2>&1; then
        restarted=$((restarted + 1))
      else
        warn "  Could not restart deployment/${name}"
      fi
    fi
  done
  log "  Restarted ${restarted} Deployment(s) not on ECR's image for tag ${tag}"
  return 0
}

# agentcore_image_stale <tag>: true when the digest ECR holds for
# ${STACK_NAME}-agentcore:<tag> is not the one recorded after the last successful
# Step 10, or either cannot be read. Sets AGENTCORE_STALE_WHY.
agentcore_image_stale() {
  local tag="$1" recorded current
  AGENTCORE_STALE_WHY=""
  recorded="$(state_read "${STACK_PREFIX}" resolved.agentcoreImageDigest)"
  current="$(ecr_image_digest "${STACK_NAME}-agentcore" "${tag}")"
  if [[ -z "${current}" ]]; then
    AGENTCORE_STALE_WHY="ECR has no readable digest for ${STACK_NAME}-agentcore:${tag}"
  elif [[ -z "${recorded}" ]]; then
    AGENTCORE_STALE_WHY="no AgentCore image digest is recorded for this prefix in ${DEPLOY_STATE_FILE}"
  elif [[ "${recorded}" != "${current}" ]]; then
    AGENTCORE_STALE_WHY="the MCP runtime was registered with AgentCore image ${recorded:7:12}, and ECR now holds ${current:7:12} for ${STACK_NAME}-agentcore:${tag}"
  else
    return 1
  fi
  return 0
}

# record_agentcore_digest <digest>: after Step 10 registers the MCP runtime,
# remember the image digest it was registered with for the next run's check.
record_agentcore_digest() {
  if [[ "$1" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    state_set "${STACK_PREFIX}" resolved "agentcoreImageDigest=$1"
    log "  AgentCore image ${1:7:12} recorded for the next run's Phase 5 check"
  else
    warn "  Could not read the AgentCore image digest from ECR; the next run registers the MCP runtime again."
  fi
  return 0
}

# prebid_fingerprint <repo-root> <deals-seed-sha256>: sha256 over the git-visible
# files deploy_prebid.sh builds or applies (source/prebid/, source/demand/,
# deployment/prebid_cfn.yaml, deployment/deploy_prebid.sh,
# deployment/scripts/prebid_config_template.yaml, its three eks/ manifests,
# source/Dockerfile.demand and lib/k8s_rollout.sh) and the deals seed sha256.
# Fails on a malformed seed sha.
prebid_fingerprint() {
  local root="$1" seed_sha="$2" files="" p f
  local list=()
  [[ "${seed_sha}" =~ ^[0-9a-f]{64}$ ]] || return 1
  for p in source/prebid source/demand; do
    if [[ -d "${root}/${p}" ]]; then files+="$(_hashable_files "${root}/${p}")"$'\n'; fi
  done
  for p in deployment/prebid_cfn.yaml deployment/deploy_prebid.sh \
           deployment/scripts/prebid_config_template.yaml \
           deployment/eks/prebid-server-deployment.yaml deployment/eks/amt-simulator-deployment.yaml \
           deployment/eks/artfhouse-demand-deployment.yaml source/Dockerfile.demand \
           deployment/lib/k8s_rollout.sh; do
    files+="${root}/${p}"$'\n'
  done
  while IFS= read -r f; do
    [ -n "${f}" ] || continue
    list+=("${f}")
  done <<PREBID_FILES
${files}
PREBID_FILES
  "${PYTHON}" - "${root}" "${seed_sha}" ${list[@]+"${list[@]}"} <<'PY'
import hashlib
import os
import sys

root, seed = os.path.realpath(sys.argv[1]), sys.argv[2]
lines = []
for path in sorted({os.path.realpath(p) for p in sys.argv[3:]}):
    rel = os.path.relpath(path, root)
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            lines.append(f"{rel}:{hashlib.sha256(handle.read()).hexdigest()}")
    else:
        lines.append(f"{rel}:absent")
lines.append(f"deals_seed_sha256:{seed}")
print(hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest())
PY
}

# prebid_step_needed <phase5-ran>: whether Step 12 (deploy_prebid.sh) runs. Never
# without --with-prebid; always when Phase 5 ran, as before; otherwise when
# PREBID_FINGERPRINT differs from the one recorded after the last successful
# Step 12, none is recorded, or it could not be computed. Sets PREBID_STEP_WHY.
prebid_step_needed() {
  local ran="$1" recorded
  PREBID_STEP_WHY=""
  if [[ "${WITH_PREBID}" -ne 1 ]]; then return 1; fi
  if [[ "${ran}" -eq 1 ]]; then return 0; fi
  recorded="$(state_read "${STACK_PREFIX}" resolved.prebidFingerprint)"
  if [[ -z "${PREBID_FINGERPRINT}" ]]; then
    PREBID_STEP_WHY="the Prebid host's sources could not be fingerprinted"
  elif [[ -z "${recorded}" ]]; then
    PREBID_STEP_WHY="no Prebid host fingerprint is recorded for this prefix in ${DEPLOY_STATE_FILE}"
  elif [[ "${recorded}" != "${PREBID_FINGERPRINT}" ]]; then
    PREBID_STEP_WHY="the Prebid host's sources or deals seed changed since its last deploy (${recorded:0:12} to ${PREBID_FINGERPRINT:0:12})"
  else
    return 1
  fi
  return 0
}

# check_unprobed_staleness: Phase 3 and Phase 5 staleness, checked once before
# the phases run; each finding becomes force_phase 3 or 5. Phase 1 and the Deal
# Scorer engine are the probe's (see compute_phase_expectations). Sets globals,
# so call it bare.
check_unprobed_staleness() {
  local tag line
  tag="$(recorded_image_tag)"
  if [[ "${START_AT}" -le 3 ]]; then
    # The probe's context for THIS cluster, never kubectl's current one: before
    # Step 5.5 that may belong to another deployment.
    if [[ -n "${DEPLOY_KUBE_CONTEXT:-}" ]]; then
      stale_deployments "${tag}" "${DEPLOY_KUBE_CONTEXT}"
      if [[ ${#STALE_DEPLOYMENTS[@]} -gt 0 ]]; then
        force_phase 3 "${#STALE_DEPLOYMENTS[@]} Deployment(s) do not run ECR's image for tag ${tag} (${STALE_DEPLOYMENTS[*]})"
        for line in "${STALE_DEPLOYMENT_LINES[@]}"; do log "    ${line}"; done
      fi
    else
      force_phase 3 "the running pods' image digests could not be read (no kubeconfig context for ${CLUSTER_NAME})"
    fi
  fi
  if [[ "${SKIP_AGENTCORE}" -eq 0 ]] && agentcore_image_stale "${tag}"; then
    force_phase 5 "${AGENTCORE_STALE_WHY}"
  fi
  return 0
}

if [[ "${EXPORT_ONLY}" -eq 0 ]]; then
  check_unprobed_staleness
fi

if _run_phase 1; then
phase 1 "Preparing models"
step "Step 1: Ensuring ECR repositories"
if [[ "${LOCAL_BUILD}" -eq 1 ]]; then
  # --local-build means every image is built here, so the daemon has to be up. Start
  # it rather than failing at the first `docker login` with a connect error.
  # shellcheck source=lib/deploy_docker.sh
  source "${SCRIPT_DIR}/lib/deploy_docker.sh"
  docker_ensure_running "build the container images locally (--local-build)"
  docker_ensure_buildx
  quiet "ECR login (${REGISTRY})" _ecr_docker_login
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
step "Step 1.5: Ensuring DynamoDB table for load test history"
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

# Container registry table. The partition key is a CONSTANT ("artf-containers")
# and the sort key is the container name, so one Query returns every record and
# no Scan is ever needed -- Scan is not granted to the orchestrator's node role
# and this keeps it that way. SSE with the AWS-owned key rather than the
# closed-loop stack's CMK, because that key belongs to an optional stack.
if ! aws dynamodb describe-table --table-name "${CONTAINER_REGISTRY_TABLE}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  aws dynamodb create-table \
    --table-name "${CONTAINER_REGISTRY_TABLE}" \
    --attribute-definitions AttributeName=registry,AttributeType=S AttributeName=name,AttributeType=S \
    --key-schema AttributeName=registry,KeyType=HASH AttributeName=name,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --sse-specification Enabled=true \
    --region "${AWS_REGION}" >/dev/null
  log "  Created DynamoDB table: ${CONTAINER_REGISTRY_TABLE}"
  aws dynamodb wait table-exists --table-name "${CONTAINER_REGISTRY_TABLE}" --region "${AWS_REGION}" 2>/dev/null || true
else
  log "  DynamoDB table exists: ${CONTAINER_REGISTRY_TABLE}"
fi

# Seed the template container's record, guarded by attribute_not_exists so a
# redeploy NEVER overwrites a display name, description, intent list or -- most
# importantly -- an activation choice the user made in the UI. Ships inactive:
# the container is deployed and reachable but out of the flow until switched on.
# ConditionalCheckFailedException here means "already seeded", which is the
# expected outcome on every deploy after the first, so it is not an error.
if aws dynamodb put-item \
  --table-name "${CONTAINER_REGISTRY_TABLE}" \
  --region "${AWS_REGION}" \
  --condition-expression "attribute_not_exists(#n)" \
  --expression-attribute-names '{"#n":"name"}' \
  --item '{
    "registry":     {"S": "artf-containers"},
    "name":         {"S": "artf-template"},
    "display_name": {"S": "ARTF Template"},
    "description":  {"S": "Template container. Deployed and wired but not implemented — returns no mutations. Edit source/containers/artf_template/app.py, rebuild that one image, restart that one Deployment, then activate here."},
    "intents":      {"L": [{"S": "ADD_CIDS"}]},
    "endpoint":     {"S": "http://artf-template:8081"},
    "active":       {"BOOL": false},
    "updated_at":   {"S": "'"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"'"},
    "updated_by":   {"S": "deploy.sh"}
  }' >/dev/null 2>&1; then
  log "  Seeded registry record: artf-template (inactive)"
else
  log "  Registry record artf-template already present — left untouched"
fi
# Deals table stack and seed. The table is a CloudFormation resource (on-demand,
# KMS encryption at rest, point-in-time recovery); the seed is written on every
# deploy through the stack's put-only DealsSeedRole, so a seed edit reaches the
# table here and the Deal Scorer picks it up on its next 30 second refresh.
DEALS_STATUS="$(aws cloudformation describe-stacks --stack-name "${DEALS_STACK}" --region "${AWS_REGION}" \
  --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo '')"
if [[ "${DEALS_STATUS}" == "ROLLBACK_COMPLETE" ]]; then
  log "  ${DEALS_STACK} is ${DEALS_STATUS}; deleting before recreate"
  aws cloudformation delete-stack --stack-name "${DEALS_STACK}" --region "${AWS_REGION}"
  aws cloudformation wait stack-delete-complete --stack-name "${DEALS_STACK}" --region "${AWS_REGION}"
  DEALS_STATUS=""
fi
DEALS_ACTION="create-stack"
[[ -n "${DEALS_STATUS}" && "${DEALS_STATUS}" != "None" ]] && DEALS_ACTION="update-stack"
DEALS_CFN_LOG="/tmp/cfn-deploy-${DEALS_STACK}-$(date +%s).log"
if aws cloudformation "${DEALS_ACTION}" --stack-name "${DEALS_STACK}" \
    --template-body "file://${SCRIPT_DIR}/deals_table_cfn.yaml" \
    --capabilities CAPABILITY_IAM --region "${AWS_REGION}" \
    --parameters "ParameterKey=TableName,ParameterValue=${DEALS_TABLE}" \
    >"${DEALS_CFN_LOG}" 2>&1; then
  DEALS_WAIT="stack-create-complete"
  [[ "${DEALS_ACTION}" == "update-stack" ]] && DEALS_WAIT="stack-update-complete"
  aws cloudformation wait "${DEALS_WAIT}" --stack-name "${DEALS_STACK}" --region "${AWS_REGION}" \
    || fail 1 "Deals table stack ${DEALS_STACK} did not reach ${DEALS_WAIT}; see: aws cloudformation describe-stack-events --stack-name ${DEALS_STACK}"
elif ! grep -q "No updates are to be performed" "${DEALS_CFN_LOG}"; then
  fail 1 "Deals table stack ${DEALS_STACK} ${DEALS_ACTION} failed: $(cat "${DEALS_CFN_LOG}")"
fi
DEALS_SEED_ROLE_ARN="$(_cfn_output "${DEALS_STACK}" DealsSeedRoleArn)"
[[ -n "${DEALS_SEED_ROLE_ARN}" ]] || fail 1 "${DEALS_STACK} has no DealsSeedRoleArn output."
log "  Deals table stack ready: ${DEALS_STACK} (table ${DEALS_TABLE})"
"${PYTHON}" "${SCRIPT_DIR}/scripts/seed_deals_table.py" \
  --table "${DEALS_TABLE}" \
  --role-arn "${DEALS_SEED_ROLE_ARN}" \
  --seed "${SCRIPT_DIR}/seed/deals_and_campaigns.json" \
  --region "${AWS_REGION}" \
  --origin-prefix "${STACK_PREFIX}" \
  ${AWS_PROFILE:+--profile "${AWS_PROFILE}"} \
  || fail 1 "seed_deals_table.py could not seed ${DEALS_TABLE} from deployment/seed/deals_and_campaigns.json (see its error above)."

# =========================================================================
# Step 2: Export PyTorch models to ONNX
# =========================================================================
step "Step 2: Exporting PyTorch models to ONNX (source for the Model Optimizer)"
# Part 2: Triton serves TensorRT engines (tensorrt_plan). The exported ONNX is the
# SOURCE the Model Optimizer compiles into engines — it is uploaded to onnx-source/
# (Step 3a), NOT served directly. Export to a staging dir so it does not collide
# with the router/engine configs committed under triton/model_repository/.
ONNX_STAGING="${SCRIPT_DIR}/../source/triton/onnx_export"
rm -rf "${ONNX_STAGING}"
run_step "Exporting DLRM to ONNX; training, exporting and gating the Deal Scorer (NCF)" \
  "${PYTHON}" "${SCRIPT_DIR}/../source/triton/export_models.py" \
    --output-dir "${ONNX_STAGING}"

# Step 2.5: Genesis XGBoost export for the Yield Optimizer (floor/margin).
# Best-effort, non-blocking (per explicit user instruction: train the
# genesis models, but never make deployment depend on it completing).
# xgboost/onnxmltools are intentionally NOT added to REQUIRED_PY_PACKAGES
# above (that check uses fail(), which would block the deploy) -- checked
# and installed independently here, with a warn()-only fallback. If genesis
# export doesn't succeed,
# register_genesis_models.py (Step 3, deploy_closed_loop.sh) skips that
# model type (its own existing "missing artifact" path) and
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
if "${PYTHON}" -c "
import sys
try:
    import onnxmltools, xgboost
except ImportError:
    sys.exit(1)
sys.exit(0 if xgboost.__version__ == '1.7.6' else 1)
" 2>/dev/null; then
  XGBOOST_OK=1
fi

step "Step 2.5: Exporting genesis XGBoost models (Yield Optimizer floor/margin)"
if [[ "${XGBOOST_OK}" -ne 1 ]]; then
  log "  Installing ${XGBOOST_PIN}/onnxmltools (best-effort, non-blocking)..."
  quiet "pip install ${XGBOOST_PIN} onnxmltools" "${PYTHON}" -m pip install --quiet "${XGBOOST_PIN}" onnxmltools || true
  if "${PYTHON}" -c "
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
  run_step "Exporting genesis XGBoost floor/margin models" \
    "${PYTHON}" "${SCRIPT_DIR}/../source/training/export_xgboost_genesis.py" \
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
step "Step 3: Ensuring S3 model bucket ${MODEL_BUCKET}"
if ! aws s3api head-bucket --bucket "${MODEL_BUCKET}" 2>/dev/null; then
  quiet "s3 mb ${MODEL_BUCKET}" aws s3 mb "s3://${MODEL_BUCKET}" --region "${AWS_REGION}"
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
  # The Deal Scorer build also carries its training interactions and its
  # fingerprint (upload_ncf_source above).
  if [[ "${m}" == "ncf_deal_manager" ]]; then
    upload_ncf_source "${ONNX_STAGING}/${m}/1"
    continue
  fi
  quiet "s3 cp ${m}/model.onnx -> onnx-source/" \
    aws s3 cp "${ONNX_STAGING}/${m}/1/model.onnx" \
      "s3://${MODEL_BUCKET}/onnx-source/${m}/model.onnx" --region "${AWS_REGION}"
  # manifest.json goes NEXT TO the ONNX, which is where the optimizer looks for it.
  # Only the exporters that write one produce this file; a model without one is
  # uploaded without one, and the optimizer then refuses it if its bootstrap entry
  # declares an expected version.
  if [[ -f "${ONNX_STAGING}/${m}/1/manifest.json" ]]; then
    quiet "s3 cp ${m}/manifest.json -> onnx-source/" \
      aws s3 cp "${ONNX_STAGING}/${m}/1/manifest.json" \
        "s3://${MODEL_BUCKET}/onnx-source/${m}/manifest.json" --region "${AWS_REGION}"
  fi
done
log "  ONNX uploaded to s3://${MODEL_BUCKET}/onnx-source/"
# Phase 5 gate-skips on a healthy stack, so a new Deal Scorer build is registered
# here as well (the script registers a version only for a new fingerprint).
register_ncf_build

# 3a2. Yield Optimizer genesis: ONNX form for registry bookkeeping
# (onnx-source/, matches the DLRM/NCF convention exactly so
# register_genesis_models.py needs no format-specific branching) AND the
# native XGBoost JSON form (triton-models/<model>/1/xgboost.json) --
# Triton's FIL backend does NOT read ONNX, only the native format. Skipped
# per-model if Step 2.5 did not produce that model's artifacts.
for m in "${YIELD_MODELS[@]}"; do
  if [[ -f "${ONNX_STAGING}/${m}/1/model.onnx" && -f "${ONNX_STAGING}/${m}/1/xgboost.json" ]]; then
    quiet "s3 cp ${m}/model.onnx -> onnx-source/" \
      aws s3 cp "${ONNX_STAGING}/${m}/1/model.onnx" \
        "s3://${MODEL_BUCKET}/onnx-source/${m}/model.onnx" --region "${AWS_REGION}"
    quiet "s3 cp ${m}/xgboost.json -> triton-models/" \
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
quiet "s3 sync served Triton repo -> triton-models/" \
  aws s3 sync "${SERVED_STAGING}/" "s3://${MODEL_BUCKET}/triton-models/" \
    --exclude "*.onnx" --region "${AWS_REGION}"
rm -rf "${SERVED_STAGING}"
log "  Triton served repo (routers + engine configs) uploaded; engines built at Step 8b"

# 3c. Upload the Model Optimizer bootstrap spec (base-engine build instructions).
#
# __FEATURE_SPEC_VERSION__ is read from source/shared/dlrm_features.py rather than
# written into the JSON, so the version this deployment expects has one definition.
# The optimizer refuses to compile a DLRM artifact whose manifest.json declares a
# different version, or none at all -- an ONNX graph with the right input names and
# widths is not evidence that the producer and the serving container agree on what
# each position means.
FEATURE_SPEC_VERSION="$("${PYTHON}" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}/../source')
from shared import dlrm_features
print(dlrm_features.FEATURE_SPEC_VERSION)
")" || fail "Could not read FEATURE_SPEC_VERSION from source/shared/dlrm_features.py"
log "  DLRM feature spec version: ${FEATURE_SPEC_VERSION}"
BOOTSTRAP_TMP="$(mktemp)"
sed -e "s|__MODEL_BUCKET__|${MODEL_BUCKET}|g" \
    -e "s|__FEATURE_SPEC_VERSION__|${FEATURE_SPEC_VERSION}|g" \
  "${SCRIPT_DIR}/optimizer-bootstrap.json" > "${BOOTSTRAP_TMP}"
grep -q '__FEATURE_SPEC_VERSION__' "${BOOTSTRAP_TMP}" \
  && fail "optimizer-bootstrap.json still has an unsubstituted __FEATURE_SPEC_VERSION__"
quiet "s3 cp optimizer bootstrap spec" \
  aws s3 cp "${BOOTSTRAP_TMP}" \
    "s3://${MODEL_BUCKET}/optimizer-bootstrap/spec.json" --region "${AWS_REGION}"
rm -f "${BOOTSTRAP_TMP}"
log "  Model Optimizer bootstrap spec uploaded"

# 3d. Phase 1 marker: the deals seed and the Deal Scorer build this phase put in
#     place. Written last, so a phase that stops part way leaves the previous
#     marker (or none) behind.
write_phase1_marker \
  "$(file_sha256 "${SCRIPT_DIR}/seed/deals_and_campaigns.json")" \
  "${NCF_FINGERPRINT}"
ok "Models exported and uploaded"
fi # START_AT <= 1 (Phase 1)

# =========================================================================
# Step 4: Build and push container images
# =========================================================================
if _run_phase 2; then
phase 2 "Building containers & provisioning infrastructure"
log "  This typically takes 15-20 minutes (bounded by EKS cluster creation)."
fi
_PHASE2_START=$(date +%s)
# eksctl create cluster and the Phase-3 kubectl apply loop write to the run's detail
# log like every other step. They used to have per-run files under /tmp; one log per
# run, rotated by _deploy_log_init, is easier to find and to hand to someone.
_EKSCTL_LOG="${DEPLOY_LOG}"
_KUBECTL_LOG="${DEPLOY_LOG}"
IMAGE_OUTPUTS="${SCRIPT_DIR}/.image-outputs.json"

# Always read the outputs file if it exists (needed for --start-at to use the correct
# tag). recorded_image_tag reads ${IMAGE_OUTPUTS}; the staleness checks before Phase 1
# use the same function, so they compare against the tag deployed here.
IMAGE_TAG="$(recorded_image_tag)"

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

# List the files under directory $1 that count as image source, one absolute
# path per line, sorted. Uses git so gitignored artifacts (__pycache__/*.pyc
# from local pytest runs, .pytest_cache, .hypothesis, .DS_Store, node_modules)
# do not change the hash: a plain `find` included them, so running the tests
# changed every Python image's hash and the next deploy rebuilt all of them.
# remote_build.sh already drops __pycache__ and .DS_Store from the CodeBuild
# zip, so those files were never in the built image either.
#
# -c lists tracked files and -o --exclude-standard adds untracked files that are
# not ignored. Contents are read from the working tree, so uncommitted edits
# and new uncommitted files still change the hash.
#
# Outside a git checkout (e.g. a source tarball) it falls back to find with the
# same artifacts excluded explicitly.
_hashable_files() {
  local dir="$1" root
  root="$(git -C "${dir}" rev-parse --show-toplevel 2>/dev/null || echo '')"
  if [[ -n "${root}" ]]; then
    git -C "${root}" -c core.quotePath=false ls-files -co --exclude-standard -- "${dir}" 2>/dev/null \
      | while IFS= read -r f; do [ -n "${f}" ] && printf '%s/%s\n' "${root}" "${f}"; done \
      | LC_ALL=C sort
  else
    find "${dir}" \( -name __pycache__ -o -name .pytest_cache -o -name .hypothesis \
        -o -name .mypy_cache -o -name .ruff_cache -o -name node_modules -o -name dist \) -prune \
      -o -type f ! -name '*.pyc' ! -name .DS_Store -print | LC_ALL=C sort
  fi
}

# Compute a short content hash for the given image key over exactly the files
# that Dockerfile actually COPYs (kept in sync with build_image_local's cases
# and the Dockerfiles under source/). Returns empty on an unrecognized key so
# callers can fail safe (rebuild) instead of silently trusting a stale image.
_source_hash() {
  local key="$1"
  # Physical path, so it matches the git toplevel prefix _hashable_files emits
  # and ${f#${src}/} below strips it.
  local src
  src="$(cd "${SCRIPT_DIR}/../source" 2>/dev/null && pwd -P || echo "${SCRIPT_DIR}/../source")"
  local paths=()
  case "${key}" in
    dlrm-bid-shader|ncf-deal-manager|yield-optimizer-floor|yield-optimizer-margin)
      paths=("${src}/triton/Dockerfile.triton-artf" "${src}/shared" "${src}/containers/${key//-/_}") ;;
    widedeep-segment-activator|metrics-enricher|artf-template)
      paths=("${src}/Dockerfile" "${src}/shared" "${src}/containers/${key//-/_}") ;;
    orchestrator)
      # The three deployment/ files are listed individually because
      # Dockerfile.orchestrator COPYs exactly those, not the whole directory
      # (governance_api imports TritonModelLoader from deployment.model_deployer).
      # They were missing here, so editing one changed the image without changing
      # the hash — the gate would report "content unchanged" and reuse a stale
      # image, which is the exact failure it exists to prevent.
      paths=("${src}/Dockerfile.orchestrator" "${src}/shared" "${src}/agents" \
             "${src}/closed_loop_demo" "${src}/orchestrator" \
             "${src}/deployment/model_deployer.py" "${src}/deployment/canary_deployer.py" \
             "${src}/deployment/__init__.py") ;;
    agentcore)
      paths=("${src}/Dockerfile.agentcore" "${src}/shared" "${src}/containers" "${src}/agentcore") ;;
    model-optimizer)
      paths=("${src}/Dockerfile.optimizer" "${src}/optimizer") ;;
    *)
      return 1 ;;
  esac
  local f rel manifest="" _found=""
  for p in "${paths[@]}"; do
    if [[ -f "${p}" ]]; then
      rel="${p#${src}/}"
      manifest+="${rel}:$(_sha256 < "${p}" | awk '{print $1}')"$'\n'
    elif [[ -d "${p}" ]]; then
      # Collected into a variable and fed through a heredoc rather than
      # `done < <(find ...)`. Process substitution is a bash extension that
      # /bin/sh -- bash in POSIX mode -- rejects at PARSE time, so its presence
      # made the ENTIRE script unparseable when invoked as `sh deploy.sh`: it
      # failed with a syntax error on this line before any phase ran.
      #
      # A pipe into the loop is not an alternative: it would run the loop in a
      # subshell and lose `manifest`, which is why process substitution was used.
      # The `[ -n ]` guard matters because a heredoc over an empty `find` result
      # still yields one empty line, which would otherwise add a bogus entry.
      #
      # Verified output-identical to the previous form over nine cases, including
      # every directory this function is called with, a missing path, a single
      # file, a mixed set, and an empty directory.
      _found="$(_hashable_files "${p}")"
      while IFS= read -r f; do
        [ -n "${f}" ] || continue
        # ls-files still lists a tracked file deleted in the working tree.
        [ -f "${f}" ] || continue
        rel="${f#${src}/}"
        manifest+="${rel}:$(_sha256 < "${f}" | awk '{print $1}')"$'\n'
      done <<MANIFEST_FILES
${_found}
MANIFEST_FILES
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
    dlrm-bid-shader|ncf-deal-manager|yield-optimizer-floor|yield-optimizer-margin)
      run_step "Building ${repo} (amd64, tritonclient)" \
        docker buildx build --platform linux/amd64 --build-arg CONTAINER="containers/${key//-/_}" \
          -f "${src}/triton/Dockerfile.triton-artf" -t "${image}" --load "${src}"
      quiet "docker push ${repo}" docker push "${image}" ;;
    widedeep-segment-activator|metrics-enricher|artf-template)
      # No Triton dependency, so the plain ARTF Dockerfile rather than
      # Dockerfile.triton-artf. Segment activation was switched from the Wide &
      # Deep Triton model to deterministic rules (see
      # source/containers/widedeep_segment_activator/app.py); metrics enrichment
      # was always rules; and artf-template ships as a pass-through with no
      # model at all. A user who adds a Triton model to the template should move
      # its key to the tritonclient case above.
      run_step "Building ${repo} (amd64)" \
        docker buildx build --platform linux/amd64 \
          --build-arg CONTAINER="containers/${key//-/_}" --build-arg AGENT_NAME="${key}" \
          -f "${src}/Dockerfile" -t "${image}" --load "${src}"
      quiet "docker push ${repo}" docker push "${image}" ;;
    orchestrator)
      run_step "Building ${repo} (amd64)" \
        docker buildx build --platform linux/amd64 --build-arg AGENT_NAME="artf-orchestrator" \
          -f "${src}/Dockerfile.orchestrator" -t "${image}" --load "${src}"
      quiet "docker push ${repo}" docker push "${image}" ;;
    agentcore)
      run_step "Building ${repo} (arm64)" \
        docker buildx build --platform linux/arm64 \
          -f "${src}/Dockerfile.agentcore" -t "${image}" --load "${src}"
      quiet "docker push ${repo}" docker push "${image}" ;;
    model-optimizer)
      run_step "Building ${repo} (amd64)" \
        docker buildx build --platform linux/amd64 \
          -f "${src}/Dockerfile.optimizer" -t "${image}" --load "${src}"
      quiet "docker push ${repo}" docker push "${image}" ;;
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
  STEP4_KEYS=(dlrm-bid-shader widedeep-segment-activator ncf-deal-manager metrics-enricher yield-optimizer-floor yield-optimizer-margin artf-template orchestrator model-optimizer)
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

  if [[ ${#MISSING_KEYS[@]} -gt 0 ]]; then
    # Read by _run_phase: a rebuilt image must be rolled out even when the probe
    # says the phase that rolls it is already complete.
    _IMAGES_REBUILT=1
    for key in "${MISSING_KEYS[@]}"; do
      [[ "${key}" == "agentcore" ]] && _AGENTCORE_IMAGE_REBUILT=1
    done
  fi
  if [[ ${#MISSING_KEYS[@]} -eq 0 ]]; then
    step "Step 4: All required images content-matched. Nothing to build."
  elif [[ "${LOCAL_BUILD}" -eq 0 ]]; then
    step "Step 4: Building changed images via CodeBuild (tag=${IMAGE_TAG}): ${MISSING_KEYS[*]}"
    NGC_FLAG=()
    if [[ -n "${NGC_KEY}" ]]; then NGC_FLAG=(--ngc-key "${NGC_KEY}")
    elif [[ -n "${NGC_SECRET}" ]]; then NGC_FLAG=(--ngc-secret "${NGC_SECRET}"); fi
    # CodeBuild adds the src-<hash> tags itself once every image is pushed, so a
    # deploy.sh killed while waiting below still leaves the cache populated and
    # the next run reuses those images. The client-side tagging loop after the
    # build stays as the fallback and is a no-op when the tags already exist.
    _SRC_TAG_PAIRS=""
    for i in "${!HASH_KEYS[@]}"; do
      _SRC_TAG_PAIRS+="${STACK_NAME}-$(display_name "${HASH_KEYS[$i]}")=${HASH_VALS[$i]} "
    done
    SRC_TAGS_FLAG=()
    if [[ -n "${_SRC_TAG_PAIRS}" ]]; then SRC_TAGS_FLAG=(--src-tags "${_SRC_TAG_PAIRS% }"); fi
    # remote_build.sh narrates on stderr and polls CodeBuild every 15 s; all of that is
    # in the detail log. The terminal shows this one label until the build finishes.
    run_step "CodeBuild: ${#MISSING_KEYS[@]} image(s), tag ${IMAGE_TAG}" \
      bash "${SCRIPT_DIR}/codebuild/remote_build.sh" \
        --stack-name "${STACK_NAME}" \
        --only "${MISSING_KEYS[*]}" \
        --tag "${IMAGE_TAG}" \
        --region "${AWS_REGION}" \
        --profile "${AWS_PROFILE}" \
        ${SRC_TAGS_FLAG[@]+"${SRC_TAGS_FLAG[@]}"} \
        ${NGC_FLAG[@]+"${NGC_FLAG[@]}"}
        # ^ bash 3.2 (macOS) treats an EMPTY array as unbound under `set -u`, so a
        #   plain "${NGC_FLAG[@]}" aborts the run when no NGC flag was given.
  else
    step "Step 4: Building changed images locally (tag=${IMAGE_TAG}): ${MISSING_KEYS[*]}"
    quiet "ECR login (${REGISTRY})" _ecr_docker_login
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
# update_managed_policy ARN NAME DOC: make DOC the default version of an existing
# customer-managed policy. IAM caps a managed policy at five versions and refuses
# CreatePolicyVersion beyond that, so while five exist the oldest non-default
# version is deleted first. A failed prune or update stops the deploy: a swallowed
# failure leaves the previous default in force while the step reports success,
# which is how the closed-loop policy lost its yield training ARNs and how the
# Step 7.5 DealsBook statement could be dropped.
update_managed_policy() {
  local arn="$1" name="$2" doc="$3" count oldest err
  while :; do
    count="$(aws iam list-policy-versions --policy-arn "${arn}" --query 'length(Versions)' --output text 2>/dev/null || echo 0)"
    [[ "${count}" =~ ^[0-9]+$ && "${count}" -ge 5 ]] || break
    oldest="$(aws iam list-policy-versions --policy-arn "${arn}" \
      --query 'sort_by(Versions[?IsDefaultVersion==`false`], &CreateDate)[0].VersionId' --output text 2>/dev/null || echo '')"
    if [[ -z "${oldest}" || "${oldest}" == "None" ]]; then
      fail "${name} has ${count} versions and no non-default version to prune"
    fi
    log "  Pruning old ${name} version ${oldest} (IAM 5-version limit)"
    aws iam delete-policy-version --policy-arn "${arn}" --version-id "${oldest}" >/dev/null \
      || fail "could not prune old version ${oldest} of ${name}"
  done
  if ! err="$(aws iam create-policy-version --policy-arn "${arn}" \
      --policy-document "${doc}" --set-as-default 2>&1 >/dev/null)"; then
    fail "could not update ${name}: ${err}"
  fi
  return 0
}
create_eks_cluster() {
  log "  GPU node group: desired 1, max ${MAX_GPUS} (set via --maxGPUs)"
  local CLUSTER_CONFIG="/tmp/${CLUSTER_NAME}-config.yaml"
  sed -e "s/__STACK_NAME__/${STACK_NAME}/g" \
      -e "s/__REGION__/${AWS_REGION}/g" \
      -e "s/__MAX_GPUS__/${MAX_GPUS}/g" \
      "${SCRIPT_DIR}/eks/cluster-config.yaml" > "${CLUSTER_CONFIG}"
  # eksctl streams one line per CloudFormation waiter for 15-20 minutes, and it
  # runs CONCURRENTLY with the image build, so the two interleave. Captured to a
  # log file by default; the heartbeat in the orchestration block below is what
  # keeps a working deploy from looking hung.
  #
  # Under --verbose it is NOT redirected: verbose exists for someone who wants the
  # underlying stream, and a log file is a worse answer for them than the stream.
  if [[ "${VERBOSE}" -eq 1 ]]; then
    eksctl create cluster -f "${CLUSTER_CONFIG}"
  else
    _log_section "eksctl create cluster ${CLUSTER_NAME}"
    eksctl create cluster -f "${CLUSTER_CONFIG}" >>"${_EKSCTL_LOG}" 2>&1
  fi
}

# Factored into a function (was previously inline) so it can run concurrently
# with build_images() above (FR-8) — cluster creation has no dependency on the
# images existing. Any internal `fail` call here runs in the backgrounded
# subshell (see the orchestration block below) and only exits that subshell;
# the caller checks the real exit code via `wait`.
ensure_eks_cluster() {
  if eksctl get cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    step "Step 5: EKS cluster ${CLUSTER_NAME} already exists"
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
    step "Step 5: Creating EKS cluster ${CLUSTER_NAME} (15-20 min)"
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
        quiet "eksctl delete cluster ${CLUSTER_NAME} (orphaned stack)" \
          eksctl delete cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --wait \
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
  # Heartbeat while the cluster is created. eksctl's own stream is now captured to
  # a log file (see create_eks_cluster), and it was the only sign of life during a
  # 15-20 minute wait -- without this a working deploy reads as hung.
  _heartbeat_pid=""
  if [[ "${VERBOSE}" -eq 0 ]]; then
    (
      while kill -0 "${_cluster_pid}" 2>/dev/null; do
        sleep 60
        kill -0 "${_cluster_pid}" 2>/dev/null || break
        printf '  … still creating the EKS cluster (%dm elapsed; eksctl output in %s)\n' \
          "$(( ( $(date +%s) - _PHASE2_START ) / 60 ))" "${_EKSCTL_LOG}"
      done
    ) &
    _heartbeat_pid=$!
  fi
  build_images
  if ! wait "${_cluster_pid}"; then
    stop_bg "${_heartbeat_pid}"
    fail 2 "EKS cluster creation failed. eksctl output is in ${_EKSCTL_LOG}; the resource-level reason is in CloudFormation: aws cloudformation describe-stack-events --stack-name <the eksctl-${CLUSTER_NAME}-* stack that failed> --region ${AWS_REGION} --query \"StackEvents[?ResourceStatus=='CREATE_FAILED'].[LogicalResourceId,ResourceStatusReason]\""
  fi
  stop_bg "${_heartbeat_pid}"
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
step "Step 5.5: Disabling termination protection on eksctl-managed stacks"
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

# --profile is written into the kubeconfig's exec block, so a kubectl run later from
# a shell with a different AWS_PROFILE still authenticates as this deployment did.
quiet "eks update-kubeconfig ${CLUSTER_NAME}" \
  aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --profile "${AWS_PROFILE}"


# =========================================================================
# Step 6: Install NVIDIA Kubernetes Device Plugin + Prometheus Operator CRDs
# =========================================================================
step "Step 6: Ensuring NVIDIA Kubernetes Device Plugin"
quiet "kubectl apply nvidia-device-plugin" \
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml || true

log "  Ensuring Prometheus Operator CRDs (for Triton metrics)"
if ! kubectl get crd podmonitors.monitoring.coreos.com >/dev/null 2>&1; then
  quiet "kubectl apply PodMonitor CRD" \
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.75.0/example/prometheus-operator-crd/monitoring.coreos.com_podmonitors.yaml || \
    warn "Could not install PodMonitor CRD"
fi
if ! kubectl get crd prometheusrules.monitoring.coreos.com >/dev/null 2>&1; then
  quiet "kubectl apply PrometheusRule CRD" \
    kubectl apply --server-side -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/v0.75.0/example/prometheus-operator-crd/monitoring.coreos.com_prometheusrules.yaml || \
    warn "Could not install PrometheusRule CRD"
fi

# =========================================================================
# Step 7: IRSA — IAM role for Triton S3 model access
# =========================================================================
step "Step 7: Ensuring IRSA for Triton S3 access"
TRITON_POLICY_NAME="${STACK_NAME}-triton-s3-policy-${STACK_UID}"
TRITON_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${TRITON_POLICY_NAME}"

POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${MODEL_BUCKET}\",\"arn:aws:s3:::${MODEL_BUCKET}/*\"]}]}"
if aws iam get-policy --policy-arn "${TRITON_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${TRITON_POLICY_ARN}" \
    --policy-document "${POLICY_DOC}" \
    --set-as-default >>"${DEPLOY_LOG:-/dev/null}" 2>&1 || true
else
  aws iam create-policy \
    --policy-name "${TRITON_POLICY_NAME}" \
    --policy-document "${POLICY_DOC}" >/dev/null
fi

quiet "eksctl create iamserviceaccount triton-sa" \
  eksctl create iamserviceaccount \
    --name triton-sa \
    --namespace default \
    --cluster "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --attach-policy-arn "${TRITON_POLICY_ARN}" \
    --approve \
    --override-existing-serviceaccounts || true

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
step "Step 7.1: Ensuring IRSA for the Model Optimizer S3 access"
OPTIMIZER_POLICY_NAME="${STACK_NAME}-model-optimizer-s3-${STACK_UID}"
OPTIMIZER_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${OPTIMIZER_POLICY_NAME}"
OPTIMIZER_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\"],\"Resource\":\"arn:aws:s3:::${MODEL_BUCKET}/*\"},{\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::${MODEL_BUCKET}\"}]}"
if aws iam get-policy --policy-arn "${OPTIMIZER_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${OPTIMIZER_POLICY_ARN}" \
    --policy-document "${OPTIMIZER_POLICY_DOC}" \
    --set-as-default >>"${DEPLOY_LOG:-/dev/null}" 2>&1 || true
else
  aws iam create-policy \
    --policy-name "${OPTIMIZER_POLICY_NAME}" \
    --policy-document "${OPTIMIZER_POLICY_DOC}" >/dev/null
fi

quiet "eksctl create iamserviceaccount model-optimizer-sa" \
  eksctl create iamserviceaccount \
    --name model-optimizer-sa \
    --namespace default \
    --cluster "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --attach-policy-arn "${OPTIMIZER_POLICY_ARN}" \
    --approve \
    --override-existing-serviceaccounts || true

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
# Step 7.2: IRSA for the Deal Scorer (deals table read)
# The Deal Scorer primes its deal book from the deals table at startup and
# refreshes it every 30 seconds off the request path. Read-only, on the one
# table. Its own service account rather than the node role, which every pod on
# the node shares. The floor and margin containers get no table access.
# =========================================================================
step "Step 7.2: Ensuring IRSA for the Deal Scorer deals table read"
DEAL_SCORER_POLICY_NAME="${STACK_NAME}-deal-scorer-deals-read-${STACK_UID}"
DEAL_SCORER_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DEAL_SCORER_POLICY_NAME}"
DEAL_SCORER_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:Query\",\"dynamodb:Scan\"],\"Resource\":\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${DEALS_TABLE}\"}]}"
if aws iam get-policy --policy-arn "${DEAL_SCORER_POLICY_ARN}" >/dev/null 2>&1; then
  update_managed_policy "${DEAL_SCORER_POLICY_ARN}" "${DEAL_SCORER_POLICY_NAME}" "${DEAL_SCORER_POLICY_DOC}"
else
  aws iam create-policy \
    --policy-name "${DEAL_SCORER_POLICY_NAME}" \
    --policy-document "${DEAL_SCORER_POLICY_DOC}" >/dev/null
fi
quiet "eksctl create iamserviceaccount deal-scorer-sa" \
  eksctl create iamserviceaccount \
    --name deal-scorer-sa \
    --namespace default \
    --cluster "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --attach-policy-arn "${DEAL_SCORER_POLICY_ARN}" \
    --approve \
    --override-existing-serviceaccounts || true
DEAL_SCORER_ROLE_ARN=""
for attempt in $(seq 1 10); do
  DEAL_SCORER_ROLE_ARN="$(kubectl get sa deal-scorer-sa -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo '')"
  [[ -n "${DEAL_SCORER_ROLE_ARN}" ]] && break
  sleep 3
done
if [[ -z "${DEAL_SCORER_ROLE_ARN}" ]]; then
  # Fatal: without the annotation the Deal Scorer cannot read its deal book, so
  # it fails its prime and never becomes ready.
  fail "Deal Scorer IRSA role annotation did not populate after 30s. If deal-scorer-sa's ServiceAccount is missing from the cluster but its CFN stack (eksctl-${CLUSTER_NAME}-addon-iamserviceaccount-default-deal-scorer-sa) still exists, eksctl will not recreate it automatically. Fix by deleting that CFN stack first, or manually: kubectl annotate sa deal-scorer-sa eks.amazonaws.com/role-arn=<role-arn> --overwrite"
fi
log "  Deal Scorer IRSA role: ${DEAL_SCORER_ROLE_ARN}"

# =========================================================================
# Step 7.5: DynamoDB permissions for orchestrator pods
# =========================================================================
step "Step 7.5: Ensuring DynamoDB access for orchestrator"
DYNAMO_POLICY_NAME="${STACK_NAME}-dynamo-loadtest-${STACK_UID}"
DYNAMO_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${DYNAMO_POLICY_NAME}"
# Three statements, each scoped to one table ARN, no wildcard resource. The
# registry statement grants UpdateItem (the activation toggle uses a SET
# expression so it cannot clobber a display name or description written
# concurrently) but deliberately NOT Scan: the table's constant partition key
# means one Query returns every record. DealsBook is read-only on the deals
# table: shared/deal_library.py Queries the base table (no index) for the
# auction type of each deal the Deal Scorer activates.
DYNAMO_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[\
{\"Sid\":\"LoadTestHistory\",\"Effect\":\"Allow\",\"Action\":[\"dynamodb:PutItem\",\"dynamodb:GetItem\",\"dynamodb:Query\",\"dynamodb:Scan\"],\"Resource\":\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${LOADTEST_TABLE}\"},\
{\"Sid\":\"ContainerRegistry\",\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:Query\",\"dynamodb:PutItem\",\"dynamodb:UpdateItem\"],\"Resource\":\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${CONTAINER_REGISTRY_TABLE}\"},\
{\"Sid\":\"DealsBook\",\"Effect\":\"Allow\",\"Action\":[\"dynamodb:GetItem\",\"dynamodb:Query\"],\"Resource\":\"arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${DEALS_TABLE}\"}\
]}"

if aws iam get-policy --policy-arn "${DYNAMO_POLICY_ARN}" >/dev/null 2>&1; then
  update_managed_policy "${DYNAMO_POLICY_ARN}" "${DYNAMO_POLICY_NAME}" "${DYNAMO_POLICY_DOC}"
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
step "Step 7.6: Ensuring EKS nodegroup scaling access for orchestrator"
EKS_SCALE_POLICY_NAME="${STACK_NAME}-eks-gpu-scale-${STACK_UID}"
EKS_SCALE_POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${EKS_SCALE_POLICY_NAME}"
EKS_SCALE_POLICY_DOC="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"eks:UpdateNodegroupConfig\",\"eks:DescribeNodegroup\"],\"Resource\":\"arn:aws:eks:${AWS_REGION}:${ACCOUNT_ID}:nodegroup/${CLUSTER_NAME}/*\"}]}"

if aws iam get-policy --policy-arn "${EKS_SCALE_POLICY_ARN}" >/dev/null 2>&1; then
  aws iam create-policy-version \
    --policy-arn "${EKS_SCALE_POLICY_ARN}" \
    --policy-document "${EKS_SCALE_POLICY_DOC}" \
    --set-as-default >>"${DEPLOY_LOG:-/dev/null}" 2>&1 || true
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
step "Step 7.7: Ensuring closed-loop (Part 2) access for orchestrator"
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
{\"Sid\":\"TrainingTriggerFromGovernanceUI\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:CreateTrainingJob\",\"sagemaker:AddTags\",\"sagemaker:DescribeTrainingJob\"],\"Resource\":[\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:training-job/dlrm-bid-shader-*\",\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:training-job/ncf-deal-manager-*\",\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:training-job/deal-yield-manager-floor-*\",\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:training-job/deal-yield-manager-margin-*\"]},\
{\"Sid\":\"ListTrainingJobsFromGovernanceUI\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:ListTrainingJobs\"],\"Resource\":\"*\"},\
{\"Sid\":\"GlueJobRunsForTrainableRunFilter\",\"Effect\":\"Allow\",\"Action\":[\"glue:GetJobRuns\"],\"Resource\":[\"arn:aws:glue:${AWS_REGION}:${ACCOUNT_ID}:job/*feature-engineering-etl\"]},\
{\"Sid\":\"GlueOnDemandSweepAfterLoadTest\",\"Effect\":\"Allow\",\"Action\":[\"glue:StartJobRun\"],\"Resource\":[\"arn:aws:glue:${AWS_REGION}:${ACCOUNT_ID}:job/*feature-engineering-etl\"]},\
{\"Sid\":\"PassSageMakerTrainingRole\",\"Effect\":\"Allow\",\"Action\":[\"iam:PassRole\"],\"Resource\":\"arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_TRAINING_ROLE_NAME}\",\"Condition\":{\"StringEquals\":{\"iam:PassedToService\":\"sagemaker.amazonaws.com\"}}},\
{\"Sid\":\"PromoteFromGovernanceUI\",\"Effect\":\"Allow\",\"Action\":[\"sagemaker:UpdateModelPackage\"],\"Resource\":[\"arn:aws:sagemaker:${AWS_REGION}:${ACCOUNT_ID}:model-package/*artf-*/*\"]},\
{\"Sid\":\"TritonModelRepoReadWrite\",\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${MODEL_BUCKET}\",\"arn:aws:s3:::${MODEL_BUCKET}/triton-models/*\"]},\
{\"Sid\":\"KmsForDynamoDb\",\"Effect\":\"Allow\",\"Action\":[\"kms:Decrypt\",\"kms:GenerateDataKey\",\"kms:DescribeKey\"],\"Resource\":\"*\",\"Condition\":{\"StringEquals\":{\"kms:ViaService\":[\"dynamodb.${AWS_REGION}.amazonaws.com\"]}}}\
]}"

if aws iam get-policy --policy-arn "${CLOSED_LOOP_POLICY_ARN}" >/dev/null 2>&1; then
  update_managed_policy "${CLOSED_LOOP_POLICY_ARN}" "${CLOSED_LOOP_POLICY_NAME}" "${CLOSED_LOOP_POLICY_DOC}"
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
if _run_phase 3; then
phase 3 "Deploying workloads"
step "Step 8: Applying Kubernetes manifests"

# The deal-scorer Deployment's NCF_MODEL_LABEL (substituted below). Resolved first
# so a missing build stops the phase before anything is applied.
resolve_ncf_model_label

# --- Provision Cognito BEFORE applying manifests so the orchestrator gets the real pool ID ---
run_step "Provisioning the Cognito user pool (orchestrator auth)" \
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
    --action deploy \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
    --cloudfront-domain "${CF_DOMAIN:-localhost}"

# Bedrock grant for the Auction Theater captions. Applied HERE, in the base
# deploy, and not alongside the closed-loop agent grant: that one needs AgentCore
# runtime ARNs so it lives behind --with-retraining, whereas this one needs
# nothing but the role that was just created. Gating it the same way would ship a
# theater whose captions never generate, with no error anywhere but the browser
# console -- the fallback caption is correct, so nothing would look broken.
quiet "deploy_cognito.py grant-caption-invoke" \
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
    --action grant-caption-invoke \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
  || warn "Caption grant failed - theater captions will fall back to factual text."

COGNITO_OUTPUTS="${SCRIPT_DIR}/.cognito-outputs.json"
COGNITO_USER_POOL_ID="$(jq -r '.UserPoolId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
COGNITO_CLIENT_ID="$(jq -r '.ClientId // empty' "${COGNITO_OUTPUTS}" 2>/dev/null || echo '')"
log "  Cognito Pool: ${COGNITO_USER_POOL_ID}  Client: ${COGNITO_CLIENT_ID}"
# Recorded so a later --resume/--start-at run that skips this phase can still
# print a working credential-reset command (see print_demo_credentials).
state_set "${STACK_PREFIX}" resolved \
  "cognitoUserPoolId=${COGNITO_USER_POOL_ID}" \
  "cognitoClientId=${COGNITO_CLIENT_ID}" \
  "modelBucket=${MODEL_BUCKET}" \
  "imageTag=${IMAGE_TAG}"

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
state_set "${STACK_PREFIX}" resolved \
  "identityPoolId=${IDENTITY_POOL_ID:-}" \
  "authRoleName=${ID_POOL_AUTH_ROLE_NAME:-}"

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
quiet "kubectl apply model-optimizer-bootstrap Job" kubectl apply -f "${BOOTSTRAP_MANIFEST}"

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
_APPLIED_MANIFESTS=(triton-deployment.yaml triton-internal-nlb.yaml artf-containers-deployment.yaml orchestrator-deployment.yaml orchestrator-internal-nlb.yaml triton-hpa.yaml)
for manifest in "${_APPLIED_MANIFESTS[@]}"; do
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
      -e "s|__ARTF_MUTATIONS_REQUIRED_SCOPE__|${ARTF_MUTATIONS_REQUIRED_SCOPE}|g" \
      -e "s|__PARAMETER_STORE_TABLE__|${STACK_PREFIX:+${STACK_PREFIX}-}parameter-store|g" \
      -e "s|__AUDIT_TRAIL_TABLE__|${STACK_PREFIX:+${STACK_PREFIX}-}audit-trail|g" \
      -e "s|__CONTAINER_REGISTRY_TABLE__|${CONTAINER_REGISTRY_TABLE}|g" \
      -e "s|__DLRM_MODEL_GROUP__|${STACK_PREFIX:+${STACK_PREFIX}-}artf-dlrm-bid-shader|g" \
      -e "s|__NCF_MODEL_GROUP__|${STACK_PREFIX:+${STACK_PREFIX}-}artf-ncf-deal-manager|g" \
      -e "s|__YIELD_FLOOR_MODEL_GROUP__|${YIELD_FLOOR_MODEL_GROUP}|g" \
      -e "s|__YIELD_MARGIN_MODEL_GROUP__|${YIELD_MARGIN_MODEL_GROUP}|g" \
      -e "s|__SAGEMAKER_TRAINING_ROLE_ARN__|arn:aws:iam::${ACCOUNT_ID}:role/${SAGEMAKER_TRAINING_ROLE_NAME}|g" \
      -e "s|__TRAINING_DATA_BUCKET__|${TRAINING_DATA_BUCKET}|g" \
      -e "s|__TRAINING_IMAGE_REGISTRY__|${REGISTRY}|g" \
      -e "s|__XGBOOST_TRAINING_IMAGE_URI__|${XGBOOST_TRAINING_IMAGE_URI}|g" \
      -e "s|__FEEDBACK_STREAM_NAME__|${FEEDBACK_STREAM_NAME}|g" \
      -e "s|__DEAL_YIELD_FEEDBACK_STREAM_NAME__|${DEAL_YIELD_FEEDBACK_STREAM_NAME}|g" \
      -e "s|__OUTCOME_SIMULATOR_ENABLED__|${OUTCOME_SIMULATOR_ENABLED}|g" \
      -e "s|__OUTCOME_SIMULATOR_WIN_RATE__|${OUTCOME_SIMULATOR_WIN_RATE}|g" \
      -e "s|__OUTCOME_SIMULATOR_IMPRESSION_RATE__|${OUTCOME_SIMULATOR_IMPRESSION_RATE}|g" \
      -e "s|__OUTCOME_SIMULATOR_CLICK_RATE__|${OUTCOME_SIMULATOR_CLICK_RATE}|g" \
      -e "s|__OUTCOME_SIMULATOR_CONVERSION_RATE__|${OUTCOME_SIMULATOR_CONVERSION_RATE}|g" \
      -e "s|__OUTCOME_SIMULATOR_CONVERSION_VALUE__|${OUTCOME_SIMULATOR_CONVERSION_VALUE}|g" \
      -e "s|__GLUE_JOB_NAME__|${GLUE_JOB_NAME}|g" \
      -e "s|__DEAL_YIELD_GLUE_JOB_NAME__|${DEAL_YIELD_GLUE_JOB_NAME}|g" \
      -e "s|__NCF_MODEL_LABEL__|${NCF_MODEL_LABEL}|g" \
      "${SCRIPT_DIR}/eks/${manifest}" > "${PROCESSED}"
  grep -q '__NCF_MODEL_LABEL__' "${PROCESSED}" \
    && fail 3 "${manifest} still has an unsubstituted __NCF_MODEL_LABEL__"
  quiet "kubectl apply ${manifest}" kubectl apply -f "${PROCESSED}"
done
log "  Applied ${#_APPLIED_MANIFESTS[@]} manifests"

# --- Prune pre-split Yield Optimizer objects.
# `kubectl apply` only creates and updates; it NEVER deletes objects that were
# removed from (or renamed within) a manifest. The Yield Optimizer used to be a
# single `yield-optimizer` Deployment/Service/HPA and is now
# `yield-optimizer-floor` + `yield-optimizer-margin`, so applying the new
# manifest leaves the old trio Running on any cluster deployed before the split
# (7 ARTF pods instead of 6, the stale pod still on the pre-split combined
# image).
#
# It receives no traffic -- the orchestrator's CONTAINERS registry no longer
# lists it -- so this is not a correctness problem, but it holds a pod slot, its
# HPA can still scale it to 5 replicas, and it contradicts the six-container
# topology the docs describe. Deleting exactly these three names (never a broad
# prune) is idempotent and a no-op on a fresh cluster.
for OBJ in "deployment/yield-optimizer" "service/yield-optimizer" "hpa/yield-hpa"; do
  if kubectl get "${OBJ}" >/dev/null 2>&1; then
    kubectl delete "${OBJ}" --ignore-not-found >/dev/null 2>&1 && \
      say "  Removed pre-split ${OBJ} (replaced by yield-optimizer-floor/-margin)"
  fi
done

# --- Roll out a Deployment when only its image CONTENT changed.
# IMAGE_TAG is pinned to the tag recorded in .image-outputs.json whenever the
# registry matches, so a rebuild that changes only the source inside an image
# produces an identical pod template. `kubectl apply` is then a no-op, and
# `imagePullPolicy: Always` only takes effect for pods that are newly created -- so
# the old pod keeps serving the old code while every step above reports success.
#
# Each managed Deployment's running pods are compared with the digest ECR holds for
# its image at IMAGE_TAG, and only the ones that differ are restarted. A digest that
# cannot be read on either side (a pod still starting, an unreadable cluster or
# repository) counts as different, so a missed restart is never the result of not
# knowing. Applies to the orchestrator too, which shares source/shared/ with the
# containers.
log "  Restarting the Deployments whose pods do not run ECR's image for ${IMAGE_TAG}..."
restart_stale_deployments "${IMAGE_TAG}"

# Every Deployment this step applied or restarted is waited on, and a rollout that
# does not complete fails the deploy with the pods' reasons and last log lines. A
# step that carried on here once reported "Deployed" over a crash-looping Deal
# Scorer. Triton gets 900 s: a new GPU node can take several minutes to join and
# pull the image. The rest get 300 s.
wait_managed_rollouts() {
  local name timeout failed=()
  for name in triton-inference-server "${MANAGED_DEPLOYMENTS[@]}"; do
    kubectl get "deployment/${name}" >/dev/null 2>&1 || continue
    timeout=300
    [[ "${name}" == "triton-inference-server" ]] && timeout=900
    say "  Waiting for deployment/${name} to roll out (up to ${timeout}s)..."
    if ! wait_rollout "${name}" "" "${timeout}"; then
      failed+=("${ROLLOUT_FAILURE}")
      [[ "${name}" == "triton-inference-server" ]] && \
        warn "  GPU capacity: kubectl get nodes -l nvidia.com/gpu=present"
    fi
  done
  if [[ ${#failed[@]} -gt 0 ]]; then
    local joined
    joined="$(printf '%s; ' "${failed[@]}")"
    _DEPLOY_DEGRADED=1
    _DEGRADED_DETAIL="${_DEGRADED_DETAIL:+${_DEGRADED_DETAIL}, and }rollouts that did not complete (${joined%; })"
    warn "Rollouts did not complete: ${joined%; }"
    return 1
  fi
  return 0
}
wait_managed_rollouts || true

# NOTE: the orchestrator is intentionally NOT patched with any agent runtime ARN.
# It must never invoke the closed-loop agents (FR-6) — the browser invokes them
# directly via SigV4. Agent ARNs are wired into the frontend build in Step 11.

# =========================================================================
# Step 8.4: UI API proxy Lambda -- the orchestrator's only way in
# =========================================================================
# The browser reaches the orchestrator through this Lambda, which forwards from
# inside the VPC to the orchestrator's internal NLB (eks/orchestrator-internal-nlb.yaml,
# applied with the other manifests above). The NLB hostname is waited for here, passed
# to the stack, and recorded; then the function is invoked once to prove the path
# works before the UI build in Step 9 bakes the function ARN into the bundle. The
# Identity Pool authenticated role is granted lambda:InvokeFunction on exactly this
# function.
step "Step 8.4: Deploying the UI API proxy Lambda (${UI_API_PROXY_STACK})"
resolve_orchestrator_internal_url 3
state_set "${STACK_PREFIX}" resolved "orchestratorInternalUrl=${ORCHESTRATOR_INTERNAL_URL}"
deploy_ui_api_proxy 3 "${ORCHESTRATOR_INTERNAL_URL}"
state_set "${STACK_PREFIX}" resolved "uiApiProxyArn=${UI_API_PROXY_ARN}"
verify_ui_api_proxy 3
quiet "deploy_cognito.py grant-ui-api-invoke" \
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
    --action grant-ui-api-invoke \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
    --ui-api-proxy-arn "${UI_API_PROXY_ARN}" \
  || fail 3 "Could not grant the Identity Pool authenticated role lambda:InvokeFunction on ${UI_API_PROXY_ARN}; the UI would have no route to the orchestrator."

# =========================================================================
# Step 8.5: Schedule GPU node group shutdown at 8pm ET daily
# =========================================================================
step "Step 8.5: Configuring GPU scheduled shutdown (8pm ET daily)"

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
if _run_phase 4; then
phase 4 "Setting up access"
step "Step 9: Deploying frontend"

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
VITE_BEDROCK_REGION=${AWS_REGION}
VITE_CAPTION_INFERENCE_PROFILE_ID=${CAPTION_INFERENCE_PROFILE_ID}
VITE_ADAPTIVE_BIDDING_RUNTIME_ARN=
VITE_GOVERNANCE_RUNTIME_ARN=
VITE_UI_API_PROXY_ARN=${UI_API_PROXY_ARN}
EOF

# React UI distribution (static-only CloudFront; API calls go browser -> ui-api-proxy)
run_step "Building and publishing the frontend (S3 + CloudFront)" \
  "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_frontend.py" \
    --action deploy \
    --stack-name "${STACK_NAME}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}"

PRIMARY_OUTPUTS="${SCRIPT_DIR}/.frontend-outputs.json"
CF_DOMAIN="$(jq -r '.CloudFrontDomain // empty' "${PRIMARY_OUTPUTS}" 2>/dev/null || echo '')"
state_set "${STACK_PREFIX}" resolved "cloudFrontDomain=${CF_DOMAIN}"

# Update Cognito callback URLs now that we know the CF domain
if [[ -n "${CF_DOMAIN}" && -n "${COGNITO_USER_POOL_ID}" ]]; then
  quiet "deploy_cognito.py: callback URLs for ${CF_DOMAIN}" \
    "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_cognito.py" \
      --action deploy \
      --stack-name "${STACK_NAME}" \
      --region "${AWS_REGION}" \
      --profile "${AWS_PROFILE}" \
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
      TEMP_PASS="$("${PYTHON}" - <<'PY'
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
# Phase 5 is gated like the others, but it had no START_AT guard to replace, so the
# whole region is wrapped here. Everything the Summary reads from inside it
# (COGNITO_USER_POOL_ID in particular) is referenced with a `:-` default, which is
# what makes skipping this block safe.
_PHASE5_RAN=0
if _run_phase 5; then
_PHASE5_RAN=1
phase 5 "Registering agents"
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
  step "Step 10: Deploying AgentCore MCP runtime"

  # The AgentCore SDK logs progress at INFO, which put dozens of lines into the
  # middle of an otherwise quiet deployment. It goes to the run's detail log like
  # every other step; run_logged still tails it inline if the deploy fails.
  _AC_LOG="${DEPLOY_LOG}"

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
  # Read before the registration, which resolves the same tag; recorded only once
  # the registration succeeds (record_agentcore_digest).
  _AC_DIGEST="$(ecr_image_digest "${STACK_NAME}-agentcore" "${IMAGE_TAG}")"
  run_logged "${_AC_LOG}" "Registering MCP runtime ${AC_RUNTIME_NAME}" \
    "${PYTHON}" "${SCRIPT_DIR}/scripts/deploy_to_agentcore.py" \
      --action deploy \
      --runtime-name "${AC_RUNTIME_NAME}" \
      --role-arn "${ROLE_ARN}" \
      --container-uri "${AC_IMAGE}" \
      --region "${AWS_REGION}" \
      --profile "${AWS_PROFILE}" \
    || fail "AgentCore MCP runtime registration failed (detail in ${_AC_LOG})"
  record_agentcore_digest "${_AC_DIGEST}"
else
  warn "Skipping AgentCore deployment (--skip-agentcore)"
fi

# =========================================================================
# Step 11 (optional): Deploy closed-loop retraining infrastructure
# =========================================================================
if [[ "${WITH_RETRAINING}" -eq 1 ]]; then
  log ""
  step "Step 11: Deploying closed-loop retraining infrastructure (--with-retraining)"
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
  # The resolved profile travels as an explicit argument, not only as inherited
  # environment, so the child's own output names it and a standalone run of the
  # child behaves the same way.
  CLOSED_LOOP_ARGS="--profile ${AWS_PROFILE}"
  if [[ -n "${STACK_PREFIX}" ]]; then
    CLOSED_LOOP_ARGS="${CLOSED_LOOP_ARGS} --prefix ${STACK_PREFIX}"
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
  # One --verbose on this script turns on the child too. Without this pass-through,
  # deploy_closed_loop.sh's newly gated log() would hide its detail even from
  # someone who explicitly asked for it.
  if [[ "${VERBOSE}" -eq 1 ]]; then
    CLOSED_LOOP_ARGS="${CLOSED_LOOP_ARGS} --verbose"
  fi

  # Phase 5's output is TEE'd, not just printed. deploy_closed_loop.sh writes to
  # stdout only, so when it failed for real the error existed nowhere but the
  # terminal scrollback -- and "check the output above" is useless to anyone who
  # detached, closed the tab, or came back the next morning. It now lands in a file
  # named by the failure message. It is the run's detail log: the child's own quiet
  # mode (same contract as this script) sends its chatty children there directly,
  # and its step lines reach both the terminal and the log through this tee.
  _CL_LOG="${DEPLOY_LOG}"
  _log_section "deploy_closed_loop.sh"
  _CL_TEE=("${_CL_LOG}")

  # PIPESTATUS[0], not $?: a pipeline reports its LAST element, so `... | tee`
  # would report tee's success and a failed Phase 5 would look fine. `set -o
  # pipefail` is on and would also catch it, but reading the child's status
  # directly does not depend on an option somebody could turn off later.
  set +e
  CL_DETAIL_LOG="${_CL_LOG}" \
  VPC_ID="${CL_VPC_ID}" \
  SUBNET_IDS="${CL_SUBNET_IDS}" \
  EKS_NODE_ROLE="${CL_NODE_ROLE_ARN}" \
  MODEL_BUCKET="${MODEL_BUCKET}" \
  CLUSTER_NAME="${CLUSTER_NAME}" \
  BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID}" \
  ADAPTIVE_BIDDING_MODEL_ID="${ADAPTIVE_BIDDING_MODEL_ID}" \
  GOVERNANCE_MODEL_ID="${GOVERNANCE_MODEL_ID}" \
  PATH="${PYTHON_BIN_DIR:+${PYTHON_BIN_DIR}:}${PATH}" \
  bash "${SCRIPT_DIR}/deploy_closed_loop.sh" ${CLOSED_LOOP_ARGS} 2>&1 \
    | tee -a "${_CL_TEE[@]}"
  _CL_RC=${PIPESTATUS[0]}
  set -e

  if [[ "${_CL_RC}" -ne 0 ]]; then
    # This block used to print two warnings and let the run continue to the full
    # success banner. That is how a detached deploy reported success while the
    # vpc-proxy and governance-eventbridge stacks did not exist -- the failure was
    # visible only as two lines scrolled off the top of a 40-minute log.
    _DEPLOY_DEGRADED=1
    _DEGRADED_DETAIL="the closed-loop stack (deploy_closed_loop.sh exited non-zero)"
    warn "Closed-loop deployment exited ${_CL_RC}."
    # The last lines are almost always the error, and they are what a reader wants
    # without opening a file. The full log is named right after.
    warn "Last 15 lines of its output:"
    tail -15 "${_CL_LOG}" 2>/dev/null | while IFS= read -r _cl_line; do
      printf '         %s\n' "${_cl_line}"
    done
    warn "Full Phase 5 output: ${_CL_LOG}"
    report_closed_loop_stacks
  fi

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
# Phase 5 is the one phase whose ok() can be reached after a failure: the
# closed-loop and Prebid calls below deliberately do not abort the run, they set
# _DEPLOY_DEGRADED instead. So its completion is conditional, where the other four
# phases can attach unconditionally to their ok().
if [[ "${_DEPLOY_DEGRADED}" -eq 0 ]]; then
  ok "Agents registered"
else
  warn "Phase 5/5 incomplete: ${_DEGRADED_DETAIL}"
fi

fi # _run_phase 5 (Phase 5)

# =========================================================================
# Step 12 (optional): Deploy Prebid Server as the sell-side ARTF host
#
# Delegated, mirroring Step 11's call into deploy_closed_loop.sh. Everything
# specific to Prebid lives in deploy_prebid.sh: this block only decides whether
# to call it and passes through what it cannot discover for itself.
#
# Runs with --with-prebid whenever Phase 5 runs, and also when Phase 5 is skipped
# but the Prebid host's sources or the deals seed changed since the last
# successful Step 12 (prebid_fingerprint, recorded in .deploy-state.json), so a
# re-run picks up a Prebid change without calling deploy_prebid.sh by hand.
#
# --yes is passed because deploy.sh has already been invoked deliberately with
# --with-prebid. deploy_prebid.sh still PRINTS its cost disclosure; it just does
# not stop mid-deployment to ask a question the operator answered by adding the
# flag. Run deploy_prebid.sh directly for the interactive confirmation.
# =========================================================================
if [[ "${WITH_PREBID}" -eq 1 ]]; then
  _PB_SEED_SHA="$(file_sha256 "${SCRIPT_DIR}/seed/deals_and_campaigns.json" 2>/dev/null || true)"
  PREBID_FINGERPRINT="$(prebid_fingerprint "${SCRIPT_DIR}/.." "${_PB_SEED_SHA}" 2>/dev/null || true)"
  if ! [[ "${PREBID_FINGERPRINT}" =~ ^[0-9a-f]{64}$ ]]; then PREBID_FINGERPRINT=""; fi
fi
if prebid_step_needed "${_PHASE5_RAN}"; then
  if [[ "${_PHASE5_RAN}" -eq 0 ]]; then
    say "  Phase 5/5 reads as complete in AWS, but ${PREBID_STEP_WHY}; running Step 12."
  fi
  log ""
  step "Step 12: Deploying Prebid Server as the sell-side ARTF host (--with-prebid)"
  log ""
  # COGNITO_USER_POOL_ID is assigned in Phase 3 and re-read in Phase 4, so on a
  # --start-at 5 / --resume run it is unset here -- and the --user-pool-id
  # pass-through below is conditional on it being non-empty. That combination
  # silently deployed the Prebid host with no user pool. Fall back to the recorded
  # value so a resumed run behaves like a full one.
  COGNITO_USER_POOL_ID="${COGNITO_USER_POOL_ID:-$(state_read "${STACK_PREFIX}" resolved.cognitoUserPoolId)}"
  PREBID_ARGS=""
  if [[ -n "${STACK_PREFIX}" ]]; then
    PREBID_ARGS="--prefix ${STACK_PREFIX}"
  fi
  PREBID_ARGS="${PREBID_ARGS} --cluster ${CLUSTER_NAME} --region ${AWS_REGION} --profile ${AWS_PROFILE} --yes"
  if [[ -n "${COGNITO_USER_POOL_ID:-}" ]]; then
    PREBID_ARGS="${PREBID_ARGS} --user-pool-id ${COGNITO_USER_POOL_ID}"
  fi
  # Invoked through `bash` rather than executed directly: the execute bit is a
  # property of the checkout, not of the code, and losing it turned a working deploy
  # into "Permission denied" at Step 12 with Phases 1-4 already applied. Nothing here
  # needs the mode bit to be right.
  #
  # PREBID_DETAIL_LOG switches the child into the same quiet contract as this script:
  # its step lines come to the terminal, its narration and its chatty children go to
  # the detail log. Run standalone (no PREBID_DETAIL_LOG) it streams everything, as it
  # always did. --verbose is passed through so one flag governs both.
  if [[ "${VERBOSE}" -eq 1 ]]; then PREBID_ARGS="${PREBID_ARGS} --verbose"; fi
  _log_section "deploy_prebid.sh"
  set +e
  PREBID_DETAIL_LOG="${DEPLOY_LOG}" \
  PATH="${PYTHON_BIN_DIR:+${PYTHON_BIN_DIR}:}${PATH}" \
  bash "${SCRIPT_DIR}/deploy_prebid.sh" ${PREBID_ARGS} 2>&1 \
    | tee -a "${DEPLOY_LOG}"
  _PB_RC=${PIPESTATUS[0]}
  set -e
  if [[ "${_PB_RC}" -ne 0 ]]; then
    _DEPLOY_DEGRADED=1
    _DEGRADED_DETAIL="${_DEGRADED_DETAIL:+${_DEGRADED_DETAIL}, and }the Prebid ARTF host (deploy_prebid.sh exited ${_PB_RC})"
    warn "Prebid deployment exited ${_PB_RC}. Last 15 lines of its output:"
    tail -15 "${DEPLOY_LOG}" 2>/dev/null | while IFS= read -r _pb_line; do
      printf '         %s\n' "${_pb_line}"
    done
    warn "Full output: ${DEPLOY_LOG}"
    warn "Re-run on its own: ${SCRIPT_DIR}/deploy_prebid.sh ${PREBID_ARGS}"
  elif [[ -n "${PREBID_FINGERPRINT}" ]]; then
    # Recorded only after a successful Step 12, so a failed or interrupted one runs
    # again on the next deploy.
    state_set "${STACK_PREFIX}" resolved "prebidFingerprint=${PREBID_FINGERPRINT}"
    log "  Prebid host fingerprint ${PREBID_FINGERPRINT:0:12} recorded for the next run"
  fi
elif [[ "${WITH_PREBID}" -eq 1 ]]; then
  ok "Prebid ARTF host already deployed from these sources (${PREBID_FINGERPRINT:0:12}); skipping Step 12"
elif [[ "${_PHASE5_RAN}" -eq 1 ]]; then
  log ""
  log "  Skipping the Prebid ARTF host (pass --with-prebid to enable)."
  log "  This includes: a pinned Prebid Server build, a Cognito domain and M2M client,"
  log "  a Secrets Manager credential, and the artfhouse demand endpoint."
fi

# =========================================================================
# Summary
# =========================================================================
# Always printed regardless of --verbose ("print ONLY what the user needs to get
# started"), so this block uses say() (always-visible) rather than log().
#
# Values a SKIPPED phase would have resolved are read back from the state file.
# Without this, every --start-at 4 / --resume run printed "(pending)" for the
# frontend URL and endpoint even though both already existed -- the phase that
# assigns those variables simply had not run this time.
CF_DOMAIN="${CF_DOMAIN:-$(state_read "${STACK_PREFIX}" resolved.cloudFrontDomain)}"
UI_API_PROXY_ARN="${UI_API_PROXY_ARN:-$(state_read "${STACK_PREFIX}" resolved.uiApiProxyArn)}"
COGNITO_USER_POOL_ID="${COGNITO_USER_POOL_ID:-$(state_read "${STACK_PREFIX}" resolved.cognitoUserPoolId)}"

# print_demo_credentials(): the login block, called from BOTH the success and the
# partial-completion summaries.
#
# The reset command prints in every case, with the real pool ID substituted, so it
# is copy-pasteable rather than a template. Cognito cannot disclose an existing
# user's password -- it is hashed, AdminGetUser returns no credential, and
# AdminCreateUser does not return the temporary password it set -- so a password
# that has scrolled past can only be reset, never recovered. That is what this
# command is for, and why it is not hidden behind the 'existing' case.
print_demo_credentials() {
  say "  Demo Login (Cognito):"
  case "${DEMO_LOGIN_STATUS:-no-auth}" in
    created)
      say "    Username:  ${DEMO_USER_EMAIL}"
      say "    Password:  ${DEMO_USER_TEMP_PASSWORD}"
      say "    Note:      Temporary, and shown ONLY here — save it now. You'll set a"
      say "               permanent password on first login."
      ;;
    existing)
      say "    Username:  ${DEMO_USER_EMAIL}"
      say "    Password:  (existing user — Cognito does not disclose it; reset it below)"
      ;;
    *)
      say "    (Cognito auth not configured — orchestrator auth is disabled)"
      ;;
  esac
  if [[ -n "${COGNITO_USER_POOL_ID:-}" ]]; then
    say "    Lost it?   aws cognito-idp admin-set-user-password \\"
    say "                 --user-pool-id ${COGNITO_USER_POOL_ID} \\"
    say "                 --username ${DEMO_USER_EMAIL:-admin@example.com} \\"
    say "                 --password '<new-password>' --permanent --region ${AWS_REGION}"
    say "               See README \"Demo credentials\" for the full walkthrough."
  elif [[ "${DEMO_LOGIN_STATUS:-no-auth}" != "no-auth" ]]; then
    # Better to say the ID is unknown than to print a command with an empty
    # --user-pool-id that fails on paste.
    say "    Reset:     the user pool ID could not be resolved from this run or from"
    say "               deployment/.deploy-state.json. Find it with:"
    say "                 aws cognito-idp list-user-pools --max-results 60 --region ${AWS_REGION}"
    say "               then see README \"Demo credentials\"."
  fi
  return 0
}

# ---- Partial completion: a failed subsystem must not print a success banner ----
if [[ "${_DEPLOY_DEGRADED}" -ne 0 ]]; then
  say ""
  say "========================================================="
  say "  Accelerator-optimized Agentic Bidding — PARTIALLY DEPLOYED"
  say "========================================================="
  say ""
  warn "This deploy did not finish. What failed: ${_DEGRADED_DETAIL}."
  say ""
  # Re-probed AFTER the failure, so this describes what the account actually holds
  # rather than how far the script believed it got. Those are different things, and
  # the difference is the whole reason this report is trustworthy now.
  gate_refresh "${STACK_PREFIX}"
  gate_print "${STACK_PREFIX}"
  say ""
  say "  What already works:"
  say "    Frontend:  https://${CF_DOMAIN:-'(not deployed)'}"
  say "    UI API:    ${UI_API_PROXY_ARN:-'(not deployed)'}  (Lambda; the orchestrator has no public address)"
  say ""
  print_demo_credentials
  say ""
  say "  To finish it, run the same command again:"
  say "    ./deploy.sh${STACK_PREFIX:+ --prefix ${STACK_PREFIX}}"
  say ""
  say "  It will skip everything above that is already done — no flags needed, and"
  say "  nothing needs tearing down first. To look without deploying:"
  say "    ./deploy.sh${STACK_PREFIX:+ --prefix ${STACK_PREFIX}} --status"
  say ""
  # Non-zero, so the failure is visible to anything scripting around this and is
  # not mistaken for a clean deploy.
  exit 1
fi

say ""
say "========================================================="
say "  Accelerator-optimized Agentic Bidding — Deployed (EKS + Triton)"
say "========================================================="
say ""
say "  Frontend:    https://${CF_DOMAIN:-'(pending)'}"
say "  UI API:      ${UI_API_PROXY_ARN:-'(pending)'}"
say "               (Lambda invoked by the browser; forwards to the orchestrator's"
say "               internal NLB. The orchestrator has no public address.)"
say "  Orchestrator internal NLB: ${ORCHESTRATOR_INTERNAL_URL:-'(pending)'}/health/ready"
say "               (reachable only from inside the VPC)"
say "  Orchestrator in-cluster:  http://orchestrator.default.svc.cluster.local/health/ready"
say "               (kubectl port-forward svc/orchestrator 8080:80 to reach it from here)"
if [[ "${SKIP_AGENTCORE}" -eq 0 ]]; then
say "  AgentCore:   See runtime ARN above"
fi
say ""
print_demo_credentials
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
say "  Check this deployment any time, from anywhere, without deploying:"
say "    ./deploy.sh${STACK_PREFIX:+ --prefix ${STACK_PREFIX}} --status"
say ""
say "  Remembered inputs (so --ngc-key and friends need not be repeated):"
say "    ${DEPLOY_STATE_FILE}"
say ""
