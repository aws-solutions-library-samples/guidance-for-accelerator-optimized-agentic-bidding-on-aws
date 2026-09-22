#!/usr/bin/env bash
#
# attach_artf_container.sh — attach an externally prepared ARTF container.
#
# The container's producer already describes it completely: an
# `artf-registry-record.json` carrying the registry row plus image coordinates,
# and their own `register-artf-container.sh` that writes the row. This script is
# the consuming half and does the work only this repository can do:
#
#   * resolves the registry table from THIS stack, not their default
#     (`artf-container-registry`), which names a table nothing here reads;
#   * REFUSES an endpoint that is not cluster-internal (ARTF conformance);
#   * copies the image into this stack's ECR so nodes never pull cross-account;
#   * checks the intents against the Prebid hook's configured ceiling, which is
#     an allowlist that silently never asks for an intent it does not list;
#   * warns when an intent is already claimed, naming who would win;
#   * renders THEIR Kubernetes example against this repo's conventions;
#   * probes POST /mutate — never /health/ready, which answers ok unconditionally.
#
# A sibling script, deliberately not part of deploy.sh: attaching a container is
# not a deploy, and should not require one.
#
# All validation lives in deployment/scripts/artf_attach.py, which is pure and
# unit-tested. This file is the AWS and cluster orchestration around it.
#
# Usage:
#   ./attach_artf_container.sh --record /path/to/artf-registry-record.json \
#       --stack my-stack [--region us-east-1] [--namespace default] \
#       [--variant cpu|gpu] [--priority 0] [--no-copy] [--apply] [--yes]
#
#   ./attach_artf_container.sh --detach contextual-yield-agent --stack my-stack
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="${SCRIPT_DIR}/scripts/artf_attach.py"

RECORD=""
STACK_NAME="${STACK_NAME:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
NAMESPACE=""
VARIANT="cpu"
PRIORITY="0"
DO_COPY=1
DO_APPLY=0
DETACH=""
ASSUME_YES=0
OUT=""
DEST_REPO=""
TEMPLATE=""
TABLE_OVERRIDE=""
ORCHESTRATOR_DEPLOY="orchestrator"
PREBID_CONFIG_KEY="prebid-server/current/prebid-config.yaml"

log()  { printf '  %s\n' "$*"; }
head1() { printf '\n%s\n' "$*"; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '3,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --record)      RECORD="$2"; shift 2 ;;
    --stack)       STACK_NAME="$2"; shift 2 ;;
    --region)      AWS_REGION="$2"; shift 2 ;;
    --namespace)   NAMESPACE="$2"; shift 2 ;;
    --variant)     VARIANT="$2"; shift 2 ;;
    --priority)    PRIORITY="$2"; shift 2 ;;
    --dest-repo)   DEST_REPO="$2"; shift 2 ;;
    --template)    TEMPLATE="$2"; shift 2 ;;
    --out)         OUT="$2"; shift 2 ;;
    --table)       TABLE_OVERRIDE="$2"; shift 2 ;;
    --no-copy)     DO_COPY=0; shift ;;
    --apply)       DO_APPLY=1; shift ;;
    --detach)      DETACH="$2"; shift 2 ;;
    --yes|-y)      ASSUME_YES=1; shift ;;
    -h|--help)     usage ;;
    *)             fail "Unknown argument: $1" ;;
  esac
done

command -v aws >/dev/null     || fail "The AWS CLI is required."
command -v python3 >/dev/null || fail "python3 is required."
[[ -f "${HELPER}" ]]          || fail "Helper not found at ${HELPER}"
[[ -n "${STACK_NAME}" ]]      || fail "--stack is required (or set STACK_NAME)."

# Every call into the helper goes through this, so the module is imported exactly
# one way and a path problem surfaces once.
helper() { PYTHONPATH="${SCRIPT_DIR}/scripts" python3 -c "$1" "${@:2}"; }

TABLE="$(helper '
import sys, artf_attach as A
print(A.resolve_table(sys.argv[1], sys.argv[2] or None))
' "${STACK_NAME}" "${TABLE_OVERRIDE}")" || fail "Could not resolve the registry table."

# ---------------------------------------------------------------------------
# Detach
# ---------------------------------------------------------------------------
if [[ -n "${DETACH}" ]]; then
  head1 "Detaching '${DETACH}'"
  log "Table: ${TABLE}"

  # A code-defined container has no registry row, so there is nothing to delete
  # and the attempt would silently succeed. Refuse by name instead.
  for builtin in dlrm-bid-shader widedeep-segment-activator ncf-deal-manager \
                 metrics-enricher yield-optimizer-floor yield-optimizer-margin; do
    if [[ "${DETACH}" == "${builtin}" ]]; then
      fail "'${DETACH}' is built into the orchestrator. It has no registry record and cannot be detached."
    fi
  done

  EXISTING="$(aws dynamodb get-item --table-name "${TABLE}" --region "${AWS_REGION}" \
    --key "{\"registry\":{\"S\":\"artf-containers\"},\"name\":{\"S\":\"${DETACH}\"}}" \
    --query 'Item.name.S' --output text 2>/dev/null || echo "")"
  if [[ -z "${EXISTING}" || "${EXISTING}" == "None" ]]; then
    fail "No registry record named '${DETACH}' in ${TABLE}. Nothing to detach."
  fi

  if [[ "${ASSUME_YES}" -eq 0 ]]; then
    printf '\n  This deletes the registry record for %s, then its workload.\n' "${DETACH}"
    printf '  Continue? [y/N] '
    read -r reply
    [[ "${reply}" == "y" || "${reply}" == "Y" ]] || fail "Aborted."
  fi

  # ORDER MATTERS. The record goes first: a live record pointing at a deleted
  # Service reports unreachable on every request, which reads as a broken
  # container rather than a removed one.
  log "Deleting the registry record first, so nothing routes to a vanishing Service..."
  aws dynamodb delete-item --table-name "${TABLE}" --region "${AWS_REGION}" \
    --key "{\"registry\":{\"S\":\"artf-containers\"},\"name\":{\"S\":\"${DETACH}\"}}" \
    || fail "Could not delete the registry record. Workload left untouched."
  log "Record deleted."

  if command -v kubectl >/dev/null; then
    NS_ARG=()
    [[ -n "${NAMESPACE}" ]] && NS_ARG=(-n "${NAMESPACE}")
    log "Deleting workload objects named '${DETACH}' (if present)..."
    kubectl delete "deployment/${DETACH}" "${NS_ARG[@]}" --ignore-not-found >/dev/null 2>&1 || true
    kubectl delete "service/${DETACH}"    "${NS_ARG[@]}" --ignore-not-found >/dev/null 2>&1 || true
    kubectl delete "hpa/${DETACH}"        "${NS_ARG[@]}" --ignore-not-found >/dev/null 2>&1 || true
    log "Workload objects removed."
  else
    warn "kubectl not found. Delete the workload yourself: kubectl delete deployment,service,hpa ${DETACH}"
  fi

  log "The copied image is left in ECR. Remove it separately if you want it gone."
  head1 "Detached."
  exit 0
fi

# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------
[[ -n "${RECORD}" ]]   || fail "--record is required. It is the artf-registry-record.json the container's own deploy step produced."
[[ -f "${RECORD}" ]]   || fail "No record file at ${RECORD}"
[[ "${VARIANT}" == "cpu" || "${VARIANT}" == "gpu" ]] || fail "--variant must be cpu or gpu."

head1 "1. Reading the record"

read -r NAME DISPLAY IMAGE DIGEST REC_NS ENDPOINT < <(helper '
import sys, artf_attach as A
c = A.parse_record(open(sys.argv[1]).read())
print(c.name, c.display_name.replace(" ", "_"), c.image, c.image_digest or "-", c.namespace, c.endpoint)
' "${RECORD}") || fail "The record could not be read. See the message above."

INTENTS="$(helper '
import sys, artf_attach as A
print(",".join(A.parse_record(open(sys.argv[1]).read()).intents))
' "${RECORD}")"

log "Container : ${NAME}"
log "Intents   : ${INTENTS}"
log "Image     : ${IMAGE}"
log "Digest    : ${DIGEST}"
log "Table     : ${TABLE}"

[[ -n "${NAMESPACE}" ]] || NAMESPACE="${REC_NS}"
log "Namespace : ${NAMESPACE}"

head1 "2. Enforcing a cluster-internal endpoint"

ENDPOINT="$(helper '
import sys, artf_attach as A
c = A.parse_record(open(sys.argv[1]).read())
check = A.check_endpoint(c.endpoint)
if not check.ok:
    sys.exit("REFUSED: " + check.reason)
print(A.retarget_endpoint(c.endpoint, sys.argv[2]))
' "${RECORD}" "${NAMESPACE}")" || fail "The endpoint is not acceptable. ARTF containers must be reachable only inside the cluster."
log "Endpoint  : ${ENDPOINT}"

SERVICE="$(helper '
import sys, artf_attach as A
print(A.service_name(sys.argv[1]))
' "${ENDPOINT}")"

if command -v kubectl >/dev/null; then
  if kubectl get "service/${SERVICE}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    log "Service '${SERVICE}' exists in namespace '${NAMESPACE}'."
  else
    warn "Service '${SERVICE}' does not exist in namespace '${NAMESPACE}' yet."
    warn "That is expected if the workload is not deployed. The record is written"
    warn "inactive, so nothing routes to it until you deploy and activate."
  fi
else
  warn "kubectl not found, so the Service could not be verified."
fi

head1 "3. Checking the Prebid hook's intent ceiling"

CONFIG_BUCKET="$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='ConfigBucketName'].OutputValue" \
  --output text 2>/dev/null || echo "")"

CEILING_FILE=""
if [[ -n "${CONFIG_BUCKET}" && "${CONFIG_BUCKET}" != "None" ]]; then
  CEILING_FILE="$(mktemp)"
  if aws s3 cp "s3://${CONFIG_BUCKET}/${PREBID_CONFIG_KEY}" "${CEILING_FILE}" \
       --region "${AWS_REGION}" --quiet 2>/dev/null; then
    log "Read the live Prebid config overlay."
  else
    rm -f "${CEILING_FILE}"; CEILING_FILE=""
    log "No Prebid config overlay found — the Prebid variant may not be deployed."
  fi
else
  log "No config bucket output on the stack — skipping the ceiling check."
fi

helper '
import re, sys, artf_attach as A
intents = sys.argv[1].split(",")
path = sys.argv[2]
configured = None
if path:
    text = open(path).read()
    block = re.search(r"^\s*intents:\s*$((?:\s*-\s*\w+\s*$)+)", text, re.M)
    if block:
        configured = re.findall(r"-\s*(\w+)", block.group(1))
check = A.check_prebid_ceiling(intents, configured)
if configured is None:
    print("  The ceiling could not be read, so it is unknown rather than assumed.")
elif check.ok:
    print("  All intents are in the configured ceiling: " + ", ".join(check.covered))
else:
    print()
    print(A.ceiling_remedy(check.missing, "'"${PREBID_CONFIG_KEY}"'"))
for u in check.unknown:
    print("  NOTE: " + u + " is not a request-side intent, so the hook cannot apply it at")
    print("        processed-auction-request even if it is listed.")
' "${INTENTS}" "${CEILING_FILE}"
[[ -n "${CEILING_FILE}" ]] && rm -f "${CEILING_FILE}"

head1 "4. Checking for an intent already claimed"

EXISTING_JSON="$(aws dynamodb query --table-name "${TABLE}" --region "${AWS_REGION}" \
  --key-condition-expression 'registry = :r' \
  --expression-attribute-values '{":r":{"S":"artf-containers"}}' \
  --output json 2>/dev/null || echo '{"Items":[]}')"

# The six built-ins are code-defined and always active, so they are not in the
# table and have to be supplied for the contention check to see them.
printf '%s' "${EXISTING_JSON}" | helper '
import json, sys, artf_attach as A
record_path, intents_csv, priority = sys.argv[1], sys.argv[2], int(sys.argv[3])
container = A.parse_record(open(record_path).read())
items = json.load(sys.stdin).get("Items", [])
existing = [A.item_to_record(i) | {"source": "store"} for i in items]
existing += [
    {"name": n, "active": True, "intents": [i], "source": "code", "priority": 0}
    for n, i in [
        ("dlrm-bid-shader", "BID_SHADE"),
        ("widedeep-segment-activator", "ACTIVATE_SEGMENTS"),
        ("ncf-deal-manager", "ACTIVATE_DEALS"),
        ("metrics-enricher", "ADD_METRICS"),
        ("yield-optimizer-floor", "ADJUST_DEAL_FLOOR"),
        ("yield-optimizer-margin", "ADJUST_DEAL_MARGIN"),
    ]
]
warnings = A.describe_contention(container, existing, priority=priority)
if not warnings:
    print("  No intent is claimed by another active container.")
for w in warnings:
    print("  " + w)
' "${RECORD}" "${INTENTS}" "${PRIORITY}"

head1 "5. Copying the image into this stack's ECR"

FINAL_IMAGE="${IMAGE}"
if [[ "${DO_COPY}" -eq 1 ]]; then
  [[ -n "${DEST_REPO}" ]] || DEST_REPO="${STACK_NAME}-external-artf/${NAME}"
  SRC_REPO="${IMAGE%:*}"; SRC_TAG="${IMAGE##*:}"
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
  DEST_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  # The tag is preserved verbatim: the producer encodes both code and data
  # provenance in it (src-<hash>-cache-<hash>), and a retag destroys that.
  FINAL_IMAGE="${DEST_REGISTRY}/${DEST_REPO}:${SRC_TAG}"

  log "Source      : ${IMAGE}"
  log "Destination : ${FINAL_IMAGE}"

  aws ecr describe-repositories --repository-names "${DEST_REPO}" --region "${AWS_REGION}" \
    >/dev/null 2>&1 || {
    log "Creating ECR repository ${DEST_REPO}..."
    aws ecr create-repository --repository-name "${DEST_REPO}" --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true \
      --image-tag-mutability IMMUTABLE >/dev/null \
      || fail "Could not create ${DEST_REPO}."
  }

  BUILDSPEC=$(cat <<SPEC
version: 0.2
phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${DEST_REGISTRY}
      # The SOURCE registry is in another account. This login needs a repository
      # policy on their side granting this account ecr:BatchGetImage and
      # ecr:GetDownloadUrlForLayer. Copying does not remove that requirement --
      # it moves it from every node, continuously, to this one step, once.
      - aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${SRC_REPO%%/*}
  build:
    commands:
      - docker pull ${IMAGE}
      - docker tag ${IMAGE} ${FINAL_IMAGE}
      - docker push ${FINAL_IMAGE}
  post_build:
    commands:
      - |
        PUSHED=\$(aws ecr describe-images --repository-name ${DEST_REPO} \\
          --image-ids imageTag=${SRC_TAG} --region ${AWS_REGION} \\
          --query 'imageDetails[0].imageDigest' --output text)
        echo "Pushed digest: \$PUSHED"
        if [ -n "${DIGEST}" ] && [ "${DIGEST}" != "-" ] && [ "\$PUSHED" != "${DIGEST}" ]; then
          echo "DIGEST MISMATCH: the record says ${DIGEST} but the copy is \$PUSHED"
          exit 1
        fi
        echo "Digest verified against the record."
SPEC
)
  log "Starting the copy in CodeBuild (in-region; a 10.7 GB GPU image over a laptop connection is the wrong shape)..."
  BUILD_ID="$(aws codebuild start-build \
    --project-name "${STACK_NAME}-image-builder" \
    --region "${AWS_REGION}" \
    --buildspec-override "${BUILDSPEC}" \
    --query 'build.id' --output text)" || fail "Could not start the copy build."
  log "Build: ${BUILD_ID}"
  log "Follow it with: aws codebuild batch-get-builds --ids ${BUILD_ID} --region ${AWS_REGION} --query 'builds[0].buildStatus'"
  log ""
  log "If it fails on the source pull, the image OWNER must add a repository policy"
  log "granting this account (${ACCOUNT_ID}) ecr:BatchGetImage and"
  log "ecr:GetDownloadUrlForLayer on ${SRC_REPO}. That permission is in their"
  log "account and cannot be granted from here."
else
  log "--no-copy: registering the foreign image reference directly."
  log "Every node will then need cross-account pull permission on ${IMAGE%:*},"
  log "on every scale-out and every node replacement."
fi

head1 "6. Rendering the manifest"

if [[ -z "${TEMPLATE}" ]]; then
  log "No --template given, so no manifest is rendered."
  log "Pass --template <their artf-container.example.yaml> to render one."
else
  [[ -f "${TEMPLATE}" ]] || fail "No template at ${TEMPLATE}"
  [[ -n "${OUT}" ]] || OUT="${SCRIPT_DIR}/eks/external-${NAME}.yaml"

  if [[ "${VARIANT}" == "gpu" ]]; then
    GPU_INSTANCES="$(helper '
import re, sys
text = open(sys.argv[1]).read()
print(",".join(re.findall(r"g\d+\.\w+|p\d+d?\.\w+", text)) or "unknown")
' "${SCRIPT_DIR}/eks/cluster-config.yaml" 2>/dev/null || echo unknown)"
    helper '
import sys, artf_attach as A
c = A.parse_record(open(sys.argv[1]).read())
ok, reason = A.gpu_guard(c, instance_types=sys.argv[2].split(","), image_exists=False)
sys.exit("REFUSED: " + reason)
' "${RECORD}" "${GPU_INSTANCES}" || true
    fail "The GPU variant is refused. See the reason above. Only the CPU image is published today; the GPU path is untested and cannot be verified from here."
  fi

  helper '
import sys, artf_attach as A
rendered = A.render_manifest(
    open(sys.argv[1]).read(), image=sys.argv[2], namespace=sys.argv[3]
)
open(sys.argv[4], "w").write(rendered)
' "${TEMPLATE}" "${FINAL_IMAGE}" "${NAMESPACE}" "${OUT}" || fail "Could not render the manifest."
  log "Rendered: ${OUT}"
  log "Review it before applying. Rendering and applying are separate on purpose."

  if [[ "${DO_APPLY}" -eq 1 ]]; then
    command -v kubectl >/dev/null || fail "--apply needs kubectl."
    log "Applying..."
    kubectl apply -f "${OUT}" || fail "kubectl apply failed. No record written."
    log "Applied."
  fi
fi

head1 "7. Probing POST /mutate"

# NOT /health/ready. The shared server answers readiness with an unconditional
# {"status":"ok"} even with nothing loaded, so a health probe here is a check
# that cannot fail. Probed from inside the cluster because the endpoint is
# cluster-internal by requirement.
if command -v kubectl >/dev/null && kubectl get "deployment/${ORCHESTRATOR_DEPLOY}" >/dev/null 2>&1; then
  PROBE_JSON="$(helper '
import json, sys, artf_attach as A
print(json.dumps(A.probe_payload(A.parse_record(open(sys.argv[1]).read()))))
' "${RECORD}")"
  POD="$(kubectl get pods -l app="${ORCHESTRATOR_DEPLOY}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")"
  if [[ -n "${POD}" ]]; then
    RESULT="$(kubectl exec "${POD}" -- python3 -c "
import json, urllib.request
req = urllib.request.Request('${ENDPOINT}/mutate', data=json.dumps(${PROBE_JSON}).encode(),
                             headers={'Content-Type': 'application/json'})
try:
    print(urllib.request.urlopen(req, timeout=5).read().decode())
except Exception as exc:
    print(json.dumps({'__error__': f'{type(exc).__name__}: {exc}'}))
" 2>/dev/null || echo '{"__error__":"could not exec into the orchestrator pod"}')"
    helper '
import json, sys, artf_attach as A
raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception:
    print("  " + A.summarise_probe(None, error="the response was not JSON")); raise SystemExit
if isinstance(data, dict) and "__error__" in data:
    print("  " + A.summarise_probe(None, error=data["__error__"]))
else:
    print("  " + A.summarise_probe(data))
' "${RESULT}"
  else
    warn "No orchestrator pod found, so the container could not be probed from inside the cluster."
  fi
else
  warn "kubectl or the orchestrator deployment is unavailable, so no probe was made."
  warn "A failed or skipped probe does not block registration."
fi

head1 "8. Writing the registry record"

ITEM="$(helper '
import json, sys, artf_attach as A
c = A.parse_record(open(sys.argv[1]).read())
print(json.dumps(A.build_item(
    c, endpoint=sys.argv[2], priority=int(sys.argv[3]),
    updated_at=sys.argv[4], updated_by=sys.argv[5],
)))
' "${RECORD}" "${ENDPOINT}" "${PRIORITY}" \
  "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" \
  "$(aws sts get-caller-identity --query Arn --output text 2>/dev/null || echo unknown)")"

if aws dynamodb put-item \
     --table-name "${TABLE}" --region "${AWS_REGION}" \
     --item "${ITEM}" \
     --condition-expression 'attribute_not_exists(#n)' \
     --expression-attribute-names '{"#n":"name"}' >/dev/null 2>&1; then
  log "Record written, active=false, priority=${PRIORITY}."
else
  # Not an error. The row may have been activated since it was first written, and
  # replacing it would silently switch the container off.
  warn "A record named '${NAME}' already exists and was LEFT UNTOUCHED."
  warn "It may have been activated since it was written; overwriting it would"
  warn "switch the container off without saying so. Delete it first if you"
  warn "genuinely want to re-register: ./attach_artf_container.sh --detach ${NAME} --stack ${STACK_NAME}"
fi

head1 "Done — and what happens next"
cat <<NEXT
  The container is registered but INACTIVE. Nothing routes to it yet.

  Activate it from the Container Health panel, or:
    aws dynamodb update-item --table-name ${TABLE} --region ${AWS_REGION} \\
      --key '{"registry":{"S":"artf-containers"},"name":{"S":"${NAME}"}}' \\
      --update-expression 'SET active = :a' \\
      --expression-attribute-values '{":a":{"BOOL":true}}'

  Precedence: it is registered at priority ${PRIORITY}. At equal priority,
  registry order decides and store containers follow built-in ones -- so it
  WINS against a built-in claiming the same intent, and the built-in's mutation
  is computed and then discarded. Set a negative priority to keep the
  built-in's value:

    --priority -1

  Teardown:
    ./attach_artf_container.sh --detach ${NAME} --stack ${STACK_NAME}
NEXT
