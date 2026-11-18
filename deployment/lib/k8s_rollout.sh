#!/usr/bin/env bash
# lib/k8s_rollout.sh -- wait for a Deployment to roll out, and say why when it does not.
#
# Sourced by deploy.sh and deploy_prebid.sh. bash 3.2 and `set -u` safe.
#
#   ROLLOUT_KUBECTL=(kubectl --context X)   optional; defaults to plain `kubectl`
#   wait_rollout <deployment> <namespace or ""> <timeout-seconds>
#       0 when `kubectl rollout status` reports the rollout complete. Otherwise
#       prints, for every pod of the Deployment that is not Ready, its phase, the
#       reason it is not running (CrashLoopBackOff, ImagePullBackOff, ErrImagePull,
#       CreateContainerConfigError, Unschedulable with the scheduler's message) and
#       its last log lines, sets ROLLOUT_FAILURE to a one-line summary, and returns 1.
#   pinned_image_ref <registry> <repo> <digest>
#       <registry>/<repo>@<digest>; returns 1 unless <digest> is sha256:<64 hex>.
#
# Callers decide what a failure means; both scripts treat it as a failed deploy.

ROLLOUT_FAILURE=""
ROLLOUT_LOG_LINES="${ROLLOUT_LOG_LINES:-20}"

_rollout_kubectl() {
  if [[ -n "${ROLLOUT_KUBECTL+x}" && ${#ROLLOUT_KUBECTL[@]} -gt 0 ]]; then
    "${ROLLOUT_KUBECTL[@]}" "$@"
  else
    kubectl "$@"
  fi
}

pinned_image_ref() {
  local registry="$1" repo="$2" digest="$3"
  [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || return 1
  printf '%s/%s@%s\n' "${registry}" "${repo}" "${digest}"
}

# _pod_diagnosis <pods-json-file>: one line per not-Ready pod,
# "<pod><TAB><reason><TAB><detail><TAB><container-or-empty><TAB><restarts>".
_pod_diagnosis() {
  python3 - "$1" <<'PY'
import json, sys
try:
    pods = json.load(open(sys.argv[1])).get("items", [])
except Exception:
    pods = []
for pod in pods:
    meta, status = pod.get("metadata", {}), pod.get("status", {})
    if meta.get("deletionTimestamp"):
        continue
    conds = {c.get("type"): c for c in status.get("conditions") or []}
    if conds.get("Ready", {}).get("status") == "True":
        continue
    name = meta.get("name", "?")
    sched = conds.get("PodScheduled", {})
    if sched.get("status") == "False":
        print(f"{name}\t{sched.get('reason') or 'Pending'}\t{(sched.get('message') or '').strip()}\t\t0")
        continue
    reason, detail, cname, restarts = status.get("phase") or "Unknown", "", "", 0
    for cs in (status.get("initContainerStatuses") or []) + (status.get("containerStatuses") or []):
        st = cs.get("state") or {}
        waiting, term = st.get("waiting"), st.get("terminated")
        last = (cs.get("lastState") or {}).get("terminated") or {}
        if waiting:
            reason, detail = waiting.get("reason") or "Waiting", (waiting.get("message") or "").strip()
        elif term:
            reason, detail = term.get("reason") or "Terminated", f"exit {term.get('exitCode')}"
        elif not cs.get("ready"):
            reason, detail = "NotReady", "running but failing its readiness probe"
        else:
            continue
        if last and not detail:
            detail = f"last exit {last.get('exitCode')} ({last.get('reason')})"
        cname, restarts = cs.get("name", ""), cs.get("restartCount", 0)
        break
    print(f"{name}\t{reason}\t{detail}\t{cname}\t{restarts}")
PY
}

# resolve_ecr_digest <repo> <tag> <region>: the manifest digest ECR holds for
# <repo>:<tag> (DescribeImages imageDetails[].imageDigest), or return 1.
resolve_ecr_digest() {
  local digest
  digest="$(aws ecr describe-images --repository-name "$1" --image-ids "imageTag=$2" \
    --region "$3" --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true)"
  [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || return 1
  printf '%s\n' "${digest}"
}

# pin_image_in_manifest <file> <placeholder> <registry> <repo> <tag> <region>:
# replaces <placeholder> in <file> (in place) with <registry>/<repo>@<digest> for the
# digest ECR holds under <tag>. A tag can be pushed again with new content, and a pod
# spec that names the tag with imagePullPolicy IfNotPresent keeps the old image; a
# digest changes whenever the content does, so every rebuild rolls out. Sets
# PINNED_IMAGE_REF. Returns 1 when the digest cannot be read.
PINNED_IMAGE_REF=""
pin_image_in_manifest() {
  local file="$1" placeholder="$2" registry="$3" repo="$4" tag="$5" region="$6" digest ref
  PINNED_IMAGE_REF=""
  digest="$(resolve_ecr_digest "${repo}" "${tag}" "${region}")" || return 1
  ref="$(pinned_image_ref "${registry}" "${repo}" "${digest}")" || return 1
  sed -e "s|${placeholder}|${ref}|g" "${file}" >"${file}.pinned" && mv "${file}.pinned" "${file}"
  PINNED_IMAGE_REF="${ref}"
}

wait_rollout() {
  local name="$1" ns="$2" timeout="$3"
  local nsargs=() selector tmp line pod reason detail cname restarts summary=""
  [[ -n "${ns}" ]] && nsargs=(-n "${ns}")
  ROLLOUT_FAILURE=""
  if _rollout_kubectl rollout status "deployment/${name}" ${nsargs[@]+"${nsargs[@]}"} \
       --timeout="${timeout}s" >/dev/null 2>&1; then
    printf '  rollout ok: deployment/%s\n' "${name}"
    return 0
  fi
  printf '  rollout FAILED: deployment/%s did not complete within %ss\n' "${name}" "${timeout}"
  selector="$(_rollout_kubectl get "deployment/${name}" ${nsargs[@]+"${nsargs[@]}"} -o json 2>/dev/null \
    | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
print(",".join(f"{k}={v}" for k,v in sorted((d.get("spec",{}).get("selector",{}).get("matchLabels") or {}).items())))' 2>/dev/null)"
  if [[ -z "${selector}" ]]; then
    ROLLOUT_FAILURE="deployment/${name}: not found or unreadable"
    printf '    %s\n' "${ROLLOUT_FAILURE}"
    return 1
  fi
  tmp="$(mktemp "${TMPDIR:-/tmp}/rollout.XXXXXX")"
  _rollout_kubectl get pods ${nsargs[@]+"${nsargs[@]}"} -l "${selector}" -o json >"${tmp}" 2>/dev/null || true
  while IFS="$(printf '\t')" read -r pod reason detail cname restarts; do
    [[ -n "${pod}" ]] || continue
    printf '    pod %s: %s%s (restarts %s)\n' "${pod}" "${reason}" "${detail:+: ${detail}}" "${restarts:-0}"
    summary="${summary:+${summary}; }${pod} ${reason}"
    case "${reason}" in
      Unschedulable|Pending|ImagePullBackOff|ErrImagePull|CreateContainerConfigError) ;;
      *)
        printf '    last %s log lines of %s:\n' "${ROLLOUT_LOG_LINES}" "${pod}"
        { _rollout_kubectl logs "${pod}" ${nsargs[@]+"${nsargs[@]}"} ${cname:+-c "${cname}"} \
            --previous --tail="${ROLLOUT_LOG_LINES}" 2>/dev/null \
          || _rollout_kubectl logs "${pod}" ${nsargs[@]+"${nsargs[@]}"} ${cname:+-c "${cname}"} \
            --tail="${ROLLOUT_LOG_LINES}" 2>/dev/null; } | sed 's/^/      | /'
        ;;
    esac
  done <<EOF
$(_pod_diagnosis "${tmp}")
EOF
  rm -f "${tmp}"
  ROLLOUT_FAILURE="deployment/${name}: ${summary:-rollout incomplete, no failing pod reported}"
  return 1
}
