# =============================================================================
# deploy_gate.sh — decide what to do by asking AWS, not by reading a local file.
#
# Wraps scripts/deploy_status.py for deploy.sh. One probe answers all five phases,
# and each phase then does one of three things:
#
#   ok           -> skip it, and say why it was skipped
#   in_progress  -> WAIT for the AWS-side operation to finish, then re-probe
#   anything else-> run it
#
# This replaces the local progress file that recorded what the *script* had done.
# The difference is not academic: that file reported "Phase 3 complete" for a
# deployment whose Triton pod had never started, and it could not be read from
# another machine, after a reboot, or by a colleague.
#
# Because the probe only reads AWS, any number of terminals can run this at the
# same time. There is no process to attach to, nothing to lock, and no way for two
# invocations to deadlock waiting on each other.
#
# Usage:
#   STATE_PYTHON="${PYTHON}"
#   source "${SCRIPT_DIR}/lib/deploy_gate.sh"
#   gate_refresh "${STACK_PREFIX}"
#   if gate_should_run 2; then ... ; fi
# =============================================================================

_GATE_PYTHON="${STATE_PYTHON:-python3}"
_GATE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_GATE_HELPER="${_GATE_LIB_DIR}/../scripts/deploy_status.py"

# Seconds between polls while waiting for an AWS-side operation. CloudFormation,
# CodeBuild and EKS all move on the order of minutes; polling faster only burns API
# calls.
GATE_POLL="${GATE_POLL:-20}"
# Give up waiting eventually, and say so, rather than blocking forever on something
# that is genuinely stuck.
GATE_WAIT_LIMIT="${GATE_WAIT_LIMIT:-3600}"

# Populated by gate_refresh.
DEPLOY_PHASE_1=unknown; DEPLOY_PHASE_2=unknown; DEPLOY_PHASE_3=unknown
DEPLOY_PHASE_4=unknown; DEPLOY_PHASE_5=unknown
DEPLOY_BLOCKER_1=''; DEPLOY_BLOCKER_2=''; DEPLOY_BLOCKER_3=''
DEPLOY_BLOCKER_4=''; DEPLOY_BLOCKER_5=''
DEPLOY_OVERALL=unknown
DEPLOY_KUBE_CONTEXT=''
GATE_AVAILABLE=0

# The probe's flags are derived from this run's own options, because what counts as
# "complete" depends on them: --no-retraining means the closed-loop stacks and the
# two reasoning agents are not expected to exist at all.
_gate_flags() {
  local out=()
  [[ "${WITH_RETRAINING:-1}" -eq 1 ]] || out+=(--no-retraining)
  [[ "${WITH_PREBID:-0}" -eq 1 ]] && out+=(--with-prebid)
  [[ "${SKIP_AGENTCORE:-0}" -eq 0 ]] || out+=(--skip-agentcore)
  # deploy.sh resolves and exports AWS_PROFILE before sourcing this file. The probe
  # would inherit it anyway; passing it explicitly keeps the probe's credentials
  # visible in the command line and identical to every other child's.
  [[ -n "${AWS_PROFILE:-}" ]] && out+=(--profile "${AWS_PROFILE}")
  printf '%s\n' "${out[@]+"${out[@]}"}"
}

_gate_available() {
  [[ -f "${_GATE_HELPER}" ]] && command -v "${_GATE_PYTHON}" >/dev/null 2>&1
}

# gate_refresh <prefix> — re-probe AWS and update the DEPLOY_* variables.
#
# Returns 0 even when the probe cannot run. A probe failure must degrade to
# "unknown" (and therefore "do the work") rather than abort a deployment.
gate_refresh() {
  local prefix="${1:-}" line flags=()
  if ! _gate_available; then
    GATE_AVAILABLE=0
    return 0
  fi
  while IFS= read -r line; do [[ -n "${line}" ]] && flags+=("${line}"); done < <(_gate_flags)

  # Only DEPLOY_*= assignments are eval'd. A stray warning on stdout then cannot be
  # executed as shell -- the difference between a degraded probe and a surprise.
  local evaluated=0
  while IFS= read -r line; do
    if [[ "${line}" =~ ^DEPLOY_[A-Z0-9_]+= ]]; then
      eval "${line}"
      evaluated=1
    fi
  done < <("${_GATE_PYTHON}" "${_GATE_HELPER}" \
             --prefix "${prefix}" --region "${AWS_REGION:-us-east-1}" --shell \
             "${flags[@]+"${flags[@]}"}" 2>/dev/null || true)
  GATE_AVAILABLE="${evaluated}"
  return 0
}

# gate_print <prefix> — the human-readable status table.
#
# ALWAYS returns 0, and sets GATE_PRINT_RC to the probe's real exit code (0 when
# everything expected is present, 1 otherwise). Two callers with opposite needs:
# mid-deploy this is informational and must not trip `set -e`, while `--status`
# wants the code so it can be used in a script.
GATE_PRINT_RC=0
gate_print() {
  local prefix="${1:-}" flags=() line
  _gate_available || { printf '  (status probe unavailable)\n'; GATE_PRINT_RC=0; return 0; }
  while IFS= read -r line; do [[ -n "${line}" ]] && flags+=("${line}"); done < <(_gate_flags)
  set +e
  "${_GATE_PYTHON}" "${_GATE_HELPER}" \
    --prefix "${prefix}" --region "${AWS_REGION:-us-east-1}" \
    "${flags[@]+"${flags[@]}"}" 2>/dev/null
  GATE_PRINT_RC=$?
  set -e
  return 0
}

# gate_status <1-5> — echo the probed status for a phase.
gate_status() {
  case "$1" in
    1) printf '%s' "${DEPLOY_PHASE_1}" ;;
    2) printf '%s' "${DEPLOY_PHASE_2}" ;;
    3) printf '%s' "${DEPLOY_PHASE_3}" ;;
    4) printf '%s' "${DEPLOY_PHASE_4}" ;;
    5) printf '%s' "${DEPLOY_PHASE_5}" ;;
    *) printf 'unknown' ;;
  esac
}

# gate_blocker <1-5> — echo the first non-OK check for a phase, or nothing.
gate_blocker() {
  case "$1" in
    1) printf '%s' "${DEPLOY_BLOCKER_1}" ;;
    2) printf '%s' "${DEPLOY_BLOCKER_2}" ;;
    3) printf '%s' "${DEPLOY_BLOCKER_3}" ;;
    4) printf '%s' "${DEPLOY_BLOCKER_4}" ;;
    5) printf '%s' "${DEPLOY_BLOCKER_5}" ;;
    *) printf '' ;;
  esac
}

# gate_should_run <1-5> — true when this phase has work to do.
#
# `unknown` runs the phase. Every phase is idempotent, so re-running one we cannot
# verify costs time; SKIPPING one we cannot verify costs a broken deployment that
# claims success.
gate_should_run() {
  local n="$1" st
  st="$(gate_status "${n}")"
  [[ "${GATE_AVAILABLE}" -eq 1 ]] || return 0
  [[ "${st}" != "ok" ]]
}

# gate_wait <1-5> <prefix> — block while a phase is mid-flight in AWS.
#
# This is the piece that makes a second terminal useful instead of destructive: if
# CloudFormation is already creating a stack, or CodeBuild is already building, the
# right move is to wait for it, not to start a competing operation.
#
# MUST be called as a bare statement, never as `$(gate_wait ...)`. It re-probes via
# gate_refresh, which assigns the DEPLOY_* globals, and a command substitution runs
# in a subshell where every one of those assignments is thrown away -- leaving the
# caller believing the phase is still in progress after the wait succeeded.
gate_wait() {
  local n="$1" prefix="${2:-}" waited=0 st
  st="$(gate_status "${n}")"
  [[ "${st}" == "in_progress" ]] || return 0
  say "  Phase ${n}/5 is already running in AWS — waiting for it rather than starting another."
  say "    $(gate_blocker "${n}")"
  while [[ "${st}" == "in_progress" ]]; do
    if [[ "${waited}" -ge "${GATE_WAIT_LIMIT}" ]]; then
      warn "Phase ${n}/5 has been in progress for $(( waited / 60 ))m; not waiting longer."
      warn "Check it with: ./deploy.sh --prefix ${prefix} --status"
      return 1
    fi
    sleep "${GATE_POLL}"
    waited=$(( waited + GATE_POLL ))
    gate_refresh "${prefix}"
    st="$(gate_status "${n}")"
    say "    still working ($(( waited / 60 ))m) — $(gate_blocker "${n}")"
  done
  say "  Phase ${n}/5 finished in AWS (now: ${st})."
  return 0
}
