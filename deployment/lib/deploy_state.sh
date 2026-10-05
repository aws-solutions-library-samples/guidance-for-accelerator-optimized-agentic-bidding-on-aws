# =============================================================================
# deploy_state.sh — remembered INPUTS for deploy.sh. Not progress.
#
# Sourced by deploy.sh. Phase progress is deliberately NOT recorded here: a file
# records what the *script* did, which is not what *exists* (it would say "Phase 3
# complete" for a deployment whose Triton pod never started), and it cannot be read
# from another machine or by a colleague. Progress comes from AWS -- see
# scripts/deploy_status.py and lib/deploy_gate.sh.
#
# What is left is the part AWS genuinely cannot answer: the flags a previous run
# was GIVEN. The NGC secret name is the case that justifies the file existing at
# all -- pass --ngc-key once and later runs find the stored secret on their own.
#
# Two rules every function here obeys:
#
#   1. ALWAYS return 0. Callers run under `set -euo pipefail`; a non-zero return
#      from a state read would end a 30-minute deployment over a cache miss.
#   2. NEVER call fail(). This is a convenience cache, never a requirement.
#
# Usage:
#   STATE_PYTHON="${PYTHON}"                       # optional; defaults to python3
#   source "${SCRIPT_DIR}/lib/deploy_state.sh"
# =============================================================================

# Resolved once. deploy.sh runs Python through its own virtualenv (see its
# preflight), so it passes that interpreter in; the tests fall back to python3.
_STATE_PYTHON="${STATE_PYTHON:-python3}"
_STATE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_STATE_HELPER="${_STATE_LIB_DIR}/../scripts/deploy_state.py"
# Overridable so the test harness can point at a temp file.
DEPLOY_STATE_FILE="${DEPLOY_STATE_FILE:-${_STATE_LIB_DIR}/../.deploy-state.json}"

# True when the helper is usable at all. Checked per call rather than cached, so
# a helper deleted mid-run degrades instead of erroring.
_state_available() {
  [[ -f "${_STATE_HELPER}" ]] && command -v "${_STATE_PYTHON}" >/dev/null 2>&1
}

# Every command funnels through here: one place that adds --file/--prefix,
# swallows failure, and guarantees exit 0.
_state() {
  _state_available || return 0
  "${_STATE_PYTHON}" "${_STATE_HELPER}" \
    --file "${DEPLOY_STATE_FILE}" \
    --prefix "${1:-}" \
    "${@:2}" 2>/dev/null || true
  return 0
}

# --- Reads -------------------------------------------------------------------

# state_read <prefix> <dotted.field>  -> prints the value, or nothing
state_read() { _state "$1" read --field "$2"; }

# --- Writes ------------------------------------------------------------------

# state_set <prefix> <remembered|resolved> key=value [key=value ...]
#
# Empty values are passed through deliberately: the helper's set_values() refuses
# to overwrite a stored non-empty value with an empty one, which is what lets a
# run that skipped a phase leave that phase's resolved values intact.
state_set() {
  local prefix="$1" section="$2"; shift 2
  local args=() kv
  for kv in "$@"; do args+=(--kv "${kv}"); done
  [[ ${#args[@]} -gt 0 ]] || return 0
  _state "${prefix}" set --section "${section}" "${args[@]}"
}

# state_start_run <prefix> <account> <region> <stack> <cluster> [argv...]
#
# Records this invocation's identity and (redacted) argv, and prints a
# comma-separated list of identity fields that DISAGREE with the stored record --
# a different account or STACK_NAME under the same prefix key. Remembered values
# are not reused across such a conflict, because doing so would silently point a
# run at the wrong environment.
state_start_run() {
  local prefix="$1" account="$2" region="$3" stack="$4" cluster="$5"; shift 5
  _state "${prefix}" start-run \
    --account "${account}" --region "${region}" \
    --stack "${stack}" --cluster "${cluster}" \
    --argv "$@"
}

# state_clear <prefix>
state_clear() { _state "$1" clear; }
