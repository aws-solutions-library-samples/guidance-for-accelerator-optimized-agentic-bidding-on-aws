# =============================================================================
# deploy_progress.sh — status marks, step narration, and an in-place spinner.
#
# Three jobs:
#
#   1. MARKS. One pair of glyphs for "this exists" and "this does not", used by both
#      the deploy output and `--status`, so the same thing looks the same
#      everywhere. Check and cross, NOT emoji: they are one column wide in every
#      terminal, they survive copy/paste into a ticket, and they do not depend on a
#      font that happens to ship colour pictographs.
#
#   2. step(). Always-visible "you are here" narration. Distinct from log(), which
#      is --verbose only. Losing the step markers was a real complaint: the phase
#      header appeared, then minutes of silence, and nothing said which of a
#      phase's eleven steps was running.
#
#   3. run_logged(). Run a noisy command with its output in a FILE and a spinner on
#      screen, then one line saying what happened. The AgentCore deploy printed
#      dozens of lines of INFO logging into the middle of an otherwise quiet
#      deployment; that detail belongs in a log, like every other phase's.
#
# Failures are never hidden by any of this: run_logged tails the log inline when a
# command fails, and names the file. Quiet output must not mean unexplained output.
# =============================================================================

# --- marks -------------------------------------------------------------------
# UTF-8 when the locale supports it, ASCII when it does not. A mark that renders as
# a replacement box is worse than 'x'.
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
  *UTF-8*|*utf-8*|*UTF8*|*utf8*)
    MARK_OK="✓"
    MARK_BAD="✗"
    MARK_WAIT="·"
    MARK_UNKNOWN="?"
    ;;
  *)
    MARK_OK="+"
    MARK_BAD="x"
    MARK_WAIT="."
    MARK_UNKNOWN="?"
    ;;
esac

# Colour only when writing to a terminal, so a redirected log stays clean text.
if [[ -t 1 ]]; then
  _C_OK=$'\033[0;32m'; _C_BAD=$'\033[0;31m'; _C_DIM=$'\033[0;90m'; _C_OFF=$'\033[0m'
else
  _C_OK=""; _C_BAD=""; _C_DIM=""; _C_OFF=""
fi

# --- step narration ----------------------------------------------------------

# step <text> — always-visible marker for entering a step within a phase.
# Indented under the phase header so the hierarchy is readable at a glance.
step() { printf '  %s%s%s %s\n' "${_C_DIM}" "->" "${_C_OFF}" "$*"; }

# done_ok <text> / done_bad <text> — a completed step, with its mark.
done_ok()  { printf '  %s%s%s %s\n' "${_C_OK}"  "${MARK_OK}"  "${_C_OFF}" "$*"; }
done_bad() { printf '  %s%s%s %s\n' "${_C_BAD}" "${MARK_BAD}" "${_C_OFF}" "$*"; }

# --- spinner -----------------------------------------------------------------

# Braille dots animate smoothly and are a single column; the ASCII set is the
# classic four-frame spin for terminals without UTF-8.
case "${MARK_OK}" in
  "✓") _SPIN_FRAMES=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏) ;;
  *)   _SPIN_FRAMES=('|' '/' '-' '\') ;;
esac
SPIN_INTERVAL="${SPIN_INTERVAL:-0.12}"

_SPIN_PID=""
_SPIN_MSG=""
# Whatever EXIT/INT/TERM handlers the caller had, captured as re-evaluable `trap`
# commands. deploy.sh has none today, but a spinner must not be the reason a
# cleanup handler someone adds later silently stops running.
_SPIN_PREV_TRAPS=""
# Only true when THIS file installed a trap. Without it, the non-terminal path --
# which installs nothing, and is the usual path under nohup or a tee'd pipeline --
# would still run `trap -` on the way out and delete the caller's handler.
_SPIN_TRAPPED=0

# True only when an animation makes sense: a real terminal. Under nohup, in CI, or
# with stdout redirected, a spinner would write thousands of \r frames into a file.
_spin_possible() { [[ -t 1 ]] && [[ "${DEPLOY_NO_SPINNER:-0}" -ne 1 ]]; }

_spin_loop() {
  local msg="$1" i=0 n=${#_SPIN_FRAMES[@]}
  while :; do
    printf '\r  %s%s%s %s' "${_C_DIM}" "${_SPIN_FRAMES[$(( i % n ))]}" "${_C_OFF}" "${msg}"
    i=$(( i + 1 ))
    sleep "${SPIN_INTERVAL}"
  done
}

# spin_start <message> — begin animating. No-op (with one static line) when stdout
# is not a terminal, so redirected output reads normally.
spin_start() {
  _SPIN_MSG="$1"
  if ! _spin_possible; then
    step "${_SPIN_MSG}"
    return 0
  fi
  command -v tput >/dev/null 2>&1 && tput civis 2>/dev/null || true
  _spin_loop "${_SPIN_MSG}" &
  _SPIN_PID=$!
  # Never leave a spinner running or the cursor hidden if the deploy dies. `trap -p`
  # prints the current handlers as commands that can be eval'd back, which is how
  # spin_stop restores rather than discards them.
  _SPIN_PREV_TRAPS="$(trap -p EXIT INT TERM)"
  trap _spin_cleanup EXIT INT TERM
  _SPIN_TRAPPED=1
  return 0
}

# stop_bg <pid> — kill a background job AND reap it.
#
# The `wait` is the point. Without it bash keeps the job in its table, notices it
# died from a signal, and at its next opportunity prints the job-control notice to
# stderr -- the pid, "Terminated: 15", and the ENTIRE body of the subshell:
#
#   ./deploy.sh: line 1998: 14035 Terminated: 15  ( while kill -0 "${_cluster_pid}"
#       ...
#
# It surfaces wherever bash next checks jobs, which is nowhere near the code that
# started it, so it reads as a crash in an unrelated step. `wait` reaps the job
# before that notification can happen. Errors are ignored: the job is already gone
# in the common case, and failing to kill a heartbeat must never fail a deploy.
stop_bg() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  return 0
}

_spin_cleanup() {
  # No spinner was ever started, so there is nothing to erase and no cursor to
  # restore. Returning early matters: `tput cnorm` writes an escape sequence to
  # stdout, and emitting it on the non-terminal path would put control characters
  # into a redirected log that is supposed to be plain text.
  [[ -n "${_SPIN_PID}" ]] || return 0
  kill "${_SPIN_PID}" 2>/dev/null || true
  wait "${_SPIN_PID}" 2>/dev/null || true
  _SPIN_PID=""
  printf '\r\033[K'
  command -v tput >/dev/null 2>&1 && tput cnorm 2>/dev/null || true
  return 0
}

# spin_stop <ok|fail> [message] — stop animating and print the outcome line.
spin_stop() {
  local result="${1:-ok}" msg="${2:-${_SPIN_MSG}}"
  _spin_cleanup
  # Restore only what we replaced. Touching traps we never installed is how a
  # caller's cleanup handler disappears.
  if [[ "${_SPIN_TRAPPED}" -eq 1 ]]; then
    trap - EXIT INT TERM
    if [[ -n "${_SPIN_PREV_TRAPS}" ]]; then
      eval "${_SPIN_PREV_TRAPS}"
    fi
    _SPIN_PREV_TRAPS=""
    _SPIN_TRAPPED=0
  fi
  case "${result}" in
    ok) done_ok "${msg}" ;;
    *)  done_bad "${msg}" ;;
  esac
  return 0
}

# --- run a noisy command quietly --------------------------------------------

# run_logged <logfile> <label> <command...>
#
# Output goes to the log; the screen gets a spinner and one result line. Returns
# the COMMAND's exit status, so callers keep their own error handling.
#
# On failure the tail of the log is printed inline. Sending detail to a file is
# only an improvement if a failure still explains itself without the user having to
# know which file to open.
run_logged() {
  local log="$1" label="$2"; shift 2
  local rc=0
  mkdir -p "$(dirname "${log}")" 2>/dev/null || true
  {
    printf '\n===== %s =====\n' "${label}"
    printf '===== %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >> "${log}" 2>/dev/null || true

  spin_start "${label}"
  # Remember whether the CALLER had errexit on. Restoring it unconditionally with
  # `set -e` would switch it on for callers that never had it, and the next failing
  # top-level command would then abort the script with no output.
  local _ee=0; case "$-" in *e*) _ee=1 ;; esac
  set +e
  "$@" >> "${log}" 2>&1
  rc=$?
  if [[ "${_ee}" -eq 1 ]]; then set -e; fi

  if [[ "${rc}" -eq 0 ]]; then
    spin_stop ok "${label}"
  else
    spin_stop fail "${label} — exit ${rc}"
    printf '    %slast 15 lines of %s:%s\n' "${_C_DIM}" "${log}" "${_C_OFF}" >&2
    tail -15 "${log}" 2>/dev/null | sed 's/^/      /' >&2
  fi
  return "${rc}"
}

# run_logged_capture <var> <logfile> <label> <command...>
#
# For commands whose STDOUT is a value the caller needs (an ARN, an ID) while their
# stderr is noise. stdout is captured into <var>, stderr goes to the log.
#
# `eval` on a caller-supplied variable NAME (not a value) is how bash 3.2 does an
# out-parameter; there is no `declare -n` before bash 4.3, and macOS ships 3.2.
run_logged_capture() {
  local __var="$1" log="$2" label="$3"; shift 3
  local __out rc=0
  mkdir -p "$(dirname "${log}")" 2>/dev/null || true
  {
    printf '\n===== %s =====\n' "${label}"
    printf '===== %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >> "${log}" 2>/dev/null || true

  spin_start "${label}"
  local _ee=0; case "$-" in *e*) _ee=1 ;; esac
  set +e
  __out="$("$@" 2>> "${log}")"
  rc=$?
  if [[ "${_ee}" -eq 1 ]]; then set -e; fi

  if [[ "${rc}" -eq 0 ]]; then
    spin_stop ok "${label}"
  else
    spin_stop fail "${label} — exit ${rc}"
    printf '    %slast 15 lines of %s:%s\n' "${_C_DIM}" "${log}" "${_C_OFF}" >&2
    tail -15 "${log}" 2>/dev/null | sed 's/^/      /' >&2
  fi
  eval "${__var}=\"\${__out}\""
  return "${rc}"
}
