#!/usr/bin/env bash
# =============================================================================
# test_deploy_phases.sh — static/mocked verification for deploy.sh's phased
# output, --start-at renumbering, Phase-2 concurrency, and the non-blocking
# model-optimizer bootstrap watcher (FR-6..FR-9, NFR-6).
#
# This harness does NOT make real AWS/eksctl/kubectl/docker calls. It mocks
# each of those commands as shell functions exported on PATH, sources the
# relevant pieces of deploy.sh's logic in isolation, and asserts on captured
# output / exit codes / files written. Per the workspace `testing` steering,
# this is the verification ceiling achievable without a live AWS account —
# a real `./deploy.sh` run is the final confirmation step (see tasks.md
# Group 11) and is NOT claimed as done by this test file.
#
# Usage: bash deployment/tests/test_deploy_phases.sh
# Exit code 0 = all assertions passed. Non-zero = at least one failure
# (printed with a FAIL line identifying which assertion).
# =============================================================================

set -uo pipefail

# NOTE ON ERREXIT: this suite deliberately runs WITHOUT `set -e` (above), because
# tests routinely run commands expected to fail and then assert on the status.
# Several tests wrap such a command in `set +e` ... `set -e` -- and that trailing
# `set -e` used to ENABLE errexit rather than restore it, since it was never on.
# The effect was invisible until a test had a top-level command legitimately
# return non-zero (`wait` on a killed process, status 143): the whole suite then
# exited silently at that line, reporting no results at all. Restores are `set +e`
# for that reason. If you add one, restore with `set +e`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SH="${SCRIPT_DIR}/../deploy.sh"

PASS_COUNT=0
FAIL_COUNT=0

assert_contains() {
  local haystack="$1" needle="$2" label="$3"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: %s — expected to find %q\n' "${label}" "${needle}" >&2
    printf '  --- actual output (last 20 lines) ---\n' >&2
    printf '%s\n' "${haystack}" | tail -20 >&2
  fi
}

assert_not_contains() {
  local haystack="$1" needle="$2" label="$3"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: %s — did NOT expect to find %q\n' "${label}" "${needle}" >&2
  fi
}

assert_eq() {
  local actual="$1" expected="$2" label="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: %s — expected %q, got %q\n' "${label}" "${expected}" "${actual}" >&2
  fi
}

# run_limited SECONDS OUTFILE COMMAND... — run a command under a wall-clock limit
# and return its exit code, or 124 if it had to be killed.
#
# Hand-rolled rather than using timeout(1): that is GNU coreutils and is NOT
# present on a stock macOS, which is a supported platform for these scripts (the
# same reason this codebase avoids bash-4 features). Without a limit, a test that
# is meant to prove "this does not hang" would itself hang forever when it fails,
# which is the least useful possible outcome.
run_limited() {
  local secs="$1" outfile="$2"; shift 2
  local pid elapsed=0
  "$@" >"${outfile}" 2>&1 &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    if [[ "${elapsed}" -ge "${secs}" ]]; then
      kill -9 "${pid}" 2>/dev/null
      wait "${pid}" 2>/dev/null
      return 124
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  wait "${pid}"
  return $?
}

# =========================================================================
# Test 1: --start-at rejects values outside 1-5 (FR-7)
# =========================================================================
test_start_at_validation() {
  local rc

  # Exercise the exact regex deploy.sh uses to validate --start-at (FR-7):
  # in-range values 1-5 must pass, everything else (including the entire old
  # ad-hoc step-number range) must fail.
  for val in 0 6 8 11 abc ""; do
    if [[ "${val}" =~ ^[1-5]$ ]]; then rc=0; else rc=1; fi
    assert_eq "${rc}" "1" "start-at regex rejects out-of-range value '${val}'"
  done

  for val in 1 2 3 4 5; do
    if [[ "${val}" =~ ^[1-5]$ ]]; then rc=0; else rc=1; fi
    assert_eq "${rc}" "0" "start-at regex accepts in-range value '${val}'"
  done
}

# =========================================================================
# Test 2: display_name() maps old container keys to new job-oriented names,
# and passes through unknown keys unchanged (FR-4)
# =========================================================================
test_display_name() {
  # Extract and source just the display_name() function definition from
  # deploy.sh so this test exercises the real implementation, not a copy.
  local fn_src
  fn_src="$(sed -n '/^display_name() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"

  assert_eq "$(display_name dlrm-bid-shader)" "bid-pricer" "display_name dlrm-bid-shader"
  assert_eq "$(display_name widedeep-segment-activator)" "audience-activator" "display_name widedeep-segment-activator"
  assert_eq "$(display_name ncf-deal-manager)" "deal-scorer" "display_name ncf-deal-manager"
  assert_eq "$(display_name metrics-enricher)" "signals-enricher" "display_name metrics-enricher"
  assert_eq "$(display_name orchestrator)" "orchestrator" "display_name passes through unknown key unchanged"
}

# =========================================================================
# Test 3: phase()/ok()/log()/warn()/fail()/say() output gating (FR-10, FR-11)
# =========================================================================
test_output_gating() {
  local fn_src out

  # The emitters now route through _journal()/_emit() (so a follower can tail their
  # output), so those have to come along too — extracted from deploy.sh rather than
  # reimplemented here, which is the point of this test: it asserts on the real
  # definitions, not on a copy that could drift.
  fn_src="$(sed -n '/^_journal() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  fn_src="$(sed -n '/^_emit() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  # Journaling off for this test: it asserts on what reaches the console.
  DEPLOY_JOURNAL_FILE=""

  fn_src="$(sed -n '/^say() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  fn_src="$(sed -n '/^log()  {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  fn_src="$(sed -n '/^warn() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  fn_src="$(sed -n '/^phase() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  fn_src="$(sed -n '/^ok() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"

  # Default (VERBOSE=0): log() must be silent, warn()/phase()/ok()/say() visible.
  VERBOSE=0
  out="$(log "hidden detail" 2>&1)"
  assert_eq "${out}" "" "log() prints nothing when VERBOSE=0"

  out="$(warn "a warning" 2>&1)"
  assert_contains "${out}" "a warning" "warn() always prints"

  out="$(phase 2 "Test Phase" 2>&1)"
  assert_contains "${out}" "Phase 2/5: Test Phase" "phase() prints phase header"

  out="$(ok "did the thing" 2>&1)"
  assert_contains "${out}" "did the thing" "ok() always prints"
  assert_contains "${out}" "[OK]" "ok() includes the OK marker"

  # --verbose (VERBOSE=1): log() becomes visible too.
  VERBOSE=1
  out="$(log "now visible" 2>&1)"
  assert_contains "${out}" "now visible" "log() prints when VERBOSE=1"
}

# =========================================================================
# Test 4: fail() with a phase number prints the phase + remediation hint,
# without suppressing the real error message (FR-11)
# =========================================================================
test_fail_with_hint() {
  local fn_src out rc

  fn_src="$(sed -n '/^phase_hint() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"
  fn_src="$(sed -n '/^fail() {/,/^}/p' "${DEPLOY_SH}")"
  eval "${fn_src}"

  out="$(fail 2 "cluster creation failed" 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "fail() exits non-zero"
  assert_contains "${out}" "Phase 2/5" "fail() with phase number includes the phase label"
  assert_contains "${out}" "cluster creation failed" "fail() never suppresses the real error text"
  assert_contains "${out}" "NGC" "fail() includes Phase 2's remediation hint"

  # Single-arg form (existing call sites) must still work unchanged.
  out="$(fail "plain error, no phase" 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "fail() single-arg form still exits non-zero"
  assert_contains "${out}" "plain error, no phase" "fail() single-arg form prints the message"
  assert_not_contains "${out}" "Phase" "fail() single-arg form does not fabricate a phase label"
}

# =========================================================================
# Test 5: Phase-2 concurrency exit-code propagation (FR-8) — the core
# job-control correctness this feature adds. Mocks a backgrounded function
# that fails, and confirms `wait` surfaces that failure (this is exactly the
# pattern deploy.sh uses for `( ensure_eks_cluster ) & ... wait "$pid"`).
# =========================================================================
test_concurrency_exit_code_propagation() {
  local rc

  # Success case: backgrounded function exits 0, wait must report 0.
  ( exit 0 ) &
  local pid_ok=$!
  wait "${pid_ok}"; rc=$?
  assert_eq "${rc}" "0" "wait() propagates a successful backgrounded exit code"

  # Failure case: backgrounded function exits 1, wait must report non-zero
  # (this is the exact mechanism deploy.sh relies on to catch a failed
  # `eksctl create cluster` running in the background instead of silently
  # continuing to Step 6+ against a cluster that doesn't exist).
  ( exit 1 ) &
  local pid_fail=$!
  wait "${pid_fail}"; rc=$?
  assert_eq "${rc}" "1" "wait() propagates a failed backgrounded exit code"
}

# =========================================================================
# Test 6: model-optimizer bootstrap watcher writes the expected status file
# shape on completion and on timeout/failure (FR-9), using a mocked
# `kubectl` so no real cluster is contacted.
# =========================================================================
test_bootstrap_watcher() {
  local tmpdir status_file

  tmpdir="$(mktemp -d)"
  status_file="${tmpdir}/.bootstrap-status.json"

  # Mock `kubectl wait` succeeding immediately.
  kubectl() { return 0; }
  export -f kubectl

  (
    if kubectl wait --for=condition=complete job/model-optimizer-bootstrap --timeout=900s >/dev/null 2>&1; then
      printf '{"status":"complete","finishedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
    else
      printf '{"status":"timeout_or_failed","checkedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
    fi
  )
  local content
  content="$(cat "${status_file}" 2>/dev/null || echo '')"
  assert_contains "${content}" '"status":"complete"' "bootstrap watcher writes complete status on kubectl success"

  # Mock `kubectl wait` failing (simulating a timeout).
  rm -f "${status_file}"
  kubectl() { return 1; }
  export -f kubectl

  (
    if kubectl wait --for=condition=complete job/model-optimizer-bootstrap --timeout=900s >/dev/null 2>&1; then
      printf '{"status":"complete","finishedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
    else
      printf '{"status":"timeout_or_failed","checkedAt":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_file}"
    fi
  )
  content="$(cat "${status_file}" 2>/dev/null || echo '')"
  assert_contains "${content}" '"status":"timeout_or_failed"' "bootstrap watcher writes timeout_or_failed status on kubectl failure"

  rm -rf "${tmpdir}"
  unset -f kubectl
}

# =========================================================================
# Test 7: the NON-INTERACTIVE remote_build.sh paths.
#
# This is the regression test for the defect the deploy-state feature exists to
# fix. remote_build.sh read `</dev/tty` at two prompts; with no controlling
# terminal the REDIRECT itself fails, `read` returns non-zero, and `set -euo
# pipefail` exits the script. deploy_closed_loop.sh calls it in a command
# substitution under its own `set -e`, so it died at Step 4b -- leaving the
# vpc-proxy and governance-eventbridge stacks uncreated while deploy.sh printed
# its success banner.
#
# Runs the REAL script with a mocked `aws` and stdin closed. Two things are
# asserted that a code read cannot establish: that it does not HANG, and that it
# exits non-zero with a message naming the remedy.
# =========================================================================
test_remote_build_non_interactive() {
  local tmpdir out rc
  tmpdir="$(mktemp -d)"

  # Mock `aws`: resolve an account, and report NO existing NGC secret (which is
  # what makes the NGC prompt reachable).
  cat > "${tmpdir}/aws" <<'MOCK'
#!/usr/bin/env bash
case "$*" in
  *"sts get-caller-identity"*) echo "123456789012" ;;
  *"secretsmanager describe-secret"*) exit 1 ;;
  *"codebuild list-builds-for-project"*) echo "None" ;;
  *) echo "None" ;;
esac
exit 0
MOCK
  chmod +x "${tmpdir}/aws"

  # --target nemo is what sets _BUILDS_NEMO=1 and so makes the prompt reachable.
  # stdin from /dev/null: not a terminal, which is exactly the detached case.
  set +e
  PATH="${tmpdir}:${PATH}" run_limited 60 "${tmpdir}/run1.txt" \
    bash "${SCRIPT_DIR}/../codebuild/remote_build.sh" \
    --stack-name teststack --target nemo --tag latest --region us-east-1 --no-wait \
    </dev/null
  rc=$?
  set +e
  out="$(cat "${tmpdir}/run1.txt" 2>/dev/null || echo '')"

  # 124 means run_limited had to kill it -- i.e. it hung, which is precisely the
  # failure mode this test exists to catch.
  if [[ "${rc}" -eq 124 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: remote_build.sh HUNG with no terminal attached\n' >&2
  else
    PASS_COUNT=$((PASS_COUNT + 1))
  fi
  if [[ "${rc}" -eq 0 || "${rc}" -eq 124 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: remote_build.sh should exit non-zero when NGC credentials are missing non-interactively (got %s)\n' "${rc}" >&2
  else
    PASS_COUNT=$((PASS_COUNT + 1))
  fi
  assert_contains "${out}" "NGC credentials are required" \
    "remote_build.sh fails with an actionable NGC message instead of a failed tty read"
  assert_contains "${out}" "--ngc-key" \
    "the non-interactive NGC failure names the flag that fixes it"
  assert_not_contains "${out}" "Continue without NGC credentials?" \
    "the interactive prompt is not emitted when there is no terminal"

  # With a secret already stored for this stack, the prompt is never reached at
  # all -- the reuse path at remote_build.sh:155-165 resolves it silently. This is
  # what makes a re-run that omits --ngc-key work.
  cat > "${tmpdir}/aws" <<'MOCK'
#!/usr/bin/env bash
case "$*" in
  *"sts get-caller-identity"*) echo "123456789012" ;;
  *"secretsmanager describe-secret"*) echo '{"Name":"teststack-ngc-api-key"}' ;;
  *) echo "None" ;;
esac
exit 0
MOCK
  chmod +x "${tmpdir}/aws"
  # --no-wait so it exits after starting the (mocked) build rather than entering
  # the 15-second poll loop, which a mock can never satisfy.
  set +e
  PATH="${tmpdir}:${PATH}" run_limited 90 "${tmpdir}/run2.txt" \
    bash "${SCRIPT_DIR}/../codebuild/remote_build.sh" \
    --stack-name teststack --target nemo --tag latest --region us-east-1 --no-wait \
    </dev/null
  rc=$?
  set +e
  out="$(cat "${tmpdir}/run2.txt" 2>/dev/null || echo '')"
  assert_not_contains "${out}" "NGC credentials are required" \
    "an existing NGC secret is reused with no prompt and no failure"
  assert_contains "${out}" "reusing, no prompt" \
    "the reuse path is taken, which is what makes a re-run without --ngc-key work"
  if [[ "${rc}" -eq 124 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: remote_build.sh HUNG on the NGC-secret-present path\n' >&2
  else
    PASS_COUNT=$((PASS_COUNT + 1))
  fi

  rm -rf "${tmpdir}"
}

# =========================================================================
# Test 10: a degraded run must not print the success banner, and must exit
# non-zero. This is the reporting half of the detach defect: the closed-loop
# failure was swallowed into two warnings and the deploy claimed success.
# =========================================================================
test_degraded_summary_suppresses_success() {
  local out rc

  # Mirrors deploy.sh's summary branch.
  _summary() {
    local degraded="$1"
    if [[ "${degraded}" -ne 0 ]]; then
      echo "  Accelerator-optimized Agentic Bidding — PARTIALLY DEPLOYED"
      echo "  To finish it:"
      echo "    ./deploy.sh --resume"
      return 1
    fi
    echo "  Accelerator-optimized Agentic Bidding — Deployed (EKS + Triton)"
    return 0
  }

  set +e
  out="$(_summary 1)"; rc=$?
  set +e
  assert_contains "${out}" "PARTIALLY DEPLOYED" "a degraded run reports partial completion"
  assert_not_contains "${out}" "— Deployed (EKS + Triton)" "a degraded run does NOT print the success banner"
  assert_contains "${out}" "--resume" "the partial summary names the command that finishes the deploy"
  assert_eq "${rc}" "1" "a degraded run exits non-zero"

  set +e
  out="$(_summary 0)"; rc=$?
  set +e
  assert_contains "${out}" "— Deployed (EKS + Triton)" "a clean run still prints the success banner"
  assert_eq "${rc}" "0" "a clean run exits zero"
}

# =========================================================================
# Test 11: the credential block prints a fully-resolved reset command, and never
# a command with an empty --user-pool-id (which would fail on paste).
# =========================================================================
test_demo_credentials_block() {
  local out

  _creds() {
    local status="$1" pool="$2" region="us-east-1" email="admin@example.com"
    case "${status}" in
      created)  echo "    Username:  ${email}"; echo "    Password:  TEMPPASS123x" ;;
      existing) echo "    Username:  ${email}"; echo "    Password:  (existing user — Cognito does not disclose it; reset it below)" ;;
      *)        echo "    (Cognito auth not configured — orchestrator auth is disabled)" ;;
    esac
    if [[ -n "${pool}" ]]; then
      echo "    Lost it?   aws cognito-idp admin-set-user-password \\"
      echo "                 --user-pool-id ${pool} \\"
    elif [[ "${status}" != "no-auth" ]]; then
      echo "    Reset:     the user pool ID could not be resolved"
      echo "                 aws cognito-idp list-user-pools --max-results 60 --region ${region}"
    fi
  }

  out="$(_creds created us-east-1_AbC123)"
  assert_contains "${out}" "admin-set-user-password" "the reset command prints for a newly created user"
  assert_contains "${out}" "us-east-1_AbC123" "the reset command carries the real pool ID"

  out="$(_creds existing us-east-1_AbC123)"
  assert_contains "${out}" "admin-set-user-password" "the reset command prints for an existing user too"
  assert_contains "${out}" "does not disclose" "an existing user's password is stated as unrecoverable, not omitted"

  out="$(_creds existing "")"
  assert_not_contains "${out}" "--user-pool-id " "no reset command is printed with an empty pool ID"
  assert_contains "${out}" "list-user-pools" "an unresolved pool ID points at how to find it"
}

# =========================================================================
# Test 10: the AWS gate decides which phases run
# =========================================================================
# Replaces the old resume/state-progress tests. Those asserted on a local file's
# idea of progress, which is the thing that turned out to be untrustworthy -- it
# reported a phase complete for a deployment whose pod had never started.
#
# No AWS calls here: the probe is stubbed so the DECISION logic is what gets
# tested. Whether the probe reads AWS correctly is a separate concern, verified
# against a real account.
test_gate_decisions() {
  local tmp out
  tmp="$(mktemp -d)"

  # A stub standing in for scripts/deploy_status.py, emitting the same --shell
  # contract: DEPLOY_PHASE_n=<status> plus DEPLOY_OVERALL.
  mkdir -p "${tmp}/scripts" "${tmp}/lib"
  cp "${SCRIPT_DIR}/../lib/deploy_gate.sh" "${tmp}/lib/"
  cat > "${tmp}/scripts/deploy_status.py" <<'STUB'
import os, sys
statuses = os.environ.get("STUB_STATUSES", "ok,ok,ok,ok,ok").split(",")
for i, s in enumerate(statuses, 1):
    print("DEPLOY_PHASE_%d=%s" % (i, s))
    print("DEPLOY_BLOCKER_%d='because %s'" % (i, s))
print("DEPLOY_OVERALL=%s" % ("ok" if set(statuses) == {"ok"} else "missing"))
print("DEPLOY_KUBE_CONTEXT='stub'")
sys.exit(0 if set(statuses) == {"ok"} else 1)
STUB

  _gate_env() {
    AWS_REGION=us-east-1; WITH_RETRAINING=1; WITH_PREBID=0; SKIP_AGENTCORE=0
    STACK_PREFIX=t; START_AT=1; _GIVEN_START_AT=0
    say()  { printf '%s\n' "$*"; }
    warn() { printf '[warn] %s\n' "$*"; }
    ok()   { printf '  [OK] %s\n' "$*"; }
    STATE_PYTHON=python3
    GATE_POLL=1
    # shellcheck source=/dev/null
    source "${tmp}/lib/deploy_gate.sh"
    # _run_phase and force_phase, lifted from deploy.sh so the real precedence is
    # what is tested.
    _IMAGES_REBUILT=0
    _AGENTCORE_IMAGE_REBUILT=0
    FORCE_PHASE_1=0; FORCE_PHASE_1_WHY=""
    FORCE_PHASE_3=0; FORCE_PHASE_3_WHY=""
    FORCE_PHASE_5=0; FORCE_PHASE_5_WHY=""
    eval "$(sed -n '/^_run_phase() {/,/^}/p; /^force_phase() {/,/^}/p' "${DEPLOY_SH}")"
  }
  _gate_env
  assert_eq "$(type -t _run_phase)" "function" "the real _run_phase is lifted from deploy.sh"
  assert_eq "$(type -t force_phase)" "function" "the real force_phase is lifted from deploy.sh"

  # --- everything already deployed -> every phase skips -------------------
  STUB_STATUSES="ok,ok,ok,ok,ok" gate_refresh t
  assert_eq "${GATE_AVAILABLE}" "1" "gate reads the probe output"
  assert_eq "${DEPLOY_OVERALL}" "ok" "a fully deployed stack reports overall ok"
  local ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "" "no phase runs when AWS says everything exists"

  # --- partially deployed -> only the incomplete phases run ---------------
  STUB_STATUSES="ok,missing,ok,failed,missing" gate_refresh t
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "245" "only phases AWS reports incomplete are run"

  # --- unknown must RUN, never skip ---------------------------------------
  # Skipping something unverifiable ships a deployment that claims to be complete
  # and is not. Re-running an idempotent phase only costs time.
  STUB_STATUSES="unknown,unknown,unknown,unknown,unknown" gate_refresh t
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "12345" "an unknown phase is run, not skipped"

  # --- --start-at forces exactly the phase it names -----------------------
  # The header documents --start-at as the override "for when you know something
  # the probe cannot". A healthy cluster used to make it a no-op. The named phase
  # runs; the later ones still consult AWS, so a forced Phase 3 does not drag the
  # frontend and agents along with it.
  STUB_STATUSES="ok,ok,ok,ok,ok" gate_refresh t
  START_AT=3; _GIVEN_START_AT=1
  ran=""
  for n in 1 2 3 4 5; do if _run_phase "$n" >/dev/null 2>&1; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "3" "--start-at 3 forces Phase 3 and only Phase 3 on a complete stack"
  _run_phase 3 > "${tmp}/startat.out" 2>&1 || true
  assert_contains "$(cat "${tmp}/startat.out")" "because --start-at 3 names it" "the forced phase says why it is running"
  # The default START_AT=1 with no flag given is not a force: a complete stack still skips.
  START_AT=1; _GIVEN_START_AT=0
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "" "the implicit START_AT=1 forces nothing"

  # --- a rebuilt image forces the rollout phase -----------------------------
  # The probe reads what is RUNNING; pods on an image rebuilt under the same tag
  # read as ok. Nine images were once rebuilt and every phase skipped, so the old
  # orchestrator kept serving. Phase 3 carries the rollout restart and must run.
  STUB_STATUSES="ok,ok,ok,ok,ok" gate_refresh t
  _IMAGES_REBUILT=1
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "3" "a rebuilt image forces Phase 3 (rollout) and nothing else"
  _run_phase 3 > "${tmp}/rebuilt.out" 2>&1 || true
  assert_contains "$(cat "${tmp}/rebuilt.out")" "rebuilt images under the same tag" "the forced rollout says why it is running"
  _AGENTCORE_IMAGE_REBUILT=1
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "35" "a rebuilt AgentCore image also forces Phase 5 (runtime update)"
  _IMAGES_REBUILT=0
  _AGENTCORE_IMAGE_REBUILT=0

  # --- a stale phase (the probe's own verdict) runs -------------------------
  # A changed deals seed or Deal Scorer fingerprint reads as `stale`: present, but
  # built from other inputs. It is not ok, so the phase runs.
  STUB_STATUSES="stale,ok,stale,ok,ok" gate_refresh t
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "13" "phases the probe reads as stale run, the ok ones skip"

  # --- force_phase: the read-only checks before Phase 1 ----------------------
  STUB_STATUSES="ok,ok,ok,ok,ok" gate_refresh t
  force_phase 3 "2 Deployment(s) do not run ECR's image for tag abc (bid-pricer deal-scorer)"
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "3" "FORCE_PHASE_3 forces Phase 3 and nothing else on a complete stack"
  _run_phase 3 > "${tmp}/force3.out" 2>&1 || true
  assert_contains "$(cat "${tmp}/force3.out")" "do not run ECR's image for tag abc" "the forced Phase 3 names the stale Deployments"
  force_phase 5 "the MCP runtime was registered with AgentCore image 111111111111"
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "35" "FORCE_PHASE_5 forces Phase 5 as well"
  _run_phase 5 > "${tmp}/force5.out" 2>&1 || true
  assert_contains "$(cat "${tmp}/force5.out")" "registered with AgentCore image 111111111111" "the forced Phase 5 says why"
  force_phase 1 "the local Deal Scorer fingerprint could not be computed"
  force_phase 1 "the local deals seed could not be hashed"
  ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "135" "FORCE_PHASE_1 forces Phase 1"
  _run_phase 1 > "${tmp}/force1.out" 2>&1 || true
  assert_contains "$(cat "${tmp}/force1.out")" "could not be computed; the local deals seed could not be hashed" "a second force_phase reason is appended to the first"
  # --start-at still wins: a forced phase before START_AT does not run.
  START_AT=4; ran=""
  for n in 1 2 3 4 5; do if _gate_run_quiet "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "5" "--start-at 4 skips forced Phases 1 and 3, runs forced Phase 5"
  START_AT=1
  # An operator's own FORCE_PHASE_3=1 (exported, no reason given) is honoured too.
  FORCE_PHASE_1=0; FORCE_PHASE_5=0; FORCE_PHASE_3=1; FORCE_PHASE_3_WHY=""
  _run_phase 3 > "${tmp}/force3env.out" 2>&1; local rc3=$?
  assert_eq "${rc3}" "0" "FORCE_PHASE_3=1 without a reason still forces Phase 3"
  assert_contains "$(cat "${tmp}/force3env.out")" "FORCE_PHASE_3=1 is set" "the hand-set force says so"
  FORCE_PHASE_3=0; FORCE_PHASE_3_WHY=""

  # --- a probe that cannot run -> everything runs -------------------------
  rm -f "${tmp}/scripts/deploy_status.py"
  gate_refresh t
  assert_eq "${GATE_AVAILABLE}" "0" "a missing probe is detected"
  ran=""
  for n in 1 2 3 4 5; do if gate_should_run "$n"; then ran="${ran}${n}"; fi; done
  assert_eq "${ran}" "12345" "an unavailable probe degrades to running every phase"

  # --- in_progress is waited on, not raced --------------------------------
  # gate_wait must poll until the status changes. The stub flips after 2 polls.
  cat > "${tmp}/scripts/deploy_status.py" <<'STUB2'
import os, sys
c = "/tmp/gate_poll_count"
try:
    n = int(open(c).read().strip())
except Exception:
    n = 0
n += 1
open(c, "w").write(str(n))
st = "in_progress" if n <= 2 else "ok"
for i in range(1, 6):
    print("DEPLOY_PHASE_%d=%s" % (i, st if i == 2 else "ok"))
    print("DEPLOY_BLOCKER_%d=''" % i)
print("DEPLOY_OVERALL=missing")
print("DEPLOY_KUBE_CONTEXT='stub'")
sys.exit(1)
STUB2
  rm -f /tmp/gate_poll_count
  gate_refresh t
  assert_eq "$(gate_status 2)" "in_progress" "an AWS operation in flight reads as in_progress"
  # Output to a FILE, not `$(...)`: gate_wait updates globals via gate_refresh, and a
  # command substitution runs in a subshell where those updates are discarded. The
  # first version of this test did exactly that and reported a polling failure that
  # was purely its own. deploy.sh calls it bare inside `if _run_phase`, which is the
  # current shell, so the real path is unaffected.
  gate_wait 2 t > "${tmp}/wait.out" 2>&1
  out="$(cat "${tmp}/wait.out")"
  assert_eq "$(gate_status 2)" "ok" "gate_wait polls until the AWS operation finishes"
  assert_contains "${out}" "already running in AWS" "waiting says why it is waiting"
  assert_contains "${out}" "rather than starting another" "waiting explains it is not racing"
  rm -f /tmp/gate_poll_count

  rm -rf "${tmp}"
}

# Helper: _run_phase with its output suppressed, for the loops above.
_gate_run_quiet() { _run_phase "$1" >/dev/null 2>&1; }


# =========================================================================
# Test 11: Docker is started when a local build needs it
# =========================================================================
# Phase 5 builds the two AgentCore agent images locally by design. It used to die
# with a raw connect error when the daemon was down, after having already created
# the ECR repository -- leaving a repo with zero images and no agent runtimes, which
# is not a legible symptom. It now starts Docker instead.
#
# `docker` and the launchers are stubbed, so nothing here touches the real daemon.
test_docker_autostart() {
  local tmp out rc
  tmp="$(mktemp -d)"
  local D="${SCRIPT_DIR}/.."

  _dk_run() {   # _dk_run <stubdir> <timeout> ; echoes rc then output
    local stub="$1" limit="$2" o r
    o="$(MARK="${stub}/up" PATH="${stub}:/usr/bin:/bin" D="${D}" DKT="${limit}" bash -c '
      set -uo pipefail
      SCRIPT_DIR="$D"; DOCKER_START_TIMEOUT="$DKT"; DOCKER_POLL=1; VERBOSE=0
      log(){ return 0; }; say(){ printf "%s\n" "$*"; }; ok(){ printf "[OK] %s\n" "$*"; }
      warn(){ :; }; fail(){ printf "[fail] %s\n" "$*" >&2; exit 1; }
      source "$D/lib/deploy_docker.sh"
      docker_ensure_running "build the AgentCore agent images"
      echo REACHED_BUILD
    ' 2>&1)"; r=$?
    printf '%s\n%s' "${r}" "${o}"
  }

  # --- daemon already up: silent, no start attempted ----------------------
  mkdir -p "${tmp}/up"
  printf '#!/bin/sh\nexit 0\n' > "${tmp}/up/docker"; chmod +x "${tmp}/up/docker"
  out="$(_dk_run "${tmp}/up" 6)"; rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  assert_eq "${rc}" "0" "a running daemon needs no intervention"
  assert_contains "${out}" "REACHED_BUILD" "a running daemon proceeds straight to the build"
  assert_not_contains "${out}" "Starting it" "a running daemon is not restarted"

  # --- daemon down but startable: waits, then proceeds --------------------
  mkdir -p "${tmp}/start"
  cat > "${tmp}/start/docker" <<'DKS'
#!/bin/sh
if [ "$1" = "info" ]; then
  [ -f "$MARK" ] && exit 0
  n=0; [ -f "$MARK.n" ] && n=$(cat "$MARK.n")
  n=$((n+1)); echo $n > "$MARK.n"
  [ "$n" -ge 3 ] && touch "$MARK"
  exit 1
fi
exit 0
DKS
  chmod +x "${tmp}/start/docker"
  printf '#!/bin/sh\nexit 0\n' > "${tmp}/start/open"; chmod +x "${tmp}/start/open"
  out="$(_dk_run "${tmp}/start" 20)"; rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  assert_eq "${rc}" "0" "a startable daemon results in success"
  assert_contains "${out}" "Starting it" "the user is told Docker is being started"
  assert_contains "${out}" "build the AgentCore agent images" "the reason Docker is needed is named"
  assert_contains "${out}" "start requested via" "a request is claimed, not a success"
  assert_contains "${out}" "Docker daemon ready" "readiness is reported only once confirmed"
  assert_contains "${out}" "REACHED_BUILD" "the build proceeds after the daemon comes up"

  # --- daemon down and unstartable: honest failure, no build --------------
  mkdir -p "${tmp}/dead"
  printf '#!/bin/sh\n[ "$1" = "info" ] && exit 1\nexit 0\n' > "${tmp}/dead/docker"
  chmod +x "${tmp}/dead/docker"
  for m in open colima rdctl systemctl service sudo; do
    printf '#!/bin/sh\nexit 1\n' > "${tmp}/dead/${m}"; chmod +x "${tmp}/dead/${m}"
  done
  out="$(_dk_run "${tmp}/dead" 4)"; rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  if [[ "${rc}" -ne 0 ]]; then PASS_COUNT=$((PASS_COUNT + 1)); else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: an unstartable daemon was expected to fail the deploy\n' >&2
  fi
  assert_not_contains "${out}" "REACHED_BUILD" "an unstartable daemon does not proceed to the build"
  assert_contains "${out}" "docker info" "the failure says how to check Docker"
  assert_contains "${out}" "skip everything already deployed" "the failure says re-running is safe"

  # --- no docker CLI at all: install guidance -----------------------------
  mkdir -p "${tmp}/nocli"
  out="$(_dk_run "${tmp}/nocli" 4)"; rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  if [[ "${rc}" -ne 0 ]]; then PASS_COUNT=$((PASS_COUNT + 1)); else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf 'FAIL: a missing docker CLI was expected to fail the deploy\n' >&2
  fi
  assert_contains "${out}" "Docker Desktop" "a missing CLI points at how to install one"

  rm -rf "${tmp}"
}

# =========================================================================
# Test: --help prints the WHOLE documentation block
# =========================================================================
# This used to be `sed -n '2,96p'`, so adding a paragraph above line 96 silently
# truncated --help mid-sentence. Rather than pin a length, assert that the first
# and last lines of the in-file doc block both survive to the output.
test_help_is_complete() {
  local out first last
  out="$(bash "${DEPLOY_SH}" --help 2>&1)"

  # The doc block is whatever sits between the first and second `# =====` banner.
  first="$(awk '/^# ={10,}/{n++; next} n==1' "${DEPLOY_SH}" | head -1)"
  last="$(awk '/^# ={10,}/{n++; next} n==1' "${DEPLOY_SH}" | tail -1)"

  assert_contains "${out}" "${first}" "--help includes the first line of the doc block"
  assert_contains "${out}" "${last}" "--help includes the LAST line of the doc block"
  assert_not_contains "${out}" "set -euo pipefail" "--help stops before the code"
}

# =========================================================================
# Test: lib/deploy_progress.sh — quiet output that still explains failures
# =========================================================================
# The AgentCore deploy used to print dozens of INFO lines into the middle of the
# terminal. run_logged moves that into a file. The risk in doing so is obvious:
# quiet output must not become unexplained output, and a wrapper that swallows an
# exit code turns a failed deploy into a successful-looking one. Both are asserted.
test_progress_output() {
  local tmp out rc log
  tmp="$(mktemp -d)"
  log="${tmp}/detail.log"
  local D="${SCRIPT_DIR}/.."

  _pg_run() {   # _pg_run <script-body> ; echoes rc then output
    local body="$1" o r
    o="$(D="${D}" L="${log}" bash -c "
      set -uo pipefail
      source \"\$D/lib/deploy_progress.sh\"
      ${body}
    " 2>&1)"; r=$?
    printf '%s\n%s' "${r}" "${o}"
  }

  # --- success: the command's output is in the FILE, not on screen ---------
  out="$(_pg_run 'run_logged "$L" "Registering MCP runtime" bash -c "for i in 1 2 3; do echo INFO_NOISE_\$i; done"')"
  rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  assert_eq "${rc}" "0" "run_logged returns 0 when the command succeeds"
  assert_not_contains "${out}" "INFO_NOISE_2" "a succeeding command's chatter stays off the screen"
  assert_contains "${out}" "Registering MCP runtime" "the step is still named on screen"
  assert_contains "$(cat "${log}")" "INFO_NOISE_2" "the chatter is in the log file"

  # --- failure: the real exit code survives, and the log is shown inline ---
  # A wrapper that returned 0 here would let the deploy march on to its success
  # banner, which is the exact failure mode the degraded-summary work fixed.
  out="$(_pg_run 'run_logged "$L" "A step that fails" bash -c "echo THE_REAL_REASON; exit 7"')"
  rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  assert_eq "${rc}" "7" "run_logged propagates the command's exit code, not tail's"
  assert_contains "${out}" "THE_REAL_REASON" "a failure tails its log inline"
  assert_contains "${out}" "${log}" "a failure names the log file"

  # --- run_logged_capture: stdout is a value, stderr is noise -------------
  out="$(_pg_run 'run_logged_capture V "$L" "Creating runtime" bash -c "echo CHATTER >&2; echo arn:aws:x:1:runtime/abc"; printf "GOT[%s]\n" "$V"')"
  rc="${out%%$'\n'*}"; out="${out#*$'\n'}"
  assert_eq "${rc}" "0" "run_logged_capture returns the command's status"
  assert_contains "${out}" "GOT[arn:aws:x:1:runtime/abc]" "stdout is captured into the named variable"
  assert_contains "$(cat "${log}")" "CHATTER" "stderr chatter goes to the log"

  # --- no spinner frames when stdout is not a terminal --------------------
  # Under nohup or a tee'd pipeline a spinner would write thousands of \r frames
  # into a file. The test's own stdout is a pipe, so this is the real condition.
  out="$(_pg_run 'run_logged "$L" "Quiet when redirected" true')"
  out="${out#*$'\n'}"
  assert_not_contains "${out}" $'\r' "no carriage returns leak into non-terminal output"
  # `tput civis`/`cnorm` write escape sequences to stdout. A log that is meant to be
  # plain text must not collect cursor-visibility codes either.
  assert_not_contains "${out}" $'\033' "no terminal escape sequences leak into non-terminal output"

  # --- errexit is left exactly as the caller had it -----------------------
  # A `set +e ... set -e` pair that restores unconditionally switches errexit ON
  # for callers that never had it; the next failing top-level command then aborts
  # the script with no output. That bug cost a debugging session in the shell
  # harness, so it is pinned here in both directions.
  out="$(_pg_run 'run_logged "$L" "s" true >/dev/null; case "$-" in *e*) echo EE_ON;; *) echo EE_OFF;; esac')"
  out="${out#*$'\n'}"
  assert_contains "${out}" "EE_OFF" "errexit stays off for a caller that had it off"

  out="$(_pg_run 'set -e; run_logged "$L" "s" true >/dev/null; case "$-" in *e*) echo EE_ON;; *) echo EE_OFF;; esac')"
  out="${out#*$'\n'}"
  assert_contains "${out}" "EE_ON" "errexit stays on for a caller that had it on"

  # --- a pre-existing EXIT trap survives the spinner ----------------------
  # spin_stop used to `trap - EXIT`, which would silently delete a cleanup handler
  # added to deploy.sh later. It now restores what it found.
  out="$(_pg_run 'trap "echo CALLER_CLEANUP" EXIT; run_logged "$L" "s" true >/dev/null')"
  out="${out#*$'\n'}"
  assert_contains "${out}" "CALLER_CLEANUP" "a caller's EXIT trap is restored, not discarded"

  # --- stop_bg kills a background job WITHOUT a job-control notice --------
  # The EKS-cluster heartbeat was killed but never reaped, so bash printed
  # "<pid> Terminated: 15" plus the whole subshell body at its next job check --
  # which landed in the middle of Step 5.5, reading like a crash in a step that had
  # nothing to do with it. Assert both that the job dies and that nothing is said.
  out="$(_pg_run '( while :; do sleep 30; done ) & p=$!; sleep 0.3; stop_bg "$p"; sleep 0.4; kill -0 "$p" 2>/dev/null && echo STILL_ALIVE || echo REAPED')"
  out="${out#*$'\n'}"
  assert_contains "${out}" "REAPED" "stop_bg actually stops the job"
  assert_not_contains "${out}" "Terminated" "stop_bg reaps the job, so bash prints no job-control notice"
  assert_not_contains "${out}" "while :" "the subshell body is not dumped to the terminal"

  # --- the spinner leaves no orphan process ------------------------------
  out="$(_pg_run 'run_logged "$L" "s" sleep 0.2 >/dev/null; jobs -r | wc -l | tr -d " "')"
  out="${out#*$'\n'}"
  assert_contains "${out}" "0" "no background spinner is left running"

  # --- marks are one glyph, and OK/BAD differ ----------------------------
  out="$(_pg_run 'printf "[%s][%s]\n" "$MARK_OK" "$MARK_BAD"')"
  out="${out#*$'\n'}"
  assert_not_contains "${out}" "[][]" "the marks are not empty"
  case "${out}" in
    *'[✓][✗]'*|*'[+][x]'*) PASS_COUNT=$((PASS_COUNT + 1)) ;;
    *) FAIL_COUNT=$((FAIL_COUNT + 1))
       printf 'FAIL: marks are neither the UTF-8 nor the ASCII pair — got %q\n' "${out}" >&2 ;;
  esac

  rm -rf "${tmp}"
}

# =========================================================================
# Test: nodes land in private subnets behind NAT
# =========================================================================
# An account with VPC Block Public Access in block-ingress mode drops the
# kubelet's traffic to the cluster endpoint when the node sits in a public
# subnet, so the node never joins and eksctl's CloudFormation waiter times out
# after 25 minutes with no useful message. Private nodes behind NAT are the fix,
# and this pins both halves of it in the RENDERED config -- the same sed deploy.sh
# runs -- so a stray edit to the template cannot quietly undo it.
test_cluster_config_private_nodes() {
  local tmp rendered
  tmp="$(mktemp -d)"
  rendered="${tmp}/config.yaml"
  sed -e "s/__STACK_NAME__/t-stack/g" -e "s/__REGION__/us-east-1/g" -e "s/__MAX_GPUS__/1/g" \
      "${SCRIPT_DIR}/../eks/cluster-config.yaml" > "${rendered}"

  local n_private n_groups
  n_private="$(grep -c '^[[:space:]]*privateNetworking: true' "${rendered}")"
  # Nodegroups only: `- name:` also appears under addons:, which have no subnets.
  n_groups="$(awk '/^managedNodeGroups:/{g=1; next} /^[^[:space:]]/{g=0} g && /^[[:space:]]*- name: /{c++} END{print c+0}' "${rendered}")"
  assert_eq "${n_private}" "${n_groups}" "every managed nodegroup sets privateNetworking: true (${n_groups} groups)"
  assert_eq "${n_private}" "2" "both nodegroups (gpu-inference, cpu-services) are private"
  assert_contains "$(cat "${rendered}")" "gateway: HighlyAvailable" "NAT is HighlyAvailable (one gateway per AZ), the option chosen in the Express"
  assert_not_contains "$(cat "${rendered}")" "__" "no template placeholder survives rendering"

  rm -rf "${tmp}"
}

# =========================================================================
# Test: a mistyped flag stops the run instead of being dropped
# =========================================================================
# The run that motivated this was started with `—with-prebid` (an em dash, the
# kind a chat client or word processor substitutes for two hyphens). The old loop
# ignored it, so the deploy ran without Prebid and nobody knew until the end. Every
# script a human invokes now rejects anything it does not recognise, and names the
# dash when that is what happened.
test_unknown_args_rejected() {
  local D="${SCRIPT_DIR}/.." out rc
  local em=$'\xe2\x80\x94'

  # Fake credentials so nothing here can reach AWS even if parsing let it through.
  _arg_run() { AWS_ACCESS_KEY_ID=x AWS_SECRET_ACCESS_KEY=y env -u AWS_PROFILE bash "$@" 2>&1; }

  out="$(_arg_run "${D}/deploy.sh" --prefix abc --bogus-flag)"; rc=$?
  assert_eq "${rc}" "1" "deploy.sh exits 1 on an unknown flag"
  assert_contains "${out}" "unknown argument '--bogus-flag'" "deploy.sh names the unknown flag"

  out="$(_arg_run "${D}/deploy.sh" --prefix abc "${em}with-prebid")"; rc=$?
  assert_eq "${rc}" "1" "deploy.sh exits 1 on an em-dash flag"
  assert_contains "${out}" "em/en dash" "deploy.sh says the first character is a dash, not two hyphens"

  out="$(_arg_run "${D}/deploy_prebid.sh" --nope)"; rc=$?
  assert_eq "${rc}" "1" "deploy_prebid.sh exits 1 on an unknown flag"
  assert_contains "${out}" "unknown argument '--nope'" "deploy_prebid.sh names the unknown flag"

  out="$(_arg_run "${D}/deploy_closed_loop.sh" --nope)"; rc=$?
  assert_eq "${rc}" "1" "deploy_closed_loop.sh exits 1 on an unknown flag"
  assert_contains "${out}" "unknown argument '--nope'" "deploy_closed_loop.sh names the unknown flag"

  out="$(_arg_run "${D}/codebuild/remote_build.sh" --nope)"; rc=$?
  assert_eq "${rc}" "1" "remote_build.sh exits 1 on an unknown flag"
  assert_contains "${out}" "unknown argument '--nope'" "remote_build.sh names the unknown flag"

  out="$(_arg_run "${D}/codebuild/remote_build.sh" --tag)"; rc=$?
  assert_eq "${rc}" "1" "remote_build.sh exits 1 when a value-taking flag is last"
  assert_contains "${out}" "--tag requires a value" "remote_build.sh names the flag missing its value"

  out="$(_arg_run "${D}/codebuild/remote_build.sh" "${em}tag" x)"; rc=$?
  assert_eq "${rc}" "1" "remote_build.sh exits 1 on an em-dash flag"
  assert_contains "${out}" "two ASCII hyphens: --tag" "remote_build.sh shows the corrected flag"
}

# =========================================================================
# Test: the UI API path is private (no public ingress anywhere)
# =========================================================================
# An account running VPC Block Public Access in block-ingress mode drops every
# packet to an internet-facing load balancer at the internet gateway, and the
# operator may not be able to change that. The design has NO internet-facing resource
# in the VPC: the orchestrator is a ClusterIP Service and the browser reaches it by
# invoking the <prefix>-ui-api-proxy Lambda, attached to the cluster's private
# subnets. These assertions pin that shape in the manifest, in deploy.sh, and in
# the proxy stack helper, and pin the ABSENCE of the old gate (it must not come
# back: it stopped deploys that now work, and it asked a person to override an
# account security control).
test_private_ui_api_path() {
  local tmp fn_src out rc manifest
  tmp="$(mktemp -d)"
  manifest="${SCRIPT_DIR}/../eks/orchestrator-deployment.yaml"

  # --- the manifest: ClusterIP, never LoadBalancer -------------------------
  local svc
  svc="$(awk '/^kind: Service$/{s=1} s && /^  name: orchestrator$/{n=1} s && n && /^  type: /{print $2; exit}' "${manifest}")"
  assert_eq "${svc}" "ClusterIP" "the orchestrator Service is ClusterIP"
  assert_not_contains "$(cat "${manifest}")" "type: LoadBalancer" "no Service in the orchestrator manifest asks for a load balancer"

  # --- the internal NLB the proxy forwards to -------------------------------
  # A VPC Lambda cannot resolve cluster.local and has no route to a ClusterIP; the
  # first live use of that URL 502'd every UI call. The orchestrator therefore gets
  # an INTERNAL NLB, same shape as triton-internal, and nothing internet-facing.
  local nlb; nlb="${SCRIPT_DIR}/../eks/orchestrator-internal-nlb.yaml"
  [[ -f "${nlb}" ]]; assert_eq "$?" "0" "eks/orchestrator-internal-nlb.yaml exists"
  local nlb_src; nlb_src="$(cat "${nlb}")"
  assert_contains "${nlb_src}" "name: orchestrator-internal" "the Service is named orchestrator-internal"
  assert_contains "${nlb_src}" "type: LoadBalancer" "it is a LoadBalancer Service"
  assert_contains "${nlb_src}" 'service.beta.kubernetes.io/aws-load-balancer-type: "nlb"' "it asks for an NLB"
  assert_contains "${nlb_src}" 'service.beta.kubernetes.io/aws-load-balancer-internal: "true"' "the NLB is internal"
  assert_not_contains "${nlb_src}" "internet-facing" "nothing in it is internet-facing"
  assert_contains "${nlb_src}" "targetPort: 8000" "it targets the orchestrator's HTTP port"
  assert_contains "${nlb_src}" "component: orchestrator" "it selects the orchestrator pods"
  assert_contains "$(cat "${DEPLOY_SH}")" "orchestrator-deployment.yaml orchestrator-internal-nlb.yaml" "the manifest is applied right after the orchestrator"

  # --- deploy.sh: the gate and the ELB wait are gone, the proxy is wired in --
  local src; src="$(cat "${DEPLOY_SH}")"
  assert_not_contains "${src}" "ensure_public_ingress_allowed" "the public-ingress gate is removed"
  assert_not_contains "${src}" "create-vpc-block-public-access-exclusion" "deploy.sh never asks for a BPA exclusion"
  assert_not_contains "${src}" "describe-vpc-block-public-access" "deploy.sh does not read the account's BPA mode"
  # The only load-balancer hostname deploy.sh waits on is the orchestrator's INTERNAL
  # NLB, in resolve_orchestrator_internal_url; the old public ELB wait must not return.
  assert_eq "$(grep -c 'loadBalancer.ingress' "${DEPLOY_SH}")" "1" "exactly one load balancer hostname wait, the internal NLB"
  assert_contains "${src}" 'kubectl get svc "${ORCHESTRATOR_INTERNAL_SVC}"' "that wait reads the orchestrator-internal Service"
  assert_contains "${src}" "resolve_orchestrator_internal_url 3" "Phase 3 resolves the internal NLB hostname"
  assert_contains "${src}" 'deploy_ui_api_proxy 3 "${ORCHESTRATOR_INTERNAL_URL}"' "the proxy stack is given the internal NLB URL"
  assert_contains "${src}" "verify_ui_api_proxy 3" "Phase 3 drives one request through the proxy before building the UI"
  # The probe goes through boto3. `aws lambda invoke --cli-binary-format` is CLI v2
  # only; on a shell whose `aws` is v1 every invoke failed locally and the first live
  # run reported a network fault that did not exist.
  assert_contains "${src}" 'boto3.client("lambda", region_name=region).invoke(' "the readiness probe invokes the function through boto3"
  assert_not_contains "${src}" "cli-binary-format" "deploy.sh uses no AWS CLI v2-only flag"
  assert_contains "${src}" 'resolved "orchestratorInternalUrl=${ORCHESTRATOR_INTERNAL_URL}"' "the NLB URL is recorded in the prefix state"
  assert_not_contains "${src}" "ParameterValue=http://orchestrator.default.svc.cluster.local" "the proxy is never pointed at a cluster.local name"
  assert_not_contains "${src}" "orchestrator-url" "deploy_frontend.py is no longer given an orchestrator URL"
  assert_contains "${src}" 'UI_API_PROXY_STACK="${STACK_PREFIX:+${STACK_PREFIX}-}ui-api-proxy"' "the proxy stack follows the prefix convention"
  assert_contains "${src}" "deploy_ui_api_proxy 3" "Phase 3 deploys the UI API proxy"
  assert_contains "${src}" "--action grant-ui-api-invoke" "the Identity Pool role is granted invoke on the proxy"
  assert_contains "${src}" 'VITE_UI_API_PROXY_ARN=${UI_API_PROXY_ARN}' "the proxy ARN is baked into the UI build"
  # Both .env.production writers (Step 9 and --ui-only) carry it.
  assert_eq "$(grep -c 'VITE_UI_API_PROXY_ARN=' "${DEPLOY_SH}")" "2" "both UI env writers set VITE_UI_API_PROXY_ARN"
  # --destroy removes the proxy's VPC ENIs before eksctl deletes the subnets.
  assert_contains "${src}" 'for _vpc_lambda_stack in "${UI_API_PROXY_STACK}" "${VPC_PROXY_STACK}"' "--destroy deletes the proxy stack before the cluster"

  # --- deploy_closed_loop.sh: the UI rebuild keeps the proxy ARN ------------
  local cl; cl="$(cat "${SCRIPT_DIR}/../deploy_closed_loop.sh")"
  assert_not_contains "${cl}" "orchestrator-url" "deploy_closed_loop.sh is no longer given an orchestrator URL"
  assert_not_contains "${cl}" "get svc orchestrator" "deploy_closed_loop.sh does not look for an orchestrator load balancer (Triton's internal NLB lookup is unrelated)"
  assert_contains "${cl}" 'VITE_UI_API_PROXY_ARN=${UI_API_PROXY_ARN}' "the Step 7 rebuild carries the proxy ARN"

  # --- deploy_frontend.py: CloudFront is static-only ------------------------
  local fe; fe="$(cat "${SCRIPT_DIR}/../scripts/deploy_frontend.py")"
  assert_not_contains "${fe}" "alb-api" "no load-balancer origin in the distribution"
  assert_not_contains "${fe}" '"PathPattern"' "no cache behaviour of any kind in the distribution"
  assert_not_contains "${fe}" "--orchestrator-url" "deploy_frontend.py has no orchestrator URL argument"

  # --- deploy_ui_api_proxy: subnet + SG discovery and the stack call ---------
  # The real function, lifted from deploy.sh, run against a mocked aws that records
  # every call. The stack must land on the private (internal-elb tagged) subnets
  # with the cluster security group, and nothing public may be created.
  fn_src="$(sed -n '/^_cfn_output() {/,/^}/p; /^deploy_ui_api_proxy() {/,/^}/p' "${DEPLOY_SH}")"
  out="$(CALLS="${tmp}/calls" bash -c '
    set -uo pipefail
    AWS_REGION=us-east-1; AWS_PROFILE=t-profile; CLUSTER_NAME=t-cluster; STACK_PREFIX=t
    SCRIPT_DIR=/deploy; UI_API_PROXY_STACK=t-ui-api-proxy
    ORCH_URL=http://t-orch-0123456789abcdef.elb.us-east-1.amazonaws.com
    log()  { printf "%s\n" "$*"; }
    ok()   { printf "[OK] %s\n" "$*"; }
    fail() { local p=""; if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then p="Phase $1/5: "; shift; fi; printf "[fail] %s%s\n" "$p" "$*" >&2; exit 1; }
    aws() {
      printf "%s\n" "$*" >> "$CALLS"
      case "$*" in
        *eks\ describe-cluster*vpcId*) printf "vpc-0abc\n" ;;
        *eks\ describe-cluster*clusterSecurityGroupId*) printf "sg-0cafe\n" ;;
        *ec2\ describe-subnets*) printf "subnet-aaa\tsubnet-bbb\n" ;;
        *cloudformation\ describe-stacks*StackStatus*) printf "\n"; return 1 ;;
        *cloudformation\ create-stack*) printf "{\"StackId\":\"arn:x\"}\n" ;;
        *cloudformation\ wait*) ;;
        *cloudformation\ describe-stacks*ProxyFunctionArn*) printf "arn:aws:lambda:us-east-1:123456789012:function:t-ui-api-proxy\n" ;;
        *) printf "UNEXPECTED aws %s\n" "$*" >&2; return 1 ;;
      esac
      return 0
    }
    '"${fn_src}"'
    deploy_ui_api_proxy 3 "${ORCH_URL}"
    echo "ARN=${UI_API_PROXY_ARN}"
  ' 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "deploy_ui_api_proxy succeeds against a healthy mocked account (output: ${out})"
  assert_contains "${out}" "ARN=arn:aws:lambda:us-east-1:123456789012:function:t-ui-api-proxy" "the function ARN is read from the stack output"
  local calls; calls="$(cat "${tmp}/calls")"
  assert_contains "${calls}" "tag:kubernetes.io/role/internal-elb,Values=1" "subnets are the private ones eksctl tags for internal load balancers"
  assert_contains "${calls}" 'ParameterKey=SubnetIds,ParameterValue="subnet-aaa,subnet-bbb"' "both private subnets are passed to the stack"
  assert_contains "${calls}" "ParameterKey=SecurityGroupIds,ParameterValue=sg-0cafe" "the cluster security group is passed to the stack"
  assert_contains "${calls}" "ParameterKey=OrchestratorBaseUrl,ParameterValue=http://t-orch-0123456789abcdef.elb.us-east-1.amazonaws.com" "the Lambda targets the orchestrator's internal NLB hostname"
  assert_contains "${calls}" "create-stack --stack-name t-ui-api-proxy" "a missing stack is created"
  assert_not_contains "${calls}" "elbv2" "nothing creates a load balancer"
  assert_not_contains "${calls}" "block-public-access" "nothing reads or changes the account's BPA settings"

  # --- no private subnets: a clear Phase-3 failure, no stack call -----------
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" bash -c '
    set -uo pipefail
    AWS_REGION=us-east-1; CLUSTER_NAME=t-cluster; STACK_PREFIX=t; SCRIPT_DIR=/deploy; UI_API_PROXY_STACK=t-ui-api-proxy
    log() { :; }; ok() { :; }
    fail() { local p=""; if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then p="Phase $1/5: "; shift; fi; printf "[fail] %s%s\n" "$p" "$*" >&2; exit 1; }
    aws() {
      printf "%s\n" "$*" >> "$CALLS"
      case "$*" in
        *vpcId*) printf "vpc-0abc\n" ;;
        *clusterSecurityGroupId*) printf "sg-0cafe\n" ;;
        *describe-subnets*) printf "\n" ;;
        *) printf "UNEXPECTED aws %s\n" "$*" >&2; return 1 ;;
      esac
    }
    '"${fn_src}"'
    deploy_ui_api_proxy 3 http://t-orch-0123456789abcdef.elb.us-east-1.amazonaws.com
  ' 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "no private subnets stops the deploy"
  assert_contains "${out}" "Phase 3/5" "the failure is attributed to Phase 3"
  assert_contains "${out}" "internal-elb" "the failure names the subnet tag it looked for"
  assert_not_contains "$(cat "${tmp}/calls")" "cloudformation" "no stack call is made without subnets"

  # --- a cluster.local URL, or none, is refused before any AWS call -----------
  for bad in "" "http://orchestrator.default.svc.cluster.local"; do
    : > "${tmp}/calls"
    out="$(CALLS="${tmp}/calls" bash -c '
      set -uo pipefail
      AWS_REGION=us-east-1; CLUSTER_NAME=t-cluster; STACK_PREFIX=t; SCRIPT_DIR=/deploy; UI_API_PROXY_STACK=t-ui-api-proxy
      log() { :; }; ok() { :; }
      fail() { local p=""; if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then p="Phase $1/5: "; shift; fi; printf "[fail] %s%s\n" "$p" "$*" >&2; exit 1; }
      aws() { printf "%s\n" "$*" >> "$CALLS"; }
      '"${fn_src}"'
      deploy_ui_api_proxy 3 "$1"
    ' _ "${bad}" 2>&1)"; rc=$?
    assert_eq "${rc}" "1" "deploy_ui_api_proxy refuses the URL '${bad:-<empty>}'"
    assert_contains "${out}" "internal NLB URL" "the refusal names what it wanted"
    assert_eq "$(cat "${tmp}/calls" | wc -l | tr -d ' ')" "0" "no AWS call is made for a refused URL"
  done

  # --- resolve_orchestrator_internal_url: hostname appears -> URL exported ----
  local wait_src; wait_src="$(sed -n '/^resolve_orchestrator_internal_url() {/,/^}/p' "${DEPLOY_SH}")"
  out="$(bash -c '
    set -uo pipefail
    ORCHESTRATOR_INTERNAL_SVC=orchestrator-internal; ORCHESTRATOR_INTERNAL_URL=""
    ok() { printf "[OK] %s\n" "$*"; }
    fail() { local p=""; if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then p="Phase $1/5: "; shift; fi; printf "[fail] %s%s\n" "$p" "$*" >&2; exit 1; }
    kubectl() { printf "t-orch-0123456789abcdef.elb.us-east-1.amazonaws.com"; }
    '"${wait_src}"'
    resolve_orchestrator_internal_url 3
    echo "URL=${ORCHESTRATOR_INTERNAL_URL}"
  ' 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "resolve_orchestrator_internal_url succeeds when the Service has a hostname"
  assert_contains "${out}" "URL=http://t-orch-0123456789abcdef.elb.us-east-1.amazonaws.com" "it exports http://<hostname>"

  # --- resolve_orchestrator_internal_url: never a hostname -> Phase 3 failure --
  # The deadline is forced to the past so the test does not wait five minutes.
  out="$(bash -c '
    set -uo pipefail
    ORCHESTRATOR_INTERNAL_SVC=orchestrator-internal; ORCHESTRATOR_INTERNAL_URL=""
    ok() { :; }
    fail() { local p=""; if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then p="Phase $1/5: "; shift; fi; printf "[fail] %s%s\n" "$p" "$*" >&2; exit 1; }
    kubectl() { printf ""; }
    '"${wait_src/+ 300/- 1}"'
    resolve_orchestrator_internal_url 3
  ' 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "no hostname within the deadline stops the deploy"
  assert_contains "${out}" "Phase 3/5" "the failure is attributed to Phase 3"
  assert_contains "${out}" "kubectl describe svc orchestrator-internal" "the failure names the command that explains why"

  # --- the CFN parameter has no default and rejects cluster.local --------------
  local cfn; cfn="$(cat "${SCRIPT_DIR}/../ui_api_proxy_cfn.yaml")"
  assert_not_contains "${cfn}" "Default: http://orchestrator.default.svc.cluster.local" "OrchestratorBaseUrl has no cluster.local default"
  assert_contains "${cfn}" "AllowedPattern: '^http://[A-Za-z0-9.-]+\\.amazonaws\\.com" "OrchestratorBaseUrl only accepts an amazonaws.com hostname"
  assert_contains "${cfn}" 'BASE_URL = os.environ["ORCHESTRATOR_BASE_URL"]' "the handler has no fallback URL"

  rm -rf "${tmp}"
}

# =========================================================================
# Deal Scorer build identity: upload, Phase 1 marker, Phase 3 label
# =========================================================================
# The real constants and functions, lifted from deploy.sh and run under
# /bin/bash (3.2 on macOS) with set -euo pipefail against a mocked aws whose
# s3 cp reads and writes a local directory standing in for the model bucket.
test_ncf_build_wiring() {
  echo "test_ncf_build_wiring"
  local tmp; tmp="$(mktemp -d)"
  local sh=/bin/bash; [[ -x "${sh}" ]] || sh=bash
  local fn_src
  fn_src="$(sed -n '/^NCF_LABEL_RE=/,/^NCF_MODEL_LABEL=""/p; /^ncf_manifest_field() {/,/^}/p; /^file_sha256() {/,/^}/p; /^upload_ncf_source() {/,/^}/p; /^write_phase1_marker() {/,/^}/p; /^resolve_ncf_model_label() {/,/^}/p' "${DEPLOY_SH}")"
  local harness='
    set -euo pipefail
    AWS_REGION=us-east-1; MODEL_BUCKET=t-models; VERBOSE=1; DEPLOY_LOG=""; PYTHON=python3
    log()  { printf "%s\n" "$*"; }
    warn() { printf "[warn] %s\n" "$*"; }
    quiet() { shift; "$@"; }
    fail() { local p=""; if [[ "$1" =~ ^[1-5]$ && $# -ge 2 ]]; then p="Phase $1/5: "; shift; fi; printf "[fail] %s%s\n" "$p" "$*" >&2; exit 1; }
    aws() {
      printf "%s\n" "$*" >> "$CALLS"
      if [[ "$1" == s3 && "$2" == cp ]]; then
        local src="$3" dst="$4" k
        case "${src}" in
          s3://*) k="${src#s3://*/}"; [[ -f "${STORE}/${k}" ]] || return 1; cp "${STORE}/${k}" "${dst}" ;;
          *) k="${dst#s3://*/}"; mkdir -p "${STORE}/$(dirname "${k}")"; cp "${src}" "${STORE}/${k}" ;;
        esac
        return 0
      fi
      printf "UNEXPECTED aws %s\n" "$*" >&2; return 1
    }
  '
  local fp="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  local label="ncf-neumf-sample-0123456789ab"
  local seed_sha="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
  local build="${tmp}/build" out rc calls
  mkdir -p "${build}" "${tmp}/store"
  printf 'onnx' > "${build}/model.onnx"
  printf 'user,deal,label\n' > "${build}/interactions.csv"
  printf '{"fingerprint": "%s", "model_version": "%s"}\n' "${fp}" "${label}" > "${build}/manifest.json"

  # --- upload_ncf_source: three objects, fingerprint as S3 metadata -----------
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    upload_ncf_source "$1"
    echo "FP=${NCF_FINGERPRINT}"
    echo "LABEL=${NCF_MODEL_LABEL}"
  ' _ "${build}" 2>&1)"; rc=$?
  calls="$(cat "${tmp}/calls")"
  assert_eq "${rc}" "0" "upload_ncf_source succeeds on a complete build (output: ${out})"
  assert_contains "${out}" "FP=${fp}" "the manifest fingerprint becomes NCF_FINGERPRINT"
  assert_contains "${out}" "LABEL=${label}" "the manifest model_version becomes NCF_MODEL_LABEL"
  assert_contains "${calls}" "s3://t-models/onnx-source/ncf_deal_manager/model.onnx --metadata source-fingerprint=${fp}" "model.onnx carries the full fingerprint as source-fingerprint metadata"
  assert_contains "${calls}" "s3://t-models/onnx-source/ncf_deal_manager/manifest.json" "manifest.json is uploaded beside the ONNX"
  assert_contains "${calls}" "s3://t-models/onnx-source/ncf_deal_manager/interactions.csv" "interactions.csv is uploaded beside the ONNX"
  # The registry version names builds/<fp12>/, which a later build never overwrites.
  local kept="onnx-source/ncf_deal_manager/builds/${fp:0:12}"
  assert_contains "${calls}" "s3://t-models/${kept}/model.onnx --metadata source-fingerprint=${fp}" "model.onnx is also kept under builds/<fp12>/ with its fingerprint"
  assert_contains "${calls}" "s3://t-models/${kept}/manifest.json" "manifest.json is also kept under builds/<fp12>/"
  assert_contains "${calls}" "s3://t-models/${kept}/interactions.csv" "interactions.csv is also kept under builds/<fp12>/"
  assert_eq "$(grep -c "${kept}/" "${tmp}/calls" | tr -d ' ')" "3" "exactly three objects go under builds/<fp12>/"
  assert_eq "$(grep -n "${kept}/model.onnx" "${tmp}/calls" | head -1 | cut -d: -f1)" "1" "the kept copy is written before the keys the next build overwrites"
  # A second build, into a copy of the bucket so the checks below still see the first.
  local other_fp="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
  local build2="${tmp}/build2"
  cp -R "${tmp}/store" "${tmp}/store2"
  mkdir -p "${build2}"
  printf 'onnx-2' > "${build2}/model.onnx"
  printf 'user,deal,label\n' > "${build2}/interactions.csv"
  printf '{"fingerprint": "%s", "model_version": "ncf-neumf-sample-%s"}\n' "${other_fp}" "${other_fp:0:12}" > "${build2}/manifest.json"
  out="$(CALLS="${tmp}/calls2" STORE="${tmp}/store2" "${sh}" -c "${harness}${fn_src}"'
    upload_ncf_source "$1"
  ' _ "${build2}" 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "a second build uploads (output: ${out})"
  assert_eq "$(cat "${tmp}/store2/${kept}/model.onnx")" "onnx" "a second build leaves the first build's kept model.onnx unchanged"
  assert_eq "$(cat "${tmp}/store2/onnx-source/ncf_deal_manager/builds/${other_fp:0:12}/model.onnx")" "onnx-2" "the second build is kept under its own fingerprint"
  assert_eq "$(cat "${tmp}/store2/onnx-source/ncf_deal_manager/model.onnx")" "onnx-2" "the optimizer's key holds the latest build"

  # --- an incomplete build stops Phase 1 before any upload --------------------
  rm -f "${build}/interactions.csv"
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    upload_ncf_source "$1"
  ' _ "${build}" 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "a build without interactions.csv stops the deploy"
  assert_contains "${out}" "Phase 1/5" "the failure is attributed to Phase 1"
  assert_eq "$(wc -l < "${tmp}/calls" | tr -d ' ')" "0" "nothing is uploaded from an incomplete build"
  printf 'user,deal,label\n' > "${build}/interactions.csv"

  # --- a malformed model_version stops Phase 1 before any upload --------------
  printf '{"fingerprint": "%s", "model_version": "ncf-neumf-v1"}\n' "${fp}" > "${build}/manifest.json"
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    upload_ncf_source "$1"
  ' _ "${build}" 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "a model_version outside ncf-neumf-sample-<12 hex> stops the deploy"
  assert_contains "${out}" "does not match" "the failure names the label rule"
  assert_eq "$(wc -l < "${tmp}/calls" | tr -d ' ')" "0" "nothing is uploaded with a malformed label"
  printf '{"fingerprint": "%s", "model_version": "%s"}\n' "${fp}" "${label}" > "${build}/manifest.json"

  # --- write_phase1_marker: JSON with both values at deploy-markers/phase1.json --
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    write_phase1_marker "$1" "$2"
  ' _ "${seed_sha}" "${fp}" 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "write_phase1_marker succeeds (output: ${out})"
  assert_contains "$(cat "${tmp}/calls")" "s3://t-models/deploy-markers/phase1.json --content-type application/json" "the marker is written as JSON at deploy-markers/phase1.json"
  local marker; marker="$(cat "${tmp}/store/deploy-markers/phase1.json" 2>/dev/null)"
  assert_contains "${marker}" "\"deals_seed_sha256\": \"${seed_sha}\"" "the marker records the deals seed sha256"
  assert_contains "${marker}" "\"ncf_fingerprint\": \"${fp}\"" "the marker records the Deal Scorer fingerprint"
  assert_contains "${marker}" "\"written_at\": \"" "the marker records when it was written"

  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    write_phase1_marker "$1" ""
  ' _ "${seed_sha}" 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "an empty Deal Scorer fingerprint is refused"
  assert_eq "$(wc -l < "${tmp}/calls" | tr -d ' ')" "0" "no marker is written without a fingerprint"

  # --- file_sha256 matches shasum ---------------------------------------------
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    file_sha256 "$1"
  ' _ "${build}/model.onnx" 2>&1)"
  assert_eq "${out}" "$(shasum -a 256 "${build}/model.onnx" | cut -d' ' -f1)" "file_sha256 prints the file's sha256"

  # --- resolve_ncf_model_label: label from this run, no AWS call --------------
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    NCF_MODEL_LABEL="$1"
    resolve_ncf_model_label
    echo "LABEL=${NCF_MODEL_LABEL}"
  ' _ "${label}" 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "a label set by Phase 1 in this run is used"
  assert_contains "${out}" "LABEL=${label}" "the Phase 1 label is kept"
  assert_eq "$(wc -l < "${tmp}/calls" | tr -d ' ')" "0" "no AWS call is made when Phase 1 set the label"

  # --- resolve_ncf_model_label: label read from the bucket manifest -----------
  : > "${tmp}/calls"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    resolve_ncf_model_label
    echo "LABEL=${NCF_MODEL_LABEL}"
  ' 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "the label is read from the bucket when Phase 1 did not run (output: ${out})"
  assert_contains "${out}" "LABEL=${label}" "the bucket manifest's model_version becomes NCF_MODEL_LABEL"
  assert_contains "$(cat "${tmp}/calls")" "s3 cp s3://t-models/onnx-source/ncf_deal_manager/manifest.json" "the manifest is read from onnx-source/ncf_deal_manager/"

  # --- resolve_ncf_model_label: no manifest in the bucket -> Phase 3 failure ---
  rm -rf "${tmp}/store/onnx-source"
  out="$(CALLS="${tmp}/calls" STORE="${tmp}/store" "${sh}" -c "${harness}${fn_src}"'
    resolve_ncf_model_label
  ' 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "no label anywhere stops the deploy"
  assert_contains "${out}" "Phase 3/5" "the failure is attributed to Phase 3"
  assert_contains "${out}" "--start-at 1" "the failure names the command that builds the Deal Scorer"

  # --- the manifest placeholder is substituted and checked in Phase 3 ---------
  local deploy; deploy="$(cat "${DEPLOY_SH}")"
  assert_contains "${deploy}" '-e "s|__NCF_MODEL_LABEL__|${NCF_MODEL_LABEL}|g"' "the Phase 3 sed loop substitutes __NCF_MODEL_LABEL__"
  assert_contains "${deploy}" "grep -q '__NCF_MODEL_LABEL__'" "Phase 3 refuses a manifest with the placeholder left in"

  rm -rf "${tmp}"
}

# =========================================================================
# Phase staleness: probe expectations, digest-stale Deployments, AgentCore, Prebid
# =========================================================================
# The real constants and functions, lifted from deploy.sh and run under /bin/bash
# (3.2 on macOS) with set -euo pipefail. kubectl and aws are shell functions that
# answer from fixture files and record every call, so no test reaches AWS or a
# cluster; .deploy-state.json is a temp file behind the real lib/deploy_state.sh.
_staleness_src() {
  sed -n '/^MANAGED_DEPLOYMENTS=(/,/orchestrator)$/p
          /^STALE_DEPLOYMENTS=()/p; /^STALE_DEPLOYMENT_LINES=()/p
          /^AGENTCORE_STALE_WHY=""/p; /^PREBID_FINGERPRINT=""/p; /^PREBID_STEP_WHY=""/p
          /^force_phase() {/,/^}/p; /^file_sha256() {/,/^}/p
          /^compute_phase_expectations() {/,/^}/p
          /^recorded_image_tag() {/,/^}/p; /^ecr_image_digest() {/,/^}/p
          /^_stale_verdicts() {/,/^}/p; /^stale_deployments() {/,/^}/p
          /^restart_stale_deployments() {/,/^}/p
          /^agentcore_image_stale() {/,/^}/p; /^record_agentcore_digest() {/,/^}/p
          /^prebid_fingerprint() {/,/^}/p; /^prebid_step_needed() {/,/^}/p
          /^check_unprobed_staleness() {/,/^}/p
          /^_hashable_files() {/,/^}/p' "${DEPLOY_SH}"
}

# _kube_fixture <out.json> <name=imageID[|imageID...]>...: a `kubectl get
# deployments,replicasets,pods -o json` answer. One Deployment per name, its
# ReplicaSet, and one pod per imageID ("" = a pod with no digest yet; a leading
# "~" marks a terminating pod).
_kube_fixture() {
  python3 - "$@" <<'PY'
import json
import sys

out, specs = sys.argv[1], sys.argv[2:]
items = []
for spec in specs:
    name, _, ids = spec.partition("=")
    image = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/t-stack-{name}:tag1"
    items.append({"kind": "Deployment", "metadata": {"name": name},
                  "spec": {"template": {"spec": {"containers": [{"name": "app", "image": image}]}}}})
    rs = f"{name}-5d9f"
    items.append({"kind": "ReplicaSet", "metadata": {"name": rs, "ownerReferences": [
        {"kind": "Deployment", "name": name}]}})
    for i, image_id in enumerate(ids.split("|") if ids else []):
        meta = {"name": f"{rs}-p{i}", "ownerReferences": [{"kind": "ReplicaSet", "name": rs}]}
        if image_id.startswith("~"):
            meta["deletionTimestamp"] = "2026-10-07T00:00:00Z"
            image_id = image_id[1:]
        status = {"phase": "Running",
                  "containerStatuses": [{"name": "app", "image": image, "imageID": image_id}]}
        items.append({"kind": "Pod", "metadata": meta, "status": status})
with open(out, "w") as handle:
    json.dump({"apiVersion": "v1", "kind": "List", "items": items}, handle)
PY
}

test_phase_staleness() {
  echo "test_phase_staleness"
  local tmp; tmp="$(mktemp -d)"
  local sh=/bin/bash; [[ -x "${sh}" ]] || sh=bash
  local fn_src; fn_src="$(_staleness_src)"
  local reg="123456789012.dkr.ecr.us-east-1.amazonaws.com"
  local d1="sha256:1111111111111111111111111111111111111111111111111111111111111111"
  local d2="sha256:2222222222222222222222222222222222222222222222222222222222222222"
  local all="bid-pricer audience-activator deal-scorer signals-enricher yield-optimizer-floor yield-optimizer-margin artf-template orchestrator"
  local out rc calls n spec

  # ECR answers from ${ECR}: one "repo:tag=digest" per line; anything else is
  # ImageNotFound. kubectl answers `get deployments,replicasets,pods` from ${KUBE}
  # (missing file = the cluster cannot be read) and `get deployment/<n>` from the
  # Deployments in it.
  local harness='
    set -euo pipefail
    AWS_REGION=us-east-1; STACK_NAME=t-stack; STACK_PREFIX=t; VERBOSE=1
    REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com; IMAGE_TAG=tag1
    SCRIPT_DIR="${FAKE_SCRIPT_DIR}"; START_AT="${START_AT:-1}"
    SKIP_AGENTCORE="${SKIP_AGENTCORE:-0}"; WITH_PREBID="${WITH_PREBID:-1}"
    CLUSTER_NAME=t-stack-triton; DEPLOY_KUBE_CONTEXT="${CTX:-}"
    PYTHON="${PYTHON:-python3}"; STATE_PYTHON=python3
    FORCE_PHASE_1=0; FORCE_PHASE_1_WHY=""; FORCE_PHASE_3=0; FORCE_PHASE_3_WHY=""
    FORCE_PHASE_5=0; FORCE_PHASE_5_WHY=""
    DEPLOY_EXPECT_DEALS_SEED_SHA=""; DEPLOY_EXPECT_NCF_FINGERPRINT=""
    log()  { printf "%s\n" "$*"; }
    warn() { printf "[warn] %s\n" "$*"; }
    say()  { printf "%s\n" "$*"; }
    source "${LIB}/deploy_state.sh"
    aws() {
      printf "aws %s\n" "$*" >> "$CALLS"
      if [[ "$1 $2" == "ecr describe-images" ]]; then
        local repo="" tag="" a
        while [[ $# -gt 0 ]]; do
          case "$1" in
            --repository-name) repo="$2"; shift ;;
            --image-ids) tag="${2#imageTag=}"; shift ;;
          esac
          shift
        done
        a="$(grep -F "${repo}:${tag}=" "$ECR" 2>/dev/null | head -n 1 || true)"
        if [[ -n "${a}" ]]; then printf "%s\n" "${a#*=}"; return 0; fi
        printf "An error occurred (ImageNotFoundException)\n" >&2; return 254
      fi
      printf "UNEXPECTED aws %s\n" "$*" >&2; return 1
    }
    kubectl() {
      printf "kubectl %s\n" "$*" >> "$CALLS"
      if [[ "$1" == --context ]]; then shift 2; fi
      case "$*" in
        "get deployments,replicasets,pods -o json") [[ -f "$KUBE" ]] || return 1; cat "$KUBE" ;;
        get\ deployment/*) [[ -f "$KUBE" ]] && grep -q "\"name\": \"${2#deployment/}\"" "$KUBE" ;;
        rollout\ restart\ deployment/*) return 0 ;;
        *) printf "UNEXPECTED kubectl %s\n" "$*" >&2; return 1 ;;
      esac
    }
  '
  _ecr_all() {  # _ecr_all <digest>: ECR holds <digest> for every managed repo at tag1
    : > "${tmp}/ecr"
    for n in ${all}; do printf 't-stack-%s:tag1=%s\n' "${n}" "$1" >> "${tmp}/ecr"; done
  }
  _run() {  # _run <script>: the harness plus the lifted functions plus <script>
    CALLS="${tmp}/calls" ECR="${tmp}/ecr" KUBE="${tmp}/kube.json" LIB="${SCRIPT_DIR}/../lib" \
      DEPLOY_STATE_FILE="${tmp}/state.json" FAKE_SCRIPT_DIR="${tmp}/deployment" \
      "${sh}" -c "${harness}${fn_src}"$'\n'"$1" 2>&1
  }
  mkdir -p "${tmp}/deployment"

  # --- no stale Deployment: nothing restarted, empty arrays under set -u -----
  _ecr_all "${d1}"
  spec=(); for n in ${all}; do spec+=("${n}=${reg}/t-stack-${n}@${d1}|${reg}/t-stack-${n}@${d1}"); done
  _kube_fixture "${tmp}/kube.json" "${spec[@]}"
  : > "${tmp}/calls"
  out="$(_run '
    restart_stale_deployments tag1
    echo "STALE=${#STALE_DEPLOYMENTS[@]}"
  ')"; rc=$?
  calls="$(cat "${tmp}/calls")"
  assert_eq "${rc}" "0" "restart_stale_deployments runs under /bin/bash set -u with nothing stale (output: ${out})"
  assert_contains "${out}" "STALE=0" "every pod on ECR's digest leaves the stale list empty"
  assert_contains "${out}" "none restarted" "the rollout says nothing needed restarting"
  assert_not_contains "${calls}" "rollout restart" "no Deployment is restarted when every pod runs ECR's digest"
  assert_contains "${calls}" "ecr describe-images --repository-name t-stack-orchestrator --image-ids imageTag=tag1" "ECR is asked for <stack>-<name>:<tag>"
  assert_not_contains "${calls}" "put-image" "the check never writes to ECR"

  # --- one stale Deployment: only it is restarted ----------------------------
  spec=(); for n in ${all}; do
    if [[ "${n}" == deal-scorer ]]; then spec+=("${n}=${reg}/t-stack-${n}@${d2}|${reg}/t-stack-${n}@${d1}")
    else spec+=("${n}=${reg}/t-stack-${n}@${d1}"); fi
  done
  _kube_fixture "${tmp}/kube.json" "${spec[@]}"
  : > "${tmp}/calls"
  out="$(_run 'restart_stale_deployments tag1')"; rc=$?
  calls="$(cat "${tmp}/calls")"
  assert_eq "${rc}" "0" "one stale Deployment: restart_stale_deployments succeeds"
  assert_contains "${calls}" "rollout restart deployment/deal-scorer" "the stale Deployment is restarted"
  assert_eq "$(grep -c 'rollout restart' "${tmp}/calls")" "1" "only the stale Deployment is restarted"
  assert_contains "${out}" "deal-scorer: stale, ECR holds 111111111111; pod deal-scorer-5d9f-p0 runs 222222222222" "the reason names the pod and both digests"

  # --- unknown digests count as stale ----------------------------------------
  # bid-pricer: a pod with no imageID yet; signals-enricher: a containerd image id
  # without a repo digest; orchestrator: ECR has no image at the tag.
  grep -v '^t-stack-orchestrator:' "${tmp}/ecr" > "${tmp}/ecr2"; mv "${tmp}/ecr2" "${tmp}/ecr"
  spec=(); for n in ${all}; do
    case "${n}" in
      bid-pricer) spec+=("${n}=${reg}/t-stack-${n}@${d1}|") ;;
      signals-enricher) spec+=("${n}=${d1}") ;;
      *) spec+=("${n}=${reg}/t-stack-${n}@${d1}") ;;
    esac
  done
  _kube_fixture "${tmp}/kube.json" "${spec[@]}"
  : > "${tmp}/calls"
  out="$(_run 'restart_stale_deployments tag1')"; rc=$?
  calls="$(cat "${tmp}/calls")"
  assert_eq "${rc}" "0" "unknown digests: restart_stale_deployments succeeds"
  assert_contains "${calls}" "rollout restart deployment/bid-pricer" "a pod with no digest yet counts as stale"
  assert_contains "${calls}" "rollout restart deployment/signals-enricher" "an image id with no repo digest counts as stale"
  assert_contains "${calls}" "rollout restart deployment/orchestrator" "no ECR digest for the tag counts as stale"
  assert_eq "$(grep -c 'rollout restart' "${tmp}/calls")" "3" "only the three unknown Deployments are restarted"
  assert_contains "${out}" "orchestrator: unknown, ECR has no readable digest" "the unknown reason is logged"

  # --- Docker-style imageID, a terminating old pod, an absent Deployment -----
  _ecr_all "${d1}"
  spec=(); for n in ${all}; do
    case "${n}" in
      audience-activator) spec+=("${n}=docker-pullable://${reg}/t-stack-${n}@${d1}") ;;
      yield-optimizer-floor) spec+=("${n}=~${reg}/t-stack-${n}@${d2}|${reg}/t-stack-${n}@${d1}") ;;
      artf-template) ;;
      *) spec+=("${n}=${reg}/t-stack-${n}@${d1}") ;;
    esac
  done
  _kube_fixture "${tmp}/kube.json" "${spec[@]}"
  : > "${tmp}/calls"
  out="$(_run '
    restart_stale_deployments tag1
    echo "STALE=${STALE_DEPLOYMENTS[*]+${STALE_DEPLOYMENTS[*]}}."
  ')"; rc=$?
  assert_eq "${rc}" "0" "mixed fixture: restart_stale_deployments succeeds (output: ${out})"
  assert_contains "${out}" "STALE=." "docker-pullable digests match, terminating pods and absent Deployments are not stale"
  assert_not_contains "$(cat "${tmp}/calls")" "rollout restart" "nothing is restarted for them"

  # --- an unreadable cluster: every managed name is stale, none is restarted blind
  rm -f "${tmp}/kube.json"
  : > "${tmp}/calls"
  out="$(_run '
    restart_stale_deployments tag1
    echo "STALE=${#STALE_DEPLOYMENTS[@]}"
  ')"; rc=$?
  assert_eq "${rc}" "0" "an unreadable cluster does not stop the deploy"
  assert_contains "${out}" "STALE=8" "an unreadable cluster makes every managed Deployment unknown"
  assert_not_contains "$(cat "${tmp}/calls")" "rollout restart" "a Deployment kubectl cannot find is not restarted"

  # --- the startup check uses the probe's context and forces Phase 3 ----------
  spec=(); for n in ${all}; do
    if [[ "${n}" == orchestrator ]]; then spec+=("${n}=${reg}/t-stack-${n}@${d2}"); else spec+=("${n}=${reg}/t-stack-${n}@${d1}"); fi
  done
  _kube_fixture "${tmp}/kube.json" "${spec[@]}"
  printf '{"Registry": "%s", "ImageTag": "tag1"}\n' "${reg}" > "${tmp}/deployment/.image-outputs.json"
  python3 "${SCRIPT_DIR}/../scripts/deploy_state.py" --file "${tmp}/state.json" --prefix t set --section resolved --kv "agentcoreImageDigest=${d1}"
  printf 't-stack-agentcore:tag1=%s\n' "${d1}" >> "${tmp}/ecr"
  : > "${tmp}/calls"
  out="$(CTX=arn:aws:eks:us-east-1:123456789012:cluster/t-stack-triton _run '
    IMAGE_TAG=other-tag
    check_unprobed_staleness
    echo "F3=${FORCE_PHASE_3} F5=${FORCE_PHASE_5} WHY3=${FORCE_PHASE_3_WHY}"
  ')"; rc=$?
  calls="$(cat "${tmp}/calls")"
  assert_eq "${rc}" "0" "check_unprobed_staleness runs under set -u (output: ${out})"
  assert_contains "${out}" "F3=1 F5=0" "a digest-stale Deployment forces Phase 3 and a matching AgentCore digest does not force Phase 5"
  assert_contains "${out}" "1 Deployment(s) do not run ECR's image for tag tag1 (orchestrator)" "the reason names the tag from .image-outputs.json and the Deployment"
  assert_contains "${calls}" "kubectl --context arn:aws:eks:us-east-1:123456789012:cluster/t-stack-triton get deployments,replicasets,pods -o json" "the startup check reads the probe's context for this cluster"
  assert_not_contains "${calls}" "rollout restart" "the startup check restarts nothing"

  : > "${tmp}/calls"
  out="$(CTX="" _run '
    check_unprobed_staleness
    echo "F3=${FORCE_PHASE_3} WHY3=${FORCE_PHASE_3_WHY}"
  ')"
  assert_contains "${out}" "F3=1 WHY3=the running pods' image digests could not be read" "without the probe's context Phase 3 is forced, not guessed"
  assert_not_contains "$(cat "${tmp}/calls")" "kubectl" "without the probe's context kubectl's current context is never read"

  spec=(); for n in ${all}; do spec+=("${n}=${reg}/t-stack-${n}@${d1}"); done
  _kube_fixture "${tmp}/kube.json" "${spec[@]}"
  out="$(CTX=ctx-t _run '
    check_unprobed_staleness
    echo "F3=${FORCE_PHASE_3} F5=${FORCE_PHASE_5}"
  ')"
  assert_contains "${out}" "F3=0 F5=0" "nothing stale forces nothing"
  : > "${tmp}/calls"
  out="$(CTX=ctx-t START_AT=4 _run '
    check_unprobed_staleness
    echo "F3=${FORCE_PHASE_3}"
  ')"
  assert_contains "${out}" "F3=0" "--start-at 4 skips the Deployment check"
  assert_not_contains "$(cat "${tmp}/calls")" "kubectl" "--start-at 4 makes no kubectl call"

  # --- AgentCore: recorded digest vs ECR ------------------------------------
  out="$(CTX=ctx-t _run '
    if agentcore_image_stale tag1; then echo "STALE ${AGENTCORE_STALE_WHY}"; else echo "FRESH"; fi
  ')"
  assert_contains "${out}" "FRESH" "the recorded AgentCore digest equal to ECR's is not stale"
  grep -v '^t-stack-agentcore:' "${tmp}/ecr" > "${tmp}/ecr2"; mv "${tmp}/ecr2" "${tmp}/ecr"
  printf 't-stack-agentcore:tag1=%s\n' "${d2}" >> "${tmp}/ecr"
  out="$(CTX=ctx-t _run '
    check_unprobed_staleness
    echo "F5=${FORCE_PHASE_5} WHY5=${FORCE_PHASE_5_WHY}"
  ')"
  assert_contains "${out}" "F5=1" "an AgentCore image that changed in ECR forces Phase 5"
  assert_contains "${out}" "registered with AgentCore image 111111111111, and ECR now holds 222222222222" "the reason names both digests"
  out="$(CTX=ctx-t SKIP_AGENTCORE=1 _run '
    check_unprobed_staleness
    echo "F5=${FORCE_PHASE_5}"
  ')"
  assert_contains "${out}" "F5=0" "--skip-agentcore skips the AgentCore check"
  grep -v '^t-stack-agentcore:' "${tmp}/ecr" > "${tmp}/ecr2"; mv "${tmp}/ecr2" "${tmp}/ecr"
  out="$(_run 'if agentcore_image_stale tag1; then echo "STALE ${AGENTCORE_STALE_WHY}"; fi')"
  assert_contains "${out}" "STALE ECR has no readable digest for t-stack-agentcore:tag1" "an unreadable ECR digest counts as stale"
  rm -f "${tmp}/state.json"
  printf 't-stack-agentcore:tag1=%s\n' "${d2}" >> "${tmp}/ecr"
  out="$(_run 'if agentcore_image_stale tag1; then echo "STALE ${AGENTCORE_STALE_WHY}"; fi')"
  assert_contains "${out}" "STALE no AgentCore image digest is recorded" "no recorded digest counts as stale"
  out="$(_run '
    record_agentcore_digest "$(ecr_image_digest t-stack-agentcore tag1)"
    record_agentcore_digest "not-a-digest"
    if agentcore_image_stale tag1; then echo "STALE"; else echo "FRESH"; fi
  ')"
  assert_contains "$(cat "${tmp}/state.json" 2>/dev/null)" "\"agentcoreImageDigest\": \"${d2}\"" "record_agentcore_digest writes resolved.agentcoreImageDigest"
  assert_contains "${out}" "[warn]" "a malformed digest is not recorded, and says so"
  assert_contains "${out}" "FRESH" "after recording, the same ECR digest is not stale"

  # --- Prebid fingerprint: what changes it ------------------------------------
  local root="${tmp}/repo" seed_a seed_b fp0 fp
  seed_a="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  seed_b="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  mkdir -p "${root}/source/prebid/artf-hook" "${root}/source/demand/artfhouse" "${root}/source/other" \
           "${root}/deployment/scripts" "${root}/deployment/eks"
  printf 'class A {}\n' > "${root}/source/prebid/artf-hook/A.java"
  printf 'x = 1\n' > "${root}/source/demand/artfhouse/catalog.py"
  printf 'y = 1\n' > "${root}/source/other/b.py"
  for f in deployment/prebid_cfn.yaml deployment/deploy_prebid.sh deployment/scripts/prebid_config_template.yaml \
           deployment/eks/prebid-server-deployment.yaml deployment/eks/amt-simulator-deployment.yaml; do
    printf '%s\n' "${f}" > "${root}/${f}"
  done
  _fp() { _run 'prebid_fingerprint "$PB_ROOT" "$PB_SEED"'; }
  fp0="$(PB_ROOT="${root}" PB_SEED="${seed_a}" _fp)"
  assert_eq "$([[ "${fp0}" =~ ^[0-9a-f]{64}$ ]] && echo yes)" "yes" "prebid_fingerprint prints 64 hex (got: ${fp0})"
  assert_eq "$(PB_ROOT="${root}" PB_SEED="${seed_a}" _fp)" "${fp0}" "the same sources and seed give the same fingerprint"
  printf 'z = 1\n' >> "${root}/source/other/b.py"
  assert_eq "$(PB_ROOT="${root}" PB_SEED="${seed_a}" _fp)" "${fp0}" "a file outside the Prebid host leaves it unchanged"
  assert_not_contains "$(PB_ROOT="${root}" PB_SEED="${seed_b}" _fp)" "${fp0}" "a changed deals seed changes it"
  printf 'class B {}\n' >> "${root}/source/prebid/artf-hook/A.java"
  fp="$(PB_ROOT="${root}" PB_SEED="${seed_a}" _fp)"
  assert_not_contains "${fp}" "${fp0}" "a changed file under source/prebid/ changes it"
  fp0="${fp}"
  printf 'x = 2\n' > "${root}/source/demand/artfhouse/catalog.py"
  fp="$(PB_ROOT="${root}" PB_SEED="${seed_a}" _fp)"
  assert_not_contains "${fp}" "${fp0}" "a changed demand Lambda source changes it"
  fp0="${fp}"
  printf 'changed\n' >> "${root}/deployment/prebid_cfn.yaml"
  fp="$(PB_ROOT="${root}" PB_SEED="${seed_a}" _fp)"
  assert_not_contains "${fp}" "${fp0}" "a changed prebid_cfn.yaml changes it"
  out="$(PB_ROOT="${root}" PB_SEED="not-hex" _run 'if prebid_fingerprint "$PB_ROOT" "$PB_SEED"; then echo OK; else echo REFUSED; fi')"
  assert_contains "${out}" "REFUSED" "a malformed seed sha is refused"

  # --- Prebid: when Step 12 runs ------------------------------------------------
  rm -f "${tmp}/state.json"
  _step12() {  # _step12 <phase5-ran>: RUN or SKIP, with the reason
    _run 'PREBID_FINGERPRINT="${PB_FP}"
      if prebid_step_needed "$PB_RAN"; then echo "RUN ${PREBID_STEP_WHY}"; else echo "SKIP"; fi'
  }
  out="$(WITH_PREBID=1 PB_FP="${fp}" PB_RAN=0 _step12)"
  assert_contains "${out}" "RUN no Prebid host fingerprint is recorded" "a missing record runs Step 12 on a healthy stack"
  python3 "${SCRIPT_DIR}/../scripts/deploy_state.py" --file "${tmp}/state.json" --prefix t set --section resolved --kv "prebidFingerprint=${fp}"
  out="$(WITH_PREBID=1 PB_FP="${fp}" PB_RAN=0 _step12)"
  assert_eq "${out}" "SKIP" "an unchanged fingerprint skips Step 12 on a healthy stack"
  out="$(WITH_PREBID=1 PB_FP="${fp0}" PB_RAN=0 _step12)"
  assert_contains "${out}" "RUN the Prebid host's sources or deals seed changed since its last deploy (${fp:0:12} to ${fp0:0:12})" "a changed fingerprint runs Step 12"
  out="$(WITH_PREBID=1 PB_FP="" PB_RAN=0 _step12)"
  assert_contains "${out}" "RUN the Prebid host's sources could not be fingerprinted" "an uncomputable fingerprint runs Step 12"
  out="$(WITH_PREBID=1 PB_FP="${fp}" PB_RAN=1 _step12)"
  assert_contains "${out}" "RUN" "Step 12 runs whenever Phase 5 runs, as before"
  for _ran in 0 1; do
    out="$(WITH_PREBID=0 PB_FP="${fp0}" PB_RAN="${_ran}" _step12)"
    assert_eq "${out}" "SKIP" "without --with-prebid Step 12 never runs (Phase 5 ran: ${_ran})"
  done
  local deploy; deploy="$(cat "${DEPLOY_SH}")"
  local close_line step12_line record_line
  close_line="$(grep -n '^fi # _run_phase 5 (Phase 5)$' "${DEPLOY_SH}" | cut -d: -f1)"
  step12_line="$(grep -n '^if prebid_step_needed "${_PHASE5_RAN}"; then$' "${DEPLOY_SH}" | cut -d: -f1)"
  record_line="$(grep -n 'state_set "${STACK_PREFIX}" resolved "prebidFingerprint=${PREBID_FINGERPRINT}"' "${DEPLOY_SH}" | cut -d: -f1)"
  assert_eq "$([[ -n "${close_line}" && -n "${step12_line}" && "${close_line}" -lt "${step12_line}" ]] && echo yes)" "yes" "Step 12 sits after the Phase 5 block, so a skipped Phase 5 can still run it"
  assert_eq "$([[ -n "${record_line}" && "${record_line}" -gt "${step12_line}" ]] && echo yes)" "yes" "the Prebid fingerprint is recorded inside Step 12"
  assert_contains "${deploy}" 'elif [[ -n "${PREBID_FINGERPRINT}" ]]; then' "the fingerprint is recorded only when deploy_prebid.sh exited 0"
  assert_contains "${deploy}" 'record_agentcore_digest "${_AC_DIGEST}"' "Step 10 records the AgentCore digest after registering the runtime"
  assert_contains "${deploy}" 'restart_stale_deployments "${IMAGE_TAG}"' "Step 8b restarts only stale Deployments"
  assert_not_contains "${deploy}" 'for DEPLOY in bid-pricer audience-activator' "the unconditional restart loop is gone"

  # --- the probe is given what this checkout would deploy ----------------------
  printf '#!/bin/sh\ncase "$1" in *build.py) [ -n "$FAKE_FP" ] && echo "log line" && echo "$FAKE_FP"; exit ${FAKE_RC:-0} ;; esac\nexec python3 "$@"\n' > "${tmp}/fakepy"
  chmod +x "${tmp}/fakepy"
  mkdir -p "${tmp}/deployment/seed"
  printf '{"deals": []}\n' > "${tmp}/deployment/seed/deals_and_campaigns.json"
  local want_seed; want_seed="$(shasum -a 256 "${tmp}/deployment/seed/deals_and_campaigns.json" | cut -d' ' -f1)"
  local nfp="9509e1edf0273ebbb9efa03b17fcbe4ad9c1264119dd120dedff93f40f41fc48"
  out="$(PYTHON="${tmp}/fakepy" FAKE_FP="${nfp}" _run '
    compute_phase_expectations
    echo "SEED=${DEPLOY_EXPECT_DEALS_SEED_SHA} FP=${DEPLOY_EXPECT_NCF_FINGERPRINT} F1=${FORCE_PHASE_1}"
  ')"; rc=$?
  assert_eq "${rc}" "0" "compute_phase_expectations runs under set -u (output: ${out})"
  assert_contains "${out}" "SEED=${want_seed}" "the expected seed sha is the seed file's sha256"
  assert_contains "${out}" "FP=${nfp} F1=0" "the expected fingerprint is build.py's last output line, and nothing is forced"
  out="$(PYTHON="${tmp}/fakepy" FAKE_FP="" FAKE_RC=1 _run '
    compute_phase_expectations
    echo "FP=${DEPLOY_EXPECT_NCF_FINGERPRINT}. F1=${FORCE_PHASE_1} WHY1=${FORCE_PHASE_1_WHY}"
  ')"
  assert_contains "${out}" "FP=. F1=1 WHY1=the local Deal Scorer fingerprint could not be computed" "an uncomputable fingerprint forces Phase 1 instead of skipping its check"
  assert_contains "${out}" "[warn]" "and warns"

  local flags
  flags="$(bash -c '
    set -u
    source "$1"
    WITH_RETRAINING=1; WITH_PREBID=0; SKIP_AGENTCORE=0; AWS_PROFILE=""
    DEPLOY_EXPECT_DEALS_SEED_SHA=aaa; DEPLOY_EXPECT_NCF_FINGERPRINT=bbb
    _gate_flags
  ' _ "${SCRIPT_DIR}/../lib/deploy_gate.sh")"
  assert_contains "${flags}" $'--expect-deals-seed-sha\naaa' "_gate_flags passes the expected deals seed sha"
  assert_contains "${flags}" $'--expect-ncf-fingerprint\nbbb' "_gate_flags passes the expected Deal Scorer fingerprint"
  flags="$(bash -c '
    set -u
    source "$1"
    WITH_RETRAINING=1; WITH_PREBID=0; SKIP_AGENTCORE=0; AWS_PROFILE=""
    DEPLOY_EXPECT_DEALS_SEED_SHA=""; DEPLOY_EXPECT_NCF_FINGERPRINT=""
    _gate_flags
  ' _ "${SCRIPT_DIR}/../lib/deploy_gate.sh")"
  assert_not_contains "${flags}" "--expect-" "no expectation flag is passed without a value"
  assert_contains "$(cat "${DEPLOY_SH}")" $'compute_phase_expectations\nfi\n\n# --status' "the expectations are computed before --status and the first probe"

  rm -rf "${tmp}"
}

# =========================================================================
# update_managed_policy (review R8): at the IAM 5-version cap it prunes exactly
# one old non-default version before create-policy-version, and a failed
# create stops the step instead of leaving the old default in force.
# =========================================================================
test_update_managed_policy() {
  local fn_src calls out rc
  calls="$(mktemp)"
  fn_src="$(sed -n '/^fail() {/,/^}/p; /^update_managed_policy() {/,/^}/p' "${DEPLOY_SH}")"
  (
    eval "${fn_src}"
    VERBOSE=0; DEPLOY_LOG=""
    log() { return 0; }
    phase_hint() { return 0; }
    MOCK_COUNT=5
    aws() {
      echo "$*" >>"${calls}"
      if [[ "$2" == "list-policy-versions" && "$*" == *"length(Versions)"* ]]; then
        echo "${MOCK_COUNT}"
      elif [[ "$2" == "list-policy-versions" ]]; then
        echo "v1"
      elif [[ "$2" == "delete-policy-version" ]]; then
        MOCK_COUNT=$((MOCK_COUNT - 1))
      fi
      return 0
    }
    update_managed_policy "arn:aws:iam::1:policy/p" "p" '{"Version":"2012-10-17"}'
  ) >/dev/null 2>&1; rc=$?
  assert_eq "${rc}" "0" "update_managed_policy succeeds at the 5-version cap"
  assert_eq "$(grep -c 'delete-policy-version' "${calls}")" "1" "exactly one delete-policy-version at five versions"
  assert_eq "$(grep -n 'iam' "${calls}" | grep -E 'delete-policy-version|create-policy-version' | awk -F: '{print $2}' | awk '{print $2}' | tr '\n' ' ')" \
    "delete-policy-version create-policy-version " "the delete runs before the create"
  : >"${calls}"
  out="$( (
    eval "${fn_src}"
    VERBOSE=0; DEPLOY_LOG=""
    log() { return 0; }
    phase_hint() { return 0; }
    aws() {
      echo "$*" >>"${calls}"
      if [[ "$2" == "list-policy-versions" ]]; then echo 2; fi
      if [[ "$2" == "create-policy-version" ]]; then echo "LimitExceeded" >&2; return 254; fi
      return 0
    }
    update_managed_policy "arn:aws:iam::1:policy/p" "p" '{}'
    echo "continued past the failed update"
  ) 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "a failed create-policy-version fails the step"
  assert_contains "${out}" "could not update p: LimitExceeded" "the failure names the policy and the AWS error"
  assert_not_contains "${out}" "continued past the failed update" "nothing runs after the failed update"
  assert_eq "$(grep -c 'delete-policy-version' "${calls}")" "0" "no prune below five versions"
  rm -f "${calls}"
}
# =========================================================================
# lib/k8s_rollout.sh: digest-pinned images and rollout waits that fail loudly
# =========================================================================
_rollout_pods_fixture() {
  # $1: crash | pull | unschedulable
  case "$1" in
    crash) cat <<'J'
{"items":[{"metadata":{"name":"deal-scorer-abc"},"status":{"phase":"Running",
 "conditions":[{"type":"PodScheduled","status":"True"},{"type":"Ready","status":"False"}],
 "containerStatuses":[{"name":"deal-scorer","ready":false,"restartCount":4,
   "state":{"waiting":{"reason":"CrashLoopBackOff","message":"back-off 40s restarting failed container"}},
   "lastState":{"terminated":{"exitCode":1,"reason":"Error"}}}]}}]}
J
    ;;
    pull) cat <<'J'
{"items":[{"metadata":{"name":"prebid-server-xyz"},"status":{"phase":"Pending",
 "conditions":[{"type":"PodScheduled","status":"True"},{"type":"Ready","status":"False"}],
 "containerStatuses":[{"name":"prebid-server","ready":false,"restartCount":0,
   "state":{"waiting":{"reason":"ImagePullBackOff","message":"manifest unknown"}}}]}}]}
J
    ;;
    unschedulable) cat <<'J'
{"items":[{"metadata":{"name":"triton-1"},"status":{"phase":"Pending",
 "conditions":[{"type":"PodScheduled","status":"False","reason":"Unschedulable",
   "message":"0/3 nodes are available: 3 Insufficient nvidia.com/gpu."}]}}]}
J
    ;;
  esac
}

test_k8s_rollout() {
  echo "--- test_k8s_rollout ---"
  local lib="${SCRIPT_DIR}/../lib/k8s_rollout.sh" out rc tmp
  bash -n "${lib}" && PASS_COUNT=$((PASS_COUNT + 1)) || { FAIL_COUNT=$((FAIL_COUNT + 1)); echo "FAIL: k8s_rollout.sh syntax" >&2; }

  # A digest-pinned image is rendered from what ECR returns for the pushed tag.
  tmp="$(mktemp)"
  printf 'image: "__IMAGE__"\n' >"${tmp}"
  out="$( (
    set -u
    source "${lib}"
    aws() { [[ "$*" == *"describe-images --repository-name ex7-prebid-server --image-ids imageTag=v1-abc"* ]] \
              && echo "sha256:$(printf 'a%.0s' $(seq 1 64))"; }
    pin_image_in_manifest "${tmp}" "__IMAGE__" "1.dkr.ecr.us-east-1.amazonaws.com" \
      "ex7-prebid-server" "v1-abc" "us-east-1" && echo "pinned ${PINNED_IMAGE_REF}"
  ) 2>&1)"
  assert_contains "$(cat "${tmp}")" "ex7-prebid-server@sha256:aaaaaaaa" "the rendered Prebid image is digest-pinned"
  assert_not_contains "$(cat "${tmp}")" ":v1-abc" "the rendered image does not name the tag"
  assert_contains "${out}" "pinned 1.dkr.ecr.us-east-1.amazonaws.com/ex7-prebid-server@sha256:" "PINNED_IMAGE_REF is set"

  # No readable digest: the render fails instead of falling back to the tag.
  printf 'image: "__IMAGE__"\n' >"${tmp}"
  ( source "${lib}"; aws() { echo None; }
    pin_image_in_manifest "${tmp}" "__IMAGE__" "r" "repo" "t" "us-east-1" ); rc=$?
  assert_eq "${rc}" "1" "an unreadable digest fails the render"
  assert_contains "$(cat "${tmp}")" "__IMAGE__" "the manifest is left unrendered"
  rm -f "${tmp}"

  # Rollout completes: ok, exit 0.
  out="$( (
    set -u
    source "${lib}"
    kubectl() { [[ "$1" == "rollout" ]] && return 0; return 1; }
    wait_rollout orchestrator "" 120
  ) 2>&1)"; rc=$?
  assert_eq "${rc}" "0" "a completed rollout passes"
  assert_contains "${out}" "rollout ok: deployment/orchestrator" "a completed rollout is reported"

  # Rollout fails on a crash-looping pod: exit 1, the reason and its logs are printed.
  out="$( (
    set -u
    source "${lib}"
    kubectl() {
      case "$1" in
        (rollout) return 1 ;;
        (get) if [[ "$2" == deployment/* ]]; then echo '{"spec":{"selector":{"matchLabels":{"app":"deal-scorer"}}}}'
             else _rollout_pods_fixture crash; fi ;;
        (logs) [[ "$*" == *"--previous"* ]] && printf 'Traceback\nRuntimeError: model label missing\n' ;;
      esac
    }
    wait_rollout deal-scorer default 60 || echo "failed: ${ROLLOUT_FAILURE}"
  ) 2>&1)"
  assert_contains "${out}" "rollout FAILED: deployment/deal-scorer" "a failed rollout is reported"
  assert_contains "${out}" "CrashLoopBackOff" "the pod reason is printed"
  assert_contains "${out}" "RuntimeError: model label missing" "the crashed container's last log lines are printed"
  assert_contains "${out}" "failed: deployment/deal-scorer: deal-scorer-abc CrashLoopBackOff" "ROLLOUT_FAILURE summarises the reason"

  out="$( ( source "${lib}"
    kubectl() { case "$1" in (rollout) return 1 ;; (get) [[ "$2" == deployment/* ]] \
      && echo '{"spec":{"selector":{"matchLabels":{"app":"prebid-server"}}}}' || _rollout_pods_fixture pull ;; esac; }
    wait_rollout prebid-server default 60 ) 2>&1)"; rc=$?
  assert_eq "${rc}" "1" "an image pull failure fails the rollout"
  assert_contains "${out}" "ImagePullBackOff: manifest unknown" "the pull failure is named"

  out="$( ( source "${lib}"
    kubectl() { case "$1" in (rollout) return 1 ;; (get) [[ "$2" == deployment/* ]] \
      && echo '{"spec":{"selector":{"matchLabels":{"app":"triton"}}}}' || _rollout_pods_fixture unschedulable ;; esac; }
    wait_rollout triton-inference-server "" 900 ) 2>&1)"
  assert_contains "${out}" "Unschedulable: 0/3 nodes are available: 3 Insufficient nvidia.com/gpu." "an unschedulable pod names the scheduler's reason"

  # The managed Deployments and Prebid are waited on, and a failure is not 'Deployed'.
  local src
  src="$(cat "${DEPLOY_SH}")"
  assert_contains "${src}" 'source "${SCRIPT_DIR}/lib/k8s_rollout.sh"' "deploy.sh sources the rollout helper"
  assert_contains "${src}" 'wait_managed_rollouts' "deploy.sh waits on every managed Deployment"
  src="$(cat "${SCRIPT_DIR}/../deploy_prebid.sh")"
  assert_contains "${src}" 'wait_rollout prebid-server' "deploy_prebid.sh waits on the Prebid rollout"
  assert_contains "${src}" 'wait_rollout artfhouse-demand' "deploy_prebid.sh waits on the demand rollout"
  assert_contains "${src}" 'pin_image_in_manifest "${PROCESSED}" "__IMAGE__"' "deploy_prebid.sh renders Prebid by digest"
  assert_not_contains "${src}" '__IMAGE__|${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}' "deploy_prebid.sh no longer renders Prebid by tag"
}

# =========================================================================
# Run all tests
# =========================================================================
test_k8s_rollout
test_cluster_config_private_nodes
test_unknown_args_rejected
test_private_ui_api_path
test_gate_decisions
test_docker_autostart
test_progress_output
test_help_is_complete
test_start_at_validation
test_display_name
test_output_gating
test_fail_with_hint
test_concurrency_exit_code_propagation
test_bootstrap_watcher
test_remote_build_non_interactive
test_degraded_summary_suppresses_success
test_demo_credentials_block
test_ncf_build_wiring
test_phase_staleness
test_update_managed_policy

printf '\n%d passed, %d failed\n' "${PASS_COUNT}" "${FAIL_COUNT}"
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  exit 1
fi
exit 0
