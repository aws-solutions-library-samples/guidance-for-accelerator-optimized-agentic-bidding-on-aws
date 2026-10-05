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
    # _run_phase, lifted from deploy.sh so the real precedence is what is tested.
    _IMAGES_REBUILT=0
    _AGENTCORE_IMAGE_REBUILT=0
    _run_phase() {
      local n="$1" st
      [[ "${START_AT}" -le "${n}" ]] || return 1
      gate_wait "${n}" "${STACK_PREFIX}" || true
      if gate_should_run "${n}"; then return 0; fi
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
      st="$(gate_status "${n}")"
      ok "Phase ${n}/5 already complete in AWS (${st}) — skipping"
      return 1
    }
  }
  _gate_env

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
# The orchestrator used to sit behind an internet-facing load balancer, and an
# account running VPC Block Public Access in block-ingress mode dropped every
# packet to it at the internet gateway, so the
# deploy stopped. The design now has NO internet-facing resource
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
# Run all tests
# =========================================================================
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

printf '\n%d passed, %d failed\n' "${PASS_COUNT}" "${FAIL_COUNT}"
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  exit 1
fi
exit 0
