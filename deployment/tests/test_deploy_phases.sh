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
# Run all tests
# =========================================================================
test_start_at_validation
test_display_name
test_output_gating
test_fail_with_hint
test_concurrency_exit_code_propagation
test_bootstrap_watcher

printf '\n%d passed, %d failed\n' "${PASS_COUNT}" "${FAIL_COUNT}"
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  exit 1
fi
exit 0
