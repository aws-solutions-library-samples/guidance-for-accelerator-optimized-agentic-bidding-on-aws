# =============================================================================
# deploy_docker.sh — make sure the Docker daemon is running, by starting it.
#
# Phase 5 builds the two AgentCore agent images LOCALLY and on purpose. AgentCore
# runtimes need arm64 images, the build is small, and it is the one place in this
# deployment where a local build is the right tool. That is not the defect.
#
# The defect was the failure mode: with the daemon down, `docker buildx build` died
# with a raw connect error, the ECR repository had already been created, and the
# whole of Phase 5 aborted -- no agent runtimes, no governance-eventbridge stack.
# The only trace was an ECR repository holding zero images. Nothing said "start
# Docker".
#
# So this starts it. A deployment tool that knows exactly what is wrong, and knows
# the one command that fixes it, should run that command rather than print it.
#
# Usage:
#   source "${SCRIPT_DIR}/lib/deploy_docker.sh"
#   docker_ensure_running "build the AgentCore agent images"
# =============================================================================

# How long to wait for the daemon to accept connections after starting it. Docker
# Desktop on a cold start is routinely 30-60s; 180 leaves headroom without hanging
# a deploy indefinitely.
DOCKER_START_TIMEOUT="${DOCKER_START_TIMEOUT:-180}"
DOCKER_POLL="${DOCKER_POLL:-3}"

_docker_daemon_up() { docker info >/dev/null 2>&1; }

# Try every plausible way to start a daemon on this machine, and report which was
# used. Returns 0 if a start was ATTEMPTED (not that it succeeded -- the caller
# polls for that).
_docker_try_start() {
  local uname_s
  uname_s="$(uname -s 2>/dev/null || echo unknown)"

  if [[ "${uname_s}" == "Darwin" ]]; then
    # Docker Desktop registers as "Docker"; some installs as "Docker Desktop".
    if open -a Docker 2>/dev/null; then
      printf 'Docker Desktop'
      return 0
    fi
    if open -a "Docker Desktop" 2>/dev/null; then
      printf 'Docker Desktop'
      return 0
    fi
    # Colima and Rancher Desktop are common Docker Desktop replacements and both
    # provide a docker socket; try them before giving up.
    if command -v colima >/dev/null 2>&1; then
      colima start >/dev/null 2>&1 &
      printf 'colima'
      return 0
    fi
    if command -v rdctl >/dev/null 2>&1; then
      rdctl start >/dev/null 2>&1 &
      printf 'Rancher Desktop'
      return 0
    fi
    return 1
  fi

  # Linux. Prefer the rootless/user service, then the system one. sudo is only
  # attempted non-interactively: a deploy must never block on a password prompt.
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl --user start docker 2>/dev/null; then
      printf 'systemctl --user start docker'
      return 0
    fi
    if sudo -n systemctl start docker 2>/dev/null; then
      printf 'sudo systemctl start docker'
      return 0
    fi
  fi
  if command -v service >/dev/null 2>&1 && sudo -n service docker start >/dev/null 2>&1; then
    printf 'sudo service docker start'
    return 0
  fi
  return 1
}

# docker_ensure_running [what it is needed for] — start Docker if it is not up.
#
# Exits via fail() only when the daemon genuinely cannot be brought up, and then
# says what was tried and what to do. Everything before that point is handled
# without involving the user.
docker_ensure_running() {
  local purpose="${1:-build a container image}"
  local waited=0 started_by

  if ! command -v docker >/dev/null 2>&1; then
    fail "Docker is required to ${purpose}, and the 'docker' command was not found.
       Install Docker Desktop (https://docs.docker.com/desktop/), Colima, or Rancher
       Desktop, then re-run this command -- it will skip everything already deployed."
  fi

  if _docker_daemon_up; then
    log "  Docker daemon is running."
    return 0
  fi

  say "  Docker is needed to ${purpose}, and its daemon is not running. Starting it..."
  if ! started_by="$(_docker_try_start)"; then
    fail "Could not start the Docker daemon automatically, and ${purpose} needs it.
       Tried: Docker Desktop, Colima, Rancher Desktop, systemctl, service.
       Start Docker yourself and re-run this command -- it will skip everything
       already deployed and pick up from here."
  fi
  # "requested", not "started": the launchers are asynchronous (and some are
  # backgrounded), so success is not knowable here. The poll below is what
  # establishes whether it actually came up, and saying otherwise would be a
  # fabricated success.
  say "    start requested via ${started_by}; waiting for it to accept connections..."

  while ! _docker_daemon_up; do
    if [[ "${waited}" -ge "${DOCKER_START_TIMEOUT}" ]]; then
      fail "Asked ${started_by} to start Docker, but the daemon was still not accepting
       connections after ${DOCKER_START_TIMEOUT}s, so ${purpose} cannot proceed.
       Check it with 'docker info' (and that ${started_by} is installed and working),
       then re-run this command -- it will skip everything already deployed."
    fi
    sleep "${DOCKER_POLL}"
    waited=$(( waited + DOCKER_POLL ))
    # Every ~15s, so a long cold start does not look like a hang.
    if [[ $(( waited % 15 )) -eq 0 ]]; then
      say "    still waiting for Docker (${waited}s)..."
    fi
  done

  ok "Docker daemon ready after ${waited}s"
  return 0
}

# docker_ensure_buildx — buildx is what provides --platform builds.
docker_ensure_buildx() {
  docker buildx version >/dev/null 2>&1 && return 0
  fail "docker buildx is required to build the arm64 AgentCore agent images, and it
       is not available. It ships with Docker Desktop; on a plain Docker Engine
       install the buildx plugin separately."
}
