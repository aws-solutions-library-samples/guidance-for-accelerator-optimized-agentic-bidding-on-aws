#!/usr/bin/env bash
# =============================================================================
# copy-bidder-files.sh - inject the ARTF sources into the Prebid Server build
#
# WHERE THIS RUNS, AND WHY IT IS THE ONLY PLACE IT CAN
#
# The upstream AWS guidance Dockerfile (deployment/ecr/prebid-server/Dockerfile,
# v1.4.0) contains exactly one extension point for code we supply:
#
#   ARG INCLUDE_AMT_BIDDER=false
#   ...
#   if [ "$INCLUDE_AMT_BIDDER" = "true" ]; then \
#       cp -r ../amt-bidder . && \
#       chmod +x ../amt-bidder/copy-bidder-files.sh && \
#       ../amt-bidder/copy-bidder-files.sh ; \
#   fi && \
#   mvn clean package $(jq -r .MVN_CLI_OPTIONS ../docker-build-config.json)
#
# So this script runs with the working directory at the prebid-server-java
# checkout, and BEFORE Maven. deploy_prebid.sh copies source/prebid/ into the
# build context's amt-bidder/ slot and passes INCLUDE_AMT_BIDDER=true only when
# this file is present.
#
# WHY NOT extra-modules/, WHICH FR-34 NAMES
#
#   - The Dockerfile's `COPY extra-modules extra-modules/` puts the directory in
#     the build context and nothing more. It then names, by hand, three paths
#     under extra-modules/log-module-reporter/ and copies those into the Prebid
#     source tree. It does not iterate the directory, so a module placed
#     alongside is never compiled: the image builds, starts, serves auctions,
#     and contains none of our code.
#   - Prebid's own module system (extra/modules with an all-modules aggregator,
#     extra/bundle producing a fat jar) is also not in play: this Dockerfile
#     builds the ROOT pom and ships target/prebid-server.jar, which is PBS-Core
#     with no modules at all.
#
# Both were established by the NFR-6 verification gate; see
#
# NO UPSTREAM FILE IS EDITED. Sources are ADDED to the checkout, which is what
# keeps this an integration rather than a fork (U1-NFR-16).
#
# SHARED WITH U4. This script injects the ARTF hook module. The artfhouse
# adapter's own copies are appended by that unit, which is why the helper below
# is generic rather than hook-specific.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '[artf-inject] %s\n' "$*"; }
fail() { printf '[artf-inject][fail] %s\n' "$*" >&2; exit 1; }

# The working directory must be the prebid-server-java checkout. Verified rather
# than assumed: a mis-copy into the wrong tree produces an image that builds
# cleanly and contains none of our code, which is the failure this whole route
# exists to make impossible.
[[ -f pom.xml ]] \
  || fail "no pom.xml in $(pwd) - expected to run from the prebid-server-java checkout"
[[ -d src/main/java/org/prebid/server ]] \
  || fail "no src/main/java/org/prebid/server in $(pwd) - this is not the Prebid Server source tree"
[[ -d src/main/java/org/prebid/server/hooks/v1 ]] \
  || fail "no org/prebid/server/hooks/v1 - this checkout has no hooks API to build against"

log "injecting into $(pwd)"

# ---------------------------------------------------------------------------
# copy_tree <absolute-source-dir> <destination-path-relative-to-checkout>
#
# Source and destination are given SEPARATELY rather than derived from one
# another. They genuinely differ: our sources live under
# source/prebid/artf-hook/src/main/java/... and must land at src/main/java/...
# in the checkout. An earlier version joined a single relative path onto both and
# produced src/main/java/src/main/java/... -- caught by the verification below on
# the first run, which is what it is there for.
#
# Every failure is loud: a silent mis-copy is precisely what produces an
# ARTF-free image that looks healthy.
# ---------------------------------------------------------------------------
copy_tree() {
  local from="$1"
  local to="$2"

  [[ -d "${from}" ]] || fail "expected source directory ${from}, which does not exist"

  mkdir -p "$(dirname "${to}")" \
    || fail "could not create $(dirname "${to}")"
  rm -rf "${to}"
  cp -R "${from}" "${to}" \
    || fail "could not copy ${from} to ${to}"
  [[ -d "${to}" ]] || fail "copy reported success but ${to} is absent"

  local count
  count="$(find "${to}" -name '*.java' -type f | wc -l | tr -d ' ')"
  [[ "${count}" -gt 0 ]] || fail "${to} contains no .java files - nothing would compile"
  log "  ${to} (${count} java files)"
}

# ------------------------------------------------------------ ARTF hook module
HOOK_ROOT="${SCRIPT_DIR}/artf-hook/src/main/java"
ARTF_PACKAGE="org/prebid/server/hooks/modules/artf"

if [[ -d "${HOOK_ROOT}/${ARTF_PACKAGE}" ]]; then
  log "ARTF hook module:"
  copy_tree "${HOOK_ROOT}/${ARTF_PACKAGE}" "src/main/java/${ARTF_PACKAGE}"
else
  fail "no ARTF hook sources at ${HOOK_ROOT}/${ARTF_PACKAGE}"
fi

# The module's @Configuration lives inside the copied package, at
# org/prebid/server/hooks/modules/artf/spring/config/. It needs no separate
# placement: @SpringBootApplication on org.prebid.server.Application scans the
# whole org.prebid.server tree, so component scan finds it there. Nothing in
# org/prebid/server/spring/config/ is touched.
[[ -f "src/main/java/${ARTF_PACKAGE}/spring/config/ArtfModuleConfiguration.java" ]] \
  || fail "ArtfModuleConfiguration.java is missing after the copy - without it no beans are registered and the hook is never invoked"

# --------------------------------------------------------- artfhouse adapter
# Absent, the hook still builds: it calls the orchestrator, not the demand
# endpoint, so the two are independent at build time. Reported rather than
# silent, because an image with the hook and no adapter has no ARTF demand.
#
# A bidder needs FOUR artifacts, and all four are copied here because missing any
# one of them is a STARTUP failure rather than a runtime one:
#   1. the bidder package
#   2. the @Configuration under spring/config/bidder
#   3. bidder-config/artfhouse.yaml   (the @PropertySource reads it off the classpath)
#   4. static/bidder-params/artfhouse.json  (BidderParamValidator loads one per bidder)
ADAPTER_DIR="${SCRIPT_DIR}/artfhouse-adapter"
ADAPTER_ROOT="${ADAPTER_DIR}/src/main/java"
ADAPTER_PACKAGE="org/prebid/server/bidder/artfhouse"
ADAPTER_CONFIG_PACKAGE="org/prebid/server/spring/config/bidder"

if [[ -d "${ADAPTER_ROOT}/${ADAPTER_PACKAGE}" ]]; then
  log "artfhouse adapter:"
  copy_tree "${ADAPTER_ROOT}/${ADAPTER_PACKAGE}" "src/main/java/${ADAPTER_PACKAGE}"

  # The adapter's @Configuration goes into the EXISTING upstream config package
  # alongside every other adapter's, so the files are merged into that directory
  # rather than replacing it -- copy_tree would delete the upstream classes.
  if [[ -d "${ADAPTER_ROOT}/${ADAPTER_CONFIG_PACKAGE}" ]]; then
    local_count="$(find "${ADAPTER_ROOT}/${ADAPTER_CONFIG_PACKAGE}" -name '*.java' -type f | wc -l | tr -d ' ')"
    [[ "${local_count}" -gt 0 ]] || fail "no .java files in ${ADAPTER_CONFIG_PACKAGE}"
    mkdir -p "src/main/java/${ADAPTER_CONFIG_PACKAGE}"
    cp -R "${ADAPTER_ROOT}/${ADAPTER_CONFIG_PACKAGE}/." "src/main/java/${ADAPTER_CONFIG_PACKAGE}/" \
      || fail "could not copy the adapter's Spring configuration"
    [[ -f "src/main/java/${ADAPTER_CONFIG_PACKAGE}/ArtfhouseConfiguration.java" ]] \
      || fail "ArtfhouseConfiguration.java is missing after the copy - without it the bidder is never registered"
    log "  src/main/java/${ADAPTER_CONFIG_PACKAGE} (${local_count} java files, merged)"
  fi

  # Resources are MERGED into the existing trees for the same reason: bidder-config/
  # and static/bidder-params/ already hold ~200 upstream adapters.
  if [[ -d "${ADAPTER_DIR}/src/main/resources" ]]; then
    cp -R "${ADAPTER_DIR}/src/main/resources/." src/main/resources/ \
      || fail "could not copy adapter resources"
    [[ -f "src/main/resources/bidder-config/artfhouse.yaml" ]] \
      || fail "bidder-config/artfhouse.yaml is missing - the @PropertySource would fail at startup"
    [[ -f "src/main/resources/static/bidder-params/artfhouse.json" ]] \
      || fail "static/bidder-params/artfhouse.json is missing - BidderParamValidator requires one per bidder"
    log "  bidder-config/artfhouse.yaml and static/bidder-params/artfhouse.json"
  else
    fail "adapter sources present but no resources - the bidder would not register"
  fi
else
  log "artfhouse adapter: not present in this build (hook only)"
fi

# ---------------------------------------------------------------- final report
log "injection complete. Maven will now compile:"
find "src/main/java/${ARTF_PACKAGE}" -name '*.java' -type f | sort | sed 's/^/  /'
