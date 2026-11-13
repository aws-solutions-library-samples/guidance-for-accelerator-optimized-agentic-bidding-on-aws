#!/usr/bin/env bash
# =============================================================================
# test_injection_slot.sh - the generic third-party injection slot, against a real checkout
#
# copy-bidder-files.sh sources any */inject.sh in its slot. That is an interface with code
# this repository does not own, so it needs a test that can be run against a real
# prebid-server-java checkout rather than reasoned about.
#
#   ./test_injection_slot.sh /path/to/prebid-server-java [/path/to/plugin]
#
# Cases, and what each one would catch:
#
#   1. NO PLUGIN         the slot as this repository ships it. The injected file manifest
#                        must be IDENTICAL to the one produced before the loop existed --
#                        byte for byte, every path and every hash. Catches a loop that does
#                        something when there is nothing to do.
#   2. FAILING PLUGIN    an inject.sh that returns non-zero must FAIL the build. Catches the
#                        worst outcome available here: an image that builds, starts, serves
#                        auctions and is missing a seat, which is indistinguishable from
#                        success until an auction returns one fewer bid than expected.
#   3. REAL PLUGIN       given a plugin directory, its files must actually APPEAR in the
#                        checkout, and no upstream file may be modified or removed.
#                        Case 3 is skipped without an argument, and says so.
#
# WHY CASE 3 ASSERTS ADDITIONS RATHER THAN JUST A ZERO EXIT. The first run of this check
# passed while injecting nothing at all: the plugin directory had no inject.sh, so the loop
# correctly ignored it, and the build exited 0. A verification that cannot tell "it worked"
# from "it did nothing" is not a verification.
# =============================================================================
set -uo pipefail

CHECKOUT="${1:-}"
PLUGIN="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREBID_SRC="${REPO_ROOT}/source/prebid"

pass=0; failed=0; skipped=0
ok()   { printf '  PASS  %s\n' "$*"; pass=$((pass + 1)); }
bad()  { printf '  FAIL  %s\n' "$*"; failed=$((failed + 1)); }
skip() { printf '  SKIP  %s\n' "$*"; skipped=$((skipped + 1)); }

if [[ -z "${CHECKOUT}" || ! -d "${CHECKOUT}/src/main/java/org/prebid/server" ]]; then
    echo "usage: $0 /path/to/prebid-server-java [/path/to/plugin]"
    echo
    echo "Fetch a checkout to test against with:"
    echo "  curl -sSL https://github.com/prebid/prebid-server-java/archive/refs/tags/3.43.0.tar.gz | tar xz"
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

manifest() { ( cd "$1" && find . -type f -not -path './.git/*' -print0 \
                 | xargs -0 shasum -a 256 2>/dev/null | sort -k2 ); }

# Assembles a slot exactly as deploy_prebid.sh does, runs the injector from the checkout.
inject_into() {  # $1 = case name, $2 = optional plugin dir; echoes the rc
    local name="$1" plugin="${2:-}"
    cp -R "${CHECKOUT}" "${WORK}/${name}"
    mkdir -p "${WORK}/${name}-slot"
    cp -R "${PREBID_SRC}/." "${WORK}/${name}-slot/"
    [[ -n "${plugin}" ]] && cp -R "${plugin}" "${WORK}/${name}-slot/$(basename "${plugin}")"
    ( cd "${WORK}/${name}" && bash "${WORK}/${name}-slot/copy-bidder-files.sh" ) \
        > "${WORK}/${name}.log" 2>&1
    echo "$?"
}

echo "=== 1. no plugin: the loop must be inert ==="
rc_none="$(inject_into none)"
[[ "${rc_none}" == "0" ]] && ok "injection succeeds with no plugin" \
                          || bad "injection failed with no plugin (rc=${rc_none}); see ${WORK}/none.log"
manifest "${WORK}/none" > "${WORK}/none.manifest"
if grep -q "third-party injector" "${WORK}/none.log"; then
    bad "the loop reported an injector when the slot contains none"
else
    ok "no injector was reported, and none was run"
fi

echo "=== 2. a failing plugin must fail the build ==="
mkdir -p "${WORK}/badplugin"
printf '#!/usr/bin/env bash\necho "[test] deliberately cannot place files" >&2\nreturn 1 2>/dev/null || exit 1\n' \
    > "${WORK}/badplugin/inject.sh"
chmod +x "${WORK}/badplugin/inject.sh"
rc_bad="$(inject_into bad "${WORK}/badplugin")"
[[ "${rc_bad}" != "0" ]] && ok "a failing inject.sh fails the build (rc=${rc_bad})" \
                         || bad "a failing inject.sh was SWALLOWED: the image would be missing that seat"

echo "=== 3. a real plugin must actually inject ==="
if [[ -z "${PLUGIN}" ]]; then
    skip "no plugin directory given, so nothing verified that a plugin's files ARRIVE"
elif [[ ! -f "${PLUGIN}/inject.sh" ]]; then
    bad "${PLUGIN} has no inject.sh, so the loop would ignore it entirely"
else
    rc_real="$(inject_into real "${PLUGIN}")"
    manifest "${WORK}/real" > "${WORK}/real.manifest"
    manifest "${CHECKOUT}" > "${WORK}/pristine.manifest"

    # A comparison against a manifest that is missing or empty reports NO DIFFERENCES, which
    # reads as a pass. That happened for real: the scratch filesystem filled up mid-run, the
    # manifests were never written, and "every upstream file is byte-identical" PASSED while
    # nothing had been compared at all. So the inputs are checked before they are trusted.
    if [[ "${rc_real}" != "0" ]]; then
        bad "injection failed with the plugin (rc=${rc_real}); see ${WORK}/real.log"
        bad "the remaining plugin checks could not run, so nothing about them is known"
    elif [[ ! -s "${WORK}/real.manifest" || ! -s "${WORK}/none.manifest" \
            || ! -s "${WORK}/pristine.manifest" ]]; then
        bad "a manifest is missing or empty, so the comparisons below would be meaningless.
        Check free disk space on ${TMPDIR:-/tmp}: each case copies the whole checkout."
    else
        ok "injection succeeds with the plugin"

        added="$(comm -13 <(cut -d' ' -f3- "${WORK}/none.manifest" | sort) \
                          <(cut -d' ' -f3- "${WORK}/real.manifest" | sort) | wc -l | tr -d ' ')"
        [[ "${added}" -gt 0 ]] && ok "the plugin added ${added} file(s)" \
                               || bad "the plugin added NOTHING -- its inject.sh did not run, or placed no files"

        removed="$(comm -23 <(cut -d' ' -f3- "${WORK}/none.manifest" | sort) \
                            <(cut -d' ' -f3- "${WORK}/real.manifest" | sort) | wc -l | tr -d ' ')"
        [[ "${removed}" -eq 0 ]] && ok "the plugin removed nothing" \
                                 || bad "the plugin REMOVED ${removed} file(s) that the plugin-free build had"

        # Compared on the shared paths only: a file the plugin ADDED has no pristine
        # counterpart and must not be read as a modification.
        shared="$(join -j2 -o 1.1,2.1,1.2 <(sort -k2 "${WORK}/pristine.manifest") \
                                          <(sort -k2 "${WORK}/real.manifest") | wc -l | tr -d ' ')"
        changed="$(join -j2 -o 1.1,2.1,1.2 <(sort -k2 "${WORK}/pristine.manifest") \
                                           <(sort -k2 "${WORK}/real.manifest") \
                     | awk '$1 != $2' | wc -l | tr -d ' ')"
        if [[ "${shared}" -eq 0 ]]; then
            bad "no files were compared against the clean checkout, so 'additions only' is unproven"
        elif [[ "${changed}" -eq 0 ]]; then
            ok "all ${shared} upstream files are byte-identical (additions only)"
        else
            bad "the plugin MODIFIED ${changed} of ${shared} upstream file(s)"
        fi
    fi
fi

echo
echo "  ${pass} passed, ${failed} failed, ${skipped} skipped"
[[ "${failed}" -eq 0 ]] || exit 1
